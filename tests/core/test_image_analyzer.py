# tests/core/test_image_analyzer.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from resolution_suggester.core.image_analyzer import ImageAnalyzer, postprocess_metric_value
from resolution_suggester.config import QualityMetrics, InterpolationMethods


@pytest.fixture
def mock_args():
    """Create a mock args object for testing."""
    args = MagicMock()
    args.ml = False
    args.no_parallel = True
    args.min_size = 32
    args.interpolation = InterpolationMethods.MITCHELL
    args.metric = QualityMetrics.PSNR
    args.channels = False
    args.no_gpu = True
    args.lpips_net = 'alex'
    args.save_im_down = False
    args.save_im_up = False
    args.chart = False
    return args


@pytest.fixture
def test_image():
    """Create a test image for analysis."""
    return np.ones((100, 100, 3), dtype=np.float32)


@patch('resolution_suggester.core.image_analyzer.get_resize_function')
def test_image_analyzer_init(mock_get_resize, mock_args):
    """Test ImageAnalyzer initialization."""
    # Setup mock
    mock_resize_fn = MagicMock()
    mock_get_resize.return_value = mock_resize_fn
    
    # Create analyzer
    analyzer = ImageAnalyzer(mock_args)
    
    # Verify initialization
    assert analyzer.args == mock_args
    assert analyzer.reporters == []
    assert analyzer.predictor is None  # No ML mode
    assert analyzer.resize_fn is mock_resize_fn
    assert analyzer.resize_fn_upscale is mock_resize_fn
    
    # Test ML initialization
    mock_args.ml = True
    with patch('resolution_suggester.core.image_analyzer.QuickPredictor') as mock_predictor_class:
        mock_predictor = MagicMock()
        mock_predictor.load.return_value = True
        mock_predictor_class.return_value = mock_predictor
        
        analyzer = ImageAnalyzer(mock_args)
        
        assert analyzer.predictor is mock_predictor
        mock_predictor.set_mode.assert_called_once_with(mock_args.channels)


@patch('resolution_suggester.core.image_analyzer.load_image')
@patch('resolution_suggester.core.image_analyzer.get_resize_function')
@patch('resolution_suggester.core.image_analyzer.compute_resolutions')
@patch('resolution_suggester.core.image_analyzer.calculate_metrics')
def test_analyze_resize_real(mock_calculate_metrics, mock_compute_resolutions, 
                             mock_get_resize, mock_load_image, mock_args, test_image):
    """Test the _analyze_resize_real method."""
    # Setup mocks
    mock_resize_fn = MagicMock()
    mock_resize_fn.return_value = test_image
    mock_get_resize.return_value = mock_resize_fn
    
    # Mock metrics calculation
    mock_calculate_metrics.return_value = 45.0  # A typical PSNR value
    
    # Create analyzer
    analyzer = ImageAnalyzer(mock_args)
    
    # Run method
    result = analyzer._analyze_resize_real(
        test_image, 1.0, ['R', 'G', 'B'], 50, 50, 100, 100, "test_file.png"
    )
    
    # Verify results - standard mode without channels
    assert isinstance(result, tuple)
    assert result[0] == "50x50"
    assert result[1] == 45.0  # The metric value
    assert len(result) == 3  # Resolution, metric, hint
    
    # Now test with channels enabled
    mock_args.channels = True
    mock_calculate_metrics.return_value = {'R': 45.0, 'G': 48.0, 'B': 42.0}
    
    analyzer = ImageAnalyzer(mock_args)
    
    result = analyzer._analyze_resize_real(
        test_image, 1.0, ['R', 'G', 'B'], 50, 50, 100, 100, "test_file.png"
    )
    
    # Verify channel-based results
    assert isinstance(result, tuple)
    assert result[0] == "50x50"
    assert isinstance(result[1], dict)
    assert len(result[1]) == 3
    assert result[1]['R'] == 45.0
    assert result[1]['G'] == 48.0
    assert result[1]['B'] == 42.0
    assert result[2] == 42.0  # Minimum metric value
    assert len(result) == 4  # Resolution, metrics dict, min metric, hint
    
    # Verify resize function calls
    mock_resize_fn.assert_any_call(test_image, 50, 50)  # Downscale
    mock_resize_fn.assert_any_call(test_image, 100, 100)  # Upscale
    
    # Verify metrics calculation
    mock_calculate_metrics.assert_called()


@patch('resolution_suggester.core.image_analyzer.get_resize_function')
def test_analyze_resize_ml(mock_get_resize, mock_args, test_image):
    """Test the _analyze_resize_ml method."""
    # Setup for ML prediction
    mock_args.ml = True
    mock_args.metric = QualityMetrics.PSNR
    
    with patch('resolution_suggester.core.image_analyzer.QuickPredictor') as mock_predictor_class, \
         patch('resolution_suggester.core.image_analyzer.extract_features_of_original_img') as mock_extract:
        
        # Setup predictor mock
        mock_predictor = MagicMock()
        mock_predictor.load.return_value = True
        mock_predictor.predict.return_value = {QualityMetrics.PSNR.value: 40.0}
        mock_predictor_class.return_value = mock_predictor
        
        # Setup features mock
        mock_extract.return_value = {'feature1': 1.0, 'feature2': 2.0}
        
        # Create analyzer with ML
        analyzer = ImageAnalyzer(mock_args)
        
        # Test without channels
        result = analyzer._analyze_resize_ml(
            test_image, ['R', 'G', 'B'], 50, 50, 100, 100
        )
        
        # Verify results
        assert isinstance(result, tuple)
        assert result[0] == "50x50"
        assert result[1] == 40.0  # The predicted metric value
        assert len(result) == 3  # Resolution, metric, hint
        
        # Verify predictor was called with features
        mock_predictor.predict.assert_called()
        called_features = mock_predictor.predict.call_args[0][0]
        assert 'feature1' in called_features
        assert 'scale_factor' in called_features
        assert called_features['channel'] == 'combined'
        
        # Now test with channels enabled
        mock_args.channels = True
        analyzer = ImageAnalyzer(mock_args)
        
        # Setup batch prediction for channel mode
        mock_predictor.predict_batch.return_value = [[40.0], [42.0], [38.0]]
        
        result = analyzer._analyze_resize_ml(
            test_image, ['R', 'G', 'B'], 50, 50, 100, 100
        )
        
        # Verify channel-based results
        assert isinstance(result, tuple)
        assert result[0] == "50x50"
        assert isinstance(result[1], dict)
        assert len(result[1]) == 3
        assert result[1]['R'] == 40.0
        assert result[1]['G'] == 42.0
        assert result[1]['B'] == 38.0
        assert result[2] == 38.0  # Minimum metric value
        assert len(result) == 4  # Resolution, metrics dict, min metric, hint


def test_postprocess_metric_value():
    """Test the postprocess_metric_value function."""
    # Import constant for test
    from resolution_suggester.config import PSNR_IS_LARGE_AS_INF
    
    # Test PSNR with scalars
    assert postprocess_metric_value(45.0, QualityMetrics.PSNR) == 45.0
    assert postprocess_metric_value(45.0 + 1.0, QualityMetrics.PSNR) == 45.0 + 1.0
    
    # Test PSNR threshold for infinity
    assert postprocess_metric_value(PSNR_IS_LARGE_AS_INF - 1.0, QualityMetrics.PSNR) == PSNR_IS_LARGE_AS_INF - 1.0
    assert postprocess_metric_value(PSNR_IS_LARGE_AS_INF, QualityMetrics.PSNR) == float('inf')
    assert postprocess_metric_value(PSNR_IS_LARGE_AS_INF + 10.0, QualityMetrics.PSNR) == float('inf')
    
    # Test PSNR with dictionaries
    channel_data = {'R': 45.0, 'G': PSNR_IS_LARGE_AS_INF, 'B': 30.0}
    result = postprocess_metric_value(channel_data, QualityMetrics.PSNR)
    assert result['R'] == 45.0
    assert result['G'] == float('inf')
    assert result['B'] == 30.0
    
    # Test other metrics with scalars (clamp to [0,1])
    assert postprocess_metric_value(-0.5, QualityMetrics.SSIM) == 0.0
    assert postprocess_metric_value(0.7, QualityMetrics.SSIM) == 0.7
    assert postprocess_metric_value(1.5, QualityMetrics.SSIM) == 1.0
    
    # Test other metrics with dictionaries
    channel_data = {'R': -0.1, 'G': 0.7, 'B': 1.2}
    result = postprocess_metric_value(channel_data, QualityMetrics.SSIM)
    assert result['R'] == 0.0
    assert result['G'] == 0.7
    assert result['B'] == 1.0
    
    # Test invalid type
    with pytest.raises(TypeError):
        postprocess_metric_value("invalid", QualityMetrics.PSNR)