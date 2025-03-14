# tests/core/test_metrics_lpips.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from resolution_suggester.core.metrics import calculate_lpips, calculate_lpips_channels


@pytest.fixture
def mock_images():
    """Create test images for metrics testing."""
    # Simple test images
    img1 = np.ones((64, 64, 3), dtype=np.float32)  # All ones
    img2 = np.ones((64, 64, 3), dtype=np.float32) * 0.9  # Slightly different
    
    # Single channel
    img_gray1 = np.ones((64, 64), dtype=np.float32)
    img_gray2 = np.ones((64, 64), dtype=np.float32) * 0.9
    
    # Multi-channel
    img_multichannel1 = np.ones((64, 64, 5), dtype=np.float32)
    img_multichannel2 = np.ones((64, 64, 5), dtype=np.float32) * 0.9
    
    return {
        'rgb1': img1, 
        'rgb2': img2,
        'gray1': img_gray1, 
        'gray2': img_gray2,
        'multi1': img_multichannel1, 
        'multi2': img_multichannel2
    }


@pytest.mark.parametrize("net_type", ["alex", "vgg", "squeeze"])
def test_calculate_lpips_mock(mock_images, net_type):
    """Test LPIPS calculation using mocks to avoid actual model loading."""
    # Create a mock LPIPS model and result
    mock_lpips_model = MagicMock()
    mock_lpips_model.return_value.item.return_value = 0.2  # Return distance of 0.2
    
    # Mock torch methods to avoid actual tensor operations
    mock_from_numpy = MagicMock()
    mock_from_numpy.return_value = MagicMock()  # Return a MagicMock when called
    
    mock_zeros = MagicMock()
    mock_zeros.return_value = MagicMock()  # Return a MagicMock when called
    mock_zeros.return_value.clone.return_value = MagicMock()  # Support clone() calls
    mock_zeros.return_value.zero_ = MagicMock()  # Support zero_() calls
    
    # Patch all the PyTorch functions used
    with patch('resolution_suggester.core.metrics.get_lpips_model', return_value=mock_lpips_model), \
         patch('resolution_suggester.core.metrics.torch.from_numpy', mock_from_numpy), \
         patch('resolution_suggester.core.metrics.torch.zeros', mock_zeros), \
         patch('resolution_suggester.core.metrics.torch.no_grad'):
        
        # Test with RGB images
        similarity = calculate_lpips(
            mock_images['rgb1'], 
            mock_images['rgb2'], 
            max_val=1.0, 
            net_type=net_type,
            no_gpu=True
        )
        # Expected similarity = 1.0 - 0.2 = 0.8
        assert similarity == 0.8
        
        # Test with grayscale images (should be converted to 3-channel)
        similarity_gray = calculate_lpips(
            mock_images['gray1'], 
            mock_images['gray2'], 
            max_val=1.0, 
            net_type=net_type,
            no_gpu=True
        )
        assert similarity_gray == 0.8
        
        # Test with max_val > 1.0 (normalization)
        similarity_norm = calculate_lpips(
            mock_images['rgb1'] * 255, 
            mock_images['rgb2'] * 255, 
            max_val=255.0, 
            net_type=net_type,
            no_gpu=True
        )
        assert similarity_norm == 0.8


def test_calculate_lpips_channels_mock(mock_images):
    """Test LPIPS calculation per channel with mocking."""
    channels = ['R', 'G', 'B', 'A', 'X']  # Channel names matching the 5-channel test images
    
    # Create a mock LPIPS model and result
    mock_lpips_model = MagicMock()
    mock_lpips_model.return_value.item.return_value = 0.3  # Return distance of 0.3
    
    # Mock torch methods to avoid actual tensor operations
    mock_from_numpy = MagicMock()
    mock_from_numpy.return_value = MagicMock()  # Return a MagicMock when called
    
    mock_zeros = MagicMock()
    mock_zeros.return_value = MagicMock()  # Return a MagicMock when called
    mock_zeros.return_value.clone.return_value = MagicMock()  # Support clone() calls
    mock_zeros.return_value.zero_ = MagicMock()  # Support zero_() calls
    
    # Patch all the PyTorch functions used
    with patch('resolution_suggester.core.metrics.get_lpips_model', return_value=mock_lpips_model), \
         patch('resolution_suggester.core.metrics.torch.from_numpy', mock_from_numpy), \
         patch('resolution_suggester.core.metrics.torch.zeros', mock_zeros), \
         patch('resolution_suggester.core.metrics.torch.no_grad'):
        
        # Test with multi-channel images
        result = calculate_lpips_channels(
            mock_images['multi1'], 
            mock_images['multi2'], 
            max_val=1.0, 
            channels=channels,
            net_type='alex',
            no_gpu=True
        )
        
        # Expected similarity = 1.0 - 0.3 = 0.7 for each channel
        assert len(result) == 5
        for channel in channels:
            assert result[channel] == 0.7
        
        # Test with single channel (grayscale)
        result_gray = calculate_lpips_channels(
            mock_images['gray1'], 
            mock_images['gray2'], 
            max_val=1.0, 
            channels=['L'],
            net_type='alex',
            no_gpu=True
        )
        
        assert len(result_gray) == 1
        assert result_gray['L'] == 0.7


@pytest.mark.parametrize("max_val", [1.0, 255.0])
def test_calculate_lpips_normalization(mock_images, max_val):
    """Test normalization behavior with different max_val values."""
    # Create a mock LPIPS model and result
    mock_lpips_model = MagicMock()
    mock_lpips_model.return_value.item.return_value = 0.1  # Return distance of 0.1
    
    # Scale images according to max_val
    scaled_img1 = mock_images['rgb1'] * max_val
    scaled_img2 = mock_images['rgb2'] * max_val
    
    # Mock torch methods to avoid actual tensor operations
    mock_from_numpy = MagicMock()
    mock_from_numpy.return_value = MagicMock()  # Return a MagicMock when called
    
    mock_zeros = MagicMock()
    mock_zeros.return_value = MagicMock()  # Return a MagicMock when called
    mock_zeros.return_value.clone.return_value = MagicMock()  # Support clone() calls
    mock_zeros.return_value.zero_ = MagicMock()  # Support zero_() calls
    
    # Patch the required functions
    with patch('resolution_suggester.core.metrics.get_lpips_model', return_value=mock_lpips_model), \
         patch('resolution_suggester.core.metrics.torch.from_numpy', mock_from_numpy), \
         patch('resolution_suggester.core.metrics.torch.zeros', mock_zeros), \
         patch('resolution_suggester.core.metrics.torch.no_grad'):
        
        similarity = calculate_lpips(
            scaled_img1, 
            scaled_img2, 
            max_val=max_val, 
            net_type='alex',
            no_gpu=True
        )
        
        assert similarity == 0.9  # 1.0 - 0.1