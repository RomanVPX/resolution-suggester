# tests/tests_main.py
import pytest
import numpy as np

from resolution_suggester.config import PSNR_IS_LARGE_AS_INF, QualityMetrics
from resolution_suggester.core.image_analyzer import postprocess_metric_value


@pytest.mark.parametrize("value, metric_type, expected", [
    # Test PSNR values above threshold
    (PSNR_IS_LARGE_AS_INF, QualityMetrics.PSNR, float('inf')),
    (PSNR_IS_LARGE_AS_INF + 1, QualityMetrics.PSNR, float('inf')),
    (PSNR_IS_LARGE_AS_INF - 1, QualityMetrics.PSNR, PSNR_IS_LARGE_AS_INF - 1),
    (1000.0, QualityMetrics.PSNR, float('inf')),

    # Test if PSNR values below threshold
    (PSNR_IS_LARGE_AS_INF - 1, QualityMetrics.PSNR, PSNR_IS_LARGE_AS_INF - 1),
    (100.0, QualityMetrics.PSNR, 100.0),
    (0.0, QualityMetrics.PSNR, 0.0),
    (-1.0, QualityMetrics.PSNR, -1.0),

    # Testing other metrics (should clamp to [0;1]])
    (100, QualityMetrics.SSIM, 1.0),
    (0.99, QualityMetrics.MS_SSIM, 0.99),
    (-0.343, QualityMetrics.MS_SSIM, 0.0),
    (1.0, QualityMetrics.MS_SSIM, 1.0),
    (0.0, QualityMetrics.MS_SSIM, 0.0),
    (-0.0, QualityMetrics.MS_SSIM, 0.0),
])
def test_postprocess_metric_value_scalar(value, metric_type, expected):
    """Test postprocess_metric_value with scalar values"""
    result = postprocess_metric_value(value, metric_type)

    if expected == float('inf'):
        assert result == float('inf')
    else:
        assert result == expected


def test_postprocess_metric_value_invalid_type():
    """Test handling of invalid types"""
    with pytest.raises(TypeError):
        postprocess_metric_value([1, 2, 3], QualityMetrics.PSNR)

    with pytest.raises(TypeError):
        postprocess_metric_value(None, QualityMetrics.PSNR)


def test_postprocess_metric_value_dict():
    """Test postprocess_metric_value with dictionary values"""
    # Словарь для PSNR
    metrics_psnr = {'R': 140.0, 'G': 100.0, 'B': PSNR_IS_LARGE_AS_INF, 'A': 50.0}
    result_psnr = postprocess_metric_value(metrics_psnr, QualityMetrics.PSNR)
    assert result_psnr['R'] == float('inf')
    assert result_psnr['G'] == 100.0
    assert result_psnr['B'] == float('inf')
    assert result_psnr['A'] == 50.0

    # Словарь для SSIM
    metrics_ssim = {'R': 0.95, 'G': 0.98, 'B': 0.99}
    result_ssim = postprocess_metric_value(metrics_ssim, QualityMetrics.SSIM)
    assert result_ssim == metrics_ssim  # Должен вернуться без изменений

    # Словарь для SSIM
    metrics_ssim = {'R': 1.001, 'G': 0.98, 'B': 0.99}
    result_ssim = postprocess_metric_value(metrics_ssim, QualityMetrics.SSIM)
    assert result_ssim == {'R': 1.00, 'G': 0.98, 'B': 0.99}  # Должен клемпнуть

def test_postprocess_metric_value_nan():
    """Test handling of NaN values"""
    result = postprocess_metric_value(float('nan'), QualityMetrics.PSNR)
    assert np.isnan(result)

    # Словарь с NaN
    metrics_with_nan = {'R': float('nan'), 'G': 100.0}
    result_dict = postprocess_metric_value(metrics_with_nan, QualityMetrics.PSNR)
    assert np.isnan(result_dict['R'])
    assert result_dict['G'] == 100.0
