import pytest
from resolution_suggester.config import PSNR_IS_LARGE_AS_INF, QualityMetrics
from resolution_suggester.core.metrics import postprocess_psnr_value


@pytest.mark.parametrize("psnr_value, metric_type, expected", [
    # Test PSNR values above threshold
    (PSNR_IS_LARGE_AS_INF, QualityMetrics.PSNR, float('inf')),
    (PSNR_IS_LARGE_AS_INF + 1, QualityMetrics.PSNR, float('inf')),
    (1000.0, QualityMetrics.PSNR, float('inf')),

    # Test if PSNR values below threshold
    (PSNR_IS_LARGE_AS_INF - 1, QualityMetrics.PSNR, PSNR_IS_LARGE_AS_INF - 1),
    (100.0, QualityMetrics.PSNR, 100.0),
    (0.0, QualityMetrics.PSNR, 0.0),
    (-1.0, QualityMetrics.PSNR, -1.0),

    # Testing other metrics (should return input values without changes)
    (PSNR_IS_LARGE_AS_INF + 1, QualityMetrics.SSIM, PSNR_IS_LARGE_AS_INF + 1),
    (PSNR_IS_LARGE_AS_INF, QualityMetrics.MS_SSIM, PSNR_IS_LARGE_AS_INF),
    (100.0, QualityMetrics.SSIM, 100.0),
    (0.0, QualityMetrics.MS_SSIM, 0.0),
])
def test_postprocess_psnr_value(psnr_value, metric_type, expected):
    """
    Test postprocess_psnr_value function.

    Test cases:
    1. PSNR values above threshold should return infinity
    2. PSNR values below threshold should return unchanged
    3. Non-PSNR metrics should return unchanged values
    """
    result = postprocess_psnr_value(psnr_value, metric_type)

    if expected == float('inf'):
        assert result == float('inf')
    else:
        assert result == expected


# Test for invalid input data
def test_postprocess_psnr_value_invalid_metric():
    """Test that function raises ValueError for invalid metric type"""
    with pytest.raises(ValueError):
        postprocess_psnr_value(100.0, "invalid_metric")


# Test for special float values
@pytest.mark.parametrize("special_value, metric_type", [
    (float('nan'), QualityMetrics.PSNR),
    (float('-inf'), QualityMetrics.PSNR),
])
def test_postprocess_psnr_value_special_values(special_value, metric_type):
    """Test handling of special float values"""
    result = postprocess_psnr_value(special_value, metric_type)
    assert result == special_value
