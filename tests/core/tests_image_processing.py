# tests/core/tests_image_processing.py
import numpy as np
import pytest

from resolution_suggester.core.image_processing import resize_mitchell, get_resize_function
from resolution_suggester.config import InterpolationMethods


def test_resize_mitchell():
    """Test Mitchell-Netravali filter resize function with various dimensions."""
    # Test with 2D grayscale image
    img_2d = np.ones((100, 100), dtype=np.float32)
    resized_2d = resize_mitchell(img_2d, 50, 50)  # Target: width=50, height=50
    assert resized_2d.shape == (50, 50)
    assert resized_2d.dtype == np.float32
    assert np.allclose(resized_2d, 1.0)  # Uniform image should remain uniform

    # Test with 3D RGB image
    img_3d = np.ones((100, 100, 3), dtype=np.float32)
    resized_3d = resize_mitchell(img_3d, 50, 50)  # Target: width=50, height=50
    assert resized_3d.shape == (50, 50, 3)
    assert resized_3d.dtype == np.float32
    assert np.allclose(resized_3d, 1.0)  # Uniform image should remain uniform
    
    # Test with 3D single-channel image
    img_3d_1ch = np.ones((100, 100, 1), dtype=np.float32)
    resized_3d_1ch = resize_mitchell(img_3d_1ch, 50, 50)  # Target: width=50, height=50
    assert resized_3d_1ch.shape == (50, 50, 1)
    assert resized_3d_1ch.dtype == np.float32
    assert np.allclose(resized_3d_1ch, 1.0)
    
    # Test upscaling
    small_img = np.ones((50, 50, 3), dtype=np.float32)
    upscaled = resize_mitchell(small_img, 100, 100)  # Target: width=100, height=100
    assert upscaled.shape == (100, 100, 3)
    assert upscaled.dtype == np.float32
    assert np.allclose(upscaled, 1.0)
    
    # Test with gradient image
    gradient_img = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)
    resized_gradient = resize_mitchell(gradient_img, 50, 50)  # Target: width=50, height=50
    assert resized_gradient.shape == (50, 50)
    # Values should change smoothly
    assert resized_gradient[0, 0] < resized_gradient[-1, -1]


def test_resize_mitchell_edge_cases():
    """Test edge cases for Mitchell resize function."""
    # Test with original dimensions (no resize should occur)
    img = np.ones((100, 100, 3), dtype=np.float32)
    resized = resize_mitchell(img, 100, 100)
    assert resized.shape == (100, 100, 3)
    assert np.array_equal(img, resized)  # Should be exact copy
    
    # Test with extreme aspect ratio - resize_mitchell takes (target_width, target_height) as 2nd/3rd args
    img_wide = np.ones((10, 100, 3), dtype=np.float32)  # 10 rows, 100 cols
    resized_wide = resize_mitchell(img_wide, 50, 5)  # Target: width=50, height=5
    assert resized_wide.shape == (5, 50, 3)  # Output will be (height, width, channels)
    assert np.allclose(resized_wide, 1.0)
    
    # Test with small image
    img_tiny = np.ones((4, 4, 3), dtype=np.float32)
    resized_tiny = resize_mitchell(img_tiny, 2, 2)
    assert resized_tiny.shape == (2, 2, 3)
    assert np.allclose(resized_tiny, 1.0)
    
    # Test with odd dimensions
    img_odd = np.ones((99, 101, 3), dtype=np.float32)
    resized_odd = resize_mitchell(img_odd, 51, 49)  # Target: width=51, height=49
    assert resized_odd.shape == (49, 51, 3)
    assert np.allclose(resized_odd, 1.0)


def test_get_resize_function():
    """Test the factory function for resize functions."""
    # Test Mitchell resize
    mitchell_fn = get_resize_function(InterpolationMethods.MITCHELL)
    assert callable(mitchell_fn)
    
    # Test some OpenCV methods
    bilinear_fn = get_resize_function(InterpolationMethods.BILINEAR)
    assert callable(bilinear_fn)
    
    bicubic_fn = get_resize_function(InterpolationMethods.BICUBIC)
    assert callable(bicubic_fn)
    
    # Test with a small image to confirm function works
    img = np.ones((10, 10, 3), dtype=np.float32)
    resized = bilinear_fn(img, 5, 5)
    assert resized.shape == (5, 5, 3)
    assert np.allclose(resized, 1.0)
    
    # Test with grayscale image
    img_gray = np.ones((10, 10), dtype=np.float32)
    resized_gray = bilinear_fn(img_gray, 5, 5)
    assert resized_gray.shape == (5, 5, 1)  # Should add channel dimension
    assert np.allclose(resized_gray, 1.0)
    
    # Test that the function is memoized (same instance returned)
    mitchell_fn2 = get_resize_function(InterpolationMethods.MITCHELL)
    assert mitchell_fn2 is mitchell_fn  # Same function instance (due to lru_cache)
    
    # Test that different methods return different functions
    assert bilinear_fn is not mitchell_fn


def test_get_resize_function_invalid():
    """Test get_resize_function with invalid interpolation method."""
    with pytest.raises(ValueError):
        get_resize_function("invalid_method")