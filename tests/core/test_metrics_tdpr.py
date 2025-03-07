# tests/core/test_metrics_tdpr.py
import numpy as np
import pytest

from resolution_suggester.core.metrics import calculate_tdpr, calculate_tdpr_channels, detect_texture_edges


@pytest.fixture
def edge_images():
    """Create test images with clear edges for TDPR testing."""
    # Create a 100x100 image with a clear edge in the middle
    img1 = np.zeros((100, 100), dtype=np.float32)
    img1[40:60, :] = 1.0  # Horizontal band
    
    # Same pattern but with some noise/blur to simulate processing
    img2 = np.zeros((100, 100), dtype=np.float32)
    img2[42:62, :] = 1.0  # Shifted band
    
    # Complete loss of detail
    img3 = np.zeros((100, 100), dtype=np.float32)
    
    # RGB versions
    img1_rgb = np.zeros((100, 100, 3), dtype=np.float32)
    img1_rgb[40:60, :, 0] = 1.0  # Edge in R channel
    img1_rgb[45:65, :, 1] = 1.0  # Edge in G channel
    img1_rgb[50:70, :, 2] = 1.0  # Edge in B channel
    
    img2_rgb = np.zeros((100, 100, 3), dtype=np.float32)
    img2_rgb[42:62, :, 0] = 1.0  # Shifted edge in R
    img2_rgb[47:67, :, 1] = 1.0  # Shifted edge in G
    img2_rgb[52:72, :, 2] = 1.0  # Shifted edge in B
    
    # Multi-channel image with 5 channels
    img1_multi = np.zeros((100, 100, 5), dtype=np.float32)
    for i in range(5):
        img1_multi[40+i*2:60+i*2, :, i] = 1.0
    
    img2_multi = np.zeros((100, 100, 5), dtype=np.float32)
    for i in range(5):
        img2_multi[42+i*2:62+i*2, :, i] = 1.0
    
    return {
        'edge1': img1,
        'edge2': img2,
        'edge3': img3,
        'rgb1': img1_rgb,
        'rgb2': img2_rgb,
        'multi1': img1_multi,
        'multi2': img2_multi
    }


def test_detect_texture_edges(edge_images):
    """Test the edge detection functionality used by TDPR."""
    # Test basic edge detection
    edges = detect_texture_edges(edge_images['edge1'])
    assert edges.shape == (100, 100)
    assert edges.dtype == bool
    
    # There should be edges detected at the transition points
    assert np.any(edges)
    
    # Check edge positions (should be roughly at rows 40 and 60)
    edge_rows = np.where(np.any(edges, axis=1))[0]
    assert any(abs(row - 40) <= 3 for row in edge_rows)  # Allow for some margin due to edge detector
    assert any(abs(row - 60) <= 3 for row in edge_rows)
    
    # Test edge detection on RGB image
    edges_rgb = detect_texture_edges(edge_images['rgb1'])
    assert edges_rgb.shape == (100, 100)
    assert np.any(edges_rgb)
    
    # Test edge detection on blank image (should have minimal or no edges)
    blank = np.zeros((50, 50), dtype=np.float32)
    edges_blank = detect_texture_edges(blank)
    assert edges_blank.shape == (50, 50)
    # Most likely will have very few or no edges
    assert np.sum(edges_blank) < 10


def test_calculate_tdpr_basic(edge_images):
    """Test basic TDPR calculation functionality."""
    # Perfect preservation case
    tdpr_perfect = calculate_tdpr(edge_images['edge1'], edge_images['edge1'])
    assert tdpr_perfect == 1.0
    
    # Imperfect preservation case (shifted edges)
    # Test with dilation to help detect the overlapping regions
    tdpr_shifted = calculate_tdpr(edge_images['edge1'], edge_images['edge2'], dilation_size=3)
    assert 0.0 < tdpr_shifted <= 1.0
    
    # Complete loss case
    tdpr_lost = calculate_tdpr(edge_images['edge1'], edge_images['edge3'])
    assert tdpr_lost == 0.0
    
    # TDPR should be symmetric in perfect case
    tdpr_symmetric = calculate_tdpr(edge_images['edge2'], edge_images['edge2'])
    assert tdpr_symmetric == 1.0
    
    # Test with different sigma values
    tdpr_high_sigma = calculate_tdpr(edge_images['edge1'], edge_images['edge2'], edge_sigma=2.0, dilation_size=3)
    tdpr_low_sigma = calculate_tdpr(edge_images['edge1'], edge_images['edge2'], edge_sigma=0.5, dilation_size=3)
    # Different sigmas will give different results, but both should be in valid range
    assert 0.0 <= tdpr_high_sigma <= 1.0
    assert 0.0 <= tdpr_low_sigma <= 1.0
    
    # Test with different dilation sizes
    tdpr_small_dilation = calculate_tdpr(edge_images['edge1'], edge_images['edge2'], dilation_size=1)
    tdpr_large_dilation = calculate_tdpr(edge_images['edge1'], edge_images['edge2'], dilation_size=5)
    
    # Larger dilation should find more overlapping edges
    assert tdpr_large_dilation >= tdpr_small_dilation


def test_calculate_tdpr_channels(edge_images):
    """Test TDPR calculation on multi-channel images."""
    # Test with RGB images
    channels = ['R', 'G', 'B']
    tdpr_channels = calculate_tdpr_channels(
        edge_images['rgb1'], 
        edge_images['rgb2'], 
        channels=channels
    )
    
    # Should return a dict with values for each channel
    assert isinstance(tdpr_channels, dict)
    assert len(tdpr_channels) == 3
    assert all(channel in tdpr_channels for channel in channels)
    assert all(0.0 <= value <= 1.0 for value in tdpr_channels.values())
    
    # Test with 5-channel images
    channels_multi = ['A', 'B', 'C', 'D', 'E']
    tdpr_multi = calculate_tdpr_channels(
        edge_images['multi1'], 
        edge_images['multi2'], 
        channels=channels_multi
    )
    
    assert isinstance(tdpr_multi, dict)
    assert len(tdpr_multi) == 5
    assert all(channel in tdpr_multi for channel in channels_multi)
    assert all(0.0 <= value <= 1.0 for value in tdpr_multi.values())
    
    # Test with grayscale image
    tdpr_gray = calculate_tdpr_channels(
        edge_images['edge1'], 
        edge_images['edge2'], 
        channels=['L']
    )
    
    assert isinstance(tdpr_gray, dict)
    assert len(tdpr_gray) == 1
    assert 'L' in tdpr_gray
    assert 0.0 <= tdpr_gray['L'] <= 1.0


def test_tdpr_edge_cases():
    """Test TDPR calculation with edge cases."""
    # Empty images
    img_empty = np.zeros((10, 10), dtype=np.float32)
    
    # No edges in original means perfect preservation (nothing to preserve)
    tdpr_empty = calculate_tdpr(img_empty, img_empty)
    assert tdpr_empty == 1.0
    
    # Images with just noise
    np.random.seed(42)  # For reproducibility
    img_noise1 = np.random.random((50, 50)).astype(np.float32) * 0.1
    img_noise2 = np.random.random((50, 50)).astype(np.float32) * 0.1
    
    # Almost white images
    img_white1 = np.ones((50, 50), dtype=np.float32) - 0.01
    img_white2 = np.ones((50, 50), dtype=np.float32) - 0.02
    
    # Calculate TDPR for edge cases
    tdpr_noise = calculate_tdpr(img_noise1, img_noise2)
    tdpr_white = calculate_tdpr(img_white1, img_white2)
    
    # Both should give sensible results between 0 and 1
    assert 0.0 <= tdpr_noise <= 1.0
    assert 0.0 <= tdpr_white <= 1.0