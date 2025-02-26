# tests/core/tests_metrics.py
import numpy as np
import pytest

from resolution_suggester.core.metrics import compute_resolutions, calculate_metrics
from resolution_suggester.config import MIN_DOWNSCALE_SIZE, QualityMetrics


def test_compute_resolutions():
    # Импортируем константу для использования в тестах

    # Базовый случай
    resolutions = compute_resolutions(1024, 1024)
    expected = []
    w, h = 1024, 1024
    while w >= MIN_DOWNSCALE_SIZE and h >= MIN_DOWNSCALE_SIZE:
        expected.append((w, h))
        w //= 2
        h //= 2
    assert resolutions == expected

    # Случай с разными размерами
    resolutions = compute_resolutions(800, 600)
    expected = []
    w, h = 800, 600
    while w >= MIN_DOWNSCALE_SIZE and h >= MIN_DOWNSCALE_SIZE:
        expected.append((w, h))
        w //= 2
        h //= 2
    assert resolutions == expected

    # Случай с пользовательским min_size больше константы
    custom_min_size = MIN_DOWNSCALE_SIZE * 2
    resolutions = compute_resolutions(1024, 1024, min_size=custom_min_size)
    expected = []
    w, h = 1024, 1024
    while w >= custom_min_size and h >= custom_min_size:
        expected.append((w, h))
        w //= 2
        h //= 2
    assert resolutions == expected

    # Случай с пользовательским min_size меньше константы (должен использоваться MIN_DOWNSCALE_SIZE)
    resolutions = compute_resolutions(1024, 1024, min_size=MIN_DOWNSCALE_SIZE // 2)
    expected = []
    w, h = 1024, 1024
    while w >= MIN_DOWNSCALE_SIZE and h >= MIN_DOWNSCALE_SIZE:
        expected.append((w, h))
        w //= 2
        h //= 2
    assert resolutions == expected

    # Случай с граничным значением
    resolutions = compute_resolutions(MIN_DOWNSCALE_SIZE, MIN_DOWNSCALE_SIZE)
    assert resolutions == [(MIN_DOWNSCALE_SIZE, MIN_DOWNSCALE_SIZE)]  # Только оригинальное разрешение

    # Случай с изображением меньше min_size
    resolutions = compute_resolutions(MIN_DOWNSCALE_SIZE - 1, MIN_DOWNSCALE_SIZE - 1)
    assert resolutions == []  # Пустой список, т.к. изображение меньше min_size

    # Случай с другим делителем
    resolutions = compute_resolutions(100, 100, divider=4)
    expected = []
    w, h = 100, 100
    while w >= MIN_DOWNSCALE_SIZE and h >= MIN_DOWNSCALE_SIZE:
        expected.append((w, h))
        w //= 4
        h //= 4
    assert resolutions == expected

@pytest.fixture
def mock_image():
    # Создаём тестовое изображение
    img = np.zeros((100, 100, 3), dtype=np.float32)
    return img

def test_calculate_metrics(mock_image):
    result = calculate_metrics(QualityMetrics.PSNR, mock_image, mock_image, 1.0)
    assert result == float('inf')
