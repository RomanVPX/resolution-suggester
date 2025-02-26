# tests/core/test_image_loader.py
from resolution_suggester.core.image_loader import load_image


def test_load_image(tmp_path):
    # Создаём тестовое изображение
    from PIL import Image
    import numpy as np

    # RGB изображение
    img_rgb = Image.new('RGB', (100, 100), color=(255, 0, 0))
    rgb_path = tmp_path / "test_rgb.png"
    img_rgb.save(rgb_path)

    # Grayscale изображение
    img_gray = Image.new('L', (100, 100), color=128)
    gray_path = tmp_path / "test_gray.png"
    img_gray.save(gray_path)

    # Тестируем загрузку RGB
    result_rgb = load_image(str(rgb_path))
    assert result_rgb.error is None
    assert result_rgb.data.shape == (100, 100, 3)
    assert result_rgb.max_value == 1.0
    assert result_rgb.channels == ['R', 'G', 'B']

    # Тестируем загрузку Grayscale
    result_gray = load_image(str(gray_path))
    assert result_gray.error is None
    assert result_gray.data.shape == (100, 100, 1)
    assert result_gray.max_value == 1.0
    assert result_gray.channels == ['L']

    # Тестируем обработку ошибок
    result_error = load_image("nonexistent_file.png")
    assert result_error.error is not None
    assert result_error.data is None
