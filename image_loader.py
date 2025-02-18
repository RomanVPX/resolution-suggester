# image_loader.py
import logging
import numpy as np
import pyexr
import os
from PIL import Image, ImageFile
from typing import Dict
from dataclasses import dataclass

ImageFile.LOAD_TRUNCATED_IMAGES = True

BIT_DEPTH_16 = 65535.0
BIT_DEPTH_8 = 255.0

MODE_CHANNEL_MAP: Dict[str, list[str]] = {
    'L': ['L'],
    'LA': ['L', 'A'],
    'RGB': ['R', 'G', 'B'],
    'RGBA': ['R', 'G', 'B', 'A'],
    'I;16': ['L'],
    'I;16L': ['L'],
    'I;16B': ['L'],
    'CMYK': ['C', 'M', 'Y', 'K'],
    'YCbCr': ['Y', 'Cb', 'Cr'],
    'LAB': ['L', 'a', 'b']
}

@dataclass
class ImageLoadResult:
    data: np.ndarray | None
    max_value: float | None
    channels: list[str] | None
    error: str | None = None

def load_image(file_path: str) -> ImageLoadResult:
    """
    Загружает изображение из файла и возвращает массив numpy, максимальное значение и каналы.
    Поддерживаемые форматы: EXR, PNG, TGA.

    Args:
        file_path: Путь к файлу изображения

    Returns:
        Tuple: (image_array, max_value, channels) или (None, None, None) при ошибке
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.exr':
            return load_exr(file_path)
        if ext in ('.png', '.tga'):
            return load_raster(file_path)

        logging.warning(f"Unsupported format: {file_path}")
        return ImageLoadResult(None, None, None)

    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        return ImageLoadResult(None, None, None, str(e))

def load_exr(file_path: str) -> ImageLoadResult:
    """Загружает EXR файл с обработкой каналов"""
    try:
        exr_file = pyexr.open(file_path)

        # Получаем каналы через атрибут channels
        channels = exr_file.channels if hasattr(exr_file, 'channels') else []


        img = exr_file.get()
        img = img.astype(np.float32)

        # Автодетект каналов
        if not channels:
            num_channels = img.shape[2] if img.ndim > 2 else 1
            channels = ['R', 'G', 'B', 'A'][:num_channels] if num_channels > 1 else ['L']

        max_val = np.max(np.abs(img))
        return ImageLoadResult(img, max_val, channels)

    except Exception as e:
        logging.error(f"EXR processing error {file_path}: {str(e)}")
        return ImageLoadResult(None, None, None, str(e))

def load_raster(file_path: str) -> ImageLoadResult:
    img = Image.open(file_path)

    if img.mode not in MODE_CHANNEL_MAP:
        img = img.convert('RGB')

    divisor = BIT_DEPTH_16 if img.mode.startswith('I;16') else BIT_DEPTH_8
    img_array = np.array(img).astype(np.float32) / divisor
    channels = MODE_CHANNEL_MAP[img.mode]

    # Максимальное значение после нормализации всегда 1.0
    return ImageLoadResult(img_array, 1.0, channels)