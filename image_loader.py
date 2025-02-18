# image_loader.py
import logging
import numpy as np
import pyexr
import os
from PIL import Image, ImageFile
from typing import Tuple, Optional, Dict
import numpy.typing as npt

ImageFile.LOAD_TRUNCATED_IMAGES = True

BIT_DEPTH_16 = 65535.0
BIT_DEPTH_8 = 255.0

def load_image(file_path: str) -> Tuple[Optional[npt.NDArray[np.float32]], Optional[float], Optional[list[str]]]:
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
        return None, None, None

    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None, None, None

MODE_CHANNEL_MAP: Dict[str, list[str]] = {
    'L': ['L'],
    'LA': ['L', 'A'],
    'RGB': ['R', 'G', 'B'],
    'RGBA': ['R', 'G', 'B', 'A'],
    'I;16': ['L'],
    'CMYK': ['C', 'M', 'Y', 'K'],
    'YCbCr': ['Y', 'Cb', 'Cr'],
    'LAB': ['L', 'a', 'b']
}

def load_exr(file_path: str) -> Tuple[np.ndarray, float, list[str]]: # Исправлено: List -> list
    """Загружает EXR файл с обработкой каналов"""
    img = pyexr.read(file_path).astype(np.float32)
    max_val = np.max(np.abs(img))
    max_val = max(max_val, 1e-6)

    channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L']
    return img, max_val, channels

def load_raster(file_path: str) -> Tuple[npt.NDArray[np.float32], float, list[str]]:
    img = Image.open(file_path)

    if img.mode not in MODE_CHANNEL_MAP:
        img = img.convert('RGB')

    divisor = BIT_DEPTH_16 if img.mode.startswith('I;16') else BIT_DEPTH_8
    img_array = np.array(img).astype(np.float32) / divisor
    channels = MODE_CHANNEL_MAP[img.mode]

    return img_array, 1.0, channels
    # return img_array, float(divisor), channels