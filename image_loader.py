# image_loader.py
import logging
import numpy as np
import pyexr
from PIL import Image, ImageFile
from typing import Tuple, Optional
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(file_path: str) -> Tuple[Optional[np.ndarray], Optional[float], Optional[list[str]]]:
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

def load_exr(file_path: str) -> Tuple[np.ndarray, float, list[str]]: # Исправлено: List -> list
    """Загружает EXR файл с обработкой каналов"""
    img = pyexr.read(file_path).astype(np.float32)
    max_val = np.max(np.abs(img))
    max_val = max(max_val, 1e-6)

    channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L']
    return img, max_val, channels

def load_raster(file_path: str) -> tuple[np.ndarray, float, list[str]]:
    img = Image.open(file_path)
    mode_conversion = {
        '1': 'L',
        'L': 'L',
        'LA': 'LA',
        'RGB': 'RGB',
        'RGBA': 'RGBA',
        'CMYK': 'RGB',
        'YCbCr': 'RGB',
        'LAB': 'RGB',
        'HSV': 'RGB',
        'I;16': 'I;16'
    }

    target_mode = mode_conversion.get(img.mode, 'RGB')
    if target_mode and img.mode != target_mode:
        img = img.convert(target_mode)

    # Улучшенная обработка битности
    divisor = 65535.0 if img.mode.startswith('I;16') else 255.0

    img_array = np.array(img).astype(np.float32) / divisor
    channels = _get_channels_description(img.mode)

    return img_array, float(divisor), channels

def _get_channels_description(mode: str) -> list[str]:
    channel_map = {
        'L': ['L'],
        'LA': ['L', 'A'],
        'RGB': ['R', 'G', 'B'],
        'RGBA': ['R', 'G', 'B', 'A'],
        'I;16': ['L']
    }
    return channel_map.get(mode, ['Unknown'])