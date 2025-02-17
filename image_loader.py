# image_loader.py
import logging
import numpy as np
import pyexr
from PIL import Image, ImageFile
from typing import Tuple, Optional, List
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

def load_raster(file_path: str) -> Tuple[np.ndarray, float, list[str]]: # Исправлено: List -> list
    """Загружает PNG/TGA файл с нормализацией"""
    img = np.array(Image.open(file_path)).astype(np.float32) / 255.0
    channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L']
    return img, 1.0, channels
