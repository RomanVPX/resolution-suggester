import cv2
import numpy as np
from numba import njit, prange
from typing import Tuple
from config import INTERPOLATION_METHODS

@njit(cache=True)
def mitchell_netravali(x: float, B: float = 1 / 3, C: float = 1 / 3) -> float:
    """Фильтр Митчелла-Нетравали для резамплинга"""
    x = np.abs(x)
    if x < 1:
        return (12 - 9 * B - 6 * C) * x**3 + (-18 + 12 * B + 6 * C) * x**2 + (6 - 2 * B)
    if x < 2:
        return (-B - 6 * C) * x**3 + (6 * B + 30 * C) * x**2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)
    return 0.0


@njit(parallel=True, cache=True)
def _resize_mitchell_impl(
    img: np.ndarray, target_width: int, target_height: int, B: float = 1 / 3, C: float = 1 / 3
) -> np.ndarray:
    """Ядро реализации алгоритма Митчелла-Нетравали"""
    height, width = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    resized = np.zeros((target_height, target_width, channels), dtype=img.dtype)

    x_ratio = width / target_width
    y_ratio = height / target_height

    for i in prange(target_height):
        for j in range(target_width):
            x = j * x_ratio
            y = i * y_ratio
            x_floor = np.floor(x)
            y_floor = np.floor(y)

            accumulator = np.zeros(channels, dtype=np.float64)
            weight_sum = 0.0

            for m in range(-2, 2):
                for n in range(-2, 2):
                    x_idx = int(x_floor + n)
                    y_idx = int(y_floor + m)

                    if 0 <= x_idx < width and 0 <= y_idx < height:
                        weight = mitchell_netravali(x - x_floor - n, B, C) * mitchell_netravali(
                            y - y_floor - m, B, C
                        )
                        pixel = img[y_idx, x_idx]
                        accumulator += weight * pixel
                        weight_sum += weight

            if weight_sum > 0:
                resized[i, j] = accumulator / weight_sum
            else:
                resized[i, j] = np.zeros_like(accumulator)

    return resized


def resize_mitchell(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Публичная функция для ресайза с фильтром Митчелла-Нетравали"""
    resized = _resize_mitchell_impl(img, target_width, target_height)
    return resized[:, :, 0] if img.ndim == 2 else resized


def get_resize_function(interpolation: str):
    """Фабрика функций для ресайза"""
    if interpolation == "mitchell":
        return resize_mitchell

    cv2_flag_name = INTERPOLATION_METHODS[interpolation] # Получаем строковое имя флага cv2
    cv2_flag = getattr(cv2, cv2_flag_name) # Используем строковое имя для получения атрибута cv2
    return lambda img, w, h: cv2.resize(img, (w, h), interpolation=cv2_flag)
