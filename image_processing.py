# image_processing.py
import cv2
import numpy as np
from functools import lru_cache
from numba import njit, prange
from config import INTERPOLATION_METHODS, InterpolationMethod # Импорт Enum


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

            for row_offset in range(-2, 2):
                for col_offset in range(-2, 2):
                    x_idx = int(x_floor + col_offset)  # col_offset для горизонтального смещения
                    y_idx = int(y_floor + row_offset)  # row_offset для вертикального смещения

                    if 0 <= x_idx < width and 0 <= y_idx < height: # Проверка границ изображения
                        weight_x = mitchell_netravali(x - x_floor - col_offset, B, C) # Вес по X
                        weight_y = mitchell_netravali(y - y_floor - row_offset, B, C) # Вес по Y
                        weight = weight_x * weight_y # Общий вес как произведение весов по X и Y
                        pixel = img[y_idx, x_idx]
                        accumulator += weight * pixel # Накопление взвешенных значений пикселей
                        weight_sum += weight # Суммирование весов

            if weight_sum > 0:
                resized[i, j] = accumulator / weight_sum # Нормализация накопленного значения на сумму весов
            else:
                resized[i, j] = np.zeros_like(accumulator) # Заполнение нулями, если сумма весов равна 0

    return resized


def resize_mitchell(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Публичная функция для ресайза с фильтром Митчелла-Нетравали"""
    resized = _resize_mitchell_impl(img, target_width, target_height)
    return resized[:, :, 0] if img.ndim == 2 else resized


@lru_cache(maxsize=4)
def get_resize_function(interpolation: str):
    """Фабрика функций для ресайза"""
    interpolation_method = InterpolationMethod(interpolation) # Преобразование строки в Enum
    if interpolation_method == InterpolationMethod.MITCHELL:
        return resize_mitchell

    cv2_flag = getattr(cv2, INTERPOLATION_METHODS[interpolation_method], cv2.INTER_LINEAR) # Используем Enum как ключ
    return lambda img, w, h: cv2.resize(img, (w, h), interpolation=cv2_flag)
