# image_processing.py
import cv2
import numpy as np
import numpy.typing as npt
from functools import lru_cache
from typing import Callable
from numba import njit, prange
from config import INTERPOLATION_METHODS, InterpolationMethod, MITCHELL_B, MITCHELL_C


ResizeFunction = Callable[[npt.NDArray[np.float32], int, int], npt.NDArray[np.float32]]

@njit(cache=True)
def _resize_single_channel(
    channel: npt.NDArray[np.float32],
    target_width: int,
    target_height: int,
    B: float,
    C: float
) -> npt.NDArray[np.float32]:
    height, width = channel.shape
    resized = np.zeros((target_height, target_width), dtype=channel.dtype)

    x_ratio = width / target_width
    y_ratio = height / target_height

    for i in range(target_height):
        for j in range(target_width):
            x = j * x_ratio
            y = i * y_ratio
            x_floor = np.floor(x)
            y_floor = np.floor(y)

            accumulator = 0.0
            weight_sum = 0.0

            for row_offset in range(-1, 3):
                for col_offset in range(-1, 3):
                    x_idx = int(x_floor + col_offset)
                    y_idx = int(y_floor + row_offset)

                    if 0 <= x_idx < width and 0 <= y_idx < height:
                        weight_x = mitchell_netravali(x - x_floor - col_offset, B, C)
                        weight_y = mitchell_netravali(y - y_floor - row_offset, B, C)
                        weight = weight_x * weight_y
                        accumulator += weight * channel[y_idx, x_idx]
                        weight_sum += weight

            resized[i, j] = accumulator / (weight_sum + np.finfo(np.float32).tiny)

    return resized

@njit(parallel=True, cache=True)
def _resize_mitchell_impl(
    img: npt.NDArray[np.float32],
    target_width: int,
    target_height: int,
    B: float = MITCHELL_B,
    C: float = MITCHELL_C
) -> npt.NDArray[np.float32]:
    if img.ndim == 2:
        return _resize_single_channel(img, target_width, target_height, B, C)

    channels = img.shape[2]
    resized = np.empty((target_height, target_width, channels), dtype=img.dtype)

    for c in prange(channels):
        resized[:, :, c] = _resize_single_channel(img[:, :, c], target_width, target_height, B, C)

    return resized


@njit(cache=True)
def mitchell_netravali(x: float, B: float = MITCHELL_B, C: float = MITCHELL_C) -> float:
    x = np.abs(x)
    x2 = x * x
    x3 = x * x2

    if x < 1:
        return (12 - 9*B - 6*C)*x3 + (-18 + 12*B + 6*C)*x2 + (6 - 2*B)
    elif x < 2:
        return (-B - 6*C)*x3 + (6*B + 30*C)*x2 + (-12*B - 48*C)*x + (8*B + 24*C)
    return 0.0


def resize_mitchell(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    B: float = MITCHELL_B,
    C: float = MITCHELL_C) -> np.ndarray:
    """Добавлены параметры B и C в публичный интерфейс"""
    resized = _resize_mitchell_impl(img, target_width, target_height, B, C)
    return resized.squeeze() if img.ndim == 2 else resized


@lru_cache(maxsize=4)
def get_resize_function(interpolation: str) -> ResizeFunction:
    """
    Фабрика функций для ресайза с улучшенной типизацией

    Args:
        interpolation: метод интерполяции

    Returns:
        Callable: функция ресайза

    Raises:
        ValueError: при неподдерживаемом методе интерполяции
    """
    try:
        interpolation_method = InterpolationMethod(interpolation)
    except ValueError:
        raise ValueError(f"Unsupported interpolation method: {interpolation}")

    if interpolation_method == InterpolationMethod.MITCHELL:
        return resize_mitchell

    try:
        cv2_flag = getattr(cv2, INTERPOLATION_METHODS[interpolation_method])
        return lambda img, w, h: np.asarray(cv2.resize(img, (w, h), interpolation=cv2_flag), dtype=np.float32)
    except AttributeError:
        raise ValueError(f"OpenCV interpolation method not found: {interpolation_method}")