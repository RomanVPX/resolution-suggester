# image_processing.py
import cv2
import numpy as np
import numpy.typing as npt
from functools import lru_cache
from typing import Callable
from numba import njit, prange
from config import MITCHELL_B, MITCHELL_C, TINY_EPSILON, INTERPOLATION_METHODS_CV2, InterpolationMethod

ResizeFunction = Callable[[npt.NDArray[np.float32], int, int], npt.NDArray[np.float32]]

@njit(cache=True)
def mitchell_netravali(
        x: float,
        cf_b: float = MITCHELL_B,
        cf_c: float = MITCHELL_C
) -> float:
    """
    Mitchell-Netravali filter kernel.
    """
    x = abs(x)
    x2 = x * x
    x3 = x2 * x
    if x < 1.0:
        cf_1 = 12 - 9 * cf_b - 6 * cf_c
        cf_2 = -18 + 12 * cf_b + 6 * cf_c
        cf_3 = 6 - 2 * cf_b
        return (cf_1 * x3 +
                cf_2 * x2 +
                cf_3)
    if x < 2.0:
        cf_1 = -cf_b - 6 * cf_c
        cf_2 = 6 * cf_b + 30 * cf_c
        cf_3 = -12 * cf_b - 48 * cf_c
        cf_4 = 8 * cf_b + 24 * cf_c
        return (cf_1 * x3 +
                cf_2 * x2 +
                cf_3 * x +
                cf_4)
    return 0.0

@njit(cache=True)
def _resize_mitchell_single_channel(
    channel: npt.NDArray[np.float32],
    target_width: int,
    target_height: int,
    cf_b: float,
    cf_c: float
) -> npt.NDArray[np.float32]:
    """
    Resizes a single-channel image using the Mitchell-Netravali filter.
    """
    height, width = channel.shape
    resized = np.zeros((target_height, target_width), dtype=channel.dtype)

    # Ранний выход, если размеры совпадают
    if target_width == width and target_height == height:
        return channel.copy()

    x_ratio = width / target_width
    y_ratio = height / target_height

    # Предварительный расчет весов для X и Y
    x_weights = np.zeros((target_width, 4), dtype=np.float32)
    x_indices = np.zeros((target_width, 4), dtype=np.int32)
    y_weights = np.zeros((target_height, 4), dtype=np.float32)
    y_indices = np.zeros((target_height, 4), dtype=np.int32)

    # Предварительный расчет весов и индексов
    for j in range(target_width):
        x = j * x_ratio
        x_floor = int(np.floor(x))
        for k in range(4):
            x_idx = x_floor + k - 1
            if 0 <= x_idx < width:
                x_indices[j, k] = x_idx
                x_weights[j, k] = mitchell_netravali(x - x_floor - (k - 1), cf_b, cf_c)

    for i in range(target_height):
        y = i * y_ratio
        y_floor = int(np.floor(y))
        for k in range(4):
            y_idx = y_floor + k - 1
            if 0 <= y_idx < height:
                y_indices[i, k] = y_idx
                y_weights[i, k] = mitchell_netravali(y - y_floor - (k - 1), cf_b, cf_c)

    # Основной цикл ресайза с использованием предварительно рассчитанных значений
    for i in range(target_height):
        for j in range(target_width):
            accumulator = 0.0
            weight_sum = 0.0
            for ky in range(4):
                y_idx = y_indices[i, ky]
                wy = y_weights[i, ky]
                if wy == 0.0:
                    continue

                for kx in range(4):
                    x_idx = x_indices[j, kx]
                    wx = x_weights[j, kx]
                    if wx == 0.0:
                        continue

                    if 0 <= x_idx < width and 0 <= y_idx < height:
                        weight = wx * wy
                        accumulator += weight * channel[y_idx, x_idx]
                        weight_sum += weight

            resized[i, j] = accumulator / (weight_sum + TINY_EPSILON)

    return resized

@njit(parallel=True, cache=True, fastmath=True)
def _resize_mitchell(
    img: npt.NDArray[np.float32],
    target_width: int,
    target_height: int,
    cf_b: float = MITCHELL_B,
    cf_c: float = MITCHELL_C
) -> npt.NDArray[np.float32]:
    """
    Full Mitchell-based resizing without chunking (channel parallelization).
    """
    if img.ndim == 2:
        return _resize_mitchell_single_channel(img, target_width, target_height, cf_b, cf_c)

    channels = img.shape[2]
    resized = np.empty((target_height, target_width, channels), dtype=img.dtype)

    for c in prange(channels):
        resized[:, :, c] = _resize_mitchell_single_channel(
            img[:, :, c], target_width, target_height, cf_b, cf_c
        )

    return resized

def resize_mitchell(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    cf_b: float = MITCHELL_B,
    cf_c: float = MITCHELL_C,
) -> np.ndarray:
    """
    Public interface for Mitchell resizing.
    """
    return _resize_mitchell(img, target_width, target_height, cf_b, cf_c)

@lru_cache(maxsize=4)
def get_resize_function(interpolation: InterpolationMethod) -> ResizeFunction:
    """
    Factory function for resize operations.
    If 'mitchell' is selected, uses resize_mitchell.
    Otherwise, uses OpenCV's interpolation methods.
    """
    if interpolation == InterpolationMethod.MITCHELL:
        return resize_mitchell
    else:
        cv2_interpolation_flag_name = INTERPOLATION_METHODS_CV2.get(interpolation)
        if cv2_interpolation_flag_name is None:
            raise ValueError(f"Метод интерполяции OpenCV не найден в списке INTERPOLATION_METHODS_CV2: {interpolation}")

        try:
            cv2_flag = getattr(cv2, cv2_interpolation_flag_name)
        except AttributeError as exc:
            raise ValueError(f"Метод интерполяции не найден в OpenCV: {interpolation}") from exc

        def opencv_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
            # Perform resize using OpenCV
            out = cv2.resize(img, (w, h), interpolation=cv2_flag)

            # Если out.ndim == 2, то OpenCV вернул двумерное изображение (H, W);
            # для согласованности с данными вида (H, W, 1) добавляем ось канала.
            if out.ndim == 2:
                out = out[..., np.newaxis]

            # Переводим к float32
            return np.asarray(out, dtype=np.float32)

        return opencv_resize