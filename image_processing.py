import cv2
import numpy as np
import numpy.typing as npt
from functools import lru_cache, partial
from typing import Callable
from numba import njit, prange
from config import MITCHELL_B, MITCHELL_C, TINY_EPSILON, INTERPOLATION_METHODS_CV2, InterpolationMethod

ResizeFunction = Callable[[npt.NDArray[np.float32], int, int], npt.NDArray[np.float32]]

@njit(cache=True)
def mitchell_netravali(x: float, B: float = MITCHELL_B, C: float = MITCHELL_C) -> float:
    """
    Ядро фильтра Mitchell-Netravali.
    """
    x = abs(x)
    x2 = x * x
    x3 = x2 * x
    if x < 1.0:
        coeff1 = 12 - 9 * B - 6 * C
        coeff2 = -18 + 12 * B + 6 * C
        coeff3 = 6 - 2 * B
        return (coeff1 * x3 +
                coeff2 * x2 +
                coeff3)
    if x < 2.0:
        coeff1 = -B - 6 * C
        coeff2 = 6 * B + 30 * C
        coeff3 = -12 * B - 48 * C
        coeff4 = 8 * B + 24 * C
        return (coeff1 * x3 +
                coeff2 * x2 +
                coeff3 * x +
                coeff4)
    return 0.0

@njit(cache=True)
def _resize_mitchell_single_channel(
    channel: npt.NDArray[np.float32],
    target_width: int,
    target_height: int,
    B: float,
    C: float
) -> npt.NDArray[np.float32]:
    """
    Ресайз одноканального изображения с использованием фильтра Mitchell-Netravali..
    """
    height, width = channel.shape
    resized = np.zeros((target_height, target_width), dtype=channel.dtype)

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
                x_weights[j, k] = mitchell_netravali(x - x_floor - (k - 1), B, C)

    for i in range(target_height):
        y = i * y_ratio
        y_floor = int(np.floor(y))
        for k in range(4):
            y_idx = y_floor + k - 1
            if 0 <= y_idx < height:
                y_indices[i, k] = y_idx
                y_weights[i, k] = mitchell_netravali(y - y_floor - (k - 1), B, C)

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

@njit(parallel=True, cache=True)
def _resize_mitchell(
    img: npt.NDArray[np.float32],
    target_width: int,
    target_height: int,
    B: float = MITCHELL_B,
    C: float = MITCHELL_C
) -> npt.NDArray[np.float32]:
    """
    Полный митчелловский ресайз без чанкинга (канальная параллелизация).
    """
    if img.ndim == 2:
        return _resize_mitchell_single_channel(img, target_width, target_height, B, C)

    channels = img.shape[2]
    resized = np.empty((target_height, target_width, channels), dtype=img.dtype)

    for c in prange(channels):
        resized[:, :, c] = _resize_mitchell_single_channel(img[:, :, c], target_width, target_height, B, C)

    return resized

def resize_mitchell(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    B: float = MITCHELL_B,
    C: float = MITCHELL_C,
) -> np.ndarray:
    """
    Публичный интерфейс Митчелла.
    """
    return _resize_mitchell(img, target_width, target_height, B, C)

@lru_cache(maxsize=4)
def get_resize_function(interpolation: InterpolationMethod) -> ResizeFunction:
    """
    Фабрика функций для ресайза:
    Если выбрали 'mitchell', берём resize_mitchell.
    """
    if interpolation == InterpolationMethod.MITCHELL:
        return partial(resize_mitchell)

    try:
        cv2_flag = getattr(cv2, INTERPOLATION_METHODS_CV2[interpolation])
    except AttributeError as exc:
        raise ValueError(f'Метод интерполяции OpenCV не найден: {interpolation}') from exc

    def opencv_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
        # Выполняем ресайз через OpenCV
        out = cv2.resize(img, (w, h), interpolation=cv2_flag)

        # Если out.ndim == 2, то OpenCV вернул двумерное изображение (H, W);
        # для согласованности с данными вида (H, W, 1) добавляем ось канала.
        if out.ndim == 2:
            out = out[..., np.newaxis]

        # Переводим к float32
        return np.asarray(out, dtype=np.float32)

    return opencv_resize