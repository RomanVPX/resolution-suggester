import cv2
import numpy as np
import numpy.typing as npt
from functools import lru_cache
from typing import Callable
from numba import njit, prange
from config import INTERPOLATION_METHODS, InterpolationMethod, MITCHELL_B, MITCHELL_C

TINY = 1e-8
MITCHELL_RADIUS = 2  # Радиус фильтра Митчелла

ResizeFunction = Callable[[npt.NDArray[np.float32], int, int], npt.NDArray[np.float32]]

@njit(cache=True)
def mitchell_netravali(x: float, B: float = MITCHELL_B, C: float = MITCHELL_C) -> float:
    x = abs(x)
    x2 = x * x
    x3 = x2 * x
    if x < 1.0:
        coeff1 = 12 - 9 * B - 6 * C
        coeff2 = -18 + 12 * B + 6 * C
        coeff3 = 6 - 2 * B
        return coeff1 * x3 + coeff2 * x2 + coeff3
    elif x < 2.0:
        coeff1 = -B - 6 * C
        coeff2 = 6 * B + 30 * C
        coeff3 = -12 * B - 48 * C
        coeff4 = 8 * B + 24 * C
        return coeff1 * x3 + coeff2 * x2 + coeff3 * x + coeff4
    return 0.0

@njit(cache=True)
def _resize_single_channel(
    channel: npt.NDArray[np.float32],
    target_width: int,
    target_height: int,
    B: float,
    C: float
) -> npt.NDArray[np.float32]:
    """Ресайз одного канала (2D) фильтром Митчелла без чанкинга."""
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

            resized[i, j] = accumulator / (weight_sum + TINY)

    return resized

@njit(parallel=True, cache=True)
def _resize_mitchell_impl(
    img: npt.NDArray[np.float32],
    target_width: int,
    target_height: int,
    B: float = MITCHELL_B,
    C: float = MITCHELL_C
) -> npt.NDArray[np.float32]:
    """Полный митчелловский ресайз без чанкинга (канальная параллелизация)."""
    if img.ndim == 2:
        return _resize_single_channel(img, target_width, target_height, B, C)

    channels = img.shape[2]
    resized = np.empty((target_height, target_width, channels), dtype=img.dtype)

    for c in prange(channels):
        resized[:, :, c] = _resize_single_channel(img[:, :, c], target_width, target_height, B, C)

    return resized

def _resize_mitchell_chunked_refined(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    chunk_size: int,
    B: float,
    C: float
) -> np.ndarray:
    """
    Более точный чанковый ресайз:
     - Пробегаемся по тайлам в пространстве (target).
     - Для каждого тайла вычисляем соответствующий фрагмент (source)
       и слегка расширяем границы (на 2 пикселя, радиус Митчелла).
     - Масштабируем фрагмент, вставляем результат в нужное место выходного массива.
    """
    src_h, src_w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]

    out_shape = (target_height, target_width) if channels == 1 else (target_height, target_width, channels)
    resized_final = np.zeros(out_shape, dtype=np.float32)

    # Коэффициенты (сколько пикселей source на 1 пиксель target)
    x_ratio = float(src_w) / float(target_width)
    y_ratio = float(src_h) / float(target_height)

    def resize_one_channel(local_channel: np.ndarray, tw: int, th: int) -> np.ndarray:
        return _resize_single_channel(local_channel, tw, th, B, C)

    # Функция ресайза цветного фрагмента или моно
    def resize_local_block(local_block: np.ndarray, tw: int, th: int) -> np.ndarray:
        if local_block.ndim == 2:
            return resize_one_channel(local_block, tw, th)
        else:
            # Раскладываем каналы
            ch_count = local_block.shape[2]
            out_block = []
            for cc in range(ch_count):
                out_block.append(resize_one_channel(local_block[..., cc], tw, th))
            return np.stack(out_block, axis=-1)

    # Бежим по чанкам в target-пространстве
    for ty0 in range(0, target_height, chunk_size):
        ty1 = min(ty0 + chunk_size, target_height)

        for tx0 in range(0, target_width, chunk_size):
            tx1 = min(tx0 + chunk_size, target_width)

            # Размер текущего тайла в target
            tile_w = tx1 - tx0
            tile_h = ty1 - ty0

            # Соответствующие координаты в source (с учётом округления)
            sx0_float = tx0 * x_ratio
            sx1_float = tx1 * x_ratio
            sy0_float = ty0 * y_ratio
            sy1_float = ty1 * y_ratio

            # Округляем (используем floor/ceil)
            sx0 = int(np.floor(sx0_float))
            sx1 = int(np.ceil(sx1_float))
            sy0 = int(np.floor(sy0_float))
            sy1 = int(np.ceil(sy1_float))

            # Расширяем на 2 пикселя радиуса Митчелла
            sx0_ext = max(sx0 - MITCHELL_RADIUS, 0)
            sx1_ext = min(sx1 + MITCHELL_RADIUS, src_w)
            sy0_ext = max(sy0 - MITCHELL_RADIUS, 0)
            sy1_ext = min(sy1 + MITCHELL_RADIUS, src_h)

            # Вырезаем фрагмент из исходника
            if channels == 1:
                local_data = img[sy0_ext:sy1_ext, sx0_ext:sx1_ext]
            else:
                local_data = img[sy0_ext:sy1_ext, sx0_ext:sx1_ext, :]

            # Масштабируем фрагмент до размеров (tile_h x tile_w)
            local_resized = resize_local_block(local_data, tile_w, tile_h)

            # Кладём в итоговый массив
            if channels == 1:
                resized_final[ty0:ty1, tx0:tx1] = local_resized
            else:
                resized_final[ty0:ty1, tx0:tx1, :] = local_resized

    return resized_final.squeeze() if channels == 1 else resized_final


def resize_mitchell(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    B: float = MITCHELL_B,
    C: float = MITCHELL_C,
    chunk_size: int = 0
) -> np.ndarray:
    """
    Публичный интерфейс Митчелла. Если chunk_size>0, включается refined-чанкинг.
    """
    if chunk_size > 0:
        return _resize_mitchell_chunked_refined(img, target_width, target_height, chunk_size, B, C)
    else:
        return _resize_mitchell_impl(img, target_width, target_height, B, C)

@lru_cache(maxsize=4)
def get_resize_function(interpolation: str, chunk_size: int = 0) -> ResizeFunction:
    """
    Фабрика функций для ресайза:
    Если выбрали 'mitchell', берём resize_mitchell. chunk_size управляет чанкингом.
    """
    try:
        interpolation_method = InterpolationMethod(interpolation)
    except ValueError:
        raise ValueError(f"Unsupported interpolation method: {interpolation}")

    if interpolation_method == InterpolationMethod.MITCHELL:
        # Возвращаем функцию Митчелла с нужным chunk_size
        from functools import partial
        return partial(resize_mitchell, chunk_size=chunk_size)

    try:
        cv2_flag = getattr(cv2, INTERPOLATION_METHODS[interpolation_method])
    except AttributeError:
        raise ValueError(f"OpenCV interpolation method not found: {interpolation_method}")

    def opencv_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
        return np.asarray(cv2.resize(img, (w, h), interpolation=cv2_flag), dtype=np.float32)

    return opencv_resize
