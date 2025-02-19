import math
import numpy as np
from numba import njit, prange
from typing import Dict, Tuple
from config import TINY_EPSILON

@njit(cache=True)
def calculate_psnr(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float
) -> float:
    if original.shape != processed.shape:
        raise ValueError("Размеры изображений должны совпадать для расчета PSNR")

    diff = original - processed
    mse = np.mean(diff * diff)

    if mse < TINY_EPSILON:
        return float('inf')

    log_max = 20 * math.log10(max_val)
    return log_max - 10 * math.log10(mse)

def calculate_channel_psnr(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    channels: list[str]
) -> Dict[str, float]:
    return {
        channel: calculate_psnr(original[..., i], processed[..., i], max_val)
        for i, channel in enumerate(channels)
    }

@njit(cache=True)
def gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """
    Формирует 2D-ядро Гаусса в nopython-режиме Numba
    (без np.meshgrid и np.linspace).
    """
    result = np.zeros((window_size, window_size), dtype=np.float32)
    half = (window_size - 1) / 2.0

    sum_val = 0.0
    for i in range(window_size):
        for j in range(window_size):
            x = i - half
            y = j - half
            val = math.exp(-0.5 * (x*x + y*y) / (sigma*sigma))
            result[i, j] = val
            sum_val += val

    # Нормируем ядро
    for i in range(window_size):
        for j in range(window_size):
            result[i, j] /= sum_val

    return result

@njit(cache=True)
def filter_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Свёртка (H,W) или (H,W,C) с ядром kernel (kH,kW) с отражающим (reflect) паддингом.
    Все операции совместимы с nopython-режимом.
    """
    if img.ndim == 2:
        return _filter_2d_singlechannel(img, kernel)
    else:
        (h, w, c) = img.shape
        out = np.zeros_like(img)
        for ch in range(c):
            out[..., ch] = _filter_2d_singlechannel(img[..., ch], kernel)
        return out

@njit(parallel=True, cache=True)
def _filter_2d_singlechannel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    result = np.zeros((h, w), dtype=channel.dtype)

    for i in prange(h):  # <-- заменяем range на prange
        for j in range(w):
            accum = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    src_i = i + ki - pad_h
                    src_j = j + kj - pad_w
                    src_i = reflect_coordinate(src_i, h)
                    src_j = reflect_coordinate(src_j, w)
                    accum += channel[src_i, src_j] * kernel[ki, kj]
            result[i, j] = accum

    return result


@njit(cache=True)
def reflect_coordinate(x: int, size: int) -> int:
    """
    Отражающее переотображение индекса x, если он выходит за границы [0..size-1].
    Симметричный reflect, аналог OpenCV BORDER_REFLECT_101.
    """
    if x < 0:
        return -x - 1  # отражаем ниже нуля
    elif x >= size:
        return 2*size - x - 1  # отражаем выше размера
    return x

@njit(cache=True)
def calculate_ssim_gauss_single(
    original: np.ndarray,
    processed: np.ndarray,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5
) -> float:
    if original.shape != processed.shape:
        raise ValueError("SSIM: размеры изображений должны совпадать")

    kernel = gaussian_kernel(window_size, sigma)

    muX = filter_2d(original, kernel)
    muY = filter_2d(processed, kernel)

    sigmaX = filter_2d(original * original, kernel) - muX * muX
    sigmaY = filter_2d(processed * processed, kernel) - muY * muY
    sigmaXY = filter_2d(original * processed, kernel) - muX * muY

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    numerator   = (2.0 * muX * muY + C1) * (2.0 * sigmaXY + C2)
    denominator = (muX**2 + muY**2 + C1) * (sigmaX + sigmaY + C2)
    ssim_map    = numerator / np.maximum(denominator, TINY_EPSILON)

    return float(np.mean(ssim_map))

def calculate_ssim_gauss(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    K1: float = 0.01,
    K2: float = 0.03,
    window_size: int = 11,
    sigma: float = 1.5
) -> float:
    if original.shape != processed.shape:
        raise ValueError("SSIM: размеры изображений должны совпадать")

    if max_val > 1.00001:
        original  = original  / max_val
        processed = processed / max_val

    if original.ndim == 2:
        return calculate_ssim_gauss_single(original, processed, K1, K2, 1.0, window_size, sigma)
    elif original.ndim == 3:
        c = original.shape[2]
        ssim_sum = 0.0
        for i in range(c):
            ssim_sum += calculate_ssim_gauss_single(
                original[..., i],
                processed[..., i],
                K1, K2, 1.0, window_size, sigma
            )
        return ssim_sum / c
    else:
        raise ValueError("Неподдерживаемая размерность изображения для SSIM")

def calculate_channel_ssim_gauss(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    channels: list[str],
    window_size: int = 11,
    sigma: float = 1.5
) -> Dict[str, float]:
    if original.shape != processed.shape:
        raise ValueError("SSIM: размеры изображений должны совпадать")

    if max_val > 1.00001:
        original  = original  / max_val
        processed = processed / max_val

    # Если у нас оказалось (H,W), тогда это один канал:
    if original.ndim == 2:
        return {'L': calculate_ssim_gauss_single(original, processed, window_size=window_size, sigma=sigma)}

    # Для (H,W,C)
    ssim_dict: Dict[str, float] = {}
    for i, ch in enumerate(channels):
        ssim_dict[ch] = calculate_ssim_gauss_single(
            original[..., i],
            processed[..., i],
            window_size=window_size,
            sigma=sigma
        )

    return ssim_dict

def compute_resolutions(
    original_width: int,
    original_height: int,
    min_size: int = 16
) -> list[Tuple[int, int]]:
    resolutions = []
    w, h = original_width, original_height
    while w >= min_size * 2 and h >= min_size * 2:
        w //= 2
        h //= 2
        resolutions.append((w, h))
    return resolutions
