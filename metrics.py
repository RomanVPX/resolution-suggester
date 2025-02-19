import math
import numpy as np
from numba import njit, prange
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
) -> dict[str, float]:
    return {
        channel: calculate_psnr(original[..., i], processed[..., i], max_val)
        for i, channel in enumerate(channels)
    }

@njit(parallel=True, cache=True)
def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Формирует 1D-ядро Гаусса длиной size с параметром sigma.
    Все операции совместимы с nopython-режимом Numba.
    """
    kernel_1d = np.empty(size, dtype=np.float32)
    half = (size - 1) / 2.0

    sum_val = 0.0
    for i in prange(size):
        x = i - half
        val = math.exp(-0.5 * (x*x) / (sigma*sigma))
        kernel_1d[i] = val
        sum_val += val

    # Нормируем, чтобы сумма коэффициентов была = 1
    for i in prange(size):
        kernel_1d[i] /= sum_val

    return kernel_1d

@njit(cache=True)
def reflect_coordinate(x: int, size: int) -> int:
    """
    Отражающее переотображение индекса x внутри [0..size-1]
    (аналог OpenCV BORDER_REFLECT_101).
    """
    if x < 0:
        return -x - 1
    elif x >= size:
        return 2*size - x - 1
    return x

@njit(parallel=True, cache=True)
def convolve_1d_horiz(src: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """
    Горизонтальная 1D-свёртка изображения src (H,W).
    Отражающий паддинг по горизонтали.
    """
    h, w = src.shape
    ksize = kernel_1d.size
    pad = ksize // 2

    out = np.zeros_like(src)
    for i in prange(h):
        for j in prange(w):
            accum = 0.0
            for k in prange(ksize):
                jj = j + k - pad
                jj = reflect_coordinate(jj, w)
                accum += src[i, jj] * kernel_1d[k]
            out[i, j] = accum
    return out

@njit(parallel=True, cache=True)
def convolve_1d_vert(src: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """
    Вертикальная 1D-свёртка изображения src (H,W).
    Отражающий паддинг по вертикали.
    """
    h, w = src.shape
    ksize = kernel_1d.size
    pad = ksize // 2

    out = np.zeros_like(src)
    for j in prange(w):
        for i in prange(h):
            accum = 0.0
            for k in prange(ksize):
                ii = i + k - pad
                ii = reflect_coordinate(ii, h)
                accum += src[ii, j] * kernel_1d[k]
            out[i, j] = accum
    return out

@njit(cache=True)
def filter_2d_separable(img: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """
    Разделяемая свёртка Гаусса для (H,W) или (H,W,C).
    1) Генерируем 1D ядро gauss_1d.
    2) Горизонтальная свёртка.
    3) Вертикальная свёртка.
    """
    # Создаём 1D ядро Гаусса
    kernel_1d = gaussian_kernel_1d(size, sigma)

    if img.ndim == 2:
        # (H,W) — одиночный канал
        tmp = convolve_1d_horiz(img, kernel_1d)
        out = convolve_1d_vert(tmp, kernel_1d)
        return out
    else:
        # (H,W,C) — многоканальное изображение
        h, w, c = img.shape
        result = np.zeros_like(img)
        for ch in range(c):
            tmp = convolve_1d_horiz(img[..., ch], kernel_1d)
            out = convolve_1d_vert(tmp, kernel_1d)
            result[..., ch] = out
        return result


@njit(cache=True)
def calculate_ssim_gauss_single(
    original: np.ndarray,
    processed: np.ndarray,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float  = 1.0,
    window_size: int = 11,
    sigma: float = 1.5
) -> float:
    if original.shape != processed.shape:
        raise ValueError("SSIM: размеры изображений должны совпадать")

    # Фильтруем средние
    muX = filter_2d_separable(original, window_size, sigma)
    muY = filter_2d_separable(processed, window_size, sigma)

    # Фильтруем X^2, Y^2 и X*Y
    sigmaX = filter_2d_separable(original*original, window_size, sigma) - muX*muX
    sigmaY = filter_2d_separable(processed*processed, window_size, sigma) - muY*muY
    sigmaXY = filter_2d_separable(original*processed, window_size, sigma) - muX*muY

    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    numerator   = (2.0 * muX * muY + C1) * (2.0 * sigmaXY + C2)
    denominator = (muX**2 + muY**2 + C1) * (sigmaX + sigmaY + C2)

    ssim_map = numerator / np.maximum(denominator, TINY_EPSILON)
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

    # Нормализуем, если max_val > 1
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
) -> dict[str, float]:
    if original.shape != processed.shape:
        raise ValueError("SSIM: размеры изображений должны совпадать")

    if max_val > 1.00001:
        original  = original  / max_val
        processed = processed / max_val

    # Если (H,W), то один канал:
    if original.ndim == 2:
        return {
            'L': calculate_ssim_gauss_single(
                original, processed,
                window_size=window_size, sigma=sigma
            )
        }

    # (H,W,C)
    ssim_dict: dict[str, float] = {}
    for i, ch in enumerate(channels):
        ssim_val = calculate_ssim_gauss_single(
            original[..., i],
            processed[..., i],
            window_size=window_size, sigma=sigma
        )
        ssim_dict[ch] = ssim_val

    return ssim_dict

def compute_resolutions(
    original_width: int,
    original_height: int,
    min_size: int = 16
) -> list[tuple[int, int]]:
    resolutions = []
    w, h = original_width, original_height
    while w >= min_size * 2 and h >= min_size * 2:
        w //= 2
        h //= 2
        resolutions.append((w, h))
    return resolutions
