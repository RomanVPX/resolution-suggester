# metrics.py
import math
import numpy as np
from numba import njit, prange
from sewar.full_ref import msssim
from config import TINY_EPSILON, QualityMetric


def calculate_ms_ssim(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float
) -> float:
    if original.shape != processed.shape:
        raise ValueError("MS-SSIM: размеры изображений должны совпадать")

    # Нормализация для EXR
    if max_val > 1.0 + 1e-5:
        original = original.astype(np.float32) / max_val
        processed = processed.astype(np.float32) / max_val
        data_range = 1.0
    else:
        data_range = max_val

    # Для многоканальных изображений sewar ожидает (H,W,C)
    if original.ndim == 2:
        original = original[..., np.newaxis]
        processed = processed[..., np.newaxis]

    # Защита от артефактов округления
    original = np.clip(original, 0.0, 1.0)
    processed = np.clip(processed, 0.0, 1.0)

    return float(np.real(msssim(original, processed, MAX=data_range)))

def calculate_ms_ssim_channels(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    channels: list[str]
) -> dict[str, float]:
    results = {}
    for i, ch in enumerate(channels):
        orig_ch = original[..., i] if original.ndim == 3 else original
        proc_ch = processed[..., i] if processed.ndim == 3 else processed
        results[ch] = calculate_ms_ssim(orig_ch, proc_ch, max_val)
    return results

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

def calculate_psnr_channels(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    channels: list[str]
) -> dict[str, float]:
    return {
        channel: calculate_psnr(original[..., i], processed[..., i], max_val)
        for i, channel in enumerate(channels)
    }

def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Формирует 1D-ядро Гаусса длиной size с параметром sigma.
    Все операции совместимы с nopython-режимом Numba.
    """
    kernel_1d = np.empty(size, dtype=np.float32)
    half = (size - 1) / 2.0

    sum_val = 0.0
    for i in range(size):
        x = i - half
        val = math.exp(-0.5 * (x*x) / (sigma*sigma))
        kernel_1d[i] = val
        sum_val += val

    # Нормируем, чтобы сумма коэффициентов была = 1
    for i in range(size):
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
    if x >= size:
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

    # (H,W,C) — многоканальное изображение
    h, w, c = img.shape
    result = np.zeros_like(img)
    for ch in range(c):
        tmp = convolve_1d_horiz(img[..., ch], kernel_1d)
        out = convolve_1d_vert(tmp, kernel_1d)
        result[..., ch] = out
    return result


def _calculate_ssim_gauss_single(
    original: np.ndarray,
    processed: np.ndarray,
    k_1: float = 0.01,
    k_2: float = 0.03,
    l: float  = 1.0,
    window_size: int = 11,
    sigma: float = 1.5
) -> float:
    if original.shape != processed.shape:
        raise ValueError("SSIM: размеры изображений должны совпадать")

    # Фильтруем средние
    mu_x = filter_2d_separable(original, window_size, sigma)
    mu_y = filter_2d_separable(processed, window_size, sigma)

    # Фильтруем X^2, Y^2 и X*Y
    sigma_x = filter_2d_separable(original*original, window_size, sigma) - mu_x*mu_x
    sigma_y = filter_2d_separable(processed*processed, window_size, sigma) - mu_y*mu_y
    sigma_xy = filter_2d_separable(original*processed, window_size, sigma) - mu_x*mu_y

    c1 = (k_1 * l) ** 2
    c2 = (k_2 * l) ** 2

    numerator   = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

    ssim_map = numerator / np.maximum(denominator, TINY_EPSILON)
    return float(np.mean(ssim_map))


def calculate_ssim_gauss(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    k_1: float = 0.01,
    k_2: float = 0.03,
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
        return _calculate_ssim_gauss_single(original, processed, k_1, k_2, 1.0, window_size, sigma)
    if original.ndim == 3:
        c = original.shape[2]
        ssim_sum = 0.0
        for i in range(c):
            ssim_sum += _calculate_ssim_gauss_single(
                original[..., i],
                processed[..., i],
                k_1, k_2, 1.0, window_size, sigma
            )
        return ssim_sum / c

    raise ValueError("Неподдерживаемая размерность изображения для SSIM")

def calculate_ssim_gauss_channels(
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
            'L': _calculate_ssim_gauss_single(
                original, processed,
                window_size=window_size, sigma=sigma
            )
        }

    # (H,W,C)
    ssim_dict: dict[str, float] = {}
    for i, ch in enumerate(channels):
        ssim_val = _calculate_ssim_gauss_single(
            original[..., i],
            processed[..., i],
            window_size=window_size, sigma=sigma
        )
        ssim_dict[ch] = ssim_val

    return ssim_dict

def calculate_metrics(
    quality_metric: QualityMetric,
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    channels: list[str] = None
) -> None | dict[str, float] | float:

    if original.shape != processed.shape:
        raise ValueError("compute_metrics: размеры изображений должны совпадать!")
    if original.ndim != processed.ndim:
        raise ValueError("compute_metrics: размерности изображений должны совпадать!")

    if channels is None:
        match quality_metric:
            case QualityMetric.PSNR:
                return calculate_psnr(original, processed, max_val)
            case QualityMetric.SSIM:
                return calculate_ssim_gauss(original, processed, max_val)
            case QualityMetric.MS_SSIM:
                return calculate_ms_ssim(original, processed, max_val)
    else:
        match quality_metric:
            case QualityMetric.PSNR:
                return calculate_psnr_channels(original, processed, max_val, channels)
            case QualityMetric.SSIM:
                return calculate_ssim_gauss_channels(original, processed, max_val, channels)
            case QualityMetric.MS_SSIM:
                return calculate_ms_ssim_channels(original, processed, max_val, channels)

    raise ValueError(f"Неподдерживаемая метрика: {quality_metric}")


def compute_resolutions(original_width: int, original_height: int, min_size: int = 16, divider: int = 2) -> list[tuple[int, int]]:
    resolutions = []
    w, h = original_width, original_height
    while w >= min_size and h >= min_size:
        resolutions.append((w, h))
        w //= divider
        h //= divider
    return resolutions
