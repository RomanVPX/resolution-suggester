# metrics.py
import math
import numpy as np
from numba import njit
from typing import Dict, Tuple
from config import TINY_EPSILON
from scipy.signal import convolve2d

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

    # Предварительно вычисленный логарифм для max_val
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

def gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """
    Формирует 2D-ядро Гаусса.

    Args:
        window_size: Размер окна ядра (нечетное число, по умолчанию 11).
        sigma: Стандартное отклонение Гауссианы (по умолчанию 1.5).

    Returns:
        np.ndarray: Нормированное 2D-ядро Гаусса.
    """
    ax = np.linspace(-(window_size-1)/2., (window_size-1)/2., window_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def filter_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Свёртка img (H,W) или (H,W,C) с ядром kernel (kH,kW).
    На границах используется reflect-паддинг.

    Args:
        img: Входное изображение (H,W) или (H,W,C).
        kernel: Ядро свертки (kH,kW).

    Returns:
        np.ndarray: Результат свертки, изображение той же размерности, что и входное.
    """
    if img.ndim == 2:
        return _filter_2d_singlechannel(img, kernel)
    else:
        # (H,W,C)
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[..., c] = _filter_2d_singlechannel(img[..., c], kernel)
        return out

def _filter_2d_singlechannel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Реализация 2D свёртки с reflect-паддингом с использованием scipy.signal.convolve2d.
    channel: (H,W)
    kernel: (kH,kW)
    """
    # Исправленная граница 'reflect' -> 'symm' (symmetric)
    return convolve2d(channel, kernel, mode='same', boundary='symm')

def calculate_ssim_gauss_single(
    original: np.ndarray,
    processed: np.ndarray,
    K1: float = 0.01, # Consider making these constants in config.py
    K2: float = 0.03,
    L: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5
) -> float:
    """
    Более точный расчёт SSIM на одном канале. Используем Гауссово окно:
      muX = conv(X) ...
      sigmaX = conv(X^2) - muX^2
      и т.д.
    Результат — среднее по всей карте SSIM.
    """
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
    ssim_map = numerator / np.maximum(denominator, TINY_EPSILON)

    # Возвращаем среднее по всей карте
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
    """
    SSIM с Гауссовым окном.
    Если изображение многоканальное — берём среднее по каналам.
    Предполагаем, что original, processed в диапазоне [0..max_val].
    Если max_val > 1, нормализуем.
    """
    if original.shape != processed.shape:
        raise ValueError("SSIM: размеры изображений должны совпадать")

    # Нормализуем если max_val > 1
    if max_val > 1.00001:
        original = original / max_val
        processed = processed / max_val

    # Если (H,W)
    if original.ndim == 2:
        return calculate_ssim_gauss_single(original, processed, K1, K2, 1.0, window_size, sigma)
    elif original.ndim == 3:
        # Среднее по каналам
        c = original.shape[2]
        ssim_sum = 0.0
        for i in range(c):
            ssim_sum += calculate_ssim_gauss_single(
                original[..., i],
                processed[..., i],
                K1, K2, 1.0, window_size, sigma
            )
        return ssim_sum / c

    raise ValueError("Неподдерживаемая размерность изображения для SSIM")

def calculate_channel_ssim_gauss(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    channels: list[str],
    window_size: int = 11,
    sigma: float = 1.5
) -> Dict[str, float]:
    """
    Померный SSIM для (H,W,C). Возвращаем словарь {channel_name: ssim_value}.
    """
    if original.shape != processed.shape:
        raise ValueError("SSIM: размеры изображений должны совпадать")

    ssim_dict: Dict[str, float] = {}

    if max_val > 1.00001:
        original = original / max_val
        processed = processed / max_val

    # Если вдруг (H,W) — значит один канал
    if original.ndim == 2:
        ssim_dict['L'] = calculate_ssim_gauss_single(original, processed, window_size=window_size, sigma=sigma)
        return ssim_dict

    for i, ch in enumerate(channels):
        s = calculate_ssim_gauss_single(original[..., i], processed[..., i],
                                        window_size=window_size, sigma=sigma)
        ssim_dict[ch] = s

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
