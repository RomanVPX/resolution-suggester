# metrics.py
import math
import numpy as np
from numba import njit
from typing import Dict, Tuple
from config import TINY_EPSILON

@njit(cache=True)
def calculate_psnr(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float
) -> float:
    if original.shape != processed.shape:
        raise ValueError("Image dimensions must match for PSNR calculation")

    # Оптимизированный расчет MSE
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
