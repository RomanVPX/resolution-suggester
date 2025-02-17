# metrics.py
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

def calculate_psnr(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float
) -> float:
    """Вычисляет PSNR между двумя изображениями"""
    mse = np.mean((original - processed) ** 2)
    return 20 * math.log10(max_val) - 10 * math.log10(mse) if mse != 0 else float('inf')

def calculate_channel_psnr(
    original: np.ndarray,
    processed: np.ndarray,
    max_val: float,
    channels: list[str] # Исправлено: List -> list
) -> Dict[str, float]:
    """Вычисляет PSNR для каждого канала отдельно"""
    return {
        channel: calculate_psnr(original[..., i], processed[..., i], max_val)
        for i, channel in enumerate(channels)
    }

def compute_resolutions(
    original_width: int,
    original_height: int,
    min_size: int = 16
) -> list[Tuple[int, int]]: # Исправлено: List -> list
    """Генерирует последовательность уменьшающихся разрешений"""
    resolutions = []
    w, h = original_width, original_height

    while w >= min_size*2 and h >= min_size*2:
        w //= 2
        h //= 2
        resolutions.append((w, h))

    return resolutions
