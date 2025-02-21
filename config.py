# config.py
"""
Configuration constants for the image quality analysis tool.

This module provides constants for configuration and output formatting.
"""

from enum import Enum
from colorama import Fore, Back, Style

SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png', '.jpg', '.jpeg']
CSV_SEPARATOR = ';'

SAVE_INTERMEDIATE_DIR = '_intermediate'
ML_DATA_DIR = '_ml-data'

MITCHELL_B = 1/3
MITCHELL_C = 1/3

TINY_EPSILON = 1e-8

# === Quality levels ===
class QualityLevelHints(Enum):
    """
    Enumeration of quality level hints for image analysis output.
    """

    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    NOTICEABLE_LOSS = "noticeable_loss"

QUALITY_LEVEL_HINTS_DESCRIPTIONS = {
    QualityLevelHints.EXCELLENT: "практически идентичные изображения",
    QualityLevelHints.VERY_GOOD: "очень хорошее качество",
    QualityLevelHints.GOOD: "приемлемое качество",
    QualityLevelHints.NOTICEABLE_LOSS: "заметные потери",
}

# === Quality metrics ===
class QualityMetrics(str, Enum):
    """
    Enumeration of quality metrics for image analysis.
    """

    PSNR = 'psnr'
    SSIM = 'ssim'
    MS_SSIM = 'ms_ssim'

# Описания метрик для справки
QUALITY_METRICS_INFO = {
    QualityMetrics.PSNR: 'Пиковое отношение сигнала к шуму',
    QualityMetrics.SSIM: 'Индекс структурного сходства',
    QualityMetrics.MS_SSIM: 'Многоуровневый индекс структурного сходства',
}

ML_TARGET_COLUMNS = [m.value for m in QualityMetrics]

DEFAULT_QUALITY_METRIC = QualityMetrics.PSNR

# --- Quality thresholds for metrics ---
METRIC_QUALITY_THRESHOLDS = {
    QualityMetrics.PSNR: {
        QualityLevelHints.EXCELLENT: 50,
        QualityLevelHints.VERY_GOOD: 40,
        QualityLevelHints.GOOD: 30,
        QualityLevelHints.NOTICEABLE_LOSS: 0,
    },
    QualityMetrics.SSIM: {
        QualityLevelHints.EXCELLENT: 0.92,
        QualityLevelHints.VERY_GOOD: 0.82,
        QualityLevelHints.GOOD: 0.75,
        QualityLevelHints.NOTICEABLE_LOSS: 0.0,
    },
    QualityMetrics.MS_SSIM: {
        QualityLevelHints.EXCELLENT: 0.97,
        QualityLevelHints.VERY_GOOD: 0.95,
        QualityLevelHints.GOOD: 0.90,
        QualityLevelHints.NOTICEABLE_LOSS: 0.0,
    }
}

# === Interpolation methods ===
class InterpolationMethods(str, Enum):
    """
    Enumeration of interpolation methods for image resampling.

    This class defines different interpolation methods that can be used to
    resample images, such as bilinear and bicubic interpolation. These
    methods are used to calculate the pixel values at fractional coordinates
    when resizing an image.
    """

    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    MITCHELL = 'mitchell'

INTERPOLATION_METHODS_CV2 = {
    InterpolationMethods.BILINEAR: 'INTER_LINEAR',
    InterpolationMethods.BICUBIC: 'INTER_CUBIC',
    # 'mitchell' is implemented separately
}

INTERPOLATION_METHODS_INFO = {
    InterpolationMethods.BILINEAR: 'Билинейная интерполяция',
    InterpolationMethods.BICUBIC: 'Бикубическая интерполяция',
    InterpolationMethods.MITCHELL: 'Фильтр Митчелла-Нетравали',
}

DEFAULT_INTERPOLATION_METHOD = InterpolationMethods.MITCHELL

# === Styling for console output ===
STYLES = {
    'header': f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}",
    'warning': f"{Style.DIM}{Back.LIGHTYELLOW_EX}",
    'original': Fore.CYAN,
    'good': Fore.LIGHTGREEN_EX,
    'ok': Fore.GREEN,
    'medium': Fore.YELLOW,
    'bad': Fore.RED,
}

def get_output_csv_header(analyze_channels: bool, metric: str) -> list[str]:
    """
    Forms the CSV header considering the metric.

    Args:
        analyze_channels: Whether to include per-channel analysis.
        metric: The quality metric used.

    Returns:
        List of column names for the CSV output.
    """
    header = ["Файл", "Разрешение"]
    metric_str = metric.upper()
    if analyze_channels:
        header.extend([
            f"R(L) {metric_str}",
            f"G {metric_str}",
            f"B {metric_str}",
            f"A {metric_str}",
            f"Min {metric_str}",
            "Качество (min)"
        ])
    else:
        header.extend([f"{metric_str}", "Качество"])
    return header