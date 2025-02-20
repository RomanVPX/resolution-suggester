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
class QualityLevel(Enum):
    """
    Enumeration of quality levels for image analysis.

    This class defines different quality levels for image analysis results,
    ranging from 'excellent' to 'noticeable loss'. These levels can be used
    to categorize the perceived quality of images based on specific metrics.
    """

    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    NOTICEABLE_LOSS = "noticeable_loss"

QUALITY_LEVEL_DESCRIPTIONS = {
    QualityLevel.EXCELLENT: "практически идентичные изображения",
    QualityLevel.VERY_GOOD: "очень хорошее качество",
    QualityLevel.GOOD: "приемлемое качество",
    QualityLevel.NOTICEABLE_LOSS: "заметные потери",
}

# === Quality metrics ===
class QualityMetric(str, Enum):
    """
    Enumeration of quality metrics for image analysis.

    This class defines different quality metrics that can be used to assess
    the quality of images, such as PSNR (Peak Signal-to-Noise Ratio) and SSIM
    (Structural Similarity Index). These metrics provide numerical values that
    can be used to determine the perceived quality of images.
    """

    PSNR = 'psnr'
    SSIM = 'ssim'
    MS_SSIM = 'ms_ssim'

# Описания метрик для справки
METRIC_DESCRIPTIONS = {
    QualityMetric.PSNR: 'Пиковое отношение сигнала к шуму',
    QualityMetric.SSIM: 'Индекс структурного сходства',
    QualityMetric.MS_SSIM: 'Многоуровневый индекс структурного сходства',
}

ML_TARGET_COLUMNS = [m.value for m in QualityMetric]

DEFAULT_METRIC = QualityMetric.PSNR

# === Quality thresholds for metrics ===
METRIC_QUALITY_THRESHOLDS = {
    QualityMetric.PSNR: {
        QualityLevel.EXCELLENT: 50,
        QualityLevel.VERY_GOOD: 40,
        QualityLevel.GOOD: 30,
        QualityLevel.NOTICEABLE_LOSS: 0,
    },
    QualityMetric.SSIM: {
        QualityLevel.EXCELLENT: 0.92,
        QualityLevel.VERY_GOOD: 0.82,
        QualityLevel.GOOD: 0.75,
        QualityLevel.NOTICEABLE_LOSS: 0.0,
    },
    QualityMetric.MS_SSIM: {
        QualityLevel.EXCELLENT: 0.97,
        QualityLevel.VERY_GOOD: 0.95,
        QualityLevel.GOOD: 0.90,
        QualityLevel.NOTICEABLE_LOSS: 0.0,
    }
}

# === Interpolation methods ===
class InterpolationMethod(str, Enum):
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
    InterpolationMethod.BILINEAR: 'INTER_LINEAR',
    InterpolationMethod.BICUBIC: 'INTER_CUBIC',
    InterpolationMethod.MITCHELL: None, # placeholder: Mitchell реализован отдельно
}

INTERPOLATION_DESCRIPTIONS = {
    InterpolationMethod.BILINEAR: 'Билинейная интерполяция',
    InterpolationMethod.BICUBIC: 'Бикубическая интерполяция',
    InterpolationMethod.MITCHELL: 'Фильтр Митчелла-Нетравали',
}

DEFAULT_INTERPOLATION = 'mitchell'

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
    Формирует заголовок CSV с учётом метрики
    """
    header = ["Файл", "Разрешение"]
    metric_str = str(metric.upper())
    if analyze_channels:
        header.extend([f"R(L) {metric_str}", f"G {metric_str}", f"B {metric_str}",
                        f"A {metric_str}", f"Min {metric_str}", "Качество (min)"])
    else:
        header.extend([f"{metric_str}", "Качество"])
    return header
