# config.py
"""
Configuration constants for the image quality analysis tool.

This module provides constants for configuration and output formatting.
"""

from .i18n import _
from enum import Enum
from pathlib import Path
from typing import Final


SUPPORTED_EXTENSIONS: Final = frozenset({'.exr', '.tga', '.png', '.jpg', '.jpeg'})
CSV_SEPARATOR: Final = ';'

ROOT_DIR = Path(__file__).parent.parent.parent

LOGS_DIR = ROOT_DIR / "logs"

DATA_DIR = ROOT_DIR / "data"

GENERATED_IMAGES_DIR = DATA_DIR / "generated_images"
INTERMEDIATE_DIR = GENERATED_IMAGES_DIR / "intermediate"

ML_DATA_DIR = DATA_DIR / "ml"
ML_MODELS_DIR = ML_DATA_DIR / "models"
ML_FEATURES_DIR = ML_DATA_DIR / "features"
ML_DATASETS_DIR = ML_DATA_DIR / "datasets"


MITCHELL_B = 1/3
MITCHELL_C = 1/3

TINY_EPSILON = 1e-8
PSNR_IS_LARGE_AS_INF = 139.0

MIN_DOWNSCALE_SIZE = 16

# === Quality levels ===
class QualityLevelHints(Enum):
    """
    Enumeration of quality level hints for image analysis output.
    """
    ORIGINAL = "original"
    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    NOTICEABLE_LOSS = "noticeable_loss"

QUALITY_LEVEL_HINTS_DESCRIPTIONS = {
    QualityLevelHints.ORIGINAL: _("original"),
    QualityLevelHints.EXCELLENT: _("excellent"),
    QualityLevelHints.VERY_GOOD: _("very_good"),
    QualityLevelHints.GOOD: _("good"),
    QualityLevelHints.NOTICEABLE_LOSS: _("noticeable_loss"),
}

# === Quality metrics ===
class QualityMetrics(str, Enum):
    """
    Enumeration of quality metrics for image analysis.
    """
    PSNR = 'psnr'
    SSIM = 'ssim'
    MS_SSIM = 'ms_ssim'
    TDPR = 'tdpr'

# Описания метрик для справки
QUALITY_METRICS_INFO = {
    QualityMetrics.PSNR: _("Peak Signal-to-Noise Ratio"),
    QualityMetrics.SSIM: _("Structural Similarity Index"),
    QualityMetrics.MS_SSIM: _("Multi-Scale Structural Similarity Index"),
    QualityMetrics.TDPR: _("Texture Detail Preservation Ratio"),
}

ML_TARGET_COLUMNS: Final = [m.value for m in QualityMetrics]

QUALITY_METRIC_DEFAULT = QualityMetrics.PSNR

# --- Quality thresholds for metrics ---
QUALITY_METRIC_THRESHOLDS = {
    QualityMetrics.PSNR: {
        QualityLevelHints.ORIGINAL: PSNR_IS_LARGE_AS_INF + 1.0,
        QualityLevelHints.EXCELLENT: PSNR_IS_LARGE_AS_INF,
        QualityLevelHints.VERY_GOOD: 50,
        QualityLevelHints.GOOD: 40,
        QualityLevelHints.NOTICEABLE_LOSS: 30,
    },
    QualityMetrics.SSIM: {
        QualityLevelHints.ORIGINAL: 1.001,
        QualityLevelHints.EXCELLENT: 1.00,
        QualityLevelHints.VERY_GOOD: 0.92,
        QualityLevelHints.GOOD: 0.82,
        QualityLevelHints.NOTICEABLE_LOSS: 0.75,
    },
    QualityMetrics.MS_SSIM: {
        QualityLevelHints.ORIGINAL: 1.001,
        QualityLevelHints.EXCELLENT: 1.00,
        QualityLevelHints.VERY_GOOD: 0.97,
        QualityLevelHints.GOOD: 0.95,
        QualityLevelHints.NOTICEABLE_LOSS: 0.90,
    },
    QualityMetrics.TDPR: {
        QualityLevelHints.ORIGINAL: 1.001,
        QualityLevelHints.EXCELLENT: 1.00,
        QualityLevelHints.VERY_GOOD: 0.90,
        QualityLevelHints.GOOD: 0.80,
        QualityLevelHints.NOTICEABLE_LOSS: 0.70,
    }
}

# === Interpolation methods ===
class InterpolationMethods(str, Enum):
    """
    Enumeration of interpolation methods for image resampling.
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
    InterpolationMethods.BILINEAR: _("Bilinear interpolation"),
    InterpolationMethods.BICUBIC: _("Bicubic interpolation"),
    InterpolationMethods.MITCHELL: _("Mitchell-Netravali filter"),
}

INTERPOLATION_METHOD_DEFAULT = InterpolationMethods.MITCHELL
INTERPOLATION_METHOD_UPSCALE = InterpolationMethods.BICUBIC

# === Styling for console output ===
RICH_STYLES = {
    'header': "bold cyan on grey35",
    'original': "bold cyan",
    'excellent': "bright_green",
    'very_good': "green",
    'good': "yellow",
    'poor': "red",
    'filename': "bold bright_cyan on black",
}


def get_output_csv_header(analyze_channels: bool, metric_type: QualityMetrics) -> list[str]:
    """
    Forms the CSV header considering the metric.

    Args:
        analyze_channels: Whether to include per-channel analysis.
        metric_type: The quality metric used.

    Returns:
        List of column names for the CSV output.
    """
    header = [_("File"), _("Resolution")]
    if analyze_channels:
        # Фиксированные столбцы, не зависящие от реального количества каналов для лучшей читаемости таблицы
        header.extend([
            f"R(L) {metric_type.value.upper()}",  # Red для многоканальных, Luminance для одноканальных изображений
            f"G {metric_type.value.upper()}",
            f"B {metric_type.value.upper()}",
            f"A {metric_type.value.upper()}",
            f"Min {metric_type.value.upper()}",
            "Качество (min)"
        ])
    else:
        header.extend([f"{metric_type.value.upper()}", "Качество"])
    return header
