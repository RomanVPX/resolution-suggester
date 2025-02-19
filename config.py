# config.py

from enum import Enum
from colorama import Fore, Back, Style

SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png', '.jpg', '.jpeg']
CSV_SEPARATOR = ';'

SAVE_INTERMEDIATE_DIR = '_intermediate'

MITCHELL_B = 1/3
MITCHELL_C = 1/3

TINY_EPSILON = 1e-8

# === Quality levels ===
class QualityLevel(Enum):
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
    PSNR = 'psnr'
    SSIM = 'ssim'

# Описания метрик для справки
METRIC_DESCRIPTIONS = {
    QualityMetric.PSNR: 'Пиковое отношение сигнала к шуму',
    QualityMetric.SSIM: 'Индекс структурного сходства',
}

DEFAULT_METRIC = QualityMetric.PSNR.value

# === Quality thresholds for metrics ===
METRIC_QUALITY_THRESHOLDS = {
    QualityMetric.PSNR: {
        QualityLevel.EXCELLENT: 50,
        QualityLevel.VERY_GOOD: 40,
        QualityLevel.GOOD: 30,
        QualityLevel.NOTICEABLE_LOSS: 0,
    },
    QualityMetric.SSIM: {
        QualityLevel.EXCELLENT: 0.9,
        QualityLevel.VERY_GOOD: 0.8,
        QualityLevel.GOOD: 0.65,
        QualityLevel.NOTICEABLE_LOSS: 0.0,
    }
}

# === Interpolation methods ===
class InterpolationMethod(str, Enum):
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    MITCHELL = 'mitchell'

INTERPOLATION_METHODS_CV2 = {
    InterpolationMethod.BILINEAR: 'INTER_LINEAR',
    InterpolationMethod.BICUBIC: 'INTER_CUBIC',
    InterpolationMethod.MITCHELL: 'mitchell', # 'mitchell' - это placeholder, т.к. Mitchell реализован отдельно
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
        header.extend([f"R(L) {metric_str}", f"G {metric_str}", f"B {metric_str}", f"A {metric_str}", f"Min {metric_str}", "Качество (min)"])
    else:
        header.extend([f"{metric_str}", "Качество"])
    return header
