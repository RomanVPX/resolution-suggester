# config.py

from enum import Enum
from colorama import Fore, Back, Style

SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png', '.jpg', '.jpeg']
CSV_SEPARATOR = ';'
DEFAULT_INTERPOLATION = 'mitchell'

MITCHELL_B = 1/3
MITCHELL_C = 1/3

MITCHELL_RADIUS = 2
TINY_EPSILON = 1e-8

class QualityHintPSNR(Enum):
    EXCELLENT = 50
    VERY_GOOD = 40
    GOOD = 30
    NOTICEABLE_LOSS = 0

QUALITY_HINTS = {
    QualityHintPSNR.EXCELLENT.value: "практически идентичные изображения",
    QualityHintPSNR.VERY_GOOD.value: "очень хорошее качество",
    QualityHintPSNR.GOOD.value: "приемлемое качество",
    QualityHintPSNR.NOTICEABLE_LOSS.value: "заметные потери",
}

PSNR_QUALITY_THRESHOLDS = sorted(QUALITY_HINTS.keys(), reverse=True)

SAVE_INTERMEDIATE_DIR = '_intermediate'

class InterpolationMethod(str, Enum):
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    MITCHELL = 'mitchell'

# Соответствие методов интерполяции OpenCV
INTERPOLATION_METHODS = {
    InterpolationMethod.BILINEAR: 'INTER_LINEAR',
    InterpolationMethod.BICUBIC: 'INTER_CUBIC',
    InterpolationMethod.MITCHELL: 'mitchell', # 'mitchell' - это placeholder, т.к. Mitchell реализован отдельно
}

# Описания методов интерполяции для справки
INTERPOLATION_DESCRIPTIONS = {
    InterpolationMethod.BILINEAR: 'Билинейная интерполяция',
    InterpolationMethod.BICUBIC: 'Бикубическая интерполяция',
    InterpolationMethod.MITCHELL: 'Фильтр Митчелла-Нетравали',
}

class QualityMetric(str, Enum):
    PSNR = 'psnr'
    SSIM = 'ssim'

# Описания метрик для справки
METRIC_DESCRIPTIONS = {
    QualityMetric.PSNR: 'Пиковое отношение сигнала к шуму',
    QualityMetric.SSIM: 'Индекс структурного сходства',
}

DEFAULT_METRIC = QualityMetric.PSNR.value

# Стили для консольного вывода
STYLES = {
    'header': "\033[1;96;100m",
    'warning': "\033[2;33m",
    'original': Fore.CYAN,
    'good': Fore.LIGHTGREEN_EX,
    'ok': Fore.GREEN,
    'medium': Fore.YELLOW,
    'bad': Fore.RED,
}


def get_output_csv_header(analyze_channels: bool) -> list[str]:
    header = ["Файл", "Разрешение"]
    if analyze_channels:
        header.extend(["R(L) PSNR", "G PSNR", "B PSNR", "A PSNR", "Min PSNR", "Качество (min)"])
    else:
        header.extend(["PSNR", "Качество"])
    return header
