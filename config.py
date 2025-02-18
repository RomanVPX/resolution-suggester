# config.py
from enum import Enum

SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png']
CSV_SEPARATOR = ';'
DEFAULT_INTERPOLATION = 'mitchell'

QUALITY_HINTS = {
    50: "практически идентичные изображения",
    40: "очень хорошее качество",
    30: "приемлемое качество",
    0: "заметные потери",
}

class InterpolationMethod(str, Enum):
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    MITCHELL = 'mitchell'

INTERPOLATION_METHODS = {
    InterpolationMethod.BILINEAR: 'INTER_LINEAR',
    InterpolationMethod.BICUBIC: 'INTER_CUBIC',
    InterpolationMethod.MITCHELL: 'mitchell',
}

INTERPOLATION_DESCRIPTIONS = {
    InterpolationMethod.BILINEAR: 'Билинейная интерполяция',
    InterpolationMethod.BICUBIC: 'Бикубическая интерполяция',
    InterpolationMethod.MITCHELL: 'Фильтр Митчелла-Нетравали',
}

STYLES = {
    'header': "\033[1;96;100m",
    'warning': "\033[2;33m",
    'original': "\033[36m",
    'good': "\033[92m",
    'ok': "\033[32m",
    'medium': "\033[33m",
    'bad': "\033[31m",
}

PSNR_QUALITY_THRESHOLDS = sorted(QUALITY_HINTS.keys(), reverse=True)


def get_output_csv_header(analyze_channels: bool) -> list[str]:
    header = ["Файл", "Разрешение"]
    if analyze_channels:
        header.extend(["R(L) PSNR", "G PSNR", "B PSNR", "A PSNR", "Min PSNR", "Качество (min)"])
    else:
        header.extend(["PSNR", "Качество"])
    return header
