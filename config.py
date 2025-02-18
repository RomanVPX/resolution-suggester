# config.py
from enum import Enum
from colorama import Fore, Back, Style

SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png']
CSV_SEPARATOR = ';'
DEFAULT_INTERPOLATION = 'mitchell'

MITCHELL_B = 1/3
MITCHELL_C = 1/3

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
    'original': Fore.CYAN,
    'good': Fore.LIGHTGREEN_EX,
    'ok': Fore.GREEN,
    'medium': Fore.YELLOW,
    'bad': Fore.RED,
}

PSNR_QUALITY_THRESHOLDS = sorted(QUALITY_HINTS.keys(), reverse=True)


def get_output_csv_header(analyze_channels: bool) -> list[str]:
    header = ["Файл", "Разрешение"]
    if analyze_channels:
        header.extend(["R(L) PSNR", "G PSNR", "B PSNR", "A PSNR", "Min PSNR", "Качество (min)"])
    else:
        header.extend(["PSNR", "Качество"])
    return header
