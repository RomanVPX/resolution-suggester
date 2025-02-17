# config.py
from typing import List
import cv2 # Добавлен импорт cv2

SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png']
CSV_SEPARATOR = ';'
DEFAULT_INTERPOLATION = 'mitchell'

QUALITY_HINTS = {
    50: "практически идентичные изображения",
    40: "очень хорошее качество",
    30: "приемлемое качество",
    0: "заметные потери",
}

INTERPOLATION_METHODS = {
    'bilinear': 'INTER_LINEAR', # Исправлено: 'cv2.INTER_LINEAR' -> 'INTER_LINEAR'
    'bicubic': 'INTER_CUBIC', # Исправлено: 'cv2.INTER_CUBIC' -> 'INTER_CUBIC'
    'mitchell': 'mitchell',
}

INTERPOLATION_DESCRIPTIONS = {
    'bilinear': 'Билинейная интерполяция',
    'bicubic': 'Бикубическая интерполяция',
    'mitchell': 'Фильтр Митчелла-Нетравали',
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
    return [
        "Файл",
        "Разрешение",
        "R(L) PSNR",
        "G PSNR",
        "B PSNR",
        "A PSNR",
        "Min PSNR",
        "Качество (min)"
    ] if analyze_channels else [
        "Файл",
        "Разрешение",
        "PSNR",
        "Качество"
    ]
