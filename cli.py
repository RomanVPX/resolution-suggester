# cli.py
import argparse
import logging
import os

from config import (
    INTERPOLATION_DESCRIPTIONS,
    DEFAULT_INTERPOLATION,
    SUPPORTED_EXTENSIONS,
    InterpolationMethod  # Импорт Enum
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Анализ потерь качества текстур при масштабировании.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'paths',
        nargs='+',
        help='Пути к файлам/директориям для анализа'
    )

    parser.add_argument(
        '-c', '--channels',
        action='store_true',
        help='Анализ по цветовым каналам'
    )

    parser.add_argument(
        '-o', '--csv-output',
        action='store_true',
        help='Экспорт результатов в CSV'
    )

    parser.add_argument(
        '-i', '--interpolation',
        default=DEFAULT_INTERPOLATION,
        choices=[m.value for m in InterpolationMethod], # Используем значения Enum для choices
        metavar='METHOD',
        help=format_interpolation_help()
    )

    return parser.parse_args()

def format_interpolation_help() -> str:
    """Форматирование справки по методам интерполяции"""
    methods = [
        f"{m.value:<8}{' (default)' if m == DEFAULT_INTERPOLATION else '':<10} {desc}" # Используем m.value
        for m, desc in INTERPOLATION_DESCRIPTIONS.items()
    ]
    return "Доступные методы интерполяции:\n" + "\n".join(methods)

def validate_paths(paths: list[str]) -> list[str]:
    """Валидация и сбор файлов для обработки"""
    valid_paths = []
    for path in paths:
        if os.path.isfile(path):
            valid_paths.append(path)
        elif os.path.isdir(path):
            valid_paths.extend(collect_files_from_dir(path))
        else:
            logging.warning(f"Invalid path: {path}")

    if not valid_paths: # Проверка на наличие валидных путей
        raise ValueError("No valid files/directories found. Please check the provided paths.")
    return valid_paths

def collect_files_from_dir(directory: str) -> list[str]:
    """Рекурсивный сбор поддерживаемых файлов из директории"""
    collected = []
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                collected.append(os.path.join(root, f))
    return collected
