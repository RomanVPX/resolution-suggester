# cli.py
import argparse
import logging
import os

from config import (
    INTERPOLATION_DESCRIPTIONS,
    DEFAULT_INTERPOLATION,
    SUPPORTED_EXTENSIONS,
    InterpolationMethod
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments() -> argparse.Namespace:
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
        choices=[m.value for m in InterpolationMethod],
        metavar='METHOD',
        help=format_interpolation_help()
    )

    parser.add_argument(
        '-m', '--min-size',
        type=int,
        default=16,
        metavar='SIZE',
        help='Минимальный размер (по ширине и высоте) для анализа (по умолчанию 16)'
    )

    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=4,
        metavar='N',
        help='Число параллельных процессов (по умолчанию 4)'
    )

    # Новый параметр для чанкинга применительно к Митчеллу
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=0,
        metavar='SIZE',
        help='Размер тайла для чанкинга при митчелловском ресайзе (0 — без чанкинга)'
    )

    return parser.parse_args()

def format_interpolation_help() -> str:
    methods = [
        f"{m.value:<8}{' (default)' if m.value == DEFAULT_INTERPOLATION else '':<10} {desc}"
        for m, desc in INTERPOLATION_DESCRIPTIONS.items()
    ]
    return "Доступные методы интерполяции:\n" + "\n".join(methods)

def validate_paths(paths: list[str]) -> list[str]:
    valid_paths = []
    for path in paths:
        if os.path.isfile(path):
            valid_paths.append(path)
        elif os.path.isdir(path):
            valid_paths.extend(collect_files_from_dir(path))
        else:
            logging.warning(f"Invalid path: {path}")

    if not valid_paths:
        raise ValueError("No valid files/directories found. Please check the provided paths.")
    return valid_paths

def collect_files_from_dir(directory: str) -> list[str]:
    collected = []
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                collected.append(os.path.join(root, f))
    return collected
