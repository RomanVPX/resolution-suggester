# cli.py
import argparse
import logging
import os

from config import (
    INTERPOLATION_DESCRIPTIONS,
    DEFAULT_INTERPOLATION,
    SUPPORTED_EXTENSIONS,
    InterpolationMethod,
    METRIC_DESCRIPTIONS,
    DEFAULT_METRIC,
    QualityMetric
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
        '-m', '--metric',
        default=DEFAULT_METRIC,
        choices=[m.value for m in QualityMetric],
        metavar='METRIC',
        help=format_metric_help()
    )

    parser.add_argument(
        '-i', '--interpolation',
        default=DEFAULT_INTERPOLATION,
        choices=[m.value for m in InterpolationMethod],
        metavar='METHOD',
        help=format_interpolation_help()
    )

    parser.add_argument(
        '--min-size',
        type=int,
        default=16,
        metavar='SIZE',
        help='Минимальный размер (по ширине и высоте) для анализа (по умолчанию 16)'
    )

    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=8,
        metavar='N',
        help='Число параллельных процессов для обработки файлов (игнорируется при --no-parallel, по умолчанию 8)'
    )

    parser.add_argument(
        '-s', '--save-intermediate',
        action='store_true',
        help='Сохранять результаты даунскейла'
    )

    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Отключить параллельную обработку и использовать однопоточную схему'
    )

    args = parser.parse_args()

    return args

def format_metric_help() -> str:
    metrics = [
        f"{m.value:<8}{' (default)' if m.value == DEFAULT_METRIC else '':<10} {desc}"
        for m, desc in METRIC_DESCRIPTIONS.items()
    ]
    return "Доступные метрики качества:\n" + "\n".join(metrics)

def format_interpolation_help() -> str:
    methods = [
        f"{m.value:<8}{' (default)' if m.value == DEFAULT_INTERPOLATION else '':<10} {desc}"
        for m, desc in INTERPOLATION_DESCRIPTIONS.items()
    ]
    return "Доступные методы интерполяции:\n" + "\n".join(methods)

def validate_paths(paths: list[str]) -> list[str]:
    valid_paths = []
    invalid_paths_str = [] # To collect invalid paths for better error message
    for path in paths:
        if os.path.isfile(path):
            valid_paths.append(path)
        elif os.path.isdir(path):
            valid_paths.extend(collect_files_from_dir(path))
        else:
            logging.warning(f"Неверный путь: {path}")
            invalid_paths_str.append(path) # Collect invalid paths

    if not valid_paths:
        error_message = "Не найдено ни одного валидного файла или директории."
        if invalid_paths_str: # Add details about invalid paths if any
            error_message += " Проверьте следующие пути: " + ", ".join(invalid_paths_str)
        raise ValueError(error_message)
    return valid_paths

def collect_files_from_dir(directory: str) -> list[str]:
    collected = []
    try:
        for root, _, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    collected.append(os.path.join(root, f))
    except PermissionError as e:
        logging.error(f"Ошибка доступа к директории {directory}: {str(e)}")
    except Exception as e:
        logging.error(f"Неожиданная ошибка при обходе директории {directory}: {str(e)}")
    return collected