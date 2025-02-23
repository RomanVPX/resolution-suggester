"""
Command line interface for image quality analysis.

This module provides the command line interface for image quality analysis.
It uses argparse to define the command line interface and parse the
arguments.

"""
import argparse
import logging
import os
import multiprocessing

from config import (
    INTERPOLATION_METHODS_INFO,
    INTERPOLATION_METHOD_DEFAULT,
    SUPPORTED_EXTENSIONS,
    InterpolationMethods,
    QUALITY_METRICS_INFO,
    QUALITY_METRIC_DEFAULT,
    QualityMetrics
)

def setup_logging():
    """
    Initialize logging module.

    Set the logging level to INFO and format each log entry as
    '%(asctime)s - %(levelname)s - %(message)s'.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns a Namespace object with the following attributes:
      - paths: list of paths to files or directories to analyze
      - channels: boolean flag to analyze by color channels
      - csv_output: boolean flag to export results to CSV
      - metric: string, one of the values from QualityMetric enum
      - interpolation: string, one of the values from InterpolationMethod enum
      - min_size: int, minimum size (width and height) for analysis (default 16)
      - threads: int, number of parallel processes for file processing (default 8)
      - save_intermediate: boolean flag to save downscaled results
      - no_parallel: boolean flag to disable parallel processing and use single-threaded scheme

    Raises:
      - argparse.ArgumentError if arguments are invalid
    """
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
        '-m', '--metric', type=QualityMetrics,
        default=QUALITY_METRIC_DEFAULT,
        choices=[m.value for m in QualityMetrics],
        metavar='METRIC',
        help=format_metric_help()
    )

    parser.add_argument(
        '-i', '--interpolation', type=InterpolationMethods,
        default=INTERPOLATION_METHOD_DEFAULT,
        choices=[m.value for m in InterpolationMethods],
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
        default=multiprocessing.cpu_count(),
        metavar='N',
        help=format_threads_help()
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

    parser.add_argument(
        '--generate-dataset',
        action='store_true',
        help='Сгенерировать датасет (features/targets) для обучения модели'
    )

    parser.add_argument(
        '--train-ml',
        action='store_true',
        help='После генерации датасета обучить модель ML'
    )

    parser.add_argument(
        '--ml',
        action='store_true',
        help='Использовать ML-модель для предсказания PSNR/SSIM (вместо реального вычисления)'
    )

    args = parser.parse_args()

    return args

def format_threads_help() -> str:
    return ("Число параллельных процессов для обработки файлов. Игнорируется при --no-parallel,\n"
            "по умолчанию равно количеству логических ядер процессора (сейчас обнаружено " +
            str(multiprocessing.cpu_count()) + ")")

def format_metric_help() -> str:
    """
    Return a string containing the list of available quality metrics.
    Each metric is represented as a string with the following format:
    <metric name> <(default)> <metric description>
    The <(default)> part is only present if the metric is the default one.
    """
    metrics = [
        f"{m.value:<8}{' (default)' if m.value == QUALITY_METRIC_DEFAULT else '':<10} {desc}"
        for m, desc in QUALITY_METRICS_INFO.items()
    ]
    return "Доступные метрики качества:\n" + "\n".join(metrics)

def format_interpolation_help() -> str:
    """
    Return a string containing the list of available interpolation methods.
    Each method is represented as a string with the following format:
    <method name> <(default)> <method description>
    The <(default)> part is only present if the method is the default one.
    """
    methods = [
        f"{m.value:<8}{' (default)' if m.value == INTERPOLATION_METHOD_DEFAULT else '':<10} {desc}"
        for m, desc in INTERPOLATION_METHODS_INFO.items()
    ]
    return "Доступные методы интерполяции:\n" + "\n".join(methods)

def validate_paths(paths: list[str]) -> list[str]:
    """
    Validate paths and return a list of valid paths.

    For each path in the input list, the function checks if it is a valid file or directory.
    If the path is a file, it is added to the output list.
    If the path is a directory, the function calls collect_files_from_dir to get a list of
    all files in the directory and adds them to the output list.

    If no valid paths are found, the function logs an error message and returns an empty list.
    """
    valid_paths = []
    invalid_paths_str = []

    for path in paths:
        if os.path.isfile(path):
            valid_paths.append(path)
        elif os.path.isdir(path):
            valid_paths.extend(collect_files_from_dir(path))
        else:
            logging.warning("Неверный путь: %s", path)
            invalid_paths_str.append(path)

    if not valid_paths:
        error_message = "Не найдено ни одного валидного файла или директории."
        if invalid_paths_str:
            error_message += " Проверьте следующие пути: " + ", ".join(invalid_paths_str)
        logging.error(error_message)
        return []  # Ноу эксепшенс!
    return valid_paths

def collect_files_from_dir(directory: str) -> list[str]:
    """
    Recursively collects and returns a list of file paths from the specified directory.
    Args:
        directory: The path to the directory to search for files.
    Returns:
        A list of file paths with extensions matching the SUPPORTED_EXTENSIONS.
    Logs:
        Logs an error if access to a directory is denied or if an unexpected error occurs.
    """
    collected = []
    try:
        for root, _, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    collected.append(os.path.join(root, f))
    except PermissionError as e:
        logging.error("Ошибка доступа к директории %s: %s", directory, str(e))
    except OSError as e:
        logging.error("Неожиданная ошибка при обходе директории %s: %s", directory, str(e))
    return collected
