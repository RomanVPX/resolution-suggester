"""
Command line interface for image quality analysis.

This module provides the command line interface for image quality analysis.
It uses argparse to define the command line interface and parse the arguments.
"""
import argparse
import logging
import multiprocessing
import os

from ..config import (
    INTERPOLATION_METHOD_DEFAULT,
    INTERPOLATION_METHODS_INFO,
    MIN_DOWNSCALE_SIZE,
    QUALITY_METRIC_DEFAULT,
    QUALITY_METRICS_INFO,
    SUPPORTED_EXTENSIONS,
    InterpolationMethods,
    QualityMetrics,
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

cpu_count = multiprocessing.cpu_count()
default_threads_count = (cpu_count - 2, cpu_count)[cpu_count < 8]

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

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
        default=MIN_DOWNSCALE_SIZE,
        metavar='SIZE',
        help="Минимальный размер (по ширине и высоте) для анализа (по умолчанию и минимально: " +
             str(MIN_DOWNSCALE_SIZE) + ")"
    )

    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=default_threads_count,
        metavar='N',
        help=format_threads_help()
    )

    parser.add_argument(
        '--save-im-down',
        action='store_true',
        help='Сохранять результаты даунскейла, производимого во время анализа\n'
             '(не работает с --ml, --train-ml и --generate-dataset)'
    )

    parser.add_argument(
        '--save-im-up',
        action='store_true',
        help='Сохранять результаты апскейла, произведённого после даунскейла\n'
             '(не работает с --ml, --train-ml и --generate-dataset)'
    )

    parser.add_argument(
        '-s', '--save-im-all',
        action='store_true',
        help='Сохранять результаты все результаты масштабирования изображений (даунскейл и апскейл)\n'
             '(не работает с --ml, --train-ml и --generate-dataset)'
    )

    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Отключить параллельную обработку и использовать однопоточную схему'
    )

    parser.add_argument(
        '--no-pytorch',
        action='store_true',
        help='Не использовать PyTorch для расчёта метрик (на случай проблем с CUDA, MPS и т.д.)'
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
        help='Использовать ML-модель для предсказания метрик вместо реального вычисления (быстро)'
    )

    args = parser.parse_args()

    if args.save_im_all:
        args.save_im_down = True
        args.save_im_up = True

    if args.generate_dataset or args.train_ml or args.ml:
        if args.save_im_down or args.save_im_up or args.save_im_all:
            logging.warning("Нельзя использовать --save-im-* и --generate-dataset, --train-ml, --ml с одновременно!\n"
                            "параметры --save-im-* будут проигнорированы.")
            args.save_im_down = False
            args.save_im_up = False

    if args.min_size < MIN_DOWNSCALE_SIZE:
        logging.warning(
            "Минимальный размер (по ширине и высоте) для анализа должен быть >= %s. "
            "Установлено значение по умолчанию: %s",
            MIN_DOWNSCALE_SIZE, MIN_DOWNSCALE_SIZE
        )
        args.min_size = MIN_DOWNSCALE_SIZE
    if args.threads < 1:
        logging.warning(
            "Число параллельных процессов должно быть >= 1. "
            "Установлено минимальное значение: 1"
        )
        args.threads = 1

    return args

def format_threads_help() -> str:
    return ("Число параллельных процессов для обработки файлов. Игнорируется при --no-parallel,\n"
            "по умолчанию равно " + str(default_threads_count) + " (логических ядер процессора обнаружено " +
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
    For each path, if it's a file, add it.
    If it's a directory, collect files from that directory.
    Raises:
        ValueError: if no valid paths are found.
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
        raise ValueError(error_message)
    return valid_paths

def collect_files_from_dir(directory: str) -> list[str]:
    """
    Recursively collects and returns a list of file paths from the specified directory.
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
