# utils/cli.py
"""
Command line interface for ResolutionSuggester.
"""
from ..i18n import _
import argparse
import sys

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
    """
    # Создаем парсер с локализованным описанием
    from ..i18n import _, setup_localization
    pre_parser = argparse.ArgumentParser(
        description=_('Texture quality analysis tool'),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )

    # Добавляем аргумент справки вручную
    pre_parser.add_argument(
        '-h', '--help',
        action='store_true',
        help=_('Show this help message and exit')
    )

    # Добавляем аргумент для выбора языка (должен быть обработан рано)
    pre_parser.add_argument(
        '--lang',
        choices=['en', 'ru', 'auto'],
        default='auto',
        help=_('Interface language (default: auto)')
    )

    # Сначала парсим только аргументы языка и справки
    pre_args, _ = pre_parser.parse_known_args()

    # Если указан язык, переустанавливаем локализацию
    if pre_args.lang != 'auto':
        setup_localization(pre_args.lang)

    parser = create_parser()

    # Если запрошена справка, показываем её и выходим
    if pre_args.help:
        parser.print_help()
        sys.exit(0)

    # Теперь парсим все аргументы
    args = parser.parse_args()

    if args.lang != 'auto':
        from ..i18n import setup_localization
        setup_localization(args.lang)

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

def create_parser() -> argparse.ArgumentParser:
    """
    Создает и настраивает парсер аргументов с текущими переводами.
    """
    parser = argparse.ArgumentParser(
        description=_('Texture quality analysis tool'),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'paths',
        nargs='+',
        help=_('Paths to files/directories for analysis')
    )

    parser.add_argument(
        '--lang',
        choices=['en', 'ru', 'auto'],
        default='auto',
        help=_('Interface language (default: auto)')
    )

    parser.add_argument(
        '-c', '--channels',
        action='store_true',
        help=_('Analysis by color channels')
    )

    parser.add_argument(
        '-o', '--csv-output',
        action='store_true',
        help=_('Export results to CSV')
    )

    parser.add_argument(
        '-j', '--json-output',
        action='store_true',
        help=_('Export results to JSON')
    )

    parser.add_argument(
        '--chart',
        action='store_true',
        help=_('Generate quality vs. resolution charts')
    )

    parser.add_argument(
        '--theme',
        choices=['light', 'dark'],
        default='dark',
        help=_('Charts theme (default: dark)')
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
        help=_("Minimum size (width and height) for analysis (default and minimum: ") +
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
        help=_('Save downscale results produced during analysis\n') +
             _('(does not work with --ml, --train-ml and --generate-dataset)')
    )

    parser.add_argument(
        '--save-im-up',
        action='store_true',
        help=_('Save upscale results produced after downscale\n') +
             _('(does not work with --ml, --train-ml and --generate-dataset)')
    )

    parser.add_argument(
        '-s', '--save-im-all',
        action='store_true',
        help=_('Save all image scaling results (downscale and upscale)\n') +
             _('(does not work with --ml, --train-ml and --generate-dataset)')
    )

    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help=_('Disable parallel processing and use single-threaded scheme')
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help=_('Do not use GPU for metrics calculation (in case of problems with CUDA, MPS, etc. in PyTorch)')
    )

    parser.add_argument(
        '--generate-dataset',
        action='store_true',
        help=_('Generate dataset (features/targets) for model training')
    )

    parser.add_argument(
        '--train-ml',
        action='store_true',
        help=_('Train ML model after dataset generation')
    )

    parser.add_argument(
        '--ml',
        action='store_true',
        help=_('Use ML model to predict metrics instead of real calculation (fast)')
    )

    return parser

def format_threads_help() -> str:
    return (_("Number of parallel processes for file processing. Ignored with --no-parallel,\n") +
            _("default is ") + str(default_threads_count) + " (" +
            _("logical processor cores detected: ") +
            str(multiprocessing.cpu_count()) + ")")

def format_metric_help() -> str:
    """
    Return a string containing the list of available quality metrics.
    """
    metrics = [
        f"{m.value:<12} {QUALITY_METRICS_INFO[m]:<20}{' (' + _('default') + ')' if m.value == QUALITY_METRIC_DEFAULT else '':>10}"
        for m in QualityMetrics
    ]
    return _("Available quality metrics") + ":\n" + "\n".join(metrics)

def format_interpolation_help() -> str:
    """
    Return a string containing the list of available interpolation methods.
    """
    methods = [
        f"{m.value:<12}{INTERPOLATION_METHODS_INFO[m]:<20}{' (' + _('default') + ')' if m.value == INTERPOLATION_METHOD_DEFAULT else '':>10}"
        for m in InterpolationMethods
    ]
    return _("Available interpolation methods") + ":\n" + "\n".join(methods)


def validate_paths(paths: list[str]) -> list[str]:
    """
    Validate paths and return a list of valid paths with supported extensions.
    """
    valid_paths = []
    invalid_paths = []

    for path in paths:
        if os.path.isfile(path):
            # Проверяем расширение для отдельных файлов
            if os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS:
                valid_paths.append(path)
            else:
                logging.warning("Неподдерживаемое расширение файла: %s", path)
                invalid_paths.append(path)
        elif os.path.isdir(path):
            # Собираем файлы с поддерживаемыми расширениями из директории
            dir_files = collect_files_from_dir(path)
            if not dir_files:
                logging.warning("В директории %s не найдено файлов с поддерживаемыми расширениями", path)
                invalid_paths.append(path)
            valid_paths.extend(dir_files)
        else:
            logging.warning("Неверный путь: %s", path)
            invalid_paths.append(path)

    if not valid_paths:
        error_message = "Не найдено ни одного валидного файла с поддерживаемым расширением."
        if invalid_paths:
            error_message += " Проверьте следующие пути: " + ", ".join(invalid_paths)
        logging.error(error_message)
        raise ValueError(error_message)

    return valid_paths


def collect_files_from_dir(directory: str) -> list[str]:
    """
    Recursively collects and returns a list of file paths with supported extensions
    from the specified directory.
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
