# utils/cli.py
"""
Command line interface for ResolutionSuggester.
"""
import argparse
import logging
import multiprocessing
import os
import sys

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
from ..i18n import _


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
    # Создаём парсер с локализованным описанием
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
    pre_args, remaining_args = pre_parser.parse_known_args()

    # Если указан язык, переустанавливаем локализацию
    if pre_args.lang != 'auto':
        setup_localization(pre_args.lang)

    parser = create_parser()

    # Если запрошена справка, показываем справку для соответствующей команды и выходим
    if pre_args.help:
        # Проверяем, запрошена ли справка для конкретной подкоманды
        subcommand = None
        for arg in remaining_args:
            if arg in ['analyze', 'model']:
                subcommand = arg
                break
        
        if subcommand:
            # Получаем субпарсер для указанной команды и показываем его справку
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    if subcommand in action.choices:
                        action.choices[subcommand].print_help()
                        sys.exit(0)
        else:
            # Показываем общую справку
            parser.print_help()
            sys.exit(0)

    # Теперь парсим все аргументы
    args = parser.parse_args()
    
    # Обрабатываем аргументы и преобразуем их для обратной совместимости
    args = handle_backward_compatibility(args)

    if args.lang != 'auto':
        from ..i18n import setup_localization
        setup_localization(args.lang)

    if hasattr(args, 'save_im_all') and args.save_im_all:
        args.save_im_down = True
        args.save_im_up = True

    # Model subcommand validation
    if hasattr(args, 'subcommand') and args.subcommand == 'model':
        if hasattr(args, 'no_parallel') and args.no_parallel and hasattr(args, 'generate_dataset') and args.generate_dataset:
            logging.info(_("Parameters --generate-dataset and --no-parallel are used simultaneously.\n"
                         "The --no-parallel parameter will be ignored, and --threads will be forced to 1."))
            args.no_parallel = False
            args.threads = 1
    # Main command validation
    elif hasattr(args, 'compare_ml') and args.compare_ml:
        if hasattr(args, 'ml') and args.ml:
            logging.warning(_("Cannot use --ml together with --compare-ml!\n"
                            "The --ml parameter will be ignored."))
            args.ml = False

    if hasattr(args, 'ml') and args.ml:
        if hasattr(args, 'save_im_down') and (args.save_im_down or args.save_im_up or args.save_im_all):
            logging.warning(_("Cannot use --save-im-* together with --ml!\n"
                            "The --save-im-* parameters will be ignored."))
            args.save_im_down = False
            args.save_im_up = False

    if hasattr(args, 'min_size') and args.min_size < MIN_DOWNSCALE_SIZE:
        logging.warning(
            _("Minimum size (width and height) for analysis must be >= %s. "
            "Set to default value: %s"),
            MIN_DOWNSCALE_SIZE, MIN_DOWNSCALE_SIZE
        )
        args.min_size = MIN_DOWNSCALE_SIZE

    if hasattr(args, 'threads') and args.threads < 1:
        logging.warning(
            _("Number of parallel processes must be >= 1. "
            "Set to minimum value: 1")
        )
        args.threads = 1

    return args


def handle_backward_compatibility(args: argparse.Namespace) -> argparse.Namespace:
    """
    Обрабатывает аргументы командной строки для обеспечения обратной совместимости.
    Если не указана подкоманда, но указаны пути или параметры, 
    автоматически преобразует в соответствующую подкоманду.
    """
    # Проверяем наличие подкоманды
    has_explicit_subcommand = hasattr(args, 'subcommand') and args.subcommand in ['analyze', 'model']
    
    # Если подкоманда явно указана, используем пути из соответствующего аргумента
    if has_explicit_subcommand:
        # Пути уже должны быть правильно установлены парсером
        pass
    else:
        # Для старого стиля: перенос из "default_*" в прямые параметры
        default_paths = getattr(args, 'default_paths', []) if hasattr(args, 'default_paths') else []
        
        # Проверяем, нужно ли использовать команду model
        if hasattr(args, 'generate_dataset_default') and args.generate_dataset_default:
            # Это команда model с --generate-dataset
            args.subcommand = 'model'
            args.paths = default_paths
            args.generate_dataset = True
            
            # Копируем другие параметры, которые могут быть использованы с model
            if hasattr(args, 'train_ml_default') and args.train_ml_default:
                args.train_ml = args.train_ml_default
                
            if hasattr(args, 'threads_default'):
                args.threads = args.threads_default
                
            if hasattr(args, 'no_parallel_default'):
                args.no_parallel = args.no_parallel_default
                
            if hasattr(args, 'no_gpu_default'):
                args.no_gpu = args.no_gpu_default
                
            if hasattr(args, 'min_size_default'):
                args.min_size = args.min_size_default
                
            if hasattr(args, 'lpips_net_default'):
                args.lpips_net = args.lpips_net_default
            
            # Не даем запуститься, если не указаны пути для model
            if not default_paths:
                logging.error(_("No paths specified for 'model' subcommand. Please provide paths to image files or directories."))
                sys.exit(1)
        else:
            # Это обычная команда analyze
            args.subcommand = 'analyze'
            args.paths = default_paths
            
            # Не даем запуститься, если не указаны пути для analyze
            if not default_paths:
                if not has_explicit_subcommand:  # Только если команда была неявной
                    logging.error(_("No paths specified for analysis. Please provide paths to image files or directories."))
                    sys.exit(1)
            
            # Копируем все параметры из "default_*" в обычные параметры
            copy_default_params = [
                ('channels', 'channels_default'),
                ('csv_output', 'csv_output_default'),
                ('json_output', 'json_output_default'),
                ('chart', 'chart_default'),
                ('theme', 'theme_default'),
                ('metric', 'metric_default'),
                ('lpips_net', 'lpips_net_default'),
                ('interpolation', 'interpolation_default'),
                ('min_size', 'min_size_default'),
                ('threads', 'threads_default'),
                ('save_im_down', 'save_im_down_default'),
                ('save_im_up', 'save_im_up_default'),
                ('save_im_all', 'save_im_all_default'),
                ('no_parallel', 'no_parallel_default'),
                ('no_gpu', 'no_gpu_default'),
                ('ml', 'ml_default'),
                ('compare_ml', 'compare_ml_default')
            ]
            
            for target, source in copy_default_params:
                if hasattr(args, source) and getattr(args, source) is not None:
                    setattr(args, target, getattr(args, source))
        
    # Обработка предупреждений для старого стиля команд
    if hasattr(args, 'generate_dataset_default') and args.generate_dataset_default and \
       args.subcommand != 'model':
        logging.warning(_("The --generate-dataset parameter is now part of the 'model' subcommand.\n"
                        "Please use 'res-suggest model --generate-dataset' instead."))
        
    if hasattr(args, 'train_ml_default') and args.train_ml_default and \
       args.subcommand != 'model':
        logging.warning(_("The --train-ml parameter is now part of the 'model' subcommand.\n"
                        "Please use 'res-suggest model --train-ml' instead."))
    
    return args

def create_parser() -> argparse.ArgumentParser:
    """
    Создаёт и настраивает парсер аргументов с текущими переводами.
    Включает поддержку подкоманд.
    """
    parser = argparse.ArgumentParser(
        description=_('Texture quality analysis tool'),
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Аргументы, общие для всех команд
    parser.add_argument(
        '--lang',
        choices=['en', 'ru', 'auto'],
        default='auto',
        help=_('Interface language (default: auto)')
    )

    # Создаем подпарсеры для разных команд
    subparsers = parser.add_subparsers(dest='subcommand', help=_('Available subcommands'))
    
    # Парсер для основного режима анализа (по умолчанию)
    main_parser = subparsers.add_parser(
        'analyze', 
        help=_('Analyze image quality (default mode)'),
        description=_('Analyze image quality'),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True
    )
    
    # Добавляем аргументы для основного режима анализа
    main_parser.add_argument(
        'paths',
        nargs='+',
        help=_('Paths to files/directories for analysis')
    )

    main_parser.add_argument(
        '-c', '--channels',
        action='store_true',
        help=_('Analysis by color channels')
    )

    main_parser.add_argument(
        '-o', '--csv-output',
        action='store_true',
        help=_('Export results to CSV')
    )

    main_parser.add_argument(
        '-j', '--json-output',
        action='store_true',
        help=_('Export results to JSON')
    )

    main_parser.add_argument(
        '--chart',
        action='store_true',
        help=_('Generate quality vs. resolution charts')
    )

    main_parser.add_argument(
        '--theme',
        choices=['light', 'dark'],
        default='dark',
        help=_('Charts theme (default: dark)')
    )

    main_parser.add_argument(
        '-m', '--metric', type=QualityMetrics,
        default=QUALITY_METRIC_DEFAULT,
        choices=[m.value for m in QualityMetrics],
        metavar='QUALITY_METRIC',
        help=format_metric_help()
    )

    main_parser.add_argument(
        '--lpips-net',
        choices=['alex', 'vgg', 'squeeze'],
        default='alex',
        metavar='NEURAL_NETWORK',
        help=_('LPIPS neural network backbone:\nalex (balanced)\nvgg (memory-hungry)\nsqueeze (fast but less accurate)')
    )

    main_parser.add_argument(
        '-i', '--interpolation', type=InterpolationMethods,
        default=INTERPOLATION_METHOD_DEFAULT,
        choices=[m.value for m in InterpolationMethods],
        metavar='METHOD',
        help=format_interpolation_help()
    )

    main_parser.add_argument(
        '--min-size',
        type=int,
        default=MIN_DOWNSCALE_SIZE,
        metavar='SIZE',
        help=_("Minimum size (width and height) for analysis (default and minimum: ") +
             str(MIN_DOWNSCALE_SIZE) + ")"
    )

    main_parser.add_argument(
        '-t', '--threads',
        type=int,
        default=default_threads_count,
        metavar='N',
        help=format_threads_help()
    )

    main_parser.add_argument(
        '--save-im-down',
        action='store_true',
        help=_('Save downscale results produced during analysis')
    )

    main_parser.add_argument(
        '--save-im-up',
        action='store_true',
        help=_('Save upscale results produced after downscale')
    )

    main_parser.add_argument(
        '-s', '--save-im-all',
        action='store_true',
        help=_('Save all image scaling results (downscale and upscale)')
    )

    main_parser.add_argument(
        '--no-parallel',
        action='store_true',
        help=_('Disable parallel processing and use a single-threaded scheme')
    )

    main_parser.add_argument(
        '--no-gpu',
        action='store_true',
        help=_('Do not use GPU for metrics calculation (in case of problems with CUDA, MPS, etc. in PyTorch)')
    )

    main_parser.add_argument(
        '--ml',
        action='store_true',
        help=_('Use ML model to predict metrics instead of real calculation (fast)')
    )

    main_parser.add_argument(
        '--compare-ml',
        action='store_true',
        help=_('Run comparison of real and ML results')
    )
    
    # Парсер для подкоманды model (работа с ML моделями)
    model_parser = subparsers.add_parser(
        'model',
        help=_('ML model operations (dataset generation, training)'),
        description=_('ML model operations (dataset generation, training)'),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True
    )
    
    # Аргументы для подкоманды model
    model_parser.add_argument(
        'paths',
        nargs='+',
        help=_('Paths to files/directories for analysis')
    )
    
    model_parser.add_argument(
        '--generate-dataset',
        action='store_true',
        help=_('Generate dataset (features/targets) for model training')
    )
    
    model_parser.add_argument(
        '--train-ml',
        action='store_true',
        help=_('Train ML model after dataset generation')
    )
    
    model_parser.add_argument(
        '--min-size',
        type=int,
        default=MIN_DOWNSCALE_SIZE,
        metavar='SIZE',
        help=_("Minimum size (width and height) for analysis (default and minimum: ") +
             str(MIN_DOWNSCALE_SIZE) + ")"
    )
    
    model_parser.add_argument(
        '-t', '--threads',
        type=int,
        default=default_threads_count,
        metavar='N',
        help=format_threads_help()
    )
    
    model_parser.add_argument(
        '--no-parallel',
        action='store_true',
        help=_('Disable parallel processing and use a single-threaded scheme\n') +
             _('(ignored and forces --threads 1 instead, when used with --generate-dataset)')
    )
    
    model_parser.add_argument(
        '--no-gpu',
        action='store_true',
        help=_('Do not use GPU for metrics calculation (in case of problems with CUDA, MPS, etc. in PyTorch)')
    )
    
    model_parser.add_argument(
        '--lpips-net',
        choices=['alex', 'vgg', 'squeeze'],
        default='alex',
        metavar='NEURAL_NETWORK',
        help=_('LPIPS neural network backbone:\nalex (balanced)\nvgg (memory-hungry)\nsqueeze (fast but less accurate)')
    )
    
    # Для совместимости с прежними скриптами добавляем возможность запускать без указания подкоманды
    parser.add_argument(
        'default_paths',
        nargs='*',
        help=argparse.SUPPRESS
    )
    
    # Добавляем все другие аргументы основной команды анализа, которые должны быть доступны на верхнем уровне
    # для обратной совместимости
    parser.add_argument(
        '-c', '--channels',
        action='store_true',
        help=_('Analysis by color channels'),
        dest='channels_default'
    )
    
    parser.add_argument(
        '-o', '--csv-output',
        action='store_true',
        help=_('Export results to CSV'),
        dest='csv_output_default'
    )
    
    parser.add_argument(
        '-j', '--json-output',
        action='store_true',
        help=_('Export results to JSON'),
        dest='json_output_default'
    )
    
    parser.add_argument(
        '--chart',
        action='store_true',
        help=_('Generate quality vs. resolution charts'),
        dest='chart_default'
    )
    
    parser.add_argument(
        '--theme',
        choices=['light', 'dark'],
        default='dark',
        help=_('Charts theme (default: dark)'),
        dest='theme_default'
    )
    
    parser.add_argument(
        '-m', '--metric', type=QualityMetrics,
        default=QUALITY_METRIC_DEFAULT,
        choices=[m.value for m in QualityMetrics],
        metavar='QUALITY_METRIC',
        help=format_metric_help(),
        dest='metric_default'
    )
    
    parser.add_argument(
        '--lpips-net',
        choices=['alex', 'vgg', 'squeeze'],
        default='alex',
        metavar='NEURAL_NETWORK',
        help=_('LPIPS neural network backbone:\nalex (balanced)\nvgg (memory-hungry)\nsqueeze (fast but less accurate)'),
        dest='lpips_net_default'
    )
    
    parser.add_argument(
        '-i', '--interpolation', type=InterpolationMethods,
        default=INTERPOLATION_METHOD_DEFAULT,
        choices=[m.value for m in InterpolationMethods],
        metavar='METHOD',
        help=format_interpolation_help(),
        dest='interpolation_default'
    )
    
    parser.add_argument(
        '--min-size',
        type=int,
        default=MIN_DOWNSCALE_SIZE,
        metavar='SIZE',
        help=_("Minimum size (width and height) for analysis (default and minimum: ") +
             str(MIN_DOWNSCALE_SIZE) + ")",
        dest='min_size_default'
    )
    
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=default_threads_count,
        metavar='N',
        help=format_threads_help(),
        dest='threads_default'
    )
    
    parser.add_argument(
        '--save-im-down',
        action='store_true',
        help=_('Save downscale results produced during analysis'),
        dest='save_im_down_default'
    )
    
    parser.add_argument(
        '--save-im-up',
        action='store_true',
        help=_('Save upscale results produced after downscale'),
        dest='save_im_up_default'
    )
    
    parser.add_argument(
        '-s', '--save-im-all',
        action='store_true',
        help=_('Save all image scaling results (downscale and upscale)'),
        dest='save_im_all_default'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help=_('Disable parallel processing and use a single-threaded scheme'),
        dest='no_parallel_default'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help=_('Do not use GPU for metrics calculation (in case of problems with CUDA, MPS, etc. in PyTorch)'),
        dest='no_gpu_default'
    )
    
    parser.add_argument(
        '--ml',
        action='store_true',
        help=_('Use ML model to predict metrics instead of real calculation (fast)'),
        dest='ml_default'
    )
    
    parser.add_argument(
        '--compare-ml',
        action='store_true',
        help=_('Run comparison of real and ML results'),
        dest='compare_ml_default'
    )
    
    # Устаревшие параметры для обратной совместимости
    parser.add_argument(
        '--generate-dataset',
        action='store_true',
        help=argparse.SUPPRESS,
        dest='generate_dataset_default'
    )
    
    parser.add_argument(
        '--train-ml',
        action='store_true',
        help=argparse.SUPPRESS,
        dest='train_ml_default'
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
                logging.warning(_("Unsupported file extension: %s"), path)
                invalid_paths.append(path)
        elif os.path.isdir(path):
            # Собираем файлы с поддерживаемыми расширениями из директории
            dir_files = collect_files_from_dir(path)
            if not dir_files:
                logging.warning(_("No files with supported extensions found in directory %s"), path)
                invalid_paths.append(path)
            valid_paths.extend(dir_files)
        else:
            logging.warning(_("Invalid path: %s"), path)
            invalid_paths.append(path)

    if not valid_paths:
        error_message = _("No valid files with supported extensions found.")
        if invalid_paths:
            error_message += _(" Check the following paths: ") + ", ".join(invalid_paths)
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
        for root, __, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    collected.append(os.path.join(root, f))
    except PermissionError as e:
        logging.error(f"{_('Permission error accessing directory')} {directory}, {str(e)}")
    except OSError as e:
        logging.error(f"{_('Unexpected error walking directory')} {directory}, {str(e)}")
    return collected
