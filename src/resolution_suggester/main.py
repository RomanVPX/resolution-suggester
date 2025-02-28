# main.py
import os

from resolution_suggester.core.image_analyzer import ImageAnalyzer

# отключаем предупреждение omp_set_nested routine deprecated от PyTorch
os.environ["KMP_WARNINGS"] = "off"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

import sys

if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

import argparse
import concurrent.futures
import logging

import pandas as pd

from .i18n import _
from tqdm import tqdm

from .config import (
    INTERPOLATION_METHOD_UPSCALE,
    ML_DATA_DIR,
    ML_DATASETS_DIR,
    InterpolationMethods,
    QualityMetrics,
)
from .core.image_loader import load_image
from .core.image_processing import get_resize_function
from .core.metrics import calculate_metrics, compute_resolutions
from .ml.predictor import QuickPredictor, extract_features_of_original_img
from .utils.cli import parse_arguments, setup_logging, validate_paths

from .utils.reporters import (
    IReporter,
    CSVReporter,
    JSONReporter,
    get_csv_log_filename,
    get_json_log_filename
)


def main() -> None:
    """Main function of the program."""
    setup_logging()

    try:
        # Обработка аргументов
        args = parse_and_validate_arguments()

        # Получение списка файлов
        files = get_file_list(args.paths)

        # Запуск нужного режима работы
        if args.generate_dataset:
            run_dataset_generation(files, args)
        else:
            run_image_analysis(files, args)
    except Exception as e:
        logging.error(f"{_('Unexpected error')}: {str(e)}")
        sys.exit(1)


def parse_and_validate_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    try:
        args = parse_arguments()
        return args
    except SystemExit:
        # Перехватываем выход из программы при показе справки
        sys.exit(0)
    except Exception as e:
        logging.error(f"{_('Error parsing arguments')}: {str(e)}")
        sys.exit(1)


def get_file_list(paths: list[str]) -> list[str]:
    """Validates paths and returns a list of files for processing."""
    try:
        return validate_paths(paths)
    except ValueError as e:
        logging.error(f"{str(e)} Завершение работы.")
        sys.exit(1)


def run_dataset_generation(files: list[str], args: argparse.Namespace) -> None:
    """Launches dataset generation and optionally trains the model."""
    features_path, targets_path = generate_dataset(files, args)
    logging.info(f"{_("Dataset generated")}: features={features_path}, targets={targets_path}")

    if args.train_ml:
        predictor = QuickPredictor()
        predictor.train(features_path, targets_path)
        logging.info(_("Model trained!"))


def run_image_analysis(files: list[str], args: argparse.Namespace) -> None:
    """Launches image analysis with reporter configuration."""
    reporters, output_paths = setup_reporters(args)

    if getattr(args, 'chart', False):
        try:
            import matplotlib
            matplotlib.use('Agg')  # Не-интерактивный бэкенд
            logging.info(_("Chart generation enabled"))
        except ImportError:
            import matplotlib
            logging.error(_("Failed to import matplotlib."))
            args.chart = False

    try:
        analyzer = ImageAnalyzer(args, reporters)
        analyzer.analyze_files(files)
    finally:
        close_reporters(reporters)

    if 'csv' in output_paths:
        print(f"\n{_("Metrics(CSV) saved to")}: {output_paths['csv']}")
    if 'json' in output_paths:
        print(f"\n{_("Metrics(JSON) saved to")}: {output_paths['json']}")


def setup_reporters(args: argparse.Namespace) -> tuple[list[IReporter], dict[str, str]]:
    """Configures reporters for analysis results output."""
    reporters = []
    output_paths = {}

    if args.csv_output:
        csv_path = get_csv_log_filename(args)
        csv_reporter = CSVReporter(csv_path, QualityMetrics(args.metric))
        csv_reporter.__enter__()
        csv_reporter.write_header(args.channels)
        reporters.append(csv_reporter)
        output_paths['csv'] = csv_path
        logging.info(f"{_("CSV output enabled, file")}: {csv_path}")

    if args.json_output:
        json_path = get_json_log_filename(args)
        json_reporter = JSONReporter(json_path, QualityMetrics(args.metric))
        json_reporter.__enter__()
        reporters.append(json_reporter)
        output_paths['json'] = json_path
        logging.info(f"{_("JSON output enabled, file")}: {json_path}")

    return reporters, output_paths


def close_reporters(reporters: list[IReporter]) -> None:
    """Closes all reporters."""
    for rep in reporters:
        try:
            rep.__exit__(None, None, None)
        except Exception as e:
            logging.error(f"{_("Error when closing reporter")}: {e}")


def process_file_for_dataset(
        file_path: str,
        interpolations_methods: list[InterpolationMethods],
        args: argparse.Namespace
) -> tuple[list[dict], list[dict]]:
    """
    Process a single file to extract features and calculate metrics for the dataset.

    Args:
        file_path: Path to the image file
        interpolations_methods: List of interpolation methods to test
        args: Command line arguments

    Returns:
        Tuple of (features, targets) lists for the processed file
    """
    features_all = []
    all_targets = []

    image_load_result = load_image(file_path)
    if image_load_result.error or image_load_result.data is None:
        logging.warning(f"{_("Skipping")} {file_path}, {_("because failed to load")}.")
        return features_all, all_targets

    img_original = image_load_result.data
    max_val = image_load_result.max_value
    original_h, original_w = img_original.shape[:2]
    channels = image_load_result.channels or ['L']

    resolutions_to_test = compute_resolutions(original_w, original_h, args.min_size)
    if not resolutions_to_test:
        return features_all, all_targets

    for method in interpolations_methods:
        try:
            resize_fn = get_resize_function(method)
            resize_fn_upscale = get_resize_function(INTERPOLATION_METHOD_UPSCALE)
        except ValueError as e:
            logging.error(f"{_("Error when selecting interpolation function for")} {file_path}: {e}")
            continue

        for (w, h) in resolutions_to_test:
            if w == original_w and h == original_h:
                continue

            scale_factor = (w / original_w + h / original_h) / 2

            # Базовые фичи (для общего режима)
            features_dict_base = {
                'scale_factor': scale_factor,
                'method': method.value,
                'original_width': original_w,
                'original_height': original_h,
            }

            img_downscaled = resize_fn(img_original, w, h)
            img_upscaled = resize_fn_upscale(img_downscaled, original_w, original_h)

            # Создаём две версии данных: без каналов и с каналами
            for analyze_channels in [0, 1]:
                if analyze_channels:
                    # Теперь итерируемся по каждому каналу
                    for c in channels:
                        features_entry = features_dict_base.copy()  # используем базовый словарь
                        features_entry['analyze_channels'] = analyze_channels
                        features_entry['channel'] = c

                        img_channel = img_original[..., channels.index(c)] if img_original.ndim == 3 else img_original
                        img_upscaled_channel = img_upscaled[..., channels.index(c)] if img_upscaled.ndim == 3 else img_original

                        channel_features = extract_features_of_original_img(img_channel)
                        features_entry.update(channel_features)  # Добавляем канальные фичи

                        targets_entry = {}
                        for metric in QualityMetrics:
                            channel_metric_value = calculate_metrics(metric, img_channel, img_upscaled_channel, max_val, no_gpu=args.no_gpu)
                            targets_entry[metric.value] = channel_metric_value

                        features_all.append(features_entry)
                        all_targets.append(targets_entry)
                else:
                    features_entry = features_dict_base.copy()  # используем базовый словарь
                    features_entry['analyze_channels'] = analyze_channels

                    features_original = extract_features_of_original_img(img_original)
                    features_entry.update(features_original)  # Добавляем общие фичи

                    metrics_combined = {
                        'psnr': calculate_metrics(QualityMetrics.PSNR, img_original, img_upscaled, max_val, no_gpu=args.no_gpu),
                        'ssim': calculate_metrics(QualityMetrics.SSIM, img_original, img_upscaled, max_val, no_gpu=args.no_gpu),
                        'ms_ssim': calculate_metrics(QualityMetrics.MS_SSIM, img_original, img_upscaled, max_val, no_gpu=args.no_gpu),
                        'tdpr': calculate_metrics(QualityMetrics.TDPR, img_original, img_upscaled, max_val, no_gpu=args.no_gpu)
                    }
                    targets_entry = metrics_combined.copy()

                    features_all.append(features_entry)
                    all_targets.append(targets_entry)

            del img_downscaled, img_upscaled

    return features_all, all_targets


def generate_dataset(files: list[str], args: argparse.Namespace) -> tuple[str, str]:
    """
    Generate a dataset from a list of image files.

    Args:
        files: List of image file paths
        args: Command line arguments

    Returns:
        Tuple of (features_csv_path, targets_csv_path) as strings
    """
    features_all = []
    all_targets = []

    # Пути для сохранения датасета
    features_csv_path = ML_DATASETS_DIR / 'features.csv'
    targets_csv_path  = ML_DATASETS_DIR / 'targets.csv'
    features_csv = str(features_csv_path)
    targets_csv = str(targets_csv_path)

    if not ML_DATA_DIR.exists():
        ML_DATA_DIR.mkdir(parents=True, exist_ok=True)

    interpolations_methods_to_test = [
        InterpolationMethods.BILINEAR,
        InterpolationMethods.BICUBIC,
        InterpolationMethods.MITCHELL
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_file_for_dataset, file_path,
                            interpolations_methods_to_test, args)
            for file_path in files
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=_("Creating dataset")):
            try:
                features, targets = future.result()
                features_all.extend(features)
                all_targets.extend(targets)
            except Exception as e:
                logging.error(f"{_("Error when processing file")}: {e}")

    if features_all:
        df_features = pd.DataFrame(features_all)
        df_features.to_csv(features_csv, index=False)
    else:
        logging.warning(f"{_("No data to save in")} features.csv")

    if all_targets:
        df_targets = pd.DataFrame(all_targets)
        df_targets.to_csv(targets_csv, index=False)
    else:
        logging.warning(f"{_("No data to save in")} targets.csv")

    return features_csv, targets_csv


if __name__ == "__main__":
    main()
