# main.py
import os

# отключаем предупреждение omp_set_nested routine deprecated от PyTorch
os.environ["KMP_WARNINGS"] = "off"

import sys

if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

import argparse
import concurrent.futures
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import (
    INTERMEDIATE_DIR,
    INTERPOLATION_METHOD_UPSCALE,
    ML_DATA_DIR,
    ML_DATASETS_DIR,
    PSNR_IS_LARGE_AS_INF,
    InterpolationMethods,
    QualityMetrics,
)
from .core.image_loader import load_image
from .core.image_processing import get_resize_function
from .core.metrics import calculate_metrics, compute_resolutions
from .ml.predictor import QuickPredictor, extract_features_of_original_img
from .utils.cli import parse_arguments, setup_logging, validate_paths
from .utils.reporting import ConsoleReporter, QualityHelper

from .utils.reporters import (
    IReporter,
    CSVReporter,
    JSONReporter,
    get_csv_log_filename,
    get_json_log_filename
)


def main() -> None:
    setup_logging()
    args = parse_arguments()
    try:
        files = validate_paths(args.paths)
    except ValueError as e:
        logging.error(str(e) + " Завершение работы.")
        return

    if args.generate_dataset:
        features_path, targets_path = generate_dataset(files, args)
        logging.info(f"Датасет сгенерирован: features={features_path}, targets={targets_path}")
        if args.train_ml:
            predictor = QuickPredictor()
            predictor.train(features_path, targets_path)
            logging.info("Модель обучена!")
        return

    reporters: list[IReporter] = []

    if args.csv_output:
        csv_path = get_csv_log_filename(args)
        csv_reporter = CSVReporter(csv_path, QualityMetrics(args.metric))
        csv_reporter.__enter__()
        csv_reporter.write_header(args.channels)
        reporters.append(csv_reporter)
        logging.info("CSV output включён, файл: %s", csv_path)

    if args.json_output:
        json_path = get_json_log_filename(args)
        json_reporter = JSONReporter(json_path, QualityMetrics(args.metric))
        json_reporter.__enter__()
        reporters.append(json_reporter)
        logging.info("JSON output включён, файл: %s", json_path)

    process_files(files, args, reporters)

    for rep in reporters:
        rep.__exit__(None, None, None)

    if args.csv_output:
        print(f"\nМетрики (CSV) сохранены в: {csv_path}")
    if args.json_output:
        print(f"\nМетрики (JSON) сохранены в: {json_path}")


def process_files(files: list[str], args: argparse.Namespace, reporters: list[IReporter]):
    if args.no_parallel:
        for file_path in files:
            try:
                results, meta = process_single_file(file_path, args)
            except Exception as e:
                logging.error(f"Ошибка обработки {file_path}: {e}")
                continue
            if results:
                print_console_results(file_path, results, args.channels, meta, QualityMetrics(args.metric))
                for rep in reporters:
                    rep.write_results(os.path.basename(file_path), results, args.channels)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
            future_to_file = {
                executor.submit(process_single_file, file_path, args): file_path
                for file_path in files
            }
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    results, meta = future.result()
                except Exception as e:
                    logging.error(f"Ошибка обработки {file_path}: {e}")
                    continue
                if results:
                    print_console_results(file_path, results, args.channels, meta, QualityMetrics(args.metric))
                    for rep in reporters:
                        rep.write_results(os.path.basename(file_path), results, args.channels)

def process_single_file(file_path: str, args: argparse.Namespace) -> Tuple[Optional[list], Optional[dict]]:
    image_load_result = load_image(file_path)
    if image_load_result.error or image_load_result.data is None:
        logging.error(f"Ошибка загрузки изображения {file_path}: {image_load_result.error}")
        return None, None

    img_original = image_load_result.data
    max_val = image_load_result.max_value
    channels = image_load_result.channels
    height, width = img_original.shape[:2]

    if height < args.min_size or width < args.min_size:
        logging.info(f"Пропуск {file_path} (размер {width}x{height}) - меньше, чем min_size = {args.min_size}")
        return None, None

    results = [create_original_entry(width, height, channels, args.channels)]
    resolutions = compute_resolutions(width, height, args.min_size)

    use_prediction = args.ml
    predictor = None
    if use_prediction:
        predictor = QuickPredictor()
        # Важно: сначала устанавливаем режим, затем загружаем модель
        predictor.set_mode(args.channels)
        if not predictor.load():
            logging.info("ML-модель не найдена, будем вычислять реальные метрики.")
            use_prediction = False
    else:
        try:
            resize_fn = get_resize_function(args.interpolation)
            resize_fn_upscale = get_resize_function(INTERPOLATION_METHOD_UPSCALE)
        except ValueError as e:
            logging.error(f"Ошибка при выборе функции интерполяции для {file_path}: {e}")
            return None, None

    for (w, h) in resolutions:
        if w == width and h == height:
            continue

        if not use_prediction:
            img_downscaled = resize_fn(img_original, w, h)
            if args.save_im_down:
                _save_intermediate(img_downscaled, file_path, w, h,
                                   InterpolationMethods(args.interpolation), '')
            img_upscaled = resize_fn_upscale(img_downscaled, width, height)
            if args.save_im_up:
                _save_intermediate(img_upscaled, file_path, w, h,
                                   InterpolationMethods(args.interpolation), 'upscaled')

        if args.channels:
            if use_prediction:
                channels_metrics = _predict_channel_metrics(img_original, w, width, h, height,
                                                            args, predictor, channels)
                min_metric = min(channels_metrics.values())
                results_entry = (f"{w}x{h}", channels_metrics, min_metric)
            else:
                channels_metrics = calculate_metrics(QualityMetrics(args.metric), img_original,
                                                     img_upscaled, max_val, channels, no_gpu=args.no_gpu)
                channels_metrics = postprocess_metric_value(channels_metrics, args.metric)
                min_metric = min(channels_metrics.values())
                results_entry = (f"{w}x{h}", channels_metrics, min_metric)
        else:
            if use_prediction:
                metric_value = _predict_combined_metric(img_original, w, width, h, height, args, predictor
                )
                results_entry = (f"{w}x{h}", metric_value)
            else:
                metric_value = calculate_metrics(QualityMetrics(args.metric), img_original, img_upscaled, max_val, no_gpu=args.no_gpu)
                metric_value = postprocess_metric_value(metric_value, args.metric)
                results_entry = (f"{w}x{h}", metric_value)

        if args.channels:
            results.append((*results_entry, QualityHelper.get_hint(results_entry[2], QualityMetrics(args.metric))))
        else:
            results.append((*results_entry, QualityHelper.get_hint(results_entry[1], QualityMetrics(args.metric))))

    return results, {'max_val': max_val, 'channels': channels}


def _predict_channel_metrics(img_original, w, width, h, height, args, predictor, channels):
    """Вспомогательная функция для предсказания поканальных метрик."""
    channels_metrics = {}
    for c in channels:
        features_for_ml_channel = extract_features_of_original_img(img_original[..., channels.index(c)])
        features_for_ml_channel.update({
            'scale_factor': (w / width + h / height) / 2,
            'original_width': width,
            'original_height': height,
            'channel': c,
            'method': args.interpolation,
        })
        prediction_channel = predictor.predict(features_for_ml_channel)
        val = prediction_channel.get(args.metric.value, 0.0)
        channels_metrics[c] = float('inf') if val >= PSNR_IS_LARGE_AS_INF else val
    return channels_metrics

def _predict_combined_metric(img_original, w, width, h, height, args, predictor):
    """Вспомогательная функция для предсказания общей метрики (без каналов)."""
    features_for_ml = extract_features_of_original_img(img_original)
    features_for_ml.update({
        'scale_factor': (w / width + h / height) / 2,
        'original_width': width,
        'original_height': height,
        'method': args.interpolation,
        'channel': 'combined'
    })
    prediction = predictor.predict(features_for_ml)
    metric_value = prediction.get(args.metric.value, 0.0)
    metric_value = float('inf') if metric_value >= PSNR_IS_LARGE_AS_INF else metric_value
    return metric_value

def postprocess_channel_metrics(channels_metrics, metric_type):
    """Заменяет значения PSNR >= PSNR_IS_LARGE_AS_INF на float('inf') в словаре поканальных метрик."""
    processed_metrics = {}
    for c, metric_value in channels_metrics.items():
        if QualityMetrics(metric_type) == QualityMetrics.PSNR:
            processed_metrics[c] = float('inf') if metric_value >= PSNR_IS_LARGE_AS_INF else metric_value
        else:
            processed_metrics[c] = metric_value
    return processed_metrics


def postprocess_metric_value(metric_value, metric_type):
    """
    Method for postprocessing metric value.
    """
    if QualityMetrics(metric_type) != QualityMetrics.PSNR:
        return metric_value

    try:
        return {
            channel: float('inf') if value >= PSNR_IS_LARGE_AS_INF else value
            for channel, value in metric_value.items()
        }
    except (AttributeError, TypeError):
        try:
            return float('inf') if metric_value >= PSNR_IS_LARGE_AS_INF else metric_value
        except (TypeError, ValueError):
            # Не удалось обработать и как скаляр
            raise TypeError(f"Ожидается число или словарь, получено: {type(metric_value).__name__}")


def process_file_for_dataset(
        file_path: str,
        interpolations_methods: list[InterpolationMethods],
        args: argparse.Namespace
) -> tuple[list[dict], list[dict]]:
    features_all = []
    all_targets = []

    image_load_result = load_image(file_path)
    if image_load_result.error or image_load_result.data is None:
        logging.warning(f"Пропуск {file_path}, т.к. не удалось загрузить.")
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
            logging.error(f"Ошибка при выборе функции интерполяции для {file_path}: {e}")
            continue

        for (w, h) in resolutions_to_test:
            if w == original_w and h == original_h:
                continue

            scale_factor = (w / original_w + h / original_h) / 2

            # Базовые фичи (для общего режима)
            features_dict_base = {  # выносим базовые фичи в отдельный словарь
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
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Создание датасета"):
            try:
                features, targets = future.result()
                features_all.extend(features)
                all_targets.extend(targets)
            except Exception as e:
                logging.error(f"Ошибка при обработке файла: {e}")

    if features_all:
        df_features = pd.DataFrame(features_all)
        df_features.to_csv(features_csv, index=False)
    else:
        logging.warning("Нет данных для сохранения в features.csv")

    if all_targets:
        df_targets = pd.DataFrame(all_targets)
        df_targets.to_csv(targets_csv, index=False)
    else:
        logging.warning("Нет данных для сохранения в targets.csv")

    return features_csv, targets_csv

def print_console_results(
    file_path: str, results: list, analyze_channels: bool, meta: dict, metric_type: QualityMetrics
):
    ConsoleReporter.print_file_header(file_path)
    if meta['max_val'] < 0.001:
        logging.warning(f"Низкое максимальное значение: {meta['max_val']:.3e}")
    ConsoleReporter.print_quality_table(results, analyze_channels, meta.get('channels'), metric_type)

def create_original_entry(width: int, height: int, channels: Optional[list[str]] = None, analyze_channels: bool = False) -> tuple:
    base_entry = (f"{width}x{height}",)
    if analyze_channels and channels:
        return *base_entry, {c: float('inf') for c in channels}, float('inf'), "Оригинал"
    return *base_entry, float('inf'), "Оригинал"

def _save_intermediate(img_array: np.ndarray, file_path: str, width: int, height: int, interpolation: InterpolationMethods, suffix: str):
    """
    Saves intermediate result as PNG.
    """
    file_path_dir = INTERMEDIATE_DIR
    if not os.path.exists(file_path_dir):
        os.makedirs(file_path_dir, exist_ok=True)

    output_filename = (
        os.path.splitext(os.path.basename(file_path))[0] +
        f"_({interpolation.value}_{width}x{height}){'_' + suffix if suffix else ''}.png"
    )
    output_path = os.path.join(file_path_dir, output_filename)

    arr_for_save = img_array
    if arr_for_save.ndim == 3 and arr_for_save.shape[2] == 1:
        arr_for_save = arr_for_save.squeeze(axis=-1)

    arr_uint8 = np.clip(arr_for_save * 255.0, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(arr_uint8)
    pil_img.save(output_path, format="PNG", optimize=True)

if __name__ == "__main__":
    main()
