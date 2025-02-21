# main.py
import os
import argparse
import logging
import concurrent.futures

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional
from cli import parse_arguments, setup_logging, validate_paths
from image_loader import load_image
from image_processing import get_resize_function
from metrics import (
    compute_resolutions,
    calculate_metrics
)
from reporting import ConsoleReporter, CSVReporter, QualityHelper, generate_csv_filename
from config import SAVE_INTERMEDIATE_DIR, ML_DATA_DIR, InterpolationMethod, QualityMetric
from ml_predictor import QuickPredictor, extract_features_of_original_img

def main():
    setup_logging()
    args = parse_arguments()
    files = validate_paths(args.paths)
    if not files:
        logging.error("Не найдено ни одного валидного файла или директории. Завершение работы.")
        return

    if args.generate_dataset:
        features_path, targets_path = generate_dataset(files)
        logging.info(f"Датасет сгенерирован: features={features_path}, targets={targets_path}")
        if args.train_ml:
            predictor = QuickPredictor()
            predictor.train(features_path, targets_path)
            logging.info("Модель обучена!")
        return  # завершаем работу после создания датасета

    if args.csv_output:
        csv_path = generate_csv_filename(args.metric, args.interpolation)
        with CSVReporter(csv_path, args.metric) as reporter:
            reporter.write_header(args.channels)
            process_files(files, args, reporter)
        print(f"\nМетрики сохранены в: {csv_path}")
    else:
        process_files(files, args)

def process_files(files: list[str], args: argparse.Namespace, reporter: Optional[CSVReporter] = None):
    if args.no_parallel:
        for file_path in files:
            try:
                results, meta = process_single_file(file_path, args)
            except Exception as e:
                logging.error(f"Ошибка обработки {file_path}: {e}")
                continue
            if results:
                print_console_results(file_path, results, args.channels, meta, args.metric)
                if reporter:
                    reporter.write_results(os.path.basename(file_path), results, args.channels)
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
                    print_console_results(file_path, results, args.channels, meta, args.metric)
                    if reporter:
                        reporter.write_results(os.path.basename(file_path), results, args.channels)

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
    if use_prediction:
        ml_predictor = QuickPredictor()
        if ml_predictor.load():
            features_original = extract_features_of_original_img(img_original)
        else:
            logging.warning("ML-модель не найдена, будем вычислять реальные метрики.")
            use_prediction = False

    try:
        resize_fn = get_resize_function(args.interpolation)
    except ValueError as e:
        logging.error(f"Ошибка при выборе функции интерполяции для {file_path}: {e}")
        return None, None

    for (w, h) in resolutions:
        if w == width and h == height:
            continue

        # Предсказываем или вычисляем метрики?
        if use_prediction:
            features_for_ml = {
                **features_original,
                'scale_factor': (w / width + h / height) / 2,
                'method': args.interpolation # TODO: Вынести бы — в цикле это не меняется
            }

            prediction = ml_predictor.predict(features_for_ml)
        else:
            img_downscaled = resize_fn(img_original, w, h)
            if args.save_intermediate:
                _save_intermediate(img_downscaled, file_path, w, h)
            img_upscaled = resize_fn(img_downscaled, width, height)

        if args.channels:
            if use_prediction:
                # TODO: Поменять на предсказание отдельных каналов
                channels_metrics = {c: prediction[args.metric] for c in channels}
            else:
                channels_metrics = calculate_metrics(args.metric, img_original, img_upscaled, max_val, channels)

            min_metric = min(channels_metrics.values())
            results.append(
                (
                    f"{w}x{h}",
                    channels_metrics,
                    min_metric,
                    QualityHelper.get_hint(min_metric, args.metric),
                )
            )
        else:
            if use_prediction:
                metric_value = prediction[args.metric]
            else:
                metric_value = calculate_metrics(args.metric, img_original, img_upscaled, max_val)

            results.append(
                (
                    f"{w}x{h}",
                    metric_value,
                    QualityHelper.get_hint(metric_value, args.metric),
                )
            )

    return results, {'max_val': max_val, 'channels': channels}

def generate_dataset(files: list[str]) -> tuple[str, str]:
    features_all = []
    all_targets = []

    # Для упрощения сохраним в CSV (можно parquet)
    features_csv = os.path.join(ML_DATA_DIR, 'features.csv')
    targets_csv  = os.path.join(ML_DATA_DIR, 'targets.csv')

    if not os.path.exists(ML_DATA_DIR):
        os.makedirs(ML_DATA_DIR, exist_ok=True)

    methods_to_test = [
        InterpolationMethod.BILINEAR,
        InterpolationMethod.BICUBIC,
        InterpolationMethod.MITCHELL
    ]

    with tqdm(total=len(files), desc=f"Создание датасета", leave=False) as progressbar_files:
        for file_path in files:
            result = load_image(file_path)
            if result.error or result.data is None:
                logging.warning(f"Пропуск {file_path}, т.к. не удалось загрузить.")
                continue

            img_original = result.data
            max_val = result.max_value
            original_h, original_w = img_original.shape[:2]

            resolutions_to_test = compute_resolutions(original_w, original_h, divider=4)
            if not resolutions_to_test:
                continue

            with tqdm(total=len(methods_to_test), desc=f"Файл: {file_path}", leave=False) as progressbar_methods:
                for method in methods_to_test:
                    resize_fn = get_resize_function(method)

                    with tqdm(total=len(resolutions_to_test), desc=f"Интерполяция: {method.value}", leave=False) as progressbar_res:
                        for (w, h) in resolutions_to_test:
                            scale_factor = (w / original_w + h / original_h) / 2
                            features_original = extract_features_of_original_img(img_original)

                            features_dict = {
                                **features_original,
                                'scale_factor': scale_factor,
                                'method': method.value,
                            }

                            if w == original_w and h == original_h:
                                continue

                            img_downscaled = resize_fn(img_original, w, h)
                            img_upscaled = resize_fn(img_downscaled, original_w, original_h)

                            features_all.append(features_dict)
                            all_targets.append({
                                'psnr': calculate_metrics(QualityMetric.PSNR, img_original, img_upscaled, max_val),
                                'ssim': calculate_metrics(QualityMetric.SSIM, img_original, img_upscaled, max_val),
                                'ms_ssim': calculate_metrics(QualityMetric.MS_SSIM, img_original, img_upscaled, max_val)
                            })
                            progressbar_res.update(1)
                    progressbar_methods.update(1)
            progressbar_files.update(1)

    df_features = pd.DataFrame(features_all)
    df_targets = pd.DataFrame(all_targets)
    df_features.to_csv(features_csv, index=False)
    df_targets.to_csv(targets_csv, index=False)

    return features_csv, targets_csv
def print_console_results(
    file_path: str, results: list, analyze_channels: bool, meta: dict, metric: str
):
    ConsoleReporter.print_file_header(file_path)

    if meta['max_val'] < 0.001:
        logging.warning(f"Низкое максимальное значение: {meta['max_val']:.3e}")
    ConsoleReporter.print_quality_table(
        results, analyze_channels, meta.get('channels'), metric
    )

def create_original_entry(
    width: int, height: int, channels: Optional[list[str]] = None, analyze_channels: bool = False
) -> tuple:
    base_entry = (f"{width}x{height}",)
    if analyze_channels and channels:
        return *base_entry, {c: float('inf') for c in channels}, float('inf'), "Оригинал"
    return *base_entry, float('inf'), "Оригинал"

def _save_intermediate(img_array: np.ndarray, file_path: str, width: int, height: int):
    """
    Saves intermediate result as PNG.
    """
    file_path_dir = os.path.join(os.path.dirname(file_path), SAVE_INTERMEDIATE_DIR)
    if not os.path.exists(file_path_dir):
        os.makedirs(file_path_dir, exist_ok=True)

    output_filename = (
        os.path.splitext(os.path.basename(file_path))[0] + f"_{width}x{height}.png"
    )
    output_path = os.path.join(file_path_dir, output_filename)

    arr_for_save = img_array
    if arr_for_save.ndim == 3 and arr_for_save.shape[2] == 1:
        arr_for_save = arr_for_save.squeeze(axis=-1)

    arr_uint8 = np.clip(arr_for_save * 255.0, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(arr_uint8)
    pil_img.save(output_path, format="PNG")

if __name__ == "__main__":
    main()