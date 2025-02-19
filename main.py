import os
import argparse
import logging
import concurrent.futures

from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
from cli import parse_arguments, setup_logging, validate_paths
from image_loader import load_image
from image_processing import get_resize_function
from metrics import (
    calculate_psnr,
    calculate_channel_psnr,
    compute_resolutions,
    calculate_ssim_gauss,
    calculate_channel_ssim_gauss
)
from reporting import ConsoleReporter, CSVReporter, QualityHelper, generate_csv_filename
from config import SAVE_INTERMEDIATE_DIR

def main():
    setup_logging()
    args = parse_arguments()
    files = validate_paths(args.paths)

    if not files:
        logging.error("Не найдено ни одного валидного файла")
        return

    if args.csv_output:
        csv_path = generate_csv_filename()
        with CSVReporter(csv_path) as reporter:
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
                print_console_results(file_path, results, args.channels, meta)
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
                    print_console_results(file_path, results, args.channels, meta)
                    if reporter:
                        reporter.write_results(os.path.basename(file_path), results, args.channels)

def process_single_file(
    file_path: str,
    args: argparse.Namespace
) -> Tuple[Optional[list], Optional[dict]]:
    result = load_image(file_path)
    if result.error or result.data is None:
        logging.error(f"Ошибка загрузки изображения {file_path}: {result.error}") # More informative log
        return None, None

    img = result.data
    max_val = result.max_value
    channels = result.channels

    height, width = img.shape[:2]

    if height < args.min_size or width < args.min_size:
        logging.info(f"Пропуск {file_path} (размер {width}x{height}) - меньше, чем min_size = {args.min_size}")
        return None, None

    try:
        resize_fn = get_resize_function(args.interpolation)
    except ValueError as e:
        logging.error(f"Ошибка при выборе функции интерполяции для {file_path}: {e}") # More informative log
        return None, None

    results = []
    results.append(create_original_entry(width, height, channels, args.channels)) # Pass analyze_channels flag
    resolutions = compute_resolutions(width, height, args.min_size)
    use_psnr = (args.metric == 'psnr')

    for (w, h) in resolutions:
        if w == width and h == height:
            continue

        downscaled = resize_fn(img, w, h)
        if args.save_intermediate:
            _save_intermediate(downscaled, file_path, w, h)

        upscaled = resize_fn(downscaled, width, height)

        if args.channels:
            if use_psnr:
                channel_metrics = calculate_channel_psnr(img, upscaled, max_val, channels)
            else:
                # Считаем SSIM для каждого канала
                channel_metrics = calculate_channel_ssim_gauss(img, upscaled, max_val, channels)

            min_metric = min(channel_metrics.values())
            hint = QualityHelper.get_hint(min_metric)
            results.append((
                f"{w}x{h}",
                channel_metrics,
                min_metric,
                hint
            ))
        else:
            if use_psnr:
                metric_value = calculate_psnr(img, upscaled, max_val)
            else:
                metric_value = calculate_ssim_gauss(img, upscaled, max_val)

            hint = QualityHelper.get_hint(metric_value)
            results.append((
                f"{w}x{h}",
                metric_value,
                hint
            ))

    return results, {'max_val': max_val, 'channels': channels}

def print_console_results(
    file_path: str,
    results: list,
    analyze_channels: bool,
    meta: dict
):
    ConsoleReporter.print_file_header(file_path)

    if meta['max_val'] < 0.001:
        logging.warning(f"Низкое максимальное значение: {meta['max_val']:.3e}")

    ConsoleReporter.print_quality_table(
        results,
        analyze_channels,
        meta.get('channels')
    )

def create_original_entry(width: int, height: int, channels: Optional[List[str]] = None, analyze_channels: bool = False) -> tuple:
    base_entry = (f"{width}x{height}",)
    if analyze_channels and channels:
        return (*base_entry, {c: float('inf') for c in channels}, float('inf'), "Оригинал")
    return (*base_entry, float('inf'), "Оригинал")


def _save_intermediate(img_array: np.ndarray, file_path: str, width: int, height: int):
    """
    Сохраняет промежуточный результат в PNG.
    Если монохром (H,W,1) - избавляемся от оси каналов для записи.
    Если (H,W) - тоже прям пишем как есть (Pillow поймёт такую матрицу как grayscale).
    Если (H,W,3) или (H,W,4) - запишем как RGB / RGBA.
    """
    # Убедимся, что папка существует

    file_path_dir = os.path.dirname(file_path) + os.sep + SAVE_INTERMEDIATE_DIR
    if not os.path.exists(file_path_dir):
        os.makedirs(file_path_dir, exist_ok=True)

    output_path = os.path.join(file_path_dir, os.path.basename(file_path).split('.')[0] + f"_{width}x{height}.png")
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    arr_for_save = img_array
    if arr_for_save.ndim == 3 and arr_for_save.shape[2] == 1:
        # (H, W, 1) -> (H, W)
        arr_for_save = arr_for_save.squeeze(axis=-1)

    # Pillow требует, чтобы данные были в [0..255], uint8 - конвертируем
    arr_uint8 = np.clip(arr_for_save * 255.0, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(arr_uint8)
    pil_img.save(output_path, format="PNG")

if __name__ == "__main__":
    main()