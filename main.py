# main.py
import os
import argparse
import logging

from typing import List, Tuple, Optional
from tqdm import tqdm
from cli import parse_arguments, setup_logging, validate_paths
from image_loader import load_image
from image_processing import get_resize_function
from metrics import calculate_psnr, calculate_channel_psnr, compute_resolutions
from reporting import ConsoleReporter, CSVReporter, QualityHelper, generate_csv_filename


def main():
    setup_logging()
    args = parse_arguments()
    files = validate_paths(args.paths)

    if not files:
        logging.error("No valid files found")
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
    """Обработка списка файлов с выводом результатов"""
    with tqdm(total=len(files), desc="Обрабатываю файлы") as pbar:
        for file_path in files:
            results, meta = process_single_file(file_path, args)

            if results:
                print_console_results(file_path, results, args.channels, meta)
                if reporter:
                    reporter.write_results(os.path.basename(file_path), results, args.channels)
            pbar.update(1)


def process_single_file(
    file_path: str,
    args: argparse.Namespace
) -> Tuple[Optional[list], Optional[dict]]:
    """Обработка одного файла"""
    img, max_val, channels = load_image(file_path)
    if img is None:
        logging.error(f"Failed to load image: {file_path}")
        return None, None

    if img.shape[0] < 16 or img.shape[1] < 16:
        logging.warning(f"Image too small for analysis: {file_path}")
        return None, None

    original_h, original_w = img.shape[:2]
    try:
        resize_fn = get_resize_function(args.interpolation)
    except ValueError as e:
        logging.error(f"Error for {file_path}: {e}")
        return None, None

    results = []
    if args.channels:
        results.append(create_original_entry(original_w, original_h, channels))
    else:
        results.append(create_original_entry(original_w, original_h))
    resolutions = compute_resolutions(original_w, original_h)
    with tqdm(total=len(resolutions), desc=f"Анализ {file_path}") as fbar:
        for w, h in resolutions:
            downscaled = resize_fn(img, w, h)
            upscaled = resize_fn(downscaled, original_w, original_h)

            if args.channels:
                channel_psnr = calculate_channel_psnr(img, upscaled, max_val, channels)
                min_psnr = min(channel_psnr.values())
                results.append((
                    f"{w}x{h}",
                    channel_psnr,
                    min_psnr,
                    QualityHelper.get_hint(min_psnr)
                ))
            else:
                psnr = calculate_psnr(img, upscaled, max_val)
                results.append((
                    f"{w}x{h}",
                    psnr,
                    QualityHelper.get_hint(psnr)
                ))
            fbar.update(1)

    return results, {'max_val': max_val, 'channels': channels}


def print_console_results(
    file_path: str,
    results: list,
    analyze_channels: bool,
    meta: dict
):
    """Вывод результатов в консоль"""
    ConsoleReporter.print_file_header(file_path)

    if meta['max_val'] < 0.001:
        logging.warning(f"Low max value: {meta['max_val']:.3e}")

    ConsoleReporter.print_quality_table(
        results,
        analyze_channels,
        meta.get('channels')
    )


def create_original_entry(w: int, h: int, channels: Optional[List[str]] = None) -> tuple:
    """Создает запись для оригинального изображения."""
    if channels:
        return (
            f"{w}x{h}",
            {c: float('inf') for c in channels},
            float('inf'),
            f"{QualityHelper.get_style(float('inf'))}Оригинал"
        )
    else:
        return (
            f"{w}x{h}",
            float('inf'),
            f"{QualityHelper.get_style(float('inf'))}Оригинал"
        )

if __name__ == "__main__":
    main()
