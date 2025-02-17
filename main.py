# main.py
import os
import numpy as np
from typing import List, Tuple, Optional, Dict
import datetime
import argparse # Добавлено: argparse

from config import SUPPORTED_EXTENSIONS
from cli import parse_arguments, setup_logging, validate_paths
from image_loader import load_image
from image_processing import get_resize_function
from metrics import calculate_psnr, calculate_channel_psnr, compute_resolutions
from reporting import ConsoleReporter, CSVReporter, QualityHelper

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
    for file_path in files:
        results, meta = process_single_file(file_path, args)

        if results:
            print_console_results(file_path, results, args.channels, meta)
            if reporter:
                reporter.write_results(os.path.basename(file_path), results, args.channels)

def process_single_file(
    file_path: str,
    args: argparse.Namespace
) -> Tuple[Optional[list], Optional[dict]]:
    """Обработка одного файла"""
    img, max_val, channels = load_image(file_path)
    if img is None:
        return None, None

    original_h, original_w = img.shape[:2]
    resize_fn = get_resize_function(args.interpolation)

    results = []
    if args.channels:
        results.append(create_original_channel_entry(original_w, original_h, channels))
    else:
        results.append(create_original_entry(original_w, original_h))

    for w, h in compute_resolutions(original_w, original_h):
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

    return results, {'max_val': max_val, 'channels': channels}

def generate_csv_filename() -> str:
    """Генерация имени CSV файла с временной меткой"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"texture_analysis_{timestamp}.csv"

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

def create_original_channel_entry(w: int, h: int, channels: list[str]) -> Tuple:
    return (
        f"{w}x{h}",
        {c: float('inf') for c in channels},
        float('inf'),
        f"{QualityHelper.get_style(float('inf'))}Оригинал"
    )

def create_original_entry(w: int, h: int) -> Tuple:
    return (
        f"{w}x{h}",
        float('inf'),
        f"{QualityHelper.get_style(float('inf'))}Оригинал"
    )

if __name__ == "__main__":
    main()
