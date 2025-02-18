# main.py
import os
import argparse
import logging
import concurrent.futures

from typing import List, Tuple, Optional
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
        # Сопоставляем Future -> file_path, чтобы знать, какой файл обрабатывался
        future_to_file = {
            executor.submit(process_single_file, file_path, args): file_path
            for file_path in files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                results, meta = future.result()
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
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
        return None, None

    img = result.data
    max_val = result.max_value
    channels = result.channels

    height, width = img.shape[:2]

    if height < args.min_size or width < args.min_size:
        return None, None

    try:
        resize_fn = get_resize_function(args.interpolation)
    except ValueError:
        return None, None

    results = []
    if args.channels:
        results.append(create_original_entry(width, height, channels))
    else:
        results.append(create_original_entry(width, height))

    resolutions = compute_resolutions(width, height, args.min_size)

    for (w, h) in resolutions:
        downscaled = resize_fn(img, w, h)
        upscaled = resize_fn(downscaled, width, height)

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

def print_console_results(
    file_path: str,
    results: list,
    analyze_channels: bool,
    meta: dict
):
    ConsoleReporter.print_file_header(file_path)

    if meta['max_val'] < 0.001:
        logging.warning(f"Low max value: {meta['max_val']:.3e}")

    ConsoleReporter.print_quality_table(
        results,
        analyze_channels,
        meta.get('channels')
    )

def create_original_entry(width: int, height: int, channels: Optional[List[str]] = None) -> tuple:
    if channels:
        return (
            f"{width}x{height}",
            {c: float('inf') for c in channels},
            float('inf'),
            f"{QualityHelper.get_style(float('inf'))}Оригинал"
        )
    else:
        return (
            f"{width}x{height}",
            float('inf'),
            f"{QualityHelper.get_style(float('inf'))}Оригинал"
        )

if __name__ == "__main__":
    main()