#!/usr/bin/env python3
import os
import math
import argparse
import csv
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageFile
import pyexr
from colorama import init, Fore, Back, Style

# Инициализация colorama и настройка PIL
init(autoreset=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Разрешает загрузку повреждённых изображений

# Constants
SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png']
STYLES = {
    'header': f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}",
    'warning': f"{Style.DIM}{Fore.YELLOW}",
    'original': f"{Fore.CYAN}",
    'good': f"{Fore.LIGHTGREEN_EX}",
    'ok': f"{Fore.GREEN}",
    'medium': f"{Fore.YELLOW}",
    'bad': f"{Fore.RED}",
}

def load_image(file_path: str) -> tuple:
    """
    Loads an image and returns the numpy array, max value, and channels.

    Args:
    file_path (str): Path to the image file.

    Returns:
    tuple: (image_array, max_value, channels)
    """
    try:
        if file_path.lower().endswith('.exr'):
            img = pyexr.read(file_path).astype(np.float32)
            max_val = np.max(np.abs(img))
            max_val = max(max_val, 1e-6)
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L']
            return img, max_val, channels
        elif file_path.lower().endswith(('.png', '.tga')):
            img = np.array(Image.open(file_path)).astype(np.float32)
            max_val = 255.0
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L']
            return img, max_val, channels
        else:
            print(f"Unsupported file format: {file_path}")
            return None, None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None

def calculate_psnr(original: np.ndarray, processed: np.ndarray, max_val: float) -> float:
    """
    Calculates PSNR between two images.

    Args:
    original (np.ndarray): Original image array.
    processed (np.ndarray): Processed image array.
    max_val (float): Maximum value in the image.

    Returns:
    float: PSNR value.
    """
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse)
    return psnr

def calculate_channel_psnr(original: np.ndarray, processed: np.ndarray, max_val: float, channels: list) -> dict:
    """
    Calculates PSNR for each channel separately.

    Args:
    original (np.ndarray): Original image array.
    processed (np.ndarray): Processed image array.
    max_val (float): Maximum value in the image.
    channels (list): List of channel names.

    Returns:
    dict: Channel PSNR values.
    """
    results = {}
    for i, channel in enumerate(channels):
        mse = np.mean((original[..., i] - processed[..., i]) ** 2)
        psnr = 20 * math.log10(max_val) - 10 * math.log10(mse) if mse != 0 else float('inf')
        results[channel] = psnr
    return results

def get_quality_hint(psnr: float, for_csv: bool = False) -> str:
    """
    Returns a quality hint string based on the PSNR value.

    Args:
    psnr (float): PSNR value.
    for_csv (bool, optional): Flag for CSV output. Defaults to False.

    Returns:
    str: Quality hint string.
    """
    if psnr >= 50:
        hint = "практически идентичные изображения"
        return hint if for_csv else f"{STYLES['good']}{hint}"
    elif psnr >= 40:
        hint = "очень хорошее качество"
        return hint if for_csv else f"{STYLES['ok']}{hint}"
    elif psnr >= 30:
        hint = "приемлемое качество"
        return hint if for_csv else f"{STYLES['medium']}{hint}"
    else:
        hint = "заметные потери"
        return hint if for_csv else f"{STYLES['bad']}{hint}"

### Image Analysis Functions ###

def compute_resolutions(original_w: int, original_h: int, min_size: int = 16) -> list:
    """
    Computes a list of resolutions for scaling.

    Args:
    original_w (int): Original image width.
    original_h (int): Original image height.
    min_size (int, optional): Minimum size. Defaults to 16.

    Returns:
    list: List of resolutions.
    """
    resolutions = []
    current_w, current_h = original_w, original_h
    while current_w >= min_size * 2 and current_h >= min_size * 2:
        current_w //= 2
        current_h //= 2
        resolutions.append((current_w, current_h))
    return resolutions

def process_image(file_path: str, analyze_channels: bool) -> tuple:
    """
    Processes an image and calculates PSNR at various scales.

    Args:
    file_path (str): Path to the image file.
    analyze_channels (bool): Flag to analyze channels separately.

    Returns:
    tuple: (results, max_value, channels)
    """
    img, max_val, channels = load_image(file_path)
    if img is None:
        return [], None, None

    original_h, original_w = img.shape[:2]
    orig_res_str = f"{original_w}x{original_h}"

    results = []
    if analyze_channels and channels:
        channel_psnr_original = {c: float('inf') for c in channels}
        results.append((orig_res_str, channel_psnr_original, float('inf'), f"{STYLES['original']}Оригинал{Style.RESET_ALL}"))
    else:
        results.append((orig_res_str, float('inf'), f"{STYLES['original']}Оригинал{Style.RESET_ALL}"))

    resolutions = compute_resolutions(original_w, original_h)

    for target_w, target_h in resolutions:
        downscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        if analyze_channels and channels:
            channel_psnr = calculate_channel_psnr(img, upscaled, max_val, channels)
            min_psnr = min(channel_psnr.values())
            results.append((f"{target_w}x{target_h}", channel_psnr, min_psnr, get_quality_hint(min_psnr)))
        else:
            psnr = calculate_psnr(img, upscaled, max_val)
            results.append((f"{target_w}x{target_h}", psnr, get_quality_hint(psnr)))

    return results, max_val if file_path.lower().endswith('.exr') else None, channels


def main():
    parser = argparse.ArgumentParser(description='Анализ потерь качества текстур')
    parser.add_argument('paths', nargs='+', help='Пути к файлам текстур или директориям')
    parser.add_argument('--channels', '-c', action='store_true', help='Анализировать отдельные цветовые каналы')
    parser.add_argument('--csv-output', action='store_true', help='Выводить метрики в CSV файл')
    args = parser.parse_args()

    files_to_process = []
    for path in args.paths:
        if os.path.isfile(path):
            files_to_process.append(path)
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
                        files_to_process.append(os.path.join(dirpath, filename))
        else:
            print(f"Путь не является файлом или директорией, или не существует: {path}")

    if not files_to_process:
        print("Не найдено поддерживаемых файлов для обработки.")
        return

    if args.csv_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}.csv"
        csv_filepath = os.path.join(os.getcwd(), csv_filename)

        with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)

            general_csv_header = ["Файл", "Разрешение", "R(L) PSNR", "G PSNR", "B PSNR", "A PSNR", "Min PSNR", "Качество (min)"]
            csv_writer.writerow(general_csv_header)

            for file_path in files_to_process:
                print(f"\n\n{STYLES['header']}--- {os.path.basename(file_path):^50} ---{Style.RESET_ALL}")

                results, max_val, channels = process_image(file_path, args.channels)
                if not results:
                    continue

                if max_val is not None and max_val < 0.001:
                    print(f"{STYLES['warning']}Warning: Max value {max_val:.3e}!{Style.RESET_ALL}")

                if args.channels and channels:
                    channel_header_labels = channels
                    print (f"\n{Style.BRIGHT}{'Разрешение':<12} | {' | '.join([c.center(9) for c in ['R(L)', 'G', 'B', 'A'][:len(channels)]])} | {'Min':^9} | {'Качество (min)':<36}{Style.RESET_ALL})")
                    print (f"{'-'*12}-+-" + "-+-".join(["-"*9]*len(['R(L)', 'G', 'B', 'A'][:len(channels)])) + f"-+-{'-'*9}-+-{'-'*32}")

                    first_row = True
                    for res, ch_psnr, min_psnr, hint in results:
                        if first_row:
                            csv_writer.writerow([os.path.basename(file_path), res, "", "", "", "", "", ""])
                            first_row = False

                        else:
                            csv_row_values = ["", res]
                            csv_row_values.append(f"{ch_psnr.get('R', ch_psnr.get('L', float('inf'))):.2f}")
                            csv_row_values.append(f"{ch_psnr.get('G', float('inf')):.2f}")
                            csv_row_values.append(f"{ch_psnr.get('B', float('inf')):.2f}")
                            csv_row_values.append(f"{ch_psnr.get('A', float('inf')):.2f}")
                            csv_row_values.append(f'{min_psnr:.2f}')
                            csv_row_values.append(get_quality_hint(min_psnr, for_csv=True))
                            csv_writer.writerow(csv_row_values)

                        ch_values_console = ' | '.join([f"{ch_psnr.get(c, 0):9.2f}" for c in ['R', 'G', 'B', 'A'][:len(channels)]])
                        print(f"{res:<12} | {ch_values_console} | {min_psnr:9.2f} | {hint:<36}")
                else:
                    print(f"\n{Style.BRIGHT}{'Resolution':<12} | {'PSNR (dB)':^10} | {'Quality':<36}{Style.RESET_ALL}")
                    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")

                    first_row = True
                    for res, psnr, hint in results:
                        if first_row:
                            csv_writer.writerow([os.path.basename(file_path), res, "", "", "", "", "", ""])
                            first_row = False
                        else:
                            csv_writer.writerow(["", res, f'{psnr:.2f}', "", "", "", "", get_quality_hint(psnr, for_csv=True)])

                        print(f"{res:<12} {Style.DIM}|{Style.NORMAL} {psnr:^10.2f} {Style.DIM}|{Style.NORMAL} {hint:<36}")

                print()


        print(f"\nМетрики сохранены в: {csv_filepath}")

    else:
        for file_path in files_to_process:
            print(f"\n\n{STYLES['header']}--- {os.path.basename(file_path):^50} ---{Style.RESET_ALL}")

            results, max_val, channels = process_image(file_path, args.channels)
            if not results:
                continue

            if max_val is not None and max_val < 0.001:
                print(f"{STYLES['warning']}Warning: Max value {max_val:.3e}!{Style.RESET_ALL}")

            if args.channels and channels:
                channel_header_labels = channels
                header = f"\n{Style.BRIGHT}{'Разрешение':<12} | {' | '.join([c.center(6) for c in channel_header_labels])} | {'Min':^6} | {'Качество (min)':<36}{Style.RESET_ALL}"
                print(header)
                header_line = f"{'-'*12}-+-" + "-+-".join(["-"*6]*len(channel_header_labels)) + f"-+-{'-'*6}-+-{'-'*32}"
                print(header_line)

                for res, ch_psnr, min_psnr, hint in results:
                    ch_values = ' | '.join([f"{ch_psnr.get(c, 0):6.2f}" for c in channel_header_labels])
                    print(f"{res:<12} | {ch_values} | {min_psnr:6.2f} | {hint:<36}")
            else:
                print(f"\n{Style.BRIGHT}{'Разрешение':<12} | {'PSNR (dB)':^10} | {'Качество':<36}{Style.RESET_ALL}")
                print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")

                for res, psnr, hint in results:
                    print(f"{res:<12} {Style.DIM}|{Style.NORMAL} {psnr:^10.2f} {Style.DIM}|{Style.NORMAL} {hint:<36}")

            print()

if __name__ == "__main__":
    main()
