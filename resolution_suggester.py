#!/usr/bin/env python3
import os
import math
import argparse
import csv
from datetime import datetime
import re

import numpy as np
import cv2
from PIL import Image, ImageFile
import pyexr
from colorama import init, Fore, Back, Style

# Инициализация colorama и настройка PIL
init(autoreset=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Разрешает загрузку повреждённых изображений

# Стили для вывода
STYLES = {
    'header': f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}",
    'warning': f"{Style.DIM}{Fore.RED}",
    'original': f"{Fore.CYAN}",
    'good': f"{Fore.LIGHTGREEN_EX}",
    'ok': f"{Fore.GREEN}",
    'medium': f"{Fore.YELLOW}",
    'bad': f"{Fore.RED}",
}

SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png']

def calculate_psnr(original: np.ndarray, processed: np.ndarray, max_val: float) -> float:
    """
    Вычисляет PSNR между двумя изображениями.
    """
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse)
    return psnr

def calculate_channel_psnr(original: np.ndarray, processed: np.ndarray, max_val: float, channels: list) -> dict:
    """
    Вычисляет PSNR для каждого канала отдельно.
    """
    results = {}
    for i, channel in enumerate(channels):
        mse = np.mean((original[..., i] - processed[..., i]) ** 2)
        psnr = 20 * math.log10(max_val) - 10 * math.log10(mse) if mse != 0 else float('inf')
        results[channel] = psnr
    return results

def get_quality_hint(psnr: float, for_csv=False) -> str:
    """
    Возвращает строку с подсказкой о качестве изображения на основе PSNR.
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

def load_image(image_path: str) -> tuple:
    """
    Загружает изображение и возвращает numpy-массив, максимальное значение и каналы.
    """
    try:
        if image_path.lower().endswith('.exr'):
            img = pyexr.read(image_path).astype(np.float32)
            max_val = np.max(np.abs(img))
            max_val = max(max_val, 1e-6)
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L'] # Handle grayscale EXR
            return img, max_val, channels
        elif image_path.lower().endswith(('.png', '.tga')):
            img = np.array(Image.open(image_path)).astype(np.float32)
            max_val = 255.0
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L'] # Handle grayscale PNG/TGA
            return img, max_val, channels
        else:
            print(f"Неподдерживаемый формат файла: {image_path}")
            return None, None, None
    except Exception as e:
        print(f"Ошибка чтения {image_path}: {e}")
        return None, None, None

def compute_resolutions(original_w: int, original_h: int, min_size: int = 16) -> list:
    """
    Вычисляет список разрешений для масштабирования.
    """
    resolutions = []
    current_w, current_h = original_w, original_h
    while current_w >= min_size * 2 and current_h >= min_size * 2:
        current_w //= 2
        current_h //= 2
        resolutions.append((current_w, current_h))
    return resolutions

def process_image(image_path: str, analyze_channels: bool):
    """
    Обрабатывает изображение и вычисляет PSNR при различных масштабированиях.
    Возвращает результаты, максимальное значение и каналы.
    """
    img, max_val, channels = load_image(image_path)
    if img is None:
        return [], None, None

    original_h, original_w = img.shape[:2]
    orig_res_str = f"{original_w}x{original_h}"

    results = []
    processed_channels = channels if analyze_channels and channels else [] # Determine channels to process for consistent logic

    if processed_channels: # Channel analysis is enabled and channels are available
        channel_psnr_original = {c: float('inf') for c in processed_channels}
        results.append((
            orig_res_str,
            channel_psnr_original,
            float('inf'),
            f'{STYLES["original"]}оригинал{Style.RESET_ALL}'
        ))
    else: # No channel analysis or no channels available
        results.append((
            orig_res_str,
            float('inf'),
            f'{STYLES["original"]}оригинал{Style.RESET_ALL}'
        ))

    # Вычисляем набор разрешений для масштабирования
    resolutions = compute_resolutions(original_w, original_h)

    # Обрабатываем каждое разрешение
    for target_w, target_h in resolutions:
        # Масштабирование вниз и вверх
        downscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        if processed_channels: # Channel analysis
            channel_psnr = calculate_channel_psnr(img, upscaled, max_val, processed_channels)
            min_psnr = min(channel_psnr.values())
            results.append((
                f"{target_w}x{target_h}",
                channel_psnr,
                min_psnr,
                get_quality_hint(min_psnr)
            ))
        else: # No channel analysis
            psnr = calculate_psnr(img, upscaled, max_val)
            results.append((
                f"{target_w}x{target_h}",
                psnr,
                get_quality_hint(psnr)
            ))

    return results, max_val if image_path.lower().endswith('.exr') else None, processed_channels # Return processed channels

def main():
    parser = argparse.ArgumentParser(description='Анализ потерь качества текстур')
    parser.add_argument('paths', nargs='+', help='Пути к файлам текстур или директориям')
    parser.add_argument('--channels', '-c', action='store_true', help='Анализировать отдельные цветовые каналы')
    parser.add_argument('--csv-output', action='store_true', help='Выводить метрики в CSV файл') # Добавлен аргумент для CSV вывода
    args = parser.parse_args()

    separator = f"{Style.DIM}|{Style.NORMAL}"

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

    if args.csv_output: # Если указан флаг --csv-output, начинаем работу с CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}.csv"
        csv_filepath = os.path.join(os.getcwd(), csv_filename) # CSV файл в текущей директории

        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_header_written = False # Флаг для записи заголовка только один раз
            general_csv_header = ['File', 'Resolution', 'R(L) PSNR', 'G PSNR', 'B PSNR', 'A PSNR', 'Min PSNR', 'Quality Hint (min)']


            for path in files_to_process:
                print(f"\n\n{STYLES['header']}--- {os.path.basename(path):^50} ---{Style.RESET_ALL}")

                results, max_val, channels = process_image(path, args.channels)
                if not results:
                    continue

                if max_val is not None and max_val < 0.001:
                    print(f"{STYLES['warning']}АХТУНГ: Максимальное значение {max_val:.3e}!{Style.RESET_ALL}")

                # Общий заголовок CSV, пишем только один раз
                if not csv_header_written:
                    csv_writer.writerow(general_csv_header)
                    csv_header_written = True

                if args.channels and channels: # CSV output with channels
                    channel_header_labels = channels # Use directly detected channels for console output
                    header = f"\n{Style.BRIGHT}{'Разрешение':<12} | {' | '.join([c.center(9) for c in [f'{channels[0]}(L)' if channels[0] in ('R','L') else 'R', 'G', 'B', 'A'][:len(channels)]])} | {'Min':^9} | {'Качество (min)':<36}{Style.RESET_ALL}" # Adjusted header for console
                    print(header)
                    header_line = f"{'-'*12}-+-" + "-+-".join(["-"*9]*len([f'{channels[0]}(L)' if channels[0] in ('R','L') else 'R', 'G', 'B', 'A'][:len(channels)])) + f"-+-{'-'*9}-+-{'-'*32}" # Adjusted line for console
                    print(header_line)

                    for res, ch_psnr, min_psnr, hint in results:
                        ch_values_console = ' | '.join([f"{ch_psnr.get(c, 0):9.2f}" for c in channels]) # Adjusted values for console
                        print(f"{res:<12} | {ch_values_console} | {min_psnr:9.2f} | {hint:<36}") # Adjusted output for console

                        # CSV row output for channels, always outputting R, G, B, A, even if not all present, for consistent columns
                        csv_row_values = [os.path.basename(path), res]
                        csv_row_values.append(f"{ch_psnr.get('R', ch_psnr.get('L', float('inf'))):.2f}") # R or L channel, default to inf if not found
                        csv_row_values.append(f"{ch_psnr.get('G', float('inf')):.2f}") # G channel, default to inf if not found
                        csv_row_values.append(f"{ch_psnr.get('B', float('inf')):.2f}") # B channel, default to inf if not found
                        csv_row_values.append(f"{ch_psnr.get('A', float('inf')):.2f}") # A channel, default to inf if not found
                        csv_row_values.append(f'{min_psnr:.2f}')
                        csv_row_values.append(get_quality_hint(min_psnr, for_csv=True))
                        csv_writer.writerow(csv_row_values)
                else: # CSV output without channels
                    print(f"\n{Style.BRIGHT}{'Разрешение':<12} | {'PSNR (dB)':^10} | {'Качество':<36}{Style.RESET_ALL}")
                    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")
                    for res, psnr, hint in results:
                        print(f"{res:<12} {separator} {psnr:^10.2f} {separator} {hint:<36}")
                        # CSV row output without channels
                        csv_writer.writerow([os.path.basename(path), res, f'{psnr:.2f}', get_quality_hint(psnr, for_csv=True), '', '', '', '']) # Pad empty columns for channel PSNRs

                print() # Пустая строка после каждого файла

        print(f"\nМетрики сохранены в: {csv_filepath}") # Сообщение о сохранении CSV

    else: # Стандартный вывод в консоль, если --csv-output не указан
        for path in files_to_process:
            print(f"\n\n{STYLES['header']}--- {os.path.basename(path):^50} ---{Style.RESET_ALL}")

            results, max_val, channels = process_image(path, args.channels)
            if not results:
                continue

            if max_val is not None and max_val < 0.001:
                print(f"{STYLES['warning']}АХТУНГ: Максимальное значение {max_val:.3e}!{Style.RESET_ALL}")

            if args.channels: # Console output with channels
                if channels is not None:
                    channel_header_labels = channels if len(channels) > 1 else channels
                    header = f"\n{Style.BRIGHT}{'Разрешение':<12} | {' | '.join([c.center(6) for c in channel_header_labels])} | {'Min':^6} | {'Качество (min)':<36}{Style.RESET_ALL}"
                    print(header)
                    header_line = f"{'-'*12}-+-" + "-+-".join(["-"*6]*len(channel_header_labels)) + f"-+-{'-'*6}-+-{'-'*32}"
                    print(header_line)
                    for res, ch_psnr, min_psnr, hint in results:
                        ch_values = ' | '.join([f"{ch_psnr.get(c, 0):6.2f}" for c in channel_header_labels])
                        print(f"{res:<12} | {ch_values} | {min_psnr:6.2f} | {hint:<36}")
            else: # Console output without channels
                print(f"\n{Style.BRIGHT}{'Разрешение':<12} | {'PSNR (dB)':^10} | {'Качество':<36}{Style.RESET_ALL}")
                print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")
                for res, psnr, hint in results:
                    print(f"{res:<12} {separator} {psnr:^10.2f} {separator} {hint:<36}")
            print() # Пустая строка после каждого файла


if __name__ == "__main__":
    main()
