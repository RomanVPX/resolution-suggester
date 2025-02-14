#!/usr/bin/env python3
import os
import math
import argparse
import csv
from datetime import datetime
import logging
from typing import Union, Tuple, List, Optional, Dict

import numpy as np
import cv2
import pyexr
from PIL import Image, ImageFile
from colorama import init, Fore, Back, Style

# Initialize colorama and PIL settings
init(autoreset=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants and Configurations
SUPPORTED_EXTENSIONS = ['.exr', '.tga', '.png']
QUALITY_HINTS = {
    50: "практически идентичные изображения",
    40: "очень хорошее качество",
    30: "приемлемое качество",
    0: "заметные потери",
}
PSNR_QUALITY_THRESHOLDS = sorted(QUALITY_HINTS.keys(), reverse=True)
STYLES = {
    'header': f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}",
    'warning': f"{Style.DIM}{Fore.YELLOW}",
    'original': f"{Fore.CYAN}",
    'good': f"{Fore.LIGHTGREEN_EX}",
    'ok': f"{Fore.GREEN}",
    'medium': f"{Fore.YELLOW}",
    'bad': f"{Fore.RED}",
}
CSV_SEPARATOR = ";"  # Explicitly define CSV separator
INTERPOLATION_METHODS = {
    'bilinear': 'cv2.INTER_LINEAR',
    'bicubic': 'cv2.INTER_CUBIC',
}
DEFAULT_INTERPOLATION = 'INTER_CUBIC'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_image(file_path: str) -> Union[Tuple[np.ndarray, float, List[str]], Tuple[None, None, None]]:
    """
    Загружает изображение из файла и возвращает массив numpy, максимальное значение и каналы.

    Поддерживает форматы EXR, PNG и TGA.

    Аргументы:
        file_path (str): Путь к файлу изображения.

    Возвращает:
        tuple: (image_array, max_value, channels) или (None, None, None) в случае ошибки.
    """
    file_extension = file_path.lower().split('.')[-1]

    try:
        if file_extension == 'exr':
            img = pyexr.read(file_path).astype(np.float32)  # Automatically detect EXR version
            max_val = np.max(np.abs(img))
            max_val = max(max_val, 1e-6)  # Prevent division by zero
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 and img.shape[2] > 1 else ['L']  # Handle grayscale EXR correctly
        elif file_extension in ('png', 'tga'):
            img = np.array(Image.open(file_path)).astype(np.float32) / 255.0  # Normalize to 0-1 range for consistency
            max_val = 1.0  # Max value is 1.0 after normalization
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 and img.shape[2] > 1 else ['L']  # Handle grayscale PNG/TGA
        else:
            logging.warning(f"Неподдерживаемый формат файла: {file_path}")
            return None, None, None
        return img, max_val, channels
    except Exception as e:
        logging.error(f"Ошибка чтения файла {file_path}: {e}")
        return None, None, None


def calculate_psnr(original: np.ndarray, processed: np.ndarray, max_val: float) -> float:
    """
    Вычисляет пиковое отношение сигнал/шум (PSNR) между двумя изображениями.

    Аргументы:
        original (np.ndarray): Оригинальное изображение в виде массива numpy.
        processed (np.ndarray): Обработанное изображение в виде массива numpy.
        max_val (float): Максимальное значение пикселя в изображении.

    Возвращает:
        float: Значение PSNR в децибелах. Возвращает float('inf'), если MSE равно 0.
    """
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse)
    return psnr


def calculate_channel_psnr(original: np.ndarray, processed: np.ndarray, max_val: float, channels: List[str]) -> Dict[str, float]:
    """
    Вычисляет PSNR для каждого канала изображения отдельно.

    Аргументы:
        original (np.ndarray): Оригинальное изображение.
        processed (np.ndarray): Обработанное изображение.
        max_val (float): Максимальное значение пикселя.
        channels (list): Список названий каналов (например, ['R', 'G', 'B']).

    Возвращает:
        dict: Словарь, где ключи - названия каналов, а значения - соответствующие PSNR.
    """
    channel_psnrs = {}
    for i, channel in enumerate(channels):
        mse = np.mean((original[..., i] - processed[..., i]) ** 2)
        psnr = 20 * math.log10(max_val) - 10 * math.log10(mse) if mse != 0 else float('inf')
        channel_psnrs[channel] = psnr
    return channel_psnrs


def get_quality_hint(psnr: float, for_csv: bool = False) -> str:
    """
    Возвращает текстовую подсказку о качестве изображения на основе значения PSNR.

    Аргументы:
        psnr (float): Значение PSNR.
        for_csv (bool, optional): Определяет, нужно ли возвращать подсказку для CSV (без стилей). По умолчанию False.

    Возвращает:
        str: Подсказка о качестве изображения.
    """
    for threshold in PSNR_QUALITY_THRESHOLDS:
        if psnr >= threshold:
            hint = QUALITY_HINTS[threshold]
            return hint if for_csv else f"{STYLES[get_quality_style(psnr)]}{hint}"
    return QUALITY_HINTS[0] if for_csv else f"{STYLES['bad']}{QUALITY_HINTS[0]}"  # Fallback to 'bad' if no threshold is met


def get_quality_style(psnr: float) -> str:
    """
    Определяет стиль форматирования на основе значения PSNR.

    Аргументы:
        psnr (float): Значение PSNR.

    Возвращает:
        str: Ключ стиля из словаря STYLES.
    """
    if psnr >= 50:
        return 'good'
    elif psnr >= 40:
        return 'ok'
    elif psnr >= 30:
        return 'medium'
    else:
        return 'bad'


def compute_resolutions(original_width: int, original_height: int, min_size: int = 16) -> List[Tuple[int, int]]:
    """
    Вычисляет список разрешений для анализа, последовательно уменьшая размеры изображения вдвое.

    Процесс продолжается, пока обе стороны изображения не станут меньше, чем min_size * 2.

    Аргументы:
        original_width (int): Исходная ширина изображения.
        original_height (int): Исходная высота изображения.
        min_size (int, optional): Минимальный размер стороны изображения. По умолчанию 16.

    Возвращает:
        list: Список кортежей (ширина, высота) для уменьшенных разрешений.
    """
    resolutions = []
    current_width, current_height = original_width, original_height
    while current_width >= min_size * 2 and current_height >= min_size * 2:
        current_width //= 2
        current_height //= 2
        resolutions.append((current_width, current_height))
    return resolutions


def process_image(file_path: str, analyze_channels: bool, interpolation: str) -> Tuple[List, Optional[float], Optional[List[str]]]:
    """
    Обрабатывает изображение, вычисляя PSNR для различных уменьшенных и увеличенных разрешений.

    Аргументы:
        file_path (str): Путь к файлу изображения.
        analyze_channels (bool): Флаг, указывающий, нужно ли анализировать каналы раздельно.
        interpolation (str): Метод интерполяции для масштабирования ('bilinear' или 'bicubic').

    Возвращает:
        tuple: (results, max_value, channels), где results - список результатов анализа,
               max_value - максимальное значение пикселя (только для EXR), channels - список каналов.
    """
    img, max_val, channels = load_image(file_path)
    if img is None:
        return [], None, None

    original_height, original_width = img.shape[:2]
    original_resolution_str = f"{original_width}x{original_height}"

    results = []
    if analyze_channels and channels:
        channel_psnr_original = {c: float('inf') for c in channels}
        results.append((original_resolution_str, channel_psnr_original, float('inf'), f"{STYLES['original']}Оригинал{Style.RESET_ALL}"))
    else:
        results.append((original_resolution_str, float('inf'), f"{STYLES['original']}Оригинал{Style.RESET_ALL}"))

    resolutions = compute_resolutions(original_width, original_height)
    interpolation_flag = INTERPOLATION_METHODS.get(interpolation, DEFAULT_INTERPOLATION)

    for target_width, target_height in resolutions:
        if interpolation_flag == 'mitchell':
            downscaled_img = resize_mitchell(img.copy(), (target_width, target_height))
            upscaled_img = resize_mitchell(downscaled_img.copy(), (original_width, original_height))
        else:
            cv2_interpolation_flag = getattr(cv2, interpolation_flag, cv2.INTER_LINEAR)
            downscaled_img = cv2.resize(img, (target_width, target_height), interpolation=cv2_interpolation_flag)
            upscaled_img = cv2.resize(downscaled_img, (original_width, original_height), interpolation=cv2_interpolation_flag)

        if analyze_channels and channels:
            channel_psnr = calculate_channel_psnr(img, upscaled_img, max_val, channels)
            min_psnr = min(channel_psnr.values())
            results.append((f"{target_width}x{target_height}", channel_psnr, min_psnr, get_quality_hint(min_psnr)))
        else:
            psnr = calculate_psnr(img, upscaled_img, max_val)
            results.append((f"{target_width}x{target_height}", psnr, get_quality_hint(psnr)))

    return results, max_val, channels


def collect_files_to_process(paths: list[str]) -> List[str]:
    """
    Собирает список файлов для обработки из указанных путей.

    Если путь - это файл, он добавляется в список. Если путь - это директория,
    функция рекурсивно обходит директорию и добавляет все файлы с поддерживаемыми расширениями.

    Аргументы:
        paths (list): Список путей к файлам или директориям.

    Возвращает:
        list: Список путей к файлам для обработки.
    """
    files_to_process = []
    for path in paths:
        if os.path.isfile(path):
            files_to_process.append(path)
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
                        files_to_process.append(os.path.join(dirpath, filename))
        else:
            logging.warning(f"Путь не является файлом или директорией, или не существует: {path}")
    return files_to_process


def output_results_console(file_path: str, results: list, analyze_channels: bool, channels: Optional[List[str]], max_val: Optional[float]):
    """
    Выводит результаты анализа в консоль в форматированном виде.

    Аргументы:
        file_path (str): Путь к обработанному файлу изображения.
        results (list): Список результатов анализа, возвращенный функцией process_image.
        analyze_channels (bool): Флаг, указывающий, проводился ли анализ каналов.
        channels (list|None): Список каналов изображения, если проводился анализ каналов.
        max_val (float|None): Максимальное значение пикселя в изображении (может быть None).
    """
    print(f"\n\n{STYLES['header']}--- {os.path.basename(file_path):^50} ---{Style.RESET_ALL}")

    if max_val is not None and max_val < 0.001:
        print(f"{STYLES['warning']}Предупреждение: Максимальное значение {max_val:.3e}!{Style.RESET_ALL}")

    if analyze_channels and channels:
        channel_header_labels = channels
        header = f"\n{Style.BRIGHT}{'Разрешение':<12} | {' | '.join([c.center(9) for c in ['R(L)', 'G', 'B', 'A'][:len(channels)]])} | {'Min':^9} | {'Качество (min)':<36}{Style.RESET_ALL}"
        print(header)
        header_line = f"{'-'*12}-+-" + "-+-".join(["-" * 9] * len(['R(L)', 'G', 'B', 'A'][:len(channels)])) + f"-+-{'-'*9}-+-{'-'*32}"
        print(header_line)

        for res, ch_psnr, min_psnr, hint in results:
            ch_values_console = ' | '.join([f"{ch_psnr.get(c, 0):9.2f}" for c in ['R', 'G', 'B', 'A'][:len(channels)]])
            print(f"{res:<12} | {ch_values_console} | {min_psnr:9.2f} | {hint:<36}")
    else:
        print(f"\n{Style.BRIGHT}{'Разрешение':<12} | {'PSNR (dB)':^10} | {'Качество':<36}{Style.RESET_ALL}")
        print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")
        for res, psnr, hint in results:
            print(f"{res:<12} {Style.DIM}|{Style.NORMAL} {psnr:^10.2f} {Style.DIM}|{Style.NORMAL} {hint:<36}")
    print()


def output_results_csv(file_path: str, results: list, analyze_channels: bool, channels: Optional[List[str]], csv_writer):
    """
    Записывает результаты анализа в CSV файл.

    Аргументы:
        file_path (str): Путь к обработанному файлу изображения.
        results (list): Список результатов анализа.
        analyze_channels (bool): Флаг, указывающий, проводился ли анализ каналов.
        channels (list|None): Список каналов изображения, если проводился анализ каналов.
        csv_writer: Объект csv.writer для записи в CSV файл.
    """
    first_row = True  # Flag to indicate the first data row for a file

    for res, psnr_values, quality_hint, *rest in results:  # Unpack results tuple
        if first_row:
            # Write filename only in the first row for each file
            csv_writer.writerow([os.path.basename(file_path), res, "", "", "", "", "", ""])
            first_row = False
        else:
            csv_row_values = ["", res]  # Empty filename for subsequent rows

            if analyze_channels and channels:
                ch_psnr = psnr_values  # psnr_values is channel PSNR dict in this case
                csv_row_values.extend([
                    f"{ch_psnr.get('R', ch_psnr.get('L', float('inf'))):.2f}",  # R or L channel
                    f"{ch_psnr.get('G', float('inf')):.2f}",
                    f"{ch_psnr.get('B', float('inf')):.2f}",
                    f"{ch_psnr.get('A', float('inf')):.2f}",
                    f'{quality_hint:.2f}',  # quality_hint is min_psnr in channel analysis
                    get_quality_hint(quality_hint, for_csv=True)  # quality_hint is min_psnr
                ])
            else:
                psnr = psnr_values  # psnr_values is single PSNR value in this case
                csv_row_values.extend([
                    f'{psnr:.2f}',  # PSNR value
                    "", "", "", "",  # Empty columns for channel PSNRs - remove these
                    "",  # Empty column for Min PSNR - remove this
                    get_quality_hint(psnr, for_csv=True)  # Quality hint for overall PSNR
                ])
                # Corrected: Insert PSNR value in "R(L) PSNR" column and quality hint in "Качество (min)"
                csv_row_values_corrected = [csv_row_values[0], csv_row_values[1], csv_row_values[2]] + [""] * 4 + [csv_row_values[-1]]
                csv_writer.writerow(csv_row_values_corrected)
                continue  # Skip original writerow to avoid duplicate write

            csv_writer.writerow(csv_row_values)  # Only write for channel analysis case


def parse_arguments():
    """
    Настраивает разбор аргументов командной строки.

    Возвращает:
        argparse.Namespace: Объект, содержащий аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description='Анализ потерь качества текстур при масштабировании.')
    parser.add_argument('paths', nargs='+', help='Пути к файлам текстур или директориям для анализа.')
    parser.add_argument('--channels', '-c', action='store_true', help='Включить анализ по цветовым каналам.')
    parser.add_argument('--csv-output', action='store_true', help='Выводить результаты в CSV файл.')
    parser.add_argument('--interpolation', '-i', default=DEFAULT_INTERPOLATION, choices=INTERPOLATION_METHODS.keys(), help=f"Метод интерполяции для масштабирования. По умолчанию: {DEFAULT_INTERPOLATION}")

    return parser.parse_args()


def main():
    """
    Главная функция скрипта, координирующая процесс анализа изображений и вывода результатов.
    """
    args = parse_arguments()
    files_to_process = collect_files_to_process(args.paths)

    if not files_to_process:
        print("Не найдено поддерживаемых файлов для обработки.")
        return

    if args.csv_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}.csv"
        csv_filepath = os.path.join(os.getcwd(), csv_filename)

        with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=CSV_SEPARATOR)  # Explicitly set delimiter

            general_csv_header = ["Файл", "Разрешение", "R(L) PSNR", "G PSNR", "B PSNR", "A PSNR", "Min PSNR", "Качество (min)"]
            csv_writer.writerow(general_csv_header)

            for file_path in files_to_process:
                results, max_val, channels = process_image(file_path, args.channels, args.interpolation)
                if results:  # Process results only if not empty
                    output_results_csv(file_path, results, args.channels, channels, csv_writer)
                    output_results_console(file_path, results, args.channels, channels, max_val)  # Still output to console for user feedback
        print(f"\nМетрики сохранены в: {csv_filepath}")

    else:
        for file_path in files_to_process:
            results, max_val, channels = process_image(file_path, args.channels, args.interpolation)
            if results:  # Process results only if not empty
                output_results_console(file_path, results, args.channels, channels, max_val)


if __name__ == "__main__":
    main()
