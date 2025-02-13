#!/usr/bin/env python3
import os
import math
import argparse

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

def get_quality_hint(psnr: float) -> str:
    """
    Возвращает строку с подсказкой о качестве изображения на основе PSNR.
    """
    if psnr >= 50:
        return f"{STYLES['good']}практически идентичные изображения"
    elif psnr >= 40:
        return f"{STYLES['ok']}очень хорошее качество"
    elif psnr >= 30:
        return f"{STYLES['medium']}приемлемое качество"
    else:
        return f"{STYLES['bad']}заметные потери"

def load_image(image_path: str) -> tuple:
    """
    Загружает изображение и возвращает numpy-массив, максимальное значение и каналы.
    """
    try:
        if image_path.lower().endswith('.exr'):
            img = pyexr.read(image_path).astype(np.float32)
            max_val = np.max(np.abs(img))
            max_val = max(max_val, 1e-6)
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]]
            return img, max_val, channels
        elif image_path.lower().endswith(('.png', '.tga')):
            img = np.array(Image.open(image_path)).astype(np.float32)
            max_val = 255.0
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]]
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
    if analyze_channels:
        # Для оригинала указываем бесконечное PSNR для каждого канала
        results.append((
            orig_res_str,
            {c: float('inf') for c in channels},
            float('inf'),
            f'{STYLES["original"]}оригинал{Style.RESET_ALL}'
        ))
    else:
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

        if analyze_channels:
            channel_psnr = calculate_channel_psnr(img, upscaled, max_val, channels)
            min_psnr = min(channel_psnr.values())
            results.append((
                f"{target_w}x{target_h}",
                channel_psnr,
                min_psnr,
                get_quality_hint(min_psnr)
            ))
        else:
            psnr = calculate_psnr(img, upscaled, max_val)
            results.append((
                f"{target_w}x{target_h}",
                psnr,
                get_quality_hint(psnr)
            ))

    return results, max_val if image_path.lower().endswith('.exr') else None, channels if analyze_channels else None

def main():
    parser = argparse.ArgumentParser(description='Анализ потерь качества текстур')
    parser.add_argument('paths', nargs='+', help='Пути к файлам текстур')
    parser.add_argument('--channels', '-c', action='store_true', help='Анализировать отдельные цветовые каналы')
    args = parser.parse_args()

    separator = f"{Style.DIM}|{Style.NORMAL}"

    for path in args.paths:
        if not os.path.isfile(path):
            print(f"Файл не найден: {path}")
            continue

        print(f"\n\n{STYLES['header']}--- {os.path.basename(path):^50} ---{Style.RESET_ALL}")

        results, max_val, channels = process_image(path, args.channels)
        if not results:
            continue

        if max_val is not None and max_val < 0.001:
            print(f"{STYLES['warning']}АХТУНГ: Максимальное значение {max_val:.3e}!{Style.RESET_ALL}")

        if args.channels:
            if channels is not None:
                header = f"\n{Style.BRIGHT}{'Разрешение':<12} | {' | '.join([c.center(6) for c in channels])} | {'Min':^6} | {'Качество (min)':<36}{Style.RESET_ALL}"
                print(header)
                header_line = f"{'-'*12}-+-" + "-+-".join(["-"*6]*len(channels)) + f"-+-{'-'*6}-+-{'-'*32}"
                print(header_line)
                for res, ch_psnr, min_psnr, hint in results:
                    ch_values = ' | '.join([f"{ch_psnr.get(c, 0):6.2f}" for c in channels])
                    print(f"{res:<12} | {ch_values} | {min_psnr:6.2f} | {hint:<36}")
        else:
            print(f"\n{Style.BRIGHT}{'Разрешение':<12} | {'PSNR (dB)':^10} | {'Качество':<36}{Style.RESET_ALL}")
            print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")
            for res, psnr, hint in results:
                print(f"{res:<12} {separator} {psnr:^10.2f} {separator} {hint:<36}")

if __name__ == "__main__":
    main()
