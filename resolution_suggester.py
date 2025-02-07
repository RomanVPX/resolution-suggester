import os
import argparse
from re import S
import numpy as np
from PIL import Image, ImageFile
import pyexr
import cv2
import math

from colorama import init, Fore, Back, Style

init(autoreset=True) # allow loading huge TARGA images

STYLES = {
    'header': f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}",
    'warning': f"{Style.DIM}{Fore.RED}",
    'original': f"{Fore.CYAN}",
    'good': f"{Fore.LIGHTGREEN_EX}",
    'ok': f"{Fore.GREEN}",
    'medium': f"{Fore.YELLOW}",
    'bad': f"{Fore.RED}",
}

ImageFile.LOAD_TRUNCATED_IMAGES = True # allow loading truncated images

def calculate_psnr(original, processed, max_val):
    mse = np.mean((original - processed) ** 2)
    return 20 * math.log10(max_val) - 10 * math.log10(mse) if mse != 0 else float('inf')

def get_hint(psnr):
    if psnr >= 50:
        return f"{Fore.LIGHTGREEN_EX}практически идентичные изображения{Style.RESET_ALL}"
        # return 'практически идентичные изображения'
    elif psnr >= 40:
        return f"{Fore.GREEN}очень хорошее качество{Style.RESET_ALL}"
    elif psnr >= 30:
        return f"{Fore.YELLOW}приемлемое качество{Style.RESET_ALL}"
    return f"{Fore.RED}заметные потери{Style.RESET_ALL}"

def process_image(image_path):
    try:
        if image_path.lower().endswith('.exr'):
            img = pyexr.read(image_path).astype(np.float32)
            max_val = np.max(np.abs(img))
            max_val = max(max_val, 1e-6)
            exr_max = max_val  # Сохраняем для проверки
        elif image_path.lower().endswith(('.png', '.tga')):
            img = np.array(Image.open(image_path)).astype(np.float32)
            max_val = 255.0
            exr_max = None
        else:
            return None, None
    except Exception as e:
        print(f"Ошибка чтения {image_path}: {str(e)}")
        return None, None

    original_h, original_w = img.shape[:2]
    results = []

    orig_res_str = f"{original_w}x{original_h}"
    results.append((f"{orig_res_str}", float('inf'), f'{Fore.CYAN}оригинал{Style.RESET_ALL}'))

    resolutions = []
    current_w, current_h = original_w, original_h
    while (current_w := current_w // 2) >= 16 and (current_h := current_h // 2) >= 16:
        resolutions.append((current_w, current_h))

    for target_w, target_h in resolutions:
        downscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        psnr = calculate_psnr(img, upscaled, max_val)
        results.append((f"{target_w}x{target_h}", psnr, get_hint(psnr)))

    return results, exr_max

def main():
    parser = argparse.ArgumentParser(description='Анализ потерь качества текстур при уменьшении разрешения')
    parser.add_argument('paths', nargs='+', help='Пути к файлам текстур (.exr, .tga, или .png)')
    args = parser.parse_args()

    for path in args.paths:
        if not os.path.isfile(path):
            print(f"Файл не найден: {path}")
            continue

        print(f"\n\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}--- {os.path.basename(path):^50} ---{Style.RESET_ALL}")
        print(Style.RESET_ALL, end='', flush=True)
        results, exr_max = process_image(path)
        if not results:
            continue

        if exr_max is not None and exr_max < 0.001:
            print(Style.DIM, Fore.RED, end='', flush=True)
            print(f"АХТУНГ: Максимальное значение текстуры всего {exr_max:.6f}!")
            print("           PSNR-метрика может быть некорректной для таких значений")
            print(Style.RESET_ALL, end='', flush=True)

        col_sep = f"{Style.DIM}|{Style.NORMAL}"
        header = '\n{:<12} | {:^10} | {:<36}'.format('Разрешение', 'PSNR (dB)', 'Словами')
        separator = '-' * 12 + '-+-' + '-' * 10 + '-+-' + '-' * 30

        print(f'{Style.BRIGHT}{header}{Style.RESET_ALL}')
        print(f'{Style.NORMAL}{separator}{Style.RESET_ALL}')

        for res, psnr, hint in results:
            psnr_str = '{:^10}'.format('+∞') if math.isinf(psnr) else f"{psnr:^10.2f}"
            print(f"{res:<12} {col_sep:<0} {psnr_str} {col_sep:<0} {hint:<36}")

if __name__ == "__main__":
    main()