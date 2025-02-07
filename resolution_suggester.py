import os
import argparse
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

def calculate_channel_psnr(original, processed, max_val, channels):
    results = {}
    for i, channel in enumerate(channels):
        orig = original[..., i]
        proc = processed[..., i]
        mse = np.mean((orig - proc) ** 2)
        psnr = 20 * math.log10(max_val) - 10 * math.log10(mse) if mse !=0 else float('inf')
        results[channel] = psnr
    return results

def get_hint(psnr):
    if psnr >= 50:
        return f"{STYLES['good']}практически идентичные изображения"
    elif psnr >= 40:
        return f"{STYLES['ok']}очень хорошее качество"
    elif psnr >= 30:
        return f"{STYLES['medium']}приемлемое качество"
    return f"{STYLES['bad']}заметные потери"

def process_image(image_path, analyze_channels):
    try:
        if image_path.lower().endswith('.exr'):
            img = pyexr.read(image_path).astype(np.float32)
            max_val = np.max(np.abs(img))
            max_val = max(max_val, 1e-6)
            exr_max = max_val
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]]
        elif image_path.lower().endswith(('.png', '.tga')):
            img = np.array(Image.open(image_path)).astype(np.float32)
            max_val = 255.0
            exr_max = None
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]]
        else:
            return [], None, None
    except Exception as e:
        print(f"Ошибка чтения {image_path}: {str(e)}")
        return [], None, None

    original_h, original_w = img.shape[:2]
    results = []
    orig_res_str = f"{original_w}x{original_h}"

    if analyze_channels:
        # Добавляем оригинал с "бесконечными" значениями для каналов
        results.append((
            orig_res_str,
            {c: float('inf') for c in channels},
            float('inf'),
            f'{Fore.CYAN}оригинал{Style.RESET_ALL}'
        ))
    else:
        # Обычный режим
        results.append((
            orig_res_str,
            float('inf'),
            f'{Fore.CYAN}оригинал{Style.RESET_ALL}'
        ))

    resolutions = []
    current_w, current_h = original_w, original_h
    while (current_w := current_w // 2) >= 16 and (current_h := current_h // 2) >= 16:
        resolutions.append((current_w, current_h))

    for target_w, target_h in resolutions:
        downscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        if analyze_channels:
            channel_psnr = calculate_channel_psnr(img, upscaled, max_val, channels)
            min_psnr = min(channel_psnr.values())
            results.append((
                f"{target_w}x{target_h}",
                channel_psnr,
                min_psnr,
                get_hint(min_psnr)
            ))
        else:
            psnr = calculate_psnr(img, upscaled, max_val)
            results.append((
                f"{target_w}x{target_h}",
                psnr,
                get_hint(psnr)
            ))

    return results, exr_max, channels if analyze_channels else None


def main():
    parser = argparse.ArgumentParser(description='Анализ потерь качества текстур')
    parser.add_argument('paths', nargs='+', help='Пути к файлам текстур')
    parser.add_argument('--channels', '-c', action='store_true',
                       help='Анализировать отдельные цветовые каналы')
    args = parser.parse_args()

    col_sep = f"{Style.DIM}|{Style.NORMAL}"

    for path in args.paths:
        if not os.path.isfile(path):
            print(f"Файл не найден: {path}")
            continue

        print(f"\n\n{STYLES['header']}--- {os.path.basename(path):^50} ---{Style.RESET_ALL}")

        results, exr_max, channels = process_image(path, args.channels)
        if not results:
            continue

        if exr_max is not None and exr_max < 0.001:
            print(f"{STYLES['warning']}АХТУНГ: Максимальное значение {exr_max:.3e}!{Style.RESET_ALL}")

        if args.channels:
            if channels is not None:
                print(f"\n{Style.BRIGHT}{'Разрешение':<12} | {'R':^6} | {'G':^6} | {'B':^6} | {'A':^6} | {'Min':^6} | {'Качество (min)':<36}{Style.RESET_ALL}")
                channel_hor_bottom = f"{'-'*6}-+-"
                print(f"{'-'*12}-+-{channel_hor_bottom*4}{'-'*6}-+-{'-'*32}")

                for res, ch_psnr, min_psnr, hint in results:
                    ch_values = ' | '.join([f"{ch_psnr[c]:6.2f}" if c in ch_psnr else ' '*6 for c in ['R','G','B','A']])
                    print(f"{res:<12} | {ch_values} | {min_psnr:6.2f} | {hint:<36}")

        else:
            print(f"\n{Style.BRIGHT}{'Разрешение':<12} | {'PSNR (dB)':^10} | {'Качество':<36}{Style.RESET_ALL}")
            print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")

            for res, psnr, hint in results:
                psnr_str = '{:^10}'.format('+∞') if math.isinf(psnr) else f"{psnr:^10.2f}"
                print(f"{res:<12} {col_sep:<0} {psnr_str} {col_sep:<0} {hint:<36}")

if __name__ == "__main__":
    main()