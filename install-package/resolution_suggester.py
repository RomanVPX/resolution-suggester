import os
import math
import argparse
import csv
import logging
import cv2
import pyexr
import numpy as np

from datetime import datetime
from typing import Tuple, List, Optional, Dict
from numba import njit, prange
from PIL import Image, ImageFile
from colorama import init, Fore, Back, Style

# Colorama & PIL setup
init(autoreset=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
_EXTS = ['.exr', '.tga', '.png']
_HINTS = {
    50: "практически идентичные изображения",
    40: "очень хорошее качество",
    30: "приемлемое качество",
    0: "заметные потери",
}
_THRESHOLDS = sorted(_HINTS.keys(), reverse=True)
_STYLES = {
    'header': f"{Style.BRIGHT}{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}",
    'warning': f"{Style.DIM}{Fore.YELLOW}",
    'original': f"{Fore.CYAN}",
    'good': f"{Fore.LIGHTGREEN_EX}",
    'ok': f"{Fore.GREEN}",
    'medium': f"{Fore.YELLOW}",
    'bad': f"{Fore.RED}",
}
_CSV_SEP = ";"
_INTERP_DOC = {
    'bilinear': ('cv2.INTER_LINEAR', 'Билинейная интерполяция'),
    'bicubic': ('cv2.INTER_CUBIC', 'Бикубическая интерполяция'),
    'mitchell': ('mitchell', 'Фильтр Митчелла-Нетравали'),
}

_INTERP_METHODS = {
    'bilinear': ('cv2.INTER_LINEAR'),
    'bicubic': ('cv2.INTER_CUBIC'),
    'mitchell': ('mitchell'),
}
_DEFAULT_INTERP = 'mitchell'

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@njit(cache=True)
def _mn(x, B=1/3, C=1/3):
    x = np.abs(x)
    if x < 1:
        return (12 - 9 * B - 6 * C) * x**3 + (-18 + 12 * B + 6 * C) * x**2 + (6 - 2 * B)
    if x < 2:
        return (-B - 6 * C) * x**3 + (6 * B + 30 * C) * x**2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)
    return 0.0

@njit(parallel=True)
def _resize_mn_impl(img, tw, th, B=1/3, C=1/3):
    h, w = img.shape[:2]
    ch = img.shape[2] if img.ndim == 3 else 1

    r_img = np.zeros((th, tw, ch), dtype=img.dtype)

    xr = w / tw
    yr = h / th

    for i in prange(th):
        for j in range(tw):
            x = j * xr
            y = i * yr

            xf = np.floor(x)
            yf = np.floor(y)

            x_f = x - xf
            y_f = y - yf

            acc = np.zeros(ch, dtype=np.float64)
            ws = 0.0

            for m in range(-2, 2):
                for n in range(-2, 2):
                    xi = int(xf + n)
                    yi = int(yf + m)

                    if 0 <= xi < w and 0 <= yi < h:
                        weight = _mn(x_f - n, B, C) * _mn(y_f - m, B, C)
                        acc += weight * img[yi, xi] if ch > 1 else weight * img[yi, xi]
                        ws += weight

            if ws > 0:
                r_img[i, j] = acc / ws
            else:
                r_img[i, j] = 0

    return r_img

def resize_mn(img, tw, th, B=1/3, C=1/3):
    r_img = _resize_mn_impl(img, tw, th, B, C)
    if img.ndim == 2:
        return r_img[:, :, 0]
    return r_img

def load_img(fp: str) -> tuple[np.ndarray, float, list[str]] | tuple[None, None, None]:
    fe = fp.lower().split('.')[-1]

    try:
        if fe == 'exr':
            img = pyexr.read(fp).astype(np.float32)
            mv = np.max(np.abs(img))
            mv = max(mv, 1e-6)
            chs = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 and img.shape[2] > 1 else ['L']
        elif fe in ('png', 'tga'):
            img = np.array(Image.open(fp)).astype(np.float32) / 255.0
            mv = 1.0
            chs = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 and img.shape[2] > 1 else ['L']
        else:
            logging.warning(f"Неподдерживаемый формат: {fp}")
            return None, None, None
        return img, mv, chs
    except Exception as e:
        logging.error(f"Ошибка чтения {fp}: {e}")
        return None, None, None


def calc_psnr(orig: np.ndarray, proc: np.ndarray, mv: float) -> float:
    mse = np.mean((orig - proc) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(mv) - 10 * math.log10(mse)
    return psnr


def calc_ch_psnr(orig: np.ndarray, proc: np.ndarray, mv: float, chs: List[str]) -> Dict[str, float]:
    ch_psnrs = {}
    for i, ch in enumerate(chs):
        mse = np.mean((orig[..., i] - proc[..., i]) ** 2)
        psnr = 20 * math.log10(mv) - 10 * math.log10(mse) if mse != 0 else float('inf')
        ch_psnrs[ch] = psnr
    return ch_psnrs


def get_hint(psnr: float, for_csv: bool = False) -> str:
    for th in _THRESHOLDS:
        if psnr >= th:
            hint = _HINTS[th]
            return hint if for_csv else f"{_STYLES[get_style(psnr)]}{hint}"
    return _HINTS[0] if for_csv else f"{_STYLES['bad']}{_HINTS[0]}"


def get_style(psnr: float) -> str:
    if psnr >= 50:
        return 'good'
    elif psnr >= 40:
        return 'ok'
    elif psnr >= 30:
        return 'medium'
    else:
        return 'bad'


def comp_res(ow: int, oh: int, min_s: int = 16) -> List[Tuple[int, int]]:
    res = []
    cw, ch = ow, oh
    while cw >= min_s * 2 and ch >= min_s * 2:
        cw //= 2
        ch //= 2
        res.append((cw, ch))
    return res


def proc_img(fp: str, analyze_chs: bool, interp: str) -> tuple[list, float | None, list[str] | None]:
    img, mv, chs = load_img(fp)
    if img is None:
        return [], None, None

    if mv is None:
        mv = 1.0

    oh, ow = img.shape[:2]
    orig_res_str = f"{ow}x{oh}"

    res = []
    if analyze_chs and chs:
        ch_psnr_orig = {c: float('inf') for c in chs}
        res.append((orig_res_str, ch_psnr_orig, float('inf'), f"{_STYLES['original']}Оригинал{Style.RESET_ALL}"))
    else:
        res.append((orig_res_str, float('inf'), f"{_STYLES['original']}Оригинал{Style.RESET_ALL}"))

    resolutions = comp_res(ow, oh)
    interp_flag = _INTERP_METHODS.get(interp, _DEFAULT_INTERP)

    for tw, th in resolutions:
        if interp_flag == 'mitchell':
            d_img = resize_mn(img.copy(), tw, th)
            u_img = resize_mn(d_img.copy(), ow, oh)
        else:
            cv2_interp_flag = getattr(cv2, interp_flag, cv2.INTER_LINEAR)
            d_img = cv2.resize(img, (tw, th), interpolation=cv2_interp_flag)
            u_img = cv2.resize(d_img, (ow, oh), interpolation=cv2_interp_flag)

        if analyze_chs and chs:
            ch_psnr = calc_ch_psnr(img, u_img, mv, chs)
            min_psnr = min(ch_psnr.values())
            res.append((f"{tw}x{th}", ch_psnr, min_psnr, get_hint(min_psnr)))
        else:
            psnr = calc_psnr(img, u_img, mv)
            res.append((f"{tw}x{th}", psnr, get_hint(psnr)))

    return res, mv, chs


def collect_files(paths: list[str]) -> List[str]:
    files = []
    for path in paths:
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.lower().endswith(tuple(_EXTS)):
                        files.append(os.path.join(dirpath, filename))
        else:
            logging.warning(f"Путь не файл/директория: {path}")
    return files


def output_console(fp: str, res: list, analyze_chs: bool, chs: Optional[List[str]], mv: Optional[float]):
    print(f"\n\n{_STYLES['header']}--- {os.path.basename(fp):^50} ---{Style.RESET_ALL}")

    if mv is not None and mv < 0.001:
        print(f"{_STYLES['warning']}Предупреждение: Max value {mv:.3e}!{Style.RESET_ALL}")

    if analyze_chs and chs:
        header = f"\n{Style.BRIGHT}{'Разрешение':<12} | {' | '.join([c.center(9) for c in ['R(L)', 'G', 'B', 'A'][:len(chs)]])} | {'Min':^9} | {'Качество (min)':<36}{Style.RESET_ALL}"
        print(header)
        header_line = f"{'-'*12}-+-" + "-+-".join(["-" * 9] * len(['R(L)', 'G', 'B', 'A'][:len(chs)])) + f"-+-{'-'*9}-+-{'-'*32}"
        print(header_line)

        for r, ch_psnr, min_psnr, hint in res:
            ch_vals_console = ' | '.join([f"{ch_psnr.get(c, 0):9.2f}" for c in ['R', 'G', 'B', 'A'][:len(chs)]])
            print(f"{r:<12} | {ch_vals_console} | {min_psnr:9.2f} | {hint:<36}")
    else:
        print(f"\n{Style.BRIGHT}{'Разрешение':<12} | {'PSNR (dB)':^10} | {'Качество':<36}{Style.RESET_ALL}")
        print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")
        for r, psnr, hint in res:
            print(f"{r:<12} {Style.DIM}|{Style.NORMAL} {psnr:^10.2f} {Style.DIM}|{Style.NORMAL} {hint:<36}")
    print()


def output_csv(fp: str, res: list, analyze_chs: bool, chs: Optional[List[str]], csv_writer):
    first_row = True

    for r, psnr_values, quality_hint, *rest in res:
        if first_row:
            csv_writer.writerow([os.path.basename(fp), r, "", "", "", "", "", ""])
            first_row = False
        else:
            csv_row_values = ["", r]

            if analyze_chs and chs:
                ch_psnr = psnr_values
                csv_row_values.extend([
                    f"{ch_psnr.get('R', ch_psnr.get('L', float('inf'))):.2f}",
                    f"{ch_psnr.get('G', float('inf')):.2f}",
                    f"{ch_psnr.get('B', float('inf')):.2f}",
                    f"{ch_psnr.get('A', float('inf')):.2f}",
                    f'{quality_hint:.2f}',
                    get_hint(quality_hint, for_csv=True)
                ])
            else:
                psnr = psnr_values
                csv_row_values.extend([
                    f'{psnr:.2f}',
                    "", "", "", "",
                    "",
                    get_hint(psnr, for_csv=True)
                ])

                csv_row_values_corrected = [csv_row_values[0], csv_row_values[1], csv_row_values[2]] + [""] * 4 + [csv_row_values[-1]]
                csv_writer.writerow(csv_row_values_corrected)
                continue

            csv_writer.writerow(csv_row_values)


def format_interp_help():
    methods = [f"{m:<8}{' (default)' if m == _DEFAULT_INTERP else '':<15} {_INTERP_DOC[m][1]}."
               for m in _INTERP_METHODS.keys()]
    return ("Метод интерполяции. Варианты:\n" +
            '\n'.join(methods))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Анализ потерь качества текстур при масштабировании.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('paths', nargs='+', help='Пути к файлам/директориям.')
    parser.add_argument('-c', '--channels', action='store_true', help='Анализ по каналам.')
    parser.add_argument('-o', '--csv-output', action='store_true', help='Вывод в CSV.')
    parser.add_argument('-i', '--interpolation', default=_DEFAULT_INTERP, choices=_INTERP_DOC.keys(), metavar='METHOD', help=format_interp_help())

    return parser.parse_args()


def main():
    args = parse_args()
    files_to_process = collect_files(args.paths)

    if not files_to_process:
        print("Файлы не найдены.")
        return

    if args.csv_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}.csv"
        csv_filepath = os.path.join(os.getcwd(), csv_filename)

        with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=_CSV_SEP)

            general_csv_header = ["Файл", "Разрешение", "R(L) PSNR", "G PSNR", "B PSNR", "A PSNR", "Min PSNR", "Качество (min)"]
            csv_writer.writerow(general_csv_header)

            for file_path in files_to_process:
                results, max_val, channels = proc_img(file_path, args.channels, args.interpolation)
                if results:
                    output_csv(file_path, results, args.channels, channels, csv_writer)
                    output_console(file_path, results, args.channels, channels, max_val)
        print(f"\nМетрики сохранены в: {csv_filepath}")

    else:
        for file_path in files_to_process:
            results, max_val, channels = proc_img(file_path, args.channels, args.interpolation)
            if results:
                output_console(file_path, results, args.channels, channels, max_val)


if __name__ == "__main__":
    main()
