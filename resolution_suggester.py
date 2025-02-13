#!/usr/bin/env python3
import os
import math
import argparse
import datetime

import numpy as np
import cv2
from PIL import Image, ImageFile
import pyexr
from colorama import init, Fore, Back, Style

# Initialize colorama and PIL
init(autoreset=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Styles for output
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
    mse = np.mean((original - processed) ** 2)
    return 20 * math.log10(max_val) - 10 * math.log10(mse) if mse != 0 else float('inf')

def calculate_channel_psnr(original: np.ndarray, processed: np.ndarray, max_val: float, channels: list) -> dict:
    results = {}
    for i, channel in enumerate(channels):
        mse = np.mean((original[..., i] - processed[..., i]) ** 2)
        psnr = 20 * math.log10(max_val) - 10 * math.log10(mse) if mse != 0 else float('inf')
        results[channel] = psnr
    return results

def get_quality_hint(psnr: float) -> str:
    if psnr >= 50:
        return f"{STYLES['good']}практически идентичные изображения"
    elif psnr >= 40:
        return f"{STYLES['ok']}очень хорошее качество"
    elif psnr >= 30:
        return f"{STYLES['medium']}приемлемое качество"
    else:
        return f"{STYLES['bad']}заметные потери"

def load_image(image_path: str) -> tuple:
    try:
        if image_path.lower().endswith('.exr'):
            img = pyexr.read(image_path).astype(np.float32)
            max_val = np.max(np.abs(img)) or 1e-6 # Ensure max_val is not zero
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L']
            return img, max_val, channels
        elif image_path.lower().endswith(('.png', '.tga')):
            img = np.array(Image.open(image_path)).astype(np.float32)
            max_val = 255.0
            channels = ['R', 'G', 'B', 'A'][:img.shape[2]] if img.ndim > 2 else ['L']
            return img, max_val, channels
        else:
            print(f"Unsupported format: {image_path}") # Keep print for console error message
            return None, None, None
    except Exception as e:
        print(f"Error reading {image_path}: {e}") # Keep print for console error message
        return None, None, None

def compute_resolutions(original_w: int, original_h: int, min_size: int = 16) -> list:
    resolutions = []
    current_w, current_h = original_w, original_h
    while current_w >= min_size * 2 and current_h >= min_size * 2:
        current_w //= 2
        current_h //= 2
        resolutions.append((current_w, current_h))
    return resolutions

def process_image(image_path: str, analyze_channels: bool, logger) -> tuple: # Pass logger
    img, max_val, channels = load_image(image_path)
    if img is None:
        return [], None, None

    original_h, original_w = img.shape[:2]
    orig_res_str = f"{original_w}x{original_h}"
    results = []

    if analyze_channels and channels and len(channels) > 1:
        results.append((orig_res_str, {c: float('inf') for c in channels}, float('inf'), f'{STYLES["original"]}оригинал{Style.RESET_ALL}'))
    elif analyze_channels and channels and len(channels) == 1:
        results.append((orig_res_str, {c: float('inf') for c in channels}, float('inf'), f'{STYLES["original"]}оригинал{Style.RESET_ALL}'))
    else:
        results.append((orig_res_str, float('inf'), f'{STYLES["original"]}оригинал{Style.RESET_ALL}'))

    resolutions = compute_resolutions(original_w, original_h)
    for target_w, target_h in resolutions:
        downscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        if analyze_channels and channels and len(channels) > 1:
            channel_psnr = calculate_channel_psnr(img, upscaled, max_val, channels)
            min_psnr = min(channel_psnr.values())
            results.append((f"{target_w}x{target_h}", channel_psnr, min_psnr, get_quality_hint(min_psnr)))
        elif analyze_channels and channels and len(channels) == 1:
            psnr = calculate_psnr(img, upscaled, max_val)
            results.append((f"{target_w}x{target_h}", {'L': psnr}, psnr, get_quality_hint(psnr)))
        else:
            psnr = calculate_psnr(img, upscaled, max_val)
            results.append((f"{target_w}x{target_h}", psnr, get_quality_hint(psnr)))

    return results, max_val if image_path.lower().endswith('.exr') else None, channels if analyze_channels else None

class Logger: # Simple logger class
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.is_logging_to_file = log_file is not None

    def log(self, text=""):
        print(text) # Always print to console
        if self.is_logging_to_file:
            try:
                self.log_file.write(text + "\n")
            except ValueError as e: # Catch file closed error more explicitly
                print(f"Logging error (file may be closed): {e}")
                self.is_logging_to_file = False # Disable further file logging

    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
            except Exception as e:
                print(f"Error closing log file: {e}")
            finally:
                self.log_file = None
                self.is_logging_to_file = False

def main():
    parser = argparse.ArgumentParser(description='Texture Quality Analysis')
    parser.add_argument('paths', nargs='+', help='Texture file or directory paths')
    parser.add_argument('--channels', '-c', action='store_true', help='Analyze color channels separately')
    parser.add_argument('--log-file', '-l', nargs='?', const=None, default=False, help='Log output to file')
    args = parser.parse_args()

    separator = f"{Style.DIM}|{Style.NORMAL}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = args.log_file if args.log_file is not None else f"{timestamp}.log"
    log_to_file = args.log_file is not False
    log_file = None # Initialize log_file outside try block
    overwrite_allowed = False # Flag to track overwrite permission

    if log_to_file:
        os.makedirs(os.path.dirname(log_filename) or '.', exist_ok=True)
        if os.path.exists(log_filename): # Check if log file already exists
            response = input(f"Log file '{log_filename}' already exists. Overwrite? (y/N): ").strip().lower()
            if response == 'y':
                overwrite_allowed = True
            else:
                print("Logging to console only, log file will not be overwritten.")
                log_to_file = False # Disable file logging
        else:
            overwrite_allowed = True # If file doesn't exist, allow overwrite (which is creation)

        if log_to_file and overwrite_allowed:
            try:
                log_file = open(log_filename, 'w', encoding='utf-8') # Open log file if overwrite allowed
                print(f"Logging to file: {log_filename}")
            except Exception as e:
                print(f"Error opening log file {log_filename}: {e}. Logging to console only.")
                log_file = None
                log_to_file = False # Disable file logging if open fails
        else:
            log_file = None # Ensure log_file is None if not logging to file
    else:
        print("Logging to console.")


    logger = Logger(log_file) # Instantiate logger

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
            logger.log(f"Path is not a file or directory, or doesn't exist: {path}") # Use logger

    if not files_to_process:
        logger.log("No supported files found for processing.") # Use logger
        logger.close() # Close log file before exiting
        return

    logger.log(f"Log filename (determined): {log_filename}") # Debug log

    try: # Main processing in a try-finally block
        for path in files_to_process:
            logger.log(f"\n\n{STYLES['header']}--- {os.path.basename(path):^50} ---{Style.RESET_ALL}")
            results, max_val, channels = process_image(path, args.channels, logger) # Pass logger
            if not results:
                continue
            if max_val is not None and max_val < 0.001:
                logger.log(f"{STYLES['warning']}WARNING: Max value {max_val:.3e}!{Style.RESET_ALL}")

            if args.channels:
                if channels:
                    channel_header_labels = channels if len(channels) > 1 else channels
                    header = f"\n{Style.BRIGHT}{'Resolution':<12} | {' | '.join([c.center(6) for c in channel_header_labels])} | {'Min':^6} | {'Quality (min)':<36}{Style.RESET_ALL}"
                    logger.log(header)
                    header_line = f"{'-'*12}-+-" + "-+-".join(["-"*6]*len(channel_header_labels)) + f"-+-{'-'*6}-+-{'-'*32}"
                    logger.log(header_line)
                    for res, ch_psnr, min_psnr, hint in results:
                        ch_values = ' | '.join([f"{ch_psnr.get(c, 0):6.2f}" for c in channel_header_labels])
                        logger.log(f"{res:<12} | {ch_values} | {min_psnr:6.2f} | {hint:<36}")
            else:
                header = f"\n{Style.BRIGHT}{'Resolution':<12} | {'PSNR (dB)':^10} | {'Quality':<36}{Style.RESET_ALL}"
                logger.log(header)
                header_line = f"{'-'*12}-+-{'-'*10}-+-{'-'*30}"
                logger.log(header_line)
                for res, psnr, hint in results:
                    logger.log(f"{res:<12} {separator} {psnr:^10.2f} {separator} {hint:<36}")

    finally: # Ensure log file is closed in finally block
        logger.close()
        if log_to_file and log_filename and overwrite_allowed: # Confirmation message only if logging to file was intended and allowed
            print(f"Output also logged to: {log_filename}")

if __name__ == "__main__":
    main()
