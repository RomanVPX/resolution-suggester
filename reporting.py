# reporting.py
from datetime import datetime
import csv
import os
from typing import List, Optional
from colorama import Style

from config import (
    STYLES,
    CSV_SEPARATOR,
    QUALITY_HINTS,
    PSNR_QUALITY_THRESHOLDS,
    get_output_csv_header
)


def generate_csv_filename() -> str:
    """Генерация имени CSV файла с временной меткой"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"texture_analysis_{timestamp}.csv"


class ConsoleReporter:
    @staticmethod
    def print_file_header(file_path: str):
        title = os.path.basename(file_path).center(50)
        print(f"\n\n{STYLES['header']}--- {title} ---{Style.RESET_ALL}")

    @staticmethod
    def print_quality_table(results: list, analyze_channels: bool, channels: Optional[List[str]]):
        if analyze_channels and channels:
            ConsoleReporter._print_channel_table(results, channels)
        else:
            ConsoleReporter._print_simple_table(results)

    @staticmethod
    def _print_channel_table(results: list, channels: List[str]):
        header = f"{Style.BRIGHT}{'Разрешение':<12} | {' | '.join([c.center(9) for c in ['R(L)', 'G', 'B', 'A'][:len(channels)]])} | {'Min':^9} | {'Качество (min)':<36}{Style.RESET_ALL}"
        print(header)
        print(f"{'-'*12}-+-{'-+-'.join(['-' * 9] * len(['R(L)', 'G', 'B', 'A'][:len(channels)]))}-+-{'-'*9}-+-{'-'*32}")

        for res, ch_psnr, min_psnr, hint in results:
            if hint.endswith("Оригинал"):
                original_style = STYLES['original']
                hint_styled = f"{original_style}Оригинал{Style.RESET_ALL}"
                min_psnr_styled = f"{original_style}{min_psnr:9.2f}{Style.RESET_ALL}"

                # Correctly handle channel names and formatting for grayscale images
                if 'L' in channels and len(channels) == 1:
                    channel_values = [f"{original_style}{ch_psnr.get('L', float('inf')):>9.2f}{Style.RESET_ALL}"] # Right-align and format L channel
                else:
                    channel_values = [f"{original_style}{ch_psnr.get(c, float('inf')):>9.2f}{Style.RESET_ALL}" for c in ['R', 'G', 'B', 'A'][:len(channels)]] # Right-align and format other channels

                ch_values_console = ' | '.join(channel_values)
                print(f"{original_style}{res:<12}{Style.RESET_ALL} | {ch_values_console} | {min_psnr_styled} | {hint_styled:<36}")
            else:
                style = QualityHelper.get_style_for_hint(hint)
                hint_styled = f"{style}{hint}{Style.RESET_ALL}"
                min_psnr_styled = f"{style}{min_psnr:9.2f}{Style.RESET_ALL}"

                # Correctly handle channel names and formatting for grayscale images
                if 'L' in channels and len(channels) == 1:
                    channel_values = [f"{style}{ch_psnr.get('L', 0):>9.2f}{Style.RESET_ALL}"] # Right-align and format L channel
                else:
                    channel_values = [f"{style}{ch_psnr.get(c, 0):>9.2f}{Style.RESET_ALL}" for c in ['R', 'G', 'B', 'A'][:len(channels)]] # Right-align and format other channels

                ch_values_console = ' | '.join(channel_values)
                print(f"{QualityHelper.get_style_for_hint(hint)}{res:<12}{Style.RESET_ALL} | {ch_values_console} | {min_psnr_styled} | {hint_styled:<36}")


    @staticmethod
    def _print_simple_table(results: list):
        header = f"{Style.BRIGHT}{'Разрешение':<12} | {'PSNR (dB)':^10} | {'Качество':<36}{Style.RESET_ALL}"
        print(header)
        print(f"{'-'*12}-+-{'-'*10}-+-{'-'*36}")
        for res, psnr, hint in results:
            if hint.endswith("Оригинал"):
                original_style = STYLES['original']
                hint_styled = f"{original_style}Оригинал{Style.RESET_ALL}"
                psnr_styled = f"{original_style}{psnr:^10.2f}{Style.RESET_ALL}"
                print(f"{original_style}{res:<12}{Style.RESET_ALL} {Style.DIM}|{Style.NORMAL} {psnr_styled} {Style.DIM}|{Style.NORMAL} {hint_styled:<36}{Style.RESET_ALL}")
            else:
                style = QualityHelper.get_style_for_hint(hint)
                hint_styled = f"{style}{hint}{Style.RESET_ALL}"
                psnr_styled = f"{style}{psnr:^10.2f}{Style.RESET_ALL}"
                print(f"{QualityHelper.get_style_for_hint(hint)}{res:<12}{Style.RESET_ALL} {Style.DIM}|{Style.NORMAL} {psnr_styled} {Style.DIM}|{Style.NORMAL} {hint_styled:<36}") # Стиль разрешения для остальных строк


class CSVReporter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = open(self.output_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file, delimiter=CSV_SEPARATOR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write_header(self, analyze_channels: bool):
        self.writer.writerow(get_output_csv_header(analyze_channels))

    def write_results(self, filename: str, results: list, analyze_channels: bool):
        for i, result_item in enumerate(results):
            res = result_item[0]
            psnr_values = result_item[1]

            row = [filename if i == 0 else '', res]

            if analyze_channels:
                ch_psnr = psnr_values
                min_psnr = result_item[2]
                hint = result_item[3]
                if i == 0:
                    row.extend([""] * 6)
                elif 'L' in ch_psnr and len(ch_psnr) == 1:
                    row.extend([
                        f"{ch_psnr.get('L', float('inf')):.2f}",
                        "", "", "",
                        f"{min_psnr:.2f}",
                        QualityHelper.get_hint(min_psnr, for_csv=True)
                    ])
                else:
                    row.extend([
                        f"{ch_psnr.get('R', float('inf')):.2f}",
                        f"{ch_psnr.get('G', float('inf')):.2f}",
                        f"{ch_psnr.get('B', float('inf')):.2f}",
                        f"{ch_psnr.get('A', float('inf')):.2f}",
                        f"{min_psnr:.2f}",
                        QualityHelper.get_hint(min_psnr, for_csv=True)
                    ])
            else:
                psnr = psnr_values
                hint = result_item[2]
                if i == 0:
                    row.extend([""] * 2)
                else:
                    row.extend([
                        f"{psnr:.2f}",
                        QualityHelper.get_hint(psnr, for_csv=True)
                    ])
                if len(row) < 4:
                    row.extend([""] * (4 - len(row)))

            self.writer.writerow(row)


class QualityHelper:
    @staticmethod
    def get_hint(psnr: float, for_csv: bool = False) -> str:
        """Возвращает текстовую оценку качества"""
        for threshold in PSNR_QUALITY_THRESHOLDS:
            if psnr >= threshold:
                return QUALITY_HINTS[threshold]
        return QUALITY_HINTS[0]

    @staticmethod
    def get_style_for_hint(hint: str) -> str:
        """Возвращает ANSI-код стиля на основе текстовой подсказки о качестве"""
        if hint == QUALITY_HINTS[50]: return STYLES['good']
        if hint == QUALITY_HINTS[40]: return STYLES['ok']
        if hint == QUALITY_HINTS[30]: return STYLES['medium']
        return STYLES['bad']

    @staticmethod
    def get_style(psnr: float) -> str:
        """Возвращает ANSI-код стиля на основе значения PSNR (больше не используется для hint-ов)"""
        if psnr >= 50: return STYLES['good']
        if psnr >= 40: return STYLES['ok']
        if psnr >= 30: return STYLES['medium']
        return STYLES['bad']