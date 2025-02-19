# reporting.py
from datetime import datetime
import csv
import os
from typing import List, Optional
from colorama import Style

from config import (
    STYLES,
    CSV_SEPARATOR,
    QUALITY_LEVEL_DESCRIPTIONS,
    METRIC_QUALITY_THRESHOLDS,
    get_output_csv_header,
    QualityLevel,
    QualityMetric # Import QualityMetric
)

def generate_csv_filename(metric: str, interpolation: str) -> str:
    """Генерация имени CSV файла с временной меткой"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tx_analysis_{timestamp}_{interpolation}_{metric}.csv"

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
        # Сформируем динамический заголовок по именам каналов
        channel_headers = [c.center(9) for c in channels]
        header = (
            f"{Style.BRIGHT}{'Разрешение':<12} | " +
            " | ".join(channel_headers) +
            f" | {'Min':^9} | {'Качество (min)':<36}{Style.RESET_ALL}"
        )
        print(header)

        separator_parts = [
            "-" * 12,
            *["-" * 9 for _ in channels],
            "-" * 9,
            "-" * 32
        ]
        print("-+-".join(separator_parts))

        for res, ch_psnr_dict, min_psnr, hint in results:
            # Проверка «Оригинала»
            if hint.endswith("Оригинал"):
                original_style = STYLES['original']
                res_str = f"{original_style}{res:<12}{Style.RESET_ALL}"
                min_str = f"{original_style}{min_psnr:9.2f}{Style.RESET_ALL}"
                hint_str = f"{original_style}Оригинал{Style.RESET_ALL}"

                channel_strs = []
                for c in channels:
                    channel_val = ch_psnr_dict.get(c, float('inf'))
                    channel_strs.append(f"{original_style}{channel_val:>9.2f}{Style.RESET_ALL}")

                print(
                    f"{res_str} | " +
                    " | ".join(channel_strs) +
                    f" | {min_str} | {hint_str:<36}"
                )
            else:
                style = QualityHelper.get_style_for_hint(hint)
                res_str = f"{style}{res:<12}{Style.RESET_ALL}"
                min_str = f"{style}{min_psnr:9.2f}{Style.RESET_ALL}"
                hint_str = f"{style}{hint}{Style.RESET_ALL}"

                channel_strs = []
                for c in channels:
                    channel_val = ch_psnr_dict.get(c, 0.0)
                    channel_strs.append(f"{style}{channel_val:>9.2f}{Style.RESET_ALL}")

                print(
                    f"{res_str} | " +
                    " | ".join(channel_strs) +
                    f" | {min_str} | {hint_str:<36}"
                )

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
                print(
                    f"{original_style}{res:<12}{Style.RESET_ALL} "
                    f"{Style.DIM}|{Style.NORMAL} {psnr_styled} {Style.DIM}|{Style.NORMAL} {hint_styled:<36}{Style.RESET_ALL}"
                )
            else:
                style = QualityHelper.get_style_for_hint(hint)
                hint_styled = f"{style}{hint}{Style.RESET_ALL}"
                psnr_styled = f"{style}{psnr:^10.2f}{Style.RESET_ALL}"
                print(
                    f"{res:<12} {Style.DIM}|{Style.NORMAL} {psnr_styled} {Style.DIM}|{Style.NORMAL} {hint_styled:<36}"
                )

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
                else:
                    # Заполняем по ключам 'R', 'G', 'B', 'A' или 'L' и т.д.
                    # При желании найти единую схему: см. get_output_csv_header()
                    if 'L' in ch_psnr and len(ch_psnr) == 1:
                        # Градации серого
                        row.extend([
                            f"{ch_psnr.get('L', float('inf')):.2f}",
                            "", "", "",
                            f"{min_psnr:.2f}",
                            QualityHelper.get_hint(min_psnr, for_csv=True) # No metric needed for CSV output?
                        ])
                    else:
                        row.extend([
                            f"{ch_psnr.get('R', float('inf')):.2f}",
                            f"{ch_psnr.get('G', float('inf')):.2f}",
                            f"{ch_psnr.get('B', float('inf')):.2f}",
                            f"{ch_psnr.get('A', float('inf')):.2f}",
                            f"{min_psnr:.2f}",
                            QualityHelper.get_hint(min_psnr, for_csv=True) # No metric needed for CSV output?
                        ])
            else:
                psnr = psnr_values
                hint = result_item[2]
                if i == 0:
                    row.extend([""] * 2)
                else:
                    row.extend([
                        f"{psnr:.2f}",
                        QualityHelper.get_hint(psnr, for_csv=True) # No metric needed for CSV output?
                    ])
                if len(row) < 4:
                    row.extend([""] * (4 - len(row)))

            self.writer.writerow(row)

class QualityHelper:
    @staticmethod
    def get_hint(metric_value: float, metric_type: QualityMetric = QualityMetric.PSNR, for_csv: bool = False) -> str:
        """Возвращает текстовую оценку качества для заданной метрики"""
        thresholds = METRIC_QUALITY_THRESHOLDS.get(metric_type)
        if thresholds is None:
            raise ValueError(f"Нет порогов качества для метрики: {metric_type}")

        sorted_levels = sorted(thresholds.keys(), key=lambda level: thresholds[level], reverse=True) # Sort levels by threshold DESC

        for level in sorted_levels:
            if metric_type == QualityMetric.PSNR and metric_value >= thresholds[level]:
                return QUALITY_LEVEL_DESCRIPTIONS[level]
            elif metric_type == QualityMetric.SSIM and metric_value >= thresholds[level]:
                return QUALITY_LEVEL_DESCRIPTIONS[level]

        return QUALITY_LEVEL_DESCRIPTIONS[QualityLevel.NOTICEABLE_LOSS]


    @staticmethod
    def get_style_for_hint(hint: str) -> str:
        """Возвращает ANSI-код стиля на основе текстовой подсказки о качестве"""
        if hint == QUALITY_LEVEL_DESCRIPTIONS[QualityLevel.EXCELLENT]: return STYLES['good']
        if hint == QUALITY_LEVEL_DESCRIPTIONS[QualityLevel.VERY_GOOD]: return STYLES['ok']
        if hint == QUALITY_LEVEL_DESCRIPTIONS[QualityLevel.GOOD]: return STYLES['medium']
        return STYLES['bad']

    # @staticmethod
    # def get_style(psnr: float) -> str:
    #     """Возвращает ANSI-код стиля на основе значения PSNR (больше не используется для hint-ов)"""
    #     thresholds = METRIC_QUALITY_THRESHOLDS[QualityMetric.PSNR] # Стиль пока привязан к PSNR уровням
    #     if psnr >= thresholds[QualityLevel.EXCELLENT]: return STYLES['good']
    #     if psnr >= thresholds[QualityLevel.VERY_GOOD]: return STYLES['ok']
    #     if psnr >= thresholds[QualityLevel.GOOD]: return STYLES['medium']
    #     return STYLES['bad']
