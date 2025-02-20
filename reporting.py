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
    QualityMetric
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
    def print_quality_table(
        results: list,
        analyze_channels: bool,
        channels: Optional[List[str]],
        metric: str
    ):
        if analyze_channels and channels:
            ConsoleReporter._print_channel_table(results, channels)
        else:
            ConsoleReporter._print_simple_table(results, metric)

    @staticmethod
    def _print_channel_table(results: list, channels: List[str]):
        channel_headers = [c.center(9) for c in channels]
        header = (
            f"{Style.BRIGHT}{'Разрешение':<12} | "
            + " | ".join(channel_headers)
            + f" | {'Min':^9} | {'Качество (min)':<36}{Style.RESET_ALL}"
        )
        print(header)

        separator_parts = [
            "-" * 12,
            *["-" * 9 for _ in channels],
            "-" * 9,
            "-" * 32
        ]
        print("-+-".join(separator_parts))

        for res, ch_values_dict, min_val, hint in results:
            if hint.endswith("Оригинал"):
                original_style = STYLES['original']
                res_str = f"{original_style}{res:<12}{Style.RESET_ALL}"
                min_str = f"{original_style}{min_val:9.2f}{Style.RESET_ALL}"
                hint_str = f"{original_style}Оригинал{Style.RESET_ALL}"

                channel_strs = []
                for c in channels:
                    channel_val = ch_values_dict.get(c, float('inf'))
                    channel_strs.append(f"{original_style}{channel_val:>9.2f}{Style.RESET_ALL}")

                print(
                    f"{res_str} | "
                    + " | ".join(channel_strs)
                    + f" | {min_str} | {hint_str:<36}"
                )
            else:
                style = QualityHelper.get_style_for_hint(hint)
                res_str = f"{style}{res:<12}{Style.RESET_ALL}"
                min_str = f"{style}{min_val:9.2f}{Style.RESET_ALL}"
                hint_str = f"{style}{hint}{Style.RESET_ALL}"

                channel_strs = []
                for c in channels:
                    channel_val = ch_values_dict.get(c, 0.0)
                    channel_strs.append(f"{style}{channel_val:>9.2f}{Style.RESET_ALL}")

                print(
                    f"{res_str} | "
                    + " | ".join(channel_strs)
                    + f" | {min_str} | {hint_str:<36}"
                )

    @staticmethod
    def _print_simple_table(results: list, metric: str):

        metric_header = str(metric.upper())

        header = f"{Style.BRIGHT}{'Разрешение':<12} | {metric_header:^10} | {'Качество':<36}{Style.RESET_ALL}"
        print(header)
        print(f"{'-'*12}-+-{'-'*10}-+-{'-'*36}")

        for res, metric_value, hint in results:
            if hint.endswith("Оригинал"):
                original_style = STYLES['original']
                hint_styled = f"{original_style}Оригинал{Style.RESET_ALL}"
                val_styled = f"{original_style}{metric_value:^10.2f}{Style.RESET_ALL}"
                print(
                    f"{original_style}{res:<12}{Style.RESET_ALL} "
                    f"{Style.DIM}|{Style.NORMAL} {val_styled} {Style.DIM}|{Style.NORMAL} {hint_styled:<36}{Style.RESET_ALL}"
                )
            else:
                style = QualityHelper.get_style_for_hint(hint)
                val_styled = f"{style}{metric_value:^10.2f}{Style.RESET_ALL}"
                hint_styled = f"{style}{hint}{Style.RESET_ALL}"
                print(
                    f"{res:<12} {Style.DIM}|{Style.NORMAL} {val_styled} {Style.DIM}|{Style.NORMAL} {hint_styled:<36}"
                )

class CSVReporter:
    def __init__(self, output_path: str, metric: str):
        self.output_path = output_path
        self.metric = metric  # <-- сохраняем тип метрики
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
        # Передаём metric в get_output_csv_header
        self.writer.writerow(get_output_csv_header(analyze_channels, self.metric))

    def write_results(self, filename: str, results: list, analyze_channels: bool):
        for i, result_item in enumerate(results):
            res = result_item[0]
            values = result_item[1]

            row = [filename if i == 0 else '', res]

            if analyze_channels:
                ch_vals = values  # dict канал->значение
                min_val = result_item[2]
                if i == 0:
                    # Пустая строка для «Оригинал»
                    row.extend([""] * 6)
                else:
                    if 'L' in ch_vals and len(ch_vals) == 1:
                        # Одноканальное grayscale
                        row.extend([
                            f"{ch_vals.get('L', float('inf')):.2f}",
                            "", "", "",
                            f"{min_val:.2f}",
                            QualityHelper.get_hint(min_val)
                        ])
                    else:
                        row.extend([
                            f"{ch_vals.get('R', float('inf')):.2f}",
                            f"{ch_vals.get('G', float('inf')):.2f}",
                            f"{ch_vals.get('B', float('inf')):.2f}",
                            f"{ch_vals.get('A', float('inf')):.2f}",
                            f"{min_val:.2f}",
                            QualityHelper.get_hint(min_val)
                        ])
            else:
                metric_val = values
                if i == 0:
                    row.extend(["", ""])
                else:
                    row.extend([
                        f"{metric_val:.2f}",
                        QualityHelper.get_hint(metric_val)
                    ])
                if len(row) < 4:
                    row.extend([""] * (4 - len(row)))

            self.writer.writerow(row)

class QualityHelper:
    @staticmethod
    def get_hint(metric_value: float, metric_type: QualityMetric = QualityMetric.PSNR) -> str:
        """Возвращает текстовую оценку качества для заданной метрики"""
        thresholds = METRIC_QUALITY_THRESHOLDS.get(metric_type)
        if thresholds is None:
            raise ValueError(f"Нет порогов качества для метрики: {metric_type}")

        sorted_levels = sorted(
            thresholds.keys(),
            key=lambda level: thresholds[level],
            reverse=True
        )

        for level in sorted_levels:
            if metric_value >= thresholds[level]:
                return QUALITY_LEVEL_DESCRIPTIONS[level]

        return QUALITY_LEVEL_DESCRIPTIONS[QualityLevel.NOTICEABLE_LOSS]

    @staticmethod
    def get_style_for_hint(hint: str) -> str:
        """Возвращает ANSI-код стиля на основе текстовой подсказки о качестве"""
        if hint == QUALITY_LEVEL_DESCRIPTIONS[QualityLevel.EXCELLENT]:
            return STYLES['good']
        if hint == QUALITY_LEVEL_DESCRIPTIONS[QualityLevel.VERY_GOOD]:
            return STYLES['ok']
        if hint == QUALITY_LEVEL_DESCRIPTIONS[QualityLevel.GOOD]:
            return STYLES['medium']
        return STYLES['bad']
