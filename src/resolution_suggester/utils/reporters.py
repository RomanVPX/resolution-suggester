# utils/reporters.py
import os
from abc import ABC, abstractmethod
import csv
import json
from datetime import datetime

from ..config import (
    CSV_SEPARATOR,
    InterpolationMethods,
    QualityMetrics,
    get_output_csv_header, LOGS_DIR
)
from .reporting import QualityHelper


class IReporter(ABC):
    """
    Базовый интерфейс репортёра. Обеспечивает единый метод write_results(),
    а также (при необходимости) context manager.
    """
    @abstractmethod
    def write_results(self, filename: str, results: list, analyze_channels: bool) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_csv_log_filename(args) -> str:
    """
    Генерирует имя CSV-файла с отметкой времени.
    """
    parts = [
        "tx_analysis",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        InterpolationMethods(args.interpolation).value.capitalize(),
        QualityMetrics(args.metric).upper()
    ]
    if args.channels:
        parts.append("ch")
    if args.ml:
        parts.append("ML")
    output_filename = "_".join(parts) + ".csv"

    return os.path.join(LOGS_DIR, output_filename)


def get_json_log_filename(args) -> str:
    """
    Генерирует имя JSON-файла с отметкой времени.
    По аналогии с get_csv_log_filename, но с расширением .json.
    """
    parts = [
        "tx_analysis",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        InterpolationMethods(args.interpolation).value.capitalize(),
        QualityMetrics(args.metric).upper()
    ]
    if args.channels:
        parts.append("ch")
    if args.ml:
        parts.append("ML")
    output_filename =  "_".join(parts) + ".json"

    return os.path.join(LOGS_DIR, output_filename)


class CSVReporter(IReporter):
    """
    Запись результатов в CSV-файл.
    """
    def __init__(self, output_path: str, metric_type: QualityMetrics):
        self.output_path = output_path
        self.metric_type = metric_type
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = open(self.output_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file, delimiter=CSV_SEPARATOR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write_header(self, analyze_channels: bool) -> None:
        """
        Записывает заголовок (первую строку).
        """
        self.writer.writerow(get_output_csv_header(analyze_channels, self.metric_type))

    def write_results(self, filename: str, results: list, analyze_channels: bool) -> None:
        """
        Записывает результаты в CSV.
        """
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
                            QualityHelper.get_hint(min_val, self.metric_type)
                        ])
                    else:
                        row.extend([
                            f"{ch_vals.get('R', float('inf')):.2f}",
                            f"{ch_vals.get('G', float('inf')):.2f}",
                            f"{ch_vals.get('B', float('inf')):.2f}",
                            f"{ch_vals.get('A', float('inf')):.2f}",
                            f"{min_val:.2f}",
                            QualityHelper.get_hint(min_val, self.metric_type)
                        ])
            else:
                metric_val = values
                if i == 0:
                    row.extend(["", ""])
                else:
                    row.extend([
                        f"{metric_val:.2f}",
                        QualityHelper.get_hint(metric_val, self.metric_type)
                    ])
                if len(row) < 4:
                    row.extend([""] * (4 - len(row)))

            self.writer.writerow(row)


class JSONReporter(IReporter):
    """
    Репорт в формате JSON. По аналогии с CSVReporter.
    При выходе из контекста (exit) сохраняет накопленные результаты.
    """
    def __init__(self, output_path: str, metric_type: QualityMetrics):
        self.output_path = output_path
        self.metric_type = metric_type
        self.file = None
        self.data = []

    def __enter__(self):
        self.file = open(self.output_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            json.dump(self.data, self.file, ensure_ascii=False, indent=2)
            self.file.close()

    def write_results(self, filename: str, results: list, analyze_channels: bool) -> None:
        file_entry = {
            "file": filename,
            "results": []
        }

        for i, row in enumerate(results):
            if analyze_channels:
                # row: (res_str, channel_dict, min_val, hint)
                resolution, channel_dict, min_val, hint = row
                file_entry["results"].append({
                    "resolution": resolution,
                    "channels": channel_dict,
                    "min_value": min_val,
                    "hint": hint
                })
            else:
                # row: (res_str, metric_value, hint)
                resolution, metric_value, hint = row
                file_entry["results"].append({
                    "resolution": resolution,
                    "value": metric_value,
                    "hint": hint
                })

        self.data.append(file_entry)
