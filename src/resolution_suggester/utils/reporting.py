# utils/reporting.py
import os
from ..i18n import _
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config import (
    QUALITY_LEVEL_HINTS_DESCRIPTIONS,
    QUALITY_METRIC_THRESHOLDS,
    QualityLevelHints,
    QualityMetrics,
    RICH_STYLES,
)

console = Console()


def add_progress_indicator(value: float, max_value: float = 100.0, width: int = 10) -> str:
    """
    Creates a visual progress indicator for the quality metric.

    Returns:
        Progress bar with Unicode indicator
    """
    if value == float('inf'):
        filled = width
    else:
        # Ограничиваем значение метрики для визуализации
        capped_value = min(value, max_value)
        filled = int((capped_value / max_value) * width)

    # Создаем индикатор с закрашенными и пустыми блоками
    return '█' * filled + '░' * (width - filled)


class ConsoleReporter:
    @staticmethod
    def print_file_header(
            file_path: str,
            metric_type: QualityMetrics
    ) -> None:
        """
        Выводит заголовок с именем файла.
        """
        filename = os.path.basename(file_path)
        full_title = f'{_("Analysis")} ({metric_type.value.upper()})'
        console.print()
        console.print(Panel(
            Text(filename, style=RICH_STYLES['filename']),
            title=full_title,
            border_style=RICH_STYLES['header'],
            expand=False
        ))

    @staticmethod
    def print_quality_table(
            results: list,
            analyze_channels: bool,
            channels: Optional[List[str]],
            metric_type: QualityMetrics
    ) -> None:
        """
        Выводит таблицу с результатами анализа качества.

        Args:
            results: Список результатов анализа
            analyze_channels: Флаг анализа по каналам
            channels: Список каналов изображения
            metric_type: Используемая метрика качества
        """
        if analyze_channels and channels:
            ConsoleReporter._print_channel_table(results, channels, metric_type)
        else:
            ConsoleReporter._print_simple_table(results, metric_type)

    @staticmethod
    def _print_channel_table(results: list, channels: List[str], metric_type: QualityMetrics) -> None:
        """Выводит таблицу результатов с разбивкой по каналам."""
        # Создаем таблицу
        table = Table(show_header=True, header_style="bold")

        # Добавляем колонки
        table.add_column(_("Resolution"), style="bold")
        for channel in channels:
            table.add_column(channel)
        table.add_column("Min")
        table.add_column(_("Quality"))
        table.add_column(_("Quality Bar"))

        # Определяем максимальное значение для шкалы в зависимости от метрики
        if metric_type == QualityMetrics.PSNR:
            max_scale = 50.0
        else:
            max_scale = 1.0

        # Добавляем данные
        for res, ch_values_dict, min_val, hint in results:
            # Определяем базовый стиль на основе качества
            if hint == QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.ORIGINAL]:
                base_style = RICH_STYLES['original']
                indicator_str = ""
            else:
                base_style = ConsoleReporter._get_style_for_hint(hint)
                indicator_str = add_progress_indicator(min_val, max_value=max_scale, width=15)

            # Создаём текст разрешения со стилем
            res_text = Text(res, style=base_style)

            # Создаём ячейки для каждого канала
            channel_cells = []
            for ch in channels:
                channel_val = ch_values_dict.get(ch, float('inf'))

                # Проверяем, является ли значение канала минимальным
                is_min = False
                if min_val != float('inf') and channel_val != float('inf'):
                    # Используем небольшую погрешность для сравнения с плавающей точкой
                    is_min = abs(channel_val - min_val) < 1e-6

                # Форматируем значение
                if channel_val == float('inf'):
                    formatted_value = "∞"
                else:
                    formatted_value = f"{channel_val:.2f}"

                # Применяем базовый стиль и делаем жирным, если это минимум
                style = f"{base_style} bold" if is_min else base_style
                channel_cells.append(Text(formatted_value, style=style))

            # Форматируем минимальное значение
            if min_val == float('inf'):
                min_value_text = Text("∞", style=base_style)
            else:
                min_value_text = Text(f"{min_val:.2f}", style=f"{base_style} bold")

            # Создаём текст подсказки качества
            hint_text = Text(hint, style=base_style)

            # Создаём индикатор со стилем строки
            indicator_text = Text(indicator_str, style=base_style)

            # Добавляем строку в таблицу
            table.add_row(
                res_text,
                *channel_cells,
                min_value_text,
                hint_text,
                indicator_text  # Используем Text вместо строки для наследования стиля
            )

        # Выводим таблицу
        console.print(table)

    @staticmethod
    def _print_simple_table(results: list, metric_type: QualityMetrics) -> None:
        """Выводит таблицу результатов без разбивки по каналам с визуализацией качества."""
        # Создаем таблицу
        table = Table(show_header=True, header_style="bold")

        # Добавляем колонки
        table.add_column((_("Resolution")), style="bold")
        table.add_column(f"{metric_type.value.upper()}")
        table.add_column(_("Quality"))
        table.add_column(_("Quality Bar"))

        # Определяем максимальное значение для шкалы в зависимости от метрики
        if metric_type == QualityMetrics.PSNR:
            max_scale = 50.0  # PSNR обычно до ~50 dB для высокого качества
        else:
            max_scale = 1.0  # SSIM, MS-SSIM и другие метрики обычно от 0 до 1

        # Добавляем данные
        for res, metric_value, hint in results:
            # Определяем стиль строки на основе качества
            if hint == QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.ORIGINAL]:
                row_style = RICH_STYLES['original']
                indicator = ""
            else:
                row_style = ConsoleReporter._get_style_for_hint(hint)
                indicator = add_progress_indicator(
                    metric_value,
                    max_value=max_scale,
                    width=15
                )

            # Форматируем значение метрики
            if metric_value == float('inf'):
                value_str = "∞"
            else:
                value_str = f"{metric_value:.2f}"

            # Добавляем строку в таблицу
            table.add_row(res, value_str, hint, indicator, style=row_style)

        # Выводим таблицу
        console.print(table)

    @staticmethod
    def _get_style_for_hint(hint: str) -> str:
        """
        Возвращает стиль Rich для текста на основе подсказки о качестве.

        Args:
            hint: Текстовая подсказка о качестве

        Returns:
            Строка стиля Rich
        """
        hint_values = {v: k for k, v in QUALITY_LEVEL_HINTS_DESCRIPTIONS.items()}

        if hint not in hint_values:
            return ""

        hint_level = hint_values[hint]

        if hint_level == QualityLevelHints.EXCELLENT:
            return RICH_STYLES['excellent']
        if hint_level == QualityLevelHints.VERY_GOOD:
            return RICH_STYLES['very_good']
        if hint_level == QualityLevelHints.GOOD:
            return RICH_STYLES['good']

        return RICH_STYLES['poor']


class QualityHelper:
    @staticmethod
    def get_hint(metric_value: float, metric_type: QualityMetrics) -> str:
        """
        Returns a text quality assessment for a given metric.
        """
        thresholds = QUALITY_METRIC_THRESHOLDS.get(metric_type)
        if thresholds is None:
            raise ValueError(f"Нет порогов качества для метрики: {metric_type}")

        sorted_levels = sorted(
            thresholds.keys(),
            key=lambda level: thresholds[level],
            reverse=False
        )

        for level in sorted_levels:
            if metric_value < thresholds[level]:
                return QUALITY_LEVEL_HINTS_DESCRIPTIONS[level]

        return QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.ORIGINAL]
