# utils/visualization.py

"""
Module for visualization of image quality analysis results.
"""
from ..i18n import _
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..config import QualityMetrics, LOGS_DIR, QUALITY_LEVEL_HINTS_DESCRIPTIONS, QualityLevelHints


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """
    Parses a resolution string in 'WxH' format and returns a tuple (width, height).

    Args:
        res_str: Resolution string in 'WxH' format

    Returns:
        Tuple (width, height)
    """
    match = re.match(r'(\d+)x(\d+)', res_str)
    if not match:
        raise ValueError(f"Invalid resolution format: {res_str}")
    return int(match.group(1)), int(match.group(2))

def get_megapixels(res_str: str) -> float:
    """
    Calculates the number of megapixels for a resolution.

    Args:
        res_str: Resolution string in 'WxH' format

    Returns:
        Number of megapixels
    """
    width, height = parse_resolution(res_str)
    return (width * height) / 1_000_000

def generate_quality_chart(
    results: list,
    output_path: str,
    title: str = _("Quality vs Resolution Relationship"),
    metric_type: QualityMetrics = QualityMetrics.PSNR,
    analyze_channels: bool = False,
    channels: Optional[List[str]] = None
) -> str:
    """
    Creates a chart showing quality vs resolution relationship.

    Args:
        results: List of analysis results
        output_path: Path to save the chart
        title: Chart title
        metric_type: Type of quality metric
        analyze_channels: Whether to analyze by channels
        channels: List of image channels

    Returns:
        Path to the saved chart
    """
    # Создаем директорию, если не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Создаем основную фигуру
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Стиль графика
    plt.style.use('ggplot')

    # Исключаем оригинал (с бесконечным качеством)
    filtered_results = [r for r in results if QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.ORIGINAL] not in r[-1]]

    # Извлекаем данные
    resolutions = [r[0] for r in filtered_results]

    # Преобразуем разрешения в мегапиксели для второй оси X
    megapixels = [get_megapixels(res) for res in resolutions]

    # Настраиваем основные цвета и маркеры
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    # Настраиваем основную ось X для мегапикселей
    ax1.set_xlabel('Megapixels')
    ax1.set_xlim(min(megapixels) * 0.9, max(megapixels) * 1.1)

    # Создаем верхнюю ось X для разрешений
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(megapixels)
    ax2.set_xticklabels(resolutions, rotation=45)
    ax2.set_xlabel(_("Resolution"))

    # Начальные значения для оси Y
    min_y, max_y = float('inf'), float('-inf')

    if analyze_channels and channels:
        # Рисуем графики для каждого канала
        for i, channel in enumerate(channels):
            channel_values = []
            for r in filtered_results:
                channel_val = r[1].get(channel, 0.0)
                # Обработка бесконечности
                if channel_val == float('inf'):
                    channel_val = np.nan
                channel_values.append(channel_val)
                if not np.isnan(channel_val):
                    min_y = min(min_y, channel_val)
                    max_y = max(max_y, channel_val)

            # Рисуем график для канала
            ax1.plot(
                megapixels,
                channel_values,
                label=f"{channel} {metric_type.value.upper()}",
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2
            )

        # Добавляем график минимальных значений
        min_values = []
        for r in filtered_results:
            min_val = r[2]
            if min_val == float('inf'):
                min_val = np.nan
            min_values.append(min_val)
            if not np.isnan(min_val):
                min_y = min(min_y, min_val)
                max_y = max(max_y, min_val)

        ax1.plot(
            megapixels,
            min_values,
            label=f"Min {metric_type.value.upper()}",
            marker='*',
            color='black',
            linewidth=1,
            linestyle='--'
        )
    else:
        # Один график для общей метрики
        metric_values = []
        for r in filtered_results:
            val = r[1]
            if val == float('inf'):
                val = np.nan
            metric_values.append(val)
            if not np.isnan(val):
                min_y = min(min_y, val)
                max_y = max(max_y, val)

        ax1.plot(
            megapixels,
            metric_values,
            label=f"{metric_type.value.upper()}",
            marker='o',
            color=colors[0],
            linewidth=1,
        )

    # Настройка оси Y
    if min_y == float('inf'):
        min_y = 0
    if max_y == float('-inf'):
        max_y = 100

    y_padding = (max_y - min_y) * 0.1
    ax1.set_ylim(min_y - y_padding, max_y + y_padding)
    ax1.set_ylabel(f"{metric_type.value.upper()}")

    # Добавляем порогвые линии для качества
    if metric_type == QualityMetrics.PSNR:
        quality_thresholds = [(30, "Good", "green"), (40, "Very Good", "blue")]
    elif metric_type in [QualityMetrics.SSIM, QualityMetrics.MS_SSIM]:
        quality_thresholds = [(0.75, "Good", "green"), (0.9, "Very Good", "blue")]
    else:
        quality_thresholds = [(0.7, "Good", "green"), (0.85, "Very Good", "blue")]

    for threshold, label, color in quality_thresholds:
        if min_y < threshold < max_y:
            ax1.axhline(y=threshold, color=color, linestyle='--', alpha=0.7)
            ax1.text(
                max(megapixels),
                threshold,
                f" {label}",
                va='center',
                color=color,
                fontweight='bold'
            )

    # Финальные настройки
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    plt.title(title)
    plt.tight_layout()

    # Сохраняем график
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def get_chart_filename(
    file_basename: str,
    metric_type: QualityMetrics,
    analyze_channels: bool = False
) -> str:
    """
    Creates a filename for the chart.

    Args:
        file_basename: Base name of the source file
        metric_type: Type of metric
        analyze_channels: Analysis by channels

    Returns:
        Path to the chart file
    """
    # Создаем директорию для графиков, если не существует
    charts_dir = LOGS_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Формируем имя файла
    chart_filename = (
        f"{file_basename}_"
        f"{metric_type.value.upper()}"
        f"{'_channels' if analyze_channels else ''}"
        f".png"
    )

    return str(charts_dir / chart_filename)