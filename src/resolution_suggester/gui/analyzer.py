"""GUI analyzer module for Resolution Suggester."""

import argparse
from typing import Optional, List, Dict, Any
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image

from resolution_suggester.core.image_analyzer import ImageAnalyzer
from resolution_suggester.core.image_loader import load_image
from resolution_suggester.core.metrics import compute_resolutions
from resolution_suggester.config import QualityMetrics, InterpolationMethods
from resolution_suggester.i18n import _
from resolution_suggester.utils.reporting import QualityHelper
from resolution_suggester.main import parse_and_validate_arguments, get_file_list, run_image_analysis
def create_cli_args(paths: List[str], metric: QualityMetrics,
                   interpolation: InterpolationMethods,
                   analyze_channels: bool = False,
                   use_ml: bool = False) -> argparse.Namespace:
    """Create argparse.Namespace for CLI functionality."""
    return argparse.Namespace(
        paths=paths,
        metric=metric.value,
        interpolation=interpolation.value,
        channels=analyze_channels,
        ml=use_ml,
        no_gpu=False,
        lpips_net='alex',
        min_size=16,
        threads=1,
        chart=False,
        csv_output=False,
        json_output=False,
        no_parallel=True,  # GUI всегда в однопотоке
        compare_ml=False,
        generate_dataset=False,
        train_ml=False,
        save_im_down=False,
        save_im_up=False
    )

def create_analyzer_args(
    metric: QualityMetrics,
    interpolation: InterpolationMethods,
    analyze_channels: bool,
    use_ml: bool
) -> argparse.Namespace:
    """Create argparse.Namespace for ImageAnalyzer."""
    return argparse.Namespace(
        metric=metric,
        interpolation=interpolation,
        channels=analyze_channels,
        ml=use_ml,
        no_gpu=False,
        lpips_net='alex',
        min_size=16,
        save_im_down=False,
        save_im_up=False,
        threads=1,
        chart=False
    )

class GUIAnalyzer:
    """GUI analyzer class that bridges GUI and core functionality."""

    def __init__(self) -> None:
        """Initialize analyzer."""
        self.current_file: Optional[Path] = None
        self.current_directory: Optional[Path] = None
        self.results: List[Dict[str, Any]] = []

    def analyze_file(
        self,
        file_path: str,
        metric: QualityMetrics = QualityMetrics.PSNR,
        interpolation: InterpolationMethods = InterpolationMethods.MITCHELL,
        analyze_channels: bool = False,
        use_ml: bool = False
    ) -> None:
        """
        Analyze single file and update GUI.

        Args:
            file_path: Path to image file
            metric: Quality metric to use
            interpolation: Interpolation method
            analyze_channels: Whether to analyze individual channels
            use_ml: Whether to use ML prediction
        """
        self.current_file = Path(file_path)
        self.current_directory = None
        self.results.clear()

        # Обновляем статус
        dpg.set_value("status_text", _("Loading image..."))

        import os

        # Проверяем, что это файл
        if not os.path.isfile(file_path):
            dpg.set_value("status_text", _("Error: Not a file"))
            return

        # Проверяем расширение файла
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tga', '.exr'}
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in valid_extensions:
            dpg.set_value("status_text", _("Error: Unsupported file format"))
            return

        try:
            # Создаем аргументы и анализатор
            args = create_analyzer_args(metric, interpolation, analyze_channels, use_ml)
            analyzer = ImageAnalyzer(args)

            # Анализируем файл
            results, meta = analyzer.analyze_file(file_path)
            if not results or not meta:
                dpg.set_value("status_text", _("Error analyzing file"))
                return
        except Exception as e:
            dpg.set_value("status_text", f"Error: {str(e)}")
            return

        # Очищаем таблицу
        dpg.delete_item("results_table", children_only=True)
        dpg.add_table_column(parent="results_table", label=_("Resolution"))
        dpg.add_table_column(parent="results_table", label=_("Quality"))
        dpg.add_table_column(parent="results_table", label=_("Rating"))

        # Подготавливаем данные для графика
        x_data: List[float] = []
        y_data: List[float] = []

        # Добавляем результаты в таблицу и график
        for result in results:
            resolution, quality, rating = result[:3]  # Распаковываем результат
            width, height = map(int, resolution.split('x'))

            # Get original dimensions from first result (original resolution)
            if not x_data:  # Если это первый результат
                orig_width, orig_height = width, height

            scale_factor = (width / orig_width + height / orig_height) / 2
            x_data.append(scale_factor)

            # Если анализ по каналам
            if analyze_channels and isinstance(quality, dict):
                min_quality = min(quality.values())
                y_data.append(min_quality)

                # Добавляем строки для каждого канала
                for channel, channel_quality in quality.items():
                    with dpg.table_row(parent="results_table"):
                        dpg.add_text(f"{resolution} ({channel})")
                        dpg.add_text(f"{channel_quality:.3f}")
                        dpg.add_text(rating)
            else:
                y_data.append(quality)
                with dpg.table_row(parent="results_table"):
                    dpg.add_text(resolution)
                    dpg.add_text(f"{quality:.3f}")
                    dpg.add_text(rating)

        # Обновляем график
        self._update_plot(x_data, y_data)

        # Активируем экспорт
        dpg.configure_item("export_menu", enabled=True)

        # Обновляем статус
        dpg.set_value("status_text", _("Analysis complete"))

    def _update_plot(self, x_data: List[float], y_data: List[float]) -> None:
        """
        Update quality plot with new data.

        Args:
            x_data: List of scale factors
            y_data: List of quality values
        """
        # Удаляем предыдущую серию, если она есть
        if dpg.does_item_exist("quality_line"):
            dpg.delete_item("quality_line")

        # Получаем оси графика
        plot = dpg.get_item_children("quality_plot", 1)[0]
        y_axis = dpg.get_item_children(plot, 1)[1]  # Первая ось - X, вторая - Y

        # Добавляем новую линию
        dpg.add_line_series(
            x_data,
            y_data,
            label=_("Quality"),
            parent=y_axis,
            tag="quality_line"
        )

        # Обновляем диапазоны
        dpg.set_axis_limits(dpg.get_item_children(plot, 1)[0], 0, 1)  # X axis
        if y_data:
            dpg.set_axis_limits(y_axis, min(y_data) * 0.9, max(y_data) * 1.1)  # Y axis

    def analyze_paths(
        self,
        paths: List[str],
        metric: QualityMetrics = QualityMetrics.PSNR,
        interpolation: InterpolationMethods = InterpolationMethods.MITCHELL,
        analyze_channels: bool = False,
        use_ml: bool = False
    ) -> None:
        """
        Analyze multiple files and/or directories.

        Args:
            paths: List of paths to files and/or directories
            metric: Quality metric to use
            interpolation: Interpolation method
            analyze_channels: Whether to analyze individual channels
            use_ml: Whether to use ML prediction
        """

        # Запускаем процесс через CLI функции
        try:
            for file_path in paths:
                # Анализируем каждый файл отдельно через GUI анализатор
                self.analyze_file(
                    file_path,
                    metric=metric,
                    interpolation=interpolation,
                    analyze_channels=analyze_channels,
                    use_ml=use_ml
                )
            dpg.set_value("status_text", _("Analysis complete"))
        except Exception as e:
            dpg.set_value("status_text", f"Error: {str(e)}")

    def export_results(self, file_path: str) -> None:
        """
        Export analysis results.

        Args:
            file_path: Path to export file (CSV or JSON)
        """
        # TODO: Implement results export
        pass
