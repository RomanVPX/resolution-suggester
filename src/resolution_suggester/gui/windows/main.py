"""Main window implementation."""

from typing import Callable, Dict, Optional

import dearpygui.dearpygui as dpg

from ...config import InterpolationMethods, QualityMetrics
from ...i18n import _
from ..analyzer import GUIAnalyzer


class MainWindow:
    """Main window of the application."""

    def __init__(self) -> None:
        """Initialize main window."""
        self.window_tag = "main_window"
        self.analyzer = GUIAnalyzer()

        # Create file dialogs
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_file_selected,
            tag="file_dialog",
            width=700,
            height=400,
            modal=True,
            default_path=".",
            file_count=0  # 0 для неограниченного количества файлов
        ):
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
            dpg.add_file_extension(".tga")
            dpg.add_file_extension(".exr")
            dpg.add_file_extension(".*")

        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self._on_directory_selected,
            tag="dir_dialog",
            width=700,
            height=400,
            modal=True,
            default_path="."
        ):
            dpg.add_file_extension("")

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_export_path_selected,
            tag="export_dialog",
            width=700,
            height=400,
            modal=True,
            default_path=".",
            default_filename="analysis_results"
        ):
            dpg.add_file_extension(".csv")
            dpg.add_file_extension(".json")

        # Create main window
        with dpg.window(
            label="Resolution Suggester",
            tag=self.window_tag,
            no_close=True,
            menubar=True,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_scrollbar=True,
            width=dpg.get_viewport_width(),
            height=dpg.get_viewport_height(),
            pos=(0, 0)
        ):
            self._create_menu()
            self._create_main_area()
            self._create_status_bar()

    def _create_menu(self) -> None:
        """Create the menu bar."""
        with dpg.menu_bar():
            # File menu
            with dpg.menu(label=_("File")):
                dpg.add_menu_item(
                    label=_("Open File..."),
                    callback=self._on_open_file
                )
                dpg.add_menu_item(
                    label=_("Open Directory..."),
                    callback=self._on_open_directory
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label=_("Export Results..."),
                    callback=self._on_export_results,
                    enabled=False,
                    tag="export_menu"
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label=_("Exit"),
                    callback=self._on_exit
                )

            # Settings menu
            with dpg.menu(label=_("Settings")):
                dpg.add_menu_item(
                    label=_("Language"),
                    enabled=False  # TODO: Implement language switching
                )

            # Help menu
            with dpg.menu(label=_("Help")):
                dpg.add_menu_item(
                    label=_("About"),
                    callback=self._on_about
                )

    def _create_main_area(self) -> None:
        """Create the main area of the window."""
        with dpg.group(horizontal=True):
            # Left panel - Settings
            with dpg.child_window(
                width=300,
                height=-30,  # -30 для статус бара
                no_scrollbar=True,
                border=False
            ):
                # Заголовок секции
                dpg.add_text(_("Analysis Settings"), color=(0, 135, 175))
                dpg.add_separator()

                # Метрика качества
                dpg.add_text(_("Quality Metric"))
                dpg.add_combo(
                    items=[
                        "PSNR - Peak Signal-to-Noise Ratio",
                        "SSIM - Structural Similarity Index",
                        "MS-SSIM - Multi-Scale SSIM",
                        "TDPR - Texture Detail Preservation Ratio",
                        "LPIPS - Learned Perceptual Image Patch Similarity"
                    ],
                    default_value="PSNR - Peak Signal-to-Noise Ratio",
                    width=-1,
                    tag="metric_combo"
                )
                dpg.add_spacer(height=8)

                # Метод интерполяции
                dpg.add_text(_("Interpolation Method"))
                dpg.add_combo(
                    items=[
                        "Bilinear",
                        "Bicubic",
                        "Mitchell-Netravali"
                    ],
                    default_value="Mitchell-Netravali",
                    width=-1,
                    tag="interpolation_combo"
                )
                dpg.add_spacer(height=8)

                # Анализ по каналам
                dpg.add_checkbox(
                    label=_("Analyze by channels"),
                    tag="analyze_channels"
                )
                dpg.add_spacer(height=16)

                # ML режим
                dpg.add_checkbox(
                    label=_("Use ML prediction (faster)"),
                    tag="use_ml"
                )
                dpg.add_spacer(height=16)

                # Разделитель перед кнопками
                dpg.add_separator()
                dpg.add_spacer(height=8)

                # Кнопки
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label=_("Open File"),
                        callback=self._on_open_file,
                        width=140
                    )
                    dpg.add_button(
                        label=_("Open Dir"),
                        callback=self._on_open_directory,
                        width=-1
                    )

            # Right panel - Results
            with dpg.child_window(
                width=-1,  # Заполнить оставшееся пространство
                height=-30,  # -30 для статус бара
                no_scrollbar=True,
                border=False
            ):
                # Заголовок секции
                dpg.add_text(_("Analysis Results"), color=(0, 135, 175))
                dpg.add_separator()
                dpg.add_spacer(height=4)

                # Таблица для результатов
                with dpg.table(
                    header_row=True,
                    resizable=True,
                    policy=dpg.mvTable_SizingStretchProp,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_innerH=True,
                    borders_outerV=True,
                    tag="results_table"
                ):
                    # Настройка столбцов
                    dpg.add_table_column(label=_("Resolution"))
                    dpg.add_table_column(label=_("Quality"))
                    dpg.add_table_column(label=_("Rating"))

                # Место для графика
                dpg.add_spacer(height=16)
                dpg.add_text(_("Quality vs Resolution Graph"), color=(0, 135, 175))
                dpg.add_separator()
                with dpg.group(tag="quality_plot"):
                    with dpg.plot(
                        label="Quality Graph",
                        height=300,
                        width=-1
                    ):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label=_("Scale"))
                        dpg.add_plot_axis(dpg.mvYAxis, label=_("Quality"))

    def _create_status_bar(self) -> None:
        """Create the status bar."""
        with dpg.group(horizontal=True, parent=self.window_tag):
            with dpg.group(horizontal=True):
                dpg.add_text(_("Ready"))
                dpg.add_separator()
                dpg.add_text("", tag="status_text")

    # === Event Handlers ===

    def _on_open_file(self) -> None:
        """Handle open file menu action."""
        dpg.show_item("file_dialog")

    def _on_open_directory(self) -> None:
        """Handle open directory menu action."""
        dpg.show_item("dir_dialog")

    def _get_analysis_settings(self) -> tuple[QualityMetrics, InterpolationMethods, bool, bool]:
        """Get analysis settings from GUI controls."""
        metric_text = dpg.get_value("metric_combo")
        metric = {
            "PSNR - Peak Signal-to-Noise Ratio": QualityMetrics.PSNR,
            "SSIM - Structural Similarity Index": QualityMetrics.SSIM,
            "MS-SSIM - Multi-Scale SSIM": QualityMetrics.MS_SSIM,
            "TDPR - Texture Detail Preservation Ratio": QualityMetrics.TDPR,
            "LPIPS - Learned Perceptual Image Patch Similarity": QualityMetrics.LPIPS
        }[metric_text]

        interpolation_text = dpg.get_value("interpolation_combo")
        interpolation = {
            "Bilinear": InterpolationMethods.BILINEAR,
            "Bicubic": InterpolationMethods.BICUBIC,
            "Mitchell-Netravali": InterpolationMethods.MITCHELL
        }[interpolation_text]

        analyze_channels = dpg.get_value("analyze_channels")
        use_ml = dpg.get_value("use_ml")

        return metric, interpolation, analyze_channels, use_ml

    def _on_file_selected(self, sender: int, app_data: dict) -> None:
        """
        Handle file selection from dialog.

        Args:
            sender: ID of the sender widget
            app_data: Dictionary containing selected file paths
        """
        print("File selection data:", app_data)  # DEBUG

        if not app_data or not app_data.get("selections"):
            dpg.set_value("status_text", _("Error: No files selected"))
            return

        # Собираем пути к файлам
        paths = list(app_data["selections"].values())

        # Получаем настройки анализа
        metric, interpolation, analyze_channels, use_ml = self._get_analysis_settings()

        # Запускаем анализ
        self.analyzer.analyze_paths(
            paths,
            metric=metric,
            interpolation=interpolation,
            analyze_channels=analyze_channels,
            use_ml=use_ml
        )

    def _on_directory_selected(self, sender: int, app_data: dict) -> None:
        """
        Handle directory selection from dialog.

        Args:
            sender: ID of the sender widget
            app_data: Dictionary containing selected directory path
        """
        print("Directory selection data:", app_data)  # DEBUG

        if not app_data or "file_path_name" not in app_data:
            dpg.set_value("status_text", _("Error: No directory selected"))
            return

        # Получаем настройки анализа
        metric, interpolation, analyze_channels, use_ml = self._get_analysis_settings()

        # Запускаем анализ
        self.analyzer.analyze_paths(
            [app_data["file_path_name"]],
            metric=metric,
            interpolation=interpolation,
            analyze_channels=analyze_channels,
            use_ml=use_ml
        )

    def _on_export_results(self) -> None:
        """Handle export results menu action."""
        dpg.show_item("export_dialog")

    def _on_export_path_selected(self, sender: int, app_data: dict) -> None:
        """
        Handle export path selection.

        Args:
            sender: ID of the sender widget
            app_data: Dictionary containing selected file path
        """
        if not app_data or "file_path_name" not in app_data:
            dpg.set_value("status_text", _("Error: No path selected"))
            return

        export_path = app_data["file_path_name"]
        self.analyzer.export_results(export_path)

    def _on_about(self) -> None:
        """Show about dialog."""
        with dpg.window(
            label=_("About Resolution Suggester"),
            modal=True,
            show=True,
            width=400,
            height=200
        ):
            dpg.add_text(_("Resolution Suggester"))
            dpg.add_text(_("Version: 0.1.0"))
            dpg.add_separator()
            dpg.add_text(_("A tool for analyzing texture quality at different resolutions"))
            dpg.add_text(_("and suggesting optimal downsizing parameters."))

    def _on_exit(self) -> None:
        """Handle exit menu action."""
        dpg.stop_dearpygui()
