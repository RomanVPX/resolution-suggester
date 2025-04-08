"""Main GUI application module for Resolution Suggester."""

from typing import Optional

import dearpygui.dearpygui as dpg

from ..i18n import _
from .windows.main import MainWindow


class ResolutionSuggesterGUI:
    """Main GUI application class."""

    def __init__(self) -> None:
        """Initialize the GUI application."""
        # Create DPG context
        dpg.create_context()

        # Initialize window
        self.main_window: Optional[MainWindow] = None

        # Configure initial window size
        self.viewport_width = 1200
        self.viewport_height = 800

        # Setup interface
        self._setup_interface()

    def _setup_interface(self) -> None:
        """Setup the main interface."""
        # Setup theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, [0, 135, 175])
                dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 119, 200])
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3)

        dpg.bind_theme(global_theme)

        # Create viewport
        dpg.create_viewport(
            title="Resolution Suggester",
            width=self.viewport_width,
            height=self.viewport_height,
            min_width=800,
            min_height=600,
            small_icon="path/to/icon.ico",  # TODO: Add icon
            large_icon="path/to/icon.ico"   # TODO: Add icon
        )

        dpg.set_viewport_resize_callback(self._on_viewport_resize)
        dpg.setup_dearpygui()

        # Create main window
        self.main_window = MainWindow()

    def _on_viewport_resize(self) -> None:
        """Handle viewport resize event."""
        if self.main_window is None:
            return

        self.viewport_width = dpg.get_viewport_width()
        self.viewport_height = dpg.get_viewport_height()

        # Обновляем размер основного окна
        dpg.configure_item(
            self.main_window.window_tag,
            width=self.viewport_width,
            height=self.viewport_height
        )

    def run(self) -> None:
        """Run the application."""
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
