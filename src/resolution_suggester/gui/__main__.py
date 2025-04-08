"""Entry point for the GUI application."""

import sys
from pathlib import Path

# Добавляем родительскую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from resolution_suggester.gui import ResolutionSuggesterGUI
from resolution_suggester.i18n import setup_localization


def main() -> None:
    """Run the GUI application."""
    # Инициализируем локализацию (пока с автоопределением)
    setup_localization('auto')

    # Создаем и запускаем GUI
    app = ResolutionSuggesterGUI()
    app.run()


if __name__ == "__main__":
    main()
