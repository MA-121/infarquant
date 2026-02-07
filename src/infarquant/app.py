"""GUI launcher for InfarQuant."""

from __future__ import annotations

import sys

from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication

from .ui import APP_STYLE_SHEET, MainWindow


def main() -> None:
    """Launch the application main window."""
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#F6F7FB"))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.AlternateBase, QColor("#F3F4F6"))
    palette.setColor(QPalette.Text, QColor("#111827"))
    palette.setColor(QPalette.ButtonText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLE_SHEET)

    window = MainWindow()
    window.resize(1100, 780)
    window.show()
    sys.exit(app.exec_())


def launch_gui() -> None:
    """Backward-compatible launcher alias."""
    main()
