from __future__ import annotations

import os
import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from gui import MainWindow
from gui.settings_dialogs import load_ui_scale


def main() -> int:
    QtCore.QCoreApplication.setOrganizationName("OpenAI Engineering Tools")
    QtCore.QCoreApplication.setApplicationName("C-PHY S-Parameter Viewer & Analyzer")
    ui_scale = load_ui_scale(QtCore.QSettings())
    if ui_scale != 100:
        os.environ["QT_SCALE_FACTOR"] = str(ui_scale / 100)
    pg.setConfigOptions(antialias=True, foreground="#d8e0e8", background="#111820")
    application = QtWidgets.QApplication(sys.argv)
    application.setStyle("Fusion")
    icon_path = Path(__file__).resolve().parent / "assets" / "cphy-viewer.ico"
    application.setWindowIcon(QtGui.QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
