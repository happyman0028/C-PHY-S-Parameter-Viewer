from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from gui import MainWindow


def main() -> int:
    QtCore.QCoreApplication.setOrganizationName("OpenAI Engineering Tools")
    QtCore.QCoreApplication.setApplicationName("C-PHY S-Parameter Viewer & Analyzer")
    pg.setConfigOptions(antialias=True, foreground="#d8e0e8", background="#111820")
    application = QtWidgets.QApplication(sys.argv)
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
