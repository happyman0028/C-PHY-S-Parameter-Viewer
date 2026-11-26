from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from core.touchstone_loader import TouchstoneError, load_touchstone

if TYPE_CHECKING:
    from gui.main_window import MainWindow


class GuidedDemo(QtWidgets.QDockWidget):
    EXAMPLES = (
        ("cphy_demo_6port.s6p", "Sxx21_1R"),
        ("diff_demo_4port.s4p", "Sdd21"),
        ("se_demo_2port.s2p", "S21"),
    )

    def __init__(self, window: MainWindow) -> None:
        super().__init__("Guided Tour", window)
        self.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.window = window
        self.step_index = 0
        self.dataset_ids: dict[str, str] = {}
        self.steps = (
            ("Import and inspect datasets",
             "Load the included synthetic S2P, S4P and S6P files. In Datasets, double-click a row "
             "to inspect points, port mapping and reference impedance.",
             "Load example files", self.load_examples),
            ("Add and compare traces",
             "Add one example trace per file. In Add Trace, choose SE, DIFF, or C-PHY as supported "
             "by the dataset. Use multiple for Ctrl/Shift selection and Plot / Traces for visibility, color and style.",
             "Add example traces", self.add_example_traces),
            ("Set a frequency range",
             "Apply a 1–10 GHz range. Try the mouse wheel to zoom, Shift + drag to pan, "
             "FIT to return to the applied view, and Autoscale to see the full sweep.",
             "Apply 1–10 GHz", self.apply_example_range),
            ("Read values with a marker",
             "Place a marker at 5 GHz and open Marker Info. Drag its line or enter a frequency "
             "to read the value of each visible trace. Click R in the Legend to edit report labels.",
             "Show 5 GHz marker", self.show_example_marker),
            ("Export and save",
             "Export Plot PNG or use Copy on the plot to copy it. Export Trace CSV saves trace values. "
             "Save Workspace lets you reopen the datasets, curves and markers later.",
             "Export demo PNG…", self.export_example),
        )

        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        self.heading = QtWidgets.QLabel()
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.heading)
        layout.addWidget(self.description)
        buttons = QtWidgets.QHBoxLayout()
        self.try_button = QtWidgets.QPushButton()
        self.try_button.setObjectName("accentButton")
        self.back_button = QtWidgets.QPushButton("Back")
        self.next_button = QtWidgets.QPushButton()
        buttons.addWidget(self.try_button)
        buttons.addStretch()
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)
        layout.addLayout(buttons)
        self.setWidget(panel)
        self.try_button.clicked.connect(self.try_step)
        self.back_button.clicked.connect(self.previous_step)
        self.next_button.clicked.connect(self.next_step)
        self.update_step()

    def update_step(self) -> None:
        title, description, action, _ = self.steps[self.step_index]
        self.heading.setText(f"<b>{self.step_index + 1}/{len(self.steps)} · {title}</b>")
        self.description.setText(description)
        self.try_button.setText(action)
        self.back_button.setEnabled(self.step_index > 0)
        self.next_button.setText("Finish" if self.step_index + 1 == len(self.steps) else "Next")

    def previous_step(self) -> None:
        self.step_index -= 1
        self.update_step()

    def next_step(self) -> None:
        if self.step_index + 1 == len(self.steps):
            self.window.close()
            return
        self.step_index += 1
        self.update_step()

    def try_step(self) -> None:
        try:
            self.steps[self.step_index][3]()
        except TouchstoneError as exc:
            QtWidgets.QMessageBox.warning(self, "Example load failed", str(exc))

    def load_examples(self) -> None:
        folder = Path(__file__).resolve().parents[1] / "examples"
        for filename, _ in self.EXAMPLES:
            if self.dataset_ids.get(filename) not in self.window.datasets:
                dataset = load_touchstone(folder / filename)
                self.dataset_ids[filename] = dataset.id
                self.window._dataset_loaded(dataset)
        self.window.control_tabs.setCurrentIndex(0)

    def add_example_traces(self) -> None:
        self.load_examples()
        if self.window.active_plot() is None:
            self.window.add_plot()
        requests = [(self.dataset_ids[filename], trace) for filename, trace in self.EXAMPLES]
        existing = {(trace.dataset_id, trace.parameter) for trace in self.window.active_plot().traces}
        missing = [request for request in requests if request not in existing]
        if missing:
            self.window.add_traces(missing)
        self.window.control_tabs.setCurrentIndex(1)

    def apply_example_range(self) -> None:
        self.add_example_traces()
        plot = self.window.plot_widgets[self.window.active_plot_id]
        plot.freq_min.setText("1")
        plot.freq_max.setText("10")
        plot.apply_ranges()

    def show_example_marker(self) -> None:
        self.add_example_traces()
        plot = self.window.plot_widgets[self.window.active_plot_id]
        if not plot.config.markers:
            plot.add_marker()
        plot.set_marker_frequency(plot.config.markers[-1].id, 5e9)
        plot.marker_splitter.set_panel_expanded(True, plot.config.marker_info_width)

    def export_example(self) -> None:
        self.add_example_traces()
        self.window.export_active_png()
