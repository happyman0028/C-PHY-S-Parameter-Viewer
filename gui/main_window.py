from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from core.models import Dataset, PlotConfig, TraceConfig
from core.network_transform import extract_dataset, transform_dataset
from core.sparameter_math import format_trace
from core.touchstone_loader import load_s4p, load_s6p
from core.workspace import load_workspace, save_workspace
from gui.settings_dialogs import GuideDialog, PlotSettingsDialog, load_png_settings, save_png_settings
from widgets.plot_widget import AnalyzerPlotWidget
from widgets.trace_selector import TraceSelector, TraceTable

TRACE_COLORS = ("#4fc3f7", "#ff8a65", "#81c784", "#ba68c8", "#ffd54f", "#4dd0e1", "#f06292", "#a1887f")


def next_plot_name(plots: list[PlotConfig]) -> str:
    """Return a session-monotonic automatic name without collisions."""
    numbers = []
    for plot in plots:
        match = re.fullmatch(r"Plot\s+(\d+)", plot.name, re.I)
        if match:
            numbers.append(int(match.group(1)))
    return f"Plot {max(numbers, default=0) + 1}"


class LoaderSignals(QtCore.QObject):
    loaded = QtCore.Signal(object)
    failed = QtCore.Signal(str, str)
    finished = QtCore.Signal()
    progress = QtCore.Signal(int, int, str)


class DatasetLoader(QtCore.QRunnable):
    def __init__(self, paths: list[str], loader) -> None:
        super().__init__()
        self.paths = paths
        self.loader = loader
        self.signals = LoaderSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            total = len(self.paths)
            for index, path in enumerate(self.paths, 1):
                try:
                    self.signals.loaded.emit(self.loader(path))
                except Exception as exc:
                    self.signals.failed.emit(path, str(exc))
                finally:
                    self.signals.progress.emit(index, total, Path(path).name)
        finally:
            self.signals.finished.emit()


class DatasetSettingsDialog(QtWidgets.QDialog):
    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle(f"Port & Reference Impedance — {dataset.name}")
        self.setModal(True)
        self.setMinimumWidth(430)

        mapping_group = QtWidgets.QGroupBox(f"DUT {dataset.nports}-port mapping")
        form = QtWidgets.QFormLayout(mapping_group)
        self.spins = []
        labels = (
            ("Side 1 / wire A", "Side 1 / wire B", "Side 1 / wire C", "Side 2 / wire A", "Side 2 / wire B", "Side 2 / wire C")
            if dataset.network_type == "cphy"
            else ("Side 1 / P+", "Side 1 / P−", "Side 2 / P+", "Side 2 / P−")
        )
        for label, value in zip(labels, dataset.port_mapping):
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, dataset.nports)
            spin.setValue(value)
            self.spins.append(spin)
            form.addRow(label, spin)
        note = QtWidgets.QLabel(f"Each physical port 1–{dataset.nports} must appear exactly once. P+/P− order sets differential polarity.")
        note.setWordWrap(True)
        form.addRow(note)

        impedance_group = QtWidgets.QGroupBox("Reference impedance (power-wave)")
        impedance_form = QtWidgets.QFormLayout(impedance_group)
        source = QtWidgets.QLabel(dataset.source_z0_summary + f" (from {dataset.file_type})")
        self.common_z0 = self._z0_spin(float(dataset.se_reference_z0))
        self.preset = QtWidgets.QComboBox()
        nominal_preset = "C-PHY nominal" if dataset.network_type == "cphy" else "DIFF nominal"
        self.preset.addItems((nominal_preset, "Custom"))
        # Retired Equal power-wave workspaces appear as Custom; their stored
        # values are treated as explicit post-transform target impedances.
        selected_preset = dataset.impedance_preset if dataset.impedance_preset in {
            nominal_preset, "Custom"
        } else "Custom"
        self.preset.setCurrentText(selected_preset)
        modal = dataset.modal_reference_z0 or ((dataset.se_reference_z0,) * (3 if dataset.network_type == "cphy" else 2))
        self.modal_spins = [self._z0_spin(float(value)) for value in modal]
        impedance_form.addRow("Source port Z0", source)
        impedance_form.addRow("Common SE Z0", self.common_z0)
        impedance_form.addRow("Preset", self.preset)
        modal_labels = ("X-mode Z0", "Y-mode Z0", "CM-mode Z0") if dataset.network_type == "cphy" else ("Differential Z0", "Common-mode Z0")
        for label, spin in zip(modal_labels, self.modal_spins):
            impedance_form.addRow(label, spin)
        info_text = (
            "Processing: source Z0 → common SE Z0 → C-PHY transform → optional custom modal Z0.\n"
            "C-PHY nominal: X=2×SE, Y=1.5×SE, CM=SE÷3; the normalized transform already implies these references."
            if dataset.network_type == "cphy" else
            "Processing: source Z0 → common SE Z0 → DIFF mixed-mode transform → optional custom modal Z0.\n"
            "DIFF nominal: Differential=2×SE, Common=SE÷2; the normalized transform already implies these references."
        )
        info = QtWidgets.QLabel(info_text)
        info.setWordWrap(True)
        impedance_form.addRow(info)
        self.apply_all = QtWidgets.QCheckBox(f"Apply impedance settings to all loaded {dataset.analysis_label} datasets")
        impedance_form.addRow(self.apply_all)
        self.preset.currentTextChanged.connect(self._preset_changed)
        self.common_z0.valueChanged.connect(self._common_changed)
        self._update_modal_controls()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(mapping_group)
        layout.addWidget(impedance_group)
        layout.addWidget(buttons)

    @staticmethod
    def _z0_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.001, 1_000_000.0)
        spin.setDecimals(6)
        spin.setSuffix(" Ω")
        spin.setValue(value)
        return spin

    @property
    def mapping(self) -> tuple[int, ...]:
        return tuple(spin.value() for spin in self.spins)

    @property
    def impedance_settings(self) -> tuple[float, tuple[float, ...], str]:
        return (
            self.common_z0.value(),
            tuple(spin.value() for spin in self.modal_spins),
            self.preset.currentText(),
        )

    def _preset_changed(self) -> None:
        self._apply_preset_values()
        self._update_modal_controls()

    def _common_changed(self) -> None:
        self._apply_preset_values()

    def _apply_preset_values(self) -> None:
        common = self.common_z0.value()
        if self.preset.currentText() == "C-PHY nominal":
            values = (2.0 * common, 1.5 * common, common / 3.0)
        elif self.preset.currentText() == "DIFF nominal":
            values = (2.0 * common, common / 2.0)
        else:
            return
        for spin, value in zip(self.modal_spins, values):
            spin.setValue(value)

    def _update_modal_controls(self) -> None:
        editable = self.preset.currentText() == "Custom"
        for spin in self.modal_spins:
            spin.setEnabled(editable)

    def _validate(self) -> None:
        if sorted(self.mapping) != list(range(1, self.dataset.nports + 1)):
            QtWidgets.QMessageBox.warning(self, "Invalid mapping", f"Use every physical port from 1 through {self.dataset.nports} exactly once.")
            return
        self.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("C-PHY S-Parameter Viewer & Analyzer — v1.0")
        self.resize(1600, 980)
        self.setMinimumSize(1100, 720)
        self.datasets: dict[str, Dataset] = {}
        self.plots: list[PlotConfig] = []
        self.plot_widgets: dict[str, AnalyzerPlotWidget] = {}
        self.active_plot_id: str | None = None
        self.plots_frozen = False
        self._freeze_reference_width: int | None = None
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self._loader_refs: list[DatasetLoader] = []
        self._pending_imported_dataset_count = 0
        self.settings = QtCore.QSettings()
        self.theme = self.settings.value("ui/theme", "Dark", str)
        self.png_settings = load_png_settings(self.settings)
        self._build_ui()
        self._create_actions()
        self._apply_theme()
        self.add_plot()

    def _build_ui(self) -> None:
        def make_dataset_tree() -> QtWidgets.QTreeWidget:
            tree = QtWidgets.QTreeWidget()
            tree.setHeaderLabels(["Dataset", "Info"])
            tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            tree.customContextMenuRequested.connect(self._dataset_menu)
            return tree

        self.dataset_tree = make_dataset_tree()

        self.plot_list = QtWidgets.QListWidget()
        self.plot_list.currentRowChanged.connect(self._plot_selected)
        self.trace_selector = TraceSelector()
        self.trace_selector.traces_requested.connect(self.add_traces)
        self.trace_table = TraceTable()
        self.trace_table.trace_changed.connect(self._active_plot_changed)
        self.trace_table.remove_requested.connect(self.remove_trace)
        self.trace_table.remove_many_requested.connect(self.remove_traces)
        self.trace_table.remove_all_requested.connect(self.remove_all_traces)

        tabs = QtWidgets.QTabWidget()
        dataset_page = QtWidgets.QWidget()
        dataset_layout = QtWidgets.QVBoxLayout(dataset_page)
        dataset_layout.addWidget(self.dataset_tree)
        mapping_button = QtWidgets.QPushButton("Edit selected port / impedance")
        mapping_button.clicked.connect(self.edit_port_mapping)
        remove_dataset = QtWidgets.QPushButton("Remove selected datasets")
        remove_dataset.clicked.connect(self.remove_dataset)
        dataset_layout.addWidget(mapping_button)
        dataset_layout.addWidget(remove_dataset)
        trace_page = QtWidgets.QWidget()
        trace_layout = QtWidgets.QVBoxLayout(trace_page)
        trace_layout.addWidget(self.trace_selector)
        active_page = QtWidgets.QWidget()
        active_layout = QtWidgets.QVBoxLayout(active_page)
        active_layout.addWidget(QtWidgets.QLabel("Plots"))
        active_layout.addWidget(self.plot_list, 1)
        active_layout.addWidget(QtWidgets.QLabel("Active plot traces"))
        active_layout.addWidget(self.trace_table, 3)
        tabs.addTab(dataset_page, "Datasets")
        tabs.addTab(trace_page, "Add Trace")
        tabs.addTab(active_page, "Plot / Traces")
        self.control_tabs = tabs
        self.active_trace_page = active_page

        dock = QtWidgets.QDockWidget("Analyzer Control", self)
        dock.setObjectName("controlDock")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        dock.setWidget(tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        self.plot_container = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(8, 8, 8, 8)
        self.plot_layout.setSpacing(10)
        self.plot_layout.addStretch()
        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.plot_scroll.setWidget(self.plot_container)
        self.setCentralWidget(self.plot_scroll)
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.import_progress = QtWidgets.QProgressBar()
        self.import_progress.setFixedWidth(220)
        self.import_progress.setVisible(False)
        self.status.addPermanentWidget(self.import_progress)

    def _create_actions(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        left_actions = (
            ("Import S4P", self.open_s4p_datasets, "Ctrl+Shift+O"),
            ("Import S6P", self.open_s6p_datasets, "Ctrl+O"),
            ("Save Workspace", self.save_workspace_dialog, "Ctrl+S"),
            ("Load Workspace", self.load_workspace_dialog, "Ctrl+L"),
            ("Add Plot", self.add_plot, "Ctrl+P"),
        )
        for text, callback, shortcut in left_actions:
            action = QtGui.QAction(text, self)
            action.triggered.connect(callback)
            if shortcut:
                action.setShortcut(shortcut)
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("  Configurable power-wave reference Z0  "))
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self.freeze_action = QtGui.QAction("🔓 Freeze Plots", self)
        self.freeze_action.setCheckable(True)
        self.freeze_action.setToolTip("Lock all plot widths at their current sizes")
        self.freeze_action.toggled.connect(self.toggle_plot_freeze)
        toolbar.addAction(self.freeze_action)
        plot_settings = QtGui.QAction("Plot Settings", self)
        plot_settings.triggered.connect(self.show_plot_settings)
        toolbar.addAction(plot_settings)
        export_png = QtGui.QAction("Export Plot PNG", self)
        export_png.triggered.connect(self.export_active_png)
        toolbar.addAction(export_png)
        export_csv = QtGui.QAction("Export Trace CSV", self)
        export_csv.triggered.connect(self.export_active_csv)
        toolbar.addAction(export_csv)
        self.theme_action = QtGui.QAction("☀ Light Mode" if self.theme == "Dark" else "🌙 Dark Mode", self)
        self.theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(self.theme_action)
        guide = QtGui.QAction("Guide", self)
        guide.triggered.connect(self.show_guide)
        toolbar.addAction(guide)

    def _apply_theme(self) -> None:
        if self.theme == "Light":
            self.setStyleSheet("""
            QMainWindow, QWidget { background:#f4f6f8; color:#202830; }
            QToolBar { background:#e8edf2; border-bottom:1px solid #b8c2cc; spacing:5px; padding:5px; }
            QToolBar QToolButton, QPushButton, QToolButton { background:#ffffff; border:1px solid #aeb8c2; border-radius:4px; padding:5px 8px; }
            QPushButton:hover, QToolButton:hover { background:#dce9f2; }
            QPushButton#accentButton { background:#007d77; color:white; font-weight:600; padding:7px; }
            QLineEdit, QComboBox, QListWidget, QTreeWidget, QTableWidget, QTextBrowser { background:#ffffff; border:1px solid #b8c2cc; selection-background-color:#1f8790; selection-color:white; padding:3px; }
            QHeaderView::section { background:#dde4ea; color:#26333f; border:0; padding:5px; }
            QTabWidget::pane { border:1px solid #b8c2cc; }
            QTabBar::tab { background:#e3e8ed; padding:7px 12px; }
            QTabBar::tab:selected { background:#007d77; color:white; }
            QDockWidget::title { background:#dde4ea; padding:6px; }
            QFrame#plotCard { background:#ffffff; border:1px solid #aeb8c2; border-radius:5px; }
            QFrame#plotCard[active="true"] { border:2px solid #008f86; }
            QStatusBar { background:#e8edf2; color:#44515c; }
            """)
        else:
            self.setStyleSheet("""
            QMainWindow, QWidget { background:#151b22; color:#d8e0e8; }
            QToolBar { background:#1c2530; border-bottom:1px solid #354250; spacing:5px; padding:5px; }
            QToolBar QToolButton, QPushButton, QToolButton { background:#263341; border:1px solid #3b4b5b; border-radius:4px; padding:5px 8px; }
            QPushButton:hover, QToolButton:hover { background:#324456; }
            QPushButton#accentButton { background:#007d77; color:white; font-weight:600; padding:7px; }
            QLineEdit, QComboBox, QListWidget, QTreeWidget, QTableWidget { background:#10161d; border:1px solid #34414f; selection-background-color:#136b75; padding:3px; }
            QHeaderView::section { background:#202b36; color:#aebdca; border:0; padding:5px; }
            QTabWidget::pane { border:1px solid #34414f; }
            QTabBar::tab { background:#202a34; padding:7px 12px; }
            QTabBar::tab:selected { background:#007d77; color:white; }
            QDockWidget::title { background:#202a34; padding:6px; }
            QFrame#plotCard { background:#19212a; border:1px solid #354452; border-radius:5px; }
            QFrame#plotCard[active="true"] { border:2px solid #00a79d; }
            QStatusBar { background:#10161d; color:#91a0ad; }
            """)
        for widget in self.plot_widgets.values():
            widget.apply_theme(self.theme)
            widget.refresh()

    def toggle_theme(self) -> None:
        self.theme = "Light" if self.theme == "Dark" else "Dark"
        self.settings.setValue("ui/theme", self.theme)
        self.theme_action.setText("☀ Light Mode" if self.theme == "Dark" else "🌙 Dark Mode")
        self._apply_theme()

    def toggle_plot_freeze(self, frozen: bool) -> None:
        self.plots_frozen = bool(frozen)
        active_widget = self.plot_widgets.get(self.active_plot_id)
        if frozen:
            self._freeze_reference_width = active_widget.width() if active_widget else None
            for widget in self.plot_widgets.values():
                widget.set_plot_width_frozen(True)
            self.freeze_action.setText("🔒 Plots Frozen")
            self.freeze_action.setToolTip("Restore responsive plot sizing")
            self.status.showMessage("Plot widths frozen", 2500)
        else:
            for widget in self.plot_widgets.values():
                widget.set_plot_width_frozen(False)
            self._freeze_reference_width = None
            self.freeze_action.setText("🔓 Freeze Plots")
            self.freeze_action.setToolTip("Lock all plot widths at their current sizes")
            self.status.showMessage("Responsive plot sizing restored", 2500)
        self.plot_layout.invalidate()
        self.plot_container.updateGeometry()

    def _scroll_main_plots(self, delta: int) -> None:
        if not delta:
            return
        bar = self.plot_scroll.verticalScrollBar()
        scale = max(1, bar.singleStep() * 3)
        movement = round(delta * scale / 120)
        if movement == 0:
            movement = 1 if delta > 0 else -1
        bar.setValue(bar.value() - movement)

    def show_plot_settings(self) -> None:
        active_plot = self.active_plot()
        font_values = {
            "legend": active_plot.legend_font_size if active_plot else 9,
            "title": active_plot.title_font_size if active_plot else 11,
            "axis": active_plot.axis_font_size if active_plot else 9,
        }
        dialog = PlotSettingsDialog(self.png_settings, font_values, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.png_settings.clear()
            self.png_settings.update(dialog.values())
            save_png_settings(self.settings, self.png_settings)
            for widget in self.plot_widgets.values():
                widget.set_curve_render_quality(self.png_settings["curve_render_quality"])
            fonts = dialog.plot_font_values()
            targets = self.plots if fonts.pop("apply_to_all") else ([active_plot] if active_plot else [])
            for plot in targets:
                plot.legend_font_size = fonts["legend"]
                plot.title_font_size = fonts["title"]
                plot.axis_font_size = fonts["axis"]
                self.plot_widgets[plot.id].set_font_sizes(
                    fonts["legend"], fonts["title"], fonts["axis"], emit=False
                )
            self.status.showMessage("Plot rendering, font, and PNG settings saved", 3500)

    def show_guide(self) -> None:
        GuideDialog(self).exec()

    def open_datasets(self) -> None:
        """Compatibility alias for the original S6P action."""
        self.open_s6p_datasets()

    def open_s6p_datasets(self) -> None:
        self._open_datasets("S6P", "Touchstone 6-port (*.s6p)", load_s6p)

    def open_s4p_datasets(self) -> None:
        self._open_datasets("S4P", "Touchstone 4-port (*.s4p)", load_s4p)

    def _open_datasets(self, file_type: str, file_filter: str, loader_function) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, f"Import {file_type} datasets", "", file_filter)
        if not paths:
            return
        self.status.showMessage(f"Loading {len(paths)} dataset(s)…")
        loader = DatasetLoader(paths, loader_function)
        loader.signals.loaded.connect(self._dataset_loaded)
        loader.signals.failed.connect(self._dataset_failed)
        loader.signals.progress.connect(self._loading_progress)
        loader.signals.finished.connect(lambda: self._loading_finished(loader))
        self._loader_refs.append(loader)
        self.import_progress.setRange(0, len(paths))
        self.import_progress.setValue(0)
        self.import_progress.setVisible(True)
        self.thread_pool.start(loader)

    def _dataset_loaded(self, dataset: Dataset) -> None:
        base = dataset.name
        existing = {item.name for item in self.datasets.values()}
        suffix = 2
        while dataset.name in existing:
            dataset.name = f"{base}_{suffix}"
            suffix += 1
        self.datasets[dataset.id] = dataset
        self._pending_imported_dataset_count += 1
        # Loader signals arrive on the GUI thread. Rebuilding the complete
        # Dataset tree, selectors, trace table, and every plot for each file
        # made a large import effectively O(N²) and starved unrelated actions.
        # During an import, accumulate results and commit the UI once all
        # active loaders have finished. Direct/synchronous callers still get
        # an immediate refresh.
        if not self._loader_refs:
            self._flush_imported_datasets()

    def _dataset_failed(self, path: str, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Dataset load failed", f"{Path(path).name}\n\n{message}")

    def _loading_finished(self, loader: DatasetLoader) -> None:
        if loader in self._loader_refs:
            self._loader_refs.remove(loader)
        if not self._loader_refs:
            imported = self._pending_imported_dataset_count
            self._flush_imported_datasets()
            self.status.showMessage(f"Dataset loading finished — {imported} imported", 3000)
            self.import_progress.setVisible(False)

    def _flush_imported_datasets(self) -> None:
        if not self._pending_imported_dataset_count:
            return
        self._pending_imported_dataset_count = 0
        # A newly imported dataset cannot yet be referenced by an existing
        # trace, so plot redraws and trace-table reconstruction are unnecessary.
        self._refresh_datasets(refresh_trace_table=False)

    def _loading_progress(self, current: int, total: int, filename: str) -> None:
        self.import_progress.setRange(0, total)
        self.import_progress.setValue(current)
        self.import_progress.setFormat(f"{current}/{total}")
        self.status.showMessage(f"Importing {filename} ({current}/{total})")

    def _refresh_datasets(self, refresh_trace_table: bool = True) -> None:
        self.dataset_tree.clear()
        ordered_datasets = sorted(
            self.datasets.values(),
            key=lambda item: (0 if item.network_type == "cphy" else 1, item.name.casefold()),
        )
        for dataset in ordered_datasets:
            low, high = dataset.frequency_range
            top = QtWidgets.QTreeWidgetItem([f"[{dataset.file_type} · {dataset.analysis_label}] {dataset.name}", f"{dataset.points} pts · SE {dataset.se_reference_z0:g} Ω"])
            top.setData(0, QtCore.Qt.UserRole, dataset.id)
            top.addChild(QtWidgets.QTreeWidgetItem(["File", dataset.filename]))
            top.addChild(QtWidgets.QTreeWidgetItem(["Frequency", f"{low / 1e9:.6g}–{high / 1e9:.6g} GHz"]))
            top.addChild(QtWidgets.QTreeWidgetItem(["Port map", ", ".join(map(str, dataset.port_mapping))]))
            top.addChild(QtWidgets.QTreeWidgetItem(["Source Z0", dataset.source_z0_summary]))
            top.addChild(QtWidgets.QTreeWidgetItem(["Common SE Z0", f"{dataset.se_reference_z0:g} Ω"]))
            if dataset.network_type == "cphy":
                x_z0, y_z0, cm_z0 = dataset.modal_reference_z0
                modal_text = f"X {x_z0:g} · Y {y_z0:g} · CM {cm_z0:g} Ω ({dataset.impedance_preset})"
            else:
                differential_z0, common_z0 = dataset.modal_reference_z0
                modal_text = f"DM {differential_z0:g} · CM {common_z0:g} Ω ({dataset.impedance_preset})"
            top.addChild(QtWidgets.QTreeWidgetItem(["Modal target Z0", modal_text]))
            self.dataset_tree.addTopLevelItem(top)
        self.trace_selector.set_datasets(self.datasets)
        if refresh_trace_table:
            self._refresh_trace_table()

    def _selected_dataset(self) -> Dataset | None:
        selected = self._selected_datasets()
        return selected[0] if selected else None

    def _selected_datasets(self) -> list[Dataset]:
        result: list[Dataset] = []
        seen: set[str] = set()
        for selected in self.dataset_tree.selectedItems():
            item = selected
            dataset_id = None
            while item is not None:
                dataset_id = item.data(0, QtCore.Qt.UserRole)
                if dataset_id in self.datasets:
                    break
                item = item.parent()
            if dataset_id in self.datasets and dataset_id not in seen:
                seen.add(dataset_id)
                result.append(self.datasets[dataset_id])
        return result

    def _dataset_menu(self, position) -> None:
        if not self._selected_dataset():
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Edit selected port / reference impedance", self.edit_port_mapping)
        menu.addAction("Remove selected datasets", self.remove_dataset)
        tree = self.sender() if isinstance(self.sender(), QtWidgets.QTreeWidget) else self.dataset_tree
        menu.exec(tree.viewport().mapToGlobal(position))

    def edit_port_mapping(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return
        types = {item.network_type for item in selected}
        if len(types) != 1:
            QtWidgets.QMessageBox.warning(self, "Mixed dataset types", "Batch editing requires only C-PHY datasets or only DIFF datasets.")
            return
        dataset = selected[0]
        dialog = DatasetSettingsDialog(dataset, self)
        if len(selected) > 1:
            dialog.apply_all.hide()
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            common_z0, modal_z0, preset = dialog.impedance_settings
            targets = (
                [item for item in self.datasets.values() if item.network_type == dataset.network_type]
                if len(selected) == 1 and dialog.apply_all.isChecked() else selected
            )
            for target in targets:
                target.port_mapping = dialog.mapping
                target.se_reference_z0 = common_z0
                target.modal_reference_z0 = modal_z0
                target.impedance_preset = preset
                transform_dataset(target)
            self._refresh_datasets()
            self._refresh_all_plots()
            scope = f"{len(targets)} selected dataset(s)"
            self.status.showMessage(f"Updated port / reference impedance for {scope}", 4000)

    def remove_dataset(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return
        names = "\n".join(f"• {item.name}" for item in selected[:8])
        if len(selected) > 8:
            names += f"\n… and {len(selected) - 8} more"
        if QtWidgets.QMessageBox.question(self, "Remove datasets", f"Remove {len(selected)} dataset(s) and their traces?\n\n{names}") != QtWidgets.QMessageBox.Yes:
            return
        ids = {item.id for item in selected}
        for dataset_id in ids:
            del self.datasets[dataset_id]
        for plot in self.plots:
            plot.traces[:] = [trace for trace in plot.traces if trace.dataset_id not in ids]
        self._refresh_datasets()
        self._refresh_all_plots()

    def add_plot(self, config: PlotConfig | None = None) -> None:
        config = config if isinstance(config, PlotConfig) else PlotConfig(name=next_plot_name(self.plots))
        self.plots.append(config)
        widget = AnalyzerPlotWidget(config, self.datasets, self.png_settings, self.theme)
        widget.activated.connect(self.set_active_plot)
        widget.wheel_scroll_requested.connect(self._scroll_main_plots)
        widget.config_changed.connect(self._plot_config_changed)
        widget.refresh_progress.connect(self._plot_refresh_progress)
        widget.rename_requested.connect(self.rename_plot)
        widget.marker_changed.connect(self.status.showMessage)
        widget.image_copied.connect(self.status.showMessage)
        widget.trace_selection_changed.connect(self._plot_trace_selection_changed)
        widget.find_trace_requested.connect(self._find_plot_trace)
        widget.find_highlight_clear_requested.connect(self._clear_plot_trace_find)
        widget.delete_requested.connect(self.delete_plot)
        self.plot_widgets[config.id] = widget
        self.plot_layout.insertWidget(self.plot_layout.count() - 1, widget)
        self._refresh_plot_list()
        self.set_active_plot(config.id)
        if self.plots_frozen:
            reference_width = self._freeze_reference_width
            QtCore.QTimer.singleShot(
                0,
                lambda current=widget, target=reference_width: (
                    current.set_plot_width_frozen(True, target)
                    if self.plots_frozen else None
                ),
            )

    def delete_plot(self, plot_id: str | None = None) -> None:
        plot_id = plot_id or self.active_plot_id
        if not plot_id:
            return
        if len(self.plots) <= 1:
            QtWidgets.QMessageBox.information(self, "Cannot delete plot", "At least one plot must remain in the workspace.")
            return
        index = next((i for i, plot in enumerate(self.plots) if plot.id == plot_id), -1)
        if index < 0:
            return
        plot = self.plots[index]
        trace_text = f"This plot contains {len(plot.traces)} trace(s)." if plot.traces else "This plot contains no traces."
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete plot",
            f'Delete "{plot.name}"?\n\n{trace_text}',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.plots.pop(index)
        widget = self.plot_widgets.pop(plot_id)
        widget.deleteLater()
        self.active_plot_id = self.plots[min(index, len(self.plots) - 1)].id
        self._refresh_plot_list()
        self.set_active_plot(self.active_plot_id)

    def rename_plot(self, plot_id: str, requested_name: str) -> None:
        plot = next((item for item in self.plots if item.id == plot_id), None)
        widget = self.plot_widgets.get(plot_id)
        if not plot or not widget:
            return
        requested_name = requested_name.strip() or plot.name
        duplicate = any(
            item.id != plot_id and item.name.casefold() == requested_name.casefold()
            for item in self.plots
        )
        if duplicate:
            QtWidgets.QMessageBox.warning(self, "Duplicate plot name", f'A plot named "{requested_name}" already exists.')
            widget.set_name(plot.name)
            return
        widget.set_name(requested_name)
        self._refresh_plot_list()
        self.status.showMessage(f'Renamed plot to "{requested_name}"', 3000)

    def _refresh_plot_list(self) -> None:
        current = self.active_plot_id
        self.plot_list.blockSignals(True)
        self.plot_list.clear()
        for plot in self.plots:
            item = QtWidgets.QListWidgetItem(plot.name)
            item.setData(QtCore.Qt.UserRole, plot.id)
            self.plot_list.addItem(item)
        if current:
            for row in range(self.plot_list.count()):
                if self.plot_list.item(row).data(QtCore.Qt.UserRole) == current:
                    self.plot_list.setCurrentRow(row)
                    break
        self.plot_list.blockSignals(False)

    def _plot_selected(self, row: int) -> None:
        if row >= 0:
            self.set_active_plot(self.plot_list.item(row).data(QtCore.Qt.UserRole))

    def set_active_plot(self, plot_id: str) -> None:
        self.active_plot_id = plot_id
        for key, widget in self.plot_widgets.items():
            widget.set_active(key == plot_id)
        self._refresh_trace_table()
        self._refresh_plot_list()
        plot = self.active_plot()
        if plot:
            self.status.showMessage(f"Active plot: {plot.name}", 4000)

    def active_plot(self) -> PlotConfig | None:
        return next((plot for plot in self.plots if plot.id == self.active_plot_id), None)

    def add_trace(self, dataset_id: str, parameter: str) -> None:
        self.add_traces([(dataset_id, parameter)])

    def add_traces(self, requests: list[tuple[str, str]]) -> None:
        plot = self.active_plot()
        if not plot:
            return
        show_progress = not self._loader_refs and len(requests) > 1
        if show_progress:
            self.import_progress.setRange(0, 100)
            self.import_progress.setValue(0)
            self.import_progress.setFormat("Adding traces… %p%")
            self.import_progress.setVisible(True)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        existing = {(trace.dataset_id, trace.parameter) for trace in plot.traces}
        added = 0
        total = max(1, len(requests))
        update_interval = max(1, total // 20)
        for request_index, (dataset_id, parameter) in enumerate(requests, 1):
            key = (dataset_id, parameter)
            if key in existing:
                pass
            else:
                color = TRACE_COLORS[len(plot.traces) % len(TRACE_COLORS)]
                plot.traces.append(TraceConfig(dataset_id=dataset_id, parameter=parameter, color=color))
                existing.add(key)
                added += 1
            if show_progress and (request_index % update_interval == 0 or request_index == total):
                self.import_progress.setValue(int(request_index / total * 35))
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        if not added:
            self.status.showMessage("Selected trace(s) already exist in active plot", 2500)
            if show_progress:
                self.import_progress.setVisible(False)
            return
        if show_progress:
            self.import_progress.setValue(40)
            self.import_progress.setFormat("Rendering plot… %p%")
            self.status.showMessage(f"Rendering {added} new trace(s)…")
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        self._active_plot_changed()
        if show_progress:
            self.import_progress.setValue(100)
            self.import_progress.setFormat("Add Trace complete")
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            QtCore.QTimer.singleShot(600, self._finish_add_trace_progress)
        self.status.showMessage(f"Added {added} trace(s) to {plot.name}", 3500)

    def _finish_add_trace_progress(self) -> None:
        if not self._loader_refs:
            self.import_progress.setVisible(False)

    def _plot_refresh_progress(self, completed: int, total: int) -> None:
        if self.import_progress.isHidden() or not self.import_progress.format().startswith("Rendering plot"):
            return
        fraction = completed / max(1, total)
        self.import_progress.setValue(40 + min(59, int(fraction * 59)))
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def remove_trace(self, index: int) -> None:
        plot = self.active_plot()
        if plot and 0 <= index < len(plot.traces):
            plot.traces.pop(index)
            self._active_plot_changed()

    def remove_traces(self, indices: list[int]) -> None:
        plot = self.active_plot()
        valid = sorted({index for index in indices if plot and 0 <= index < len(plot.traces)}, reverse=True)
        if not plot or not valid:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete selected traces",
            f'Delete {len(valid)} selected trace(s) from "{plot.name}"?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        for index in valid:
            plot.traces.pop(index)
        self._active_plot_changed()
        self.status.showMessage(f"Deleted {len(valid)} traces from {plot.name}", 4000)

    def remove_all_traces(self) -> None:
        plot = self.active_plot()
        if not plot or not plot.traces:
            return
        count = len(plot.traces)
        plot.traces.clear()
        self._active_plot_changed()
        self.status.showMessage(f"Deleted all {count} traces from {plot.name}", 4000)

    def _active_plot_changed(self) -> None:
        plot = self.active_plot()
        if not plot:
            return
        self.plot_widgets[plot.id].refresh()
        self._refresh_trace_table()

    def _plot_config_changed(self) -> None:
        self._refresh_plot_list()

    def _refresh_trace_table(self) -> None:
        plot = self.active_plot()
        self.trace_table.set_traces(plot.traces if plot else [], self.datasets, plot.id if plot else None)

    def _plot_trace_selection_changed(self, plot_id: str, _indices: list[int]) -> None:
        pass

    def _clear_plot_trace_find(self, plot_id: str) -> None:
        if plot_id == self.active_plot_id:
            self.trace_table.clear_trace_highlight()

    def _find_plot_trace(self, plot_id: str, source_indices: list[int]) -> None:
        self.set_active_plot(plot_id)
        self.control_tabs.setCurrentWidget(self.active_trace_page)
        self.trace_table.highlight_traces(source_indices)
        plot = self.active_plot()
        valid = [index for index in source_indices if plot and 0 <= index < len(plot.traces)]
        if plot and valid:
            trace = plot.traces[valid[0]]
            dataset = self.datasets.get(trace.dataset_id)
            dataset_name = dataset.name if dataset else "Missing dataset"
            detail = f"{dataset_name} — {trace.parameter}" if len(valid) == 1 else f"{len(valid)} traces"
            self.status.showMessage(f"Found: {detail}", 5000)

    def _refresh_all_plots(self) -> None:
        for widget in self.plot_widgets.values():
            widget.datasets = self.datasets
            widget.refresh()

    def export_active_png(self) -> None:
        plot = self.active_plot()
        if plot:
            self.plot_widgets[plot.id].export_png_dialog()

    def export_active_csv(self) -> None:
        plot = self.active_plot()
        if not plot or not plot.traces:
            QtWidgets.QMessageBox.information(self, "Export", "The active plot has no traces.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export trace data", f"{plot.name}.csv", "CSV (*.csv)")
        if not path:
            return
        # Datasets may use different frequency grids: export long-form rows.
        with open(path, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset", "parameter", "frequency_hz", "real", "imag", "display_format", "display_value"])
            for trace in plot.traces:
                dataset = self.datasets.get(trace.dataset_id)
                if not dataset:
                    continue
                values = extract_dataset(dataset, trace.parameter)
                displayed = format_trace(values, dataset.frequency_hz, plot.display_format)
                for frequency, value, shown in zip(dataset.frequency_hz, values, displayed):
                    writer.writerow([dataset.name, trace.parameter, frequency, value.real, value.imag, plot.display_format, shown])
        self.status.showMessage(f"Exported {Path(path).name}", 5000)

    def save_workspace_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save workspace", "cphy_workspace.json", "JSON (*.json)")
        if not path:
            return
        datasets = [
            {
                "id": d.id,
                "name": d.name,
                "path": d.path,
                "port_mapping": list(d.port_mapping),
                "se_reference_z0": d.se_reference_z0,
                "modal_reference_z0": list(d.modal_reference_z0),
                "impedance_preset": d.impedance_preset,
                "network_type": d.network_type,
            }
            for d in self.datasets.values()
        ]
        save_workspace(path, datasets, [plot.to_dict() for plot in self.plots])
        self.status.showMessage(f"Saved workspace {Path(path).name}", 5000)

    def load_workspace_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load workspace", "", "JSON (*.json)")
        if not path:
            return
        try:
            payload = load_workspace(path)
            new_datasets: dict[str, Dataset] = {}
            for item in payload["datasets"]:
                dataset_path = Path(item["path"])
                if not dataset_path.exists():
                    answer = QtWidgets.QMessageBox.question(self, "Missing dataset", f"{dataset_path}\n\nLocate this file now?")
                    if answer == QtWidgets.QMessageBox.Yes:
                        replacement, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Locate dataset", "", "Touchstone (*.s4p *.s6p)")
                        if replacement:
                            dataset_path = Path(replacement)
                    if not dataset_path.exists():
                        continue
                modal_z0 = item.get("modal_reference_z0")
                network_type = item.get("network_type", "dphy" if dataset_path.suffix.lower() == ".s4p" else "cphy")
                loader_function = load_s4p if network_type == "dphy" else load_s6p
                default_mapping = tuple(range(1, 5 if network_type == "dphy" else 7))
                default_preset = "DIFF nominal" if network_type == "dphy" else "C-PHY nominal"
                dataset = loader_function(
                    dataset_path,
                    tuple(item.get("port_mapping", default_mapping)),
                    item.get("se_reference_z0"),
                    tuple(modal_z0) if modal_z0 else None,
                    item.get("impedance_preset", default_preset),
                )
                dataset.id = item["id"]
                dataset.name = item.get("name", dataset.name)
                new_datasets[dataset.id] = dataset
            for widget in self.plot_widgets.values():
                widget.deleteLater()
            self.datasets = new_datasets
            self.plots.clear()
            self.plot_widgets.clear()
            self.active_plot_id = None
            for item in payload["plots"]:
                self.add_plot(PlotConfig.from_dict(item))
            if not self.plots:
                self.add_plot()
            self._refresh_datasets()
            self._refresh_all_plots()
            self.status.showMessage(f"Loaded workspace {Path(path).name}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Workspace error", str(exc))
