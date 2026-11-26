from __future__ import annotations

import fnmatch
import re

from PySide6 import QtCore, QtGui, QtWidgets

from core.models import Dataset, TraceConfig
from core.sparameter_math import (
    dphy_parameter_names,
    parameter_names,
    parse_dphy_parameter,
    parse_parameter,
    se_parameter_names,
)


def wildcard_search_matches(query: str, searchable_text: str) -> bool:
    """Case-insensitive `/` OR, whitespace AND, and `*` wildcard matching."""
    query = query.strip().casefold()
    if not query:
        return True
    haystack = searchable_text.casefold()
    alternatives = [group.strip() for group in query.split("/") if group.strip()]
    for group in alternatives:
        terms = group.split()
        if all(fnmatch.fnmatchcase(haystack, f"*{term}*") for term in terms):
            return True
    return False


def natural_sort_key(value: str) -> str:
    """Return a case-insensitive key whose numeric chunks sort naturally."""
    return re.sub(r"\d+", lambda match: match.group(0).zfill(12), value.casefold())


class TraceSortItem(QtWidgets.QTableWidgetItem):
    SOURCE_ROLE = QtCore.Qt.UserRole
    SORT_ROLE = QtCore.Qt.UserRole + 1

    def __init__(self, text: str, source_index: int, sort_key) -> None:
        super().__init__(text)
        self.setData(self.SOURCE_ROLE, source_index)
        self.setData(self.SORT_ROLE, sort_key)

    def __lt__(self, other) -> bool:
        if isinstance(other, TraceSortItem):
            return self.data(self.SORT_ROLE) < other.data(self.SORT_ROLE)
        return super().__lt__(other)


class TraceSelector(QtWidgets.QWidget):
    trace_requested = QtCore.Signal(str, str)
    traces_requested = QtCore.Signal(object)
    CUSTOM_DATASETS = "CUSTOM:datasets"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._datasets: dict[str, Dataset] = {}
        self._custom_dataset_ids: list[str] = []
        self.dataset_combo = QtWidgets.QComboBox()
        self.select_datasets_button = QtWidgets.QToolButton(text="multiple")
        self.select_datasets_button.setToolTip("Select multiple datasets")
        self.select_datasets_button.setEnabled(False)
        self.trace_type = QtWidgets.QComboBox()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search parameters; * wildcard, / OR, space AND")
        self.search.setClearButtonEnabled(True)
        self.response_filter, self.stimulus_filter = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.route_filter, self.r_filter = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.response_filter.addItems(["Output Mode: Any", "x", "y", "c"])
        self.stimulus_filter.addItems(["Input Mode: Any", "x", "y", "c"])
        self.route_filter.addItems(["Direction: Any", "11", "12", "21", "22"])
        self.r_filter.addItems([
            "Rotate: ALL (1R / 2R / 3R)",
            "1R — X ∝ [1,0,-1], Y ∝ [-1,2,-1]",
            "2R — X ∝ [-1,1,0], Y ∝ [-1,-1,2]",
            "3R — X ∝ [0,-1,1], Y ∝ [2,-1,-1]",
        ])
        self.parameter_list = QtWidgets.QListWidget()
        self.parameter_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.add_button = QtWidgets.QPushButton("Add selected trace(s)")
        self.add_button.setObjectName("accentButton")
        filters = QtWidgets.QGridLayout()
        filters.addWidget(self.response_filter, 0, 0)
        filters.addWidget(self.stimulus_filter, 0, 1)
        filters.addWidget(self.route_filter, 1, 0)
        filters.addWidget(self.r_filter, 1, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Dataset"))
        dataset_row = QtWidgets.QHBoxLayout()
        dataset_row.addWidget(self.dataset_combo, 1)
        dataset_row.addWidget(self.select_datasets_button)
        layout.addLayout(dataset_row)
        layout.addWidget(QtWidgets.QLabel("Trace type"))
        layout.addWidget(self.trace_type)
        layout.addWidget(self.search)
        layout.addLayout(filters)
        layout.addWidget(self.parameter_list, 1)
        layout.addWidget(self.add_button)
        self.search.textChanged.connect(self.refresh)
        self.trace_type.currentIndexChanged.connect(self.refresh)
        self.dataset_combo.currentIndexChanged.connect(self._dataset_changed)
        self.select_datasets_button.clicked.connect(self._choose_datasets)
        for widget in (self.response_filter, self.stimulus_filter, self.route_filter, self.r_filter):
            widget.currentIndexChanged.connect(self.refresh)
        self.add_button.clicked.connect(self._emit_selected)
        self.parameter_list.itemDoubleClicked.connect(lambda _item: self._emit_selected())
        self.refresh()

    def set_datasets(self, datasets: dict[str, Dataset]) -> None:
        current = self.dataset_combo.currentData()
        self._datasets = datasets
        self._custom_dataset_ids = [item for item in self._custom_dataset_ids if item in datasets]
        self.select_datasets_button.setEnabled(len(datasets) >= 2)
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        cphy = [item for item in datasets.values() if item.network_type == "cphy"]
        dphy = [item for item in datasets.values() if item.network_type == "dphy"]
        if len(cphy) >= 2:
            self.dataset_combo.addItem(f"[ALL-C-PHY] {len(cphy)} S6P datasets", "ALL:cphy")
        if len(dphy) >= 2:
            self.dataset_combo.addItem(f"[ALL-DIFF] {len(dphy)} S4P datasets", "ALL:dphy")
        by_ports: dict[int, list[Dataset]] = {}
        for item in datasets.values():
            by_ports.setdefault(item.nports, []).append(item)
        for nports, items in sorted(by_ports.items()):
            if len(items) >= 2:
                self.dataset_combo.addItem(
                    f"[ALL-S{nports}P · SE] {len(items)} datasets", f"ALL:se:{nports}"
                )
        ordered = sorted(
            datasets.values(),
            key=lambda item: ({"cphy": 0, "dphy": 1, "se": 2}[item.network_type], item.nports, item.name.casefold()),
        )
        for dataset in ordered:
            self.dataset_combo.addItem(f"[{dataset.file_type} · {dataset.analysis_label}] {dataset.name}", dataset.id)
        if self._custom_dataset_ids:
            self.dataset_combo.addItem(
                f"[Selected] {len(self._custom_dataset_ids)} datasets", self.CUSTOM_DATASETS
            )
        if current:
            index = self.dataset_combo.findData(current)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
        self.dataset_combo.blockSignals(False)
        self._dataset_changed()

    def _choose_datasets(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select datasets")
        dialog.resize(440, 460)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Ctrl + click to toggle. Shift + click to select a range."))
        choices = QtWidgets.QListWidget(dialog)
        choices.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        current = self.dataset_combo.currentData()
        selected = set(self._custom_dataset_ids) if current == self.CUSTOM_DATASETS else {current}
        for index in range(self.dataset_combo.count()):
            dataset_id = self.dataset_combo.itemData(index)
            if dataset_id in self._datasets:
                item = QtWidgets.QListWidgetItem(self.dataset_combo.itemText(index), choices)
                item.setData(QtCore.Qt.UserRole, dataset_id)
                item.setSelected(dataset_id in selected)
        if choices.selectedItems():
            choices.setCurrentItem(choices.selectedItems()[0], QtCore.QItemSelectionModel.NoUpdate)
        layout.addWidget(choices)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        ok.setEnabled(bool(choices.selectedItems()))
        choices.itemSelectionChanged.connect(lambda: ok.setEnabled(bool(choices.selectedItems())))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            selected_items = sorted(choices.selectedItems(), key=choices.row)
            self._custom_dataset_ids = [item.data(QtCore.Qt.UserRole) for item in selected_items]
            self.set_datasets(self._datasets)
            self.dataset_combo.setCurrentIndex(self.dataset_combo.findData(self.CUSTOM_DATASETS))
        dialog.deleteLater()

    def _selected_dataset_ids(self) -> list[str]:
        data = self.dataset_combo.currentData()
        if data == self.CUSTOM_DATASETS:
            return self._custom_dataset_ids
        if data in self._datasets:
            return [data]
        if isinstance(data, str) and data.startswith("ALL:"):
            parts = data.split(":")
            if parts[1] == "se":
                return [item.id for item in self._datasets.values() if item.nports == int(parts[2])]
            return [item.id for item in self._datasets.values() if item.network_type == parts[1]]
        return []

    def _shared_port_count(self) -> int:
        return min((self._datasets[item].nports for item in self._selected_dataset_ids()), default=1)

    def _dataset_changed(self) -> None:
        data = self.dataset_combo.currentData()
        kinds = {self._datasets[item].network_type for item in self._selected_dataset_ids()}
        network_type = kinds.pop() if len(kinds) == 1 else "se"
        if isinstance(data, str) and data.startswith("ALL:se:"):
            network_type = "se"
        previous = self.trace_type.currentData()
        self.trace_type.blockSignals(True)
        self.trace_type.clear()
        if network_type == "dphy":
            self.trace_type.addItem("DIFF mixed-mode (dm/cm)", "dphy")
            self.trace_type.addItem("Single-ended (SE ports 1–4)", "se")
        elif network_type == "cphy":
            self.trace_type.addItem("C-PHY modal (X / Y / CM)", "cphy")
            self.trace_type.addItem("Single-ended (SE ports 1–6)", "se")
        else:
            self.trace_type.addItem(f"Single-ended (SE ports 1–{self._shared_port_count()})", "se")
        index = self.trace_type.findData(previous)
        self.trace_type.setCurrentIndex(index if index >= 0 else 0)
        self.trace_type.blockSignals(False)
        self.refresh()

    def _configure_filters(self, trace_type: str) -> None:
        response_value, stimulus_value = self.response_filter.currentText(), self.stimulus_filter.currentText()
        self.response_filter.blockSignals(True)
        self.stimulus_filter.blockSignals(True)
        if trace_type == "cphy":
            response_items, stimulus_items = ["Output Mode: Any", "x", "y", "c"], ["Input Mode: Any", "x", "y", "c"]
        elif trace_type == "dphy":
            response_items, stimulus_items = ["Output Mode: Any", "d", "c"], ["Input Mode: Any", "d", "c"]
        else:
            response_items, stimulus_items = ["Output Mode: Any"], ["Input Mode: Any"]
        if [self.response_filter.itemText(i) for i in range(self.response_filter.count())] != response_items:
            self.response_filter.clear()
            self.response_filter.addItems(response_items)
            if response_value in response_items:
                self.response_filter.setCurrentText(response_value)
        if [self.stimulus_filter.itemText(i) for i in range(self.stimulus_filter.count())] != stimulus_items:
            self.stimulus_filter.clear()
            self.stimulus_filter.addItems(stimulus_items)
            if stimulus_value in stimulus_items:
                self.stimulus_filter.setCurrentText(stimulus_value)
        self.response_filter.blockSignals(False)
        self.stimulus_filter.blockSignals(False)

    def refresh(self) -> None:
        search = self.search.text().strip().lower()
        trace_type = self.trace_type.currentData() or "cphy"
        self._configure_filters(trace_type)
        is_se = trace_type == "se"
        availability = {
            self.response_filter: not is_se,
            self.stimulus_filter: not is_se,
            self.route_filter: not is_se,
            self.r_filter: trace_type == "cphy",
        }
        for widget, enabled in availability.items():
            widget.setEnabled(enabled)
            font = QtGui.QFont(widget.font())
            font.setStrikeOut(not enabled)
            widget.setFont(font)
        response = None if self.response_filter.currentIndex() == 0 else self.response_filter.currentText()
        stimulus = None if self.stimulus_filter.currentIndex() == 0 else self.stimulus_filter.currentText()
        route = None if self.route_filter.currentIndex() == 0 else self.route_filter.currentText()
        r_value = None if self.r_filter.currentIndex() == 0 else int(self.r_filter.currentText()[0])
        self.parameter_list.clear()
        if is_se:
            names = se_parameter_names(self._shared_port_count())
        elif trace_type == "dphy":
            names = dphy_parameter_names()
        else:
            names = parameter_names()
        for name in names:
            if is_se:
                if not search or wildcard_search_matches(search, name):
                    self.parameter_list.addItem(name)
                continue
            if trace_type == "dphy":
                a, b, i, j = parse_dphy_parameter(name)
                r_index = None
            else:
                a, b, i, j, r_index = parse_parameter(name)
            if search and not wildcard_search_matches(search, name):
                continue
            if response and a != response:
                continue
            if stimulus and b != stimulus:
                continue
            if route and f"{i}{j}" != route:
                continue
            if r_value and r_index is not None and r_index != r_value:
                continue
            self.parameter_list.addItem(name)

    def _emit_selected(self) -> None:
        requests = [
            (dataset, item.text())
            for dataset in self._selected_dataset_ids()
            for item in self.parameter_list.selectedItems()
        ]
        if not requests:
            return
        # Keep the original per-trace signal for external compatibility, while
        # the application consumes the batch signal and redraws only once.
        for dataset, parameter in requests:
            self.trace_requested.emit(dataset, parameter)
        self.traces_requested.emit(requests)


class TraceTable(QtWidgets.QWidget):
    trace_changed = QtCore.Signal()
    remove_requested = QtCore.Signal(int)
    remove_many_requested = QtCore.Signal(list)
    remove_all_requested = QtCore.Signal()
    STYLES = ("Solid", "Dash", "Dot", "Dash Dot")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._traces: list[TraceConfig] = []
        self._context_id: str | None = None
        self._datasets: dict[str, Dataset] = {}
        self._row_indices: list[int] = []
        self._sort_column = -1
        self._sort_order = QtCore.Qt.AscendingOrder
        self._highlighted_source_indices: set[int] = set()
        self._highlighted_trace_keys: set[tuple[str, str]] = set()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search dataset / file / path / parameter; * wildcard, / OR")
        self.search.setClearButtonEnabled(True)
        self.selected_label = QtWidgets.QLabel("Selected traces: 0")
        self.select_visible_button = QtWidgets.QPushButton("Select Visible")
        self.clear_button = QtWidgets.QPushButton("Clear Selection")
        self.show_button = QtWidgets.QPushButton("Show Selected")
        self.hide_button = QtWidgets.QPushButton("Hide Selected")
        self.color_button = QtWidgets.QPushButton("Set Color")
        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItems(self.STYLES)
        self.apply_style_button = QtWidgets.QPushButton("Apply Style")
        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        self.delete_all_button = QtWidgets.QPushButton("Delete All Traces")
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["On", "Dataset", "Parameter", "Color", "Style", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(False)
        header.setMinimumSectionSize(28)
        for column in range(5):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.Fixed)
        for column, width in ((0, 40), (1, 135), (2, 105), (3, 58), (4, 88), (5, 30)):
            self.table.setColumnWidth(column, width)
        header.sectionClicked.connect(self._header_clicked)
        header_tooltips = (
            "Click to sort by visibility; drag the divider to resize",
            "Click to sort by dataset; drag the divider to resize",
            "Click to sort naturally by parameter; drag the divider to resize",
            "Click to group traces by color; drag the divider to resize",
            "Click to group traces by line style; drag the divider to resize",
            "Remove trace",
        )
        for column, tooltip in enumerate(header_tooltips):
            self.table.horizontalHeaderItem(column).setToolTip(tooltip)

        selection_row = QtWidgets.QHBoxLayout()
        selection_row.addWidget(self.selected_label)
        selection_row.addStretch()
        selection_row.addWidget(self.select_visible_button)
        selection_row.addWidget(self.clear_button)
        action_row = QtWidgets.QGridLayout()
        action_row.addWidget(self.show_button, 0, 0)
        action_row.addWidget(self.hide_button, 0, 1)
        action_row.addWidget(self.color_button, 1, 0)
        action_row.addWidget(self.delete_button, 1, 1)
        action_row.addWidget(self.delete_all_button, 2, 0, 1, 2)
        action_row.addWidget(QtWidgets.QLabel("Batch line style"), 3, 0, 1, 2)
        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(self.style_combo, 1)
        style_row.addWidget(self.apply_style_button)
        action_row.addLayout(style_row, 4, 0, 1, 2)
        batch_help = QtWidgets.QLabel("Select one or more rows below, then apply an action to all selected traces.")
        batch_help.setWordWrap(True)
        batch_help.setStyleSheet("color:#91a0ad; font-size:9pt;")
        batch_box = QtWidgets.QGroupBox("BATCH OPERATIONS — SELECTED TRACES")
        batch_layout = QtWidgets.QVBoxLayout(batch_box)
        batch_layout.setContentsMargins(7, 7, 7, 7)
        batch_layout.setSpacing(5)
        batch_layout.addWidget(batch_help)
        batch_layout.addLayout(selection_row)
        batch_layout.addLayout(action_row)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.search)
        layout.addWidget(batch_box)
        layout.addWidget(self.table, 1)

        self.search.textChanged.connect(self._refresh_rows)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.select_visible_button.clicked.connect(self._select_visible)
        self.clear_button.clicked.connect(self.table.clearSelection)
        self.show_button.clicked.connect(lambda: self._batch_visibility(True))
        self.hide_button.clicked.connect(lambda: self._batch_visibility(False))
        self.color_button.clicked.connect(self._batch_color)
        self.apply_style_button.clicked.connect(self._batch_style)
        self.delete_button.clicked.connect(self._batch_delete)
        self.delete_all_button.clicked.connect(self._delete_all_traces)
        self._selection_changed()

    def set_traces(
        self,
        traces: list[TraceConfig],
        datasets: dict[str, Dataset],
        context_id: str | None = None,
    ) -> None:
        same_plot = context_id == self._context_id
        if not same_plot:
            self._highlighted_trace_keys.clear()
        self._context_id = context_id
        self._traces, self._datasets = traces, datasets
        self._highlighted_source_indices = {
            index for index, trace in enumerate(traces)
            if (trace.dataset_id, trace.parameter) in self._highlighted_trace_keys
        }
        self._refresh_rows(preserve_scroll=same_plot)

    def highlight_trace(self, source_index: int) -> None:
        self.highlight_traces([source_index])

    def highlight_traces(self, source_indices: list[int]) -> None:
        valid = {index for index in source_indices if 0 <= index < len(self._traces)}
        self._highlighted_source_indices.update(valid)
        self._highlighted_trace_keys.update(
            (self._traces[index].dataset_id, self._traces[index].parameter) for index in valid
        )
        if not self._highlighted_source_indices:
            return
        self._refresh_rows()
        first = min(self._highlighted_source_indices)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            if item is not None and int(item.data(TraceSortItem.SOURCE_ROLE)) == first:
                current = self.table.model().index(row, 1)
                self.table.selectionModel().setCurrentIndex(current, QtCore.QItemSelectionModel.NoUpdate)
                self.table.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
                break

    def clear_trace_highlight(self) -> None:
        if not self._highlighted_source_indices and not self._highlighted_trace_keys:
            return
        self._highlighted_source_indices.clear()
        self._highlighted_trace_keys.clear()
        self._refresh_rows()

    def _searchable_text(self, trace: TraceConfig) -> str:
        dataset = self._datasets.get(trace.dataset_id)
        if not dataset:
            return f"Missing {trace.parameter}"
        return " ".join((dataset.name, dataset.filename, dataset.path, trace.parameter))

    def _refresh_rows(self, *, preserve_scroll: bool = False) -> None:
        vertical = self.table.verticalScrollBar()
        horizontal = self.table.horizontalScrollBar()
        scroll_x, scroll_y = horizontal.value(), vertical.value()
        query = self.search.text()
        self._row_indices = [
            index for index, trace in enumerate(self._traces)
            if wildcard_search_matches(query, self._searchable_text(trace))
        ]
        for source_index in sorted(self._highlighted_source_indices):
            if source_index not in self._row_indices and 0 <= source_index < len(self._traces):
                self._row_indices.append(source_index)
        if self._sort_column >= 0:
            self._row_indices.sort(
                key=lambda index: self._sort_value(index, self._sort_column),
                reverse=self._sort_order == QtCore.Qt.DescendingOrder,
            )
        self._row_indices.sort(key=lambda index: index not in self._highlighted_source_indices)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for table_row, source_index in enumerate(self._row_indices):
            trace = self._traces[source_index]
            self.table.insertRow(table_row)
            visible = QtWidgets.QCheckBox()
            visible.setChecked(trace.visible)
            visible.stateChanged.connect(
                lambda _state, i=source_index, checkbox=visible: self._set_visible(i, checkbox.isChecked())
            )
            self.table.setItem(table_row, 0, TraceSortItem("", source_index, int(trace.visible)))
            self.table.setCellWidget(table_row, 0, self._centered(visible))
            dataset = self._datasets.get(trace.dataset_id)
            dataset_name = dataset.name if dataset else "Missing"
            dataset_item = TraceSortItem(dataset_name, source_index, natural_sort_key(dataset_name))
            if dataset:
                dataset_item.setToolTip(
                    f"[{dataset.file_type} · {dataset.analysis_label}]\n"
                    f"Dataset: {dataset.name}\nFile: {dataset.filename}\nPath: {dataset.path or '(not saved)'}"
                )
            self.table.setItem(table_row, 1, dataset_item)
            parameter_item = TraceSortItem(trace.parameter, source_index, natural_sort_key(trace.parameter))
            parameter_item.setToolTip(trace.parameter)
            self.table.setItem(table_row, 2, parameter_item)
            color = QtWidgets.QPushButton()
            color.setFixedSize(28, 20)
            color.setStyleSheet(f"background:{trace.color}; border:1px solid #777;")
            color.clicked.connect(lambda _checked=False, i=source_index: self._choose_color(i))
            color_item = TraceSortItem("", source_index, trace.color.casefold())
            color_item.setToolTip(trace.color)
            self.table.setItem(table_row, 3, color_item)
            self.table.setCellWidget(table_row, 3, self._centered(color))
            style = QtWidgets.QComboBox()
            style.addItems(self.STYLES)
            style.setCurrentText(trace.line_style)
            style.currentTextChanged.connect(lambda value, i=source_index: self._set_style(i, value))
            style_rank = self.STYLES.index(trace.line_style) if trace.line_style in self.STYLES else len(self.STYLES)
            self.table.setItem(table_row, 4, TraceSortItem("", source_index, style_rank))
            self.table.setCellWidget(table_row, 4, style)
            remove = QtWidgets.QToolButton(text="×")
            remove.clicked.connect(lambda _checked=False, i=source_index: self.remove_requested.emit(i))
            self.table.setItem(table_row, 5, TraceSortItem("", source_index, source_index))
            self.table.setCellWidget(table_row, 5, self._centered(remove))
            if source_index in self._highlighted_source_indices:
                highlight = QtGui.QBrush(QtGui.QColor("#f4d35e"))
                foreground = QtGui.QBrush(QtGui.QColor("#17212b"))
                for column in range(self.table.columnCount()):
                    item = self.table.item(table_row, column)
                    if item is not None:
                        item.setBackground(highlight)
                        item.setForeground(foreground)
        self.table.blockSignals(False)
        if preserve_scroll:
            self.table.doItemsLayout()
            horizontal.setValue(scroll_x)
            vertical.setValue(scroll_y)
        self._selection_changed()

    def _sort_value(self, source_index: int, column: int):
        trace = self._traces[source_index]
        dataset = self._datasets.get(trace.dataset_id)
        dataset_name = dataset.name if dataset else "Missing"
        if column == 0:
            return int(trace.visible)
        if column == 1:
            return natural_sort_key(dataset_name)
        if column == 2:
            return natural_sort_key(trace.parameter)
        if column == 3:
            return trace.color.casefold()
        if column == 4:
            return self.STYLES.index(trace.line_style) if trace.line_style in self.STYLES else len(self.STYLES)
        return source_index

    def _header_clicked(self, column: int) -> None:
        if column == 5:
            return
        if self._sort_column == column:
            self._sort_order = (
                QtCore.Qt.DescendingOrder
                if self._sort_order == QtCore.Qt.AscendingOrder
                else QtCore.Qt.AscendingOrder
            )
        else:
            self._sort_column = column
            self._sort_order = QtCore.Qt.AscendingOrder
        header = self.table.horizontalHeader()
        header.setSortIndicatorShown(True)
        header.setSortIndicator(self._sort_column, self._sort_order)
        self._refresh_rows()

    @staticmethod
    def _centered(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget, alignment=QtCore.Qt.AlignCenter)
        return holder

    def selected_source_indices(self) -> list[int]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        selected = []
        for row in rows:
            item = self.table.item(row, 1)
            if item is not None:
                selected.append(int(item.data(TraceSortItem.SOURCE_ROLE)))
        return selected

    def _selection_changed(self) -> None:
        count = len(self.selected_source_indices())
        self.selected_label.setText(f"Selected traces: {count}")
        for widget in (
            self.show_button, self.hide_button, self.color_button,
            self.style_combo, self.apply_style_button, self.delete_button,
        ):
            widget.setEnabled(count > 0)
        self.delete_all_button.setEnabled(bool(self._traces))

    def _select_visible(self) -> None:
        self.table.clearSelection()
        selection = QtCore.QItemSelection()
        model = self.table.model()
        for row in range(self.table.rowCount()):
            selection.select(model.index(row, 0), model.index(row, self.table.columnCount() - 1))
        self.table.selectionModel().select(selection, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    def _batch_visibility(self, visible: bool) -> None:
        indices = self.selected_source_indices()
        for index in indices:
            self._traces[index].visible = visible
        if indices:
            self.trace_changed.emit()

    def _batch_style(self) -> None:
        indices = self.selected_source_indices()
        for index in indices:
            self._traces[index].line_style = self.style_combo.currentText()
        if indices:
            self.trace_changed.emit()

    def _batch_color(self) -> None:
        indices = self.selected_source_indices()
        if not indices:
            return
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._traces[indices[0]].color), self)
        if color.isValid():
            for index in indices:
                self._traces[index].color = color.name()
            self.trace_changed.emit()

    def _batch_delete(self) -> None:
        indices = self.selected_source_indices()
        if indices:
            self.remove_many_requested.emit(indices)

    def _delete_all_traces(self) -> None:
        count = len(self._traces)
        if count == 0:
            return
        first = QtWidgets.QMessageBox.question(
            self,
            "Delete all traces",
            f"Delete all {count} trace(s) from the active plot?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if first != QtWidgets.QMessageBox.Yes:
            return
        second = QtWidgets.QMessageBox.question(
            self,
            "Confirm delete all traces",
            "This will remove every trace from the active plot. Confirm again?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if second == QtWidgets.QMessageBox.Yes:
            self.remove_all_requested.emit()

    def _set_visible(self, source_index: int, visible: bool) -> None:
        if source_index >= len(self._traces):
            return
        self._traces[source_index].visible = visible
        self.trace_changed.emit()

    def _set_style(self, source_index: int, value: str) -> None:
        if source_index < len(self._traces):
            self._traces[source_index].line_style = value
            self.trace_changed.emit()

    def _choose_color(self, source_index: int) -> None:
        if source_index >= len(self._traces):
            return
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._traces[source_index].color), self)
        if color.isValid():
            self._traces[source_index].color = color.name()
            self.trace_changed.emit()
