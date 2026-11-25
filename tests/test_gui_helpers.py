import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from core.models import Dataset, MarkerConfig, PlotConfig, TraceConfig
from core.network_transform import transform_dataset
from gui.main_window import DatasetSettingsDialog, MainWindow, dataset_import_signature
from gui.settings_dialogs import GuideDialog, PlotSettingsDialog, load_png_settings, save_png_settings
from gui.main_window import next_clone_name, next_plot_name
from widgets.plot_widget import (
    AnalyzerPlotWidget, CurveStylePreview, ReportLegendDialog, ShiftPanViewBox,
    next_marker_name, nice_tick_step, normalized_pair,
)
from widgets.trace_selector import TraceSelector, TraceSortItem, TraceTable, natural_sort_key


def test_plot_names_use_highest_existing_number_not_list_length():
    plots = [PlotConfig("Plot 1"), PlotConfig("Plot 3"), PlotConfig("Custom")]
    assert next_plot_name(plots) == "Plot 4"


def test_multiple_plots_keep_readable_height_and_use_vertical_scroll():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1100, 720)
    window.show()
    app.processEvents()
    window.add_plot()
    app.processEvents()

    heights = [widget.height() for widget in window.plot_widgets.values()]
    assert len(heights) == 2
    assert all(height >= AnalyzerPlotWidget.MINIMUM_CARD_HEIGHT for height in heights)
    assert window.plot_scroll.verticalScrollBar().maximum() > 0
    window.close()
    assert app is not None


def test_clone_names_prefix_source_and_remain_unique():
    plots = [PlotConfig("Plot 1"), PlotConfig("Clone-Plot 1")]
    assert next_clone_name("Plot 2", plots) == "Clone-Plot 2"
    assert next_clone_name("Plot 1", plots) == "Clone-Plot 1-2"


def test_duplicate_import_signature_uses_path_and_processing_settings():
    frequency = np.array([1e9, 2e9])
    s = np.zeros((2, 4, 4), complex)
    first = Dataset("First", r"C:\Data\same.s4p", frequency, s, 50, network_type="dphy")
    repeated = Dataset("Repeated", r"c:\data\SAME.s4p", frequency, s, 50, network_type="dphy")
    remapped = Dataset(
        "Remapped",
        r"C:\Data\same.s4p",
        frequency,
        s,
        50,
        port_mapping=(2, 1, 3, 4),
        network_type="dphy",
    )

    assert dataset_import_signature(first) == dataset_import_signature(repeated)
    assert dataset_import_signature(first) != dataset_import_signature(remapped)


def test_duplicate_import_button_highlights_without_selecting_or_deleting():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    s = np.zeros((2, 4, 4), complex)
    first = Dataset("First", r"C:\Data\same.s4p", frequency, s, 50, network_type="dphy")
    repeated = Dataset("Repeated", r"c:\data\SAME.s4p", frequency, s, 50, network_type="dphy")
    different_settings = Dataset(
        "Different settings",
        r"C:\Data\same.s4p",
        frequency,
        s,
        50,
        port_mapping=(2, 1, 3, 4),
        network_type="dphy",
    )
    unique = Dataset("Unique", r"C:\Data\unique.s4p", frequency, s, 50, network_type="dphy")
    window = MainWindow()
    window.datasets = {
        item.id: item for item in (first, repeated, different_settings, unique)
    }
    window._refresh_datasets()

    window.highlight_duplicate_imports()

    assert window.highlight_duplicates_button.text() == "Highlight duplicate imports"
    assert window.duplicate_dataset_ids == {first.id, repeated.id}
    assert len(window.datasets) == 4
    assert window.dataset_tree.selectedItems() == []
    items = {
        window.dataset_tree.topLevelItem(row).data(0, QtCore.Qt.UserRole):
        window.dataset_tree.topLevelItem(row)
        for row in range(window.dataset_tree.topLevelItemCount())
    }
    assert items[first.id].background(0).style() != QtCore.Qt.BrushStyle.NoBrush
    assert items[repeated.id].background(1).style() != QtCore.Qt.BrushStyle.NoBrush
    assert items[different_settings.id].background(0).style() == QtCore.Qt.BrushStyle.NoBrush
    assert "1 duplicate group(s), 2 highlighted dataset(s)" in window.status.currentMessage()

    repeated.port_mapping = (1, 3, 2, 4)
    window.highlight_duplicate_imports()
    assert window.duplicate_dataset_ids == set()
    assert all(
        items[dataset_id].background(0).style() == QtCore.Qt.BrushStyle.NoBrush
        for dataset_id in items
    )
    assert window.status.currentMessage() == "No duplicate imports found"
    window.close()
    assert app is not None


def test_reversed_ranges_are_normalized():
    assert normalized_pair(-10, 0) == (-10, 0)
    assert normalized_pair(0, -10) == (-10, 0)
    assert normalized_pair(10e9, 1e9) == (1e9, 10e9)
    assert nice_tick_step(59.3) == 5


def test_marker_names_advance_without_reusing_deleted_numbers():
    markers = [MarkerConfig("M1", 1e9, "#fff"), MarkerConfig("M3", 2e9, "#fff")]
    assert next_marker_name(markers) == "M4"


def test_plot_name_field_keeps_twenty_character_display_width():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = AnalyzerPlotWidget(PlotConfig("Plot 2"), {}, {}, "Dark")
    required_width = widget.name_edit.fontMetrics().averageCharWidth() * 20 + 16
    assert widget.name_edit.minimumWidth() == widget.name_edit.maximumWidth()
    assert widget.name_edit.width() >= required_width
    assert widget.name_edit.maxLength() > 20
    assert app is not None


def test_workspace_plot_roundtrip_preserves_markers():
    trace = TraceConfig("dataset", "S11")
    plot = PlotConfig(
        "Plot 1", traces=[trace], markers=[MarkerConfig("M1", 2.5e9, "#f4d35e", visible=False)],
        frequency_step_ghz=0.5, y_step=5, legend_visible=False,
        legend_font_size=10, title_font_size=14, axis_font_size=12,
        marker_info_expanded=True, marker_info_width=372,
        legend_mode="report", legend_location="below", legend_below_alignment="Right",
        report_legend_labels={trace.id: "Golden"},
    )
    restored = PlotConfig.from_dict(plot.to_dict())
    assert restored.markers[0].name == "M1"
    assert restored.markers[0].frequency_hz == 2.5e9
    assert restored.markers[0].visible is False
    assert restored.frequency_step_ghz == 0.5
    assert restored.y_step == 5
    assert restored.legend_visible is False
    assert (restored.legend_font_size, restored.title_font_size, restored.axis_font_size) == (10, 14, 12)
    assert restored.marker_info_expanded is True
    assert restored.marker_info_width == 372
    assert restored.traces[0].id == trace.id
    assert restored.legend_mode == "report"
    assert restored.legend_location == "below"
    assert restored.legend_below_alignment == "Right"
    assert restored.report_legend_labels[trace.id] == "Golden"


def test_natural_sort_key_orders_numeric_parameter_parts():
    assert sorted(["S11", "S2", "S100"], key=natural_sort_key) == ["S2", "S11", "S100"]


def test_trace_table_sorting_keeps_source_indices_and_resizable_columns():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    s = np.zeros((2, 6, 6), complex)
    zebra = Dataset("Zebra", "z.s6p", frequency, s, 50)
    alpha = Dataset("Alpha", "a.s6p", frequency, s, 50)
    traces = [
        TraceConfig(zebra.id, "S66", visible=True),
        TraceConfig(alpha.id, "S11", visible=False),
    ]
    table = TraceTable()
    table.set_traces(traces, {zebra.id: zebra, alpha.id: alpha})
    assert table.table.horizontalHeader().sectionResizeMode(1) == QtWidgets.QHeaderView.Interactive
    table.table.setColumnWidth(1, 210)
    assert table.table.columnWidth(1) == 210

    table._header_clicked(1)
    assert [table.table.item(row, 1).text() for row in range(2)] == ["Alpha", "Zebra"]
    table.highlight_traces([0])
    assert [table.table.item(row, 1).text() for row in range(2)] == ["Zebra", "Alpha"]
    table.set_traces(traces, {zebra.id: zebra, alpha.id: alpha})
    assert table.table.item(0, 1).text() == "Zebra"
    table.clear_trace_highlight()
    assert [table.table.item(row, 1).text() for row in range(2)] == ["Alpha", "Zebra"]
    table.table.selectRow(0)
    assert table.selected_source_indices() == [1]
    assert table.table.item(0, 1).data(TraceSortItem.SOURCE_ROLE) == 1
    assert app is not None


def test_select_visible_selects_only_filtered_trace_rows():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dataset = Dataset("DUT", "dut.s6p", np.array([1e9, 2e9]), np.zeros((2, 6, 6), complex), 50)
    traces = [TraceConfig(dataset.id, "S11"), TraceConfig(dataset.id, "S22")]
    table = TraceTable()
    table.set_traces(traces, {dataset.id: dataset})
    table.search.setText("S22")
    table._select_visible()
    assert table.selected_source_indices() == [1]
    button_labels = [button.text() for button in table.findChildren(QtWidgets.QPushButton)]
    assert "Select Visible" in button_labels
    assert "Select All" not in button_labels
    assert "Delete All Traces" in button_labels
    assert "Apply Style" in button_labels
    assert "Apply to Selected" not in button_labels
    assert table.style_combo.isEnabled()
    assert table.apply_style_button.isEnabled()
    assert app is not None


def test_delete_all_traces_requires_two_confirmations(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dataset = Dataset("DUT", "dut.s6p", np.array([1e9]), np.zeros((1, 6, 6), complex), 50)
    table = TraceTable()
    table.set_traces([TraceConfig(dataset.id, "S11")], {dataset.id: dataset})
    answers = iter((QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Yes))
    questions = []

    def answer(*args, **kwargs):
        questions.append(args[1])
        return next(answers)

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", answer)
    emitted = []
    table.remove_all_requested.connect(lambda: emitted.append(True))
    table._delete_all_traces()
    assert questions == ["Delete all traces", "Confirm delete all traces"]
    assert emitted == [True]
    assert app is not None


def test_trace_table_find_temporarily_reveals_and_highlights_filtered_trace():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dataset = Dataset("DUT", "dut.s6p", np.array([1e9, 2e9]), np.zeros((2, 6, 6), complex), 50)
    traces = [TraceConfig(dataset.id, "S11"), TraceConfig(dataset.id, "S22")]
    table = TraceTable()
    table.set_traces(traces, {dataset.id: dataset})
    table.search.setText("S11")
    table.highlight_traces([0, 1])
    assert table.search.text() == "S11"
    assert table.table.rowCount() == 2
    highlighted = next(
        row for row in range(table.table.rowCount())
        if table.table.item(row, 1).data(TraceSortItem.SOURCE_ROLE) == 1
    )
    assert table.table.item(highlighted, 1).background().color().name() == "#f4d35e"
    assert all(table.table.item(row, 1).background().color().name() == "#f4d35e" for row in range(2))
    traces.append(TraceConfig(dataset.id, "S33"))
    table.set_traces(traces, {dataset.id: dataset})
    highlighted_keys = {
        (table.table.item(row, 1).data(TraceSortItem.SOURCE_ROLE), table.table.item(row, 2).text())
        for row in range(2)
    }
    assert highlighted_keys == {(0, "S11"), (1, "S22")}
    table.clear_trace_highlight()
    assert table.table.rowCount() == 1
    assert app is not None


def test_ctrl_curve_selection_survives_display_refresh_and_added_trace():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9, 3e9])
    dataset = Dataset("DUT", "dut.s6p", frequency, np.zeros((3, 6, 6), complex), 50)
    transform_dataset(dataset)
    config = PlotConfig("Plot", traces=[TraceConfig(dataset.id, "S11")])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget.toggle_trace_selection(0)
    config.display_format = "Linear Magnitude"
    config.traces.append(TraceConfig(dataset.id, "S22"))
    widget.refresh()
    assert widget.selected_trace_indices == {0}
    assert widget.trace_items[0].opts["pen"].widthF() == 3.4
    assert widget.find_trace_action.isVisible()
    assert len(widget.trace_items) == 2
    unchanged_pen = widget.trace_items[1].opts["pen"]
    widget._set_preview_trace(0)
    assert widget.trace_items[1].opts["pen"] is unchanged_pen
    assert app is not None


def test_legend_r_a_report_mode_edits_text_hides_hidden_and_preserves_highlight(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dataset = Dataset("Long_Dataset_Name", "dut.s4p", frequency, np.zeros((2, 4, 4), complex), 50, network_type="dphy")
    visible = TraceConfig(dataset.id, "S11")
    hidden = TraceConfig(dataset.id, "S22", visible=False)
    config = PlotConfig("Plot", traces=[visible, hidden])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget.select_trace(0)
    assert widget.selected_trace_indices == {0}
    assert widget.legend.mode_button.letter == "R"
    assert widget.legend.dock_button.letter == "↓"
    assert widget.legend.dock_button.isVisible() is False
    assert len(widget.legend.items) == 1
    assert "#ffe36e" in widget.legend.items[0][1].item.toHtml()
    analysis_legend_width = widget.legend.boundingRect().width()

    monkeypatch.setattr("widgets.plot_widget.ReportLegendDialog.exec", lambda _dialog: QtWidgets.QDialog.Accepted)
    monkeypatch.setattr(
        "widgets.plot_widget.ReportLegendDialog.values",
        lambda _dialog: {visible.id: "Golden Sample", hidden.id: "Must stay hidden"},
    )
    widget.toggle_legend_mode()
    assert config.legend_mode == "report"
    assert widget.legend.mode_button.letter == "A"
    assert widget.selected_trace_indices == {0}
    assert len(widget.legend.items) == 1
    report_html = widget.legend.items[0][1].item.toHtml()
    assert "Golden Sample" in report_html
    assert "#ffe36e" not in report_html
    report_sample = widget.legend.items[0][0]
    assert report_sample.acceptedMouseButtons() == QtCore.Qt.MouseButton.NoButton

    class FakeLegendClick:
        def __init__(self):
            self.accepted = False

        @staticmethod
        def button():
            return QtCore.Qt.MouseButton.LeftButton

        def accept(self):
            self.accepted = True

    report_click = FakeLegendClick()
    report_sample.mouseClickEvent(report_click)
    assert report_click.accepted
    assert widget.trace_items[0].isVisible() is True
    widget.legend.layout.activate()
    assert widget.legend.boundingRect().width() < analysis_legend_width
    widget.apply_theme("Light")
    light_report_html = widget.legend.items[0][1].item.toHtml()
    assert "Golden Sample" in light_report_html
    assert "#9b2c00" not in light_report_html
    assert widget.legend.mode_button.light is True

    widget.toggle_legend_mode()
    assert config.legend_mode == "analysis"
    assert widget.legend.mode_button.letter == "R"
    assert widget.selected_trace_indices == {0}
    analysis_html = widget.legend.items[0][1].item.toHtml()
    assert "Long_Dataset_Name: S11" in analysis_html
    assert "#9b2c00" in analysis_html
    analysis_sample = widget.legend.items[0][0]
    assert analysis_sample.acceptedMouseButtons() == QtCore.Qt.MouseButton.LeftButton
    analysis_click = FakeLegendClick()
    analysis_sample.mouseClickEvent(analysis_click)
    assert analysis_click.accepted
    assert widget.trace_items[0].isVisible() is False
    widget.trace_items[0].setVisible(True)
    widget.legend.layout.activate()
    assert widget.legend.boundingRect().width() >= analysis_legend_width - 1
    assert app is not None


def test_floating_legend_shrinks_for_short_report_labels_with_all_traces_visible(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dataset = Dataset(
        "Long_Dataset_Name_Used_For_Analysis_Legend_Labels",
        "dut.s4p",
        frequency,
        np.zeros((2, 4, 4), complex),
        50,
        network_type="dphy",
    )
    first, second = (TraceConfig(dataset.id, parameter) for parameter in ("S11", "S22"))
    config = PlotConfig("Plot", traces=[first, second])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget.resize(1100, 680)
    widget.show()
    app.processEvents()

    analysis_width = widget.legend.width()
    original_samples = [sample for sample, _ in widget.legend.items]
    monkeypatch.setattr("widgets.plot_widget.ReportLegendDialog.exec", lambda _: QtWidgets.QDialog.Accepted)
    monkeypatch.setattr(
        "widgets.plot_widget.ReportLegendDialog.values",
        lambda _: {first.id: "35厚度", second.id: "43厚度"},
    )

    widget.toggle_legend_mode()
    app.processEvents()
    report_width = widget.legend.width()
    assert report_width < analysis_width / 2
    assert [sample for sample, _ in widget.legend.items] == original_samples
    assert [label.text for _, label in widget.legend.items] == ["35厚度", "43厚度"]
    first_label = widget.legend.items[0][1]
    text_right = first_label.mapRectToParent(first_label.itemRect()).right()
    assert text_right < widget.legend.mode_button.pos().x()

    widget.toggle_legend_mode()
    app.processEvents()
    assert np.isclose(widget.legend.width(), analysis_width)
    widget.toggle_legend_mode()
    app.processEvents()
    assert np.isclose(widget.legend.width(), report_width)
    widget.close()


def test_cancel_report_legend_editor_keeps_analysis_mode(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dataset = Dataset(
        "DUT", "dut.s4p", frequency, np.zeros((2, 4, 4), complex), 50,
        network_type="dphy",
    )
    config = PlotConfig("Plot", traces=[TraceConfig(dataset.id, "S11")])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    original_labels = dict(config.report_legend_labels)
    monkeypatch.setattr(
        "widgets.plot_widget.ReportLegendDialog.exec",
        lambda _dialog: QtWidgets.QDialog.Rejected,
    )

    widget.toggle_legend_mode()

    assert config.legend_mode == "analysis"
    assert widget.legend.mode_button.letter == "R"
    assert config.report_legend_labels == original_labels
    assert app is not None


def test_legend_can_dock_below_frequency_wrap_rows_and_use_plot_setting_font():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dataset = Dataset("DUT", "dut.s4p", frequency, np.zeros((2, 4, 4), complex), 50, network_type="dphy")
    traces = [
        TraceConfig(dataset.id, f"S{response}{stimulus}", color=f"#{index + 1:06x}", line_style="Dash" if index % 2 else "Solid")
        for index, (response, stimulus) in enumerate(
            (response, stimulus) for response in range(1, 5) for stimulus in range(1, 5)
        )
    ]
    config = PlotConfig("Plot", traces=traces)
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget.resize(620, 520)
    widget.show()
    app.processEvents()
    plot_height = widget.plot.height()

    widget.toggle_legend_location()
    app.processEvents()
    assert config.legend_location == "below"
    assert widget.legend.isVisible() is False
    assert widget.docked_legend.isVisible() is True
    assert widget.docked_legend.dock_button.text() == "↑"
    assert widget.docked_legend.mode_button.isVisible() is False
    assert widget.docked_legend.dock_button.isVisible() is False
    assert len({item[0].top() for item in widget.docked_legend._layout_cache}) >= 2
    assert min(item[0].top() for item in widget.docked_legend._layout_cache) <= 6
    scene_rect = widget.view_box.sceneBoundingRect()
    plot_origin = widget.plot.mapTo(widget.docked_legend, QtCore.QPoint(0, 0)).x()
    expected_left = plot_origin + round(scene_rect.left())
    expected_right = plot_origin + round(scene_rect.right())
    assert abs(widget.docked_legend.content_left - expected_left) <= 2
    assert abs(widget.docked_legend.content_right - min(expected_right, widget.docked_legend.width())) <= 2
    assert widget.plot.height() >= plot_height - 2

    # A window/card resize must replace stale bounds instead of leaving only
    # a narrow strip beside the R/A and dock controls.
    widget.docked_legend.content_left = widget.docked_legend.width() - 40
    widget.docked_legend.content_right = widget.docked_legend.width()
    widget.resize(760, 520)
    app.processEvents()
    scene_rect = widget.view_box.sceneBoundingRect()
    plot_origin = widget.plot.mapTo(widget.docked_legend, QtCore.QPoint(0, 0)).x()
    expected_left = plot_origin + round(scene_rect.left())
    expected_right = plot_origin + round(scene_rect.right())
    assert abs(widget.docked_legend.content_left - expected_left) <= 2
    assert abs(widget.docked_legend.content_right - min(expected_right, widget.docked_legend.width())) <= 2
    assert widget.docked_legend.content_right - widget.docked_legend.content_left > 300
    assert widget.docked_legend.mode_button.isVisible() is False
    assert widget.docked_legend.dock_button.isVisible() is False

    widget.set_font_sizes(14, config.title_font_size, config.axis_font_size)
    app.processEvents()
    assert widget.docked_legend.font_size == 14
    assert widget.docked_legend.height() >= 64
    config.legend_below_alignment = "Right"
    widget._refresh_docked_legend()
    assert widget.docked_legend.alignment == "Right"

    widget.toggle_legend_location()
    app.processEvents()
    assert config.legend_location == "plot"
    assert widget.docked_legend.isVisible() is False
    assert widget.legend.isVisible() is True
    assert widget.legend.dock_button.letter == "↓"
    assert app is not None


def test_docked_legend_can_hide_curve_and_report_mode_omits_hidden_entry():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    values = np.zeros((2, 4, 4), complex)
    values[:, 0, 0] = (0.2, 0.3)
    values[:, 1, 1] = (0.4, 0.5)
    dataset = Dataset("DUT", "dut.s4p", frequency, values, 50, network_type="dphy")
    traces = [
        TraceConfig(dataset.id, "S11", color="#55c8ff"),
        TraceConfig(dataset.id, "S22", color="#ff8866"),
    ]
    config = PlotConfig("Plot", traces=traces)
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget.resize(720, 520)
    widget.show()
    widget.toggle_legend_location()
    app.processEvents()

    first_entry_widget = widget.docked_legend._entry_widgets[0]
    QtTest.QTest.mouseClick(
        first_entry_widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=first_entry_widget.rect().center(),
    )
    app.processEvents()
    assert widget.trace_items[0].isVisible() is False
    assert traces[0].id in widget.legend_hidden_trace_ids
    assert len(widget.docked_legend.entries) == 2
    assert widget.docked_legend.entries[0][-1] is False

    config.legend_mode = "report"
    widget._refresh_legend_labels()
    app.processEvents()
    assert [entry[0] for entry in widget.docked_legend.entries] == [1]
    report_entry_widget = widget.docked_legend._entry_widgets[1]
    QtTest.QTest.mouseClick(
        report_entry_widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=report_entry_widget.rect().center(),
    )
    app.processEvents()
    assert widget.trace_items[1].isVisible() is True
    report_editor = ReportLegendDialog(
        config, {dataset.id: dataset}, widget, widget.legend_hidden_trace_ids
    )
    assert report_editor.table.rowCount() == 1
    assert report_editor.table.item(0, 1).text() == "DUT: S22"

    widget.toggle_legend_location()
    app.processEvents()
    assert len(widget.legend.items) == 1
    report_sample, report_label = widget.legend.items[0]
    assert report_sample.item is widget.trace_items[1]
    assert report_sample.isVisible() is True
    assert report_label.isVisible() is True
    assert report_sample.acceptedMouseButtons() == QtCore.Qt.MouseButton.NoButton

    config.legend_mode = "analysis"
    widget._refresh_legend_labels()
    app.processEvents()
    assert len(widget.legend.items) == 2
    assert all(sample.isVisible() and label.isVisible() for sample, label in widget.legend.items)
    assert all(
        sample.acceptedMouseButtons() == QtCore.Qt.MouseButton.LeftButton
        for sample, _label in widget.legend.items
    )
    assert widget.trace_items[0].isVisible() is False
    assert app is not None


def test_legend_hidden_curve_is_omitted_from_marker_readings():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    values = np.zeros((2, 4, 4), complex)
    values[:, 0, 0] = (0.2, 0.3)
    values[:, 1, 1] = (0.4, 0.5)
    dataset = Dataset("DUT", "dut.s4p", frequency, values, 50, network_type="dphy")
    traces = [TraceConfig(dataset.id, "S11"), TraceConfig(dataset.id, "S22")]
    marker = MarkerConfig("M1", 1e9, "#f4d35e")
    config = PlotConfig("Plot", traces=traces, markers=[marker])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    app.processEvents()
    assert [reading[0] for reading in widget.marker_cards[marker.id].readings] == [0, 1]

    widget.trace_items[0].setVisible(False)
    app.processEvents()
    assert traces[0].id in widget.legend_hidden_trace_ids
    assert [reading[0] for reading in widget.marker_cards[marker.id].readings] == [1]

    widget.trace_items[0].setVisible(True)
    app.processEvents()
    assert [reading[0] for reading in widget.marker_cards[marker.id].readings] == [0, 1]
    assert app is not None


def test_marker_names_follow_analysis_and_report_legend_modes(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dataset = Dataset(
        "Long_DUT_Name", "dut.s4p", frequency, np.zeros((2, 4, 4), complex), 50,
        network_type="dphy",
    )
    trace = TraceConfig(dataset.id, "S11")
    marker = MarkerConfig("M1", 1e9, "#f4d35e")
    config = PlotConfig("Plot", traces=[trace], markers=[marker])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    analysis_text = widget.marker_cards[marker.id].readout.text()
    assert "Long_DUT_Name" in analysis_text and "S11" in analysis_text

    monkeypatch.setattr(
        "widgets.plot_widget.ReportLegendDialog.exec",
        lambda _dialog: QtWidgets.QDialog.Accepted,
    )
    monkeypatch.setattr(
        "widgets.plot_widget.ReportLegendDialog.values",
        lambda _dialog: {trace.id: "Report Lane A"},
    )
    widget.toggle_legend_mode()
    report_text = widget.marker_cards[marker.id].readout.text()
    assert "Report Lane A" in report_text
    assert "Long_DUT_Name" not in report_text

    widget.toggle_legend_mode()
    restored_text = widget.marker_cards[marker.id].readout.text()
    assert "Long_DUT_Name" in restored_text and "S11" in restored_text
    assert app is not None


def test_saved_below_report_legend_realigns_after_initial_layout_settles():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    values = np.zeros((2, 4, 4), complex)
    values[:, 0, 0] = (0.2, 0.3)
    dataset = Dataset("DUT", "dut.s4p", frequency, values, 50, network_type="dphy")
    config = PlotConfig(
        "Plot",
        traces=[TraceConfig(dataset.id, "S11")],
        legend_mode="report",
        legend_location="below",
    )
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Light")
    widget.resize(1200, 650)
    widget.show()
    QtTest.QTest.qWait(220)
    app.processEvents()

    legend = widget.docked_legend
    assert legend.isVisible()
    assert len(legend.entries) == 1
    assert legend.content_left < legend.width() * 0.25
    assert legend.content_right - legend.content_left > legend.width() * 0.70
    assert legend.mode_button.isVisible() is False
    assert legend.dock_button.isVisible() is False
    line_rect, text_rect = legend._layout_cache[0][:2]
    assert line_rect.left() >= legend.content_left
    assert text_rect.right() < legend.content_right - 54

    # Guard against the real nested-splitter failure: even if Qt briefly
    # reports only a 40 px ViewBox strip at the far right, the Legend must
    # recover to a usable axis-width frame rather than keeping that geometry.
    original_scene_bounds = widget.view_box.sceneBoundingRect
    widget.view_box.sceneBoundingRect = lambda: QtCore.QRectF(
        widget.plot.width() - 40, 0, 40, 100
    )
    widget._update_docked_legend_bounds()
    widget.view_box.sceneBoundingRect = original_scene_bounds
    assert legend.content_left < legend.width() * 0.25
    assert legend.content_right - legend.content_left > legend.width() * 0.70
    assert app is not None


def test_below_legend_height_does_not_accumulate_when_switching_modes():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    values = np.zeros((2, 4, 4), complex)
    dataset = Dataset(
        "Long_Dataset_Name_For_Legend_Wrapping",
        "dut.s4p",
        frequency,
        values,
        50,
        network_type="dphy",
    )
    traces = [
        TraceConfig(dataset.id, f"S{response}{stimulus}")
        for response in range(1, 5)
        for stimulus in range(1, 5)
    ]
    config = PlotConfig("Plot", traces=traces, legend_location="below")
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Light")
    widget.resize(520, 520)
    widget.show()
    QtTest.QTest.qWait(220)
    app.processEvents()

    analysis_height = widget.docked_legend.height()
    analysis_minimum = widget.minimumHeight()
    for index in range(1, len(traces)):
        widget.trace_items[index].setVisible(False)
    app.processEvents()

    report_minimum = None
    for _ in range(5):
        config.legend_mode = "report"
        widget._refresh_legend_labels()
        app.processEvents()
        assert len(widget.docked_legend.entries) == 1
        current_report_minimum = widget.minimumHeight()
        report_minimum = report_minimum or current_report_minimum
        assert current_report_minimum == report_minimum

        config.legend_mode = "analysis"
        widget._refresh_legend_labels()
        app.processEvents()
        assert len(widget.docked_legend.entries) == len(traces)
        assert widget.docked_legend.height() == analysis_height
        assert widget.minimumHeight() == analysis_minimum
    assert app is not None


def test_report_legend_editor_shows_read_only_curve_color_and_line_style():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dataset = Dataset("DUT", "dut.s4p", frequency, np.zeros((2, 4, 4), complex), 50, network_type="dphy")
    trace = TraceConfig(dataset.id, "S11", color="#ff0000", line_style="Dash")
    hidden = TraceConfig(dataset.id, "S22", color="#00ff00", line_style="Dot", visible=False)
    config = PlotConfig("Plot", traces=[trace, hidden])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    dialog = ReportLegendDialog(config, {dataset.id: dataset}, widget)
    assert dialog.table.rowCount() == 1
    assert dialog.table.horizontalHeaderItem(0).text() == "Curve"
    preview = dialog.table.cellWidget(0, 0).findChild(CurveStylePreview)
    assert preview.color == "#ff0000"
    assert preview.line_style == "Dash"
    assert preview.testAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    trace.color = "#123456"
    trace.line_style = "Dash Dot"
    refreshed = ReportLegendDialog(config, {dataset.id: dataset}, widget)
    refreshed_preview = refreshed.table.cellWidget(0, 0).findChild(CurveStylePreview)
    assert refreshed_preview.color == "#123456"
    assert refreshed_preview.line_style == "Dash Dot"
    assert app is not None


def test_export_hides_legend_controls_and_keeps_theme_right_margin():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    values = np.zeros((2, 4, 4), complex)
    values[:, 0, 0] = (0.2, 0.4)
    dataset = Dataset("DUT", "dut.s4p", frequency, values, 50, network_type="dphy")
    config = PlotConfig("Plot", traces=[TraceConfig(dataset.id, "S11")], legend_location="below")
    options = {
        "theme": "Dark", "include_markers": True, "include_marker_info": True,
        "size_mode": "Custom size", "preset_size": "1200 × 675", "width": 500, "height": 320,
    }
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, options, "Dark")
    widget.resize(700, 520)
    widget.show()
    app.processEvents()
    original_renderer = widget._render_at_native_size
    states = []
    widget._render_at_native_size = lambda size: (
        states.append((
            widget.legend.mode_button.isVisible(),
            widget.legend.dock_button.isVisible(),
            widget.docked_legend.mode_button.isVisible(),
            widget.docked_legend.dock_button.isVisible(),
        )) or QtGui.QPixmap(size)
    )
    widget.render_pixmap()
    widget._render_at_native_size = original_renderer
    assert states == [(False, False, False, False)]
    assert widget.docked_legend.mode_button.isVisible() is False
    assert widget.docked_legend.dock_button.isVisible() is False

    pixmap = widget.render_pixmap()
    assert (pixmap.width(), pixmap.height()) == (500, 320)
    image = pixmap.toImage()
    for x in range(480, 500):
        for y in range(0, 320, 20):
            assert image.pixelColor(x, y).name() == "#111820"
    widget.export_settings["theme"] = "Light"
    light_image = widget.render_pixmap().toImage()
    for x in range(480, 500):
        for y in range(0, 320, 20):
            assert light_image.pixelColor(x, y).name() == "#ffffff"
    assert app is not None


def test_trace_selector_adds_all_phy_entries_and_disables_rotate_for_dphy():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    datasets = {}
    for index in range(2):
        item = Dataset(f"C{index}", f"c{index}.s6p", frequency, np.zeros((2, 6, 6), complex), 50)
        datasets[item.id] = item
    for index in range(2):
        item = Dataset(
            f"D{index}", f"d{index}.s4p", frequency, np.zeros((2, 4, 4), complex), 50,
            (1, 2, 3, 4), network_type="dphy", impedance_preset="DIFF nominal",
        )
        datasets[item.id] = item
    selector = TraceSelector()
    selector.set_datasets(datasets)
    labels = [selector.dataset_combo.itemText(i) for i in range(selector.dataset_combo.count())]
    assert any("ALL-C-PHY" in label for label in labels)
    assert any("ALL-DIFF" in label for label in labels)
    assert any("ALL-S4P · SE" in label for label in labels)
    assert any("ALL-S6P · SE" in label for label in labels)
    dphy_index = selector.dataset_combo.findData("ALL:dphy")
    selector.dataset_combo.setCurrentIndex(dphy_index)
    assert selector.trace_type.currentData() == "dphy"
    assert selector.r_filter.isEnabled() is False
    assert selector.r_filter.font().strikeOut() is True
    assert selector.response_filter.font().strikeOut() is False
    assert selector.response_filter.itemText(0) == "Output Mode: Any"
    assert selector.stimulus_filter.itemText(0) == "Input Mode: Any"
    selector.trace_type.setCurrentIndex(selector.trace_type.findData("se"))
    for control in (selector.response_filter, selector.stimulus_filter, selector.route_filter, selector.r_filter):
        assert control.isEnabled() is False
        assert control.font().strikeOut() is True
    selector.trace_type.setCurrentIndex(selector.trace_type.findData("dphy"))
    selector.parameter_list.setCurrentRow(0)
    batches = []
    selector.traces_requested.connect(lambda requests: batches.append(requests))
    selector._emit_selected()
    assert len(batches) == 1 and len(batches[0]) == 2
    assert app is not None


def test_se_only_datasets_offer_only_se_and_same_port_batch_group():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    datasets = {}
    for index in range(2):
        item = Dataset(
            f"SE{index}", f"se{index}.s3p", frequency, np.zeros((2, 3, 3), complex), 50,
            (1, 2, 3), network_type="se", impedance_preset="SE only",
        )
        datasets[item.id] = item
    selector = TraceSelector()
    selector.set_datasets(datasets)
    group_index = selector.dataset_combo.findData("ALL:se:3")
    assert group_index >= 0
    selector.dataset_combo.setCurrentIndex(group_index)
    assert selector.trace_type.count() == 1
    assert selector.trace_type.currentData() == "se"
    assert selector.parameter_list.count() == 9
    dialog = DatasetSettingsDialog(next(iter(datasets.values())))
    assert len(dialog.spins) == 3
    assert dialog.modal_spins == []
    assert dialog.impedance_settings[2] == "SE only"
    assert app is not None


def test_batch_add_traces_redraws_only_once():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    redraws = []
    window._active_plot_changed = lambda: redraws.append(True)
    requests = [(f"dataset-{index}", "Sdd11") for index in range(120)]
    window.add_traces(requests)
    assert len(window.active_plot().traces) == 120
    assert redraws == [True]
    assert not window.import_progress.isHidden()
    assert window.import_progress.value() == 100
    assert window.import_progress.format() == "Add Trace complete"
    window.add_traces(requests)
    assert redraws == [True]
    window._finish_add_trace_progress()
    assert not window.import_progress.isVisible()
    assert app is not None


def test_dataset_import_batches_ui_commit_without_redrawing_plots():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    loader = object()
    window._loader_refs.append(loader)
    dataset_refreshes = []
    plot_refreshes = []
    window._refresh_datasets = lambda refresh_trace_table=True: dataset_refreshes.append(refresh_trace_table)
    window._refresh_all_plots = lambda: plot_refreshes.append(True)
    frequency = np.array([1e9, 2e9])
    first = Dataset("Batch", "a.s4p", frequency, np.zeros((2, 4, 4), complex), 50, network_type="dphy")
    second = Dataset("Batch", "b.s4p", frequency, np.zeros((2, 4, 4), complex), 50, network_type="dphy")

    window._dataset_loaded(first)
    window._dataset_loaded(second)
    assert len(window.datasets) == 2
    assert second.name == "Batch_2"
    assert dataset_refreshes == []
    assert plot_refreshes == []

    window._loading_finished(loader)
    assert dataset_refreshes == [False]
    assert plot_refreshes == []
    assert window._pending_imported_dataset_count == 0
    assert app is not None


def test_large_plot_uses_interactive_quality_then_restores_idle_detail():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.linspace(1e9, 3e9, 101)
    values = np.zeros((frequency.size, 4, 4), complex)
    values[:, 0, 0] = 0.1
    dataset = Dataset("DUT", "dut.s4p", frequency, values, 50, network_type="dphy")
    config = PlotConfig(
        "Plot", traces=[
            TraceConfig(dataset.id, "S11", color=f"#{index + 1:06x}")
            for index in range(AnalyzerPlotWidget.INTERACTIVE_TRACE_THRESHOLD)
        ],
    )
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget._plot_view_range_changed()
    assert {item.opts["autoDownsampleFactor"] for item in widget.trace_items.values()} == {0.4}
    assert {item.opts["antialias"] for item in widget.trace_items.values()} == {False}
    widget._restore_idle_plot_quality()
    assert {item.opts["autoDownsampleFactor"] for item in widget.trace_items.values()} == {1.5}
    assert {item.opts["antialias"] for item in widget.trace_items.values()} == {True}
    widget.set_curve_render_quality("High quality")
    widget._plot_view_range_changed()
    assert {item.opts["autoDownsampleFactor"] for item in widget.trace_items.values()} == {1.5}
    assert {item.opts["antialias"] for item in widget.trace_items.values()} == {True}
    widget.set_curve_render_quality("Performance")
    assert {item.opts["autoDownsampleFactor"] for item in widget.trace_items.values()} == {0.4}
    assert {item.opts["antialias"] for item in widget.trace_items.values()} == {False}
    export_states = []
    widget._render_at_native_size = lambda _size: (
        export_states.append({
            (item.opts["autoDownsampleFactor"], item.opts["antialias"])
            for item in widget.trace_items.values()
        }) or QtGui.QPixmap(10, 10)
    )
    widget.render_pixmap()
    assert export_states == [{(1.5, True)}]
    assert {item.opts["autoDownsampleFactor"] for item in widget.trace_items.values()} == {0.4}
    assert {item.opts["antialias"] for item in widget.trace_items.values()} == {False}
    assert app is not None


def test_first_marker_keeps_information_collapsed_until_user_opens_it():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    values = np.zeros((2, 4, 4), complex)
    dataset = Dataset("DUT", "dut.s4p", frequency, values, 50, network_type="dphy")
    config = PlotConfig("Plot", traces=[TraceConfig(dataset.id, "S11")])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget.resize(1000, 600)
    widget.show()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False

    widget.add_marker()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    assert widget.marker_expand_button.isVisible()
    widget.marker_expand_button.click()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is True
    assert widget.marker_collapse_button.text() == "▲"
    widget.marker_collapse_button.click()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    assert widget.marker_expand_button.isVisible()
    assert not widget.capture_widget.isAncestorOf(widget.marker_expand_button)

    widget.add_marker()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    widget.marker_expand_button.click()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is True
    assert app is not None


def test_marker_info_expands_plot_card_without_squeezing_graph():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dataset = Dataset(
        "DUT", "dut.s4p", frequency, np.zeros((2, 4, 4), complex), 50,
        network_type="dphy",
    )
    config = PlotConfig("Plot", traces=[TraceConfig(dataset.id, "S11")])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget.resize(1000, 600)
    widget.show()
    app.processEvents()
    collapsed_card_width = widget.width()
    collapsed_graph_width = widget.plot.width()

    widget.add_marker()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    widget.marker_expand_button.click()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is True
    assert widget.width() >= collapsed_card_width + widget.config.marker_info_width
    assert widget.plot.width() >= collapsed_graph_width - 2
    assert widget.base_width_without_marker_info() == collapsed_card_width

    widget.marker_collapse_button.click()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    assert widget.minimumWidth() < collapsed_card_width + widget.config.marker_info_width
    assert app is not None


def test_frozen_plot_keeps_action_buttons_fixed_when_marker_info_expands():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dataset = Dataset(
        "DUT", "dut.s4p", frequency, np.zeros((2, 4, 4), complex), 50,
        network_type="dphy",
    )
    config = PlotConfig("Plot", traces=[TraceConfig(dataset.id, "S11")])
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, {}, "Dark")
    widget.resize(1000, 600)
    widget.show()
    app.processEvents()
    widget.set_plot_width_frozen(True)
    app.processEvents()
    assert [button.text() for button in widget.plot_action_buttons] == [
        "Apply", "Autoscale", "+ Marker", "Legend", "C",
    ]
    frozen_toolbar_width = widget.toolbar_widget.width()
    before = [button.mapTo(widget, QtCore.QPoint(0, 0)).x() for button in widget.plot_action_buttons]

    widget.add_marker()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    widget.marker_expand_button.click()
    app.processEvents()
    after = [button.mapTo(widget, QtCore.QPoint(0, 0)).x() for button in widget.plot_action_buttons]

    assert widget.marker_splitter.panel_expanded() is True
    assert widget.toolbar_widget.width() == frozen_toolbar_width
    assert after == before
    widget.set_plot_width_frozen(False)
    assert app is not None


def test_marker_info_outward_expansion_uses_main_horizontal_scrollbar():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1000, 720)
    window.show()
    app.processEvents()
    widget = window.plot_widgets[window.active_plot_id]
    widget.add_marker()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    widget.marker_expand_button.click()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is True
    assert window.plot_scroll.horizontalScrollBar().maximum() > 0
    window.close()
    assert app is not None


def test_plot_export_settings_and_axis_steps(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9, 3e9])
    dataset = Dataset("DUT", "dut.s6p", frequency, np.zeros((3, 6, 6), complex), 50)
    transform_dataset(dataset)
    config = PlotConfig(
        "Plot", traces=[TraceConfig(dataset.id, "Sxx11_1R")],
        markers=[MarkerConfig("M1", 2e9, "#f4d35e")],
        frequency_range_hz=(1e9, 3e9), y_range=(-10, 0), frequency_step_ghz=0.5, y_step=2,
        marker_info_expanded=False,
    )
    options = {
        "theme": "Light", "include_markers": False, "include_marker_info": False,
        "size_mode": "Preset size", "preset_size": "1200 × 675", "width": 1600, "height": 900,
    }
    widget = AnalyzerPlotWidget(config, {dataset.id: dataset}, options, "Dark")
    widget.resize(1200, 600)
    widget.show()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    assert widget.marker_expand_button.text() == "▼ Marker Info"
    assert widget.marker_expand_button.isVisible()
    assert widget.marker_expand_button.parentWidget() is widget.marker_info_toolbar_row
    assert widget.marker_info_toolbar_row.height() == 30
    assert not widget.capture_widget.isAncestorOf(widget.marker_expand_button)
    widget.marker_expand_button.click()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is True
    assert widget.config.marker_info_expanded is True
    assert widget.marker_splitter.sizes()[1] >= 310
    assert widget.marker_expand_button.isVisible() is False
    assert widget.marker_collapse_button.text() == "▲"
    widget.marker_collapse_button.click()
    app.processEvents()
    assert widget.marker_splitter.panel_expanded() is False
    assert widget.config.marker_info_expanded is False
    assert widget.marker_splitter.sizes()[1] == 0
    render_progress = []
    widget.refresh_progress.connect(lambda completed, total: render_progress.append((completed, total)))
    widget.refresh()
    assert render_progress[-1] == (1, 1)
    assert widget.trace_items[0].opts["clipToView"] is True
    assert widget.trace_items[0].opts["autoDownsample"] is True
    assert widget.trace_items[0].opts["autoDownsampleFactor"] == 1.5
    assert widget.trace_items[0].opts["antialias"] is True
    assert widget.trace_items[0].opts["downsampleMethod"] == "peak"

    original_native_renderer = widget._render_at_native_size
    export_panel_states = []
    widget.export_settings["include_marker_info"] = True
    widget._render_at_native_size = lambda _size: (
        export_panel_states.append(widget.marker_splitter.panel_expanded()) or QtGui.QPixmap(10, 10)
    )
    widget.render_pixmap()
    widget._render_at_native_size = original_native_renderer
    assert export_panel_states == [True]
    assert widget.marker_splitter.panel_expanded() is False

    class LegendDragEvent:
        def __init__(self, position, last_position, start=False, finish=False):
            self._position = QtCore.QPointF(*position)
            self._last_position = QtCore.QPointF(*last_position)
            self._start = start
            self._finish = finish

        def button(self):
            return QtCore.Qt.MouseButton.LeftButton

        def accept(self):
            pass

        def pos(self):
            return self._position

        def lastPos(self):
            return self._last_position

        def isStart(self):
            return self._start

        def isFinish(self):
            return self._finish

    legend_before = QtCore.QPointF(widget.legend.pos())
    widget.legend.mouseDragEvent(LegendDragEvent((30, 18), (0, 0), start=True))
    widget.legend.mouseDragEvent(LegendDragEvent((30, 18), (30, 18), finish=True))
    assert widget.legend.pos() != legend_before
    cached_once = widget._formatted_trace(dataset, "Sxx11_1R")
    assert widget._formatted_trace(dataset, "Sxx11_1R") is cached_once
    transform_dataset(dataset)
    assert widget._formatted_trace(dataset, "Sxx11_1R") is not cached_once
    assert "Reset L" not in [button.text() for button in widget.findChildren(QtWidgets.QToolButton)]
    context_menu = widget.plot.getViewBox().menu
    visible_actions = [action for action in context_menu.actions() if action.isVisible() and not action.isSeparator()]
    assert [action.text() for action in visible_actions] == ["Plot Options", "Average"]
    plot_options = visible_actions[0].menu()
    assert "Average" not in [action.text() for action in plot_options.actions()]
    assert widget.plot.scene().contextMenu == []
    assert isinstance(widget.plot.getViewBox(), ShiftPanViewBox)
    widget._set_preview_trace(0)
    assert widget.preview_trace_index == 0
    assert widget.trace_items[0].opts["pen"].widthF() == 2.6
    preview_html = widget.legend.items[0][1].item.toHtml()
    assert "#8be9fd" in preview_html and "font-weight:700" in preview_html
    marker_card = next(iter(widget.marker_cards.values()))
    QtTest.QTest.qWait(120)
    assert "#8be9fd" in marker_card.readout.text()
    widget.apply_theme("Light")
    assert "#005a9c" in widget.legend.items[0][1].item.toHtml()
    assert "#005a9c" in marker_card.readout.text()
    widget.apply_theme("Dark")
    widget._set_preview_trace(None)
    assert widget.trace_items[0].opts["pen"].widthF() == 1.7

    class TraceClickEvent:
        def __init__(self, button, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
            self._button = button
            self._modifiers = modifiers

        def button(self):
            return self._button

        def modifiers(self):
            return self._modifiers

    widget._trace_clicked(0, TraceClickEvent(QtCore.Qt.MouseButton.LeftButton))
    widget._trace_clicked(0, TraceClickEvent(QtCore.Qt.MouseButton.RightButton))
    assert widget.selected_trace_indices == set()
    widget._trace_clicked(0, TraceClickEvent(
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.ControlModifier
    ))
    assert widget.selected_trace_indices == {0}
    assert widget.trace_items[0].opts["pen"].widthF() == 3.4
    assert widget.find_trace_action.isVisible()
    assert widget.clear_trace_action.isVisible()
    selected_legend_html = widget.legend.items[0][1].item.toHtml()
    assert "#ffe36e" in selected_legend_html and "font-weight:700" in selected_legend_html
    QtTest.QTest.qWait(120)
    assert "#ffe36e" in marker_card.readout.text()
    widget.apply_theme("Light")
    assert "#9b2c00" in widget.legend.items[0][1].item.toHtml()
    assert "#9b2c00" in marker_card.readout.text()
    widget.apply_theme("Dark")

    widget._find_selected_trace()
    assert widget.find_result_active is True
    widget._set_preview_trace(0)
    assert widget.trace_items[0].opts["pen"].widthF() == 4.2
    assert widget.trace_items[0].opts["pen"].color().name() == "#ff6b6b"
    assert "#ff6b6b" in widget.legend.items[0][1].item.toHtml()
    QtTest.QTest.qWait(120)
    assert "#ff6b6b" in marker_card.readout.text()
    widget.apply_theme("Light")
    assert widget.trace_items[0].opts["pen"].color().name() == "#b42318"
    assert "#b42318" in widget.legend.items[0][1].item.toHtml()
    assert "#b42318" in marker_card.readout.text()
    widget.apply_theme("Dark")
    widget._trace_clicked(0, TraceClickEvent(
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.ControlModifier
    ))
    assert widget.selected_trace_indices == set()
    assert widget.clear_trace_action.isVisible()
    widget._set_preview_trace(None)
    widget._plot_mouse_clicked(TraceClickEvent(QtCore.Qt.MouseButton.LeftButton))
    assert widget.selected_trace_indices == set()
    widget.clear_trace_action.trigger()
    QtTest.QTest.qWait(120)
    assert widget.selected_trace_indices == set()
    assert widget.find_result_active is False
    assert widget.trace_items[0].opts["pen"].widthF() == 1.7
    assert not widget.find_trace_action.isVisible()
    assert not widget.clear_trace_action.isVisible()
    assert "#d8e0e8" in widget.legend.items[0][1].item.toHtml()
    assert "#ffe36e" not in marker_card.readout.text()
    parent_bounds = widget.legend.parentItem().boundingRect().adjusted(8, 8, -8, -8)
    requested = widget.legend._clamped_position(QtCore.QPointF(5000, 5000))
    widget.legend.autoAnchor(requested, relative=False)
    app.processEvents()
    legend_bounds = widget.legend.mapRectToParent(widget.legend.boundingRect())
    assert legend_bounds.right() <= parent_bounds.right() + 0.1
    assert legend_bounds.bottom() <= parent_bounds.bottom() + 0.1
    widget.apply_theme("Light")
    assert widget.legend.opts["brush"].color().name() == "#ffffff"
    assert widget.legend.opts["pen"].color().name() == "#8e9aa5"
    widget.apply_theme("Dark")
    assert widget.legend.opts["brush"].color().name() == "#0e151c"
    image = QtGui.QImage(1300, 700, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    x_labels = [item[2] for item in widget.x_axis.generateDrawSpecs(painter)[2]]
    y_labels = [item[2] for item in widget.y_axis.generateDrawSpecs(painter)[2]]
    painter.end()
    assert ("1.0" in x_labels or "1" in x_labels) and ("3.0" in x_labels or "3" in x_labels)
    assert "-10" in y_labels and "0" in y_labels
    original_text = (widget.freq_step.text(), widget.y_step.text())
    widget.autoscale()
    assert widget.x_axis._tickSpacing is None
    assert widget.y_axis._tickSpacing is None
    assert widget.x_axis.requested_endpoints == (1.0, 3.0)
    assert widget.y_axis.requested_endpoints is not None
    assert widget.y_axis.requested_endpoints[1] == 0
    assert not hasattr(widget, "zero_line")
    assert widget.plot.plotItem.autoBtn._width == 36
    five_chars = max(48, min(60, widget.freq_min.fontMetrics().averageCharWidth() * 5 + 12))
    assert widget.freq_min.width() == five_chars
    home_x, home_y = tuple(widget.plot.viewRange()[0]), tuple(widget.plot.viewRange()[1])
    widget.plot.setXRange(1.4, 1.8, padding=0)
    widget.plot.setYRange(-5, -2, padding=0)
    widget.fit_view()
    np.testing.assert_allclose(widget.plot.viewRange()[0], home_x)
    np.testing.assert_allclose(widget.plot.viewRange()[1], home_y)
    widget.set_font_size("legend", 12)
    widget.set_font_size("title", 16)
    widget.set_font_size("axis", 14)
    app.processEvents()
    assert (config.legend_font_size, config.title_font_size, config.axis_font_size) == (12, 16, 14)
    legend_html = widget.legend.items[0][1].item.toHtml()
    assert "font-size:12pt" in legend_html and "#d8e0e8" in legend_html
    assert "font-size:14pt" in widget.x_axis.label.toHtml()
    assert "#d8e0e8" in widget.x_axis.label.toHtml()
    widget.apply_theme("Light")
    assert "#202830" in widget.x_axis.label.toHtml()
    assert "#202830" in widget.legend.items[0][1].item.toHtml()
    widget.apply_theme("Dark")
    assert (widget.freq_step.text(), widget.y_step.text()) == original_text
    widget.apply_ranges(silent=True)
    assert widget.x_axis._tickSpacing is not None
    assert widget.y_axis._tickSpacing is not None
    for field in (widget.freq_min, widget.freq_max, widget.y_min, widget.y_max):
        field.clear()
    widget.apply_ranges(silent=True)
    assert widget.x_axis.requested_endpoints == (1.0, 3.0)
    assert widget.y_axis.requested_endpoints is not None
    assert widget.y_axis.requested_endpoints[1] == 0
    assert widget.freq_step.text() == original_text[0]
    assert widget.y_step.text() == original_text[1]
    image = QtGui.QImage(1300, 700, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(image)
    x_labels = [item[2] for item in widget.x_axis.generateDrawSpecs(painter)[2]]
    y_labels = [item[2] for item in widget.y_axis.generateDrawSpecs(painter)[2]]
    painter.end()
    assert ("1.0" in x_labels or "1" in x_labels) and ("3.0" in x_labels or "3" in x_labels)
    assert "0" in y_labels
    original_capture_size = widget.capture_widget.size()
    pixmap = widget.render_pixmap()
    assert (pixmap.width(), pixmap.height()) == (1200, 675)
    assert widget.capture_widget.size() == original_capture_size
    widget.export_settings.update({"size_mode": "Custom size", "width": 900, "height": 900})
    square_pixmap = widget.render_pixmap()
    assert (square_pixmap.width(), square_pixmap.height()) == (900, 900)
    assert widget.capture_widget.size() == original_capture_size
    assert widget.plot.viewRange()[1][1] >= 0
    widget.set_legend_visible(False)
    assert widget.legend.isVisible() is False
    output = tmp_path / "plot.png"
    widget.export_png(output)
    assert output.exists() and output.stat().st_size > 0
    assert app is not None


def test_png_settings_persist_in_qsettings(tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    values = load_png_settings(settings)
    values.update({
        "theme": "Light", "width": 1920, "include_markers": False,
        "curve_render_quality": "High quality",
    })
    save_png_settings(settings, values)
    settings.sync()
    restored = load_png_settings(settings)
    assert restored["theme"] == "Light"
    assert restored["width"] == 1920
    assert restored["include_markers"] is False
    assert restored["curve_render_quality"] == "High quality"


def test_plot_settings_exposes_active_plot_fonts_and_apply_all():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = PlotSettingsDialog(
        load_png_settings(QtCore.QSettings()),
        {"legend": 12, "title": 16, "axis": 14},
    )
    assert dialog.windowTitle() == "Plot Settings — Fonts / Rendering / PNG / Clipboard"
    assert dialog.legend_font.value() == 12
    assert dialog.title_font.value() == 16
    assert dialog.axis_font.value() == 14
    assert dialog.curve_render_quality.currentText() == "Auto (Recommended)"
    dialog.apply_fonts_to_all.setChecked(True)
    dialog.legend_below_alignment.setCurrentText("Distribute evenly")
    assert dialog.plot_font_values() == {
        "legend": 12, "title": 16, "axis": 14,
        "legend_below_alignment": "Distribute evenly", "apply_to_all": True,
    }
    assert app is not None


def test_main_window_exposes_v1_import_theme_guide_and_plot_settings_actions():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    labels = {action.text() for action in window.findChildren(QtGui.QAction)}
    assert "v1.2" in window.windowTitle()
    assert {"Import SNP", "UnLock", "Guide", "Plot Settings", "Export Plot PNG", "Export Trace CSV"} <= labels
    assert "Import S6P" not in labels and "Import S4P" not in labels
    toolbar_labels = [action.text() for action in window.findChildren(QtWidgets.QToolBar)[0].actions()]
    assert toolbar_labels[:4] == [
        "Import SNP", "Save Workspace", "Load Workspace", "Add Plot",
    ]
    right_side = [
        "UnLock", "Plot Settings", "Export Plot PNG", "Export Trace CSV",
        "☀ Light Mode", "Guide",
    ]
    assert [toolbar_labels.index(label) for label in right_side] == sorted(
        toolbar_labels.index(label) for label in right_side
    )
    control_dock = window.findChild(QtWidgets.QDockWidget, "controlDock")
    assert control_dock.features() == QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
    assert control_dock.allowedAreas() == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
    assert app is not None


def test_guide_documents_plot_mouse_controls():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    guide = GuideDialog()
    text = "\n".join(browser.toPlainText() for browser in guide.findChildren(QtWidgets.QTextBrowser))
    for instruction in (
        "Hold Ctrl + move",
        "Ctrl + left-click",
        "Shift + left-drag",
        "Mouse wheel",
            "Lock / UnLock",
        "Main Toolbar",
        "Save Workspace",
        "FIT",
        "Find Selected Trace(s)",
        "Clear Selected Traces",
        "Legend R/A mode",
    ):
        assert instruction in text
    assert app is not None


def test_inactive_plot_wheel_is_forwarded_to_main_scroll():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = AnalyzerPlotWidget(PlotConfig("Plot 1"), {}, {}, "Dark")
    forwarded = []
    widget.wheel_scroll_requested.connect(forwarded.append)

    class WheelEvent:
        accepted = False

        def type(self):
            return QtCore.QEvent.Type.Wheel

        def angleDelta(self):
            return QtCore.QPoint(0, 120)

        def pixelDelta(self):
            return QtCore.QPoint()

        def accept(self):
            self.accepted = True

    event = WheelEvent()
    widget.set_active(False)
    assert widget.eventFilter(widget.plot.viewport(), event) is True
    assert forwarded == [120]
    assert event.accepted is True
    assert app is not None


def test_freeze_action_locks_plot_width_until_unfrozen():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    widget = window.plot_widgets[window.active_plot_id]
    original_width = widget.width()
    unlock_button_width = window.freeze_button.width()

    window.freeze_action.setChecked(True)
    assert window.plots_frozen is True
    assert window.freeze_action.text() == "Lock"
    locked_icon_key = window.freeze_action.icon().cacheKey()
    assert not window.freeze_action.icon().isNull()
    assert window.freeze_button.width() == unlock_button_width
    assert widget.plot_width_is_frozen() is True
    assert widget.minimumWidth() == widget.maximumWidth() == original_width

    window.resize(window.width() + 240, window.height())
    app.processEvents()
    assert widget.width() == original_width

    window.freeze_action.setChecked(False)
    app.processEvents()
    assert window.plots_frozen is False
    assert window.freeze_action.text() == "UnLock"
    assert not window.freeze_action.icon().isNull()
    assert window.freeze_action.icon().cacheKey() != locked_icon_key
    assert window.freeze_button.width() == unlock_button_width
    assert widget.plot_width_is_frozen() is False
    assert widget.maximumWidth() > original_width
    window.close()
    assert app is not None


def test_plot_list_go_clone_and_double_confirm_delete(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    source = window.active_plot()
    source.traces.append(TraceConfig("shared-dataset", "S11", color="#123456", line_style="Dash"))
    source.markers.append(MarkerConfig("M1", 2e9, "#f4d35e"))
    source.frequency_range_hz = (1e9, 3e9)
    source.legend_mode = "report"
    source.legend_location = "below"
    source.legend_below_alignment = "Right"
    source.report_legend_labels[source.traces[0].id] = "Golden"
    source_id = source.id
    window._refresh_plot_list()
    row = window.plot_list.itemWidget(window.plot_list.item(0))
    assert row.findChild(QtWidgets.QToolButton, "plotGoButton").text() == "Go"
    assert row.findChild(QtWidgets.QToolButton, "plotCloneButton").text() == "Clone"
    assert row.findChild(QtWidgets.QToolButton, "plotDeleteButton").text() == "×"

    window.add_plot()
    window.go_to_plot(source_id)
    assert window.active_plot_id == source_id

    confirmations = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question",
        lambda *args, **kwargs: confirmations.append(args[1]) or QtWidgets.QMessageBox.Yes,
    )
    window.clone_plot(source_id)
    assert len(confirmations) == 1
    assert len(window.plots) == 3
    clone = window.active_plot()
    assert clone.id != source.id and clone.name != source.name
    assert clone.name == f"Clone-{source.name}"
    assert clone.traces[0].id != source.traces[0].id
    assert clone.traces[0].dataset_id == source.traces[0].dataset_id
    assert clone.markers[0].id != source.markers[0].id
    assert clone.legend_location == "below"
    assert clone.legend_below_alignment == "Right"
    assert clone.report_legend_labels[clone.traces[0].id] == "Golden"

    answers = iter((QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Cancel))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: next(answers))
    window.delete_plot(clone.id)
    assert len(window.plots) == 3

    answers = iter((QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Yes))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: next(answers))
    window.delete_plot(clone.id)
    assert len(window.plots) == 2
    window.close()
    assert app is not None


def test_single_dataset_list_orders_cphy_before_dphy_then_by_name():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frequency = np.array([1e9, 2e9])
    dphy = Dataset(
        "D-Lane", "d.s4p", frequency, np.zeros((2, 4, 4), complex), 50,
        (1, 2, 3, 4), network_type="dphy", impedance_preset="DIFF nominal",
    )
    cphy_b = Dataset("C-Zebra", "cz.s6p", frequency, np.zeros((2, 6, 6), complex), 50)
    cphy_a = Dataset("C-Alpha", "ca.s6p", frequency, np.zeros((2, 6, 6), complex), 50)
    window = MainWindow()
    window.datasets = {dphy.id: dphy, cphy_b.id: cphy_b, cphy_a.id: cphy_a}
    window._refresh_datasets()
    labels = [window.dataset_tree.topLevelItem(row).text(0) for row in range(3)]
    assert labels == [
            "[S6P · C-PHY/SE] C-Alpha",
            "[S6P · C-PHY/SE] C-Zebra",
            "[S4P · DIFF/SE] D-Lane",
    ]
    assert app is not None
