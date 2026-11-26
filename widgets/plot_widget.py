from __future__ import annotations

import html
import math
import re
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from core.models import Dataset, MarkerConfig, PlotConfig
from core.network_transform import extract_dataset
from core.sparameter_math import DisplayFormat, axis_label, format_trace

STYLE_MAP = {
    "Solid": QtCore.Qt.SolidLine,
    "Dash": QtCore.Qt.DashLine,
    "Dot": QtCore.Qt.DotLine,
    "Dash Dot": QtCore.Qt.DashDotLine,
}
MARKER_COLORS = ("#f4d35e", "#ff6b6b", "#7bdff2", "#b2f7ef", "#cdb4db", "#f7a072", "#a7c957", "#f15bb5")


class EndpointAxisItem(pg.AxisItem):
    """Axis that keeps explicitly requested range endpoints visible."""

    def __init__(self, orientation: str) -> None:
        super().__init__(orientation=orientation)
        self.requested_endpoints: tuple[float, float] | None = None

    def set_requested_endpoints(self, low: float | None, high: float | None) -> None:
        self.requested_endpoints = None if low is None or high is None else (float(low), float(high))
        self.update()

    def generateDrawSpecs(self, painter):
        specs = super().generateDrawSpecs(painter)
        if specs is None or self.requested_endpoints is None or self.orientation not in {"bottom", "left"}:
            return specs
        axis_spec, tick_specs, text_specs = specs
        bounds = self.mapRectFromParent(self.geometry())
        bounding = self.boundingRect()
        range_low, range_high = self.range
        difference = range_high - range_low
        if difference == 0:
            return specs
        spacing = abs(self.requested_endpoints[1] - self.requested_endpoints[0]) / 10 or 1.0
        existing = {text for _rect, _flags, text in text_specs}
        for value in self.requested_endpoints:
            if value < min(range_low, range_high) - 1e-12 or value > max(range_low, range_high) + 1e-12:
                continue
            text = self.tickStrings([value], self.autoSIPrefixScale * self.scale, spacing)[0]
            if text in existing:
                continue
            text_rect = painter.boundingRect(
                QtCore.QRectF(0, 0, 120, 100), QtCore.Qt.AlignmentFlag.AlignCenter, text
            )
            text_rect.setHeight(text_rect.height() * 0.8)
            offset = max(0, self.style["tickLength"]) + self.style["tickTextOffset"][1 if self.orientation == "bottom" else 0]
            if self.orientation == "bottom":
                position = bounds.left() + (value - range_low) / difference * bounds.width()
                rect = QtCore.QRectF(
                    position - text_rect.width() / 2,
                    bounds.top() + offset,
                    text_rect.width(),
                    text_rect.height(),
                )
                rect.moveLeft(max(bounding.left(), min(rect.left(), bounding.right() - rect.width())))
                flags = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
            else:
                position = bounds.bottom() - (value - range_low) / difference * bounds.height()
                rect = QtCore.QRectF(
                    bounds.right() - offset - text_rect.width(),
                    position - text_rect.height() / 2,
                    text_rect.width(),
                    text_rect.height(),
                )
                rect.moveTop(max(bounding.top(), min(rect.top(), bounding.bottom() - rect.height())))
                flags = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            text_specs.append((rect, flags | QtCore.Qt.TextFlag.TextDontClip, text))
            existing.add(text)
        return axis_spec, tick_specs, text_specs


class ShiftPanViewBox(pg.ViewBox):
    """Reserve clicks for traces and pan only with Shift + left drag."""

    def mouseDragEvent(self, event, axis=None) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                previous_mode = self.state["mouseMode"]
                self.state["mouseMode"] = self.PanMode
                try:
                    super().mouseDragEvent(event, axis=axis)
                finally:
                    self.state["mouseMode"] = previous_mode
                return
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            event.accept()
            return
        super().mouseDragEvent(event, axis=axis)


class LegendModeButton(QtWidgets.QGraphicsObject):
    clicked = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.letter = "R"
        self.light = False
        self.hovered = False
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.setToolTip("R: switch to Report mode and edit Legend text")

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, 18, 18)

    def paint(self, painter: QtGui.QPainter, _option, _widget=None) -> None:
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        background = "#dce9f2" if self.light and self.hovered else "#ffffff" if self.light else "#324456" if self.hovered else "#263341"
        border = "#7f909f" if self.light else "#607386"
        foreground = "#17212b" if self.light else "#f1f5f8"
        painter.setPen(pg.mkPen(border, width=1))
        painter.setBrush(pg.mkBrush(background))
        painter.drawRoundedRect(self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(pg.mkPen(foreground))
        painter.drawText(self.boundingRect(), QtCore.Qt.AlignCenter, self.letter)

    def set_mode(self, report_mode: bool, light: bool) -> None:
        self.set_symbol(
            "A" if report_mode else "R",
            "A: return to Analysis Legend" if report_mode else "R: switch to Report mode and edit Legend text",
            light,
        )

    def set_symbol(self, letter: str, tooltip: str, light: bool) -> None:
        self.letter = letter
        self.light = bool(light)
        self.setToolTip(tooltip)
        self.update()

    def hoverEnterEvent(self, event) -> None:
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self.boundingRect().contains(event.pos()):
            self.clicked.emit()
        event.accept()


class ConstrainedLegendItem(pg.LegendItem):
    """Movable legend whose frame always remains inside its parent ViewBox."""

    def __init__(self, *args, margin: float = 8.0, mode_toggle: bool = False, **kwargs) -> None:
        self.margin = float(margin)
        self._drag_target: QtCore.QPointF | None = None
        self._drag_clock = QtCore.QElapsedTimer()
        self.mode_button: LegendModeButton | None = None
        self.dock_button: LegendModeButton | None = None
        super().__init__(*args, **kwargs)
        # Keep the rows at their natural size inside a clipped viewport.
        old_layout = self.layout
        horizontal_spacing = old_layout.horizontalSpacing()
        vertical_spacing = old_layout.verticalSpacing()
        margins = old_layout.getContentsMargins()
        self.setLayout(None)
        self._clip = QtWidgets.QGraphicsRectItem(self)
        self._clip.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self._clip.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape)
        self._contents = QtWidgets.QGraphicsWidget(self._clip)
        self.layout = QtWidgets.QGraphicsGridLayout()
        self.layout.setHorizontalSpacing(horizontal_spacing)
        self.layout.setVerticalSpacing(vertical_spacing)
        self.layout.setContentsMargins(*margins)
        self._contents.setLayout(self.layout)
        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.scrollbar.setFocusPolicy(QtCore.Qt.NoFocus)
        self.scrollbar.setFixedWidth(12)
        self.scrollbar.setSingleStep(24)
        self.scrollbar.valueChanged.connect(self._scroll_contents)
        self._scroll_proxy = QtWidgets.QGraphicsProxyWidget(self)
        self._scroll_proxy.setWidget(self.scrollbar)
        self._scroll_proxy.setZValue(2)
        self._scroll_proxy.hide()
        self._controls_visible = True
        self._layout_margins_without_controls = self.layout.getContentsMargins()
        if mode_toggle:
            left, top, right, bottom = self._layout_margins_without_controls
            self.layout.setContentsMargins(left, top, right + 22, bottom)
            self.mode_button = LegendModeButton(self)
            self.dock_button = LegendModeButton(self)
            self.dock_button.set_symbol("↓", "Move Legend below Frequency", False)
            self.dock_button.hide()
            self.updateSize()
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def set_controls_visible(self, visible: bool) -> None:
        self._controls_visible = visible
        if self.mode_button is None or self.dock_button is None:
            return
        left, top, right, bottom = self._layout_margins_without_controls
        self.layout.setContentsMargins(left, top, right + (22 if visible else 0), bottom)
        self.mode_button.setVisible(visible)
        self.dock_button.hide()
        self.layout.invalidate()
        self.layout.activate()
        self.updateSize()

    def updateSize(self) -> None:
        if self.size is None:
            preferred = self.layout.effectiveSizeHint(QtCore.Qt.SizeHint.PreferredSize)
            parent = self.parentItem()
            available = parent.boundingRect().height() - 2 * self.margin if parent else 0
            height = min(preferred.height(), available) if available > 0 else preferred.height()
            self._contents.resize(preferred)
            self.resize(preferred.width(), height)
            self._clip.setRect(self.boundingRect())
            self.scrollbar.setPageStep(max(1, int(height)))
            self.scrollbar.setRange(0, max(0, math.ceil(preferred.height() - height)))
            show_scrollbar = self._controls_visible and self.scrollbar.maximum() > 0
            scroll_top = 23 if self.mode_button is not None else 2
            self._scroll_proxy.setGeometry(QtCore.QRectF(
                preferred.width() - 15, scroll_top, 12, max(1, height - scroll_top - 3)
            ))
            self._scroll_proxy.setVisible(show_scrollbar)
        if self.mode_button is not None:
            rect = self.boundingRect()
            self.mode_button.setPos(rect.right() - 20, rect.top() + 2)
        if self.dock_button is not None:
            self.dock_button.hide()

    def _scroll_contents(self, position: int) -> None:
        self._contents.setY(-position)

    def scroll_at(self, scene_position: QtCore.QPointF, delta: int) -> bool:
        if (not delta or not self.isVisible() or not self.scrollbar.maximum()
                or not self.boundingRect().contains(self.mapFromScene(scene_position))):
            return False
        step = self.scrollbar.singleStep()
        distance = max(1, round(abs(delta) * step / 120))
        self.scrollbar.setValue(self.scrollbar.value() + (distance if delta < 0 else -distance))
        return True

    def _clamped_position(self, requested) -> QtCore.QPointF:
        requested = QtCore.QPointF(requested)
        parent = self.parentItem()
        if parent is None:
            return requested
        parent_rect = parent.boundingRect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        candidate = self.mapRectToParent(self.boundingRect()).translated(requested - self.pos())
        dx = 0.0
        dy = 0.0
        if candidate.width() <= parent_rect.width():
            if candidate.left() < parent_rect.left():
                dx = parent_rect.left() - candidate.left()
            elif candidate.right() > parent_rect.right():
                dx = parent_rect.right() - candidate.right()
        else:
            dx = parent_rect.left() - candidate.left()
        if candidate.height() <= parent_rect.height():
            if candidate.top() < parent_rect.top():
                dy = parent_rect.top() - candidate.top()
            elif candidate.bottom() > parent_rect.bottom():
                dy = parent_rect.bottom() - candidate.bottom()
        else:
            dy = parent_rect.top() - candidate.top()
        return requested + QtCore.QPointF(dx, dy)

    def ensure_inside(self) -> None:
        try:
            parent = self.parentItem()
        except RuntimeError:
            # A queued post-layout check may outlive a plot deleted during a
            # rapid refresh or test teardown.
            return
        if parent is None:
            return
        self.updateSize()
        clamped = self._clamped_position(self.pos())
        if clamped != self.pos():
            self.autoAnchor(clamped, relative=False)

    def mouseDragEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            event.accept()
            if event.isStart() or self._drag_target is None:
                self._drag_target = QtCore.QPointF(self.pos())
                self._drag_clock.start()
            self._drag_target += event.pos() - event.lastPos()
            if event.isFinish():
                self.autoAnchor(self._clamped_position(self._drag_target), relative=False)
                self._drag_target = None
            elif event.isStart() or self._drag_clock.elapsed() >= 16:
                # LegendItem remains anchored to its parent; setPos() is
                # immediately overwritten by that anchor. Updating the anchor
                # at most once per display frame keeps dragging responsive.
                self.autoAnchor(self._drag_target, relative=False)
                self._drag_clock.restart()


class DockedLegendLineSample(QtWidgets.QWidget):
    """Qt line sample used by the stable below-plot Legend panel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.color = "#4fc3f7"
        self.line_style = "Solid"
        self.entry_visible = True
        self.setFixedSize(32, 24)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

    def set_content(self, color: str, line_style: str, visible: bool) -> None:
        self.color = color
        self.line_style = line_style
        self.entry_visible = bool(visible)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        color = QtGui.QColor(self.color)
        if not self.entry_visible:
            color.setAlpha(90)
        pen = QtGui.QPen(color, 2.2, STYLE_MAP.get(self.line_style, QtCore.Qt.SolidLine))
        pen.setCapStyle(QtCore.Qt.FlatCap)
        painter.setPen(pen)
        painter.drawLine(2, self.height() // 2, self.width() - 2, self.height() // 2)


class DockedLegendEntry(QtWidgets.QFrame):
    clicked = QtCore.Signal(int)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self.interactive = True
        self.sample = DockedLegendLineSample()
        self.label = QtWidgets.QLabel()
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setWordWrap(True)
        self.label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 6, 0)
        layout.setSpacing(4)
        layout.addWidget(self.sample, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label, 0, QtCore.Qt.AlignVCenter)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def set_content(
        self, color: str, line_style: str, text: str, text_color: str,
        bold: bool, visible: bool, font_size: int, interactive: bool,
    ) -> None:
        self.interactive = bool(interactive)
        self.sample.set_content(color, line_style, visible)
        self.label.setText(text)
        font = QtGui.QFont(self.font())
        font.setPointSize(font_size)
        font.setBold(bool(bold))
        self.label.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        label_width = max(54, min(320, metrics.horizontalAdvance(text) + 8))
        self.label.setFixedWidth(label_width)
        color_value = QtGui.QColor(text_color)
        if not visible:
            color_value.setAlpha(105)
        self.label.setStyleSheet(f"color:{color_value.name(QtGui.QColor.HexArgb)}; background:transparent;")
        self.setCursor(QtCore.Qt.PointingHandCursor if interactive else QtCore.Qt.ArrowCursor)
        self.adjustSize()

    def mousePressEvent(self, event) -> None:
        if self.interactive and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
            event.accept()
            return
        super().mousePressEvent(event)


class DockedLegendWidget(QtWidgets.QWidget):
    """Real Qt wrapping panel shown beneath the PlotWidget."""

    mode_clicked = QtCore.Signal()
    dock_clicked = QtCore.Signal()
    entry_clicked = QtCore.Signal(int)
    height_changed = QtCore.Signal(int)
    geometry_changed = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.entries = []
        self.font_size = 9
        self.theme = "Dark"
        self.report_mode = False
        self.alignment = "Center"
        self.content_left = 0
        self.content_right = 120
        # Bottom controls are temporarily removed from the UI. The floating
        # Legend still owns the R/A and dock controls; keep these objects only
        # for API/export compatibility until the bottom controls are redesigned.
        self.controls_visible = False
        self._rebuilding = False
        self._entry_widgets: dict[int, DockedLegendEntry] = {}
        self._layout_cache = []

        self.frame = QtWidgets.QFrame()
        self.frame.setObjectName("dockedLegendFrame")
        self.frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.rows_layout = QtWidgets.QVBoxLayout(self.frame)
        self.rows_layout.setContentsMargins(8, 4, 8, 4)
        self.rows_layout.setSpacing(4)
        self.outer_layout = QtWidgets.QHBoxLayout(self)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.addWidget(self.frame)

        self.mode_button = QtWidgets.QToolButton(self.frame)
        self.mode_button.setFixedSize(24, 22)
        self.mode_button.clicked.connect(self.mode_clicked)
        self.mode_button.hide()
        self.dock_button = QtWidgets.QToolButton(self.frame)
        self.dock_button.setFixedSize(24, 22)
        self.dock_button.setText("↑")
        self.dock_button.setToolTip("Return Legend to the plot")
        self.dock_button.clicked.connect(self.dock_clicked)
        self.dock_button.hide()
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(120)

    def set_content(self, entries, font_size: int, report_mode: bool, theme: str, alignment: str = "Center") -> None:
        self.entries = list(entries)
        self.font_size = int(font_size)
        self.report_mode = bool(report_mode)
        self.theme = theme
        self.alignment = alignment
        self.mode_button.setText("A" if report_mode else "R")
        self.mode_button.setToolTip(
            "A: return to Analysis Legend" if report_mode
            else "R: switch to Report mode and edit Legend text"
        )
        self._sync_entry_widgets()
        self._apply_theme()
        self._rebuild_rows()

    def _sync_entry_widgets(self) -> None:
        active_indices = {entry[0] for entry in self.entries}
        for index in list(self._entry_widgets):
            if index not in active_indices:
                widget = self._entry_widgets.pop(index)
                widget.setParent(None)
                widget.deleteLater()
        foreground = "#202830" if self.theme == "Light" else "#d8e0e8"
        for index, color, line_style, text, text_color, bold, visible in self.entries:
            widget = self._entry_widgets.get(index)
            if widget is None:
                widget = DockedLegendEntry(index, self.frame)
                widget.clicked.connect(self.entry_clicked)
                self._entry_widgets[index] = widget
            widget.set_content(
                color, line_style, text, text_color or foreground, bold, visible,
                self.font_size, not self.report_mode,
            )

    def _apply_theme(self) -> None:
        light = self.theme == "Light"
        self.frame.setStyleSheet(
            "QFrame#dockedLegendFrame {"
            f"background:{'#ffffff' if light else '#0e151c'};"
            f"border:1px solid {'#8e9aa5' if light else '#5a6875'}; border-radius:3px;"
            "} QFrame#dockedLegendFrame QLabel { border:none; background:transparent; }"
        )

    def set_horizontal_bounds(self, left: int, right: int) -> None:
        left = max(0, min(int(left), max(0, self.width() - 80)))
        right = max(left + 80, min(int(right), self.width()))
        self.content_left, self.content_right = left, right
        self.outer_layout.setContentsMargins(left, 0, max(0, self.width() - right), 0)
        self._rebuild_rows()

    def set_controls_visible(self, visible: bool) -> None:
        self.controls_visible = False
        self.mode_button.hide()
        self.dock_button.hide()
        self._rebuild_rows()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.geometry_changed.emit()
        QtCore.QTimer.singleShot(0, self._rebuild_rows)

    def _clear_rows(self) -> None:
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            row_layout = item.layout()
            if row_layout is not None:
                while row_layout.count():
                    row_layout.takeAt(0)
                row_layout.deleteLater()

    def _wrapped_rows(self, available: int):
        ordered = [self._entry_widgets[entry[0]] for entry in self.entries if entry[0] in self._entry_widgets]
        rows, current, used = [], [], 0
        for widget in ordered:
            row_index = len(rows)
            row_available = max(80, available - (58 if row_index == 0 and self.controls_visible else 0))
            width = widget.sizeHint().width()
            required = width if not current else width + 6
            if current and used + required > row_available:
                rows.append(current)
                current, used = [], 0
            current.append(widget)
            used += width if len(current) == 1 else width + 6
        if current or not rows:
            rows.append(current)
        return rows

    def _add_aligned_row(self, row, row_index: int) -> None:
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        if self.alignment in ("Center", "Right"):
            layout.addStretch(1)
        for position, widget in enumerate(row):
            layout.addWidget(widget, 0, QtCore.Qt.AlignVCenter)
            # Newly restored Analysis entries must participate in sizeHint()
            # immediately, including before Qt processes pending show events.
            widget.show()
            if self.alignment == "Distribute evenly" and position < len(row) - 1:
                layout.addStretch(1)
        if self.alignment in ("Center", "Left") or not row:
            layout.addStretch(1)
        if row_index == 0 and self.controls_visible:
            layout.addWidget(self.mode_button, 0, QtCore.Qt.AlignTop)
            layout.addWidget(self.dock_button, 0, QtCore.Qt.AlignTop)
        self.rows_layout.addLayout(layout)

    def _rebuild_rows(self) -> None:
        if self._rebuilding or self.width() <= 0:
            return
        self._rebuilding = True
        try:
            self._clear_rows()
            available = max(80, self.content_right - self.content_left - 16)
            rows = self._wrapped_rows(available)
            for row_index, row in enumerate(rows):
                self._add_aligned_row(row, row_index)
            self.rows_layout.activate()
            desired = max(36, self.rows_layout.sizeHint().height() + self.frame.frameWidth() * 2)
            if self.height() != desired:
                self.setFixedHeight(desired)
                self.height_changed.emit(desired)
            QtCore.QTimer.singleShot(0, self._update_layout_cache)
        finally:
            self._rebuilding = False

    def _update_layout_cache(self) -> None:
        cache = []
        entry_map = {entry[0]: entry for entry in self.entries}
        for index, widget in self._entry_widgets.items():
            if index not in entry_map or not widget.isVisible():
                continue
            entry = entry_map[index]
            sample_top_left = widget.sample.mapTo(self, QtCore.QPoint(0, 0))
            label_top_left = widget.label.mapTo(self, QtCore.QPoint(0, 0))
            sample_rect = QtCore.QRect(sample_top_left, widget.sample.size())
            label_rect = QtCore.QRect(label_top_left, widget.label.size())
            cache.append((sample_rect, label_rect, *entry[1:6], index, entry[6]))
        self._layout_cache = cache


class MarkerSplitter(QtWidgets.QSplitter):
    """Horizontal plot/marker splitter that remembers its right-panel width."""

    panel_state_changed = QtCore.Signal(bool, int)

    def __init__(self, parent=None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setChildrenCollapsible(False)
        self.setHandleWidth(6)
        self._expanded = False
        self._marker_available = False
        self._last_panel_width = 350
        self._applying_state = False
        self.splitterMoved.connect(self._splitter_moved)

    def set_marker_available(self, available: bool) -> None:
        self._marker_available = bool(available)
        if not available and self._expanded:
            self.set_panel_expanded(False)

    def panel_expanded(self) -> bool:
        return self._expanded

    def panel_width(self) -> int:
        return self._last_panel_width

    def set_panel_expanded(self, expanded: bool, width: int | None = None, notify: bool = True) -> None:
        expanded = bool(expanded and self._marker_available)
        if width is not None:
            self._last_panel_width = max(310, min(390, int(width)))
        total = max(self.width(), sum(self.sizes()), 1)
        panel = self.widget(1) if self.count() > 1 else None
        self._applying_state = True
        try:
            if expanded:
                if panel:
                    panel.show()
                desired = min(self._last_panel_width, max(1, total - 120))
                self.setSizes([max(1, total - desired), desired])
            else:
                sizes = self.sizes()
                if len(sizes) > 1 and sizes[1] > 0:
                    self._last_panel_width = max(310, min(390, sizes[1]))
                if panel:
                    panel.hide()
            self._expanded = expanded
        finally:
            self._applying_state = False
        if notify:
            self.panel_state_changed.emit(self._expanded, self._last_panel_width)

    def toggle_marker_panel(self) -> None:
        self.set_panel_expanded(not self._expanded)

    def _splitter_moved(self, _position: int, _index: int) -> None:
        if self._applying_state or self.count() < 2:
            return
        sizes = self.sizes()
        expanded = self.widget(1).isVisible() and sizes[1] > 0
        if expanded:
            self._last_panel_width = max(310, min(390, sizes[1]))
        self._expanded = expanded
        self.panel_state_changed.emit(self._expanded, self._last_panel_width)


def normalized_pair(first: float, second: float) -> tuple[float, float]:
    """Accept either user-entered order and return an ascending axis range."""
    return (first, second) if first <= second else (second, first)


def nice_tick_step(span: float, target_intervals: int = 10) -> float:
    """Choose an engineering-friendly 1/2/5 × 10ⁿ major interval."""
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(1, target_intervals)
    exponent = np.floor(np.log10(raw))
    fraction = raw / (10.0 ** exponent)
    nice_fraction = 1.0 if fraction < 1.5 else 2.0 if fraction < 3 else 5.0 if fraction < 7 else 10.0
    return float(nice_fraction * (10.0 ** exponent))


def next_marker_name(markers: list[MarkerConfig]) -> str:
    numbers = []
    for marker in markers:
        match = re.fullmatch(r"M(\d+)", marker.name, re.I)
        if match:
            numbers.append(int(match.group(1)))
    return f"M{max(numbers, default=0) + 1}"


class MarkerCard(QtWidgets.QFrame):
    delete_requested = QtCore.Signal(str)
    selected = QtCore.Signal(str)
    frequency_changed = QtCore.Signal(str, float)
    visibility_changed = QtCore.Signal(str, bool)

    def __init__(self, marker: MarkerConfig, parent=None) -> None:
        super().__init__(parent)
        self.marker_id = marker.id
        self.color = marker.color
        self.theme = "Dark"
        self.unit = ""
        self.readings: list[tuple[int, str, str, str, str]] = []
        self.selected_trace_indices: set[int] = set()
        self.preview_trace_index: int | None = None
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.title = QtWidgets.QLabel()
        self.title.setTextFormat(QtCore.Qt.RichText)
        self.frequency = QtWidgets.QDoubleSpinBox()
        self.frequency.setDecimals(6)
        self.frequency.setRange(0.0, 1_000_000.0)
        self.frequency.setSingleStep(0.1)
        self.frequency.setSuffix(" GHz")
        self.frequency.setKeyboardTracking(False)
        self.frequency.setFixedWidth(135)
        self.frequency.editingFinished.connect(
            lambda: self.frequency_changed.emit(self.marker_id, self.frequency.value() * 1e9)
        )
        self.visibility_button = QtWidgets.QToolButton()
        self.visibility_button.clicked.connect(self._toggle_visibility)
        self.delete_button = QtWidgets.QToolButton()
        self.delete_button.setText("×")
        self.delete_button.setToolTip("Delete this marker")
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.marker_id))
        self.readout = QtWidgets.QLabel()
        self.readout.setTextFormat(QtCore.Qt.RichText)
        self.readout.setWordWrap(True)
        self.readout.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.title)
        header.addWidget(self.frequency, 1)
        header.addWidget(self.visibility_button)
        header.addWidget(self.delete_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 8)
        layout.setSpacing(3)
        layout.addLayout(header)
        layout.addWidget(self.readout)
        self.set_active(False)

    def set_content(
        self,
        marker: MarkerConfig,
        unit: str,
        readings: list[tuple[int, str, str, str, str]],
    ) -> None:
        self.color = marker.color
        self._visible = marker.visible
        self.unit = unit
        self.readings = readings
        self.title.setText(
            f"<span style='color:{html.escape(marker.color)}'>●</span> "
            f"<b>{html.escape(marker.name)}</b>"
        )
        self.frequency.blockSignals(True)
        self.frequency.setValue(marker.frequency_hz / 1e9)
        self.frequency.blockSignals(False)
        self.visibility_button.setText("Hide" if marker.visible else "Show")
        self.visibility_button.setToolTip("Hide marker line" if marker.visible else "Show marker line")
        self._render_readout()
        # A hidden marker remains manageable from its compact header, while its
        # potentially long list of trace readings is folded away with the line.
        self.readout.setVisible(marker.visible)

    def set_trace_highlights(self, selected_indices: set[int], preview_index: int | None) -> None:
        self.selected_trace_indices = set(selected_indices)
        self.preview_trace_index = preview_index
        self._render_readout()

    def _render_readout(self) -> None:
        secondary = "#536471" if self.theme == "Light" else "#9fb0bf"
        selected_color = "#9b2c00" if self.theme == "Light" else "#ffe36e"
        preview_color = "#005a9c" if self.theme == "Light" else "#8be9fd"
        removal_color = "#b42318" if self.theme == "Light" else "#ff6b6b"
        rows = []
        for index, color, dataset_name, parameter, value_text in self.readings:
            selected = index in self.selected_trace_indices
            previewed = index == self.preview_trace_index and not selected
            removal_preview = index == self.preview_trace_index and selected
            text_color = removal_color if removal_preview else selected_color if selected else preview_color if previewed else None
            if parameter:
                trace_name = f"{html.escape(dataset_name)} · <b>{html.escape(parameter)}</b>"
            else:
                trace_name = html.escape(dataset_name)
            text = f"{trace_name} · {html.escape(value_text)}"
            if text_color:
                font_style = "; font-style:italic" if removal_preview else ""
                text = f"<span style='color:{text_color}; font-weight:700{font_style}'>{text}</span>"
            rows.append(f"<span style='color:{html.escape(color)}'>■</span> {text}")
        body = "<br>".join(rows) or "<i>No visible traces</i>"
        self.readout.setText(
            f"<span style='color:{secondary}'>{html.escape(self.unit)}</span><br>{body}"
        )

    def _toggle_visibility(self) -> None:
        self.visibility_changed.emit(self.marker_id, not self._visible)

    def set_active(self, active: bool) -> None:
        light = self.theme == "Light"
        border = self.color if active else ("#aeb8c2" if light else "#3b4a58")
        width = 2 if active else 1
        self.setStyleSheet(
            f"MarkerCard {{ background:{'#ffffff' if light else '#111820'}; border:{width}px solid {border}; border-radius:5px; }}"
            "MarkerCard QLabel { background:transparent; }"
        )

    def apply_theme(self, theme: str) -> None:
        self.theme = theme
        self.set_active(False)
        self._render_readout()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.selected.emit(self.marker_id)
        super().mousePressEvent(event)


class CurveStylePreview(QtWidgets.QWidget):
    """Read-only color and line-style sample used by the report label editor."""

    def __init__(self, color: str, line_style: str, parent=None) -> None:
        super().__init__(parent)
        self.color = color
        self.line_style = line_style
        self.setFixedSize(76, 24)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setToolTip(f"{line_style} · {color} (edit in Plot / Traces)")

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(self.color), 2.4, STYLE_MAP.get(self.line_style, QtCore.Qt.SolidLine))
        pen.setCapStyle(QtCore.Qt.FlatCap)
        painter.setPen(pen)
        center = self.rect().center().y()
        painter.drawLine(7, center, self.width() - 7, center)
        painter.end()


class ReportLegendDialog(QtWidgets.QDialog):
    """Edit report-friendly labels for traces currently visible on the plot."""

    def __init__(
        self,
        config: PlotConfig,
        datasets: dict[str, Dataset],
        parent=None,
        legend_hidden_trace_ids: set[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.datasets = datasets
        self.setWindowTitle("Edit Report Legend")
        self.resize(680, 440)

        hidden_ids = legend_hidden_trace_ids or set()
        visible_traces = [
            trace for trace in config.traces
            if trace.visible and trace.id not in hidden_ids
        ]
        self.table = QtWidgets.QTableWidget(len(visible_traces), 3)
        self.table.setHorizontalHeaderLabels(["Curve", "Analysis legend", "Report legend text"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(0, 88)
        for row, trace in enumerate(visible_traces):
            dataset = datasets.get(trace.dataset_id)
            full_name = f"{dataset.name if dataset else 'Missing'}: {trace.parameter}"
            original = QtWidgets.QTableWidgetItem(full_name)
            original.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            original.setData(QtCore.Qt.UserRole, trace.id)
            label = QtWidgets.QLineEdit(config.report_legend_labels.get(trace.id, full_name))
            preview_color = (
                parent._display_color(trace.color)
                if parent is not None and hasattr(parent, "_display_color") else trace.color
            )
            preview = CurveStylePreview(preview_color, trace.line_style)
            preview_holder = QtWidgets.QWidget()
            preview_layout = QtWidgets.QHBoxLayout(preview_holder)
            preview_layout.setContentsMargins(4, 0, 4, 0)
            preview_layout.addWidget(preview, alignment=QtCore.Qt.AlignCenter)
            self.table.setCellWidget(row, 0, preview_holder)
            self.table.setItem(row, 1, original)
            self.table.setCellWidget(row, 2, label)

        self.full_names_button = QtWidgets.QPushButton("Use Full Names")
        self.full_names_button.clicked.connect(self._use_full_names)

        note = QtWidgets.QLabel(
            "Only visible traces are listed. Report text changes only the Legend; "
            "Curve samples show the current color and line style but are read-only here; "
            "edit them in Plot / Traces. Ctrl selections and Highlight state remain unchanged."
        )
        note.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.full_names_button, alignment=QtCore.Qt.AlignLeft)
        layout.addWidget(buttons)

    def _use_full_names(self) -> None:
        for row in range(self.table.rowCount()):
            editor = self.table.cellWidget(row, 2)
            original = self.table.item(row, 1)
            if isinstance(editor, QtWidgets.QLineEdit) and original is not None:
                editor.setText(original.text())

    def values(self) -> dict[str, str]:
        valid_ids = {trace.id for trace in self.config.traces}
        labels = {
            trace_id: label for trace_id, label in self.config.report_legend_labels.items()
            if trace_id in valid_ids
        }
        for row in range(self.table.rowCount()):
            original = self.table.item(row, 1)
            editor = self.table.cellWidget(row, 2)
            if original is None:
                continue
            label = editor.text().strip() if isinstance(editor, QtWidgets.QLineEdit) else ""
            labels[str(original.data(QtCore.Qt.UserRole))] = label or original.text()
        return labels


class AnalyzerPlotWidget(QtWidgets.QFrame):
    CURVE_QUALITY_AUTO = "Auto (Recommended)"
    CURVE_QUALITY_HIGH = "High quality"
    CURVE_QUALITY_PERFORMANCE = "Performance"
    CURVE_QUALITIES = (CURVE_QUALITY_AUTO, CURVE_QUALITY_HIGH, CURVE_QUALITY_PERFORMANCE)
    INTERACTIVE_DOWNSAMPLE_FACTOR = 0.4
    IDLE_DOWNSAMPLE_FACTOR = 1.5
    INTERACTIVE_TRACE_THRESHOLD = 20
    MINIMUM_CARD_HEIGHT = 590
    activated = QtCore.Signal(str)
    config_changed = QtCore.Signal()
    marker_changed = QtCore.Signal(str)
    delete_requested = QtCore.Signal(str)
    rename_requested = QtCore.Signal(str, str)
    image_copied = QtCore.Signal(str)
    wheel_scroll_requested = QtCore.Signal(int)
    trace_selection_changed = QtCore.Signal(str, object)
    find_trace_requested = QtCore.Signal(str, object)
    find_highlight_clear_requested = QtCore.Signal(str)
    refresh_progress = QtCore.Signal(int, int)

    def __init__(self, config: PlotConfig, datasets: dict[str, Dataset], export_settings: dict | None = None, theme: str = "Dark", parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.datasets = datasets
        self._is_active_plot = False
        self._width_before_freeze: tuple[int, int] | None = None
        self._marker_external_extra = 0
        self._marker_base_constraints: tuple[int, int] | None = None
        self._legend_external_extra_height = 0
        self._legend_base_minimum_height: int | None = None
        self._legend_base_card_height: int | None = None
        self.marker_lines: dict[str, pg.InfiniteLine] = {}
        self.marker_cards: dict[str, MarkerCard] = {}
        self.active_marker_id: str | None = None
        self.selected_trace_indices: set[int] = set()
        self.preview_trace_index: int | None = None
        self.legend_hidden_trace_ids: set[str] = set()
        self.find_result_active = False
        self.trace_items: dict[int, pg.PlotDataItem] = {}
        self.trace_hover_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.display_cache: dict[tuple[str, str, str, int], np.ndarray] = {}
        self.home_view: tuple[tuple[float, float], tuple[float, float], tuple[float, float] | None, tuple[float, float] | None] | None = None
        self.export_settings = export_settings if export_settings is not None else {
            "theme": "Current GUI", "include_markers": True, "include_marker_info": True,
            "size_mode": "Current GUI ratio", "preset_size": "1600 × 900", "width": 1600, "height": 900,
            "curve_render_quality": self.CURVE_QUALITY_AUTO,
        }
        self.theme = theme
        self.setObjectName("plotCard")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        # Multiple plots should scroll vertically instead of compressing each
        # graph below the comfortable single-plot reading height.
        self.setMinimumHeight(self.MINIMUM_CARD_HEIGHT)

        self.name_edit = QtWidgets.QLineEdit(config.name)
        plot_name_width = max(180, self.name_edit.fontMetrics().averageCharWidth() * 20 + 16)
        self.name_edit.setFixedWidth(plot_name_width)
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems([item.value for item in DisplayFormat])
        self.format_combo.setCurrentText(config.display_format)
        self.freq_min, self.freq_max, self.freq_step = QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.y_min, self.y_max, self.y_step = QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for edit in (self.freq_min, self.freq_max, self.freq_step, self.y_min, self.y_max, self.y_step):
            five_char_width = max(48, min(60, edit.fontMetrics().averageCharWidth() * 5 + 12))
            edit.setFixedWidth(five_char_width)
        if config.frequency_range_hz:
            self.freq_min.setText(f"{config.frequency_range_hz[0] / 1e9:g}")
            self.freq_max.setText(f"{config.frequency_range_hz[1] / 1e9:g}")
        if config.y_range:
            self.y_min.setText(f"{config.y_range[0]:g}")
            self.y_max.setText(f"{config.y_range[1]:g}")
        if config.frequency_step_ghz:
            self.freq_step.setText(f"{config.frequency_step_ghz:g}")
        if config.y_step:
            self.y_step.setText(f"{config.y_step:g}")

        self.x_axis = EndpointAxisItem("bottom")
        self.y_axis = EndpointAxisItem("left")
        self.view_box = ShiftPanViewBox()
        self.plot = pg.PlotWidget(
            background="#111820",
            axisItems={"bottom": self.x_axis, "left": self.y_axis},
            viewBox=self.view_box,
        )
        self.plot.viewport().setMouseTracking(True)
        self.plot.viewport().installEventFilter(self)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.legend = ConstrainedLegendItem(offset=(10, 10), margin=8, mode_toggle=True)
        self.legend.setParentItem(self.plot.getViewBox())
        self.plot.plotItem.legend = self.legend
        self.plot.getViewBox().sigResized.connect(lambda *_args: self.legend.ensure_inside())
        self.legend.mode_button.clicked.connect(self.toggle_legend_mode)
        self.legend.dock_button.clicked.connect(self.toggle_legend_location)
        self._configure_fit_button()
        self._build_context_menu()
        self.plot.setMinimumHeight(330)
        self._build_marker_panel()

        apply_range = QtWidgets.QToolButton(text="Apply")
        autoscale = QtWidgets.QToolButton(text="Autoscale")
        add_marker = QtWidgets.QToolButton(text="+ Marker")
        self.legend_button = QtWidgets.QToolButton(text="Legend")
        self.legend_button.setCheckable(True)
        self.legend_button.setChecked(config.legend_visible)
        self.marker_expand_button = QtWidgets.QToolButton(text="▼ Marker Info")
        self.marker_expand_button.setToolTip("Show marker information")
        self.marker_expand_button.hide()
        copy_image = QtWidgets.QToolButton(text="Copy")
        primary_row = QtWidgets.QHBoxLayout()
        primary_row.addWidget(self.name_edit)
        primary_row.addWidget(self.format_combo)
        primary_row.addStretch()
        for button in (
            apply_range, autoscale, add_marker, self.legend_button,
            copy_image,
        ):
            primary_row.addWidget(button)
        self.plot_action_buttons = (
            apply_range, autoscale, add_marker, self.legend_button, copy_image,
        )
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("X GHz"))
        range_row.addWidget(self.freq_min)
        range_row.addWidget(QtWidgets.QLabel("to"))
        range_row.addWidget(self.freq_max)
        range_row.addWidget(QtWidgets.QLabel("Step"))
        range_row.addWidget(self.freq_step)
        range_row.addSpacing(12)
        range_row.addWidget(QtWidgets.QLabel("Y"))
        range_row.addWidget(self.y_min)
        range_row.addWidget(QtWidgets.QLabel("to"))
        range_row.addWidget(self.y_max)
        range_row.addWidget(QtWidgets.QLabel("Step"))
        range_row.addWidget(self.y_step)
        range_row.addStretch()
        self.marker_info_toolbar_row = QtWidgets.QWidget()
        self.marker_info_toolbar_row.setObjectName("markerInfoToolbarRow")
        self.marker_info_toolbar_row.setFixedHeight(30)
        marker_info_row = QtWidgets.QHBoxLayout(self.marker_info_toolbar_row)
        marker_info_row.setContentsMargins(0, 0, 0, 0)
        marker_info_row.addStretch()
        marker_info_row.addWidget(self.marker_expand_button)
        toolbar = QtWidgets.QVBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        toolbar.addLayout(primary_row)
        toolbar.addLayout(range_row)
        toolbar.addWidget(self.marker_info_toolbar_row)
        self.toolbar_widget = QtWidgets.QWidget()
        self.toolbar_widget.setObjectName("plotToolbarWidget")
        self.toolbar_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.toolbar_widget.setLayout(toolbar)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.toolbar_widget, 0)
        self.capture_widget = QtWidgets.QWidget()
        capture_layout = QtWidgets.QVBoxLayout(self.capture_widget)
        capture_layout.setContentsMargins(0, 0, 0, 0)
        capture_layout.setSpacing(0)
        self.docked_legend = DockedLegendWidget()
        self.docked_legend.mode_clicked.connect(self.toggle_legend_mode)
        self.docked_legend.dock_clicked.connect(self.toggle_legend_location)
        self.docked_legend.entry_clicked.connect(self._toggle_docked_legend_trace)
        self.docked_legend.height_changed.connect(self._docked_legend_height_changed)
        self.docked_legend.geometry_changed.connect(self._schedule_docked_legend_bounds_update)
        self.docked_legend.hide()
        self.plot_column = QtWidgets.QWidget()
        plot_column_layout = QtWidgets.QVBoxLayout(self.plot_column)
        plot_column_layout.setContentsMargins(0, 0, 0, 0)
        plot_column_layout.setSpacing(4)
        plot_column_layout.addWidget(self.plot, 1)
        plot_column_layout.addWidget(self.docked_legend, 0)
        self.marker_splitter = MarkerSplitter()
        self.marker_splitter.addWidget(self.plot_column)
        self.marker_splitter.addWidget(self.marker_panel)
        self.marker_splitter.setStretchFactor(0, 1)
        self.marker_splitter.setStretchFactor(1, 0)
        self.marker_splitter.panel_state_changed.connect(self._marker_panel_state_changed)
        self.marker_collapse_button.clicked.connect(
            lambda: self.marker_splitter.set_panel_expanded(False)
        )
        self.marker_expand_button.clicked.connect(
            lambda: self.marker_splitter.set_panel_expanded(True, self.config.marker_info_width)
        )
        capture_layout.addWidget(self.marker_splitter, 1)
        self.view_box.sigResized.connect(
            lambda *_args: self._schedule_docked_legend_bounds_update()
        )
        layout.addWidget(self.capture_widget, 1)

        self.name_edit.editingFinished.connect(self._name_changed)
        self.format_combo.currentTextChanged.connect(self._format_changed)
        apply_range.clicked.connect(self.apply_ranges)
        autoscale.clicked.connect(self.autoscale)
        add_marker.clicked.connect(self.add_marker)
        self.legend_button.toggled.connect(self.set_legend_visible)
        copy_image.clicked.connect(self.copy_image_to_clipboard)
        self.plot.scene().sigMouseClicked.connect(self._plot_mouse_clicked)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(50)
        self.preview_timer.timeout.connect(self._check_preview_modifier)
        self.marker_highlight_timer = QtCore.QTimer(self)
        self.marker_highlight_timer.setSingleShot(True)
        self.marker_highlight_timer.setInterval(50)
        self.marker_highlight_timer.timeout.connect(self._refresh_marker_highlights_now)
        self.marker_drag_timer = QtCore.QTimer(self)
        self.marker_drag_timer.setSingleShot(True)
        self.marker_drag_timer.setInterval(40)
        self.marker_drag_timer.timeout.connect(self._flush_marker_drag_update)
        self.pending_marker_id: str | None = None
        self.navigation_quality_timer = QtCore.QTimer(self)
        self.navigation_quality_timer.setSingleShot(True)
        # A large plot can take more than one frame to repaint. Waiting long
        # enough prevents low/high quality from oscillating mid-gesture.
        self.navigation_quality_timer.setInterval(400)
        self.navigation_quality_timer.timeout.connect(self._restore_idle_plot_quality)
        self._force_high_plot_quality = False
        self.view_box.sigRangeChanged.connect(self._plot_view_range_changed)
        self.apply_theme(theme)
        self.refresh()
        QtCore.QTimer.singleShot(0, self._restore_marker_panel_state)

    def _restore_marker_panel_state(self) -> None:
        self.marker_splitter.set_marker_available(bool(self.config.markers))
        self.marker_splitter.set_panel_expanded(
            self.config.marker_info_expanded,
            self.config.marker_info_width,
            notify=False,
        )
        self._apply_marker_panel_external_width(
            self.marker_splitter.panel_expanded(), self.marker_splitter.panel_width()
        )
        self._sync_marker_panel_controls()

    def _marker_panel_state_changed(self, expanded: bool, width: int) -> None:
        self.config.marker_info_expanded = bool(expanded)
        self.config.marker_info_width = int(width)
        self._apply_marker_panel_external_width(expanded, width)
        self._sync_marker_panel_controls()

    def _apply_marker_panel_external_width(self, expanded: bool, panel_width: int) -> None:
        """Grow the plot card outward so Marker Info never steals graph width."""
        new_extra = (int(panel_width) + self.marker_splitter.handleWidth()) if expanded else 0
        old_extra = self._marker_external_extra
        if new_extra == old_extra:
            return

        frozen = self.plot_width_is_frozen()
        if old_extra == 0 and new_extra > 0:
            self._marker_base_constraints = (
                self._width_before_freeze
                if frozen and self._width_before_freeze is not None
                else (self.minimumWidth(), self.maximumWidth())
            )

        delta = new_extra - old_extra
        if frozen:
            self.setFixedWidth(max(1, self.width() + delta))
        elif new_extra > 0:
            # Use the current laid-out width as the baseline. The scroll area's
            # horizontal bar then exposes the added panel instead of shrinking
            # the PlotWidget inside the card.
            target = max(self.minimumWidth(), self.width()) + delta
            maximum = self._marker_base_constraints[1] if self._marker_base_constraints else self.maximumWidth()
            self.setMinimumWidth(max(1, target))
            self.setMaximumWidth(max(maximum, target))
        elif self._marker_base_constraints is not None:
            minimum, maximum = self._marker_base_constraints
            self.setMinimumWidth(minimum)
            self.setMaximumWidth(maximum)

        self._marker_external_extra = new_extra
        if new_extra == 0:
            self._marker_base_constraints = None
        self.updateGeometry()
        if self.parentWidget() is not None:
            self.parentWidget().updateGeometry()

    def base_width_without_marker_info(self) -> int:
        return max(1, self.width() - self._marker_external_extra)

    def _sync_marker_panel_controls(self) -> None:
        available = bool(self.config.markers)
        self.marker_expand_button.setVisible(
            available and not self.marker_splitter.panel_expanded()
        )

    def _build_marker_panel(self) -> None:
        self.marker_panel = QtWidgets.QFrame()
        self.marker_panel.setObjectName("markerPanel")
        self.marker_panel.setMinimumWidth(310)
        self.marker_panel.setMaximumWidth(390)
        title = QtWidgets.QLabel("MARKERS")
        title.setStyleSheet("font-weight:700; letter-spacing:1px;")
        self.hide_all_button = QtWidgets.QToolButton(text="Hide All")
        self.show_all_button = QtWidgets.QToolButton(text="Show All")
        self.delete_all_button = QtWidgets.QToolButton(text="Delete All")
        self.marker_collapse_button = QtWidgets.QToolButton(text="▲")
        self.marker_collapse_button.setToolTip("Hide marker information")
        self.hide_all_button.clicked.connect(lambda: self.set_all_markers_visible(False))
        self.show_all_button.clicked.connect(lambda: self.set_all_markers_visible(True))
        self.delete_all_button.clicked.connect(self.delete_all_markers)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.hide_all_button)
        header.addWidget(self.show_all_button)
        header.addWidget(self.delete_all_button)
        header.addWidget(self.marker_collapse_button)
        self.marker_cards_widget = QtWidgets.QWidget()
        self.marker_cards_layout = QtWidgets.QVBoxLayout(self.marker_cards_widget)
        self.marker_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.marker_cards_layout.setSpacing(6)
        self.marker_scroll = QtWidgets.QScrollArea()
        self.marker_scroll.setWidgetResizable(True)
        self.marker_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.marker_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.marker_scroll.setWidget(self.marker_cards_widget)
        layout = QtWidgets.QVBoxLayout(self.marker_panel)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.addLayout(header)
        layout.addWidget(self.marker_scroll, 1)
        self.marker_panel.setStyleSheet("QFrame#markerPanel { background:#0e151c; border:1px solid #43515f; border-radius:5px; }")
        self.marker_panel.show()

    def _configure_fit_button(self) -> None:
        button = self.plot.plotItem.autoBtn
        try:
            button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        button.clicked.connect(lambda *_args: self.fit_view())
        button.prepareGeometryChange()
        button._width = 36
        self._style_fit_button()

    def _style_fit_button(self) -> None:
        button = self.plot.plotItem.autoBtn
        pixmap = QtGui.QPixmap(36, 36)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        light = self.theme == "Light"
        painter.setPen(pg.mkPen("#6f7d89" if light else "#c4ced7", width=1.0))
        painter.setBrush(pg.mkBrush(QtGui.QColor(255, 255, 255, 210) if light else QtGui.QColor(18, 26, 34, 220)))
        painter.drawRoundedRect(QtCore.QRectF(1.5, 6.5, 33, 23), 4, 4)
        font = QtGui.QFont(self.font())
        font.setPixelSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(pg.mkPen("#24313c" if light else "#e3ebf2"))
        painter.drawText(QtCore.QRectF(1, 6, 34, 24), QtCore.Qt.AlignCenter, "FIT")
        painter.end()
        button.setPixmap(pixmap)

    def _build_context_menu(self) -> None:
        view_box = self.plot.getViewBox()
        plot_options = self.plot.plotItem.ctrlMenu
        average_action = next((action for action in plot_options.actions() if action.text() == "Average"), None)
        if average_action is not None:
            plot_options.removeAction(average_action)
        menu = QtWidgets.QMenu("Plot Context", self.plot)
        self.edit_report_legend_action = menu.addAction("Edit Report Legend Text…")
        self.edit_report_legend_action.setVisible(self.config.legend_mode == "report")
        self.edit_report_legend_action.triggered.connect(self.edit_report_legend)
        menu.addSeparator()
        self.find_trace_action = menu.addAction("Find Selected Trace")
        self.find_trace_action.setVisible(False)
        self.find_trace_action.triggered.connect(self._find_selected_trace)
        self.clear_trace_action = menu.addAction("Clear Selected Traces")
        self.clear_trace_action.setVisible(False)
        self.clear_trace_action.triggered.connect(self.clear_trace_selection)
        menu.addSeparator()
        menu.addMenu(plot_options)
        if average_action is not None:
            menu.addAction(average_action)
        view_box.menu = menu
        self.plot.plotItem.setMenuEnabled(False, enableViewBoxMenu=None)
        # Remove PyQtGraph's separate scene-level Export action; exporting is
        # provided by the application's toolbar with persistent PNG settings.
        self.plot.scene().contextMenu = []

    def _find_selected_trace(self) -> None:
        if self.selected_trace_indices:
            self.find_result_active = True
            self.clear_trace_action.setVisible(True)
            self.find_trace_requested.emit(self.config.id, sorted(self.selected_trace_indices))

    def _plot_mouse_clicked(self, event) -> None:
        self.activated.emit(self.config.id)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.plot.viewport():
            if event.type() == QtCore.QEvent.Type.Wheel:
                delta = event.angleDelta().y() or event.pixelDelta().y()
                scene_pos = self.plot.mapToScene(event.position().toPoint())
                if self.legend.scroll_at(scene_pos, int(delta)):
                    event.accept()
                    return True
                if not self._is_active_plot:
                    self.wheel_scroll_requested.emit(int(delta))
                    event.accept()
                    return True
            if event.type() == QtCore.QEvent.Type.MouseMove:
                scene_pos = self.plot.mapToScene(event.position().toPoint())
                self.update_trace_preview(scene_pos, event.modifiers())
            elif event.type() == QtCore.QEvent.Type.Leave:
                self._set_preview_trace(None)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "docked_legend"):
            self._schedule_docked_legend_bounds_update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "docked_legend"):
            self._schedule_docked_legend_bounds_update()

    def update_trace_preview(self, scene_pos: QtCore.QPointF, modifiers) -> None:
        if not modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._set_preview_trace(None)
            return
        if not self.view_box.sceneBoundingRect().contains(scene_pos):
            self._set_preview_trace(None)
            return
        view_pos = self.view_box.mapSceneToView(scene_pos)
        nearest_index = None
        nearest_distance = 8.0
        for index, (x_values, y_values) in self.trace_hover_data.items():
            if x_values.size < 2 or view_pos.x() < x_values[0] or view_pos.x() > x_values[-1]:
                continue
            upper = int(np.searchsorted(x_values, view_pos.x(), side="right"))
            upper = min(max(upper, 1), x_values.size - 1)
            lower = upper - 1
            x0, x1 = x_values[lower], x_values[upper]
            y0, y1 = y_values[lower], y_values[upper]
            if not np.all(np.isfinite((x0, x1, y0, y1))):
                continue
            ratio = 0.0 if x1 == x0 else (view_pos.x() - x0) / (x1 - x0)
            curve_y = y0 + ratio * (y1 - y0)
            curve_scene = self.trace_items[index].mapToScene(QtCore.QPointF(view_pos.x(), float(curve_y)))
            distance = QtCore.QLineF(scene_pos, curve_scene).length()
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index
        self._set_preview_trace(nearest_index)
        if nearest_index is not None and not self.preview_timer.isActive():
            self.preview_timer.start()

    def _check_preview_modifier(self) -> None:
        if not QtGui.QGuiApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._set_preview_trace(None)

    def _set_preview_trace(self, index: int | None) -> None:
        if index == self.preview_trace_index:
            return
        previous = self.preview_trace_index
        self.preview_trace_index = index
        if index is None:
            self.preview_timer.stop()
        self._update_trace_selection_styles({previous, index})

    def select_trace(self, index: int) -> None:
        if index not in self.trace_items:
            return
        changed = set(self.selected_trace_indices)
        changed.add(index)
        self.selected_trace_indices = {index}
        self._selection_updated(changed)

    def toggle_trace_selection(self, index: int) -> None:
        if index not in self.trace_items:
            return
        if index in self.selected_trace_indices:
            self.selected_trace_indices.remove(index)
        else:
            self.selected_trace_indices.add(index)
        self._selection_updated({index})

    def _selection_updated(self, changed_indices: set[int | None] | None = None) -> None:
        self._update_trace_selection_styles(changed_indices)
        count = len(self.selected_trace_indices)
        self.find_trace_action.setVisible(count > 0)
        self.clear_trace_action.setVisible(count > 0 or self.find_result_active)
        self.find_trace_action.setText("Find Selected Traces" if count > 1 else "Find Selected Trace")
        self.trace_selection_changed.emit(self.config.id, sorted(self.selected_trace_indices))

    def clear_trace_selection(self) -> None:
        if not self.selected_trace_indices and not self.find_result_active:
            return
        changed = set(self.selected_trace_indices)
        self.selected_trace_indices.clear()
        self.find_result_active = False
        self._selection_updated(changed)
        self.find_highlight_clear_requested.emit(self.config.id)

    def _trace_clicked(self, index: int, event) -> None:
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.toggle_trace_selection(index)

    def _apply_trace_selection(self) -> None:
        self._update_trace_selection_styles(None)

    def _update_trace_selection_styles(self, indices: set[int | None] | None) -> None:
        target_indices = (
            set(self.trace_items)
            if indices is None
            else {index for index in indices if index is not None and index in self.trace_items}
        )
        for index in target_indices:
            item = self.trace_items[index]
            trace = self.config.traces[index]
            selected = index in self.selected_trace_indices
            previewed = index == self.preview_trace_index and not selected
            removal_preview = index == self.preview_trace_index and selected
            removal_color = "#b42318" if self.theme == "Light" else "#ff6b6b"
            item.setPen(pg.mkPen(
                removal_color if removal_preview else self._display_color(trace.color),
                width=4.2 if removal_preview else 3.4 if selected else 2.6 if previewed else 1.7,
                style=STYLE_MAP.get(trace.line_style, QtCore.Qt.SolidLine),
            ))
            item.setZValue(120 if removal_preview else 100 if selected else 50 if previewed else 0)
        self._refresh_legend_labels(indices=target_indices if indices is not None else None)
        self.marker_highlight_timer.start()

    def set_font_size(self, kind: str, size: int) -> None:
        size = int(size)
        if kind == "legend":
            self.config.legend_font_size = size
        elif kind == "title":
            self.config.title_font_size = size
        elif kind == "axis":
            self.config.axis_font_size = size
        else:
            raise ValueError(f"Unknown plot font kind: {kind}")
        self._apply_plot_fonts()
        self.config_changed.emit()

    def set_font_sizes(self, legend: int, title: int, axis: int, emit: bool = True) -> None:
        self.config.legend_font_size = int(legend)
        self.config.title_font_size = int(title)
        self.config.axis_font_size = int(axis)
        self._apply_plot_fonts()
        if emit:
            self.config_changed.emit()

    def _name_changed(self) -> None:
        self.rename_requested.emit(self.config.id, self.name_edit.text().strip() or self.config.name)

    def set_name(self, value: str) -> None:
        self.config.name = value
        self.name_edit.setText(value)
        self.plot.setTitle(
            value,
            color="#202830" if self.theme == "Light" else "#dfe7ef",
            size=f"{self.config.title_font_size}pt",
        )

    def _format_changed(self, value: str) -> None:
        self.config.display_format = value
        self.refresh()
        self.config_changed.emit()

    def set_active(self, active: bool) -> None:
        self._is_active_plot = bool(active)
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_plot_width_frozen(self, frozen: bool, width: int | None = None) -> None:
        if frozen:
            if self._width_before_freeze is None:
                self._width_before_freeze = (
                    self._marker_base_constraints
                    if self._marker_external_extra and self._marker_base_constraints is not None
                    else (self.minimumWidth(), self.maximumWidth())
                )
            base_target = int(width if width is not None else self.base_width_without_marker_info())
            target = max(1, base_target + self._marker_external_extra)
            self.setFixedWidth(target)
            margins = self.layout().contentsMargins()
            toolbar_width = max(1, base_target - margins.left() - margins.right())
            self.toolbar_widget.setFixedWidth(toolbar_width)
        elif self._width_before_freeze is not None:
            minimum, maximum = self._width_before_freeze
            self._width_before_freeze = None
            if self._marker_external_extra:
                self._marker_base_constraints = (minimum, maximum)
                target = max(minimum, self.base_width_without_marker_info()) + self._marker_external_extra
                self.setMinimumWidth(target)
                self.setMaximumWidth(max(maximum, target))
            else:
                self.setMinimumWidth(minimum)
                self.setMaximumWidth(maximum)
            self.toolbar_widget.setMinimumWidth(0)
            self.toolbar_widget.setMaximumWidth(16_777_215)
            self.toolbar_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.updateGeometry()

    def plot_width_is_frozen(self) -> bool:
        return self._width_before_freeze is not None

    def refresh(self) -> None:
        selected_keys = {
            (self.config.traces[index].dataset_id, self.config.traces[index].parameter)
            for index in self.selected_trace_indices
            if 0 <= index < len(self.config.traces)
        }
        self.selected_trace_indices.clear()
        self.preview_trace_index = None
        self.preview_timer.stop()
        self.trace_items.clear()
        self.trace_hover_data.clear()
        self.find_trace_action.setVisible(False)
        self.clear_trace_action.setVisible(False)
        self.plot.clear()
        renderable = [
            (index, trace, self.datasets.get(trace.dataset_id))
            for index, trace in enumerate(self.config.traces)
            if trace.visible and self.datasets.get(trace.dataset_id) is not None
        ]
        render_total = len(renderable)
        progress_interval = max(1, render_total // 20)
        idle_factor, idle_antialias = self._idle_render_state()
        for rendered, (index, trace, dataset) in enumerate(renderable, 1):
            y = self._formatted_trace(dataset, trace.parameter)
            pen = pg.mkPen(self._display_color(trace.color), width=1.7, style=STYLE_MAP.get(trace.line_style, QtCore.Qt.SolidLine))
            item = self.plot.plot(
                dataset.frequency_hz / 1e9, y, pen=pen,
                name=f"{dataset.name}: {trace.parameter}", connect="finite",
                # Idle rendering favors detail. Large plots temporarily switch
                # to a lower factor only while their view range is changing.
                autoDownsampleFactor=idle_factor, antialias=idle_antialias,
            )
            item.setCurveClickable(True, width=8)
            # clipToView must be enabled only after the item is attached to its
            # ViewBox (PyQtGraph 0.14 otherwise resolves the wrong parent).
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
            item.sigClicked.connect(lambda _item, event, i=index: self._trace_clicked(i, event))
            self.trace_items[index] = item
            if trace.id in self.legend_hidden_trace_ids:
                item.hide()
            item.visibleChanged.connect(lambda i=index: self._trace_item_visibility_changed(i))
            self.trace_hover_data[index] = (dataset.frequency_hz / 1e9, np.asarray(y))
            if rendered % progress_interval == 0 or rendered == render_total:
                self.refresh_progress.emit(rendered, render_total)
        self.selected_trace_indices = {
            index for index, trace in enumerate(self.config.traces)
            if (trace.dataset_id, trace.parameter) in selected_keys
        }
        self._selection_updated()
        self._apply_plot_fonts()
        self._sync_legend_location()
        self._rebuild_markers()
        self.apply_ranges(silent=True, record_home=self.home_view is None)
        QtCore.QTimer.singleShot(0, self.legend.ensure_inside)

    def _curve_render_quality(self) -> str:
        quality = self.export_settings.get("curve_render_quality", self.CURVE_QUALITY_AUTO)
        return quality if quality in self.CURVE_QUALITIES else self.CURVE_QUALITY_AUTO

    def _idle_render_state(self) -> tuple[float, bool]:
        if self._force_high_plot_quality:
            return self.IDLE_DOWNSAMPLE_FACTOR, True
        if self._curve_render_quality() == self.CURVE_QUALITY_PERFORMANCE:
            return self.INTERACTIVE_DOWNSAMPLE_FACTOR, False
        return self.IDLE_DOWNSAMPLE_FACTOR, True

    def _set_plot_render_state(self, factor: float, antialias: bool) -> None:
        for item in self.trace_items.values():
            factor_changed = item.opts.get("autoDownsampleFactor") != factor
            antialias_changed = item.opts.get("antialias") is not antialias
            if not factor_changed and not antialias_changed:
                continue
            item.opts["autoDownsampleFactor"] = factor
            item.opts["antialias"] = antialias
            if factor_changed:
                item.viewRangeChanged()
            else:
                item.updateItems(styleUpdate=True)

    def set_curve_render_quality(self, quality: str) -> None:
        if quality not in self.CURVE_QUALITIES:
            quality = self.CURVE_QUALITY_AUTO
        self.export_settings["curve_render_quality"] = quality
        self.navigation_quality_timer.stop()
        factor, antialias = self._idle_render_state()
        self._set_plot_render_state(factor, antialias)

    def _plot_view_range_changed(self, *_args) -> None:
        if self._force_high_plot_quality:
            return
        quality = self._curve_render_quality()
        if quality == self.CURVE_QUALITY_PERFORMANCE:
            self._set_plot_render_state(self.INTERACTIVE_DOWNSAMPLE_FACTOR, False)
        elif quality == self.CURVE_QUALITY_HIGH or len(self.trace_items) < self.INTERACTIVE_TRACE_THRESHOLD:
            self._set_plot_render_state(self.IDLE_DOWNSAMPLE_FACTOR, True)
        else:
            self._set_plot_render_state(self.INTERACTIVE_DOWNSAMPLE_FACTOR, False)
            self.navigation_quality_timer.start()

    def _restore_idle_plot_quality(self) -> None:
        factor, antialias = self._idle_render_state()
        self._set_plot_render_state(factor, antialias)

    def _formatted_trace(self, dataset: Dataset, parameter: str) -> np.ndarray:
        values = np.asarray(extract_dataset(dataset, parameter))
        key = (dataset.id, parameter, self.config.display_format, dataset.transform_revision)
        cached = self.display_cache.get(key)
        if cached is None:
            cached = format_trace(values, dataset.frequency_hz, self.config.display_format)
            self.display_cache[key] = cached
            # A port/impedance transform replaces the source array. Discard
            # stale generations for this same displayed trace.
            stale = [
                item for item in self.display_cache
                if item[:3] == key[:3] and item[3] != key[3]
            ]
            for item in stale:
                self.display_cache.pop(item, None)
        return cached

    def _apply_plot_fonts(self) -> None:
        foreground = "#202830" if self.theme == "Light" else "#d8e0e8"
        axis_style = {"font-size": f"{self.config.axis_font_size}pt", "color": foreground}
        self.plot.setLabel("bottom", "Frequency", units="GHz", **axis_style)
        self.plot.setLabel("left", axis_label(self.config.display_format), **axis_style)
        tick_font = QtGui.QFont(self.font())
        tick_font.setPointSize(self.config.axis_font_size)
        self.x_axis.setTickFont(tick_font)
        self.y_axis.setTickFont(tick_font)
        self.plot.setTitle(
            self.config.name,
            color="#202830" if self.theme == "Light" else "#dfe7ef",
            size=f"{self.config.title_font_size}pt",
        )
        self._refresh_legend_labels(foreground)

    def _refresh_legend_labels(
        self,
        foreground: str | None = None,
        indices: set[int] | None = None,
    ) -> None:
        foreground = foreground or ("#202830" if self.theme == "Light" else "#d8e0e8")
        selected_color = "#9b2c00" if self.theme == "Light" else "#ffe36e"
        preview_color = "#005a9c" if self.theme == "Light" else "#8be9fd"
        removal_color = "#b42318" if self.theme == "Light" else "#ff6b6b"
        report_mode = self.config.legend_mode == "report"
        size = f"{self.config.legend_font_size}pt"
        self.legend.setLabelTextSize(size)
        self.legend.setLabelTextColor(foreground)
        if self.legend.mode_button is not None:
            self.legend.mode_button.set_mode(report_mode, self.theme == "Light")
        if self.legend.dock_button is not None:
            self.legend.dock_button.set_symbol("↓", "Move Legend below Frequency", self.theme == "Light")
        if self._sync_floating_legend_entries(report_mode):
            indices = None
        # PyQtGraph 0.14 updates LabelItem options but does not rebuild the
        # existing HTML; explicitly regenerate every live legend label.
        item_indices = {id(item): index for index, item in self.trace_items.items()}
        for sample, label in self.legend.items:
            index = item_indices.get(id(getattr(sample, "item", None)))
            if indices is not None and index not in indices:
                continue
            selected = index in self.selected_trace_indices if index is not None else False
            previewed = index == self.preview_trace_index and not selected if index is not None else False
            removal_preview = index == self.preview_trace_index and selected if index is not None else False
            trace = self.config.traces[index] if index is not None and index < len(self.config.traces) else None
            dataset = self.datasets.get(trace.dataset_id) if trace is not None else None
            # Report Legend is presentation-only: its entries cannot toggle
            # curve visibility. Analysis restores the normal click behavior.
            sample.setAcceptedMouseButtons(
                QtCore.Qt.MouseButton.NoButton
                if report_mode else QtCore.Qt.MouseButton.LeftButton
            )
            sample.setAcceptHoverEvents(not report_mode)
            self._set_legend_sample_click_enabled(sample, not report_mode)
            sample.setVisible(True)
            label.setVisible(True)
            analysis_text = (
                f"{dataset.name}: {trace.parameter}" if dataset is not None and trace is not None
                else label.text
            )
            shown_text = (
                self.config.report_legend_labels.get(trace.id, analysis_text)
                if report_mode and trace is not None else analysis_text
            )
            label.setText(
                shown_text,
                size=size,
                color=(foreground if report_mode else removal_color if removal_preview else selected_color if selected else preview_color if previewed else foreground),
                bold=False if report_mode else selected or previewed or removal_preview,
                italic=False if report_mode else removal_preview,
            )
        self.legend.layout.invalidate()
        self.legend.layout.activate()
        self.legend.updateSize()
        QtCore.QTimer.singleShot(0, self.legend.ensure_inside)
        if hasattr(self, "docked_legend"):
            self._refresh_docked_legend()

    def _sync_floating_legend_entries(self, report_mode: bool) -> bool:
        desired = [
            (index, item)
            for index, item in self.trace_items.items()
            if not report_mode or item.isVisible()
        ]
        current_items = [getattr(sample, "item", None) for sample, _label in self.legend.items]
        desired_items = [item for _index, item in desired]
        if len(current_items) == len(desired_items) and all(
            current is desired for current, desired in zip(current_items, desired_items)
        ):
            return False

        # Hiding existing LabelItems leaves their QGraphicsGridLayout cells in
        # place, producing blank rows, misplaced samples/text, and an oversized
        # frame. Rebuild the actual rows whenever the mode changes which traces
        # belong in the floating Legend.
        self.legend.clear()
        for index, item in desired:
            trace = self.config.traces[index]
            dataset = self.datasets.get(trace.dataset_id)
            if dataset is None:
                continue
            self.legend.addItem(item, f"{dataset.name}: {trace.parameter}")
        self.legend.layout.invalidate()
        self.legend.layout.activate()
        self.legend.updateSize()
        return True

    @staticmethod
    def _set_legend_sample_click_enabled(sample, enabled: bool) -> None:
        original_attribute = "_analysis_mouse_click_event"
        original_handler = getattr(sample, original_attribute, None)
        if enabled:
            if original_handler is not None:
                sample.mouseClickEvent = original_handler
                delattr(sample, original_attribute)
            return
        if original_handler is not None:
            return

        # PyQtGraph's ItemSample receives its own scene MouseClickEvent and
        # toggles PlotDataItem visibility directly, even when Qt accepted
        # mouse buttons are disabled. Replace that handler in Report mode;
        # keep the original bound method so Analysis can restore it exactly.
        setattr(sample, original_attribute, sample.mouseClickEvent)

        def ignore_report_click(event) -> None:
            event.accept()
            sample.update()

        sample.mouseClickEvent = ignore_report_click

    def apply_ranges(self, silent: bool = False, record_home: bool = True) -> None:
        try:
            automatic_bounds = self._autoscale_bounds()
            if self.freq_min.text().strip() and self.freq_max.text().strip():
                low, high = normalized_pair(float(self.freq_min.text()) * 1e9, float(self.freq_max.text()) * 1e9)
                if low == high:
                    raise ValueError("Frequency range must contain two different values.")
                self.config.frequency_range_hz = (low, high)
                self.plot.setXRange(low / 1e9, high / 1e9, padding=0)
                self.x_axis.set_requested_endpoints(low / 1e9, high / 1e9)
            else:
                self.config.frequency_range_hz = None
                if automatic_bounds is not None:
                    low, high = automatic_bounds[:2]
                    self.plot.setXRange(low, high, padding=0)
                    self.x_axis.set_requested_endpoints(low, high)
                else:
                    low, high = self.plot.viewRange()[0]
                    self.x_axis.set_requested_endpoints(low, high)
            self.config.frequency_step_ghz = self._optional_positive(self.freq_step.text(), "Frequency step")
            x_axis = self.plot.getAxis("bottom")
            x_axis.setTickSpacing(
                major=self.config.frequency_step_ghz,
                minor=self.config.frequency_step_ghz / 5 if self.config.frequency_step_ghz else None,
            )
            if self.y_min.text().strip() and self.y_max.text().strip():
                low, high = normalized_pair(float(self.y_min.text()), float(self.y_max.text()))
                if low == high:
                    raise ValueError("Y range must contain two different values.")
                self.config.y_range = (low, high)
                zero_endpoint = low == 0 or high == 0
                self.plot.setYRange(low, high, padding=0.012 if zero_endpoint else 0)
                self.y_axis.set_requested_endpoints(low, high)
            else:
                self.config.y_range = None
                if automatic_bounds is not None:
                    low, high = automatic_bounds[2:]
                    self.plot.setYRange(low, high, padding=0)
                    self.y_axis.set_requested_endpoints(low, high)
                else:
                    low, high = self.plot.viewRange()[1]
                    self.y_axis.set_requested_endpoints(low, high)
            self.config.y_step = self._optional_positive(self.y_step.text(), "Y step")
            y_axis = self.plot.getAxis("left")
            y_axis.setTickSpacing(
                major=self.config.y_step,
                minor=self.config.y_step / 5 if self.config.y_step else None,
            )
            if record_home:
                self._remember_home_view()
            if not silent:
                self.config_changed.emit()
        except ValueError as exc:
            if not silent:
                QtWidgets.QMessageBox.warning(self, "Invalid range", str(exc))

    @staticmethod
    def _optional_positive(text: str, label: str) -> float | None:
        if not text.strip():
            return None
        value = float(text)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive number.")
        return value

    def set_legend_visible(self, visible: bool) -> None:
        self.config.legend_visible = bool(visible)
        self._sync_legend_location()
        self.config_changed.emit()

    def _legend_entries(self):
        report_mode = self.config.legend_mode == "report"
        selected_color = "#9b2c00" if self.theme == "Light" else "#ffe36e"
        preview_color = "#005a9c" if self.theme == "Light" else "#8be9fd"
        foreground = "#202830" if self.theme == "Light" else "#d8e0e8"
        entries = []
        for index, item in self.trace_items.items():
            trace = self.config.traces[index]
            dataset = self.datasets.get(trace.dataset_id)
            if dataset is None:
                continue
            curve_visible = item.isVisible()
            if report_mode and not curve_visible:
                continue
            analysis_text = f"{dataset.name}: {trace.parameter}"
            text = self.config.report_legend_labels.get(trace.id, analysis_text) if report_mode else analysis_text
            selected = index in self.selected_trace_indices
            previewed = index == self.preview_trace_index and not selected
            text_color = foreground if report_mode else selected_color if selected else preview_color if previewed else foreground
            entries.append((
                index, self._display_color(trace.color), trace.line_style, text,
                text_color, False if report_mode else selected or previewed,
                curve_visible,
            ))
        return entries

    def _trace_item_visibility_changed(self, index: int) -> None:
        item = self.trace_items.get(index)
        if item is None or not (0 <= index < len(self.config.traces)):
            return
        trace_id = self.config.traces[index].id
        if item.isVisible():
            self.legend_hidden_trace_ids.discard(trace_id)
        else:
            self.legend_hidden_trace_ids.add(trace_id)
        self._refresh_legend_labels(indices={index})
        self._refresh_marker_cards()
        self.config_changed.emit()

    def _toggle_docked_legend_trace(self, index: int) -> None:
        # Report mode deliberately omits hidden entries. Return to Analysis
        # mode to reveal and restore a curve that was hidden from the Legend.
        if self.config.legend_mode != "analysis":
            return
        item = self.trace_items.get(index)
        if item is not None:
            item.setVisible(not item.isVisible())

    def _refresh_docked_legend(self) -> None:
        if not hasattr(self, "docked_legend"):
            return
        self.docked_legend.set_content(
            self._legend_entries(), self.config.legend_font_size,
            self.config.legend_mode == "report", self.theme,
            self.config.legend_below_alignment,
        )

    def _update_docked_legend_bounds(self) -> None:
        if not hasattr(self, "docked_legend"):
            return
        if not self.docked_legend.isVisible() or self.docked_legend.width() <= 0:
            return
        # The plot lives inside MarkerSplitter while the below-Legend is its
        # sibling. Avoid mapping a GraphicsScene point across those different
        # widget containers: that can acquire the split pane's offset and
        # collapse the Legend to a narrow strip at the far right. ViewBox scene
        # coordinates already use PlotWidget viewport pixels, so combine them
        # directly with the PlotWidget's origin in the below-Legend widget.
        scene_rect = self.view_box.sceneBoundingRect()
        plot_origin = self.plot.mapTo(self.docked_legend, QtCore.QPoint(0, 0)).x()
        left = plot_origin + round(scene_rect.left())
        right = plot_origin + round(scene_rect.right())

        plot_left = max(0, plot_origin)
        plot_right = min(self.docked_legend.width(), plot_origin + self.plot.width())
        left = max(plot_left, min(left, plot_right))
        right = max(left, min(right, plot_right))

        # A normal ViewBox occupies most of its PlotWidget. If a transient Qt
        # layout pass still reports a collapsed span, prefer a safe axis-like
        # inset instead of preserving the unusable 40 px far-right frame.
        expected_minimum = max(120, round(self.plot.width() * 0.35))
        if right - left < expected_minimum and self.plot.width() >= 320:
            safe_left_inset = max(44, round(self.plot.width() * 0.04))
            left = min(plot_right - 80, plot_left + safe_left_inset)
            right = max(left + 80, plot_right - 4)
        self.docked_legend.set_horizontal_bounds(left, right)

    def _schedule_docked_legend_bounds_update(self) -> None:
        if not hasattr(self, "docked_legend"):
            return
        # Layout changes occur in more than one Qt pass when the below Legend
        # changes card height. Recheck after the immediate and settled passes.
        for delay in (0, 40, 160):
            QtCore.QTimer.singleShot(delay, self._update_docked_legend_bounds)

    def _sync_legend_location(self) -> None:
        docked = self.config.legend_location == "below"
        visible = self.config.legend_visible
        self.legend.setVisible(visible and not docked)
        if docked and visible:
            self._refresh_docked_legend()
            self._apply_docked_legend_height(self.docked_legend.height())
            self.docked_legend.show()
            self._schedule_docked_legend_bounds_update()
        else:
            self.docked_legend.hide()
            self._apply_docked_legend_height(0)

    def toggle_legend_location(self) -> None:
        self.config.legend_location = "plot" if self.config.legend_location == "below" else "below"
        self._sync_legend_location()
        self.config_changed.emit()

    def _docked_legend_height_changed(self, height: int) -> None:
        if self.config.legend_location == "below" and self.config.legend_visible:
            self._apply_docked_legend_height(height)

    def _apply_docked_legend_height(self, height: int) -> None:
        height = max(0, int(height))
        old = self._legend_external_extra_height
        if height == old:
            return
        if old == 0 and height > 0:
            self._legend_base_minimum_height = self.minimumHeight()
            self._legend_base_card_height = max(self.minimumHeight(), self.height())
        if height > 0:
            # Always derive the requested size from the original card height.
            # Using the already-expanded current height here makes repeated
            # Analysis/Report switches accumulate the Legend height.
            base_height = self._legend_base_card_height
            if base_height is None:
                base_height = max(1, self.height() - old)
                self._legend_base_card_height = base_height
            self.setMinimumHeight(max(1, base_height + height))
        elif self._legend_base_minimum_height is not None:
            self.setMinimumHeight(self._legend_base_minimum_height)
        self._legend_external_extra_height = height
        if height == 0:
            self._legend_base_minimum_height = None
            self._legend_base_card_height = None
        self.updateGeometry()

    def toggle_legend_mode(self) -> None:
        if self.config.legend_mode == "report":
            self.config.legend_mode = "analysis"
            if hasattr(self, "edit_report_legend_action"):
                self.edit_report_legend_action.setVisible(False)
            self._refresh_legend_labels()
            self._refresh_marker_cards()
            self.config_changed.emit()
            return
        if not self.edit_report_legend():
            return
        self.config.legend_mode = "report"
        if hasattr(self, "edit_report_legend_action"):
            self.edit_report_legend_action.setVisible(True)
        self._refresh_legend_labels()
        self._refresh_marker_cards()
        self.config_changed.emit()

    def edit_report_legend(self) -> bool:
        dialog = ReportLegendDialog(
            self.config,
            self.datasets,
            self,
            self.legend_hidden_trace_ids,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return False
        self.config.report_legend_labels = dialog.values()
        self._refresh_legend_labels()
        self._refresh_marker_cards()
        self.config_changed.emit()
        return True

    def _remember_home_view(self) -> None:
        x_range, y_range = self.plot.viewRange()
        self.home_view = (
            (float(x_range[0]), float(x_range[1])),
            (float(y_range[0]), float(y_range[1])),
            self.x_axis.requested_endpoints,
            self.y_axis.requested_endpoints,
        )

    def fit_view(self) -> None:
        if self.home_view is None:
            self.autoscale()
            return
        x_range, y_range, x_endpoints, y_endpoints = self.home_view
        self.plot.disableAutoRange()
        self.plot.setXRange(*x_range, padding=0)
        self.plot.setYRange(*y_range, padding=0)
        self.x_axis.set_requested_endpoints(*(x_endpoints or (None, None)))
        self.y_axis.set_requested_endpoints(*(y_endpoints or (None, None)))

    def _display_color(self, color: str) -> str:
        value = QtGui.QColor(color)
        if self.theme == "Light" and value.lightness() > 175:
            return value.darker(180).name()
        return color

    def apply_theme(self, theme: str) -> None:
        self.theme = theme
        light = theme == "Light"
        background, foreground, border = ("#ffffff", "#202830", "#aeb8c2") if light else ("#111820", "#d8e0e8", "#43515f")
        self.plot.setBackground(background)
        for axis_name in ("left", "bottom", "top", "right"):
            axis = self.plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(foreground))
            axis.setTextPen(pg.mkPen(foreground))
        self.marker_panel.setStyleSheet(
            f"QFrame#markerPanel {{ background:{'#f4f6f8' if light else '#0e151c'}; border:1px solid {border}; border-radius:5px; }}"
        )
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self._style_fit_button()
        self.legend.setPen(pg.mkPen("#8e9aa5" if light else "#5a6875", width=1.0))
        self.legend.setBrush(pg.mkBrush(QtGui.QColor(255, 255, 255, 225) if light else QtGui.QColor(14, 21, 28, 225)))
        if self.legend.mode_button is not None:
            self.legend.mode_button.set_mode(self.config.legend_mode == "report", light)
        self._apply_trace_selection()
        for card in self.marker_cards.values():
            card.apply_theme(theme)

    def _output_size(self, current_size: QtCore.QSize) -> QtCore.QSize:
        mode = self.export_settings.get("size_mode", "Current GUI ratio")
        if mode == "Current GUI ratio":
            return current_size
        if mode == "Preset size":
            match = re.search(r"(\d+)\s*×\s*(\d+)", self.export_settings.get("preset_size", "1600 × 900"))
            if match:
                return QtCore.QSize(int(match.group(1)), int(match.group(2)))
        return QtCore.QSize(int(self.export_settings.get("width", 1600)), int(self.export_settings.get("height", 900)))

    def _render_at_native_size(self, target: QtCore.QSize) -> QtGui.QPixmap:
        """Reflow the live plot at the target size instead of stretching a screenshot."""
        widget = self.capture_widget
        right_margin = min(24, max(0, target.width() - 1))
        content_target = QtCore.QSize(max(1, target.width() - right_margin), target.height())
        original_size = widget.size()
        original_minimum = widget.minimumSize()
        original_maximum = widget.maximumSize()
        original_legend_position = QtCore.QPointF(self.legend.pos())
        card_layout = self.layout()
        layout_index = card_layout.indexOf(widget)
        card_layout.removeWidget(widget)
        try:
            widget.setFixedSize(content_target)
            widget.layout().activate()
            QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.LayoutRequest)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            pixmap = QtGui.QPixmap(target)
            pixmap.fill(QtGui.QColor("#ffffff" if self.theme == "Light" else "#111820"))
            painter = QtGui.QPainter(pixmap)
            widget.render(painter, QtCore.QPoint(0, 0))
            painter.end()
            return pixmap
        finally:
            widget.setMinimumSize(QtCore.QSize(0, 0))
            widget.setMaximumSize(original_maximum)
            widget.setMinimumSize(original_minimum)
            widget.resize(original_size)
            widget.layout().activate()
            card_layout.insertWidget(layout_index, widget, 1)
            card_layout.invalidate()
            card_layout.activate()
            QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.LayoutRequest)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            self.legend.autoAnchor(original_legend_position, relative=False)

    def render_pixmap(self) -> QtGui.QPixmap:
        export_theme = self.export_settings.get("theme", "Current GUI")
        desired_theme = self.theme if export_theme == "Current GUI" else export_theme
        original_theme = self.theme
        original_legend_scroll = self.legend.scrollbar.value()
        original_panel_expanded = self.marker_splitter.panel_expanded()
        original_panel_width = self.marker_splitter.panel_width()
        theme_changed = desired_theme != original_theme
        original_lines: dict[str, bool] = {}
        original_factor = next((
            item.opts.get("autoDownsampleFactor", self.IDLE_DOWNSAMPLE_FACTOR)
            for item in self.trace_items.values()
        ), self.IDLE_DOWNSAMPLE_FACTOR)
        original_antialias = next((
            bool(item.opts.get("antialias", False)) for item in self.trace_items.values()
        ), True)
        self._force_high_plot_quality = True
        self._set_plot_render_state(self.IDLE_DOWNSAMPLE_FACTOR, True)
        try:
            self.apply_theme(desired_theme)
            if theme_changed:
                self.refresh()
            original_lines = {key: line.isVisible() for key, line in self.marker_lines.items()}
            if not self.export_settings.get("include_markers", True):
                for line in self.marker_lines.values():
                    line.hide()
            self.marker_splitter.set_marker_available(bool(self.config.markers))
            self.marker_splitter.set_panel_expanded(
                bool(self.config.markers) and self.export_settings.get("include_marker_info", True),
                original_panel_width,
                notify=False,
            )
            self.legend.set_controls_visible(False)
            self.docked_legend.set_controls_visible(False)
            QtWidgets.QApplication.processEvents()
            target = self._output_size(self.capture_widget.size())
            return self._render_at_native_size(target)
        finally:
            self.apply_theme(original_theme)
            if theme_changed:
                self.refresh()
            else:
                for key, visible in original_lines.items():
                    if key in self.marker_lines:
                        self.marker_lines[key].setVisible(visible)
            self.marker_splitter.set_panel_expanded(
                original_panel_expanded,
                original_panel_width,
                notify=False,
            )
            self.legend.set_controls_visible(True)
            self.legend.scrollbar.setValue(original_legend_scroll)
            self.docked_legend.set_controls_visible(True)
            self._set_plot_render_state(original_factor, original_antialias)
            self._force_high_plot_quality = False

    def copy_image_to_clipboard(self) -> None:
        QtWidgets.QApplication.clipboard().setPixmap(self.render_pixmap())
        self.image_copied.emit(f'Copied "{self.config.name}" image to clipboard')

    def autoscale(self) -> None:
        # Derive engineering-friendly bounds from visible data while retaining
        # the user's manual range/step text for a later Apply.
        self.x_axis.setTickSpacing()
        self.y_axis.setTickSpacing()
        bounds = self._autoscale_bounds()
        if bounds is None:
            self.x_axis.set_requested_endpoints(None, None)
            self.y_axis.set_requested_endpoints(None, None)
            self.plot.enableAutoRange()
            self.plot.autoRange(padding=0)
            self._remember_home_view()
            return
        x_low, x_high, y_low, y_high = bounds
        self.plot.disableAutoRange()
        self.plot.setXRange(x_low, x_high, padding=0)
        self.plot.setYRange(y_low, y_high, padding=0)
        self.x_axis.set_requested_endpoints(x_low, x_high)
        self.y_axis.set_requested_endpoints(y_low, y_high)
        self._remember_home_view()

    def _autoscale_bounds(self) -> tuple[float, float, float, float] | None:
        bounds = self._visible_trace_bounds()
        if bounds is None:
            return None
        x_low, x_high, y_low, y_high = bounds
        if x_low == x_high:
            x_low, x_high = x_low - 0.5, x_high + 0.5
        if y_low == y_high:
            expansion = max(abs(y_high) * 0.1, 1.0)
            y_low, y_high = y_low - expansion, y_high + expansion
        y_major = nice_tick_step(y_high - y_low)
        y_low = float(np.floor(y_low / y_major) * y_major)
        if self.config.display_format == DisplayFormat.LOG_MAG.value and y_high <= 0.5:
            y_high = 0.0
        else:
            y_high = float(np.ceil(y_high / y_major) * y_major)
        if y_low == y_high:
            y_low -= y_major
        return x_low, x_high, y_low, y_high

    def _visible_trace_bounds(self) -> tuple[float, float, float, float] | None:
        x_min, x_max = np.inf, -np.inf
        y_min, y_max = np.inf, -np.inf
        found = False
        for trace in self.config.traces:
            dataset = self.datasets.get(trace.dataset_id)
            if not dataset or not trace.visible:
                continue
            displayed = self._formatted_trace(dataset, trace.parameter)
            finite = np.isfinite(displayed)
            if not np.any(finite):
                continue
            frequencies = dataset.frequency_hz[finite] / 1e9
            shown = displayed[finite]
            x_min = min(x_min, float(np.min(frequencies)))
            x_max = max(x_max, float(np.max(frequencies)))
            y_min = min(y_min, float(np.min(shown)))
            y_max = max(y_max, float(np.max(shown)))
            found = True
        return (x_min, x_max, y_min, y_max) if found else None

    def add_marker(self) -> None:
        name = next_marker_name(self.config.markers)
        match = re.fullmatch(r"M(\d+)", name)
        number = int(match.group(1)) if match else len(self.config.markers) + 1
        x_range = self.plot.viewRange()[0]
        frequency_ghz = float(sum(x_range) / 2) if np.all(np.isfinite(x_range)) else 0.0
        if not self.config.traces:
            frequency_ghz = 0.0
        else:
            dataset = self.datasets.get(self.config.traces[0].dataset_id)
            if dataset and frequency_ghz == 0.0:
                frequency_ghz = float(dataset.frequency_hz[len(dataset.frequency_hz) // 2] / 1e9)
        marker = MarkerConfig(name=name, frequency_hz=frequency_ghz * 1e9, color=MARKER_COLORS[(number - 1) % len(MARKER_COLORS)])
        self.config.markers.append(marker)
        self.active_marker_id = marker.id
        self._rebuild_markers()
        self.marker_changed.emit(f"Added {marker.name} at {frequency_ghz:.6g} GHz")
        self.config_changed.emit()

    def delete_marker(self, marker_id: str) -> None:
        marker = next((item for item in self.config.markers if item.id == marker_id), None)
        if not marker:
            return
        self.config.markers[:] = [item for item in self.config.markers if item.id != marker_id]
        if self.active_marker_id == marker_id:
            self.active_marker_id = self.config.markers[-1].id if self.config.markers else None
        self._rebuild_markers()
        self.marker_changed.emit(f"Deleted {marker.name}")
        self.config_changed.emit()

    def set_marker_frequency(self, marker_id: str, frequency_hz: float) -> None:
        marker = next((item for item in self.config.markers if item.id == marker_id), None)
        line = self.marker_lines.get(marker_id)
        if not marker or not line:
            return
        marker.frequency_hz = float(frequency_hz)
        line.setValue(marker.frequency_hz / 1e9)
        self._update_marker_card(marker)
        self.select_marker(marker_id)
        self.marker_changed.emit(f"{marker.name} = {marker.frequency_hz / 1e9:.6g} GHz")
        self.config_changed.emit()

    def set_marker_visible(self, marker_id: str, visible: bool) -> None:
        marker = next((item for item in self.config.markers if item.id == marker_id), None)
        if not marker:
            return
        marker.visible = bool(visible)
        line = self.marker_lines.get(marker_id)
        if line:
            line.setVisible(marker.visible)
        self._update_marker_card(marker)
        self.marker_changed.emit(f"{marker.name} {'shown' if marker.visible else 'hidden'}")
        self.config_changed.emit()

    def set_all_markers_visible(self, visible: bool) -> None:
        if not self.config.markers:
            return
        for marker in self.config.markers:
            marker.visible = bool(visible)
            line = self.marker_lines.get(marker.id)
            if line:
                line.setVisible(marker.visible)
            self._update_marker_card(marker)
        self.marker_changed.emit("All markers shown" if visible else "All markers hidden")
        self.config_changed.emit()

    def delete_all_markers(self) -> None:
        if not self.config.markers:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete all markers",
            f'Delete all {len(self.config.markers)} marker(s) from "{self.config.name}"?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.config.markers.clear()
        self.active_marker_id = None
        self._rebuild_markers()
        self.marker_changed.emit("Deleted all markers")
        self.config_changed.emit()

    def select_marker(self, marker_id: str) -> None:
        self.active_marker_id = marker_id
        for marker in self.config.markers:
            line = self.marker_lines.get(marker.id)
            card = self.marker_cards.get(marker.id)
            active = marker.id == marker_id
            if line:
                line.setPen(pg.mkPen(marker.color, width=2.5 if active else 1.2))
            if card:
                card.set_active(active)

    def _clear_marker_cards(self) -> None:
        while self.marker_cards_layout.count():
            item = self.marker_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.marker_cards.clear()

    def _remove_marker_items(self) -> None:
        """Remove live PyQtGraph items before dropping Python references."""
        for line in list(self.marker_lines.values()):
            try:
                self.plot.removeItem(line)
            except Exception:
                # A preceding plot.clear() may already have removed the item.
                pass
        self.marker_lines.clear()

    def _rebuild_markers(self) -> None:
        self._remove_marker_items()
        self._clear_marker_cards()
        for marker in self.config.markers:
            line = pg.InfiniteLine(
                pos=marker.frequency_hz / 1e9,
                angle=90,
                movable=True,
                pen=pg.mkPen(marker.color, width=1.2),
                label=f"{marker.name} {{value:0.6g}} GHz",
                labelOpts={
                    "position": 0.95,
                    "color": marker.color,
                    "fill": "#ffffffdd" if self.theme == "Light" else "#111820cc",
                    "movable": True,
                },
            )
            line.sigPositionChanged.connect(lambda *_, marker_id=marker.id: self._marker_moved(marker_id))
            line.sigPositionChangeFinished.connect(
                lambda *_, marker_id=marker.id: self._marker_move_finished(marker_id)
            )
            self.plot.addItem(line, ignoreBounds=True)
            line.setVisible(marker.visible)
            self.marker_lines[marker.id] = line
            card = MarkerCard(marker)
            card.apply_theme(self.theme)
            card.delete_requested.connect(self.delete_marker)
            card.selected.connect(self.select_marker)
            card.frequency_changed.connect(self.set_marker_frequency)
            card.visibility_changed.connect(self.set_marker_visible)
            self.marker_cards_layout.addWidget(card)
            self.marker_cards[marker.id] = card
            self._update_marker_card(marker)
        self.marker_cards_layout.addStretch()
        self.marker_splitter.set_marker_available(bool(self.config.markers))
        self._sync_marker_panel_controls()
        if self.active_marker_id not in self.marker_lines:
            self.active_marker_id = self.config.markers[0].id if self.config.markers else None
        if self.active_marker_id:
            self.select_marker(self.active_marker_id)

    def _marker_moved(self, marker_id: str) -> None:
        marker = next((item for item in self.config.markers if item.id == marker_id), None)
        line = self.marker_lines.get(marker_id)
        if not marker or not line:
            return
        marker.frequency_hz = float(line.value()) * 1e9
        self.pending_marker_id = marker_id
        self.marker_drag_timer.start()
        if self.active_marker_id != marker_id:
            self.select_marker(marker_id)

    def _flush_marker_drag_update(self) -> None:
        marker_id = self.pending_marker_id
        self.pending_marker_id = None
        marker = next((item for item in self.config.markers if item.id == marker_id), None)
        if marker:
            self._update_marker_card(marker)
            self.marker_changed.emit(f"{marker.name} = {marker.frequency_hz / 1e9:.6g} GHz")

    def _marker_move_finished(self, marker_id: str) -> None:
        self.marker_drag_timer.stop()
        self.pending_marker_id = marker_id
        self._flush_marker_drag_update()
        self.config_changed.emit()

    def _update_marker_card(self, marker: MarkerConfig) -> None:
        card = self.marker_cards.get(marker.id)
        if not card:
            return
        readings = []
        for trace_index, trace in enumerate(self.config.traces):
            dataset = self.datasets.get(trace.dataset_id)
            item = self.trace_items.get(trace_index)
            if (
                not dataset
                or not trace.visible
                or trace.id in self.legend_hidden_trace_ids
                or (item is not None and not item.isVisible())
            ):
                continue
            if marker.frequency_hz < dataset.frequency_hz[0] or marker.frequency_hz > dataset.frequency_hz[-1]:
                value_text = "Out of range"
            else:
                upper = int(np.searchsorted(dataset.frequency_hz, marker.frequency_hz, side="left"))
                if upper <= 0:
                    index = 0
                elif upper >= dataset.frequency_hz.size:
                    index = dataset.frequency_hz.size - 1
                else:
                    lower = upper - 1
                    index = (
                        lower
                        if marker.frequency_hz - dataset.frequency_hz[lower]
                        <= dataset.frequency_hz[upper] - marker.frequency_hz
                        else upper
                    )
                y = self._formatted_trace(dataset, trace.parameter)
                value_text = f"{y[index]:.6g}"
            if self.config.legend_mode == "report":
                report_name = self.config.report_legend_labels.get(
                    trace.id, f"{dataset.name}: {trace.parameter}"
                )
                readings.append((trace_index, trace.color, report_name, "", value_text))
            else:
                readings.append((trace_index, trace.color, dataset.name, trace.parameter, value_text))
        card.set_content(marker, axis_label(self.config.display_format), readings)
        card.set_trace_highlights(self.selected_trace_indices, self.preview_trace_index)

    def _refresh_marker_cards(self) -> None:
        for marker in self.config.markers:
            self._update_marker_card(marker)

    def _refresh_marker_highlights(self) -> None:
        self.marker_highlight_timer.start()

    def _refresh_marker_highlights_now(self) -> None:
        for card in self.marker_cards.values():
            card.set_trace_highlights(self.selected_trace_indices, self.preview_trace_index)

    def export_png_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plot", f"{self.config.name}.png", "PNG (*.png)")
        if path:
            self.export_png(path)

    def export_png(self, path: str | Path) -> None:
        if not self.render_pixmap().save(str(path), "PNG"):
            raise OSError(f"Unable to save PNG: {path}")
