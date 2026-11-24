from __future__ import annotations

from PySide6 import QtCore, QtWidgets


DEFAULT_PNG_SETTINGS = {
    "theme": "Current GUI",
    "include_markers": True,
    "include_marker_info": True,
    "size_mode": "Current GUI ratio",
    "preset_size": "1600 × 900",
    "width": 1600,
    "height": 900,
    "curve_render_quality": "Auto (Recommended)",
}

CURVE_RENDER_QUALITIES = ("Auto (Recommended)", "High quality", "Performance")
LEGEND_BELOW_ALIGNMENTS = ("Center", "Left", "Right", "Distribute evenly")


def load_png_settings(settings: QtCore.QSettings) -> dict:
    result = dict(DEFAULT_PNG_SETTINGS)
    result["theme"] = settings.value("png/theme", result["theme"], str)
    result["include_markers"] = settings.value("png/include_markers", True, bool)
    result["include_marker_info"] = settings.value("png/include_marker_info", True, bool)
    result["size_mode"] = settings.value("png/size_mode", result["size_mode"], str)
    result["preset_size"] = settings.value("png/preset_size", result["preset_size"], str)
    result["width"] = settings.value("png/width", 1600, int)
    result["height"] = settings.value("png/height", 900, int)
    quality = settings.value("png/curve_render_quality", result["curve_render_quality"], str)
    result["curve_render_quality"] = quality if quality in CURVE_RENDER_QUALITIES else "Auto (Recommended)"
    return result


def save_png_settings(settings: QtCore.QSettings, values: dict) -> None:
    for key, value in values.items():
        settings.setValue(f"png/{key}", value)


class PlotSettingsDialog(QtWidgets.QDialog):
    def __init__(self, values: dict, font_values: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Settings — Fonts / Rendering / PNG / Clipboard")
        self.setMinimumWidth(470)
        font_values = font_values or {"legend": 9, "title": 11, "axis": 9, "legend_below_alignment": "Center"}
        self.legend_font = QtWidgets.QSpinBox()
        self.title_font = QtWidgets.QSpinBox()
        self.axis_font = QtWidgets.QSpinBox()
        for spin in (self.legend_font, self.title_font, self.axis_font):
            spin.setRange(6, 48)
            spin.setSuffix(" pt")
        self.legend_font.setValue(int(font_values["legend"]))
        self.title_font.setValue(int(font_values["title"]))
        self.axis_font.setValue(int(font_values["axis"]))
        self.legend_below_alignment = QtWidgets.QComboBox()
        self.legend_below_alignment.addItems(LEGEND_BELOW_ALIGNMENTS)
        self.legend_below_alignment.setCurrentText(
            font_values.get("legend_below_alignment", "Center")
        )
        self.apply_fonts_to_all = QtWidgets.QCheckBox("Apply fonts and below-Legend alignment to all plots")
        font_form = QtWidgets.QFormLayout()
        font_form.addRow("Legend", self.legend_font)
        font_form.addRow("Title", self.title_font)
        font_form.addRow("Axis", self.axis_font)
        font_form.addRow("Below-Legend alignment", self.legend_below_alignment)
        font_form.addRow(self.apply_fonts_to_all)
        font_group = QtWidgets.QGroupBox("Active Plot Font Size")
        font_group.setLayout(font_form)
        self.curve_render_quality = QtWidgets.QComboBox()
        self.curve_render_quality.addItems(CURVE_RENDER_QUALITIES)
        self.curve_render_quality.setCurrentText(
            values.get("curve_render_quality", "Auto (Recommended)")
        )
        rendering_form = QtWidgets.QFormLayout()
        rendering_form.addRow("Curve rendering quality", self.curve_render_quality)
        rendering_note = QtWidgets.QLabel(
            "Auto smooths idle curves and temporarily favors performance while navigating large plots. "
            "High quality always smooths curves; Performance disables smoothing."
        )
        rendering_note.setWordWrap(True)
        rendering_form.addRow(rendering_note)
        rendering_group = QtWidgets.QGroupBox("On-screen Rendering")
        rendering_group.setLayout(rendering_form)
        self.theme = QtWidgets.QComboBox()
        self.theme.addItems(("Current GUI", "Light", "Dark"))
        self.theme.setCurrentText(values["theme"])
        self.include_markers = QtWidgets.QCheckBox("Include marker lines and labels")
        self.include_markers.setChecked(values["include_markers"])
        self.include_marker_info = QtWidgets.QCheckBox("Include marker information panel")
        self.include_marker_info.setChecked(values["include_marker_info"])
        self.size_mode = QtWidgets.QComboBox()
        self.size_mode.addItems(("Current GUI ratio", "Preset size", "Custom size"))
        self.size_mode.setCurrentText(values["size_mode"])
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(("1200 × 675", "1600 × 900", "1920 × 1080", "2560 × 1440"))
        self.preset.setCurrentText(values["preset_size"])
        self.width, self.height = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin in (self.width, self.height):
            spin.setRange(320, 7680)
        self.width.setValue(values["width"])
        self.height.setValue(values["height"])
        dimensions = QtWidgets.QHBoxLayout()
        dimensions.addWidget(self.width)
        dimensions.addWidget(QtWidgets.QLabel("×"))
        dimensions.addWidget(self.height)
        form = QtWidgets.QFormLayout()
        form.addRow("Export theme", self.theme)
        form.addRow(self.include_markers)
        form.addRow(self.include_marker_info)
        form.addRow("Output sizing", self.size_mode)
        form.addRow("Preset", self.preset)
        form.addRow("Custom pixels", dimensions)
        note = QtWidgets.QLabel(
            "Export Plot PNG and each plot's C (Copy) button use these shared settings. "
            "Preset/Custom sizes are re-laid out at the target pixels, so text is not stretched."
        )
        note.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(font_group)
        layout.addWidget(rendering_group)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.size_mode.currentTextChanged.connect(self._update_enabled)
        self._update_enabled()

    def _update_enabled(self) -> None:
        self.preset.setEnabled(self.size_mode.currentText() == "Preset size")
        custom = self.size_mode.currentText() == "Custom size"
        self.width.setEnabled(custom)
        self.height.setEnabled(custom)

    def values(self) -> dict:
        return {
            "theme": self.theme.currentText(),
            "include_markers": self.include_markers.isChecked(),
            "include_marker_info": self.include_marker_info.isChecked(),
            "size_mode": self.size_mode.currentText(),
            "preset_size": self.preset.currentText(),
            "width": self.width.value(),
            "height": self.height.value(),
            "curve_render_quality": self.curve_render_quality.currentText(),
        }

    def plot_font_values(self) -> dict:
        return {
            "legend": self.legend_font.value(),
            "title": self.title_font.value(),
            "axis": self.axis_font.value(),
            "legend_below_alignment": self.legend_below_alignment.currentText(),
            "apply_to_all": self.apply_fonts_to_all.isChecked(),
        }


class GuideDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("C-PHY / DIFF Viewer Guide")
        self.resize(760, 560)
        tabs = QtWidgets.QTabWidget()
        pages = (
            ("1. C-PHY Rotations", """<h2>C-PHY 1R / 2R / 3R</h2>
            <p><b>1R</b>: X ∝ [1,0,-1], Y ∝ [-1,2,-1]</p>
            <p><b>2R</b>: X ∝ [-1,1,0], Y ∝ [-1,-1,2]</p>
            <p><b>3R</b>: X ∝ [0,-1,1], Y ∝ [2,-1,-1]</p>
            <p>CM ∝ [1,1,1] for every rotation. X/Y/CM bases are power-normalized before reference-impedance renormalization.</p>"""),
            ("2. How to Use", """<h2>Workflow</h2><ol>
            <li>Use <b>Import SNP</b> to select one or more .sNp Touchstone files, including mixed port counts. Import progress appears in the status bar.</li>
            <li>In <b>Datasets</b>, verify or batch-edit port mapping and reference impedance. <b>Highlight duplicate imports</b> marks every repeated full-path + processing-settings group in yellow without selecting or deleting anything; run it again to recalculate, then remove entries manually if desired. S1P/S2P/S3P/S5P and other port counts provide SE; S4P additionally provides DIFF (dm/cm), and S6P additionally provides C-PHY.</li>
            <li>In <b>Add Trace</b>, choose a capability-labeled dataset and trace type, then filter parameters with ordinary text, <code>*</code> wildcards, <code>/</code> OR, or whitespace AND. ALL-DIFF / ALL-C-PHY and same-port-count ALL-SnP · SE groups appear when multiple matching datasets are loaded.</li>
            <li>Select one or more parameters and add them to the active plot. Use <b>Add Plot</b> for another independent graph. Every Plot keeps a readable minimum card height; multiple plots extend the page vertically and use the main scrollbar instead of compressing their graphs. In <b>Plot / Traces</b>, each plot row provides <b>Go</b>, <b>Clone</b>, and <b>×</b>: Go activates and scrolls to it, Clone copies the complete plot after one confirmation, and × deletes it after two confirmations.</li>
            <li>Set display format, X/Y limits and steps. Apply accepts either endpoint order; Autoscale preserves the typed fields and ignores manual steps.</li>
            <li>Add markers, move or enter their GHz position, and hide/show/delete marker lines or the Marker Info panel. Legend can be shown, hidden and dragged within the graph.</li>
            <li>Use <b>Plot / Traces</b> to search, sort, select visible rows, and batch-edit visibility, color, line style or deletion. Delete All Traces requires two confirmations.</li>
            <li>Use <b>Save Workspace</b> / <b>Load Workspace</b> to preserve datasets, mapping, impedance, plots, traces and markers.</li>
            <li>Use <b>Plot Settings</b> for font sizes, rendering quality and persistent PNG/Clipboard options. Export the active plot as PNG, export traces as CSV, or use its <b>C</b> button to copy the configured image.</li>
            </ol>
            <h2>Main Toolbar</h2><p>The right-side order is <b>✓ UnLock / closed-lock Lock → Plot Settings → Export Plot PNG → Export Trace CSV → Light/Dark Mode → Guide</b>. The state icons are custom theme-aware drawings rather than system emoji.</p>
            <h2>Plot Mouse Controls</h2><ul>
            <li><b>Hold Ctrl + move:</b> preview an unselected curve in blue/cyan; an already-selected curve changes to red as a removal preview.</li>
            <li><b>Ctrl + left-click:</b> add an unselected curve or remove an already-selected curve. Repeat to select multiple curves.</li>
            <li><b>Shift + left-drag:</b> pan the plot.</li>
            <li><b>Mouse wheel:</b> zoom only the selected plot (the card with the teal border). Over an unselected plot or outside the graph, the wheel scrolls the main plot list instead.</li>
            <li><b>Lock / UnLock:</b> the theme-aware check/lock control in the main toolbar keeps each graph and its plot-action toolbar at the current width while the main window is resized. Expanding Marker Info does not move Apply, Autoscale, + Marker, Legend, or C. Horizontal scrolling appears when needed.</li>
            <li><b>Marker information:</b> adding a Marker leaves its information panel collapsed by default. Use ▼ Marker Info beside the plot toolbar controls to open it and ▲ in the panel header to hide it. The panel expands outward to the right without squeezing the graph, and the main horizontal scrollbar exposes the added area. The boundary remains draggable while expanded.</li>
            <li><b>Legend R/A mode:</b> click <b>R</b> inside the Legend to enter Report mode and edit concise labels for currently visible traces. Hidden traces are omitted. The editor shows a read-only sample of each trace's current color and Solid/Dash/Dot style; change styles only in Plot / Traces. The button changes to <b>A</b>; click it to return to the full Analysis Legend. Ctrl selections and Highlight state are retained, while Report text stays in the normal theme color.</li>
            <li><b>Legend position:</b> moving the Legend below Frequency is temporarily disabled while that interaction is redesigned; the floating ↓ control and bottom controls are hidden. Existing saved below-Legend layouts can still be displayed.</li>
            <li><b>Report visibility:</b> a curve hidden through either Legend remains hidden when switching modes and is omitted from both the Report Legend and the Report Legend editor. Report Legend entries are presentation-only and cannot hide curves. Return to Analysis mode to restore or hide a curve.</li>
            <li><b>Below-Legend layout:</b> Plot Settings controls its shared Legend font and Center (default), Left, Right, or Distribute evenly alignment. These settings can be applied to the active plot or every plot.</li>
            <li><b>Right-click:</b> open Plot Options, Find Selected Trace(s), and Clear Selected Traces.</li>
            <li><b>Find Selected Trace(s):</b> open Plot / Traces, persistently highlight every selected row, and pin highlighted rows above the normal sort order.</li>
            <li><b>Clear Selected Traces:</b> clear curve, Legend, Marker, and table highlights.</li>
            <li><b>FIT:</b> restore the latest Apply/Autoscale view after zooming or panning.</li>
            </ul><p>Adding a marker does not open the information panel automatically. Use ▼ Marker Info above the capture area to open it; this control is not included in copied/exported plot pixels. Curves hidden through the Analysis Legend are omitted from Marker readings. Marker names use full Dataset/Parameter text in Analysis and the edited Report Legend text in Report mode. Ordinary left/right clicks do not select or clear curves. Ctrl selections and Find highlights survive display-property changes, newly imported files, mapping/impedance edits, and newly added traces until explicitly cleared or deleted.</p>
            <p>In Plot / Traces, <b>Select Visible</b> selects the currently filtered rows. <b>Delete All Traces</b> requires two confirmations before removing every trace from the active plot.</p>
            <p>Click a plot to select it before wheel-zooming. Marker controls continue to work independently.</p>
            <h2>Curve Rendering Quality</h2><ul>
            <li><b>Auto (Recommended):</b> smooth idle curves; large plots temporarily favor performance during navigation.</li>
            <li><b>High quality:</b> always use smooth antialiased curves.</li>
            <li><b>Performance:</b> disable smoothing and favor navigation speed.</li>
            </ul><p>PNG and Copy output always use high-quality curve rendering. R/A and ↓/↑ controls are omitted from output, and a small theme-colored margin is reserved at the right edge.</p>"""),
            ("3. Details", """<h2>Conventions</h2><p>S<sub>ij</sub> is response/output i due to stimulus/input j.</p><p>S4P DIFF nominal: Zdiff=2×SE, Zcommon=SE÷2. S6P C-PHY nominal: X=2×SE, Y=1.5×SE, CM=SE÷3.</p><p>The normalized DIFF/C-PHY transforms already imply these nominal modal references, so nominal results use the direct expanded equations without a second renormalization. Custom values apply an explicit post-transform target Z0.</p><p>SE traces are taken after port mapping and Common SE Z0 normalization. DIFF and C-PHY matrices are calculated lazily only when a corresponding trace is first used.</p>"""),
            ("4. Revision History", """<h2>v1.1</h2><ul><li>Unified Import SNP workflow for arbitrary Touchstone port counts; SE is always available, S4P adds DIFF (dm/cm), and S6P adds C-PHY</li><li>Duplicate-import finder highlights repeated full-path + processing-settings groups without selecting or deleting data</li><li>Readable fixed minimum plot-card height with vertical scrolling instead of compressing multiple plots</li><li>Report/Analysis Legend editing, visibility synchronization, theme-aware Highlight retention, and cleaner capture/export controls</li><li>Marker Information expands outward, starts collapsed, follows Legend visibility/report names, and keeps frozen plot controls fixed</li><li>Clear Lock/UnLock state icons, streamlined toolbar and per-plot controls, Clone-prefixed plot names, and safer confirmations</li><li>Large-dataset responsiveness, lazy modal transforms, rendering-quality choices, and active-plot-only wheel zoom</li></ul><h2>v1.0</h2><ul><li>Initial C-PHY/DIFF/SE viewer and analyzer release</li><li>Configurable port mapping and power-wave reference impedance</li><li>Multiple plots, ranges/steps, Autoscale, FIT, Legend, Markers, Copy, PNG/CSV export, workspace save/load, and Light/Dark themes</li><li>Ctrl curve selection with synchronized Legend, Marker, and Trace-table highlighting</li></ul>"""),
        )
        for title, html in pages:
            browser = QtWidgets.QTextBrowser()
            browser.setHtml(html)
            tabs.addTab(browser, title)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(close)
