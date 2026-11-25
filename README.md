# C-PHY S-Parameter Viewer & Analyzer — v1.2

Windows desktop engineering tool for C-PHY S6P and DIFF S4P analysis. It converts single-ended Touchstone data into C-PHY or DIFF modal S-parameters while retaining access to every SE trace.

## Install and run

Requires Windows and Python 3.11+.

```powershell
python -m pip install -r requirements.txt
python main.py
```

`python app.py` remains available as a compatibility launcher.

## Build the Windows EXE

Build in a clean virtual environment so another installed Qt binding cannot be collected accidentally:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-build.txt
$env:QT_API = "pyside6"
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean C-PHY-S-Parameter-Viewer-v1.2.spec
```

The release specification excludes an incompatible ICU DLL that may be discovered elsewhere on the build machine and uses the Windows system ICU expected by Qt. It also keeps the bundled MSVC runtime aligned with the PySide6 wheel. The resulting file is `dist\C-PHY-S-Parameter-Viewer-v1.2.exe`.

## Quick start

1. Click **Import SNP** to load one or more `.sNp` Touchstone files. Every port count supports SE; S4P also supports DIFF (dm/cm), while S6P also supports C-PHY. Multi-file import shows progress in the status bar.
2. Open the **Add Trace** tab.
3. Dataset entries are labeled `[S6P · C-PHY]` or `[S4P · DIFF]`. Select the applicable modal or SE trace type, then filter or search the parameters.
4. Select parameters and click **Add selected trace(s)**.
5. Add more plots with **Add Plot**. Each plot has its own display format and ranges and retains a readable minimum height; multiple plots use the main vertical scrollbar instead of being compressed. In **Plot / Traces**, use **Go** to activate/scroll to a plot, **Clone** to copy all of its settings, traces and markers after one confirmation (named `Clone-<original name>`), or **×** to delete it after two confirmations.
6. Use **Plot / Traces** to change visibility, color, line style, or remove traces.

**Plot Settings → Curve rendering quality** provides three persistent global modes:

- **Auto (Recommended)**: antialiased high-quality idle rendering; large plots temporarily use performance rendering while zooming or panning, then restore after 400 ms.
- **High quality**: antialiasing and detailed rendering remain enabled at all times.
- **Performance**: antialiasing is disabled and aggressive display downsampling remains enabled.

PNG export and the per-plot Copy button always render with high-quality antialiasing, regardless of the on-screen mode.

**Plot Settings → UI scale** scales the entire interface at 70%, 80%, 90%, or 100% (default). After saving a different scale, close and reopen the application to apply it.

Adding a Marker leaves the Marker Information panel collapsed by default. Use **▼ Marker Info** in the toolbar above the capture area to open it, then **▲** in the panel header to hide it. It expands the plot card outward to the right, preserving the graph width and using the main horizontal scrollbar for the added area. The toolbar control is outside copied/exported plot pixels. Each plot stores its own expanded state and width. PNG/Copy inclusion remains controlled independently by Plot Settings.

The Legend contains an **R/A** mode control. **R** opens Report mode and lets you replace long visible-trace descriptions with report-friendly text; hidden traces are omitted. Its editor includes a read-only sample of each curve's current color and Solid/Dash/Dot style, while actual style changes remain in Plot / Traces. **A** returns to the full Analysis Legend. Ctrl selections and Highlight state survive both directions, while Report-mode text remains neutral for presentation output. Labels and the selected mode are saved in the workspace.

Moving the Legend below the Frequency axis is temporarily disabled while that interaction is redesigned. The floating **↓** control and all bottom Legend controls are hidden. Existing workspaces that already store a below-Legend layout can still display that panel.

In Analysis mode, click an entry in either Legend position to hide or restore its curve. A hidden below-Legend entry is dimmed so it remains available for restoration. When switching to Report mode, hidden curves are omitted from both the Report Legend and its label editor. Report entries are presentation-only and cannot hide curves; return to Analysis mode to change curve visibility.

Marker Information follows the same visibility and naming rules: Legend-hidden curves are omitted, Analysis mode uses full Dataset/Parameter names, and Report mode uses the edited Report Legend text.

PNG/Copy output omits the R/A and position controls and reserves a small theme-colored margin at the right edge so the final trace endpoint is not visually clipped.

A validation file is included at `examples/cphy_demo_6port.s6p`. Regenerate it with `python examples/generate_demo_s6p.py`.

## S4P / DIFF definitions

The default four-port ordering is Side 1 P+/P− = physical Ports 1/2 and Side 2 P+/P− = physical Ports 3/4. It is editable per dataset; P+/P− order determines differential polarity.

The DIFF transform uses the orthonormal modal ordering `[d1, c1, d2, c2]`. Add Trace provides all 16 parameters in the `Sdd`, `Scc`, `Sdc`, and `Scd` families, with routes 11/12/21/22, plus all 16 SE parameters `S11` through `S44`.

For a 50 Ω common SE reference, **DIFF nominal** uses 100 Ω differential and 25 Ω common-mode reference impedances. The orthonormal mixed-mode transform already implies these references, so the nominal result is the standard direct `M · S · Mᵀ` expansion without a second renormalization. Custom differential and common-mode values are post-renormalized from 100/25 Ω.

## Port and modal definitions

Default ordering is Side 1 = physical Ports 1, 2, 3 and Side 2 = physical Ports 4, 5, 6. Mapping can be changed independently for each dataset; every physical port must occur exactly once.

| Definition | X | Y | CM |
|---|---|---|---|
| 1R | `[1, 0, -1] / √2` | `[-1, 2, -1] / √6` | `[1, 1, 1] / √3` |
| 2R | `[-1, 1, 0] / √2` | `[-1, -1, 2] / √6` | `[1, 1, 1] / √3` |
| 3R | `[0, -1, 1] / √2` | `[2, -1, -1] / √6` | `[1, 1, 1] / √3` |

For every definition the application calculates the full complex matrix after power-wave renormalization:

```text
S_SE,common = renormalize(S_SE, source Z0 → common SE Z0)
S_modal,equal(f) = T6 · S_SE,common(f) · T6ᵀ
Z_modal,nominal = [2×SE, 1.5×SE, SE÷3, 2×SE, 1.5×SE, SE÷3]
S_modal = renormalize(S_modal,equal, Z_modal,nominal → custom target Z0), only for Custom
```

Open **Datasets → Edit selected port / impedance** to select:

- **C-PHY nominal** (default): X = 2×SE, Y = 1.5×SE, CM = SE÷3 (100/75/16.6667 Ω for 50 Ω SE).
- **Custom**: independently enter positive real X, Y, and CM reference impedances.

The orthonormal C-PHY transform already implies the nominal physical modal references above. Therefore the default nominal result is exactly `T6 · S · T6ᵀ` and matches the expanded modal equations; it is not renormalized a second time. Custom values are post-renormalized from those nominal modal references.

The S6P source Z0 is retained. Unequal positive real per-port values are first renormalized to the selected common SE Z0. Complex, non-finite, and non-positive reference impedances are rejected. Settings can be applied to one or all loaded datasets and are stored in JSON workspaces.

## Parameter naming

Names use `Sabij_nR`: `a` is the response mode, `b` the stimulus mode, `i` the response side, and `j` the stimulus side.

- `Sxc12_1R`: Side 2 CM stimulus to Side 1 X response
- `Sxy12_3R`: Side 2 Y stimulus to Side 1 X response
- `Sxx21_2R`: Side 1 X stimulus to Side 2 X response

All nine mode families (`Sxx`, `Sxy`, `Sxc`, `Syx`, `Syy`, `Syc`, `Scx`, `Scy`, `Scc`) and four side routes are available for every R definition.

The **Single-ended** trace type provides all logical DUT-port parameters: 36 (`S11`–`S66`) for S6P or 16 (`S11`–`S44`) for S4P. These traces use the port-mapped matrix after normalization to Common SE Z0, before modal transformation. SE, C-PHY, and DIFF traces from loaded datasets can be plotted together.

## Analysis features

- Log/linear magnitude, wrapped/unwrapped phase, real and imaginary
- Group delay in ps or ns with nonuniform-frequency differentiation
- Independent display format per plot
- Mouse zoom/pan, autoscale, custom X/Y ranges, and configurable major tick/grid steps
- Mouse-wheel zoom is limited to the selected plot; the wheel scrolls the main plot list over inactive plots or non-graph areas
- Compact two-row plot controls and a FIT home-view button that restores the latest Apply/Autoscale view after mouse zoom or pan
- Light/Dark GUI switching with the last theme restored at startup
- The clear custom **check UnLock / closed-lock Lock** state beside the theme switch freezes both graph width and plot-action button positions; the theme-aware icons do not depend on system emoji fonts. Marker Info expands outward without moving Apply/Autoscale/+ Marker/Legend/C, and horizontal scrolling appears when required
- Per-plot Legend Hide/Show with drag confinement inside the plot and theme-aware text/background/border colors
- Simplified plot context menu with Plot Options and Average at the top level
- Plot Settings controls Legend, plot Title, and Axis font sizes for the active plot or all plots
- Ctrl + left-click is the only way to add curve selections; Find locates them, Clear Selected Traces removes them, and Shift + left drag pans
- Holding Ctrl previews the nearest curve with a temporary thicker line and theme-aware Legend highlight before selection
- Marker readout rows mirror Ctrl preview/selection highlighting, with all mouse controls documented in the built-in Guide
- Ctrl selections and pinned Find highlights persist through display changes and appended traces
- Plot / Traces provides Select Visible plus double-confirmed Delete All Traces
- Preset/custom PNG sizes use native target-size reflow instead of non-uniform bitmap stretching
- Per-plot **C** button to copy the configured plot image to the clipboard
- Multiple movable markers (`M1`, `M2`, …) with color-coded, scrollable readout cards beside the plot
- Direct GHz frequency entry, per-marker Hide/Show, and panel-level Hide All/Show All; hidden marker cards collapse their trace readings
- Individual marker deletion plus confirmed **Delete All**; position, color, and visibility are saved in the workspace
- Multi-dataset comparison and arbitrary parameters per plot
- Per-trace and multi-selected batch visibility, color, line style, and deletion
- Resizable active-trace columns and clickable sorting by visibility, dataset, natural parameter order, color, or style
- Active-trace search across dataset, filename, path, and parameter; supports `*` wildcard, `/` OR, and whitespace AND
- Active-plot PNG export from the main toolbar and long-form CSV trace export
- Shared Plot/PNG settings for export theme, marker lines, marker information panel, GUI ratio, presets, and custom pixel dimensions; settings persist across restarts
- JSON workspace save/load with missing-file relocation
- ALL-C-PHY / ALL-DIFF Add Trace entries when two or more matching datasets are loaded
- One ordered dataset list (C-PHY/S6P first, DIFF/S4P second) with multi-select batch mapping, impedance editing, and removal
- Duplicate-import detection highlights every dataset sharing the same normalized full path and processing settings; it never selects or deletes data automatically
- Built-in four-section Guide covering rotations, workflow, technical conventions, and revision history

Range fields accept either order (`-10 to 0` and `0 to -10` are equivalent). Autoscale changes only the current view and preserves the user's manual range text for later reuse. Automatic Plot names always advance from the highest existing `Plot N` number, and deleting a Plot requires confirmation.

## Tests

```powershell
python -m pytest -q
```

Tests cover C-PHY and DIFF orthonormal bases, direct SE-equation comparisons, 108 C-PHY names, 16 DIFF names, S4P/S6P loading, port mapping, group delay, log-zero handling, unequal-port Z0 normalization, and modal target-impedance normalization.

## Structure

```text
main.py / app.py
core/       C-PHY/DIFF transformation, loader, math, models, workspace
gui/        main application window
widgets/    PyQtGraph plot and trace selector
tests/      pytest core verification
examples/   synthetic Touchstone generator and data
```
