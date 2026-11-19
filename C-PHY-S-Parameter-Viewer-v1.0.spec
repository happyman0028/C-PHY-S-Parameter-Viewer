# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import PySide6


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'pytest', 'matplotlib'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Qt 6.11 expects the unversioned ICU exports provided by Windows 10's
# System32 ICU. A different ICU found on the build machine's PATH must not be
# bundled. Also keep the MSVC runtime at the same version as the PySide6 wheel;
# an older Python-distribution runtime can otherwise shadow it at startup.
msvc_runtime = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
}
a.binaries = [
    entry
    for entry in a.binaries
    if not Path(entry[0]).name.casefold().startswith("icu")
    and Path(entry[0]).name.casefold() not in msvc_runtime
]
pyside_dir = Path(PySide6.__file__).resolve().parent
a.binaries += [
    (name, str(pyside_dir / name), "BINARY")
    for name in sorted(msvc_runtime)
]

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='C-PHY-S-Parameter-Viewer-v1.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
