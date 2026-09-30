# PyInstaller build for the Model Forge desktop app:  python -m PyInstaller --clean ModelForge.spec
# Produces a single file, dist/ModelForge.exe (dist/ModelForge on Linux/macOS), with the app icon.
# COLMAP and ffmpeg are not bundled: they're found on PATH (or COLMAP_BIN / FFMPEG_BIN) at run time.
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["app.py"],
    pathex=[],
    datas=[("frontend", "frontend"), ("modelforge.ico", "."), ("modelforge.png", ".")],
    hiddenimports=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ModelForge",
    console=False,  # windowed app, no console
    icon="modelforge.ico",  # .exe, taskbar and title-bar icon
    runtime_tmpdir=None,
)
