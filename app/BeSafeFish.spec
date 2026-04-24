# -*- mode: python ; coding: utf-8 -*-
# BeSafeFish - PyInstaller spec (onedir - folder)
# Buduje folder z programem, potem pakowany do .zip
# Uzytkownik rozpakowuje zip, uruchamia BeSafeFish.exe jako Admin
#
# Build (z rootu repo): pyinstaller app/BeSafeFish.spec
# Output: dist/BeSafeFish/BeSafeFish.exe
#
# Uwaga: sciezki w datas/pathex sa **wzgledem katalogu z ktorego odpalasz**
# pyinstallera (zwykle root repo), NIE wzgledem pliku .spec.

a = Analysis(
    ['app\\besafefish.py'],
    pathex=[
        'app',
        'versions\\tryb1_rybka_klik\\post_cnn',
    ],
    binaries=[],
    datas=[
        ('app\\gui\\fish.ico', 'gui'),
        ('app\\gui\\assets', 'gui\\assets'),
        ('versions\\tryb1_rybka_klik\\post_cnn\\cnn\\models\\fish_patch_cnn.onnx', 'cnn\\models'),
        ('versions\\tryb1_rybka_klik\\post_cnn\\cnn\\models\\fish_patch_cnn.onnx.data', 'cnn\\models'),
    ],
    hiddenimports=[
        'PySide6.QtSvg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'matplotlib', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BeSafeFish',
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
    icon=['app\\gui\\fish.ico'],
    uac_admin=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BeSafeFish',
)
