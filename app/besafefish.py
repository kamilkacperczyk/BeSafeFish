"""
BeSafeFish — punkt wejscia aplikacji.

Uruchomienie:
    python besafefish.py

Wymagania:
    - PySide6
    - Wszystkie zaleznosci z requirements.txt
    - Windows (pydirectinput wymaga Windows)
"""

import sys
import os

# Dodaj app/ (dla importow gui.*) oraz katalog biezacego wariantu bota
# (dla importow src.* i cnn.*) do PYTHONPATH.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)

# Aktualnie jedyny wariant bota uzywany przez GUI to tryb1_rybka_klik/post_cnn.
# W przyszlosci GUI bedzie wybieralo wariant dynamicznie (Etap 2 refaktoru bota).
BOT_VARIANT_DIR = os.path.join(
    REPO_ROOT, "versions", "tryb1_rybka_klik", "post_cnn"
)

for path in (APP_DIR, BOT_VARIANT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# Pod PyInstallerem sciezka bazowa to sys._MEIPASS — tam leza spakowane zasoby
# (cnn\\models\\*.onnx itd.). Dodajemy go tez, zeby importy 'cnn.*' dzialaly
# rowniez w zbudowanym .exe (gdzie BOT_VARIANT_DIR nie istnieje).
if getattr(sys, "frozen", False):
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and meipass not in sys.path:
        sys.path.insert(0, meipass)

import ctypes
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from gui.app import BeSafeFishApp
from gui.styles import DARK_THEME


def main():
    # Windows: ustaw AppUserModelID zeby ikona byla widoczna na pasku zadan
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Kosa.BeSafeFish.1.0")
    except Exception:
        pass
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("BeSafeFish")
    app.setOrganizationName("Kosa")
    app.setStyleSheet(DARK_THEME)

    window = BeSafeFishApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
