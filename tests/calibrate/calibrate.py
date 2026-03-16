"""
Proste narzedzie kalibracyjne - robi screenshot i zapisuje do pliku.
Uruchom JAKO ADMINISTRATOR.
"""

import sys
import ctypes
import cv2
import numpy as np
import time
import mss

if not ctypes.windll.shell32.IsUserAnAdmin():
    print("BLAD: Uruchom jako Administrator!")
    sys.exit(1)

print("=== KALIBRACJA ===")
print("Aktywuje okno gry...")

import pygetwindow as gw
win = None
for w in gw.getAllWindows():
    if 'eryndos' in w.title.lower() and w.title.strip() and not w.isMinimized:
        win = w
        break
    elif 'eryndos' in w.title.lower() and w.title.strip() and w.isMinimized:
        w.restore()
        time.sleep(0.3)
        win = w
        break

if win:
    win.activate()
    print(f"Okno: {win.title} ({win.width}x{win.height})")
else:
    print("Nie znaleziono okna Eryndos!")

time.sleep(1)
print("Robie screenshot za 1 sekunde...")
time.sleep(2)

sct = mss.mss()

# Screenshot calego glownego monitora
monitor = sct.monitors[1]
screenshot = sct.grab(monitor)
frame = np.array(screenshot)
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

cv2.imwrite("screenshot_caly.png", frame)
print(f"Zapisano: screenshot_caly.png ({frame.shape[1]}x{frame.shape[0]})")

# Rysuj ramke fishing boxa na screenshocie
from src.config import FISHING_BOX_X, FISHING_BOX_Y, FISHING_BOX_WIDTH, FISHING_BOX_HEIGHT
marked = frame.copy()
cv2.rectangle(marked,
    (FISHING_BOX_X, FISHING_BOX_Y),
    (FISHING_BOX_X + FISHING_BOX_WIDTH, FISHING_BOX_Y + FISHING_BOX_HEIGHT),
    (0, 255, 0), 3)
cv2.putText(marked, "FISHING BOX", (FISHING_BOX_X, FISHING_BOX_Y - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite("screenshot_z_ramka.png", marked)
print(f"Zapisano: screenshot_z_ramka.png (z zielona ramka fishing boxa)")

# Wytnij sam fishing box
fishing_crop = frame[FISHING_BOX_Y:FISHING_BOX_Y+FISHING_BOX_HEIGHT,
                     FISHING_BOX_X:FISHING_BOX_X+FISHING_BOX_WIDTH]
cv2.imwrite("screenshot_fishbox.png", fishing_crop)
print(f"Zapisano: screenshot_fishbox.png (sam fishing box {FISHING_BOX_WIDTH}x{FISHING_BOX_HEIGHT})")

print()
print("Otworz screenshot_z_ramka.png - czy zielona ramka pokrywa okienko minigry?")
