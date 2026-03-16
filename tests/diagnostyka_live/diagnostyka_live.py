"""
Diagnostyka NA ZYWO — nagrywa klatki z miniory lowienia
i pokazuje/loguje co widzi detektor.

Uruchom: .\venv\Scripts\python.exe diagnostyka_live.py
(jako Administrator, gra musi byc otwarta z minigra aktywna)

Nacisniecia:
  q — zakoncz
  s — zapisz aktualny screenshot

Wynik: folder diagnostyka_live/ z klatkami i log.txt
"""

import os
import sys
import time
import cv2
import numpy as np
import math

# Dodaj folder projektu do path
sys.path.insert(0, os.path.dirname(__file__))

from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
from src.config import (
    CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, CIRCLE_SAFE_MARGIN,
    SCAN_INTERVAL,
)

OUTPUT_DIR = "diagnostyka_live"
os.makedirs(OUTPUT_DIR, exist_ok=True)

capture = ScreenCapture()
detector = FishingDetector()

frame_num = 0
log_lines = []
last_fish_pos = None
prev_color = "none"

print("=" * 50)
print("  DIAGNOSTYKA LIVE — nagrywanie minigry")
print("=" * 50)
print(f"Zapis do: {OUTPUT_DIR}/")
print("Nacisnij 'q' w oknie podgladu aby zakonczyc")
print("Nacisnij 's' aby zapisac klatke")
print()
print("Nagrywam od razu - nacisnij 'q' aby zakonczyc")
print("Zarzuc wedke w grze!")

save_all = True  # zapisz KAZDA klatke
start_time = time.perf_counter()
inactive_count = 0  # ile klatek z rzedu minigra nieaktywna

while True:
    frame = capture.grab_fishing_box()
    color = detector.detect_circle_color(frame)
    fish_pos = detector.find_fish_position(frame, circle_color=color)
    predicted = detector.predict_fish_position(ahead_ms=30.0)
    
    if fish_pos is not None:
        last_fish_pos = fish_pos
    
    frame_num += 1
    elapsed = time.perf_counter() - start_time
    
    # --- Rysuj na klatce ---
    display = frame.copy()
    
    # Okrag i bezpieczna strefa
    safe_r = CIRCLE_RADIUS - CIRCLE_SAFE_MARGIN
    cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, (128, 128, 128), 1)
    cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), safe_r, (0, 255, 0), 1)
    
    # Kolor ramki
    if color == "red":
        border = (0, 0, 255)
    elif color == "white":
        border = (255, 255, 255)
    else:
        border = (80, 80, 80)
    cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1), border, 3)
    
    # Fish position z detectora (ZIELONY = swieze wykrycie)
    if fish_pos is not None:
        cv2.circle(display, fish_pos, 10, (0, 255, 0), 2)
        cv2.putText(display, "FISH", (fish_pos[0]+12, fish_pos[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Last known position (ZOLTY = stara pozycja)
    if last_fish_pos is not None and fish_pos is None:
        cv2.circle(display, last_fish_pos, 10, (0, 255, 255), 1)
        cv2.putText(display, "last", (last_fish_pos[0]+12, last_fish_pos[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Predicted position (NIEBIESKI)
    if predicted is not None:
        cv2.circle(display, (int(predicted[0]), int(predicted[1])), 6, (255, 0, 0), 2)
    
    # Info
    cv2.putText(display, f"f{frame_num:04d} {color} t={elapsed:.2f}s", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, border, 1)
    
    fish_str = f"fish={fish_pos}" if fish_pos else "fish=None"
    last_str = f"last={last_fish_pos}" if last_fish_pos else "last=None"
    cv2.putText(display, f"{fish_str} {last_str}", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    # Logika: gdzie by bot kliknal?
    click_target = predicted or last_fish_pos
    if color == "red" and click_target:
        # Clamp
        dx = click_target[0] - CIRCLE_CENTER_X
        dy = click_target[1] - CIRCLE_CENTER_Y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > safe_r and dist > 0:
            scale = safe_r / dist
            cx = int(CIRCLE_CENTER_X + dx * scale)
            cy = int(CIRCLE_CENTER_Y + dy * scale)
        else:
            cx, cy = click_target[0], click_target[1]
        # CZERWONY KRZYZYK = tu by kliknal bot
        cv2.drawMarker(display, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(display, f"CLICK({cx},{cy})", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # Dodatkowe info z detectora
    debug = detector.get_debug_info(frame)
    bright_px = debug.get('bright_white_pixels', 0)
    cv2.putText(display, f"bright={bright_px}", (5, display.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    # Log
    log_line = (f"f{frame_num:04d} t={elapsed:.3f} color={color} "
                f"fish={fish_pos} last={last_fish_pos} pred={predicted} "
                f"bright={bright_px}")
    log_lines.append(log_line)
    
    # Zapisz klatke
    if save_all:
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{frame_num:04d}.png", display)
    
    # Pokaz
    # Powieksz 2x
    big = cv2.resize(display, (display.shape[1]*2, display.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Diagnostyka", big)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fn = f"{OUTPUT_DIR}/manual_{frame_num:04d}.png"
        cv2.imwrite(fn, frame)
        print(f"Zapisano: {fn}")
    
    # Nie zamykamy automatycznie — tylko 'q' zamyka
    # (is_fishing_active moze falszywie zwrocic False w czerwonej fazie)
    
    prev_color = color
    time.sleep(SCAN_INTERVAL)

# Zapisz log
log_path = f"{OUTPUT_DIR}/log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")

print(f"\nZapisano {frame_num} klatek do {OUTPUT_DIR}/")
print(f"Log: {log_path}")

cv2.destroyAllWindows()
