"""
TEST 8a - Tracking rybki BEZ klikania

Cel: Zbadac jak dobrze background subtraction trackuje rybke
gdy uzytkownik sam gra (lowienie manualne), a bot tylko obserwuje.
Brak klikniec = brak zaklocen od MISS/HIT tekstu.

Przebieg:
1. Nagrywa 500 klatek z okienka lowienia
2. Dla kazdej klatki:
   - detect_circle_color() -> faza (red/white/none)
   - find_fish_position() -> pozycja rybki (lub None)
3. Zapisuje klatki z naniesiona pozycja rybki do folderu test8a_tracking/frames/
4. Log przebiegu do test8a_tracking/log.csv
5. Raport statystyczny na koniec

Uruchomienie:
  python test8a_tracking.py
  (wymaga uruchomionej gry z aktywna minigra lowienia)
  UWAGA: bot NIE klika - gracz lowis sam!
"""

import os
import sys
import time
import csv
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
from src.config import (
    CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS,
    SCAN_INTERVAL,
)

OUTPUT_DIR = "test8a_tracking"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
NUM_FRAMES = 500
SAVE_EVERY_N = 5  # Zapisuj co N-ta klatke jako obraz (oszczednosc miejsca)


def draw_debug_frame(frame, frame_nr, color, fish_pos, detector):
    """Rysuje debug overlay na klatce."""
    display = frame.copy()
    
    # Rysuj okrag
    if color == "red":
        circle_color = (0, 0, 255)
    elif color == "white":
        circle_color = (200, 200, 200)
    else:
        circle_color = (80, 80, 80)
    
    cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, circle_color, 1)
    
    # Rysuj pozycje rybki
    if fish_pos is not None:
        fx, fy = fish_pos
        cv2.circle(display, (fx, fy), 6, (0, 255, 0), 2)  # zielony punkt
        cv2.line(display, (fx - 8, fy), (fx + 8, fy), (0, 255, 0), 1)
        cv2.line(display, (fx, fy - 8), (fx, fy + 8), (0, 255, 0), 1)
    
    # Info tekstowe
    status = f"#{frame_nr} | {color}"
    if fish_pos:
        status += f" | fish=({fish_pos[0]},{fish_pos[1]})"
    else:
        status += " | fish=NONE"
    
    cv2.putText(display, status, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return display


def main():
    print("=" * 60)
    print("  TEST 8a - Tracking rybki BEZ klikania")
    print("=" * 60)
    print()
    print("Upewnij sie ze:")
    print("  1. Gra jest uruchomiona")
    print("  2. Minigra lowienia jest aktywna")
    print("  3. TY lowisz recznie - bot tylko obserwuje!")
    print()
    print(f"Nagrywam {NUM_FRAMES} klatek...")
    print(f"Wyniki zapisze do: {OUTPUT_DIR}/")
    print()
    
    os.makedirs(FRAMES_DIR, exist_ok=True)
    
    capture = ScreenCapture()
    detector = FishingDetector()
    
    # CSV log
    log_path = os.path.join(OUTPUT_DIR, "log.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["frame", "timestamp", "color", "active", "fish_x", "fish_y", "white_px", "bright_px"])
    
    # Odliczanie
    for i in range(3, 0, -1):
        print(f"  Start za {i}...")
        time.sleep(1)
    print()
    
    start_time = time.perf_counter()
    
    # Statystyki
    stats = {
        "total": 0,
        "active": 0,
        "red": 0,
        "white": 0,
        "none": 0,
        "red_detected": 0,   # rybka znaleziona w fazie red
        "white_detected": 0, # rybka znaleziona w fazie white
        "detected_total": 0,
    }
    
    for i in range(NUM_FRAMES):
        frame = capture.grab_fishing_box()
        ts = time.perf_counter() - start_time
        
        if frame is None:
            writer.writerow([i+1, f"{ts:.3f}", "error", False, "", "", 0, 0])
            time.sleep(SCAN_INTERVAL)
            continue
        
        # Detekcja
        debug = detector.get_debug_info(frame)
        color = debug["circle_color"]
        active = debug["fishing_active"]
        white_px = debug["white_pixels"]
        bright_px = debug["bright_pixels"]
        
        fish_pos = detector.find_fish_position(frame, circle_color=color)
        
        # Statystyki
        stats["total"] += 1
        if active:
            stats["active"] += 1
        if color == "red":
            stats["red"] += 1
            if fish_pos:
                stats["red_detected"] += 1
        elif color == "white":
            stats["white"] += 1
            if fish_pos:
                stats["white_detected"] += 1
        else:
            stats["none"] += 1
        if fish_pos:
            stats["detected_total"] += 1
        
        # Log CSV
        fx = fish_pos[0] if fish_pos else ""
        fy = fish_pos[1] if fish_pos else ""
        writer.writerow([i+1, f"{ts:.3f}", color, active, fx, fy, white_px, bright_px])
        
        # Zapisz klatke co N
        if (i % SAVE_EVERY_N) == 0:
            display = draw_debug_frame(frame, i+1, color, fish_pos, detector)
            filename = f"frame_{i+1:04d}_{color}.png"
            cv2.imwrite(os.path.join(FRAMES_DIR, filename), display)
        
        # Status co 50 klatek
        if (i+1) % 50 == 0:
            det_pct = 100 * stats["detected_total"] / stats["total"] if stats["total"] else 0
            print(f"  [{i+1:3d}/{NUM_FRAMES}] det={det_pct:.1f}% | red={stats['red']} white={stats['white']} none={stats['none']}")
        
        time.sleep(SCAN_INTERVAL)
    
    log_file.close()
    elapsed = time.perf_counter() - start_time
    
    # RAPORT
    print()
    print("=" * 60)
    print("  RAPORT - TEST 8a (tracking bez klikania)")
    print("=" * 60)
    
    total = stats["total"]
    print(f"  Klatki lacznie:      {total}")
    print(f"  Czas nagrywania:     {elapsed:.1f}s ({total/elapsed:.1f} FPS)")
    print(f"  Aktywna minigra:     {stats['active']} ({100*stats['active']/total:.0f}%)")
    print()
    print(f"  Fazy:")
    print(f"    RED:   {stats['red']:4d} ({100*stats['red']/total:.0f}%)")
    print(f"    WHITE: {stats['white']:4d} ({100*stats['white']/total:.0f}%)")
    print(f"    NONE:  {stats['none']:4d} ({100*stats['none']/total:.0f}%)")
    print()
    
    red_det_pct = 100 * stats["red_detected"] / stats["red"] if stats["red"] else 0
    white_det_pct = 100 * stats["white_detected"] / stats["white"] if stats["white"] else 0
    total_det_pct = 100 * stats["detected_total"] / total if total else 0
    
    print(f"  Detekcja rybki:")
    print(f"    W RED:   {stats['red_detected']:4d}/{stats['red']:4d} ({red_det_pct:.1f}%)")
    print(f"    W WHITE: {stats['white_detected']:4d}/{stats['white']:4d} ({white_det_pct:.1f}%)")
    print(f"    TOTAL:   {stats['detected_total']:4d}/{total:4d} ({total_det_pct:.1f}%)")
    
    # Zapisz raport
    report_path = os.path.join(OUTPUT_DIR, "raport.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"TEST 8a - Tracking bez klikania ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Klatki: {total}, Czas: {elapsed:.1f}s, FPS: {total/elapsed:.1f}\n\n")
        f.write(f"Fazy: RED={stats['red']}, WHITE={stats['white']}, NONE={stats['none']}\n")
        f.write(f"Detekcja w RED: {stats['red_detected']}/{stats['red']} ({red_det_pct:.1f}%)\n")
        f.write(f"Detekcja w WHITE: {stats['white_detected']}/{stats['white']} ({white_det_pct:.1f}%)\n")
        f.write(f"Detekcja TOTAL: {stats['detected_total']}/{total} ({total_det_pct:.1f}%)\n")
    
    print(f"\n  Log CSV: {log_path}")
    print(f"  Raport: {report_path}")
    print(f"  Klatki: {FRAMES_DIR}/ (co {SAVE_EVERY_N}-ta)")
    print()
    print("GOTOWE!")


if __name__ == "__main__":
    main()
