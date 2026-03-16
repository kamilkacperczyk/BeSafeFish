"""
Diagnostyka bota - robi screeny co 100ms przez 20 sekund.
Zapisuje kazda klatke z informacja o detekcji.
Uzycie: uruchom minigre lowienia, potem odpal ten skrypt.
"""
import os
import time
import sys
import cv2

sys.path.insert(0, ".")
from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector

OUTPUT_DIR = "diagnostyka"
os.makedirs(OUTPUT_DIR, exist_ok=True)

capture = ScreenCapture()
detector = FishingDetector()

NUM_FRAMES = 200  # 200 klatek
DELAY = 0.1       # co 100ms = 10 FPS

print("=== DIAGNOSTYKA BOTA ===")
print(f"Zrobie {NUM_FRAMES} klatek co {DELAY}s ({NUM_FRAMES * DELAY:.0f}s)")
print(f"Zapisuje do: {OUTPUT_DIR}/")
print()
print("Uruchom minigre lowienia w grze!")
print("Start za 3 sekundy...")
time.sleep(3)

log_lines = []
last_fish_pos = None

for i in range(1, NUM_FRAMES + 1):
    t_start = time.time()

    frame = capture.grab_fishing_box()
    color = detector.detect_circle_color(frame)
    fish_pos = detector.find_fish_position(frame)
    debug = detector.get_debug_info(frame)

    if fish_pos is not None:
        last_fish_pos = fish_pos

    white_px = debug["white_pixels"]
    bright_px = debug["bright_pixels"]
    active = debug["fishing_active"]

    if color == "red":
        label = "CZERWONY"
    elif color == "white":
        label = "BIALY"
    else:
        label = "BRAK"

    fish_str = f"({fish_pos[0]},{fish_pos[1]})" if fish_pos else "???"
    last_str = f"({last_fish_pos[0]},{last_fish_pos[1]})" if last_fish_pos else "???"

    # Rysuj overlay na klatce
    display = frame.copy()
    # Ramka koloru
    if color == "red":
        border = (0, 0, 255)
    elif color == "white":
        border = (255, 255, 255)
    else:
        border = (128, 128, 128)
    cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1), border, 2)
    cv2.putText(display, f"#{i} {label}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, border, 1)
    cv2.putText(display, f"w={white_px} b={bright_px}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    if fish_pos:
        cv2.circle(display, fish_pos, 8, (0, 255, 255), 2)
        cv2.line(display, (fish_pos[0]-12, fish_pos[1]), (fish_pos[0]+12, fish_pos[1]), (0, 255, 255), 1)
        cv2.line(display, (fish_pos[0], fish_pos[1]-12), (fish_pos[0], fish_pos[1]+12), (0, 255, 255), 1)
    if last_fish_pos and fish_pos != last_fish_pos:
        cv2.circle(display, last_fish_pos, 5, (0, 100, 100), 1)

    filename = f"{i:03d}_{label}_f{fish_str}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), display)

    log = f"[{i:3d}] {label:9s} white={white_px:4d} bright={bright_px:4d} active={active} fish={fish_str:12s} last={last_str}"
    log_lines.append(log)

    # Print co 10 klatek
    if i % 10 == 0 or color == "red":
        print(log)

    elapsed = time.time() - t_start
    wait = max(0, DELAY - elapsed)
    if wait > 0:
        time.sleep(wait)

# Zapisz logi
log_path = os.path.join(OUTPUT_DIR, "log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print()
print(f"Gotowe! {NUM_FRAMES} klatek zapisanych w {OUTPUT_DIR}/")
print(f"Logi: {log_path}")

# Podsumowanie
red_frames = sum(1 for l in log_lines if "CZERWONY" in l)
white_frames = sum(1 for l in log_lines if "BIALY" in l and "CZERWONY" not in l)
none_frames = sum(1 for l in log_lines if "BRAK" in l)
fish_found = sum(1 for l in log_lines if "fish=(?" not in l and "fish=???" not in l)
print(f"\nPodsumowanie:")
print(f"  CZERWONY: {red_frames} klatek")
print(f"  BIALY:    {white_frames} klatek")
print(f"  BRAK:     {none_frames} klatek")
print(f"  Rybka wykryta: {fish_found}/{NUM_FRAMES} klatek")
