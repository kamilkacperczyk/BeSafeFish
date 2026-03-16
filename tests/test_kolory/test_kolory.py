"""
Test rozpoznawania kolorow okregu.
Robi 10 screenshotow co 2 sekundy i zapisuje je z wynikiem detekcji.

Uzycie:
  1. Uruchom gre i zacznij lowic ryby
  2. Odpal ten skrypt
  3. Skrypt zrobi 10 screenshotow i zapisze je w folderze test_kolory/
"""
import os
import time
import sys

sys.path.insert(0, ".")
from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
import cv2

OUTPUT_DIR = "test_kolory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

capture = ScreenCapture()
detector = FishingDetector()

NUM_SHOTS = 10
DELAY = 2.0

print("=== TEST KOLOROW OKREGU ===")
print(f"Zrobie {NUM_SHOTS} screenshotow co {DELAY}s")
print(f"Zapisuje do folderu: {OUTPUT_DIR}/")
print()
print("Uruchom minigre lowienia w grze!")
print("Start za 3 sekundy...")
time.sleep(3)

for i in range(1, NUM_SHOTS + 1):
    frame = capture.grab_fishing_box()
    debug = detector.get_debug_info(frame)

    color = debug["circle_color"]
    white_px = debug["white_pixels"]
    bright_px = debug["bright_pixels"]
    active = debug["fishing_active"]

    if color == "red":
        label = "CZERWONY"
    elif color == "white":
        label = "BIALY"
    else:
        label = "BRAK"

    filename = f"{i:02d}_{label}_w{white_px}_b{bright_px}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, frame)

    print(f"[{i:2d}/{NUM_SHOTS}] Kolor: {label:9s} | white_px={white_px:4d} | bright_px={bright_px:4d} | active={active} | -> {filename}")

    if i < NUM_SHOTS:
        time.sleep(DELAY)

print()
print("Gotowe! Screenshoty zapisane w " + OUTPUT_DIR + "/")
