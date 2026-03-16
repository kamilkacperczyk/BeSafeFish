"""
Diagnostyka z graniem - bot GRA w minigre i jednoczesnie zapisuje
kazda klatke + szczegolowe logi.

Cel: zobaczyc co sie dzieje z detekcja gdy pojawiaja sie napisy HIT/MISS.

URUCHOM JAKO ADMIN:
  Start-Process powershell -Verb RunAs
"""

import os
import sys
import time
import ctypes
import cv2
import numpy as np

# Sprawdz admina
if not ctypes.windll.shell32.IsUserAnAdmin():
    print("BLAD: Uruchom jako Administrator!")
    sys.exit(1)

from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
from src.input_simulator import InputSimulator
from src.config import (
    FISHING_BOX_X, FISHING_BOX_Y,
    FISHING_BOX_WIDTH, FISHING_BOX_HEIGHT,
    GAME_WINDOW_X, GAME_WINDOW_Y,
    SCAN_INTERVAL, CLICKS_TO_WIN,
)

# Folder na wyniki
OUT_DIR = "diagnostyka_gra"
os.makedirs(OUT_DIR, exist_ok=True)

# Ile klatek zebrac
MAX_FRAMES = 300  # ~15 sek przy 50ms interval

capture = ScreenCapture()
detector = FishingDetector()
sim = InputSimulator()

print("=" * 50)
print("  DIAGNOSTYKA Z GRANIEM")
print("=" * 50)
print(f"Zbieranie {MAX_FRAMES} klatek z graniem na zywo.")
print("Bot bedzie klikalkal w rybke na czerwonym okregu.")
print()
print("Uruchom gre, uzyj robaka (F4), zarzuc wedke (SPACJA).")
print("Bot zacznie grac gdy wykryje minigre.")
print()
print("Start za 3 sekundy...")
time.sleep(3)

# Czekanie na minigre
print("[DIAG] Czekam na minigre (max 15s)...")
start_wait = time.time()
while time.time() - start_wait < 15.0:
    frame = capture.grab_fishing_box()
    if detector.is_fishing_active(frame):
        print("[DIAG] Minigra wykryta! Zaczynam grac i logowac!")
        break
    time.sleep(0.1)
else:
    print("[DIAG] Timeout - minigra sie nie pojawila. Zaczynam i tak (loguje co widze)...")

# Reset trackera rybki
detector.reset_tracking()

# Glowna petla: graj + loguj
log_lines = []
click_count = 0
last_fish_pos = None
t0 = time.time()

for i in range(1, MAX_FRAMES + 1):
    t_start = time.perf_counter()

    # Screenshot
    frame = capture.grab_fishing_box()

    # Detekcja
    debug_info = detector.get_debug_info(frame)
    white_px = debug_info["white_pixels"]
    bright_px = debug_info["bright_pixels"]
    color = debug_info["circle_color"]
    active = debug_info["fishing_active"]

    # Sledzenie rybki
    fish_pos = detector.find_fish_position(frame)
    if fish_pos is not None:
        last_fish_pos = fish_pos

    # Czas od startu
    elapsed = time.time() - t0

    # Decyzja + akcja
    action = "---"
    click_x, click_y = -1, -1

    if color == "red":
        if last_fish_pos is not None:
            fx, fy = last_fish_pos
            click_x, click_y = fx, fy
            # Klikamy BEZ POST_CLICK_DELAY (robimy to sami, bo logujemy)
            abs_x = GAME_WINDOW_X + FISHING_BOX_X + fx
            abs_y = GAME_WINDOW_Y + FISHING_BOX_Y + fy

            import pydirectinput
            from src.input_simulator import _focus_game_window
            _focus_game_window()
            pydirectinput.click(abs_x, abs_y)

            click_count += 1
            action = f"CLICK_FISH({fx},{fy})->abs({abs_x},{abs_y}) #{click_count}"
        else:
            # Fallback na srodek
            center_x = FISHING_BOX_WIDTH // 2
            center_y = FISHING_BOX_HEIGHT // 2
            abs_x = GAME_WINDOW_X + FISHING_BOX_X + center_x
            abs_y = GAME_WINDOW_Y + FISHING_BOX_Y + center_y

            import pydirectinput
            from src.input_simulator import _focus_game_window
            _focus_game_window()
            pydirectinput.click(abs_x, abs_y)

            click_count += 1
            click_x, click_y = center_x, center_y
            action = f"CLICK_CENTER({center_x},{center_y}) #{click_count}"

    # Formatowanie pozycji rybki
    fish_str = f"({fish_pos[0]},{fish_pos[1]})" if fish_pos else "???"
    last_str = f"({last_fish_pos[0]},{last_fish_pos[1]})" if last_fish_pos else "???"

    # Nazwa koloru
    if color == "red":
        color_name = "CZERWONY"
    elif color == "white":
        color_name = "BIALY"
    else:
        color_name = "BRAK"

    # Czas przetwarzania klatki
    t_proc = (time.perf_counter() - t_start) * 1000

    # Log
    log = (
        f"[{i:3d}] {elapsed:6.2f}s {color_name:8s} "
        f"white={white_px:4d} bright={bright_px:4d} "
        f"active={str(active):5s} "
        f"fish={fish_str:15s} last={last_str:15s} "
        f"proc={t_proc:5.1f}ms "
        f"action={action}"
    )
    log_lines.append(log)
    print(log)

    # Zapisz klatke
    fn = os.path.join(OUT_DIR, f"frame_{i:03d}.png")
    cv2.imwrite(fn, frame)

    # Jesli wygrana (3 trafienia) — kontynuuj logowanie ale nie klikaj
    # (chcemy widziec co sie dzieje po wygranej)

    # Odczekaj do nastepnej klatki (minus czas przetwarzania)
    t_elapsed_ms = t_proc
    wait_ms = max(0, SCAN_INTERVAL * 1000 - t_elapsed_ms)
    if wait_ms > 0:
        time.sleep(wait_ms / 1000)

# Zapisz log
log_path = os.path.join(OUT_DIR, "log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
    f.write("\n")
    f.write(f"=== PODSUMOWANIE ===\n")

    # Statystyki
    n_red = sum(1 for l in log_lines if "CZERWONY" in l)
    n_white = sum(1 for l in log_lines if "BIALY" in l)
    n_none = sum(1 for l in log_lines if "BRAK" in l)
    n_clicks = sum(1 for l in log_lines if "CLICK" in l)
    n_fish_found = sum(1 for l in log_lines if "CLICK_FISH" in l)
    n_fish_center = sum(1 for l in log_lines if "CLICK_CENTER" in l)

    f.write(f"Klatek: {MAX_FRAMES}\n")
    f.write(f"CZERWONY: {n_red}\n")
    f.write(f"BIALY: {n_white}\n")
    f.write(f"BRAK: {n_none}\n")
    f.write(f"Klikniec ogolnie: {n_clicks}\n")
    f.write(f"  - w rybke: {n_fish_found}\n")
    f.write(f"  - w srodek (fallback): {n_fish_center}\n")

print()
print(f"=== GOTOWE ===")
print(f"Zapisano {MAX_FRAMES} klatek do: {OUT_DIR}/")
print(f"Log: {log_path}")
print(f"Klikniec: {click_count}")
