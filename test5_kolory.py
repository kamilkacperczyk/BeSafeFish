"""
TEST 5 - Weryfikacja detekcji kolorow (powtorka)

Cel: Ponowne sprawdzenie czy detekcja koloru okregu dziala poprawnie
po zmianach w kodzie (background subtraction).

Przebieg:
1. Robi 20 screenshotow okienka lowienia (co 0.5s)
2. Dla kazdego screenshota:
   - detect_circle_color() -> wynik automatyczny
   - is_fishing_active() -> czy minigra aktywna
   - get_debug_info() -> szczegoly (white_pixels, bright_pixels)
3. Zapisuje screenshoty do folderu test5_kolory/
4. Generuje raport na koniec

Uruchomienie:
  python test5_kolory.py
  (wymaga uruchomionej gry z aktywna minigra lowienia)
"""

import os
import sys
import time
import cv2
import numpy as np

# Dodaj sciezke projektu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector

OUTPUT_DIR = "test5_kolory"
NUM_SCREENSHOTS = 20
INTERVAL = 0.5  # sekundy miedzy screenshotami


def main():
    print("=" * 60)
    print("  TEST 5 - Weryfikacja detekcji kolorow (20 screenshotow)")
    print("=" * 60)
    print()
    print("Upewnij sie ze:")
    print("  1. Gra jest uruchomiona")
    print("  2. Minigra lowienia jest aktywna (wedka zarzucona)")
    print()
    print(f"Robie {NUM_SCREENSHOTS} screenshotow co {INTERVAL}s...")
    print(f"Wyniki zapisze do folderu: {OUTPUT_DIR}/")
    print()
    
    # Przygotuj folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    capture = ScreenCapture()
    detector = FishingDetector()
    
    results = []
    
    # Odliczanie
    for i in range(3, 0, -1):
        print(f"  Start za {i}...")
        time.sleep(1)
    print()
    
    for i in range(NUM_SCREENSHOTS):
        frame = capture.grab_fishing_box()
        
        if frame is None:
            print(f"  [{i+1:2d}/{NUM_SCREENSHOTS}] BLAD: nie udalo sie zrobic screenshota")
            results.append({
                "nr": i + 1,
                "active": False,
                "color": "error",
                "white_px": 0,
                "bright_px": 0,
            })
            time.sleep(INTERVAL)
            continue
        
        # Analiza
        debug = detector.get_debug_info(frame)
        color = debug["circle_color"]
        active = debug["fishing_active"]
        white_px = debug["white_pixels"]
        bright_px = debug["bright_pixels"]
        
        # Zapisz screenshot z adnotacja
        display = frame.copy()
        label = f"#{i+1} | {color} | W:{white_px} B:{bright_px}"
        
        if color == "red":
            border = (0, 0, 255)
        elif color == "white":
            border = (255, 255, 255)
        else:
            border = (128, 128, 128)
        
        cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1), border, 3)
        cv2.putText(display, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border, 2)
        
        filename = f"test5_{i+1:02d}_{color}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), display)
        
        # Wynik
        result = {
            "nr": i + 1,
            "active": active,
            "color": color,
            "white_px": white_px,
            "bright_px": bright_px,
        }
        results.append(result)
        
        # Status
        symbol = "🔴" if color == "red" else ("⚪" if color == "white" else "⬛")
        print(f"  [{i+1:2d}/{NUM_SCREENSHOTS}] {symbol} {color:6s} | white_px={white_px:5d} | bright_px={bright_px:5d} | active={active}")
        
        time.sleep(INTERVAL)
    
    print()
    print("=" * 60)
    print("  RAPORT")
    print("=" * 60)
    
    # Statystyki
    total = len(results)
    active_count = sum(1 for r in results if r["active"])
    red_count = sum(1 for r in results if r["color"] == "red")
    white_count = sum(1 for r in results if r["color"] == "white")
    none_count = sum(1 for r in results if r["color"] == "none")
    error_count = sum(1 for r in results if r["color"] == "error")
    
    print(f"  Lacznie screenshotow: {total}")
    print(f"  Aktywna minigra:      {active_count} ({100*active_count/total:.0f}%)")
    print(f"  Czerwony okrag:       {red_count} ({100*red_count/total:.0f}%)")
    print(f"  Bialy okrag:          {white_count} ({100*white_count/total:.0f}%)")
    print(f"  Brak okregu:          {none_count} ({100*none_count/total:.0f}%)")
    if error_count:
        print(f"  Bledy:                {error_count}")
    
    # White pixels stats per color
    for c in ["white", "red"]:
        px_vals = [r["white_px"] for r in results if r["color"] == c]
        if px_vals:
            print(f"\n  {c.upper()} - white_pixels: min={min(px_vals)}, max={max(px_vals)}, avg={sum(px_vals)/len(px_vals):.0f}")
    
    # Zapisz raport
    report_path = os.path.join(OUTPUT_DIR, "raport.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"TEST 5 - Detekcja kolorow ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Lacznie: {total}\n")
        f.write(f"Aktywna minigra: {active_count}\n")
        f.write(f"Czerwony: {red_count}, Bialy: {white_count}, None: {none_count}\n\n")
        f.write(f"{'Nr':>3} {'Color':>6} {'Active':>8} {'WhitePx':>8} {'BrightPx':>9}\n")
        f.write(f"{'-'*40}\n")
        for r in results:
            f.write(f"{r['nr']:3d} {r['color']:>6} {str(r['active']):>8} {r['white_px']:>8} {r['bright_px']:>9}\n")
    
    print(f"\n  Raport zapisany: {report_path}")
    print(f"  Screenshoty w: {OUTPUT_DIR}/")
    print()
    print("GOTOWE!")


if __name__ == "__main__":
    main()
