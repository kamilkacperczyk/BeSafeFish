"""
Walidacja nowego detektora (background subtraction) na zapisanych klatkach.
Porownuje nowy detektor z wynikami starego.
"""
import cv2
import numpy as np
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.fishing_detector import FishingDetector
from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS

DIAG_DIR = "diagnostyka_live"

def parse_log():
    entries = []
    with open(os.path.join(DIAG_DIR, "log.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fnum = int(parts[0][1:])
            t = float(parts[1].split("=")[1])
            color = parts[2].split("=")[1]
            fish_str = line.split("fish=")[1].split(" last=")[0]
            if fish_str.startswith("("):
                fx, fy = fish_str.strip("()").split(", ")
                fx, fy = int(fx), int(fy)
            else:
                fx, fy = None, None
            entries.append((fnum, t, color, fx, fy))
    return entries


def main():
    entries = parse_log()
    detector = FishingDetector()
    
    # Filtruj do klatek z aktywna gra
    game_entries = [e for e in entries if e[2] in ("white", "red")]
    
    print(f"=== Walidacja nowego detektora na {len(game_entries)} klatkach ===\n")
    
    old_detections = 0
    new_detections = 0
    total = 0
    
    red_old = 0
    red_new = 0
    red_total = 0
    
    white_old = 0
    white_new = 0
    white_total = 0
    
    for fnum, t, color, old_fx, old_fy in game_entries:
        frame_path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
        if not os.path.exists(frame_path):
            continue
        
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        
        new_pos = detector.find_fish_position(frame, circle_color=color)
        
        total += 1
        had_old = old_fx is not None
        has_new = new_pos is not None
        
        if had_old:
            old_detections += 1
        if has_new:
            new_detections += 1
        
        if color == "red":
            red_total += 1
            if had_old:
                red_old += 1
            if has_new:
                red_new += 1
        elif color == "white":
            white_total += 1
            if had_old:
                white_old += 1
            if has_new:
                white_new += 1
        
        # Szczegóły dla pierwszych wykryć w red phase
        if color == "red" and (has_new or had_old) and red_total <= 40:
            old_str = f"({old_fx},{old_fy})" if had_old else "None"
            new_str = f"({new_pos[0]},{new_pos[1]})" if has_new else "None"
            print(f"  f{fnum:04d} [{color}] old={old_str:15s} new={new_str:15s}")
    
    print(f"\n=== PODSUMOWANIE ===")
    print(f"Łącznie klatek gry: {total}")
    print(f"")
    print(f"  Stary detektor (frame diff):")
    print(f"    Ogółem:  {old_detections}/{total} = {100*old_detections/total:.1f}%")
    print(f"    RED:     {red_old}/{red_total} = {100*red_old/red_total:.1f}%")
    print(f"    WHITE:   {white_old}/{white_total} = {100*white_old/white_total:.1f}%")
    print(f"")
    print(f"  Nowy detektor (bg subtraction):")
    print(f"    Ogółem:  {new_detections}/{total} = {100*new_detections/total:.1f}%")
    print(f"    RED:     {red_new}/{red_total} = {100*red_new/red_total:.1f}%")
    print(f"    WHITE:   {white_new}/{white_total} = {100*white_new/white_total:.1f}%")
    
    improvement = new_detections / max(old_detections, 1)
    print(f"\n  Poprawa: {improvement:.1f}x więcej wykryć!")


if __name__ == "__main__":
    main()
