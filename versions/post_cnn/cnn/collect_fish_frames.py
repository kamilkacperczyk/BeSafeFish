"""
Zbiera WSZYSTKIE klatki z wykryta rybka do jednego folderu.
Zapisuje pelne klatki 279x247 + zaznacza pozycje rybki na kopii.
Uzytkownik wytnie sam ksztalt rybki (bez tla) i zapisze do fish_shapes/
"""

import os
import sys
import re
import csv
import json
import math
import collections
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fishing_detector import FishingDetector
from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS

BASE = Path(r'C:\Users\REDACTED-USER-PATH\Desktop\Repos\Kosa')
OUTPUT = BASE / 'rybki_do_oceny'
SHAPES_DIR = OUTPUT / 'fish_shapes'  # tu user wklei wyciety ksztalt

# Zrodla klatek
SOURCES = [
    ('test10_clean/raw',       'test10_clean/log.csv',              5),
    ('test8a_tracking/frames', 'tests/test8a_tracking/log.csv',     4),
    ('test8b_miss/frames',     'tests/test8b_miss/log.csv',         4),
    ('test8c_hit/frames',      'tests/test8c_hit/log.csv',          4),
    ('test9_long/frames',      'test9_long/log.csv',                5),
]


def load_log(log_path):
    """Zaladuj pozycje z log.csv."""
    positions = {}
    if not os.path.exists(log_path):
        return positions
    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fx = row.get('fish_x', '')
            fy = row.get('fish_y', '')
            fidx = row.get('frame', '')
            color = row.get('color', '')
            if fx and fy and fidx:
                try:
                    positions[int(fidx)] = {
                        'x': int(float(fx)),
                        'y': int(float(fy)),
                        'color': color,
                    }
                except ValueError:
                    pass
    return positions


def main():
    # Przygotuj foldery
    raw_dir = OUTPUT / 'raw'       # czyste klatki
    marked_dir = OUTPUT / 'marked' # klatki z zaznaczona pozycja
    raw_dir.mkdir(parents=True, exist_ok=True)
    marked_dir.mkdir(parents=True, exist_ok=True)
    SHAPES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ZBIERANIE KLATEK Z RYBKA DO OCENY")
    print("=" * 60)
    print(f"  Cel: {OUTPUT}")
    print()

    total = 0
    manifest = []

    for frames_rel, log_rel, digits in SOURCES:
        frames_dir = BASE / frames_rel
        log_path = BASE / log_rel
        source_name = frames_rel.split('/')[0]

        if not frames_dir.exists():
            continue

        positions = load_log(str(log_path))
        detector = FishingDetector()

        files = sorted(frames_dir.glob("frame_*.png"),
                       key=lambda f: int(re.search(r'(\d+)', f.stem).group(1)))

        prev_color = None
        batch_found = 0

        for frame_file in files:
            # Kolor z nazwy
            m = re.search(r'_(white|red|none)', frame_file.stem)
            color = m.group(1) if m else 'unknown'
            if color in ('none', 'unknown'):
                continue

            img = cv2.imread(str(frame_file))
            if img is None:
                continue

            # Reset detektora przy zmianie koloru
            if color != prev_color:
                detector.reset_tracking()
                prev_color = color

            # Szukaj rybki (bg subtraction)
            fish_pos = detector.find_fish_position(img, circle_color=color)

            # Fallback: pozycja z loga
            m_idx = re.search(r'(\d+)', frame_file.stem)
            fidx = int(m_idx.group(1)) if m_idx else -1
            if fish_pos is None and fidx in positions:
                fish_pos = (positions[fidx]['x'], positions[fidx]['y'])

            if fish_pos is None:
                continue

            fx, fy = fish_pos

            # Walidacja: wewnatrz okregu?
            dist = math.sqrt((fx - CIRCLE_CENTER_X)**2 + (fy - CIRCLE_CENTER_Y)**2)
            if dist > CIRCLE_RADIUS + 5:
                continue

            # Unikalna nazwa
            unique_name = f"{source_name}_{frame_file.name}"

            # Zapisz czysta klatke (RAW)
            cv2.imwrite(str(raw_dir / unique_name), img)

            # Zapisz klatke z zaznaczeniem
            marked = img.copy()
            # Czerwony krzyżyk na pozycji rybki
            cv2.drawMarker(marked, (fx, fy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            # Czerwony kwadrat 40x40 wokol rybki
            half = 20
            x1, y1 = max(0, fx-half), max(0, fy-half)
            x2, y2 = min(279, fx+half), min(247, fy+half)
            cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # Tekst z pozycja
            cv2.putText(marked, f"({fx},{fy})", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.imwrite(str(marked_dir / unique_name), marked)

            manifest.append({
                'file': unique_name,
                'fish_x': fx,
                'fish_y': fy,
                'color': color,
                'source': source_name,
            })

            total += 1
            batch_found += 1

        print(f"  {source_name}: {batch_found} klatek z rybka")

    # Zapisz manifest
    manifest_path = OUTPUT / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"RAZEM: {total} klatek z rybka")
    print(f"\nFoldery:")
    print(f"  raw/    — czyste klatki (279x247) — do wycinania ksztaltu ryby")
    print(f"  marked/ — klatki z zaznaczona pozycja (czerwony krzyzyk)")
    print(f"  fish_shapes/ — TU WKLEJ WYCIETY KSZTALT RYBY (PNG bez tla)")
    print(f"\nSciezka: {OUTPUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
