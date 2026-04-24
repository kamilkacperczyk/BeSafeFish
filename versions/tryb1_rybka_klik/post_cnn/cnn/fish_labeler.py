"""
Auto-labeler pozycji rybki z weryfikacja uzytkownika.

Strategia:
1. Laduje klatki z danych treningowych
2. Uruchamia klasyczny detektor (background subtraction) na kazdej klatce
3. Pokazuje wynik — uzytkownik akceptuje lub poprawia kliknieciem
4. Zapisuje zweryfikowane pozycje do JSONL

Dzieki temu nie trzeba recznie oznaczac 900+ klatek,
a tylko poprawiac bledy detektora (~20% klatek).

Sterowanie:
  SPACJA   - akceptuj pozycje detektora (zielony markerek)
  Klik LPM - popraw pozycje rybki (kliknij w rybke)
  N        - brak rybki na klatce
  D / →    - pomin klatke (nie etykietuj)
  A / ←    - cofnij do poprzedniej
  S        - zapisz etykiety
  Q        - zapisz i wyjdz
"""

import os
import sys
import json
import csv
import re
import collections
from pathlib import Path

import cv2
import numpy as np

# Dodaj src/ do path zeby importowac klasyczny detektor
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.fishing_detector import FishingDetector
from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS


# Konfiguracja wyswietlania
SCALE = 3  # powiekszenie — ryba bedzie lepiej widoczna
ORIG_W = 279
ORIG_H = 247


class FishAutoLabeler:
    """Auto-etykietowanie pozycji rybki z weryfikacja uzytkownika."""

    def __init__(self, frames_dir: str, output_file: str, log_csv: str = None):
        """
        Args:
            frames_dir: folder z klatkami PNG
            output_file: plik wyjsciowy JSONL
            log_csv: opcjonalny log.csv z pozycjami rybki (do porownania)
        """
        self.frames_dir = Path(frames_dir)
        self.output_file = Path(output_file)

        # Wczytaj klatki (sortowane po nazwie)
        self.files = sorted(self.frames_dir.glob("*.png"))
        if not self.files:
            print(f"Brak klatek w {frames_dir}")
            sys.exit(1)

        # Klasyczny detektor
        self.detector = FishingDetector()

        # Wczytaj istniejace etykiety
        self.labels = {}
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        label = json.loads(line)
                        self.labels[label['file']] = label
            print(f"Wczytano {len(self.labels)} istniejacych etykiet")

        # Opcjonalnie: pozycje z log.csv (do porownania)
        self.log_positions = {}
        if log_csv and os.path.exists(log_csv):
            self._load_log_positions(log_csv)

        # Grupuj klatki po kolorze (z nazwy pliku)
        self.frame_groups = self._group_by_color()

        # Stan
        self.current_idx = 0
        self._click_pos = None

        # Znajdz pierwsza nie-etykietowana
        for i, f in enumerate(self.files):
            if f.name not in self.labels:
                self.current_idx = i
                break

        # Statystyki
        self.stats = {'accepted': 0, 'corrected': 0, 'no_fish': 0, 'skipped': 0}

        print(f"Znaleziono {len(self.files)} klatek")
        print(f"  Do oznaczenia: {len(self.files) - len(self.labels)}")
        for color, frames in self.frame_groups.items():
            print(f"  {color}: {len(frames)} klatek")

    def _load_log_positions(self, log_csv: str):
        """Laduje pozycje z log.csv do porownania."""
        with open(log_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fx = row.get('fish_x', '')
                fy = row.get('fish_y', '')
                frame_idx = row.get('frame', '')
                if fx and fy and frame_idx:
                    try:
                        self.log_positions[int(frame_idx)] = (
                            int(float(fx)), int(float(fy))
                        )
                    except ValueError:
                        pass
        if self.log_positions:
            print(f"  Log: {len(self.log_positions)} pozycji z CSV")

    def _group_by_color(self) -> dict:
        """Grupuje klatki po kolorze z nazwy pliku."""
        groups = collections.defaultdict(list)
        for f in self.files:
            m = re.search(r'_(white|red|none)\b', f.stem)
            if m:
                groups[m.group(1)].append(f)
            else:
                groups['unknown'].append(f)
        return dict(groups)

    def _parse_frame_color(self, filename: str) -> str:
        """Wyciaga kolor z nazwy pliku."""
        m = re.search(r'_(white|red|none)', filename)
        return m.group(1) if m else 'unknown'

    def _detect_fish_in_sequence(self, idx: int) -> tuple:
        """
        Detektuje rybke uzywajac background subtraction.

        Potrzebuje kilku klatek tego samego koloru do zbudowania modelu tla.
        Przetwarza klatki sekwencyjnie od pierwszej tego samego koloru.

        Returns:
            (x, y) lub None
        """
        current_file = self.files[idx]
        current_color = self._parse_frame_color(current_file.name)

        if current_color in ('none', 'unknown'):
            return None

        # Reset detektora
        self.detector.reset_tracking()

        # Znajdz klatki tego samego koloru w okolicy (do budowy tla)
        # Przetwarzaj ostatnie 20 klatek tego samego koloru
        same_color_files = []
        for i in range(max(0, idx - 25), idx + 1):
            f = self.files[i]
            c = self._parse_frame_color(f.name)
            if c == current_color:
                same_color_files.append(f)

        # Przetwarzaj sekwencyjnie (bg subtraction potrzebuje historii)
        last_pos = None
        for f in same_color_files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            pos = self.detector.find_fish_position(img, circle_color=current_color)
            if f == current_file:
                last_pos = pos

        return last_pos

    def _mouse_callback(self, event, x, y, flags, param):
        """Klikniecie = popraw pozycje rybki."""
        if event == cv2.EVENT_LBUTTONDOWN:
            ox = x // SCALE
            oy = y // SCALE
            if 0 <= ox < ORIG_W and 0 <= oy < ORIG_H:
                self._click_pos = (ox, oy)

    def _render(self, img_bgr, filename, auto_pos, log_pos=None):
        """Renderuje klatke z wizualizacja."""
        display = cv2.resize(img_bgr, (ORIG_W * SCALE, ORIG_H * SCALE),
                             interpolation=cv2.INTER_NEAREST)

        # Okrag
        cv2.circle(display,
                   (CIRCLE_CENTER_X * SCALE, CIRCLE_CENTER_Y * SCALE),
                   CIRCLE_RADIUS * SCALE,
                   (100, 100, 100), 1)

        # Kolor z nazwy pliku
        color = self._parse_frame_color(filename)
        color_map = {'white': (255, 255, 255), 'red': (0, 0, 255), 'none': (128, 128, 128)}
        border_color = color_map.get(color, (128, 128, 128))
        cv2.rectangle(display, (0, 0),
                      (display.shape[1]-1, display.shape[0]-1),
                      border_color, 3)

        # Auto-detected fish position (zielony)
        if auto_pos is not None:
            ax, ay = auto_pos
            cv2.drawMarker(display,
                           (ax * SCALE, ay * SCALE),
                           (0, 255, 0),  # zielony
                           cv2.MARKER_CROSS, 20 * SCALE // 2, 2)
            cv2.putText(display, f"Auto: ({ax},{ay})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Log position (zolty, do porownania)
        if log_pos is not None:
            lx, ly = log_pos
            cv2.drawMarker(display,
                           (lx * SCALE, ly * SCALE),
                           (0, 255, 255),  # zolty
                           cv2.MARKER_DIAMOND, 14 * SCALE // 2, 1)

        # Kliknieta pozycja (cyan)
        if self._click_pos is not None:
            cx, cy = self._click_pos
            cv2.drawMarker(display,
                           (cx * SCALE, cy * SCALE),
                           (255, 255, 0),  # cyan
                           cv2.MARKER_CROSS, 24 * SCALE // 2, 2)
            cv2.putText(display, f"Klik: ({cx},{cy})",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Istniejaca etykieta
        existing = self.labels.get(filename)
        if existing and existing.get('fish_visible'):
            ex, ey = existing['fish_x'], existing['fish_y']
            cv2.drawMarker(display,
                           (ex * SCALE, ey * SCALE),
                           (255, 0, 255),  # magenta
                           cv2.MARKER_TILTED_CROSS, 16 * SCALE // 2, 1)
            cv2.putText(display, "[SAVED]",
                        (10, ORIG_H * SCALE - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Info
        labeled = sum(1 for f in self.files if f.name in self.labels)
        cv2.putText(display,
                    f"[{self.current_idx+1}/{len(self.files)}] "
                    f"Labeled: {labeled} | {color.upper()}",
                    (10, ORIG_H * SCALE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instrukcje
        instructions = [
            "SPACE: akceptuj auto-pozycje",
            "KLIK:  popraw pozycje rybki",
            "N:     brak rybki",
            "D:     pomin",
            "A:     cofnij",
            "S:     zapisz  Q: wyjdz",
        ]
        panel_x = ORIG_W * SCALE + 10
        for i, txt in enumerate(instructions):
            cv2.putText(display, txt,
                        (panel_x, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Statystyki
        stats_y = 25 + len(instructions) * 22 + 20
        for key, val in self.stats.items():
            cv2.putText(display, f"{key}: {val}",
                        (panel_x, stats_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            stats_y += 18

        return display

    def _save_label(self, filename: str, fish_pos: tuple, source: str):
        """Zapisuje etykiete."""
        color = self._parse_frame_color(filename)
        state_map = {'white': 'WHITE', 'red': 'RED', 'none': 'INACTIVE'}
        state = state_map.get(color, 'INACTIVE')

        label = {
            'file': filename,
            'state': state,
            'fish_x': fish_pos[0] if fish_pos else None,
            'fish_y': fish_pos[1] if fish_pos else None,
            'fish_visible': fish_pos is not None,
            'source': source,
        }
        self.labels[filename] = label

    def _save_to_file(self):
        """Zapisuje wszystkie etykiety do JSONL."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for fname in sorted(self.labels.keys()):
                f.write(json.dumps(self.labels[fname], ensure_ascii=False) + '\n')

        total = len(self.labels)
        with_pos = sum(1 for l in self.labels.values() if l.get('fish_visible'))
        print(f"\nZapisano {total} etykiet ({with_pos} z pozycja rybki)")
        print(f"  Plik: {self.output_file}")

    def run(self):
        """Glowna petla GUI."""
        win_name = "Fish Auto-Labeler"
        win_w = ORIG_W * SCALE + 260
        win_h = ORIG_H * SCALE

        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_name, self._mouse_callback)

        print("\n=== Fish Auto-Labeler ===")
        print("Zielony krzyżyk = auto-detekcja. Akceptuj SPACJA lub popraw kliknieciem.\n")

        while True:
            if self.current_idx >= len(self.files):
                print("\nWszystkie klatki przetworzone!")
                self._save_to_file()
                break

            current_file = self.files[self.current_idx]
            filename = current_file.name

            # Wczytaj klatke
            img = cv2.imread(str(current_file))
            if img is None:
                self.current_idx += 1
                continue

            # Auto-detekcja rybki
            auto_pos = self._detect_fish_in_sequence(self.current_idx)

            # Pozycja z loga (do porownania)
            frame_idx_match = re.search(r'(\d+)', filename)
            log_pos = None
            if frame_idx_match:
                fidx = int(frame_idx_match.group(1))
                log_pos = self.log_positions.get(fidx)

            # Renderuj
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            rendered = self._render(img, filename, auto_pos, log_pos)
            rh, rw = rendered.shape[:2]
            canvas[:min(rh, win_h), :min(rw, win_w)] = rendered[:min(rh, win_h), :min(rw, win_w)]

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                self._save_to_file()
                break

            elif key == ord('s'):
                self._save_to_file()

            elif key == 32:  # SPACE — akceptuj auto
                if auto_pos is not None:
                    self._save_label(filename, auto_pos, 'auto_accepted')
                    self.stats['accepted'] += 1
                    print(f"  OK {filename}: ({auto_pos[0]},{auto_pos[1]})")
                else:
                    self._save_label(filename, None, 'auto_no_fish')
                    self.stats['no_fish'] += 1
                    print(f"  -- {filename}: brak rybki (auto)")
                self._click_pos = None
                self.current_idx += 1

            elif self._click_pos is not None and key == 13:  # ENTER po kliknieciu
                pos = self._click_pos
                self._save_label(filename, pos, 'manual_corrected')
                self.stats['corrected'] += 1
                print(f"  CC {filename}: ({pos[0]},{pos[1]}) [poprawione]")
                self._click_pos = None
                self.current_idx += 1

            elif key == ord('n'):  # brak rybki
                self._save_label(filename, None, 'manual_no_fish')
                self.stats['no_fish'] += 1
                print(f"  -- {filename}: brak rybki")
                self._click_pos = None
                self.current_idx += 1

            elif key == ord('d') or key == 83:  # pomin
                self.stats['skipped'] += 1
                self._click_pos = None
                self.current_idx += 1

            elif key == ord('a') or key == 81:  # cofnij
                self.current_idx = max(0, self.current_idx - 1)
                self._click_pos = None

            # Obsluga klikniecia — pokaz od razu aktualizacje
            # (klikniecie akceptowane ENTER-em)

        cv2.destroyAllWindows()
        print(f"\nStatystyki: {self.stats}")
        print("Gotowe!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-labeler pozycji rybki")
    parser.add_argument(
        "--frames", required=True,
        help="Folder z klatkami PNG"
    )
    parser.add_argument(
        "--output", default=None,
        help="Plik wyjsciowy JSONL (domyslnie: cnn/data/fish_labels.jsonl)"
    )
    parser.add_argument(
        "--log", default=None,
        help="Opcjonalny log.csv z pozycjami (do porownania)"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = str(SCRIPT_DIR / "data" / "fish_labels.jsonl")

    tool = FishAutoLabeler(args.frames, args.output, args.log)
    tool.run()
