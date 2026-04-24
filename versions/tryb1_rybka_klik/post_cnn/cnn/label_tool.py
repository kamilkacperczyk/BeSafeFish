"""
Narzedzie do recznego etykietowania klatek minigry lowienia.

GUI w OpenCV:
- Wyswietla klatke (powiekszenie 2×)
- Klawisze 1-5: wybor stanu (INACTIVE/WHITE/RED/HIT_TEXT/MISS_TEXT)
- Klikniecie LPM: pozycja rybki
- N: brak rybki (fish_visible=False)
- ←/→ lub A/D: poprzednia/nastepna klatka
- Ctrl+Z: cofnij ostatnia etykiete
- S: zapisz etykiety
- Q: zapisz i wyjdz

Etykiety zapisywane do JSONL (append-mode).
"""

import os
import sys
import json
import glob
from pathlib import Path

import cv2
import numpy as np


# Konfiguracja
SCALE = 2                     # powiekszenie wyswietlanej klatki
ORIG_W = 279                  # oryginalna szerokosc fishing box
ORIG_H = 247                  # oryginalna wysokosc fishing box
CIRCLE_CX = 140               # srodek okregu X
CIRCLE_CY = 137               # srodek okregu Y
CIRCLE_R = 64                 # promien okregu

STATE_NAMES = ['INACTIVE', 'WHITE', 'RED', 'HIT_TEXT', 'MISS_TEXT']
STATE_KEYS = {
    ord('1'): 'INACTIVE',
    ord('2'): 'WHITE',
    ord('3'): 'RED',
    ord('4'): 'HIT_TEXT',
    ord('5'): 'MISS_TEXT',
}
STATE_COLORS = {
    'INACTIVE': (128, 128, 128),
    'WHITE': (255, 255, 255),
    'RED': (0, 0, 255),
    'HIT_TEXT': (0, 255, 255),
    'MISS_TEXT': (255, 0, 255),
}


class LabelTool:
    """Interaktywne narzedzie do etykietowania klatek."""

    def __init__(self, frames_dir: str, output_file: str):
        self.frames_dir = Path(frames_dir)
        self.output_file = Path(output_file)

        # Wczytaj listę klatek
        patterns = ['*.png', '*.jpg', '*.bmp']
        self.files = []
        for p in patterns:
            self.files.extend(sorted(self.frames_dir.glob(p)))

        if not self.files:
            print(f"[LabelTool] Brak klatek w {frames_dir}")
            sys.exit(1)

        print(f"[LabelTool] Znaleziono {len(self.files)} klatek w {frames_dir}")

        # Wczytaj istniejące etykiety (jeśli plik istnieje)
        self.labels = {}  # filename → label dict
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        label = json.loads(line)
                        self.labels[label['file']] = label
            print(f"[LabelTool] Wczytano {len(self.labels)} istniejacych etykiet")

        # Stan edycji
        self.current_idx = 0
        self.current_state = None     # wybrany stan
        self.current_fish_pos = None  # (x, y) lub None
        self.current_fish_visible = None

        # Znajdź pierwszą nie-etykietowaną klatkę
        for i, f in enumerate(self.files):
            if f.name not in self.labels:
                self.current_idx = i
                break

        # Zmienne do kliknięcia
        self._click_pos = None

    def _mouse_callback(self, event, x, y, flags, param):
        """Callback myszy — kliknięcie = pozycja rybki."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Przelicz z powiekszonego → oryginalny
            orig_x = x // SCALE
            orig_y = y // SCALE
            if 0 <= orig_x < ORIG_W and 0 <= orig_y < ORIG_H:
                self._click_pos = (orig_x, orig_y)

    def _render(self, img_bgr: np.ndarray, filename: str) -> np.ndarray:
        """Renderuje klatke z overleyem informacji."""
        # Powiekszenie
        display = cv2.resize(img_bgr, (ORIG_W * SCALE, ORIG_H * SCALE),
                             interpolation=cv2.INTER_NEAREST)

        # Narysuj okrąg (punkt odniesienia)
        cv2.circle(display,
                   (CIRCLE_CX * SCALE, CIRCLE_CY * SCALE),
                   CIRCLE_R * SCALE,
                   (100, 100, 100), 1)

        # Istniejaca etykieta
        existing = self.labels.get(filename)
        if existing:
            state = existing['state']
            color = STATE_COLORS.get(state, (200, 200, 200))
            cv2.putText(display, f"[SAVED] {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if existing.get('fish_visible') and existing.get('fish_x') is not None:
                fx = existing['fish_x'] * SCALE
                fy = existing['fish_y'] * SCALE
                cv2.drawMarker(display, (fx, fy), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, f"({existing['fish_x']},{existing['fish_y']})",
                            (fx + 15, fy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Bieżąca edycja
        if self.current_state:
            color = STATE_COLORS.get(self.current_state, (200, 200, 200))
            cv2.putText(display, f"State: {self.current_state}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if self.current_fish_pos:
            fx, fy = self.current_fish_pos
            cv2.drawMarker(display, (fx * SCALE, fy * SCALE), (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 2)
            cv2.putText(display, f"Fish: ({fx},{fy})", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.current_fish_visible is False:
            cv2.putText(display, "Fish: NONE", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Nawigacja
        labeled = sum(1 for f in self.files if f.name in self.labels)
        cv2.putText(display,
                    f"[{self.current_idx+1}/{len(self.files)}] "
                    f"Labeled: {labeled}/{len(self.files)}",
                    (10, ORIG_H * SCALE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instrukcje (prawa strona)
        instructions = [
            "1: INACTIVE  2: WHITE",
            "3: RED  4: HIT_TEXT",
            "5: MISS_TEXT",
            "Click: fish pos",
            "N: no fish",
            "ENTER: save label",
            "A/D: prev/next",
            "S: save file  Q: quit",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(display, text,
                        (ORIG_W * SCALE + 10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return display

    def _save_current_label(self):
        """Zapisuje biezaca etykiete."""
        if self.current_state is None:
            print("  ! Wybierz stan (1-5) przed zapisaniem")
            return False

        filename = self.files[self.current_idx].name

        label = {
            'file': filename,
            'state': self.current_state,
            'fish_x': self.current_fish_pos[0] if self.current_fish_pos else None,
            'fish_y': self.current_fish_pos[1] if self.current_fish_pos else None,
            'fish_visible': bool(self.current_fish_pos) if self.current_fish_visible is None
                            else self.current_fish_visible,
        }

        # Jesli fish_visible=False, wyczysc pozycje
        if not label['fish_visible']:
            label['fish_x'] = None
            label['fish_y'] = None

        self.labels[filename] = label
        print(f"  ✓ {filename}: {label['state']}, "
              f"fish={'(' + str(label['fish_x']) + ',' + str(label['fish_y']) + ')' if label['fish_visible'] else 'NONE'}")

        # Reset
        self.current_state = None
        self.current_fish_pos = None
        self.current_fish_visible = None

        return True

    def _save_to_file(self):
        """Zapisuje wszystkie etykiety do pliku JSONL."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for filename in sorted(self.labels.keys()):
                f.write(json.dumps(self.labels[filename], ensure_ascii=False) + '\n')
        print(f"[LabelTool] Zapisano {len(self.labels)} etykiet do {self.output_file}")

    def run(self):
        """Glowna petla GUI."""
        window_name = "Kosa Label Tool"
        # Szerokosc = obraz powiększony + panel instrukcji
        win_w = ORIG_W * SCALE + 220
        win_h = ORIG_H * SCALE
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n[LabelTool] Sterowanie:")
        print("  1-5: wybor stanu | klik: pozycja rybki | N: brak rybki")
        print("  ENTER: zapisz etykiete | A/D: prev/next | S: save | Q: quit")
        print()

        while True:
            # Wczytaj biezaca klatke
            img_path = self.files[self.current_idx]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  ! Nie mozna wczytac: {img_path}")
                self.current_idx = (self.current_idx + 1) % len(self.files)
                continue

            # Obsluz klikniecie myszy
            if self._click_pos is not None:
                self.current_fish_pos = self._click_pos
                self.current_fish_visible = True
                self._click_pos = None

            # Renderuj
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            rendered = self._render(img_bgr, img_path.name)
            rh, rw = rendered.shape[:2]
            canvas[:rh, :rw] = rendered

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                self._save_to_file()
                break

            elif key == ord('s'):
                self._save_to_file()

            elif key in STATE_KEYS:
                self.current_state = STATE_KEYS[key]
                print(f"  Stan: {self.current_state}")

            elif key == ord('n'):
                self.current_fish_pos = None
                self.current_fish_visible = False
                print("  Rybka: BRAK")

            elif key == 13:  # ENTER
                if self._save_current_label():
                    # Przejdz do nastepnej
                    self.current_idx = min(self.current_idx + 1, len(self.files) - 1)

            elif key == ord('d') or key == 83:  # D lub →
                self.current_idx = min(self.current_idx + 1, len(self.files) - 1)
                self.current_state = None
                self.current_fish_pos = None
                self.current_fish_visible = None

            elif key == ord('a') or key == 81:  # A lub ←
                self.current_idx = max(self.current_idx - 1, 0)
                self.current_state = None
                self.current_fish_pos = None
                self.current_fish_visible = None

            elif key == 26:  # Ctrl+Z
                filename = self.files[self.current_idx].name
                if filename in self.labels:
                    del self.labels[filename]
                    print(f"  ↩ Cofnieto etykiete: {filename}")
                self.current_state = None
                self.current_fish_pos = None
                self.current_fish_visible = None

        cv2.destroyAllWindows()
        print("[LabelTool] Zakonczono.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Narzedzie do etykietowania klatek")
    parser.add_argument(
        "--frames", default="test10_clean/raw",
        help="Folder z klatkami PNG (domyslnie: test10_clean/raw)"
    )
    parser.add_argument(
        "--output", default="cnn/data/labels.jsonl",
        help="Plik wyjsciowy JSONL (domyslnie: cnn/data/labels.jsonl)"
    )
    args = parser.parse_args()

    tool = LabelTool(args.frames, args.output)
    tool.run()
