"""
Kolekcjoner ksztaltu rybki — wycinanie i katalogowanie rybek z klatek.

Tryby pracy:
1. ZBIERANIE (--collect): Uruchom z botem — bot zapisuje klatki do folderu,
   ten skrypt je przetwarza i wycina rybki
2. PRZEGLADANIE (--review): Przegladaj wyciety snippety rybek,
   akceptuj/odrzucaj kliknieciem — RYBKA/NIE RYBKA
3. EKSTRAKCJA (--extract): Automatycznie wytnij rybki z istniejacych klatek
   uzywajac klasycznego detektora

Wynik: folder z wycinankami rybek + plik JSONL z etykietami pozycji.
Kazdy snippet to 64x64 wycinek wokol rybki — CNN nauczy sie tego ksztaltu.

Sterowanie (tryb review):
  Y / SPACJA : to jest rybka (akceptuj)
  N          : to NIE jest rybka (odrzuc)
  ← / →      : cofnij / dalej
  Q          : zapisz i wyjdz
"""

import os
import sys
import json
import re
import csv
import collections
from pathlib import Path

import cv2
import numpy as np

# Dodaj src/ do path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.fishing_detector import FishingDetector
from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS

# Rozmiar wycinanego snippetu rybki (w pikselach z oryginalnej klatki)
SNIPPET_SIZE = 64
SNIPPET_HALF = SNIPPET_SIZE // 2

# Rozmiar wyswietlania (snippet powiekszony)
DISPLAY_SCALE = 6

# Rozmiary oryginalnej klatki
ORIG_W = 279
ORIG_H = 247


def extract_fish_snippet(frame_bgr: np.ndarray, fish_x: int, fish_y: int,
                         size: int = SNIPPET_SIZE) -> np.ndarray:
    """
    Wycina kwadratowy fragment wokol rybki z klatki.

    Args:
        frame_bgr: pelna klatka (279x247 BGR)
        fish_x, fish_y: pozycja rybki
        size: rozmiar snippetu (domyslnie 64x64)

    Returns:
        Wycinek BGR (size x size), z paddingiem jesli przy krawedzi
    """
    half = size // 2
    h, w = frame_bgr.shape[:2]

    # Oblicz region z clampingiem
    x1 = max(0, fish_x - half)
    y1 = max(0, fish_y - half)
    x2 = min(w, fish_x + half)
    y2 = min(h, fish_y + half)

    # Wytnij
    crop = frame_bgr[y1:y2, x1:x2]

    # Jesli przy krawedzi — dodaj padding (czarny)
    if crop.shape[0] != size or crop.shape[1] != size:
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        # Oblicz offset paddingu
        px = half - (fish_x - x1)
        py = half - (fish_y - y1)
        padded[py:py+crop.shape[0], px:px+crop.shape[1]] = crop
        return padded

    return crop.copy()


def extract_from_frames(
    frames_dir: str,
    output_dir: str,
    log_csv: str = None,
):
    """
    Automatycznie wycinaj rybki z istniejacych klatek.

    Uzywa klasycznego detektora (background subtraction) plus
    opcjonalnie logow z pozycjami.

    Args:
        frames_dir: folder z klatkami PNG
        output_dir: folder na snippety rybek
        log_csv: opcjonalny log.csv z pozycjami
    """
    frames_path = Path(frames_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snippets_dir = output_path / "snippets"
    snippets_dir.mkdir(exist_ok=True)
    labels_file = output_path / "fish_snippets.jsonl"

    # Zaladuj logi z pozycjami
    log_positions = {}
    if log_csv and os.path.exists(log_csv):
        with open(log_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fx = row.get('fish_x', '')
                fy = row.get('fish_y', '')
                fidx = row.get('frame', '')
                if fx and fy and fidx:
                    try:
                        log_positions[int(fidx)] = (int(float(fx)), int(float(fy)))
                    except ValueError:
                        pass
        print(f"  Log: {len(log_positions)} pozycji z CSV")

    # Zbierz klatki
    files = sorted(frames_path.glob("*.png"))
    if not files:
        print(f"Brak klatek w {frames_dir}")
        return

    print(f"Przetwarzam {len(files)} klatek z {frames_dir}...")

    # Grupuj po kolorze i przetwarzaj sekwencyjnie
    detector = FishingDetector()
    labels = []
    extracted = 0

    # Sortuj klatki po numerze
    def frame_sort_key(f):
        m = re.search(r'(\d+)', f.stem)
        return int(m.group(1)) if m else 0

    files = sorted(files, key=frame_sort_key)

    # Przetwarzaj sekwencyjnie (bg subtraction potrzebuje historii)
    prev_color = None
    for fi, frame_file in enumerate(files):
        # Kolor z nazwy pliku
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

        # Szukaj rybki
        fish_pos = detector.find_fish_position(img, circle_color=color)

        # Fallback: pozycja z loga
        if fish_pos is None:
            m_idx = re.search(r'(\d+)', frame_file.stem)
            if m_idx:
                fidx = int(m_idx.group(1))
                if fidx in log_positions:
                    fish_pos = log_positions[fidx]

        if fish_pos is None:
            continue

        fx, fy = fish_pos

        # Walidacja: czy wewnatrz okregu?
        import math
        dist = math.sqrt((fx - CIRCLE_CENTER_X)**2 + (fy - CIRCLE_CENTER_Y)**2)
        if dist > CIRCLE_RADIUS + 10:
            continue

        # Wytnij snippet
        snippet = extract_fish_snippet(img, fx, fy)

        # Zapisz
        snippet_name = f"fish_{frame_file.stem}.png"
        cv2.imwrite(str(snippets_dir / snippet_name), snippet)

        label = {
            'snippet': snippet_name,
            'source_frame': frame_file.name,
            'fish_x': fx,
            'fish_y': fy,
            'color': color,
            'is_fish': None,  # do weryfikacji
        }
        labels.append(label)
        extracted += 1

    # Zapisz etykiety
    with open(labels_file, 'w', encoding='utf-8') as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')

    print(f"\nWycieto {extracted} snippetow rybek")
    print(f"  Snippety: {snippets_dir}")
    print(f"  Etykiety: {labels_file}")

    return str(snippets_dir), str(labels_file)


def review_snippets(
    snippets_dir: str,
    labels_file: str,
):
    """
    Interaktywne przegladanie snippetow — uzytkownik decyduje: rybka/nie rybka.

    Pokazuje snippet powiekszony, uzytkownik naciska Y/N.
    Wynik zapisywany do JSONL.
    """
    snippets_path = Path(snippets_dir)
    labels_path = Path(labels_file)

    # Wczytaj etykiety
    labels = []
    if labels_path.exists():
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(json.loads(line))

    if not labels:
        print("Brak snippetow do przegladu")
        return

    # Znajdz pierwsza nie-sprawdzona
    current = 0
    for i, l in enumerate(labels):
        if l.get('is_fish') is None:
            current = i
            break

    win_name = "Fish Review — Y=rybka N=nie S=zapisz Q=wyjdz"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    stats = {'yes': 0, 'no': 0, 'remaining': 0}
    for l in labels:
        if l.get('is_fish') is True:
            stats['yes'] += 1
        elif l.get('is_fish') is False:
            stats['no'] += 1
        else:
            stats['remaining'] += 1

    print(f"\n=== Przeglad snippetow rybek ===")
    print(f"Razem: {len(labels)} | Zaakceptowane: {stats['yes']} | Odrzucone: {stats['no']} | Pozostale: {stats['remaining']}")
    print("Y/SPACJA = rybka | N = nie rybka | A/D = prev/next | Q = wyjdz\n")

    while True:
        if current >= len(labels):
            current = len(labels) - 1

        label = labels[current]
        snippet_path = snippets_path / label['snippet']

        if not snippet_path.exists():
            current += 1
            continue

        snippet = cv2.imread(str(snippet_path))
        if snippet is None:
            current += 1
            continue

        # Powiekszenie
        display = cv2.resize(snippet,
                             (SNIPPET_SIZE * DISPLAY_SCALE, SNIPPET_SIZE * DISPLAY_SCALE),
                             interpolation=cv2.INTER_NEAREST)

        # Ramka statusu
        status = label.get('is_fish')
        if status is True:
            border_color = (0, 255, 0)  # zielony = rybka
            status_text = "RYBKA"
        elif status is False:
            border_color = (0, 0, 255)  # czerwony = nie rybka
            status_text = "NIE RYBKA"
        else:
            border_color = (128, 128, 128)  # szary = nie sprawdzone
            status_text = "???"

        cv2.rectangle(display, (0, 0),
                      (display.shape[1]-1, display.shape[0]-1),
                      border_color, 4)

        # Info
        color = label.get('color', '?')
        pos = f"({label.get('fish_x', '?')},{label.get('fish_y', '?')})"
        cv2.putText(display, f"[{current+1}/{len(labels)}] {status_text}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)
        cv2.putText(display, f"{color.upper()} {pos}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, label['snippet'],
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('y') or key == 32:  # Y lub SPACJA
            labels[current]['is_fish'] = True
            print(f"  ✓ [{current+1}] RYBKA: {label['snippet']}")
            current += 1
        elif key == ord('n'):
            labels[current]['is_fish'] = False
            print(f"  ✗ [{current+1}] NIE: {label['snippet']}")
            current += 1
        elif key == ord('d') or key == 83:  # D lub →
            current = min(current + 1, len(labels) - 1)
        elif key == ord('a') or key == 81:  # A lub ←
            current = max(current - 1, 0)
        elif key == ord('s'):
            # Zapisz od razu
            with open(labels_path, 'w', encoding='utf-8') as f:
                for l in labels:
                    f.write(json.dumps(l, ensure_ascii=False) + '\n')
            print(f"  Zapisano {len(labels)} etykiet")

    # Zapisz na koniec
    with open(labels_path, 'w', encoding='utf-8') as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')

    cv2.destroyAllWindows()

    # Podsumowanie
    yes = sum(1 for l in labels if l.get('is_fish') is True)
    no = sum(1 for l in labels if l.get('is_fish') is False)
    unseen = sum(1 for l in labels if l.get('is_fish') is None)
    print(f"\nPodsumowanie: RYBKA={yes}, NIE={no}, Nieoznaczone={unseen}")
    print(f"Zapisano: {labels_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kolekcjoner ksztaltu rybki")
    subparsers = parser.add_subparsers(dest='mode', help='Tryb pracy')

    # Ekstrakcja z istniejacych klatek
    p_extract = subparsers.add_parser('extract', help='Wytnij rybki z klatek')
    p_extract.add_argument('--frames', required=True, help='Folder z klatkami')
    p_extract.add_argument('--output', default='cnn/data/fish_collection',
                           help='Folder wyjsciowy')
    p_extract.add_argument('--log', default=None, help='Log CSV z pozycjami')

    # Przegladanie i weryfikacja
    p_review = subparsers.add_parser('review', help='Przegladaj snippety rybek')
    p_review.add_argument('--snippets', default='cnn/data/fish_collection/snippets',
                          help='Folder ze snippetami')
    p_review.add_argument('--labels', default='cnn/data/fish_collection/fish_snippets.jsonl',
                          help='Plik z etykietami')

    args = parser.parse_args()

    if args.mode == 'extract':
        extract_from_frames(args.frames, args.output, args.log)
    elif args.mode == 'review':
        review_snippets(args.snippets, args.labels)
    else:
        # Domyslnie: wytnij z WSZYSTKICH testow
        print("=== Ekstrakcja rybek z WSZYSTKICH testow ===\n")
        base = Path(__file__).resolve().parent.parent.parent.parent

        sources = [
            (base / 'test10_clean' / 'raw', base / 'test10_clean' / 'log.csv'),
            (base / 'test8a_tracking' / 'frames', base / 'tests' / 'test8a_tracking' / 'log.csv'),
            (base / 'test8b_miss' / 'frames', base / 'tests' / 'test8b_miss' / 'log.csv'),
            (base / 'test8c_hit' / 'frames', base / 'tests' / 'test8c_hit' / 'log.csv'),
            (base / 'test9_long' / 'frames', base / 'test9_long' / 'log.csv'),
        ]

        output_dir = str(SCRIPT_DIR / 'data' / 'fish_collection')
        all_snippets = 0

        for frames_dir, log_csv in sources:
            if frames_dir.exists():
                print(f"\n--- {frames_dir.name} ---")
                result = extract_from_frames(
                    str(frames_dir), output_dir,
                    str(log_csv) if log_csv.exists() else None
                )

        # Na koniec podsumowanie
        snippets_dir = Path(output_dir) / 'snippets'
        if snippets_dir.exists():
            count = len(list(snippets_dir.glob('*.png')))
            print(f"\n{'='*50}")
            print(f"RAZEM: {count} snippetow rybek")
            print(f"Folder: {snippets_dir}")
            print(f"\nNastepny krok: przejrzyj snippety:")
            print(f"  python -m cnn.fish_collector review")
