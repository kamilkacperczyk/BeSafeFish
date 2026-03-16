"""
Przygotowanie danych treningowych z istniejacych testow.

Skanuje foldery testowe, kopiuje klatki do centralnego katalogu,
wyciaga etykiety z nazw plikow + opcjonalnie pozycje z logow CSV.

Uruchomienie:
    python -m cnn.prepare_data
"""

import csv
import json
import os
import re
import shutil
from pathlib import Path


# ---- Konfiguracja zrodel danych ----

# Sciezki wzgledem roota projektu (C:\...\Kosa)
# (log_csv, frames_dir, frame_name_digits) — digits: ile cyfr w nazwie
DATA_SOURCES = [
    {
        'name': 'test10',
        'log': 'test10_clean/log.csv',
        'frames': 'test10_clean/raw',
        'digits': 5,  # frame_00003_white.png
    },
    {
        'name': 'test8a',
        'log': 'tests/test8a_tracking/log.csv',
        'frames': 'test8a_tracking/frames',
        'digits': 4,  # frame_0001_red.png
    },
    {
        'name': 'test8b',
        'log': 'tests/test8b_miss/log.csv',
        'frames': 'test8b_miss/frames',
        'digits': 4,
    },
    {
        'name': 'test8c',
        'log': 'tests/test8c_hit/log.csv',
        'frames': 'test8c_hit/frames',
        'digits': 4,
    },
    {
        'name': 'test9',
        'log': 'test9_long/log.csv',
        'frames': 'test9_long/frames',
        'digits': 5,
    },
]

# Mapowanie kolorow w nazwach plikow → stany CNN
COLOR_TO_STATE = {
    'white': 'WHITE',
    'red': 'RED',
    'none': 'INACTIVE',
    'inactive': 'INACTIVE',
}

# Rozmiary oryginalnego fishing box
ORIG_W = 279
ORIG_H = 247


def parse_frame_filename(filename: str):
    """
    Parsuje nazwe pliku klatki.

    Formaty:
        frame_00003_white.png  → idx=3,  color='white'
        frame_0001_red.png     → idx=1,  color='red'
        frame_0001_none.png    → idx=1,  color='none'

    Returns:
        (frame_idx, color) lub None jesli nie pasuje
    """
    m = re.match(r'frame_(\d+)_(\w+)\.png', filename)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def load_log_positions(log_path: str) -> dict:
    """
    Laduje pozycje rybek z log.csv.

    Returns:
        dict: {frame_idx: (fish_x, fish_y)} — tylko klatki z pozycja
    """
    positions = {}
    if not os.path.exists(log_path):
        return positions

    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return positions

        for row in reader:
            frame_idx = int(row.get('frame', 0))
            fish_x = row.get('fish_x', '')
            fish_y = row.get('fish_y', '')

            if fish_x and fish_y:
                try:
                    fx = int(float(fish_x))
                    fy = int(float(fish_y))
                    if 0 <= fx < ORIG_W and 0 <= fy < ORIG_H:
                        positions[frame_idx] = (fx, fy)
                except (ValueError, TypeError):
                    pass

    return positions


def collect_frames_from_source(
    source: dict,
    project_root: Path,
    output_frames_dir: Path,
    labels: list,
) -> int:
    """
    Zbiera klatki z jednego zrodla testowego.

    Args:
        source: dict z konfiguracja zrodla
        project_root: sciezka do roota projektu
        output_frames_dir: docelowy folder na skopiowane klatki
        labels: lista etykiet do rozszerzenia

    Returns:
        Liczba dodanych klatek
    """
    name = source['name']
    frames_dir = project_root / source['frames']
    log_path = project_root / source['log']

    if not frames_dir.exists():
        print(f"  [{name}] SKIP — folder {frames_dir} nie istnieje")
        return 0

    # Zaladuj pozycje z loga
    positions = load_log_positions(str(log_path))
    if positions:
        print(f"  [{name}] Log: {len(positions)} klatek z pozycja rybki")

    # Skanuj klatki
    added = 0
    for frame_file in sorted(frames_dir.glob("frame_*.png")):
        parsed = parse_frame_filename(frame_file.name)
        if parsed is None:
            continue

        frame_idx, color = parsed
        state = COLOR_TO_STATE.get(color)
        if state is None:
            print(f"  [{name}] Nieznany kolor: {color} w {frame_file.name}")
            continue

        # Unikalna nazwa: {source}_{oryginal}
        unique_name = f"{name}_{frame_file.name}"

        # Kopiuj klatke
        dst = output_frames_dir / unique_name
        if not dst.exists():
            shutil.copy2(frame_file, dst)

        # Pozycja rybki z loga
        fish_visible = False
        fish_x = None
        fish_y = None
        if frame_idx in positions:
            fish_x, fish_y = positions[frame_idx]
            fish_visible = True

        label = {
            'file': unique_name,
            'state': state,
            'fish_x': fish_x,
            'fish_y': fish_y,
            'fish_visible': fish_visible,
            'source': name,
        }
        labels.append(label)
        added += 1

    print(f"  [{name}] {added} klatek dodanych")
    return added


def prepare_data(
    project_root: str = None,
    output_dir: str = None,
):
    """
    Glowna funkcja przygotowania danych.

    Zbiera klatki z wszystkich testow, kopiuje do jednego folderu,
    generuje etykiety JSONL.
    """
    if project_root is None:
        # Auto-detect: versions/post_cnn/cnn/ → 3 levels up
        project_root = Path(__file__).resolve().parent.parent.parent.parent

    project_root = Path(project_root)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    labels_file = output_dir / "labels.jsonl"

    print("=" * 60)
    print("[Prepare Data] Zbieranie danych treningowych")
    print(f"  Zrodlo:  {project_root}")
    print(f"  Cel:     {output_dir}")
    print("=" * 60)

    labels = []

    # Zbierz z kazdego zrodla
    total = 0
    for source in DATA_SOURCES:
        count = collect_frames_from_source(
            source, project_root, frames_dir, labels
        )
        total += count

    # Podsumowanie
    print(f"\n[Prepare Data] RAZEM: {total} klatek")

    # Rozklad klas
    counts = {}
    visible_count = 0
    for label in labels:
        s = label['state']
        counts[s] = counts.get(s, 0) + 1
        if label['fish_visible']:
            visible_count += 1

    print("\nRozklad klas:")
    for state in ['INACTIVE', 'WHITE', 'RED', 'HIT_TEXT', 'MISS_TEXT']:
        c = counts.get(state, 0)
        pct = c / max(total, 1) * 100
        print(f"  {state:12s}: {c:4d} ({pct:5.1f}%)")

    print(f"\nKlatki z pozycja rybki: {visible_count} ({visible_count/max(total,1)*100:.1f}%)")

    # Sprawdz balans — ostrzezenia
    if total < 100:
        print("\n⚠ UWAGA: Bardzo malo danych (<100). Trening moze nie byc skuteczny.")
    elif total < 500:
        print("\n⚠ UWAGA: Mala ilosc danych (<500). Rozważ zebranie wiecej klatek.")

    # Sprawdz czy mamy brakujace klasy
    missing = [s for s in ['INACTIVE', 'WHITE', 'RED'] if counts.get(s, 0) == 0]
    if missing:
        print(f"\n⚠ BRAK DANYCH dla klas: {', '.join(missing)}")
        print("  Model moze nie naucyc sie tych stanow!")

    # Zapisz etykiety
    with open(labels_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + '\n')

    print(f"\nZapisano: {labels_file}")
    print("=" * 60)

    return str(labels_file), str(frames_dir), total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Przygotowanie danych treningowych")
    parser.add_argument(
        "--root", default=None,
        help="Sciezka do roota projektu (auto-detect jesli pominiety)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Folder wyjsciowy na dane (domyslnie cnn/data/)"
    )
    args = parser.parse_args()

    prepare_data(project_root=args.root, output_dir=args.output)
