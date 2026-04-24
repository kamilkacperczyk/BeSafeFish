"""
Generator pseudo-etykiet z istniejacych logow testowych.

Parsuje pliki log.csv z testow (test8a, test8b, test8c, test9, test10)
i generuje etykiety w formacie JSONL kompatybilne z FishingDataset.

Pseudo-labels maja te same bledy co klasyczny detektor, dlatego:
1. Stan (WHITE/RED/INACTIVE) — dokladny (>95% poprawny) → uzywamy
2. Pozycja rybki — czesto poprawna, ale bywa bledna → do weryfikacji
3. Klatki z missed detection → oznaczone jako "brak rybki" (conf=0)
"""

import csv
import json
import os
from pathlib import Path


def parse_test_log(
    log_path: str,
    frames_dir: str,
    output_labels: list,
    frame_extension: str = ".png",
):
    """
    Parsuje log.csv z testow i generuje pseudo-labels.

    Oczekiwany format CSV:
        frame_idx, timestamp, phase, fish_x, fish_y, action, ...

    Args:
        log_path: sciezka do log.csv
        frames_dir: folder z klatkami
        output_labels: lista do ktorej dodajemy etykiety
        frame_extension: rozszerzenie plikow klatek
    """
    if not os.path.exists(log_path):
        print(f"  [skip] {log_path} — nie istnieje")
        return

    frames_path = Path(frames_dir)
    added = 0

    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"  [skip] {log_path} — pusty lub bez naglowka")
            return

        for row in reader:
            # Znajdz plik klatki
            frame_idx = row.get('frame_idx', row.get('frame', ''))
            if not frame_idx:
                continue

            # Mozliwe nazwy plikow
            candidates = [
                f"frame_{int(frame_idx):04d}{frame_extension}",
                f"frame_{frame_idx}{frame_extension}",
                f"{frame_idx}{frame_extension}",
            ]

            frame_file = None
            for c in candidates:
                if (frames_path / c).exists():
                    frame_file = c
                    break

            if frame_file is None:
                continue

            # Mapuj fazę na stan
            phase = row.get('phase', row.get('circle_color', row.get('color', '')))
            phase = phase.strip().lower() if phase else ''

            state_map = {
                'white': 'WHITE',
                'red': 'RED',
                'none': 'INACTIVE',
                'inactive': 'INACTIVE',
            }
            state = state_map.get(phase)
            if state is None:
                continue

            # Pozycja rybki
            fish_x = row.get('fish_x', row.get('fx', ''))
            fish_y = row.get('fish_y', row.get('fy', ''))

            fish_visible = False
            fx = fy = None
            if fish_x and fish_y:
                try:
                    fx = int(float(fish_x))
                    fy = int(float(fish_y))
                    if 0 <= fx < 279 and 0 <= fy < 247:
                        fish_visible = True
                except (ValueError, TypeError):
                    pass

            label = {
                'file': frame_file,
                'state': state,
                'fish_x': fx if fish_visible else None,
                'fish_y': fy if fish_visible else None,
                'fish_visible': fish_visible,
                'source': 'pseudo',
            }

            output_labels.append(label)
            added += 1

    print(f"  {log_path}: {added} pseudo-labels")


def generate_all_pseudo_labels(output_file: str = "cnn/data/pseudo_labels.jsonl"):
    """
    Generuje pseudo-labels z wszystkich dostepnych logow.

    Skanuje foldery testowe i parsuje logi.
    """
    base = Path(".")
    labels = []

    # Znane lokalizacje logow
    test_dirs = [
        ("tests/test8a_tracking/log.csv", "test8a_tracking/frames"),
        ("tests/test8b_miss/log.csv",     "test8b_miss/frames"),
        ("tests/test8c_hit/log.csv",      "test8c_hit/frames"),
        ("test9_long/log.csv",            "test9_long/frames"),
        ("test10_clean/log.csv",          "test10_clean/raw"),
    ]

    print("[Pseudo-labels] Generowanie z logow testowych...")
    for log_path, frames_dir in test_dirs:
        parse_test_log(str(base / log_path), str(base / frames_dir), labels)

    # Dodaj znane MISS klatki
    miss_dir = base / "miss and hit" / "miss"
    if miss_dir.exists():
        miss_files = sorted(miss_dir.glob("*.png"))
        for f in miss_files:
            labels.append({
                'file': f.name,
                'state': 'MISS_TEXT',
                'fish_x': None,
                'fish_y': None,
                'fish_visible': False,
                'source': 'manual',
            })
        print(f"  miss and hit/miss/: {len(miss_files)} MISS labels")

    # HIT klatki (folder pusty, ale gdyby sie pojawily)
    hit_dir = base / "miss and hit" / "hit"
    if hit_dir.exists():
        hit_files = sorted(hit_dir.glob("*.png"))
        for f in hit_files:
            labels.append({
                'file': f.name,
                'state': 'HIT_TEXT',
                'fish_x': None,
                'fish_y': None,
                'fish_visible': False,
                'source': 'manual',
            })
        if hit_files:
            print(f"  miss and hit/hit/: {len(hit_files)} HIT labels")

    # Zapisz
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + '\n')

    print(f"\n[Pseudo-labels] Zapisano {len(labels)} etykiet do {output_file}")

    # Podsumowanie po klasach
    counts = {}
    for l in labels:
        s = l['state']
        counts[s] = counts.get(s, 0) + 1
    print("Rozklad klas:")
    for state, count in sorted(counts.items()):
        print(f"  {state}: {count}")


if __name__ == "__main__":
    generate_all_pseudo_labels()
