"""
Generator datasetu patchy 32x32 do treningu Patch CNN (fish vs not-fish).

Tryby:
  generate  — wytwarza patche z oznaczonych klatek (manifest.json)
  review    — GUI do weryfikacji patchy (Y/N)
  stats     — podsumowanie datasetu
  augment   — augmentacja zatwierdzonych patchy

Dane zrodlowe:
  rybki_do_oceny/manifest.json  — pozycje rybek (fish_x, fish_y)
  rybki_do_oceny/raw/           — klatki 279x247
  rybki_do_oceny/marked/        — klatki z zaznaczeniami (backup)

Wynik (wzgledem roota repo):
  versions/tryb1_rybka_klik/post_cnn/cnn/patches/fish/       — patche z rybka (pozytywne)
  versions/tryb1_rybka_klik/post_cnn/cnn/patches/not_fish/   — patche bez rybki (negatywne)
  versions/tryb1_rybka_klik/post_cnn/cnn/patches/labels.json — etykiety + status weryfikacji
"""

import json
import random
import sys
import math
from pathlib import Path

import cv2
import numpy as np

# Sciezki
BASE = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST = BASE / 'rybki_do_oceny' / 'manifest.json'
RAW_DIR = BASE / 'rybki_do_oceny' / 'raw'
PATCHES_DIR = Path(__file__).resolve().parent / 'patches'
FISH_DIR = PATCHES_DIR / 'fish'
NOT_FISH_DIR = PATCHES_DIR / 'not_fish'
LABELS_FILE = PATCHES_DIR / 'labels.json'

# Parametry
PATCH_SIZE = 32
PATCH_HALF = PATCH_SIZE // 2
CIRCLE_CX = 140   # srodek okregu w klatce
CIRCLE_CY = 137
CIRCLE_R = 64

# Minimalna odleglosc negatywu od rybki (piksele)
NEG_MIN_DIST = 20

# Ile negatywow na klatke
NEGS_PER_FRAME = 3

# GUI
DISPLAY_SCALE = 8  # 32*8 = 256px


def _safe_crop(frame, cx, cy, half=PATCH_HALF):
    """Wycina patch z ramki, padduje jesli wychodzi poza."""
    h, w = frame.shape[:2]
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    # Oblicz padding
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bot = max(0, y2 - h)

    # Clamp
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(w, x2)
    cy2 = min(h, y2)

    crop = frame[cy1:cy2, cx1:cx2]

    if pad_left or pad_top or pad_right or pad_bot:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bot, pad_left, pad_right,
                                  cv2.BORDER_REFLECT_101)

    # Upewnij sie ze mamy dokladny rozmiar
    if crop.shape[0] != PATCH_SIZE or crop.shape[1] != PATCH_SIZE:
        crop = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE))

    return crop


def _is_in_circle(x, y, margin=5):
    """Czy punkt jest wewnatrz okregu minigry."""
    dist = math.sqrt((x - CIRCLE_CX)**2 + (y - CIRCLE_CY)**2)
    return dist <= (CIRCLE_R - margin)


def _random_neg_pos(fish_x, fish_y, max_attempts=50):
    """Losowa pozycja w okregu, min NEG_MIN_DIST od rybki."""
    for _ in range(max_attempts):
        # Losuj w kwadracie obejmujacym okrag
        x = random.randint(CIRCLE_CX - CIRCLE_R + PATCH_HALF,
                           CIRCLE_CX + CIRCLE_R - PATCH_HALF)
        y = random.randint(CIRCLE_CY - CIRCLE_R + PATCH_HALF,
                           CIRCLE_CY + CIRCLE_R - PATCH_HALF)

        if not _is_in_circle(x, y):
            continue

        dist = math.sqrt((x - fish_x)**2 + (y - fish_y)**2)
        if dist >= NEG_MIN_DIST:
            return (x, y)

    return None


def generate():
    """Generuje dataset patchy z manifest.json."""
    print("=== Generowanie datasetu patchy ===\n")

    if not MANIFEST.exists():
        print(f"BLAD: Brak {MANIFEST}")
        return

    with open(MANIFEST, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    print(f"Manifest: {len(manifest)} klatek z pozycjami ryb")
    print(f"Raw dir: {RAW_DIR} ({len(list(RAW_DIR.glob('*.png')))} plikow)")

    # Stworz foldery
    FISH_DIR.mkdir(parents=True, exist_ok=True)
    NOT_FISH_DIR.mkdir(parents=True, exist_ok=True)

    labels = []
    fish_count = 0
    neg_count = 0
    skipped = 0

    for entry in manifest:
        fname = entry['file']
        fish_x = entry['fish_x']
        fish_y = entry['fish_y']
        color = entry['color']
        source = entry.get('source', '')

        frame_path = RAW_DIR / fname
        if not frame_path.exists():
            skipped += 1
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            skipped += 1
            continue

        # --- POZYTYWNY: patch centrowany na rybce ---
        fish_patch = _safe_crop(frame, fish_x, fish_y)
        fish_name = f"fish_{source}_{fish_x:03d}_{fish_y:03d}_{fname}"
        cv2.imwrite(str(FISH_DIR / fish_name), fish_patch)

        labels.append({
            'file': fish_name,
            'class': 'fish',
            'source_frame': fname,
            'center_x': fish_x,
            'center_y': fish_y,
            'color': color,
            'verified': None,  # None = niezweryfikowany, True/False = user
        })
        fish_count += 1

        # --- NEGATYWNE: losowe patche z tej samej klatki ---
        for neg_i in range(NEGS_PER_FRAME):
            neg_pos = _random_neg_pos(fish_x, fish_y)
            if neg_pos is None:
                continue

            nx, ny = neg_pos
            neg_patch = _safe_crop(frame, nx, ny)
            neg_name = f"neg_{source}_{nx:03d}_{ny:03d}_n{neg_i}_{fname}"
            cv2.imwrite(str(NOT_FISH_DIR / neg_name), neg_patch)

            labels.append({
                'file': neg_name,
                'class': 'not_fish',
                'source_frame': fname,
                'center_x': nx,
                'center_y': ny,
                'color': color,
                'verified': None,
            })
            neg_count += 1

    # Zapisz etykiety
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\nWygenerowano:")
    print(f"  Pozytywne (fish):     {fish_count}")
    print(f"  Negatywne (not_fish): {neg_count}")
    print(f"  Pominiete:            {skipped}")
    print(f"\nZapisano: {LABELS_FILE}")
    print(f"Patche: {FISH_DIR}")
    print(f"         {NOT_FISH_DIR}")
    print(f"\nTeraz uruchom: python -m cnn.patch_dataset review")


def review():
    """
    GUI do weryfikacji patchy.

    Pokazuje patch powiekszony 8x (32→256px).
    Obok pokazuje oryginalny fragment klatki z zaznaczeniem.

    Y/SPACJA = OK (potwierdzam klase)
    N        = ZLE (zamien klase: fish→not_fish lub odwrotnie)
    D/→      = dalej (bez zmiany)
    A/←      = cofnij
    Q        = zapisz i wyjdz
    """
    if not LABELS_FILE.exists():
        print("Brak labels.json — najpierw uruchom: python -m cnn.patch_dataset generate")
        return

    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    print(f"\n=== Weryfikacja patchy ===")
    print(f"Razem: {len(labels)} patchy")

    # Statystyki
    verified = sum(1 for l in labels if l['verified'] is not None)
    fish = sum(1 for l in labels if l['class'] == 'fish')
    neg = sum(1 for l in labels if l['class'] == 'not_fish')
    print(f"Fish: {fish} | Not_fish: {neg} | Zweryfikowane: {verified}/{len(labels)}")

    # Znajdz pierwszy niezweryfikowany
    current = 0
    for i, l in enumerate(labels):
        if l['verified'] is None:
            current = i
            break

    win_name = "Patch Review: Y=OK  N=zmien klase  Q=wyjdz"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    while 0 <= current < len(labels):
        label = labels[current]
        cls = label['class']

        # Wczytaj patch
        if cls == 'fish':
            patch_path = FISH_DIR / label['file']
        else:
            patch_path = NOT_FISH_DIR / label['file']

        if not patch_path.exists():
            current += 1
            continue

        patch = cv2.imread(str(patch_path))
        if patch is None:
            current += 1
            continue

        # Powiekszony patch
        big_patch = cv2.resize(patch,
                               (PATCH_SIZE * DISPLAY_SCALE, PATCH_SIZE * DISPLAY_SCALE),
                               interpolation=cv2.INTER_NEAREST)

        # Wczytaj oryginalny frame i pokaz kontekst
        source_frame_path = RAW_DIR / label['source_frame']
        context_img = None
        if source_frame_path.exists():
            frame = cv2.imread(str(source_frame_path))
            if frame is not None:
                context_img = frame.copy()
                cx, cy = label['center_x'], label['center_y']
                # Zaznacz patch area
                cv2.rectangle(context_img,
                              (cx - PATCH_HALF, cy - PATCH_HALF),
                              (cx + PATCH_HALF, cy + PATCH_HALF),
                              (0, 255, 255), 2)
                # Zaznacz okrag
                cv2.circle(context_img, (CIRCLE_CX, CIRCLE_CY), CIRCLE_R,
                           (128, 128, 128), 1)
                # Skaluj do 256px wysokosci
                scale = (PATCH_SIZE * DISPLAY_SCALE) / context_img.shape[0]
                context_img = cv2.resize(context_img,
                                         (int(context_img.shape[1] * scale),
                                          PATCH_SIZE * DISPLAY_SCALE))

        # Status
        v = label['verified']
        if v is True:
            border = (0, 255, 0)
            status = "OK"
        elif v is False:
            border = (0, 165, 255)  # pomaranczowy = zmieniono
            status = "ZMIENIONO"
        else:
            border = (128, 128, 128)
            status = "???"

        # Klasa
        if cls == 'fish':
            cls_color = (0, 255, 0)
            cls_text = "RYBKA"
        else:
            cls_color = (0, 0, 255)
            cls_text = "NIE RYBKA"

        # Ramka
        cv2.rectangle(big_patch, (0, 0),
                      (big_patch.shape[1]-1, big_patch.shape[0]-1),
                      border, 4)

        # Tekst na patchu
        cv2.putText(big_patch, f"[{current+1}/{len(labels)}] {cls_text}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cls_color, 2)
        cv2.putText(big_patch, f"Status: {status}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border, 1)
        cv2.putText(big_patch, f"{label['color'].upper()} ({label['center_x']},{label['center_y']})",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Instrukcja na dole
        cv2.putText(big_patch, "Y=OK  N=zmien  Q=wyjdz",
                    (10, big_patch.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Polacz patch + kontekst obok siebie
        if context_img is not None:
            # Dopasuj wysokosc
            h1 = big_patch.shape[0]
            h2 = context_img.shape[0]
            if h1 != h2:
                context_img = cv2.resize(context_img,
                                         (int(context_img.shape[1] * h1 / h2), h1))
            display = np.hstack([big_patch, context_img])
        else:
            display = big_patch

        cv2.imshow(win_name, display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('y') or key == 32:  # Y lub SPACJA = potwierdzam klase
            labels[current]['verified'] = True
            v_count = sum(1 for l in labels if l['verified'] is not None)
            print(f"  OK [{current+1}/{len(labels)}] {cls_text} — {label['file'][:50]}  (zweryfikowano: {v_count})")
            current += 1
        elif key == ord('n'):  # N = zmien klase
            old_cls = labels[current]['class']
            new_cls = 'not_fish' if old_cls == 'fish' else 'fish'

            # Przenies plik
            old_dir = FISH_DIR if old_cls == 'fish' else NOT_FISH_DIR
            new_dir = NOT_FISH_DIR if old_cls == 'fish' else FISH_DIR
            old_path = old_dir / label['file']
            new_path = new_dir / label['file']
            if old_path.exists():
                old_path.rename(new_path)

            labels[current]['class'] = new_cls
            labels[current]['verified'] = True  # zmieniono = tez zweryfikowane
            print(f"  ZMIANA [{current+1}] {old_cls} -> {new_cls}: {label['file'][:50]}")
            current += 1
        elif key == ord('d') or key == 83 or key == 3:  # D lub →
            current += 1
        elif key == ord('a') or key == 81 or key == 2:  # A lub ←
            current = max(0, current - 1)

    # Zapisz
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    cv2.destroyAllWindows()

    # Podsumowanie
    ok = sum(1 for l in labels if l['verified'] is True)
    changed = sum(1 for l in labels if l['verified'] is True and l.get('_changed'))
    fish_final = sum(1 for l in labels if l['class'] == 'fish' and l['verified'])
    neg_final = sum(1 for l in labels if l['class'] == 'not_fish' and l['verified'])
    unseen = sum(1 for l in labels if l['verified'] is None)

    print(f"\n=== Podsumowanie ===")
    print(f"Zweryfikowane: {ok}/{len(labels)}")
    print(f"  Fish: {fish_final}")
    print(f"  Not_fish: {neg_final}")
    print(f"  Nieoznaczone: {unseen}")
    print(f"Zapisano: {LABELS_FILE}")


def stats():
    """Podsumowanie datasetu."""
    if not LABELS_FILE.exists():
        print("Brak labels.json")
        return

    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    total = len(labels)
    fish = [l for l in labels if l['class'] == 'fish']
    neg = [l for l in labels if l['class'] == 'not_fish']
    verified = [l for l in labels if l['verified'] is not None]
    fish_v = [l for l in fish if l['verified'] is True]
    neg_v = [l for l in neg if l['verified'] is True]

    print(f"=== Dataset stats ===")
    print(f"Razem patchy:     {total}")
    print(f"  Fish:           {len(fish)} (zweryfikowane: {len(fish_v)})")
    print(f"  Not_fish:       {len(neg)} (zweryfikowane: {len(neg_v)})")
    print(f"  Zweryfikowane:  {len(verified)}/{total}")
    print(f"\nPliki:")
    print(f"  {FISH_DIR}: {len(list(FISH_DIR.glob('*.png')))} plikow")
    print(f"  {NOT_FISH_DIR}: {len(list(NOT_FISH_DIR.glob('*.png')))} plikow")


def augment():
    """
    Augmentacja zweryfikowanych patchy.

    Tworzy dodatkowe warianty:
    - Flipy (H, V, HV)
    - Rotacje (90, 180, 270)
    - Jasnosc (+/-15)

    Cel: z ~200 fish -> ~2000+ probek
    """
    if not LABELS_FILE.exists():
        print("Brak labels.json")
        return

    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    # Tylko zweryfikowane
    verified_fish = [l for l in labels if l['class'] == 'fish' and l['verified'] is True]
    verified_neg = [l for l in labels if l['class'] == 'not_fish' and l['verified'] is True]

    if not verified_fish:
        print("Brak zweryfikowanych patchy fish! Uruchom review najpierw.")
        return

    print(f"=== Augmentacja ===")
    print(f"Fish do augmentacji: {len(verified_fish)}")
    print(f"Not_fish do augmentacji: {len(verified_neg)}")

    aug_count = 0

    for lbl in verified_fish + verified_neg:
        cls = lbl['class']
        src_dir = FISH_DIR if cls == 'fish' else NOT_FISH_DIR
        src_path = src_dir / lbl['file']

        if not src_path.exists():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        stem = src_path.stem

        augmentations = {
            'flipH': cv2.flip(img, 1),
            'flipV': cv2.flip(img, 0),
            'flipHV': cv2.flip(img, -1),
            'rot90': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            'rot180': cv2.rotate(img, cv2.ROTATE_180),
            'rot270': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            'bright_p': cv2.convertScaleAbs(img, alpha=1.0, beta=15),
            'bright_n': cv2.convertScaleAbs(img, alpha=1.0, beta=-15),
        }

        for aug_name, aug_img in augmentations.items():
            aug_fname = f"{stem}_aug_{aug_name}.png"
            aug_path = src_dir / aug_fname
            if not aug_path.exists():
                cv2.imwrite(str(aug_path), aug_img)
                aug_count += 1

    print(f"Wygenerowano {aug_count} augmentacji")
    print(f"Fish dir: {len(list(FISH_DIR.glob('*.png')))} plikow")
    print(f"Not_fish dir: {len(list(NOT_FISH_DIR.glob('*.png')))} plikow")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uzycie:")
        print("  python -m cnn.patch_dataset generate  — generuj patche z manifest.json")
        print("  python -m cnn.patch_dataset review    — weryfikuj patche (GUI)")
        print("  python -m cnn.patch_dataset stats     — statystyki datasetu")
        print("  python -m cnn.patch_dataset augment   — augmentacja zweryfikowanych")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == 'generate':
        generate()
    elif mode == 'review':
        review()
    elif mode == 'stats':
        stats()
    elif mode == 'augment':
        augment()
    else:
        print(f"Nieznany tryb: {mode}")
