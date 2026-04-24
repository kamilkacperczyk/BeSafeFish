"""
Detektor rybki — tlo referencyjne (mediana wielu klatek) + template matching.

Rybka rozni sie od tla wody (woda jasna S=190 V=215, rybka ciemna S=144 V=134).
Strategia:
1. Tlo referencyjne = mediana pikseli z wielu klatek (statyczne elementy znikaja)
2. Roznica klatki od tla referencyjnego → maska zmian
3. Blob detection + filtr ksztaltu → kandydaci na rybke
4. Template matching → dokladna pozycja

Szablony ladowane z: rybki_do_oceny/fish_shapes/Nowy folder/
Tlo referencyjne: cache/bg_reference.npy (generowane z dostepnych klatek)
"""

import math
from pathlib import Path

import cv2
import numpy as np

from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS


# Progi roznicy od tla referencyjnego
# Rybka: diff_SV ~100-140 (S-50 + V-80)
# Statyczne artefakty: diff_SV ~0-10 (bo sa w tle referer.)
DIFF_THRESH = 18       # |piksel - bg_ref| per kanal (S lub V)
DIFF_COMBINED = 28     # suma |dS|+|dV| musi byc >= 28

# Rozmiar rybki (~20x19px, area ~150-190)
# Ale w niektorychy klatkach rybka jest mniejsza/wieksza
FISH_MIN_AREA = 18
FISH_MAX_AREA = 1000

# Maska okregu (cache)
_circle_mask_cache = {}

# Sciezka do cache tla referencyjnego
_BG_REF_FILENAME = "bg_reference.npy"


def _get_circle_mask(shape):
    """Maska okregu (cached)."""
    key = shape[:2]
    if key not in _circle_mask_cache:
        mask = np.zeros(key, dtype=np.uint8)
        cv2.circle(mask, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y),
                   CIRCLE_RADIUS - 5, 255, -1)
        # Wytnij UI gora/dol
        mask[:22, :] = 0
        mask[-18:, :] = 0
        _circle_mask_cache[key] = mask
    return _circle_mask_cache[key]


class FishShapeDetector:
    """
    Detektor rybki — tlo referencyjne + template matching.

    Tlo referencyjne = mediana pikselowa z wielu klatek.
    Statyczne elementy (woda, dekoracje) sa w tle i nie generuja detekcji.
    Rybka (w innej pozycji w kazdej klatce) rozni sie od tla.
    """

    def __init__(self, templates_dir: str = None, bg_ref_path: str = None):
        """
        Args:
            templates_dir: folder z szablonami rybki
            bg_ref_path: sciezka do .npy z tlem referencyjnym
                         Jesli None, sprobuje zaladowac z cache/
                         Jesli brak pliku, oblicza z dostepnych klatek
        """
        self.templates = []
        self.bg_ref = None  # numpy array HxWx3 BGR

        if templates_dir is None:
            base = Path(__file__).resolve().parent.parent.parent.parent
            templates_dir = base / 'rybki_do_oceny' / 'fish_shapes' / 'Nowy folder'

        self._load_templates(str(templates_dir))
        self._load_bg_reference(bg_ref_path)

    def _load_templates(self, templates_dir: str):
        """Laduje szablony rybki (wyciete na bialym tle) + obrocone wersje."""
        tdir = Path(templates_dir)
        if not tdir.exists():
            print(f"[FishShape] UWAGA: Brak szablonow w {templates_dir}")
            return

        template_files = [f for f in sorted(tdir.glob("*.png"))
                          if not f.name.startswith("mask_")]

        for tf in template_files:
            img = cv2.imread(str(tf))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

            self.templates.append((gray, mask, tf.name))

            # Obroty co 45 stopni
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            for angle in [45, 90, 135, 180, 225, 270, 315]:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cos, sin = abs(M[0, 0]), abs(M[0, 1])
                nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
                M[0, 2] += (nw - w) / 2
                M[1, 2] += (nh - h) / 2

                rg = cv2.warpAffine(gray, M, (nw, nh), borderValue=255)
                rm = cv2.warpAffine(mask, M, (nw, nh), borderValue=0)

                coords = cv2.findNonZero(rm)
                if coords is not None and len(coords) > 10:
                    x, y, bw, bh = cv2.boundingRect(coords)
                    p = 2
                    rg = rg[max(0,y-p):min(nh,y+bh+p), max(0,x-p):min(nw,x+bw+p)]
                    rm = rm[max(0,y-p):min(nh,y+bh+p), max(0,x-p):min(nw,x+bw+p)]
                    self.templates.append((rg, rm, f"{tf.stem}_r{angle}"))

        print(f"[FishShape] {len(template_files)} szablonow → "
              f"{len(self.templates)} wersji (z rotacjami)")

    def _load_bg_reference(self, bg_ref_path: str = None):
        """Laduje lub generuje tlo referencyjne (mediana wielu klatek)."""
        base = Path(__file__).resolve().parent.parent.parent.parent

        # Probuj zaladowac z pliku
        if bg_ref_path:
            ref_path = Path(bg_ref_path)
        else:
            cache_dir = base / 'versions' / 'post_cnn' / 'cnn' / 'cache'
            ref_path = cache_dir / _BG_REF_FILENAME

        if ref_path.exists():
            self.bg_ref = np.load(str(ref_path))
            print(f"[FishShape] Tlo referencyjne zaladowane: {ref_path.name} "
                  f"({self.bg_ref.shape})")
            return

        # Generuj z dostepnych klatek
        raw_dir = base / 'rybki_do_oceny' / 'raw'
        if not raw_dir.exists():
            print(f"[FishShape] UWAGA: Brak klatek do generowania tla!")
            return

        self._generate_bg_reference(raw_dir, ref_path)

    def _generate_bg_reference(self, frames_dir: Path, save_path: Path):
        """
        Generuje tlo referencyjne = mediana pikselowa z wielu klatek.

        Rybka jest w innej pozycji w kazdej klatce, wiec mediana
        da czysty obraz tla bez rybki.
        """
        frames_files = sorted(frames_dir.glob("*.png"))
        if not frames_files:
            print("[FishShape] Brak klatek do generowania tla!")
            return

        # Uzyj max 50 klatek (rownomiernie rozlozonych) — wystarczy do mediany
        step = max(1, len(frames_files) // 50)
        selected = frames_files[::step][:50]

        # Wczytaj klatki
        stack = []
        for f in selected:
            img = cv2.imread(str(f))
            if img is not None:
                stack.append(img)

        if len(stack) < 5:
            print(f"[FishShape] Za malo klatek ({len(stack)}) do generowania tla!")
            return

        # Mediana pikselowa
        stack_arr = np.array(stack, dtype=np.uint8)
        self.bg_ref = np.median(stack_arr, axis=0).astype(np.uint8)

        # Zapisz do cache
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), self.bg_ref)
        print(f"[FishShape] Tlo referencyjne wygenerowane z {len(stack)} klatek "
              f"→ {save_path.name}")

    # --- Glowna metoda: roznica od tla referencyjnego ---

    def _compute_diff_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Maska roznicy: piksele rozniace sie od tla referencyjnego.

        Uzywam kanalow S i V z HSV, bo sa najbardziej dyskryminujace
        (rybka ciemniejsza V=134 vs woda V=215, mniej nasycona S=144 vs S=197).
        """
        circle_mask = _get_circle_mask(frame_bgr.shape)

        if self.bg_ref is not None:
            # Roznica od tla referencyjnego (pikselowa)
            bg_hsv = cv2.cvtColor(self.bg_ref, cv2.COLOR_BGR2HSV)
            fr_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            diff_s = np.abs(fr_hsv[:, :, 1].astype(np.float32) -
                            bg_hsv[:, :, 1].astype(np.float32))
            diff_v = np.abs(fr_hsv[:, :, 2].astype(np.float32) -
                            bg_hsv[:, :, 2].astype(np.float32))
            combined = diff_s + diff_v

            mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            mask[(diff_s > DIFF_THRESH) | (diff_v > DIFF_THRESH)] = 255
            mask[combined < DIFF_COMBINED] = 0
        else:
            # Fallback: roznica od mediany tego samego frame'u
            fr_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            water = fr_hsv[circle_mask > 0]
            if len(water) < 100:
                return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            med_s = np.median(water[:, 1])
            med_v = np.median(water[:, 2])

            s_ch = fr_hsv[:, :, 1].astype(np.float32)
            v_ch = fr_hsv[:, :, 2].astype(np.float32)
            diff_s = np.abs(s_ch - med_s)
            diff_v = np.abs(v_ch - med_v)
            combined = diff_s + diff_v

            mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            mask[(diff_s > DIFF_THRESH) | (diff_v > DIFF_THRESH)] = 255
            mask[combined < DIFF_COMBINED] = 0

        # Ogranicz do okregu
        mask = cv2.bitwise_and(mask, circle_mask)

        # Oczyszczenie
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def find_fish(self, frame_bgr: np.ndarray) -> tuple:
        """
        Znajduje rybke — roznica od mediany tla + filtr ksztaltu.

        Returns:
            (x, y, confidence) lub None
        """
        diff_mask = self._compute_diff_mask(frame_bgr)

        contours, _ = cv2.findContours(
            diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if FISH_MIN_AREA <= area <= FISH_MAX_AREA:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, bw, bh = cv2.boundingRect(c)
                    aspect = bw / max(bh, 1)
                    # Rybka obraca sie swobodnie — aspect moze byc 0.2-5.0
                    if 0.15 <= aspect <= 6.0:
                        # Score: im blizej rozmiaru rybki (~20x19=~180px), tym lepiej
                        size_score = 1.0 - abs(area - 180) / 500
                        size_score = max(0.1, min(1.0, size_score))
                        candidates.append((size_score, cx, cy, x, y, bw, bh, area))

        if not candidates:
            return None

        # Sortuj po size_score (najlepsza rybka najpierw)
        candidates.sort(key=lambda c: c[0], reverse=True)

        # Template matching na top kandydatach (jesli mamy szablony)
        if self.templates:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            best_match = None
            best_score = 0

            for score, cx, cy, bx, by, bw, bh, area in candidates[:5]:
                pad = 12
                rx1, ry1 = max(0, bx - pad), max(0, by - pad)
                rx2 = min(frame_bgr.shape[1], bx + bw + pad)
                ry2 = min(frame_bgr.shape[0], by + bh + pad)
                region = gray[ry1:ry2, rx1:rx2]

                if region.shape[0] < 8 or region.shape[1] < 8:
                    continue

                for tmpl_g, tmpl_m, _ in self.templates:
                    th, tw = tmpl_g.shape[:2]
                    if tw > region.shape[1] or th > region.shape[0]:
                        continue
                    try:
                        res = cv2.matchTemplate(
                            region, tmpl_g, cv2.TM_CCOEFF_NORMED,
                            mask=tmpl_m
                        )
                        _, mv, _, ml = cv2.minMaxLoc(res)
                        # Laczony score: template * rozmiar
                        combined = mv * 0.6 + score * 0.4
                        if combined > best_score:
                            best_score = combined
                            mx = rx1 + ml[0] + tw // 2
                            my = ry1 + ml[1] + th // 2
                            best_match = (mx, my, combined)
                    except cv2.error:
                        continue

            if best_match:
                x, y, conf = best_match
                dist = math.sqrt((x - CIRCLE_CENTER_X)**2 +
                                 (y - CIRCLE_CENTER_Y)**2)
                if dist <= CIRCLE_RADIUS + 5:
                    return best_match

        # Fallback: najlepszy kandydat po size_score
        score, cx, cy, *_ = candidates[0]
        dist = math.sqrt((cx - CIRCLE_CENTER_X)**2 +
                         (cy - CIRCLE_CENTER_Y)**2)
        if dist <= CIRCLE_RADIUS + 5:
            return (cx, cy, score * 0.5)

        return None

    def find_fish_simple(self, frame_bgr: np.ndarray) -> tuple:
        """
        Szybka detekcja — sama roznica od tla (bez template matching).

        Returns:
            (x, y) lub None
        """
        diff_mask = self._compute_diff_mask(frame_bgr)

        contours, _ = cv2.findContours(
            diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best = None
        best_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            if FISH_MIN_AREA <= area <= FISH_MAX_AREA:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = math.sqrt((cx - CIRCLE_CENTER_X)**2 +
                                     (cy - CIRCLE_CENTER_Y)**2)
                    if dist <= CIRCLE_RADIUS:
                        # Score = bliskosc rozmiaru do ~180px
                        score = 1.0 - abs(area - 180) / 300
                        if score > best_score:
                            best = (cx, cy)
                            best_score = score

        return best

    def debug_visualize(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Wizualizacja debugowa — roznica + detekcja."""
        diff_mask = self._compute_diff_mask(frame_bgr)
        result = self.find_fish(frame_bgr)

        vis = frame_bgr.copy()

        # Zielony overlay na masce roznicy
        overlay = vis.copy()
        overlay[diff_mask > 0] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        if result:
            x, y, conf = result
            cv2.drawMarker(vis, (x, y), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, f"Fish ({x},{y}) c={conf:.2f}",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(vis, "No fish",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return vis


# --- Test ---
if __name__ == "__main__":
    import time

    print("=== Test FishShapeDetector v2 (diff-based) ===\n")
    detector = FishShapeDetector()

    BASE = Path(__file__).resolve().parent.parent.parent.parent
    raw_dir = BASE / 'rybki_do_oceny' / 'raw'
    frames = sorted(raw_dir.glob("*.png"))
    print(f"Testuje na {len(frames)} klatkach\n")

    detected = 0
    times = []
    positions = []

    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        t0 = time.perf_counter()
        result = detector.find_fish(img)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

        if result:
            detected += 1
            x, y, conf = result
            positions.append((x, y, conf, f.name))

    total = len(times)
    print(f"Wykryto: {detected}/{total} ({100*detected/total:.1f}%)")
    print(f"Czas: mean={np.mean(times):.1f}ms, "
          f"median={np.median(times):.1f}ms, max={np.max(times):.1f}ms\n")

    # Pokaz rozklad pozycji
    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        confs = [p[2] for p in positions]
        print(f"Pozycje: X={np.mean(xs):.0f}±{np.std(xs):.0f}, "
              f"Y={np.mean(ys):.0f}±{np.std(ys):.0f}")
        print(f"Conf: mean={np.mean(confs):.3f}, "
              f"min={np.min(confs):.3f}, max={np.max(confs):.3f}\n")

        # Pokaz unikalne pozycje
        from collections import Counter
        pos_counts = Counter((p[0], p[1]) for p in positions)
        print(f"Unikalnych pozycji: {len(pos_counts)}")
        print("Top 10 najczestszych:")
        for (x, y), cnt in pos_counts.most_common(10):
            print(f"  ({x:3d},{y:3d}): {cnt}x")

        print(f"\nPrzykladowe (co 20):")
        for p in positions[::20]:
            print(f"  {p[3]:50s} → ({p[0]:3d},{p[1]:3d}) c={p[2]:.3f}")
