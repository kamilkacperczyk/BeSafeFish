"""
Analiza klatek BAD vs OK - bezpośrednie porównanie pikseli.
Szukamy cech odróżniających napisy HIT/MISS od prawdziwej rybki.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS

OUTPUT_DIR = "analiza_bad"


def analyze_frame_deep(path, label):
    """Dogłębna analiza pikseli w klatce."""
    img = cv2.imread(path)
    if img is None:
        print(f"  [!] Nie mogę wczytać: {path}")
        return None

    # Usuń debug overlay tekst (górne 20px) 
    clean = img.copy()
    clean[0:20, :] = clean[20, :]

    # Maska okręgu
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, 255, -1)

    # Piksele WEWNĄTRZ okręgu
    circle_pixels = clean[mask > 0]  # shape: (N, 3) BGR

    # Konwersja do HSV wewnątrz okręgu
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    circle_hsv = hsv[mask > 0]

    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    circle_gray = gray[mask > 0]

    # Szukaj jasnych pikseli (potencjalny tekst) wewnątrz okręgu
    # Tekst HIT/MISS jest zwykle jasny na ciemnym tle
    bright_thresh = 200
    bright_mask = circle_gray > bright_thresh
    num_bright = np.count_nonzero(bright_mask)
    
    # Szukaj pikseli o wysokiej saturacji (kolorowy tekst)
    high_sat_mask = circle_hsv[:, 1] > 100
    num_high_sat = np.count_nonzero(high_sat_mask)

    # Analiza kolorów jasnych pikseli
    bright_colors = circle_pixels[bright_mask] if num_bright > 0 else np.array([])

    # Szukaj żółtego/pomarańczowego tekstu (typowe dla gier)
    # HSV: żółty H=20-40, pomarańczowy H=10-20, wysoka S i V
    yellow_mask = (circle_hsv[:, 0] >= 15) & (circle_hsv[:, 0] <= 45) & \
                  (circle_hsv[:, 1] > 80) & (circle_hsv[:, 2] > 150)
    num_yellow = np.count_nonzero(yellow_mask)

    # Szukaj białego tekstu
    white_mask = (circle_hsv[:, 1] < 40) & (circle_hsv[:, 2] > 200)
    num_white = np.count_nonzero(white_mask)

    # Szukaj zielonego (debug marker, nie tekst)
    green_mask = (circle_hsv[:, 0] >= 35) & (circle_hsv[:, 0] <= 85) & \
                 (circle_hsv[:, 1] > 100) & (circle_hsv[:, 2] > 100)
    num_green = np.count_nonzero(green_mask)
    
    # Szukaj czerwonego tekstu
    red_mask = ((circle_hsv[:, 0] <= 10) | (circle_hsv[:, 0] >= 170)) & \
               (circle_hsv[:, 1] > 80) & (circle_hsv[:, 2] > 150)
    num_red = np.count_nonzero(red_mask)

    # Analiza kontrastowa - odchylenie standardowe jasności
    gray_std = np.std(circle_gray)
    gray_mean = np.mean(circle_gray)

    # Szukaj krawędzi (tekst ma dużo krawędzi)
    edges = cv2.Canny(gray, 50, 150)
    circle_edges = edges[mask > 0]
    num_edges = np.count_nonzero(circle_edges)

    # Gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    circle_gradient = gradient[mask > 0]
    grad_mean = np.mean(circle_gradient)

    total_circle = np.count_nonzero(mask)

    print(f"\n{'='*70}")
    print(f"  {label}: {os.path.basename(path)}")
    print(f"{'='*70}")
    print(f"  Pikseli w okręgu: {total_circle}")
    print(f"  Jasność: mean={gray_mean:.1f}, std={gray_std:.1f}")
    print(f"  Gradient mean: {grad_mean:.1f}")
    print(f"  Krawędzie (Canny): {num_edges} ({100*num_edges/total_circle:.1f}%)")
    print(f"  Jasne (>{bright_thresh}): {num_bright} ({100*num_bright/total_circle:.1f}%)")
    print(f"  Wysoka saturacja: {num_high_sat} ({100*num_high_sat/total_circle:.1f}%)")
    print(f"  Żółte/Pomarańcz.: {num_yellow} ({100*num_yellow/total_circle:.1f}%)")
    print(f"  Białe: {num_white} ({100*num_white/total_circle:.1f}%)")
    print(f"  Zielone: {num_green} ({100*num_green/total_circle:.1f}%)")
    print(f"  Czerwone: {num_red} ({100*num_red/total_circle:.1f}%)")
    
    if len(bright_colors) > 0:
        print(f"  Średni kolor jasnych (BGR): ({bright_colors[:,0].mean():.0f}, {bright_colors[:,1].mean():.0f}, {bright_colors[:,2].mean():.0f})")

    return {
        'label': label,
        'gray_mean': gray_mean, 'gray_std': gray_std,
        'grad_mean': grad_mean, 'edges': num_edges,
        'bright': num_bright, 'high_sat': num_high_sat,
        'yellow': num_yellow, 'white': num_white,
        'green': num_green, 'red': num_red,
        'total': total_circle,
    }


def save_comparison_images(frames_list, output_name):
    """Zapisuje porównanie klatek obok siebie."""
    imgs = []
    for path, label in frames_list:
        img = cv2.imread(path)
        if img is not None:
            # Dodaj etykietę
            display = img.copy()
            cv2.putText(display, label[:40], (5, img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            imgs.append(display)
    
    if imgs:
        # Ułóż w rzędzie
        combined = np.hstack(imgs) if len(imgs) <= 5 else np.vstack([
            np.hstack(imgs[:len(imgs)//2 + 1]),
            np.hstack(imgs[len(imgs)//2 + 1:] + [np.zeros_like(imgs[0])] * (len(imgs)//2 + 1 - len(imgs[len(imgs)//2 + 1:])))
        ])
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, output_name), combined)
        print(f"\n  Zapisano: {OUTPUT_DIR}/{output_name}")


def main():
    print("DOGŁĘBNA ANALIZA PIKSELI - napisy HIT/MISS vs rybka")
    print("=" * 70)

    miss_frames = [
        ("test8b_miss/frames/frame_0461_red.png", "BAD-MISS #461"),
        ("test8b_miss/frames/frame_0466_red.png", "BAD-MISS #466"),
    ]

    hit_frames = [
        ("test8c_hit/frames/frame_0241_red.png", "BAD-HIT #241"),
        ("test8c_hit/frames/frame_0246_red.png", "BAD-HIT #246"),
        ("test8c_hit/frames/frame_0251_red.png", "BAD-HIT #251"),
    ]

    ok_frames = [
        ("test8a_tracking/frames/frame_0006_red.png", "OK-ryba 8a#6"),
        ("test8a_tracking/frames/frame_0011_red.png", "OK-ryba 8a#11"),
        ("test8c_hit/frames/frame_0046_red.png", "OK-ryba 8c#46"),
        ("test8c_hit/frames/frame_0076_red.png", "OK-ryba 8c#76"),
        ("test8c_hit/frames/frame_0116_red.png", "OK-ryba 8c#116"),
        ("test8c_hit/frames/frame_0236_red.png", "OK-ryba 8c#236"),
    ]

    # Analiza
    all_results = {'miss': [], 'hit': [], 'ok': []}

    print("\n\n" + "=" * 70)
    print("  KLATKI Z NAPISEM MISS")
    print("=" * 70)
    for path, label in miss_frames:
        if os.path.exists(path):
            r = analyze_frame_deep(path, label)
            if r: all_results['miss'].append(r)

    print("\n\n" + "=" * 70)
    print("  KLATKI Z NAPISEM HIT")
    print("=" * 70)
    for path, label in hit_frames:
        if os.path.exists(path):
            r = analyze_frame_deep(path, label)
            if r: all_results['hit'].append(r)

    print("\n\n" + "=" * 70)
    print("  KLATKI OK (prawdziwa ryba)")
    print("=" * 70)
    for path, label in ok_frames:
        if os.path.exists(path):
            r = analyze_frame_deep(path, label)
            if r: all_results['ok'].append(r)

    # PODSUMOWANIE
    print("\n\n" + "=" * 70)
    print("  PODSUMOWANIE ŚREDNICH")
    print("=" * 70)

    def avg(data, key):
        vals = [d[key] for d in data]
        return sum(vals)/len(vals) if vals else 0

    def avg_pct(data, key):
        return 100 * avg(data, key) / avg(data, 'total') if avg(data, 'total') else 0

    headers = f"\n  {'Cecha':<22} {'MISS(BAD)':<14} {'HIT(BAD)':<14} {'Ryba(OK)':<14} {'Różnica?'}"
    print(headers)
    print("  " + "-" * 66)

    metrics = [
        ('gray_mean', 'Jasność mean', False),
        ('gray_std', 'Jasność std', False),
        ('grad_mean', 'Gradient mean', False),
        ('edges', 'Krawędzie', True),
        ('bright', 'Jasne piksele', True),
        ('high_sat', 'Wys. saturacja', True),
        ('yellow', 'Żółte', True),
        ('white', 'Białe', True),
        ('green', 'Zielone', True),
        ('red', 'Czerwone', True),
    ]

    for key, name, is_pct in metrics:
        if is_pct:
            m = avg_pct(all_results['miss'], key)
            h = avg_pct(all_results['hit'], key)
            o = avg_pct(all_results['ok'], key)
            diff = "***" if abs(max(m,h) - o) > 2 else ""
            print(f"  {name:<22} {m:>8.1f}%     {h:>8.1f}%     {o:>8.1f}%     {diff}")
        else:
            m = avg(all_results['miss'], key)
            h = avg(all_results['hit'], key)
            o = avg(all_results['ok'], key)
            diff = "***" if abs(max(m,h) - o) / max(o, 1) > 0.2 else ""
            print(f"  {name:<22} {m:>10.1f}   {h:>10.1f}   {o:>10.1f}   {diff}")

    # Zapisz porównanie wizualne
    all_for_compare = [(p, l) for p, l in miss_frames + hit_frames + ok_frames[:3] if os.path.exists(p)]
    save_comparison_images(all_for_compare, "comparison.png")


if __name__ == "__main__":
    main()
