"""
Analiza kolorów rybki na zapisanych klatkach diagnostycznych.
Cel: znaleźć optymalny filtr HSV do bezpośredniej detekcji rybki
(zamiast frame differencing).
"""
import cv2
import numpy as np
import os
import math

DIAG_DIR = "diagnostyka_live"
LOG_FILE = os.path.join(DIAG_DIR, "log.txt")

# Środek i promień okręgu (relative to fishing box)
CX, CY, CR = 140, 137, 64

def parse_log():
    """Parsuje log i zwraca listę (frame_num, time, color, fish_x, fish_y)."""
    entries = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fnum = int(parts[0][1:])  # f0001 -> 1
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

def analyze_frame(frame_path, cx=CX, cy=CY, cr=CR):
    """Analizuje kolory pikseli wewnątrz okręgu w danej klatce."""
    img = cv2.imread(frame_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Stwórz maskę okręgu
    h, w = img.shape[:2]
    mask_circle = np.zeros((h, w), np.uint8)
    cv2.circle(mask_circle, (cx, cy), cr, 255, -1)
    
    # Piksele wewnątrz okręgu
    circle_hsv = hsv[mask_circle > 0]
    
    return {
        "h_mean": circle_hsv[:, 0].mean(),
        "s_mean": circle_hsv[:, 1].mean(),
        "v_mean": circle_hsv[:, 2].mean(),
        "h_std": circle_hsv[:, 0].std(),
        "s_std": circle_hsv[:, 1].std(),
        "v_std": circle_hsv[:, 2].std(),
    }

def find_fish_color_range(entries):
    """
    Dla klatek gdzie rybka została wykryta, analizuje kolory
    w otoczeniu pozycji rybki.
    """
    print("=== Analiza kolorów rybki ===\n")
    
    fish_colors_white = []
    fish_colors_red = []
    fish_colors_none = []
    
    bg_colors_white = []
    bg_colors_red = []
    
    for fnum, t, color, fx, fy in entries:
        frame_path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
        if not os.path.exists(frame_path):
            continue
        
        img = cv2.imread(frame_path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_img, w_img = img.shape[:2]
        
        if fx is not None and fy is not None:
            # Pobierz kolory pikseli wokół rybki (7x7 patch)
            r = 5
            y1, y2 = max(0, fy-r), min(h_img, fy+r)
            x1, x2 = max(0, fx-r), min(w_img, fx+r)
            fish_patch = hsv[y1:y2, x1:x2]
            if fish_patch.size > 0:
                avg_h = fish_patch[:,:,0].mean()
                avg_s = fish_patch[:,:,1].mean()
                avg_v = fish_patch[:,:,2].mean()
                if color == "white":
                    fish_colors_white.append((avg_h, avg_s, avg_v))
                elif color == "red":
                    fish_colors_red.append((avg_h, avg_s, avg_v))
                else:
                    fish_colors_none.append((avg_h, avg_s, avg_v))
        
        # Tło (środek okręgu, 5x5 patch)
        if color in ("white", "red"):
            # Sprawdź tło daleko od rybki (np. CX+30, CY-30)
            bx, by = CX + 30, CY - 30
            if fx and fy:
                # Wybierz punkt daleko od rybki
                dx, dy = CX - fx, CY - fy
                bx = int(CX + dx * 0.5)
                by = int(CY + dy * 0.5)
            bx = max(5, min(w_img-5, bx))
            by = max(5, min(h_img-5, by))
            bg_patch = hsv[by-3:by+3, bx-3:bx+3]
            if bg_patch.size > 0:
                bg_h = bg_patch[:,:,0].mean()
                bg_s = bg_patch[:,:,1].mean()
                bg_v = bg_patch[:,:,2].mean()
                if color == "white":
                    bg_colors_white.append((bg_h, bg_s, bg_v))
                elif color == "red":
                    bg_colors_red.append((bg_h, bg_s, bg_v))
    
    # Raport
    for label, colors in [
        ("Rybka w WHITE fazie", fish_colors_white),
        ("Rybka w RED fazie", fish_colors_red),
        ("Rybka w NONE fazie", fish_colors_none),
    ]:
        if colors:
            arr = np.array(colors)
            print(f"\n{label} ({len(colors)} próbek):")
            print(f"  H: mean={arr[:,0].mean():.1f}, std={arr[:,0].std():.1f}, min={arr[:,0].min():.1f}, max={arr[:,0].max():.1f}")
            print(f"  S: mean={arr[:,1].mean():.1f}, std={arr[:,1].std():.1f}, min={arr[:,1].min():.1f}, max={arr[:,1].max():.1f}")
            print(f"  V: mean={arr[:,2].mean():.1f}, std={arr[:,2].std():.1f}, min={arr[:,2].min():.1f}, max={arr[:,2].max():.1f}")
        else:
            print(f"\n{label}: BRAK DANYCH")
    
    for label, colors in [
        ("Tło WHITE", bg_colors_white),
        ("Tło RED", bg_colors_red),
    ]:
        if colors:
            arr = np.array(colors)
            print(f"\n{label} ({len(colors)} próbek):")
            print(f"  H: mean={arr[:,0].mean():.1f}, std={arr[:,0].std():.1f}, min={arr[:,0].min():.1f}, max={arr[:,0].max():.1f}")
            print(f"  S: mean={arr[:,1].mean():.1f}, std={arr[:,1].std():.1f}, min={arr[:,1].min():.1f}, max={arr[:,1].max():.1f}")
            print(f"  V: mean={arr[:,2].mean():.1f}, std={arr[:,2].std():.1f}, min={arr[:,2].min():.1f}, max={arr[:,2].max():.1f}")

def scan_all_colors():
    """
    Skanuje kilka wybranych klatek i sprawdza co jest wewnątrz okręgu.
    Porównuje kolory w fazie białej i czerwonej.
    """
    entries = parse_log()
    
    # Wybierz kilka klatek z różnych faz
    white_frames = [(e[0], e) for e in entries if e[2] == "white"][:5]
    red_frames = [(e[0], e) for e in entries if e[2] == "red"][:5]
    
    print("\n=== Kolory wewnątrz okręgu ===")
    for label, frames in [("WHITE", white_frames), ("RED", red_frames)]:
        print(f"\n--- Faza {label} ---")
        for fnum, entry in frames:
            frame_path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
            result = analyze_frame(frame_path)
            if result:
                print(f"  Frame {fnum:04d}: H={result['h_mean']:.1f}±{result['h_std']:.1f}, "
                      f"S={result['s_mean']:.1f}±{result['s_std']:.1f}, "
                      f"V={result['v_mean']:.1f}±{result['v_std']:.1f}")

def detailed_pixel_scan():
    """
    Dla wybranych klatek z wykrytą rybką, skanuje różne zakresy HSV
    i liczy ile pikseli wewnątrz okręgu pasuje - szukamy optymalnego filtra.
    """
    entries = parse_log()
    
    # Klatki z rybką w fazie RED
    red_fish = [e for e in entries if e[2] == "red" and e[3] is not None]
    white_fish = [e for e in entries if e[2] == "white" and e[3] is not None]
    
    print(f"\n=== Szczegółowy skan pikseli ===")
    print(f"Klatki RED z rybką: {len(red_fish)}")
    print(f"Klatki WHITE z rybką: {len(white_fish)}")
    
    # Test różnych filtrów HSV na kilku klatkach
    test_frames = red_fish[:5] + white_fish[:5]
    
    filters = [
        ("Oliwkowa rybka (H:25-50, S:100-255, V:100-200)", (25, 100, 100), (50, 255, 200)),
        ("Oliwkowa loose (H:20-55, S:80-255, V:80-220)", (20, 80, 80), (55, 255, 220)),
        ("Ciemna rybka (H:15-60, S:50-255, V:50-180)", (15, 50, 50), (60, 255, 180)),
        ("Zielonkawy (H:25-45, S:150-255, V:120-200)", (25, 150, 120), (45, 255, 200)),
        ("Szeroki (H:20-60, S:50-255, V:50-255)", (20, 50, 50), (60, 255, 255)),
        ("Bardzo szeroki (H:10-70, S:30-255, V:30-255)", (10, 30, 30), (70, 255, 255)),
    ]
    
    print("\n--- Ilość pasujących pikseli w okręgu ---")
    for fname, low, high in filters:
        print(f"\n  Filtr: {fname}")
        for entry in test_frames:
            fnum, t, color, fx, fy = entry
            frame_path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
            img = cv2.imread(frame_path)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_img, w_img = img.shape[:2]
            
            # Maska okręgu
            mask_circle = np.zeros((h_img, w_img), np.uint8)
            cv2.circle(mask_circle, (CX, CY), CR, 255, -1)
            
            # Maska koloru
            mask_color = cv2.inRange(hsv, np.array(low), np.array(high))
            
            # Połącz: piksele pasujące do koloru WEWNĄTRZ okręgu
            combined = cv2.bitwise_and(mask_color, mask_circle)
            count = cv2.countNonZero(combined)
            
            fish_str = f"fish=({fx},{fy})" if fx else "fish=None"
            print(f"    f{fnum:04d} [{color:5s}] {fish_str:20s} -> {count:4d} px")

def unique_color_analysis():
    """
    Bardziej szczegółowa analiza - na każdej klatce w okręgu  
    sprawdza histogram kolorów i szuka unikalnych cech rybki.
    """
    entries = parse_log()
    
    # Weź klatki z rybką w red phase
    red_fish = [e for e in entries if e[2] == "red" and e[3] is not None][:10]
    # I bez rybki w red phase (dla porównania)
    red_nofish = [e for e in entries if e[2] == "red" and e[3] is None][:10]
    # White z rybką
    white_fish = [e for e in entries if e[2] == "white" and e[3] is not None][:10]
    # White bez rybki
    white_nofish = [e for e in entries if e[2] == "white" and e[3] is None][:10]
    
    print("\n=== Unikalna analiza kolorów ===")
    
    for label, frames in [
        ("RED z rybką", red_fish),
        ("RED bez rybki", red_nofish),
        ("WHITE z rybką", white_fish),  
        ("WHITE bez rybki", white_nofish),
    ]:
        print(f"\n--- {label} ({len(frames)} klatek) ---")
        all_h = []
        all_s = []
        all_v = []
        
        for entry in frames:
            fnum = entry[0]
            fx, fy = entry[3], entry[4]
            frame_path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
            img = cv2.imread(frame_path)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_img, w_img = img.shape[:2]
            
            # Piksele wewnątrz okręgu
            mask_circle = np.zeros((h_img, w_img), np.uint8)
            cv2.circle(mask_circle, (CX, CY), CR, 255, -1)
            
            circle_pixels = hsv[mask_circle > 0]
            
            # Histogram H (0-179)
            h_vals = circle_pixels[:, 0]
            s_vals = circle_pixels[:, 1]
            v_vals = circle_pixels[:, 2]
            
            all_h.extend(h_vals.tolist())
            all_s.extend(s_vals.tolist())
            all_v.extend(v_vals.tolist())
        
        if all_h:
            h_arr = np.array(all_h)
            s_arr = np.array(all_s)
            v_arr = np.array(all_v)
            
            # Histogram H z binami co 10
            h_hist, _ = np.histogram(h_arr, bins=18, range=(0, 180))
            print(f"  H histogram (bins 0-180, step 10):")
            for i, count in enumerate(h_hist):
                pct = 100*count/len(h_arr)
                bar = "#" * int(pct * 2)
                print(f"    H {i*10:3d}-{(i+1)*10:3d}: {count:6d} ({pct:5.1f}%) {bar}")
            
            # Top S bins
            s_hist, _ = np.histogram(s_arr, bins=10, range=(0, 256))
            print(f"  S histogram (bins 0-256, step 25.6):")
            for i, count in enumerate(s_hist):
                pct = 100*count/len(s_arr)
                if pct > 2:
                    print(f"    S {i*25.6:.0f}-{(i+1)*25.6:.0f}: {count:6d} ({pct:5.1f}%)")

if __name__ == "__main__":
    entries = parse_log()
    find_fish_color_range(entries)
    scan_all_colors()
    detailed_pixel_scan()
    unique_color_analysis()
