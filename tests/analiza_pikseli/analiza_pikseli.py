"""
Głęboka analiza pikseli - szuka czym rybka się NAPRAWDĘ różni od tła.
Tworzy heat map z wielu klatek i próbuje znaleźć wzorzec.
"""
import cv2
import numpy as np
import os
import math

DIAG_DIR = "diagnostyka_live"
CX, CY, CR = 140, 137, 64

def parse_log():
    entries = []
    with open(os.path.join(DIAG_DIR, "log.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fnum = int(parts[0][1:])
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


def pixel_comparison():
    """
    Porównuje piksele w miejscu wykrytej rybki vs losowe miejsce w okręgu.
    """
    entries = parse_log()
    red_fish = [e for e in entries if e[2] == "red" and e[3] is not None]
    white_fish = [e for e in entries if e[2] == "white" and e[3] is not None]
    
    print("=== Porównanie pikseli: rybka vs tło ===\n")
    
    for label, frames in [("WHITE", white_fish[:5]), ("RED", red_fish[:10])]:
        print(f"\n--- {label} ---")
        for fnum, t, color, fx, fy in frames:
            path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
            img = cv2.imread(path)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h_img, w_img = img.shape[:2]
            
            # Piksele w miejscu rybki (3x3)
            r = 2
            fish_bgr = img[max(0,fy-r):min(h_img,fy+r), max(0,fx-r):min(w_img,fx+r)]
            fish_hsv = hsv[max(0,fy-r):min(h_img,fy+r), max(0,fx-r):min(w_img,fx+r)]
            fish_gray_val = gray[fy, fx] if 0<=fy<h_img and 0<=fx<w_img else 0
            
            # Tło - punkt naprzeciwko rybki (lustro względem centrum)
            bx = 2*CX - fx
            by = 2*CY - fy
            bx = max(r, min(w_img-r, bx))
            by = max(r, min(h_img-r, by))
            bg_bgr = img[by-r:by+r, bx-r:bx+r]
            bg_hsv = hsv[by-r:by+r, bx-r:bx+r]
            bg_gray_val = gray[by, bx] if 0<=by<h_img and 0<=bx<w_img else 0
            
            fh = fish_hsv[:,:,0].mean() if fish_hsv.size else 0
            fs = fish_hsv[:,:,1].mean() if fish_hsv.size else 0
            fv = fish_hsv[:,:,2].mean() if fish_hsv.size else 0
            bh = bg_hsv[:,:,0].mean() if bg_hsv.size else 0
            bs = bg_hsv[:,:,1].mean() if bg_hsv.size else 0
            bv = bg_hsv[:,:,2].mean() if bg_hsv.size else 0
            
            fb = fish_bgr[:,:,0].mean() if fish_bgr.size else 0
            fg = fish_bgr[:,:,1].mean() if fish_bgr.size else 0
            fr = fish_bgr[:,:,2].mean() if fish_bgr.size else 0
            bb = bg_bgr[:,:,0].mean() if bg_bgr.size else 0
            bg = bg_bgr[:,:,1].mean() if bg_bgr.size else 0
            br = bg_bgr[:,:,2].mean() if bg_bgr.size else 0
            
            print(f"  f{fnum:04d} fish({fx},{fy}):")
            print(f"    Rybka HSV: H={fh:.0f} S={fs:.0f} V={fv:.0f}  BGR: B={fb:.0f} G={fg:.0f} R={fr:.0f} gray={fish_gray_val}")
            print(f"    Tło   HSV: H={bh:.0f} S={bs:.0f} V={bv:.0f}  BGR: B={bb:.0f} G={bg:.0f} R={br:.0f} gray={bg_gray_val}")
            print(f"    Diff : dH={abs(fh-bh):.0f} dS={abs(fs-bs):.0f} dV={abs(fv-bv):.0f} dGray={abs(fish_gray_val-bg_gray_val)}")


def motion_heatmap():
    """
    Tworzy heatmapę ruchu - akumuluje diffs z wielu klatek. 
    Powinno pokazać szlak rybki.
    """
    entries = parse_log()
    
    # Klatki z aktywną grą
    game_frames = [e for e in entries if e[2] in ("white", "red")]
    
    print(f"\n=== Motion heatmap z {len(game_frames)} klatek gry ===")
    
    prev_gray = None
    heat = None
    
    for fnum, t, color, fx, fy in game_frames:
        path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if heat is None:
            heat = np.zeros(gray.shape, dtype=np.float64)
        
        if prev_gray is not None and gray.shape == prev_gray.shape:
            diff = cv2.absdiff(gray, prev_gray).astype(np.float64)
            heat += diff
        
        prev_gray = gray.copy()
    
    if heat is not None:
        # Normalizuj i zapisz
        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
        
        # Nałóż okrąg
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        cv2.circle(heat_color, (CX, CY), CR, (255, 255, 255), 1)
        
        cv2.imwrite("heatmap_motion.png", heat_color)
        print("  Zapisano: heatmap_motion.png")
        
        # Sprawdź max motion wewnątrz okręgu
        mask = np.zeros(heat.shape, np.uint8)
        cv2.circle(mask, (CX, CY), CR, 255, -1)
        
        inside = heat[mask > 0]
        outside = heat[mask == 0]
        print(f"  Ruch wewnątrz okręgu: mean={inside.mean():.1f}, max={inside.max():.1f}")
        print(f"  Ruch poza okręgiem:   mean={outside.mean():.1f}, max={outside.max():.1f}")


def background_model():
    """
    Tworzy model tła z median pikseli wielu klatek białej fazy.
    Potem odejmuje od każdej klatki i szuka rybki.
    """
    entries = parse_log()
    white_frames = [e for e in entries if e[2] == "white"]
    
    print(f"\n=== Model tła z {len(white_frames)} klatek WHITE ===")
    
    # Zbierz klatki białe
    imgs = []
    for fnum, t, color, fx, fy in white_frames[:30]:  # max 30 klatek
        path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
        img = cv2.imread(path)
        if img is not None:
            imgs.append(img)
    
    if len(imgs) < 3:
        print("  Za mało klatek!")
        return
    
    # Median tło
    stack = np.stack(imgs, axis=0)
    bg_median = np.median(stack, axis=0).astype(np.uint8)
    cv2.imwrite("background_white.png", bg_median)
    print(f"  Tło z {len(imgs)} klatek -> background_white.png")
    
    # Testuj odejmowanie tła na klatkach z rybką
    bg_gray = cv2.cvtColor(bg_median, cv2.COLOR_BGR2GRAY)
    
    white_fish = [e for e in entries if e[2] == "white" and e[3] is not None]
    red_fish = [e for e in entries if e[2] == "red" and e[3] is not None]
    
    print("\n  Odejmowanie tła od klatek z rybką:")
    
    for label, frames in [("WHITE", white_fish[:5]), ("RED", red_fish[:10])]:
        print(f"\n  --- {label} ---")
        for fnum, t, color, fx, fy in frames:
            path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if gray.shape != bg_gray.shape:
                continue
            
            diff = cv2.absdiff(gray, bg_gray)
            
            # Maska okręgu
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (CX, CY), CR, 255, -1)
            
            # Progowanie
            for thresh in [15, 25, 35, 50]:
                _, binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
                binary_circle = cv2.bitwise_and(binary, mask)
                count = cv2.countNonZero(binary_circle)
                
                # Znajdź kontury
                contours, _ = cv2.findContours(binary_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                biggest = 0
                bx, by = 0, 0
                for c in contours:
                    a = cv2.contourArea(c)
                    if a > biggest:
                        biggest = a
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            bx = int(M["m10"]/M["m00"])
                            by = int(M["m01"]/M["m00"])
                
                if thresh == 25:
                    print(f"    f{fnum:04d} fish({fx},{fy}) thresh={thresh}: {count}px, "
                          f"biggest={biggest:.0f}px at ({bx},{by}), "
                          f"dist={math.sqrt((bx-fx)**2+(by-fy)**2):.0f}px")
            
            # Zapisz diff wizualizację dla kilku klatek
            if fnum in [184, 185, 186, 198, 204, 342]:
                diff_vis = cv2.applyColorMap(diff * 3, cv2.COLORMAP_JET)
                cv2.circle(diff_vis, (CX, CY), CR, (255, 255, 255), 1)
                if fx and fy:
                    cv2.drawMarker(diff_vis, (fx, fy), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
                cv2.imwrite(f"diff_from_bg_{fnum:04d}_{color}.png", diff_vis)
    
    # Teraz testuj na klatkach RED (gdzie rybka jest inna niż tło białe)
    red_frames = [e for e in entries if e[2] == "red"]
    red_imgs = []
    for fnum, t, color, fx, fy in red_frames[:30]:
        path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
        img = cv2.imread(path)
        if img is not None:
            red_imgs.append(img)
    
    if len(red_imgs) >= 3:
        red_stack = np.stack(red_imgs, axis=0)
        bg_red = np.median(red_stack, axis=0).astype(np.uint8)
        cv2.imwrite("background_red.png", bg_red)
        print(f"\n  Tło RED z {len(red_imgs)} klatek -> background_red.png")
        
        bg_red_gray = cv2.cvtColor(bg_red, cv2.COLOR_BGR2GRAY)
        
        print("\n  Odejmowanie tła RED od klatek z rybką:")
        for fnum, t, color, fx, fy in red_fish[:10]:
            path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.shape != bg_red_gray.shape:
                continue
            
            diff = cv2.absdiff(gray, bg_red_gray)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (CX, CY), CR, 255, -1)
            
            for thresh in [15, 25, 35]:
                _, binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
                binary_circle = cv2.bitwise_and(binary, mask)
                count = cv2.countNonZero(binary_circle)
                
                contours, _ = cv2.findContours(binary_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                biggest = 0
                bx, by = 0, 0
                for c in contours:
                    a = cv2.contourArea(c)
                    if a > biggest:
                        biggest = a
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            bx = int(M["m10"]/M["m00"])
                            by = int(M["m01"]/M["m00"])
                
                if thresh == 25:
                    print(f"    f{fnum:04d} fish({fx},{fy}) thresh={thresh}: {count}px, "
                          f"biggest={biggest:.0f}px at ({bx},{by}), "
                          f"dist={math.sqrt((bx-fx)**2+(by-fy)**2):.0f}px")
            
            # Zapisz wizualizację
            diff_vis = cv2.applyColorMap(diff * 3, cv2.COLORMAP_JET)
            cv2.circle(diff_vis, (CX, CY), CR, (255, 255, 255), 1)
            if fx and fy:
                cv2.drawMarker(diff_vis, (fx, fy), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
            cv2.imwrite(f"diff_from_redBG_{fnum:04d}.png", diff_vis)


def edge_detection_test():
    """
    Testuje detekcję krawędzi (Canny) - rybka powinna mieć wyraźne krawędzie.
    """
    entries = parse_log()
    
    print("\n=== Test detekcji krawędzi ===")
    
    for fnum_idx, entry in enumerate([e for e in entries if e[2] in ("white", "red")][:20]):
        fnum, t, color, fx, fy = entry
        path = os.path.join(DIAG_DIR, f"frame_{fnum:04d}.png")
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur + Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Maska okręgu (bez krawędzi)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, (CX, CY), CR-3, 255, -1)
        
        edges_circle = cv2.bitwise_and(edges, mask)
        edge_count = cv2.countNonZero(edges_circle)
        
        # Kontury z krawędzi
        contours, _ = cv2.findContours(edges_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fish_str = f"({fx},{fy})" if fx else "None"
        print(f"  f{fnum:04d} [{color:5s}] fish={fish_str:12s} edges={edge_count:4d}px contours={len(contours)}")
        
        # Zapisz kilka
        if fnum_idx < 5:
            cv2.imwrite(f"edges_{fnum:04d}_{color}.png", edges_circle)


if __name__ == "__main__":
    pixel_comparison()
    motion_heatmap()
    background_model()
    edge_detection_test()
