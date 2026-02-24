"""
Analiza klatek MISS - szukamy koloru tekstu MISS.
Porownujemy klatke z MISS vs klatke OK (bliska w czasie).
"""
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS


def analyze_frame(path, label):
    img = cv2.imread(path)
    if img is None:
        print(f"  [SKIP] {path}")
        return
    
    # Wytnij region okregu
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, 255, -1)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Piksele w okregu
    circle_px = mask > 0
    
    # Analiza jasnosci
    gray_vals = gray[circle_px]
    bright_220 = np.count_nonzero(gray_vals > 220)
    bright_200 = np.count_nonzero(gray_vals > 200)
    bright_180 = np.count_nonzero(gray_vals > 180)
    total = len(gray_vals)
    
    # Analiza bialych pikseli (tekst MISS moze byc bialy)
    h_val = hsv[:,:,0][circle_px]
    s_val = hsv[:,:,1][circle_px]
    v_val = hsv[:,:,2][circle_px]
    
    # Bialy: niska saturacja, wysoka jasnosc
    white_mask = (s_val < 50) & (v_val > 180)
    num_white = np.count_nonzero(white_mask)
    
    # Bardzo jasny (V > 200) i nie-czerwony
    very_bright = (v_val > 200)
    num_very_bright = np.count_nonzero(very_bright)
    
    # Srednia jasnosc w centrze okregu (gdzie tekst sie pojawia - ok. 30px od srodka)
    center_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(center_mask, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), 35, 255, -1)
    center_px = center_mask > 0
    center_gray = gray[center_px]
    center_bright = np.count_nonzero(center_gray > 180)
    center_total = len(center_gray)
    
    print(f"\n{label} ({path})")
    print(f"  Jasnosc: mean={np.mean(gray_vals):.1f} std={np.std(gray_vals):.1f}")
    print(f"  Bright >220: {bright_220} ({100*bright_220/total:.2f}%)")
    print(f"  Bright >200: {bright_200} ({100*bright_200/total:.2f}%)")
    print(f"  Bright >180: {bright_180} ({100*bright_180/total:.2f}%)")
    print(f"  Biale (S<50, V>180): {num_white} ({100*num_white/total:.2f}%)")
    print(f"  Bardzo jasne (V>200): {num_very_bright} ({100*num_very_bright/total:.3f}%)")
    print(f"  Centrum (r=35): bright>180 = {center_bright}/{center_total} ({100*center_bright/center_total:.2f}%)")
    
    # Znajdz najjasniejsze piksele - pokaz ich kolor
    brightest_mask = gray_vals > 200
    if np.any(brightest_mask):
        brightest_bgr = img[circle_px][brightest_mask]
        brightest_hsv = hsv[circle_px][brightest_mask]
        print(f"  Najjasniejsze piksele ({np.count_nonzero(brightest_mask)} szt):")
        print(f"    BGR mean: {np.mean(brightest_bgr, axis=0).astype(int)}")
        print(f"    HSV mean: {np.mean(brightest_hsv, axis=0).astype(int)}")
        print(f"    BGR range: min={np.min(brightest_bgr, axis=0)} max={np.max(brightest_bgr, axis=0)}")
    
    # Gradient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    circle_grad = gradient[circle_px]
    print(f"  Gradient: mean={np.mean(circle_grad):.1f} std={np.std(circle_grad):.1f} max={np.max(circle_grad):.1f}")
    
    # Roznica pikseli w centrze (diff z mediana calego okregu)
    median_val = np.median(gray_vals)
    center_diff = np.abs(center_gray.astype(float) - median_val)
    center_high_diff = np.count_nonzero(center_diff > 30)
    print(f"  Centrum diff od mediany: {center_high_diff}/{center_total} px z diff>30 ({100*center_high_diff/center_total:.1f}%)")


def main():
    print("=" * 70)
    print("  ANALIZA KLATEK MISS vs OK")
    print("=" * 70)
    
    # BAD-MISS klatki
    analyze_frame("test8b_miss/frames/frame_0461_red.png", "BAD-MISS #461")
    analyze_frame("test8b_miss/frames/frame_0466_red.png", "BAD-MISS #466")
    
    # OK klatki (blisko w czasie do MISS, dla porownania)
    analyze_frame("test8b_miss/frames/frame_0456_red.png", "OK-blisko #456")
    analyze_frame("test8b_miss/frames/frame_0471_red.png", "OK-blisko #471")
    
    # Standardowe OK klatki
    analyze_frame("test8a_tracking/frames/frame_0006_red.png", "OK-8a #6")
    analyze_frame("test8a_tracking/frames/frame_0011_red.png", "OK-8a #11")
    analyze_frame("test8c_hit/frames/frame_0046_red.png", "OK-8c #46")
    analyze_frame("test8c_hit/frames/frame_0126_red.png", "OK-8c #126")
    
    # Bonus: BAD-HIT dla porownania
    analyze_frame("test8c_hit/frames/frame_0241_red.png", "BAD-HIT #241")


if __name__ == "__main__":
    main()
