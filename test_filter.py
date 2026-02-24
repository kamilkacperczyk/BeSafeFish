"""
Test filtra HIT/MISS na zapisanych klatkach z testow 8b i 8c.
Sprawdza czy filtr poprawnie odrzuca napisy i przepuszcza rybke.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.fishing_detector import FishingDetector
from src.config import CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS


def test_filter_on_frame(detector, path, expected, label):
    """
    Testuje filtr na jednej klatce.
    expected: 'reject' (napis → ma odrzucić) lub 'accept' (ryba → ma wykryć)
    """
    img = cv2.imread(path)
    if img is None:
        print(f"  [SKIP] {label}: nie mogę wczytać {path}")
        return None

    # Symuluj detekcję: reset + kilka klatek warmup + testowa klatka
    detector._frame_buffer.clear()
    detector._bg_cache = None
    detector._bg_phase = "red"
    detector._fish_history = []
    detector._prev_gray = None
    detector._last_bgr = img

    # Test _has_text_overlay
    has_text = detector._has_text_overlay(img)

    # Test na szarym obrazie z fikcyjnym bg
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sprawdź co by filtr zrobił
    if expected == 'reject':
        status = "✓ PASS" if has_text else "✗ FAIL"
        icon = "🛡️" if has_text else "⚠️"
    else:  # accept
        status = "✓ PASS" if not has_text else "✗ FAIL"
        icon = "🐟" if not has_text else "⚠️"

    print(f"  {status} | {label:<30} | text_overlay={has_text} | expected={expected}")
    return status.startswith("✓")


def test_contour_filter(detector, path, expected, label):
    """
    Testuje filtr konturów korzystając z tła stworzonego z mediany.
    """
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Stwórz sztuczne tło (mediana klatki)
    bg = np.median(gray).astype(np.uint8)
    bg_img = np.full_like(gray, bg)

    diff = cv2.absdiff(gray, bg_img)
    _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    mask = np.zeros(binary.shape, np.uint8)
    cv2.circle(mask, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS - 5, 255, -1)
    binary = cv2.bitwise_and(binary, mask)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary[:25, :] = 0
    binary[-20:, :] = 0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detector._last_bgr = img
    
    any_text = False
    for c in contours:
        area = cv2.contourArea(c)
        if area > detector.FISH_MIN_AREA:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                is_text = detector._is_text_contour(gray, img, cx, cy, c)
                x, y, w, h = cv2.boundingRect(c)
                if is_text:
                    any_text = True
                    reason = f"area={area:.0f} bbox={w}x{h} aspect={w/h:.1f}"
                    print(f"    → kontur odrzucony: {reason}")

    if expected == 'reject':
        status = "✓ PASS" if any_text else "✗ FAIL"
    else:
        status = "✓ PASS" if not any_text else "✗ FAIL"

    print(f"  {status} | {label:<30} | contour_text={any_text} | expected={expected}")
    return status.startswith("✓")


def main():
    detector = FishingDetector()

    # Klatki BAD (powinny byc ODRZUCONE = reject)
    bad_frames = [
        ("test8b_miss/frames/frame_0461_red.png", "reject", "BAD-MISS 8b#461"),
        ("test8b_miss/frames/frame_0466_red.png", "reject", "BAD-MISS 8b#466"),
        ("test8c_hit/frames/frame_0241_red.png", "reject", "BAD-HIT 8c#241"),
        ("test8c_hit/frames/frame_0246_red.png", "reject", "BAD-HIT 8c#246"),
        ("test8c_hit/frames/frame_0251_red.png", "reject", "BAD-HIT 8c#251"),
    ]

    # Klatki OK (powinny byc ZAAKCEPTOWANE = accept)
    ok_frames = [
        ("test8a_tracking/frames/frame_0006_red.png", "accept", "OK-ryba 8a#6"),
        ("test8a_tracking/frames/frame_0011_red.png", "accept", "OK-ryba 8a#11"),
        ("test8c_hit/frames/frame_0046_red.png", "accept", "OK-ryba 8c#46"),
        ("test8c_hit/frames/frame_0076_red.png", "accept", "OK-ryba 8c#76"),
        ("test8c_hit/frames/frame_0116_red.png", "accept", "OK-ryba 8c#116"),
        ("test8c_hit/frames/frame_0121_red.png", "accept", "OK-ryba 8c#121"),
        ("test8c_hit/frames/frame_0126_red.png", "accept", "OK-ryba 8c#126"),
        ("test8c_hit/frames/frame_0236_red.png", "accept", "OK-ryba 8c#236"),
        ("test8c_hit/frames/frame_0441_red.png", "accept", "OK-ryba 8c#441"),
    ]

    all_frames = bad_frames + ok_frames

    print("=" * 70)
    print("  TEST FILTRA _has_text_overlay() (wykrywanie zoltego tekstu HIT)")
    print("=" * 70)
    
    pass_count = 0
    total = 0
    for path, expected, label in all_frames:
        if os.path.exists(path):
            result = test_filter_on_frame(detector, path, expected, label)
            if result is not None:
                total += 1
                if result:
                    pass_count += 1

    print(f"\n  Wynik: {pass_count}/{total} PASS ({100*pass_count/total:.0f}%)")

    print("\n" + "=" * 70)
    print("  TEST FILTRA _is_text_contour() (analiza ksztaltu/jasnosci)")
    print("=" * 70)

    pass_count2 = 0
    total2 = 0
    for path, expected, label in all_frames:
        if os.path.exists(path):
            result = test_contour_filter(detector, path, expected, label)
            if result is not None:
                total2 += 1
                if result:
                    pass_count2 += 1

    print(f"\n  Wynik: {pass_count2}/{total2} PASS ({100*pass_count2/total2:.0f}%)")

    print("\n" + "=" * 70)
    print("  PODSUMOWANIE")
    print("=" * 70)
    print(f"  Filtr text_overlay:  {pass_count}/{total}")
    print(f"  Filtr text_contour:  {pass_count2}/{total2}")
    combined = pass_count + pass_count2
    combined_total = total + total2
    print(f"  Łącznie:             {combined}/{combined_total} ({100*combined/combined_total:.0f}%)")


if __name__ == "__main__":
    main()
