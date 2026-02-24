"""
Modul wykrywania stanu minigry lowienia ryb.

Analizuje screenshot okienka "Lowienie" i okresla:
- Czy minigra jest aktywna (czy widac okienko)
- Czy okrag jest bialy (rybka poza) czy czerwony (rybka w srodku)
- Gdzie jest rybka (background subtraction z running median)

Metoda detekcji koloru: zliczanie jasnych/bialych pikseli.
Bialy okrag ma jasny kontur (~869 px z S<40, V>220).
Czerwony okrag ma ciemniejszy kontur (~192 px z S<40, V>220).

Metoda detekcji rybki: background subtraction.
Utrzymujemy bufor ostatnich N klatek dla kazdej fazy (white/red).
Median tych klatek = tlo (rybka sie rusza, wiec usrednia sie).
Odejmujemy tlo od biezacej klatki -> rybka widoczna jako blob.
Diagnostyka potwierdzila 1-5px dokladnosc tej metody.
"""

import numpy as np
import cv2
import time
import collections

from src.config import (
    WHITE_BRIGHT_S_MAX,
    WHITE_BRIGHT_V_MIN,
    WHITE_CIRCLE_PIXEL_THRESHOLD,
    FISHING_ACTIVE_BRIGHT_THRESHOLD,
    CIRCLE_CENTER_X,
    CIRCLE_CENTER_Y,
    CIRCLE_RADIUS,
    CIRCLE_SAFE_MARGIN,
    TEXT_YELLOW_H_MIN,
    TEXT_YELLOW_H_MAX,
    TEXT_YELLOW_S_MIN,
    TEXT_YELLOW_V_MIN,
    TEXT_YELLOW_THRESHOLD,
    TEXT_BRIGHT_V_MIN,
    TEXT_BRIGHT_THRESHOLD,
    TEXT_MISS_H_MIN,
    TEXT_MISS_H_MAX,
    TEXT_MISS_S_MAX,
    TEXT_MISS_V_MIN,
    TEXT_MISS_THRESHOLD,
    TEXT_MISS_SAT_MAX,
    TEXT_MISS_LOW_SAT_RATIO,
)
import math


class FishingDetector:
    """Analizuje okienko minigry lowienia i wykrywa stan gry."""

    # Parametry background subtraction
    BG_BUFFER_SIZE = 15      # ile klatek pamietac do mediany
    BG_MIN_FRAMES = 3        # min klatek do obliczenia tla
    BG_RECOMPUTE_EVERY = 3   # przelicz mediane co N klatek (oszczednosc CPU)
    BG_DIFF_THRESHOLD = 25   # prog roznicy piksel vs tlo
    FISH_MIN_AREA = 30       # min rozmiar bloba rybki (px)
    FISH_MAX_JUMP = 50       # max skok pozycji miedzy klatkami (px)

    def __init__(self):
        # Background subtraction
        self._frame_buffer = collections.deque(maxlen=self.BG_BUFFER_SIZE)
        self._bg_cache = None         # ostatnia obliczona mediana
        self._bg_phase = None         # faza w ktorej budowalismy tlo
        self._frames_since_recompute = 0

        # Ostatnia klatka BGR (potrzebna do analizy koloru konturow)
        self._last_bgr = None

        # Fallback: frame differencing (pierwsze klatki)
        self._prev_gray = None

        # Historia pozycji rybki do predykcji
        self._fish_history = []  # lista (timestamp, x, y)
        self._max_history = 5    # ile ostatnich pozycji pamietac

        # Filtr statycznej pozycji (MISS tekst stoi w miejscu, rybka sie rusza)
        self._stale_pos = None       # (x, y) ostatnia pozycja
        self._stale_count = 0        # ile klatek z ta sama pozycja
        self.STALE_MAX_DIST = 3      # max odleglosc zeby uznac za "ta sama" pozycje
        self.STALE_FRAMES_LIMIT = 3  # po tylu klatkach odrzuc jako tekst

    def _count_bright_white_pixels(self, fishing_frame: np.ndarray) -> int:
        """Liczy piksele prawie biale (niski S, wysoki V) - kontur bialego okregu."""
        hsv = cv2.cvtColor(fishing_frame, cv2.COLOR_BGR2HSV)
        # S < WHITE_BRIGHT_S_MAX  i  V > WHITE_BRIGHT_V_MIN
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, WHITE_BRIGHT_V_MIN]),
            np.array([179, WHITE_BRIGHT_S_MAX, 255]),
        )
        return cv2.countNonZero(mask)

    def _count_bright_pixels(self, fishing_frame: np.ndarray) -> int:
        """Liczy jasne piksele (grayscale > 200) - wskaznik aktywnosci minigry."""
        gray = cv2.cvtColor(fishing_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return cv2.countNonZero(thresh)

    def _has_text_overlay(self, frame_bgr: np.ndarray) -> bool:
        """
        Sprawdza czy w okregu widac napis HIT (zolty tekst).

        Napis HIT ma charakterystyczny zolty kolor (H=15-45, S>80, V>150).
        Detekcja MISS odbywa sie na poziomie konturow (_is_text_contour)
        bo piksele lavender sa zbyt podobne do tla wody w pelnej klatce.

        Returns:
            True jesli wykryto napis tekstowy (HIT)
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Maska okregu
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, 255, -1)

        # Szukaj zoltych pikseli (napis HIT)
        yellow_mask = cv2.inRange(
            hsv,
            np.array([TEXT_YELLOW_H_MIN, TEXT_YELLOW_S_MIN, TEXT_YELLOW_V_MIN]),
            np.array([TEXT_YELLOW_H_MAX, 255, 255]),
        )
        yellow_in_circle = cv2.bitwise_and(yellow_mask, mask)
        num_yellow = cv2.countNonZero(yellow_in_circle)

        return num_yellow >= TEXT_YELLOW_THRESHOLD

    def _is_text_contour(self, gray: np.ndarray, frame_bgr: np.ndarray,
                         cx: int, cy: int, contour) -> bool:
        """
        Sprawdza czy wykryty kontur to napis (HIT/MISS) a nie rybka.

        Kryteria (konserwatywne - wolimy przepuscic tekst niz odrzucic rybke):
        1. Kontur musi byc dostatecznie duzy (area > 150) - male to szum
        2. Jasne piksele stanowia > 15% regionu - tekst jest jasny
        3. LUB aspect ratio > 2.5 z szerokosc > 30 - tekst jest szeroki

        Args:
            gray: klatka grayscale
            frame_bgr: klatka BGR (do analizy koloru)
            cx, cy: centroid konturu
            contour: kontur OpenCV

        Returns:
            True jesli kontur wyglada jak tekst (odrzucic!)
        """
        area = cv2.contourArea(contour)

        # Male kontury to prawdopodobnie szum, nie tekst
        if area < 150:
            return False

        x, y, w, h = cv2.boundingRect(contour)

        # Kryterium 1: aspect ratio - tekst jest szeroki i plaski
        aspect = w / h if h > 0 else 0
        if aspect > 3.0 and w > 40:
            return True

        # Region wokol bloba
        pad = 3
        y1 = max(0, y - pad)
        y2 = min(gray.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(gray.shape[1], x + w + pad)

        region_gray = gray[y1:y2, x1:x2]
        if region_gray.size == 0:
            return False

        # Kryterium 2: procent jasnych pikseli w regionie
        num_bright = np.count_nonzero(region_gray > TEXT_BRIGHT_V_MIN)
        bright_ratio = num_bright / region_gray.size
        if bright_ratio > 0.15 and num_bright >= TEXT_BRIGHT_THRESHOLD:
            return True

        # Kryterium 3: saturacja jasnych pikseli — tekst MISS vs rybka
        # Tekst MISS: jasne piksele (V>=170) z niska saturacja (S < 120)
        # Rybka: ciemna (malo jasnych px) lub woda z wysoka S
        # Oparte na analizie 12 przykladow MISS (miss1-12.png)
        if area >= 100:
            region_bgr = frame_bgr[y1:y2, x1:x2]
            if region_bgr.size > 0:
                region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
                bright_v = region_hsv[:, :, 2] >= 170
                n_bright = np.count_nonzero(bright_v)
                if n_bright >= 25:
                    bright_sat = region_hsv[bright_v, 1]
                    low_sat = np.count_nonzero(bright_sat < TEXT_MISS_SAT_MAX)
                    if low_sat / n_bright > TEXT_MISS_LOW_SAT_RATIO:
                        return True

        return False

    @staticmethod
    def _is_red_blob(hsv_frame: np.ndarray, cx: int, cy: int, radius: int = 4) -> bool:
        """
        Sprawdza czy blob w danym miejscu jest czerwony (MISS tekst / okrag).
        Czerwone elementy: H < 15 lub H > 165, S > 120.
        Rybka jest oliwkowa/szara: H ~ 30-50, S ~ 200+.
        """
        h_img, w_img = hsv_frame.shape[:2]
        y1 = max(0, cy - radius)
        y2 = min(h_img, cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(w_img, cx + radius)
        region = hsv_frame[y1:y2, x1:x2]
        if region.size == 0:
            return True  # pusty region — odrzuc
        avg_h = region[:, :, 0].mean()
        avg_s = region[:, :, 1].mean()
        # Czerwony: H < 15 lub H > 165, z saturation > 120
        is_red = avg_s > 120 and (avg_h < 15 or avg_h > 165)
        return is_red

    def _recompute_background(self):
        """Oblicza mediane z bufora klatek jako model tla."""
        if len(self._frame_buffer) < self.BG_MIN_FRAMES:
            return None
        stack = np.stack(list(self._frame_buffer))
        self._bg_cache = np.median(stack, axis=0).astype(np.uint8)
        self._frames_since_recompute = 0
        return self._bg_cache

    def _find_fish_bg_subtraction(self, gray: np.ndarray) -> tuple:
        """
        Znajduje rybke przez odejmowanie tla (background subtraction).

        1. Mediana ostatnich N klatek = tlo (rybka sie rusza -> usrednia)
        2. |biezaca - tlo| = roznica -> rybka jako blob
        3. Prog + circle mask -> kontur
        4. Biggest blob = rybka

        Returns:
            (x, y) pozycja rybki lub None
        """
        bg = self._bg_cache
        if bg is None:
            bg = self._recompute_background()
        if bg is None:
            return None

        if gray.shape != bg.shape:
            return None

        # Odejmij tlo
        diff = cv2.absdiff(gray, bg)
        _, binary = cv2.threshold(diff, self.BG_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Maska okregu (bez krawedzi zeby uniknac artefaktow)
        mask = np.zeros(binary.shape, np.uint8)
        cv2.circle(mask, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y),
                   CIRCLE_RADIUS - 5, 255, -1)
        binary = cv2.bitwise_and(binary, mask)

        # Oczyszczenie
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Wytnij UI (gora/dol)
        binary[:25, :] = 0
        binary[-20:, :] = 0

        # Znajdz kontury
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Biggest blob = rybka (z filtrem na napisy HIT/MISS)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.FISH_MIN_AREA:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    candidates.append((area, cx, cy, c))

        # Sortuj malejaco po area
        candidates.sort(key=lambda c: c[0], reverse=True)

        for area, cx, cy, contour in candidates:
            # Filtr: sprawdz czy kontur to tekst HIT/MISS
            if self._is_text_contour(gray, self._last_bgr, cx, cy, contour):
                continue  # pomija napis, sprawdz nastepny kontur
            return (cx, cy)

        return None

    def _find_fish_frame_diff(self, gray: np.ndarray, circle_color: str) -> tuple:
        """
        Fallback: detekcja ruchu (frame differencing).
        Uzywane tylko w pierwszych klatkach zanim bg model jest gotowy.
        """
        if self._prev_gray is None:
            return None

        h = min(gray.shape[0], self._prev_gray.shape[0])
        w = min(gray.shape[1], self._prev_gray.shape[1])
        diff = cv2.absdiff(gray[:h, :w], self._prev_gray[:h, :w])

        _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        total_motion = cv2.countNonZero(motion_mask)
        if total_motion > 1500:
            return None

        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask[:25, :] = 0
        motion_mask[-20:, :] = 0

        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if 20 < area < 1000:
                x, y, bw, bh = cv2.boundingRect(c)
                if bw < 80 and bh < 80:
                    cx, cy = x + bw // 2, y + bh // 2
                    dist = math.sqrt((cx - CIRCLE_CENTER_X)**2 + (cy - CIRCLE_CENTER_Y)**2)
                    if dist <= CIRCLE_RADIUS + 5:
                        candidates.append((area, cx, cy))

        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            return (candidates[0][1], candidates[0][2])
        return None

    def find_fish_position(self, fishing_frame: np.ndarray, circle_color: str = "white") -> tuple:
        """
        Znajduje pozycje rybki w okienku lowienia.

        Metoda glowna: BACKGROUND SUBTRACTION
        - Utrzymujemy mediane ostatnich N klatek jako model tla
        - Odejmujemy tlo od biezacej klatki
        - Rybka widoczna jako blob (potwierdzone diagnostyka: 1-5px dokladnosc)

        Fallback: frame differencing (pierwsze kilka klatek nowej fazy)

        Args:
            fishing_frame: screenshot okienka lowienia (BGR numpy array)
            circle_color: aktualny kolor okregu ("white", "red", "none")

        Returns:
            (x, y) - pozycja rybki wzgledem okienka lowienia,
            None jesli nie znaleziono rybki
        """
        gray = cv2.cvtColor(fishing_frame, cv2.COLOR_BGR2GRAY)
        self._last_bgr = fishing_frame  # zachowaj do analizy koloru konturow

        # Faza "none" -> nie szukamy rybki, nie aktualizujemy modelu
        if circle_color == "none":
            self._prev_gray = gray.copy()
            return None

        # --- FILTR NAPISOW HIT/MISS ---
        # Sprawdz czy w klatce jest napis HIT (zolty tekst)
        # Jesli tak, nie szukaj rybki — to falszywka
        if self._has_text_overlay(fishing_frame):
            self._prev_gray = gray.copy()
            # Dodaj klatke do bufora mimo wszystko
            # (napis zniknie, a mediana go usredni)
            self._frame_buffer.append(gray.copy())
            self._frames_since_recompute += 1
            if self._frames_since_recompute >= self.BG_RECOMPUTE_EVERY:
                self._recompute_background()
            return None

        # Zmiana fazy (white -> red lub red -> white)?
        # Reset modelu tla - nowa faza = inne tlo
        if circle_color != self._bg_phase:
            self._frame_buffer.clear()
            self._bg_cache = None
            self._bg_phase = circle_color
            self._frames_since_recompute = 0
            # Reset historii rybki — pozycje z innej fazy nie sa wiarygodne
            self._fish_history = []
            self._stale_pos = None
            self._stale_count = 0

        # Dodaj klatke do bufora
        self._frame_buffer.append(gray.copy())
        self._frames_since_recompute += 1

        # Przelicz mediane jesli potrzeba
        if self._frames_since_recompute >= self.BG_RECOMPUTE_EVERY:
            self._recompute_background()

        # --- DETEKCJA RYBKI ---
        fish_pos = None

        if len(self._frame_buffer) >= self.BG_MIN_FRAMES and self._bg_cache is not None:
            # Metoda glowna: background subtraction
            fish_pos = self._find_fish_bg_subtraction(gray)
        else:
            # Fallback: frame differencing (warmup)
            fish_pos = self._find_fish_frame_diff(gray, circle_color)

        # Walidacja: max skok pozycji
        if fish_pos is not None and self._fish_history:
            _, last_x, last_y = self._fish_history[-1]
            dx = abs(fish_pos[0] - last_x)
            dy = abs(fish_pos[1] - last_y)
            if dx > self.FISH_MAX_JUMP or dy > self.FISH_MAX_JUMP:
                fish_pos = None

        # Filtr statycznej pozycji (tekst MISS/HIT stoi w miejscu, rybka sie rusza)
        if fish_pos is not None:
            if self._stale_pos is not None:
                dist = math.sqrt(
                    (fish_pos[0] - self._stale_pos[0]) ** 2
                    + (fish_pos[1] - self._stale_pos[1]) ** 2
                )
                if dist <= self.STALE_MAX_DIST:
                    self._stale_count += 1
                else:
                    self._stale_pos = fish_pos
                    self._stale_count = 1
            else:
                self._stale_pos = fish_pos
                self._stale_count = 1

            if self._stale_count >= self.STALE_FRAMES_LIMIT:
                # Pozycja nie zmienila sie przez N klatek → prawdopodobnie tekst
                fish_pos = None
        else:
            # Brak detekcji — reset licznika statycznej pozycji
            self._stale_pos = None
            self._stale_count = 0

        # Zapisz do historii
        if fish_pos is not None:
            now = time.perf_counter()
            self._fish_history.append((now, fish_pos[0], fish_pos[1]))
            if len(self._fish_history) > self._max_history:
                self._fish_history.pop(0)

        self._prev_gray = gray.copy()
        return fish_pos

    def reset_tracking(self):
        """Resetuje sledzenie rybki i model tla (np. na poczatku nowej rundy)."""
        self._prev_gray = None
        self._fish_history = []
        self._frame_buffer.clear()
        self._bg_cache = None
        self._bg_phase = None
        self._frames_since_recompute = 0
        self._stale_pos = None
        self._stale_count = 0

    def predict_fish_position(self, ahead_ms: float = 50.0) -> tuple:
        """
        Przewiduje gdzie bedzie rybka za 'ahead_ms' milisekund
        na podstawie ostatnich znanych pozycji (ekstrapolacja liniowa).

        Args:
            ahead_ms: ile ms w przod przewidywac (domyslnie 50ms)

        Returns:
            (x, y) - przewidywana pozycja, lub None jesli brak danych
        """
        if len(self._fish_history) < 2:
            # Za malo danych do predykcji - zwroc ostatnia znana pozycje
            if self._fish_history:
                return (self._fish_history[-1][1], self._fish_history[-1][2])
            return None

        # Weź 2 ostatnie pozycje
        t1, x1, y1 = self._fish_history[-2]
        t2, x2, y2 = self._fish_history[-1]

        dt = t2 - t1
        if dt <= 0:
            return (x2, y2)

        # Predkosc rybki (px/s)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt

        # Ekstrapoluj pozycje
        ahead_s = ahead_ms / 1000.0
        pred_x = int(x2 + vx * ahead_s)
        pred_y = int(y2 + vy * ahead_s)

        # Ogranicz do bezpiecznego okręgu (nie wychodz poza)
        safe_r = CIRCLE_RADIUS - CIRCLE_SAFE_MARGIN
        dx = pred_x - CIRCLE_CENTER_X
        dy = pred_y - CIRCLE_CENTER_Y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > safe_r and dist > 0:
            scale = safe_r / dist
            pred_x = int(CIRCLE_CENTER_X + dx * scale)
            pred_y = int(CIRCLE_CENTER_Y + dy * scale)

        return (pred_x, pred_y)

    def detect_circle_color(self, fishing_frame: np.ndarray) -> str:
        """
        Analizuje kolor okregu w okienku lowienia.

        Metoda: liczy piksele prawie biale (S<40, V>220).
        Bialy okrag ma ich duzo (~869), czerwony malo (~192).

        Args:
            fishing_frame: screenshot okienka lowienia (BGR numpy array)

        Returns:
            'red'   - okrag czerwony (rybka w srodku - KLIKAJ!)
            'white' - okrag bialy (rybka poza - CZEKAJ)
            'none'  - nie wykryto okregu (minigra nieaktywna?)
        """
        if not self.is_fishing_active(fishing_frame):
            return "none"

        white_px = self._count_bright_white_pixels(fishing_frame)

        if white_px >= WHITE_CIRCLE_PIXEL_THRESHOLD:
            return "white"
        else:
            return "red"

    def is_fishing_active(self, fishing_frame: np.ndarray) -> bool:
        """
        Sprawdza czy minigra lowienia jest aktywna.
        Minigra aktywna = duzo jasnych pikseli (UI okienka, okrag, timer, rybka).

        Args:
            fishing_frame: screenshot okienka lowienia (BGR numpy array)

        Returns:
            True jesli minigra jest aktywna
        """
        bright_px = self._count_bright_pixels(fishing_frame)
        return bright_px >= FISHING_ACTIVE_BRIGHT_THRESHOLD

    def get_debug_info(self, fishing_frame: np.ndarray) -> dict:
        """
        Zwraca szczegolowe informacje debugowe.
        Przydatne do kalibracji progow.

        Args:
            fishing_frame: screenshot okienka lowienia (BGR numpy array)

        Returns:
            dict z informacjami o ilosciach pikseli
        """
        white_px = self._count_bright_white_pixels(fishing_frame)
        bright_px = self._count_bright_pixels(fishing_frame)
        color = self.detect_circle_color(fishing_frame)
        active = self.is_fishing_active(fishing_frame)

        # Maski do wyswietlenia (debug overlay)
        hsv = cv2.cvtColor(fishing_frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, WHITE_BRIGHT_V_MIN]),
            np.array([179, WHITE_BRIGHT_S_MAX, 255]),
        )
        gray = cv2.cvtColor(fishing_frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        return {
            "white_pixels": white_px,
            "bright_pixels": bright_px,
            "circle_color": color,
            "fishing_active": active,
            "white_mask": white_mask,
            "bright_mask": bright_mask,
        }


# --- TRYB TESTOWY ---
# Uruchom: python -m src.fishing_detector
# Pokazuje na zywo co wykrywa detektor (wymaga uruchomionej gry)
if __name__ == "__main__":
    from src.screen_capture import ScreenCapture

    print("=== Test detektora lowienia ===")
    print("Uruchom gre i rozpocznij lowienie ryb.")
    print("Nacisnij 'q' zeby zamknac.")
    print()

    capture = ScreenCapture()
    detector = FishingDetector()

    while True:
        # Zrob screenshot okienka lowienia
        fishing_frame = capture.grab_fishing_box()

        # Analizuj
        debug = detector.get_debug_info(fishing_frame)

        # Wyswietl info na klatce
        color = debug["circle_color"]
        active = debug["fishing_active"]
        white_px = debug["white_pixels"]
        bright_px = debug["bright_pixels"]

        # Kolor ramki zalezy od wykrytego stanu
        if color == "red":
            border_color = (0, 0, 255)  # czerwony
            status = "KLIKAJ!"
        elif color == "white":
            border_color = (255, 255, 255)  # bialy
            status = "Czekaj..."
        else:
            border_color = (128, 128, 128)  # szary
            status = "Brak okregu"

        # Rysuj ramke i tekst
        display = fishing_frame.copy()
        cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1), border_color, 3)
        cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
        cv2.putText(display, f"White px: {white_px}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Bright px: {bright_px}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Active: {active}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Pokaz klatke i maski
        cv2.imshow("Kosa - Fishing Box", display)
        cv2.imshow("White Mask (S<40,V>220)", debug["white_mask"])
        cv2.imshow("Bright Mask (gray>200)", debug["bright_mask"])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Zamknieto.")
