"""
Modul przechwytywania ekranu.

Uzywa biblioteki 'mss' do szybkiego robienia screenshotow.
mss jest ~10x szybszy niz pyautogui.screenshot() co jest kluczowe
dla szybkosci reakcji bota.
"""

import numpy as np
import mss
import cv2

from src.config import (
    GAME_WINDOW_X, GAME_WINDOW_Y,
    GAME_WINDOW_WIDTH, GAME_WINDOW_HEIGHT,
    FISHING_BOX_X, FISHING_BOX_Y,
    FISHING_BOX_WIDTH, FISHING_BOX_HEIGHT,
)


class ScreenCapture:
    """Klasa do przechwytywania fragmentow ekranu."""

    def __init__(self):
        self.sct = mss.mss()

    def grab_game_window(self) -> np.ndarray:
        """
        Robi screenshot calego okna gry.

        Returns:
            numpy array w formacie BGR (gotowy dla OpenCV)
        """
        monitor = {
            "left": GAME_WINDOW_X,
            "top": GAME_WINDOW_Y,
            "width": GAME_WINDOW_WIDTH,
            "height": GAME_WINDOW_HEIGHT,
        }
        screenshot = self.sct.grab(monitor)
        # mss zwraca BGRA, konwertujemy na BGR (standard OpenCV)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def grab_fishing_box(self) -> np.ndarray:
        """
        Robi screenshot samego okienka minigry lowienia.
        To jest mniejszy region = szybsze przetwarzanie.

        Returns:
            numpy array w formacie BGR (gotowy dla OpenCV)
        """
        monitor = {
            "left": GAME_WINDOW_X + FISHING_BOX_X,
            "top": GAME_WINDOW_Y + FISHING_BOX_Y,
            "width": FISHING_BOX_WIDTH,
            "height": FISHING_BOX_HEIGHT,
        }
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def grab_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Robi screenshot dowolnego regionu ekranu.

        Args:
            x: pozycja x lewego gornego rogu
            y: pozycja y lewego gornego rogu
            width: szerokosc regionu
            height: wysokosc regionu

        Returns:
            numpy array w formacie BGR
        """
        monitor = {
            "left": x,
            "top": y,
            "width": width,
            "height": height,
        }
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


# --- TRYB TESTOWY ---
# Uruchom: python -m src.screen_capture
# Zrobi screenshot okna gry i okienka lowienia, zapisze jako pliki PNG
# + otworzy podglad na zywo (nacisnij 'q' zeby zamknac)
if __name__ == "__main__":
    print("=== Test modulu przechwytywania ekranu ===")
    print(f"Okno gry: ({GAME_WINDOW_X}, {GAME_WINDOW_Y}) {GAME_WINDOW_WIDTH}x{GAME_WINDOW_HEIGHT}")
    print(f"Okienko lowienia: ({FISHING_BOX_X}, {FISHING_BOX_Y}) {FISHING_BOX_WIDTH}x{FISHING_BOX_HEIGHT}")
    print()

    capture = ScreenCapture()

    # Zapisz pojedynczy screenshot
    game_frame = capture.grab_game_window()
    cv2.imwrite("test_game_window.png", game_frame)
    print(f"Zapisano test_game_window.png ({game_frame.shape[1]}x{game_frame.shape[0]})")

    fishing_frame = capture.grab_fishing_box()
    cv2.imwrite("test_fishing_box.png", fishing_frame)
    print(f"Zapisano test_fishing_box.png ({fishing_frame.shape[1]}x{fishing_frame.shape[0]})")

    # Podglad na zywo
    print()
    print("Podglad na zywo - nacisnij 'q' zeby zamknac")
    while True:
        frame = capture.grab_game_window()
        # Rysujemy prostokat tam gdzie spodziewamy sie okienka lowienia
        cv2.rectangle(
            frame,
            (FISHING_BOX_X, FISHING_BOX_Y),
            (FISHING_BOX_X + FISHING_BOX_WIDTH, FISHING_BOX_Y + FISHING_BOX_HEIGHT),
            (0, 255, 0), 2  # zielona ramka
        )
        cv2.imshow("Kosa - Game Window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Zamknieto.")
