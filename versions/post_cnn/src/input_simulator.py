"""
Modul symulacji inputow (klikniecia myszy, naciskanie klawiszy).

Uzywa pydirectinput do symulowania akcji gracza.
pydirectinput wysyla DirectInput scan codes — gry (w tym Metin2)
odczytuja te klawisze, w przeciwienstwie do zwyklego pyautogui.

Przed kazda akcja fokusujemy okno gry, zeby klawisze trafialy we wlasciwe miejsce.
"""

import time
import pydirectinput
import pygetwindow as gw

from src.config import (
    GAME_WINDOW_X, GAME_WINDOW_Y,
    FISHING_BOX_X, FISHING_BOX_Y,
    FISHING_BOX_WIDTH, FISHING_BOX_HEIGHT,
    BAIT_KEY, CAST_KEY,
    POST_CLICK_DELAY, CAST_DELAY, BAIT_DELAY, ROUND_PAUSE,
)

# pydirectinput: minimalne opoznienie miedzy akcjami
# 0.02s = 20ms - wystarczajaco szybkie do szybkiego klikania
pydirectinput.PAUSE = 0.02

# Nazwa okna gry — uzywane do znajdowania i fokusowania okna.
# Szukamy okna zawierajacego ten tekst (case-insensitive).
GAME_WINDOW_TITLE = "eryndos"


def _find_game_window():
    """Znajduje okno gry po czesciowej nazwie (case-insensitive)."""
    for win in gw.getAllWindows():
        if GAME_WINDOW_TITLE.lower() in win.title.lower() and win.title.strip():
            return win
    return None


def _focus_game_window():
    """
    Znajduje okno gry i ustawia na nie fokus.
    Dzieki temu klawisze trafiaja do gry, a nie do terminala.
    """
    try:
        win = _find_game_window()
        if win:
            if win.isMinimized:
                win.restore()
            win.activate()
            time.sleep(0.1)  # daj chwile na aktywacje
            return True
        else:
            print(f"[INPUT] UWAGA: Nie znaleziono okna '{GAME_WINDOW_TITLE}'!")
            print("[INPUT] Sprawdz czy gra jest uruchomiona i czy GAME_WINDOW_TITLE jest poprawny.")
            return False
    except Exception as e:
        print(f"[INPUT] Nie udalo sie sfokusowac okna gry: {e}")
        return False


class InputSimulator:
    """Symuluje klikniecia myszy i naciskanie klawiszy w grze."""

    def __init__(self):
        # Srodek okienka lowienia (tam klikamy gdy okrag jest czerwony)
        self.fishing_center_x = GAME_WINDOW_X + FISHING_BOX_X + FISHING_BOX_WIDTH // 2
        self.fishing_center_y = GAME_WINDOW_Y + FISHING_BOX_Y + FISHING_BOX_HEIGHT // 2

    def click_fishing_box(self):
        """
        Klikniecie lewym przyciskiem myszy w srodek okienka lowienia.
        Uzywane jako fallback gdy nie znamy pozycji rybki.
        """
        _focus_game_window()
        pydirectinput.click(self.fishing_center_x, self.fishing_center_y)
        time.sleep(POST_CLICK_DELAY)

    def click_at_fish(self, fish_x: int, fish_y: int):
        """
        Klikniecie lewym przyciskiem myszy w pozycje rybki.
        Wersja WOLNA (z fokusem) - uzywana jako fallback.

        Args:
            fish_x: x pozycji rybki WZGLEDEM okienka lowienia
            fish_y: y pozycji rybki WZGLEDEM okienka lowienia
        """
        # Przelicz na wspolrzedne ekranowe
        abs_x = GAME_WINDOW_X + FISHING_BOX_X + fish_x
        abs_y = GAME_WINDOW_Y + FISHING_BOX_Y + fish_y
        _focus_game_window()
        pydirectinput.click(abs_x, abs_y)
        time.sleep(POST_CLICK_DELAY)

    def click_at_fish_fast(self, fish_x: int, fish_y: int):
        """
        Szybkie klikniecie w pozycje rybki BEZ fokusowania okna.
        Uzywane podczas aktywnej minigry (okno juz sfokusowane).
        Brak POST_CLICK_DELAY = maksymalna szybkosc.

        Args:
            fish_x: x pozycji rybki WZGLEDEM okienka lowienia
            fish_y: y pozycji rybki WZGLEDEM okienka lowienia
        """
        abs_x = GAME_WINDOW_X + FISHING_BOX_X + fish_x
        abs_y = GAME_WINDOW_Y + FISHING_BOX_Y + fish_y
        pydirectinput.click(abs_x, abs_y)

    def burst_click_at_fish(self, fish_x: int, fish_y: int, count: int = 3):
        """
        Seria szybkich klikniec w pozycje rybki.
        Uzywane w momencie przejscia bialy->czerwony okresag.
        Kliki sa tak szybkie jak to mozliwe (tylko pydirectinput.PAUSE miedzy nimi).

        Args:
            fish_x: x pozycji rybki WZGLEDEM okienka lowienia
            fish_y: y pozycji rybki WZGLEDEM okienka lowienia
            count: ile klikniec (domyslnie 3)
        """
        abs_x = GAME_WINDOW_X + FISHING_BOX_X + fish_x
        abs_y = GAME_WINDOW_Y + FISHING_BOX_Y + fish_y
        for _ in range(count):
            pydirectinput.click(abs_x, abs_y)

    def click_fishing_box_fast(self):
        """
        Szybkie klikniecie w srodek okienka BEZ fokusowania.
        Fallback gdy nie znamy pozycji rybki.
        """
        pydirectinput.click(self.fishing_center_x, self.fishing_center_y)

    def ensure_focus(self):
        """
        Fokusuje okno gry. Wywolaj raz przed rozpoczeciem minigry,
        potem uzywaj metod _fast() ktore pomijaja fokus.
        """
        _focus_game_window()

    def use_bait(self):
        """
        Nacisniecie klawisza robaka (F4).
        Zuzywa jednego robaka przed zarzuceniem wedki.
        """
        _focus_game_window()
        pydirectinput.press(BAIT_KEY)
        time.sleep(BAIT_DELAY)

    def cast_rod(self):
        """
        Nacisniecie SPACJI — zarzucenie wedki.
        Po tym rozpoczyna sie minigra lowienia.
        """
        _focus_game_window()
        pydirectinput.press(CAST_KEY)
        time.sleep(CAST_DELAY)

    def start_fishing_round(self):
        """
        Pelna sekwencja startu rundy:
        1. Fokus na okno gry
        2. Nacisnij F4 (uzyj robaka) — czekaj na animacje
        3. Nacisnij SPACJE (zarzuc wedke) — czekaj na animacje
        """
        _focus_game_window()
        print(f"[INPUT] Uzywam robaka ({BAIT_KEY.upper()})...")
        pydirectinput.press(BAIT_KEY)
        print(f"[INPUT] Czekam {BAIT_DELAY}s na animacje robaka...")
        time.sleep(BAIT_DELAY)
        print(f"[INPUT] Zarzucam wedke ({CAST_KEY.upper()})...")
        pydirectinput.press(CAST_KEY)
        print(f"[INPUT] Czekam {CAST_DELAY}s na animacje wedki...")
        time.sleep(CAST_DELAY)
        print("[INPUT] Wedka zarzucona, czekam na minigre...")


# --- TRYB TESTOWY ---
# Uruchom: python -m src.input_simulator
# UWAGA: To faktycznie klika i naciska klawisze!
# Masz 5 sekund na przelaczenie sie do okna gry.
if __name__ == "__main__":
    print("=== Test symulatora inputow ===")
    sim = InputSimulator()
    print(f"Srodek okienka lowienia: ({sim.fishing_center_x}, {sim.fishing_center_y})")
    print(f"Szukam okna gry zawierajacego: '{GAME_WINDOW_TITLE}'")

    win = _find_game_window()
    if win:
        print(f"Znaleziono okno: {win.title} ({win.width}x{win.height})")
    else:
        print(f"UWAGA: Nie znaleziono okna zawierajacego '{GAME_WINDOW_TITLE}'!")

    print()
    print("UWAGA: Za 5 sekund nastapi nacisniecie F4 + SPACJA + klikniecie!")
    print("Przelacz sie teraz do okna gry!")
    print()

    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("Naciskam F4 (robak)...")
    sim.use_bait()
    print("Naciskam SPACJA (wedka)...")
    sim.cast_rod()
    print("Klikam w srodek okienka lowienia!")
    sim.click_fishing_box()
    print("Gotowe.")
