"""
Glowna petla bota Kosa.

Laczy wszystkie moduly i steruje cala logika automatyzacji lowienia ryb.

Cykl zycia jednej rundy:
1. Uzyj robaka (F4)
2. Zarzuc wedke (SPACJA)
3. Czekaj az okienko "Lowienie" sie pojawi
4. Skanuj okrag:
   - Czerwony = kliknij LPM (rybka w srodku)
   - Bialy = czekaj (rybka poza)
   - Brak = minigra sie skonczyla
5. Po 3 trafieniach lub koncu czasu -> wracamy do kroku 1

Zabezpieczenia:
- pyautogui.FAILSAFE = True (rusz mysz w lewy gorny rog zeby przerwac)
- Klawisz 'q' w oknie podgladu zamyka bota
- Ctrl+C w terminalu tez zamyka
"""

import time
import sys
import ctypes
import cv2


def _check_admin():
    """Sprawdza czy bot jest uruchomiony jako Administrator."""
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("BLAD: Bot musi byc uruchomiony jako Administrator!")
        print("Gra Eryndos dziala z uprawnieniami admina - bez tego")
        print("klawisze nie dotra do okna gry (Windows UIPI).")
        print()
        print("Rozwiazanie:")
        print("  1. Otworz PowerShell jako Administrator")
        print("  2. cd do folderu projektu")
        print("  3. .\\venv\\Scripts\\python.exe -m src.bot --debug")
        print()
        print("  Lub uzyj: start_bot.bat")
        sys.exit(1)


from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
from src.input_simulator import InputSimulator
from src.config import (
    SCAN_INTERVAL, CLICKS_TO_WIN,
    CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, CIRCLE_SAFE_MARGIN,
)
import math


class KosaBot:
    """Glowny bot automatyzujacy lowienie ryb w Metin2."""

    # Bezpieczny promien — NIGDY nie klikamy dalej niz to od srodka okregu
    SAFE_RADIUS = CIRCLE_RADIUS - CIRCLE_SAFE_MARGIN  # 64 - 10 = 54 px

    # Limit klkniec w to samo miejsce (ochrona przed klikaniem w napis MISS)
    SAME_SPOT_RADIUS = 15      # piksele - jesli klik w tym promieniu = "to samo miejsce"
    SAME_SPOT_MAX_CLICKS = 3   # max klikniec w to samo miejsce

    @staticmethod
    def _clamp_to_circle(x: int, y: int) -> tuple:
        """
        Ogranicza pozycje klikniecia do wnetrza okregu z marginesem.
        Jesli (x,y) jest poza bezpieczna strefa, przesuwa punkt
        na brzeg bezpiecznego okregu w tym samym kierunku.

        Returns:
            (x, y) wewnatrz bezpiecznego okregu
        """
        dx = x - CIRCLE_CENTER_X
        dy = y - CIRCLE_CENTER_Y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= KosaBot.SAFE_RADIUS:
            return (x, y)  # juz w bezpiecznej strefie

        # Przesuń na brzeg bezpiecznego okregu
        scale = KosaBot.SAFE_RADIUS / dist
        new_x = int(CIRCLE_CENTER_X + dx * scale)
        new_y = int(CIRCLE_CENTER_Y + dy * scale)
        return (new_x, new_y)

    def __init__(self, debug: bool = False):
        """
        Args:
            debug: jesli True, pokazuje okno podgladu z wizualizacja
        """
        self.capture = ScreenCapture()
        self.detector = FishingDetector()
        self.input = InputSimulator()
        self.debug = debug
        self.running = False
        self.total_rounds = 0
        self.total_catches = 0

    def wait_for_fishing_minigame(self, timeout: float = 10.0) -> bool:
        """
        Czeka az okienko minigry lowienia sie pojawi.

        Args:
            timeout: maksymalny czas oczekiwania w sekundach

        Returns:
            True jesli minigra sie pojawila, False jesli timeout
        """
        print("[BOT] Czekam na pojawienie sie minigry...")
        start = time.time()
        while time.time() - start < timeout:
            frame = self.capture.grab_fishing_box()
            if self.detector.is_fishing_active(frame):
                print("[BOT] Minigra wykryta!")
                return True
            time.sleep(SCAN_INTERVAL)

        print("[BOT] Timeout - minigra sie nie pojawila.")
        return False

    def play_fishing_round(self) -> bool:
        """
        Gra jedna runde minigry lowienia.

        Strategia: CIAGLE KLIKANIE Z AKTUALNYM SLEDZENIEM
        - W fazie BIALEJ: sledzimy rybke (background subtraction)
        - W fazie CZERWONEJ: klikamy CO KLATKE w aktualna pozycje rybki
        - Background subtraction daje ~1-5px dokladnosc w obu fazach

        Returns:
            True jesli runda sie zakonczyla normalnie,
            False jesli przerwano (np. klawisz 'q')
        """
        print("[BOT] === Rozpoczynam runde lowienia ===")
        click_count = 0
        no_circle_count = 0
        max_no_circle = 15
        was_active = False
        last_fish_pos = None
        click_spots = []  # tracker klikniec w to samo miejsce [(x, y, count)]

        # Reset trackera rybki na poczatku rundy
        self.detector.reset_tracking()

        # Fokusuj okno RAZ na poczatku rundy
        self.input.ensure_focus()

        while self.running:
            frame = self.capture.grab_fishing_box()
            color = self.detector.detect_circle_color(frame)

            # Sledzenie rybki — dziala w obu fazach dzieki bg subtraction
            fish_pos = self.detector.find_fish_position(frame, circle_color=color)
            if fish_pos is not None:
                last_fish_pos = fish_pos

            if color == "red":
                was_active = True
                no_circle_count = 0

                # Klikaj w kazdej klatce - pozycja jest aktualna!
                click_target = fish_pos if fish_pos is not None else last_fish_pos

                if click_target is not None:
                    fx, fy = self._clamp_to_circle(click_target[0], click_target[1])

                    # Sprawdz limit klikniec w to samo miejsce
                    spot_blocked = False
                    for i, (sx, sy, cnt) in enumerate(click_spots):
                        dist = math.sqrt((fx - sx)**2 + (fy - sy)**2)
                        if dist <= self.SAME_SPOT_RADIUS:
                            if cnt >= self.SAME_SPOT_MAX_CLICKS:
                                spot_blocked = True
                            else:
                                click_spots[i] = (sx, sy, cnt + 1)
                            break
                    else:
                        click_spots.append((fx, fy, 1))

                    if not spot_blocked:
                        self.input.click_at_fish_fast(fx, fy)
                        click_count += 1
                        if click_count % 5 == 1:  # loguj co 5 klikniec
                            print(f"[BOT] Klik #{click_count} w ({fx},{fy})"
                                  f" {'[FRESH]' if fish_pos else '[LAST]'}")

            elif color == "white":
                no_circle_count = 0
                was_active = True
                click_spots = []  # reset spot trackera przy zmianie fazy na biala

            else:
                no_circle_count += 1
                if was_active and no_circle_count >= max_no_circle:
                    print(f"[BOT] Okno minigry sie zamknelo. Klikniec: {click_count}")
                    return True

            # Debug: podglad na zywo
            if self.debug:
                if not self._show_debug(frame, color, click_count, last_fish_pos):
                    return False

            time.sleep(SCAN_INTERVAL)

        return False

    def _wait_for_minigame_close(self, timeout: float = 5.0):
        """
        Czeka az okno minigry calkowicie zniknie.
        Upewnia sie ze nie zaczniemy nowej rundy za wczesnie.
        """
        start = time.time()
        while time.time() - start < timeout:
            frame = self.capture.grab_fishing_box()
            if not self.detector.is_fishing_active(frame):
                return  # okno zamkniete
            time.sleep(0.1)
        print("[BOT] Timeout czekania na zamkniecie okna minigry.")

    def _show_debug(self, frame, color, click_count, fish_pos=None) -> bool:
        """
        Wyswietla okno debugowe z podgladem.

        Returns:
            False jesli uzytkownik nacisnal 'q' (chce zakonczyc)
        """
        display = frame.copy()

        # Kolor ramki
        if color == "red":
            border = (0, 0, 255)
            text = "KLIKAJ!"
        elif color == "white":
            border = (255, 255, 255)
            text = "Czekaj..."
        else:
            border = (128, 128, 128)
            text = "Brak okregu"

        cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1), border, 3)

        # Rysuj bezpieczny okrag (zielony = strefa klikania)
        cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, (128, 128, 128), 1)
        cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), self.SAFE_RADIUS, (0, 255, 0), 1)

        cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border, 2)
        cv2.putText(display, f"Klikniecia: {click_count}/{CLICKS_TO_WIN}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display, f"Rundy: {self.total_rounds} | Zlow: {self.total_catches}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Rysuj pozycje rybki
        if fish_pos is not None:
            fx, fy = fish_pos
            cv2.circle(display, (fx, fy), 8, (0, 255, 255), 2)  # zolte kolko
            cv2.line(display, (fx - 12, fy), (fx + 12, fy), (0, 255, 255), 1)
            cv2.line(display, (fx, fy - 12), (fx, fy + 12), (0, 255, 255), 1)
            cv2.putText(display, f"Rybka: ({fx},{fy})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(display, "Rybka: ???", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 100), 1)

        cv2.imshow("Kosa Bot", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[BOT] Uzytkownik nacisnal 'q' - koncze.")
            self.running = False
            return False
        return True

    def run(self):
        """
        Glowna petla bota. Powtarza cykl lowienia w nieskonczonosc.
        Zatrzymaj przez:
        - Klawisz 'q' w oknie podgladu (tryb debug)
        - Ctrl+C w terminalu
        - Ruszenie myszy w lewy gorny rog (pyautogui failsafe)
        """
        self.running = True
        print("=" * 50)
        print("  KOSA BOT - Automatyczne lowienie ryb")
        print("=" * 50)
        print()
        print("Zabezpieczenia:")
        print("  - Rusz mysz w LEWY GORNY ROG ekranu = natychmiastowe przerwanie")
        print("  - Ctrl+C w terminalu = przerwanie")
        if self.debug:
            print("  - Klawisz 'q' w oknie podgladu = przerwanie")
        print()
        print("Start za 3 sekundy...")
        time.sleep(3)

        try:
            while self.running:
                self.total_rounds += 1
                print(f"\n{'='*40}")
                print(f"[BOT] RUNDA {self.total_rounds}")
                print(f"{'='*40}")

                # 1. Start rundy: robak + wedka
                self.input.start_fishing_round()

                # 2. Czekaj na minigre
                if not self.wait_for_fishing_minigame():
                    print("[BOT] Minigra sie nie pojawila. Probuje ponownie...")
                    continue

                # 3. Graj runde
                if not self.play_fishing_round():
                    break  # przerwano

                # 4. Czekaj az okno minigry sie zamknie + 3s przerwy
                print("[BOT] Czekam az okno minigry calkowicie zniknie...")
                self._wait_for_minigame_close()
                print("[BOT] Pauza 3 sekundy przed nastepna runda...")
                time.sleep(3.0)

        except KeyboardInterrupt:
            print("\n[BOT] Przerwano przez Ctrl+C.")
        except Exception as e:
            print(f"\n[BOT] Blad: {e}")
        finally:
            self.running = False
            if self.debug:
                cv2.destroyAllWindows()
            print()
            print(f"[BOT] Podsumowanie:")
            print(f"  Rundy: {self.total_rounds}")
            print(f"  Udane zlowienia: {self.total_catches}")
            print(f"[BOT] Koniec.")


# --- URUCHOMIENIE ---
# python -m src.bot          -> tryb normalny
# python -m src.bot --debug  -> tryb z podgladem
if __name__ == "__main__":
    _check_admin()
    debug_mode = "--debug" in sys.argv
    bot = KosaBot(debug=debug_mode)
    bot.run()
