"""KosaBot - dispatcher trybow minigry.

Po refaktorze (Etap 2) KosaBot trzyma:
- wspolny cykl rundy (znajdz okno gry, zarzuc wedke, deleguj do trybu, pauza)
- liczniki sesji
- callbacki (log, round_check)
- flage running

Logika SPECYFICZNA dla danej minigry (detekcja, klikanie / spacja, debug overlay)
zyje w `src/fishing_modes/<tryb>.py`. Aktualne tryby:
- "fish_click" -> src/fishing_modes/fish_click.py (Mini-gra lowienie ryb)

Dodanie nowego trybu = nowy plik w fishing_modes/ + dispatcher tutaj.
"""

import time
import sys
import ctypes


def _check_admin():
    """Sprawdza czy bot jest uruchomiony jako Administrator."""
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("BLAD: Bot musi byc uruchomiony jako Administrator!")
        print("Gra Eryndos dziala z uprawnieniami admina - bez tego")
        print("klawisze nie dotra do okna gry (Windows UIPI).")
        print()
        print("Rozwiazanie:")
        print("  1. Otworz PowerShell jako Administrator")
        print("  2. Przejdz do folderu wariantu bota i uruchom:")
        print()
        print('     cd "<root repo>\\versions\\tryb1_rybka_klik\\post_cnn"')
        print('     $env:PYTHONPATH="."')
        print('     py -m src.bot --debug')
        print()
        sys.exit(1)


from src.screen_capture import ScreenCapture
from src.input_simulator import InputSimulator


# Mapa nazw trybow na klasy implementujace.
# Przy dodawaniu nowego trybu: zaimportuj klase i dopisz wpis tutaj.
def _build_mode(name: str, **deps):
    """Fabryka trybow. Lazy import zeby nie ladowac niepotrzebnych modulow."""
    if name == "fish_click":
        from src.fishing_modes.fish_click import FishClickMode
        return FishClickMode(**deps)
    raise ValueError(
        f"Nieznany tryb minigry: {name!r}. "
        f"Dostepne: 'fish_click'. (Dodajesz nowy? Patrz src/fishing_modes/__init__.py)"
    )


class KosaBot:
    """Glowny bot - dispatcher do trybu minigry."""

    def __init__(
        self,
        mode: str = "fish_click",
        debug: bool = False,
        use_cnn: bool = True,
        log_callback=None,
        round_check_callback=None,
    ):
        """
        Args:
            mode: identyfikator trybu minigry (np. 'fish_click')
            debug: pokazuj okno podgladu z wizualizacja
            use_cnn: parametr przekazywany do trybu (semantyka zalezna od trybu)
            log_callback: opcjonalna funkcja(str) do przekierowania logow (np. do GUI)
            round_check_callback: opcjonalna funkcja() -> (allowed: bool, msg: str)
                                  wywoływana przed kazda runda do sprawdzenia limitu
        """
        self._log_callback = log_callback
        self._round_check_callback = round_check_callback
        self.debug = debug
        self.running = False
        self.total_rounds = 0

        # Wspolne dependencies — dzielone z trybem przez DI
        self.capture = ScreenCapture()
        self.input = InputSimulator()

        # Zbuduj wybrany tryb (laduje sie tylko ten, ktory jest potrzebny)
        self.mode = _build_mode(
            mode,
            debug=debug,
            use_cnn=use_cnn,
            log_callback=self._log,
            capture=self.capture,
            input_sim=self.input,
            is_running=lambda: self.running,
            request_stop=self.stop,
        )

    def _log(self, msg: str):
        """Loguje wiadomosc — przez callback (GUI) lub print (terminal)."""
        if self._log_callback:
            self._log_callback(msg)
        else:
            print(msg)

    def stop(self):
        """Zatrzymuje bota (thread-safe — ustawia flage)."""
        self.running = False

    def run(self):
        """Glowna petla bota.

        Powtarza cykl lowienia w nieskonczonosc. Zatrzymaj przez:
        - stop() (np. z GUI)
        - klawisz 'q' w oknie podgladu (tryb debug) — tryb sam wola self._request_stop
        - Ctrl+C w terminalu
        - ruszenie myszy w lewy gorny rog (pyautogui failsafe)
        """
        self.running = True
        self._log("=" * 50)
        self._log(f"  KOSA BOT - Tryb: {self.mode.name}")
        self._log("=" * 50)
        self._log("")

        # --- Znajdz i sfokusuj okno gry (uniwersalne dla wszystkich trybow Metin2) ---
        from src.input_simulator import _find_game_window, _focus_game_window, GAME_WINDOW_TITLE
        win = _find_game_window()
        if win:
            self._log(f"[BOT] Znaleziono okno gry: \"{win.title}\"")
            self._log(f"[BOT] Rozmiar: {win.width}x{win.height}, pozycja: ({win.left},{win.top})")
            if _focus_game_window():
                self._log("[BOT] Okno gry przeniesione na pierwszy plan!")
            else:
                self._log("[BOT] Nie udalo sie aktywowac okna - przelacz recznie!")
                time.sleep(3)
        else:
            self._log(f"[BOT] UWAGA: Nie znaleziono okna '{GAME_WINDOW_TITLE}'!")
            self._log("[BOT] Sprawdz czy gra jest uruchomiona.")
            self._log("[BOT] Przelacz sie recznie na okno gry w ciagu 5 sekund!")
            time.sleep(5)

        self._log("")
        self._log("Zabezpieczenia:")
        self._log("  - Rusz mysz w LEWY GORNY ROG ekranu = natychmiastowe przerwanie")
        if self.debug:
            self._log("  - Klawisz 'q' w oknie podgladu = przerwanie")
        self._log("")
        self._log("Start za 3 sekundy...")
        time.sleep(3)

        try:
            while self.running:
                # Sprawdz limit rund (jesli callback ustawiony)
                if self._round_check_callback:
                    allowed, check_msg = self._round_check_callback()
                    if not allowed:
                        self._log(f"[BOT] LIMIT RUND: {check_msg}")
                        break
                    if check_msg:
                        self._log(f"[LIMIT] {check_msg}")

                self.total_rounds += 1
                self._log(f"\n{'='*40}")
                self._log(f"[BOT] RUNDA {self.total_rounds}")
                self._log(f"{'='*40}")

                # 1. Start rundy: robak + wedka (uniwersalne dla obu minigier)
                self.input.start_fishing_round()

                # 2. Czekaj na minigre (delegacja do trybu)
                if not self.mode.wait_for_start(timeout=10.0):
                    self._log("[BOT] Minigra sie nie pojawila. Probuje ponownie...")
                    continue

                # 3. Graj runde (delegacja do trybu)
                if not self.mode.play_round():
                    break  # przerwano (stop / 'q')

                # 4. Czekaj az okno minigry sie zamknie + pauza
                self._log("[BOT] Czekam az okno minigry calkowicie zniknie...")
                self.mode.wait_for_end()
                self._log("[BOT] Pauza 3s przed nastepna runda...")
                time.sleep(3.0)

        except KeyboardInterrupt:
            self._log("\n[BOT] Przerwano przez Ctrl+C.")
        except Exception as e:
            self._log(f"\n[BOT] Blad: {e}")
        finally:
            self.running = False
            if self.debug:
                try:
                    import cv2
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self._log("")
            self._log(f"[BOT] Podsumowanie:")
            self._log(f"  Rundy: {self.total_rounds}")
            self._log(f"[BOT] Koniec.")


# --- URUCHOMIENIE ---
# Wymagany PowerShell jako Administrator!
# python -m src.bot                  -> tryb normalny z CNN
# python -m src.bot --debug          -> tryb z podgladem + CNN
# python -m src.bot --debug --no-cnn -> tryb klasyczny (bez CNN)
if __name__ == "__main__":
    _check_admin()
    debug_mode = "--debug" in sys.argv
    use_cnn = "--no-cnn" not in sys.argv
    bot = KosaBot(mode="fish_click", debug=debug_mode, use_cnn=use_cnn)
    bot.run()
