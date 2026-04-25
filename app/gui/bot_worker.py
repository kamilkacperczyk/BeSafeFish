"""
BotWorker — QThread wrapper dla KosaBot.

Uruchamia bota w osobnym watku, emituje sygnaly do GUI.
Zapewnia thread-safe komunikacje: bot._log() → Signal → GUI._on_log()
"""

import ctypes
from PySide6.QtCore import QThread, Signal

from gui.db import use_round


class BotWorker(QThread):
    """Watek roboczy uruchamiajacy KosaBot."""

    # Sygnaly emitowane do GUI (thread-safe przez Qt signal/slot)
    log_message = Signal(str)       # wiadomosc do wyswietlenia w logu
    status_changed = Signal(str)    # "running" / "stopped" / "error"
    finished_signal = Signal()      # bot zakonczyl prace

    def __init__(
        self,
        debug: bool = False,
        use_cnn: bool = True,
        user_id: int = None,
        enabled_modes: list | None = None,
    ):
        super().__init__()
        self._debug = debug
        self._use_cnn = use_cnn
        self._user_id = user_id
        # Lista aktywnych trybow, np. ['fish_click', 'bubble_space'].
        # Etap 1: logujemy i walidujemy; realna obsluga > 1 trybu w Etapie 2.
        self._enabled_modes = list(enabled_modes) if enabled_modes else ["fish_click"]
        self._bot = None

    def _is_admin(self) -> bool:
        """Sprawdza czy proces ma uprawnienia Administratora."""
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

    def run(self):
        """Uruchamia bota w watku (nie wywoluj bezposrednio — uzyj .start())."""
        try:
            # Sprawdz uprawnienia admina (finally ponizej wyemituje 'stopped' + finished).
            if not self._is_admin():
                self.log_message.emit("[BLAD] Bot wymaga uprawnien Administratora!")
                self.log_message.emit("[INFO] Uruchom BeSafeFish jako Administrator.")
                self.log_message.emit("[INFO] PPM na skrot → 'Uruchom jako administrator'")
                self.status_changed.emit("error")
                return

            self.status_changed.emit("running")
            self.log_message.emit("[BOT] Inicjalizacja...")
            self.log_message.emit(
                f"[BOT] Aktywne tryby: {', '.join(self._enabled_modes) or '(brak)'}"
            )

            # Etap 1: tryb 'bubble_space' nie jest jeszcze zaimplementowany.
            # (finally ponizej wyemituje status 'stopped' + finished_signal.)
            if "bubble_space" in self._enabled_modes:
                self.log_message.emit(
                    "[INFO] Tryb 'Mini-gra spacja' bedzie dostepny w kolejnej wersji."
                )
                self.log_message.emit(
                    "[INFO] Wybierz 'Mini-gra łowienie ryb' zeby uruchomic bota."
                )
                return

            # Import bota tutaj — nie blokuje startu GUI
            from src.bot import KosaBot

            # Wybierz tryb - GUI przekazuje liste enabled_modes, my bierzemy
            # pierwszy. Po Etapie 2 dispatcher trybow zyje w KosaBot.
            mode = self._enabled_modes[0] if self._enabled_modes else "fish_click"

            self._bot = KosaBot(
                mode=mode,
                debug=False,           # GUI mode — bez cv2.imshow
                use_cnn=self._use_cnn,
                log_callback=self._on_log,
                round_check_callback=self._check_round_limit,
            )

            self.log_message.emit("[BOT] Uruchamiam glowna petle...")
            self._bot.run()

        except Exception as e:
            self.log_message.emit(f"[BLAD] {e}")
            self.status_changed.emit("error")
        finally:
            self.status_changed.emit("stopped")
            self.finished_signal.emit()

    def _check_round_limit(self) -> tuple:
        """Sprawdza limit rund przez API. Fail-open: blad API = pozwol grac."""
        if not self._user_id:
            return True, ""

        try:
            result = use_round(self._user_id)
        except Exception:
            return True, "Brak polaczenia z API — pomijam limit"

        if not result.get("ok"):
            # Fail-open: blad serwera nie powinien blokowac bota
            return True, f"API niedostepne — pomijam limit"

        allowed = result.get("allowed", True)
        if not allowed:
            return False, result.get("msg", "Limit rund osiagniety")

        rounds_used = result.get("rounds_used", 0)
        max_rounds = result.get("max_rounds", -1)
        if max_rounds > 0:
            return True, f"Runda {rounds_used}/{max_rounds}"
        return True, ""

    def _on_log(self, msg: str):
        """Callback z bota — emituje sygnal do GUI."""
        self.log_message.emit(msg)

    def stop(self):
        """Zatrzymuje bota (thread-safe — ustawia flage)."""
        if self._bot is not None:
            self._bot.stop()
