"""
Dashboard BeSafeFish — panel sterowania botem.

Funkcje:
    - Wybor trybow minigry (checkboxy) — START pojawia sie po zaznaczeniu chociaz jednego
    - Start/Stop bota jednym przyciskiem
    - Log na zywo (sygnaly z BotWorker)
    - Status: Gotowy / Dziala / Blad
    - Statystyki: liczba rund
    - Opcja: PatchCNN on/off (tylko gdy Tryb 1 aktywny)
    - Zakladka Subskrypcja: aktualny plan, porownanie, historia platnosci
"""

import os
import sys

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QPlainTextEdit, QFrame, QSpacerItem, QSizePolicy,
    QCheckBox, QTabWidget,
)
from PySide6.QtCore import Signal, Qt, Slot, QSettings
from PySide6.QtGui import QTextCursor, QPixmap

from gui.bot_worker import BotWorker
from gui.subscription_tab import SubscriptionTab


def _asset_path(filename: str) -> str:
    """Sciezka do pliku w gui/assets/ dzialajaca i w dev, i w PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, "gui", "assets", filename)
    return os.path.join(os.path.dirname(__file__), "assets", filename)


class Dashboard(QWidget):
    """Panel glowny — kontrola bota i podglad logow."""

    logout_requested = Signal()

    def __init__(self, username: str, user_id: int, subscription: dict):
        super().__init__()
        self._username = username
        self._user_id = user_id
        self._subscription = subscription
        self._worker = None
        self._is_running = False
        self._round_count = 0
        self._settings = QSettings("BeSafeFish", "Desktop")
        self._setup_ui()
        self._load_mode_prefs()
        self._refresh_mode_dependent_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        # === HEADER ===
        header = QHBoxLayout()

        title = QLabel("\U0001F41F BeSafeFish")
        title.setObjectName("titleLabel")
        title.setStyleSheet("font-size: 22px;")
        header.addWidget(title)

        header.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        user_label = QLabel(f"\U0001F464 {self._username}")
        user_label.setObjectName("userLabel")
        header.addWidget(user_label)

        logout_btn = QPushButton("Wyloguj")
        logout_btn.setObjectName("logoutButton")
        logout_btn.clicked.connect(self._on_logout)
        header.addWidget(logout_btn)

        layout.addLayout(header)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # === TABS ===
        self._tabs = QTabWidget()
        self._tabs.setObjectName("mainTabs")

        # Tab 1: Bot
        bot_tab = QWidget()
        self._setup_bot_tab(bot_tab)
        self._tabs.addTab(bot_tab, "Bot")

        # Tab 2: Subskrypcja
        self._subscription_tab = SubscriptionTab(self._user_id, self._subscription)
        self._tabs.addTab(self._subscription_tab, "Subskrypcja")

        layout.addWidget(self._tabs, stretch=1)

    def _setup_bot_tab(self, tab: QWidget):
        """Zawartosc zakladki Bot (status, start/stop, log)."""
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 8, 0, 0)

        # === STATUS ===
        status_row = QHBoxLayout()

        self._status_label = QLabel("\u23F8 Gotowy")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setMinimumHeight(28)
        self._status_label.setStyleSheet(
            "color: #888; background-color: #16213e; "
            "border-radius: 4px; padding: 4px 12px;"
        )
        status_row.addWidget(self._status_label)

        status_row.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self._stats_label = QLabel("Rundy: 0")
        self._stats_label.setObjectName("statsLabel")
        status_row.addWidget(self._stats_label)

        layout.addLayout(status_row)

        # === START / STOP ===
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignCenter)

        self._start_btn = QPushButton("\u25B6  START")
        self._start_btn.setObjectName("startButton")
        self._start_btn.setMinimumWidth(200)
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("\u25A0  STOP")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setMinimumWidth(200)
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)

        layout.addLayout(btn_row)

        # === WYBOR TRYBOW ===
        modes_title = QLabel("Tryby minigry:")
        modes_title.setStyleSheet("color: #53a8b6; font-size: 12px; font-weight: bold;")
        layout.addWidget(modes_title)

        modes_row = QHBoxLayout()
        modes_row.setSpacing(16)
        modes_row.setAlignment(Qt.AlignLeft)

        # Checkboxy wygladaja normalnie, ale logika zapewnia wzajemne
        # wykluczanie — klik na jeden odznacza drugi (patrz _on_*_toggled).

        # Tryb 1: Mini-gra lowienie ryb (rybka - klik)
        self._mode_fish_checkbox = QCheckBox(
            "Mini-gra łowienie ryb (rybka - klik)"
        )
        self._mode_fish_thumb = QLabel()
        self._mode_fish_thumb.setPixmap(QPixmap(_asset_path("mode_fish_click.png")))
        self._mode_fish_thumb.setFixedSize(96, 72)
        mode1_col = QVBoxLayout()
        mode1_col.setSpacing(4)
        mode1_col.addWidget(self._mode_fish_thumb, alignment=Qt.AlignCenter)
        mode1_col.addWidget(self._mode_fish_checkbox, alignment=Qt.AlignCenter)
        modes_row.addLayout(mode1_col)

        # Tryb 2: Mini-gra spacja (dymek z cyfra - spacja)
        self._mode_bubble_checkbox = QCheckBox(
            "Mini-gra spacja (dymek z cyfrą - spacja)"
        )
        self._mode_bubble_thumb = QLabel()
        self._mode_bubble_thumb.setPixmap(QPixmap(_asset_path("mode_bubble_space.png")))
        self._mode_bubble_thumb.setFixedSize(96, 72)
        mode2_col = QVBoxLayout()
        mode2_col.setSpacing(4)
        mode2_col.addWidget(self._mode_bubble_thumb, alignment=Qt.AlignCenter)
        mode2_col.addWidget(self._mode_bubble_checkbox, alignment=Qt.AlignCenter)
        modes_row.addLayout(mode2_col)

        modes_row.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        layout.addLayout(modes_row)

        # Klik na checkbox -> wzajemne wykluczanie + odswiez UI.
        self._mode_fish_checkbox.toggled.connect(self._on_fish_toggled)
        self._mode_bubble_checkbox.toggled.connect(self._on_bubble_toggled)

        # === OPCJE ===
        opts_row = QHBoxLayout()
        opts_row.setAlignment(Qt.AlignLeft)

        self._cnn_checkbox = QCheckBox("Uzyj PatchCNN (weryfikacja rybki)")
        self._cnn_checkbox.setChecked(True)
        self._cnn_checkbox.setToolTip(
            "Dziala tylko dla trybu 'Mini-gra łowienie ryb (rybka - klik)'.\n"
            "Weryfikuje czy wykryty obiekt to faktycznie rybka."
        )
        opts_row.addWidget(self._cnn_checkbox)

        layout.addLayout(opts_row)

        # === LOG ===
        log_label = QLabel("Log:")
        log_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(log_label)

        self._log_area = QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMaximumBlockCount(500)
        layout.addWidget(self._log_area, stretch=1)

        # === FOOTER ===
        footer = QHBoxLayout()

        version_label = QLabel("BeSafeFish v1.2.5 - Kosa Post-CNN")
        version_label.setStyleSheet("color: #555; font-size: 10px;")
        footer.addWidget(version_label)

        footer.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self._clear_btn = QPushButton("Wyczysc log")
        self._clear_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        self._clear_btn.clicked.connect(self._log_area.clear)
        footer.addWidget(self._clear_btn)

        layout.addLayout(footer)

    # ------------------------------------------------------------------
    # AKCJE
    # ------------------------------------------------------------------

    def _on_start(self):
        """Uruchamia bota w osobnym watku."""
        if self._is_running:
            return

        self._round_count = 0
        self._stats_label.setText("Rundy: 0")

        self._log_area.appendPlainText("=" * 40)
        self._log_area.appendPlainText("  Uruchamianie bota...")
        self._log_area.appendPlainText("=" * 40)

        mode = self._selected_mode()
        if mode is None:
            # Nie powinno sie zdarzyc (START jest niewidoczny bez wyboru), ale dla pewnosci.
            return
        self._save_mode_prefs()

        self._worker = BotWorker(
            debug=False,
            use_cnn=self._cnn_checkbox.isChecked(),
            user_id=self._user_id,
            enabled_modes=[mode],
        )
        self._worker.log_message.connect(self._on_log)
        self._worker.status_changed.connect(self._on_status)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

        self._is_running = True
        self._start_btn.setVisible(False)
        self._stop_btn.setVisible(True)
        self._set_mode_controls_enabled(False)

    def _on_stop(self):
        """Zatrzymuje bota."""
        if not self._is_running:
            return

        self._log_area.appendPlainText("[GUI] Zatrzymywanie bota...")
        if self._worker:
            self._worker.stop()

    def _on_logout(self):
        """Wylogowuje (zatrzymuje bota jesli dziala)."""
        if self._is_running:
            self._on_stop()
        self.logout_requested.emit()

    # ------------------------------------------------------------------
    # SYGNALY Z WORKERA
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_log(self, msg: str):
        """Dodaje wiadomosc do loga z auto-scrollem."""
        self._log_area.appendPlainText(msg)

        cursor = self._log_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log_area.setTextCursor(cursor)

        if "RUNDA " in msg:
            try:
                num = int(msg.split("RUNDA ")[1].split()[0])
                self._round_count = num
                self._stats_label.setText(f"Rundy: {self._round_count}")
                # Aktualizuj progress bar w zakladce subskrypcji
                self._subscription_tab.increment_round()
            except (ValueError, IndexError):
                pass

        if "[LIMIT] Runda " in msg and "/" in msg:
            try:
                parts = msg.split("Runda ")[1].split("/")
                used = parts[0].strip()
                limit = parts[1].strip()
                self._stats_label.setText(f"Rundy: {self._round_count} ({used}/{limit})")
            except (ValueError, IndexError):
                pass

    @Slot(str)
    def _on_status(self, status: str):
        """Aktualizuje wskaznik statusu."""
        if status == "running":
            self._status_label.setText("\U0001F7E2 Dziala")
            self._status_label.setStyleSheet(
                "color: #1b998b; background-color: #16213e; "
                "border-radius: 4px; padding: 4px 12px;"
            )
        elif status == "stopped":
            self._status_label.setText("\u23F8 Zatrzymany")
            self._status_label.setStyleSheet(
                "color: #888; background-color: #16213e; "
                "border-radius: 4px; padding: 4px 12px;"
            )
        elif status == "error":
            self._status_label.setText("\U0001F534 Blad")
            self._status_label.setStyleSheet(
                "color: #e63946; background-color: #16213e; "
                "border-radius: 4px; padding: 4px 12px;"
            )

    @Slot()
    def _on_finished(self):
        """Bot zakonczyl prace — przywroc UI."""
        self._is_running = False
        self._start_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._set_mode_controls_enabled(True)
        self._refresh_mode_dependent_ui()
        self._worker = None
        self._log_area.appendPlainText("[GUI] Bot zatrzymany.")

    # ------------------------------------------------------------------
    # TRYBY MINIGRY
    # ------------------------------------------------------------------

    def _selected_mode(self) -> str | None:
        """Zwraca klucz wybranego trybu (np. 'fish_click') albo None."""
        if self._mode_fish_checkbox.isChecked():
            return "fish_click"
        if self._mode_bubble_checkbox.isChecked():
            return "bubble_space"
        return None

    def _on_fish_toggled(self, checked: bool):
        """Klik na Tryb 1 — jesli zaznaczony, odznacz Tryb 2 (wzajemne wykluczanie)."""
        if checked and self._mode_bubble_checkbox.isChecked():
            self._mode_bubble_checkbox.blockSignals(True)
            self._mode_bubble_checkbox.setChecked(False)
            self._mode_bubble_checkbox.blockSignals(False)
        self._save_mode_prefs()
        self._refresh_mode_dependent_ui()

    def _on_bubble_toggled(self, checked: bool):
        """Klik na Tryb 2 — odznacz Tryb 1 oraz PatchCNN (tryb 2 ich nie uzywa)."""
        if checked:
            if self._mode_fish_checkbox.isChecked():
                self._mode_fish_checkbox.blockSignals(True)
                self._mode_fish_checkbox.setChecked(False)
                self._mode_fish_checkbox.blockSignals(False)
            # PatchCNN dotyczy tylko Trybu 1 — automatycznie odznaczamy.
            if self._cnn_checkbox.isChecked():
                self._cnn_checkbox.setChecked(False)
        self._save_mode_prefs()
        self._refresh_mode_dependent_ui()

    def _refresh_mode_dependent_ui(self):
        """Aktywuj/dezaktywuj START i PatchCNN zaleznie od wyboru trybu."""
        if self._is_running:
            # Podczas dzialania bota kontrolki sa zablokowane przez
            # _set_mode_controls_enabled(False) — tutaj nic nie ruszamy.
            return

        mode = self._selected_mode()
        self._start_btn.setEnabled(mode is not None)

        # PatchCNN ma sens tylko gdy wybrany jest tryb rybki.
        self._cnn_checkbox.setEnabled(mode == "fish_click")

    def _set_mode_controls_enabled(self, enabled: bool):
        """Wlacza/wylacza wszystkie kontrolki konfiguracji (podczas pracy bota)."""
        self._mode_fish_checkbox.setEnabled(enabled)
        self._mode_bubble_checkbox.setEnabled(enabled)
        # PatchCNN tylko gdy enabled i gdy wybrany tryb rybki.
        self._cnn_checkbox.setEnabled(enabled and self._selected_mode() == "fish_click")

    def _load_mode_prefs(self):
        """Wczytuje ostatni wybor trybu z QSettings.

        Domyslnie: Tryb 1 (fish_click) zaznaczony, PatchCNN on.
        Jesli zapisana wartosc to pusty string — zostaje nic zaznaczonego.
        """
        mode = self._settings.value("modes/selected", "fish_click", type=str)
        cnn = self._settings.value("options/use_cnn", True, type=bool)
        # Ustawiamy bez odpalania sygnalow, zeby nie zapisac od razu tego samego.
        self._mode_fish_checkbox.blockSignals(True)
        self._mode_bubble_checkbox.blockSignals(True)
        if mode == "fish_click":
            self._mode_fish_checkbox.setChecked(True)
        elif mode == "bubble_space":
            self._mode_bubble_checkbox.setChecked(True)
        # inaczej: oba odznaczone
        self._cnn_checkbox.setChecked(cnn)
        self._mode_fish_checkbox.blockSignals(False)
        self._mode_bubble_checkbox.blockSignals(False)

    def _save_mode_prefs(self):
        """Zapisuje aktualny wybor trybu + PatchCNN do QSettings."""
        self._settings.setValue("modes/selected", self._selected_mode() or "")
        self._settings.setValue("options/use_cnn", self._cnn_checkbox.isChecked())

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def cleanup(self):
        """Czysci zasoby przy zamykaniu (czeka na watek)."""
        if self._worker and self._is_running:
            self._worker.stop()
            self._worker.wait(5000)
