"""
Ekran logowania BeSafeFish.

Obsluguje logowanie i rejestracje uzytkownikow.
Po udanym logowaniu emituje sygnal login_success(username).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFrame, QSpacerItem, QSizePolicy, QCheckBox,
    QDialog, QTextBrowser,
)
from PySide6.QtCore import Signal, Qt, QThread

from gui.db import authenticate_user, register_user, init_db


TERMS_HTML = """
<div style="font-family: Inter, sans-serif; color: #c8d0da; line-height: 1.7;">

<h2 style="color: #e2e8f0; margin-bottom: 4px;">Regulamin serwisu BeSafeFish</h2>
<p style="color: #64748b; font-size: 13px;">Ostatnia aktualizacja: 24 marca 2026</p>

<h3 style="color: #1b998b; margin-top: 20px;">1. Postanowienia ogolne</h3>
<p>Serwis BeSafeFish (dalej "Serwis") umozliwia korzystanie z oprogramowania
do automatycznego lowienia ryb w grze Metin2. Korzystanie z Serwisu oznacza
akceptacje niniejszego regulaminu.</p>
<p>Serwis ma charakter edukacyjny i demonstracyjny - prezentuje zastosowanie
sieci neuronowych (CNN) do analizy obrazu w czasie rzeczywistym.</p>

<h3 style="color: #1b998b; margin-top: 20px;">2. Konto uzytkownika</h3>
<ul>
<li>Rejestracja wymaga podania nazwy uzytkownika, adresu email i hasla.</li>
<li>Uzytkownik odpowiada za bezpieczenstwo swoich danych logowania.</li>
<li>Jedno konto na osobe. Konta wielokrotne moga zostac zablokowane.</li>
<li>Administrator moze dezaktywowac konto naruszajace regulamin.</li>
</ul>

<h3 style="color: #1b998b; margin-top: 20px;">3. Plany i subskrypcje</h3>
<ul>
<li>Darmowy plan: 50 rund dziennie, bez oplat, bez limitu czasowego.</li>
<li>Plan Premium: bez limitu rund, platny miesiecznie.</li>
<li>Po wygasnieciu planu Premium konto automatycznie wraca do planu darmowego.</li>
</ul>

<h3 style="color: #1b998b; margin-top: 20px;">4. Charakter uslugi i odpowiedzialnosc</h3>
<ul>
<li>BeSafeFish to narzedzie oparte na sztucznej inteligencji - jego skutecznosc
zalezy od wielu czynnikow (rozdzielczosc ekranu, ustawienia gry, obciazenie systemu)
i moze sie roznic w zaleznosci od konfiguracji.</li>
<li>Korzystanie z narzedzi automatyzujacych rozgrywke moze byc niezgodne
z regulaminem gry. Uzytkownik podejmuje te decyzje samodzielnie
i na wlasna odpowiedzialnosc.</li>
<li>BeSafeFish nie ponosi odpowiedzialnosci za ewentualne sankcje
ze strony wydawcy gry.</li>
</ul>

<h3 style="color: #1b998b; margin-top: 20px;">5. Jakie dane zbieramy</h3>
<p>Przetwarzamy dane osobowe zgodnie z RODO (Rozporzadzenie UE 2016/679). Zbieramy:</p>
<ul>
<li><b style="color:#e2e8f0;">Dane konta:</b> nazwa uzytkownika, adres email,
zahashowane haslo (bcrypt).</li>
<li><b style="color:#e2e8f0;">Historia logowan:</b> adres IP, User-Agent
przegladarki/aplikacji, data i wynik proby logowania.</li>
<li><b style="color:#e2e8f0;">Dane uzytkowania:</b> liczba wykorzystanych rund dziennie.</li>
</ul>

<h3 style="color: #1b998b; margin-top: 20px;">6. Cel przetwarzania danych</h3>
<ul>
<li><b style="color:#e2e8f0;">Dane konta</b> - umozliwienie logowania i korzystania
z Serwisu (art. 6 ust. 1 lit. b RODO - wykonanie umowy).</li>
<li><b style="color:#e2e8f0;">Historia logowan (IP)</b> - bezpieczenstwo systemu,
wykrywanie nieautoryzowanego dostepu (art. 6 ust. 1 lit. f RODO - uzasadniony interes).</li>
<li><b style="color:#e2e8f0;">Dane uzytkowania</b> - egzekwowanie limitow planu
subskrypcyjnego.</li>
</ul>

<h3 style="color: #1b998b; margin-top: 20px;">7. Przechowywanie i ochrona danych</h3>
<ul>
<li>Hasla sa hashowane algorytmem bcrypt - nigdy nie przechowujemy hasel
w postaci jawnej.</li>
<li>Historia logowan jest przechowywana przez maksymalnie 90 dni,
po czym jest automatycznie usuwana.</li>
<li>Dane przechowywane sa na serwerach w Unii Europejskiej (Supabase).</li>
<li>Polaczenie z serwerem jest szyfrowane (HTTPS/SSL).</li>
</ul>

<h3 style="color: #1b998b; margin-top: 20px;">8. Prawa uzytkownika (RODO)</h3>
<p>Kazdy uzytkownik ma prawo do:</p>
<ul>
<li><b style="color:#e2e8f0;">Dostepu</b> - wgladu w swoje dane osobowe.</li>
<li><b style="color:#e2e8f0;">Sprostowania</b> - poprawienia nieprawidlowych danych.</li>
<li><b style="color:#e2e8f0;">Usuniecia</b> - zadania usuniecia konta i wszystkich danych.</li>
<li><b style="color:#e2e8f0;">Przenoszenia</b> - otrzymania swoich danych w formacie
nadajacym sie do odczytu.</li>
<li><b style="color:#e2e8f0;">Sprzeciwu</b> - wobec przetwarzania danych na podstawie
uzasadnionego interesu.</li>
</ul>
<p>W celu realizacji powyzszych praw skontaktuj sie z administratorem serwisu.</p>

<h3 style="color: #1b998b; margin-top: 20px;">9. Zmiany regulaminu</h3>
<p>Administrator zastrzega sobie prawo do zmiany regulaminu.
O istotnych zmianach uzytkownicy zostana poinformowani.</p>

</div>
"""


class _ServerCheckThread(QThread):
    """Sprawdza serwer w tle (cold start Render moze trwac do 60s)."""
    result = Signal(bool, str)

    def run(self):
        try:
            init_db()
            self.result.emit(True, "")
        except RuntimeError as e:
            self.result.emit(False, str(e))


class TermsDialog(QDialog):
    """Okno dialogowe z regulaminem - uzytkownik musi przewinac do konca."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Regulamin serwisu BeSafeFish")
        self.setMinimumSize(560, 500)
        self.resize(600, 560)
        self.setModal(True)
        self._accepted = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Naglowek
        header = QLabel("Przeczytaj regulamin przed zalozeniem konta")
        header.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #e2e8f0; padding-bottom: 4px;"
        )
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Tresc regulaminu
        self._browser = QTextBrowser()
        self._browser.setObjectName("termsBrowser")
        self._browser.setOpenExternalLinks(False)
        self._browser.setHtml(TERMS_HTML)
        self._browser.setStyleSheet("""
            QTextBrowser {
                background-color: #16213e;
                border: 1px solid #0f3460;
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
            }
        """)
        self._browser.verticalScrollBar().valueChanged.connect(self._on_scroll)
        layout.addWidget(self._browser)

        # Hint
        self._hint = QLabel("Przewin regulamin do konca, aby moc zaakceptowac")
        self._hint.setStyleSheet(
            "color: #64748b; font-size: 12px; font-style: italic;"
        )
        self._hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._hint)

        # Przyciski na dole
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._decline_btn = QPushButton("Anuluj")
        self._decline_btn.setMinimumHeight(40)
        self._decline_btn.setStyleSheet("""
            QPushButton {
                background-color: #16213e;
                border: 1px solid #0f3460;
                border-radius: 6px;
                color: #94a3b8;
                font-size: 13px;
                padding: 8px 24px;
            }
            QPushButton:hover { background-color: #1a2a4a; }
        """)
        self._decline_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._decline_btn)

        self._accept_btn = QPushButton("Akceptuje regulamin")
        self._accept_btn.setMinimumHeight(40)
        self._accept_btn.setEnabled(False)
        self._accept_btn.setStyleSheet("""
            QPushButton {
                background-color: #0f3460;
                border: 1px solid #0f3460;
                border-radius: 6px;
                color: #555;
                font-size: 13px;
                font-weight: 600;
                padding: 8px 24px;
            }
            QPushButton:enabled {
                background-color: #1b998b;
                border-color: #17b890;
                color: #fff;
            }
            QPushButton:enabled:hover { background-color: #17b890; }
        """)
        self._accept_btn.clicked.connect(self._on_accept)
        btn_layout.addWidget(self._accept_btn)

        layout.addLayout(btn_layout)

    def _on_scroll(self):
        sb = self._browser.verticalScrollBar()
        if sb.value() >= sb.maximum() - 10:
            self._accept_btn.setEnabled(True)
            self._hint.setText("Mozesz teraz zaakceptowac regulamin")
            self._hint.setStyleSheet(
                "color: #1b998b; font-size: 12px; font-weight: 500;"
            )

    def _on_accept(self):
        self._accepted = True
        self.accept()

    def was_accepted(self):
        return self._accepted


class LoginScreen(QWidget):
    """Ekran logowania z mozliwoscia rejestracji."""

    login_success = Signal(str, int, object)  # username, user_id, subscription_data

    def __init__(self):
        super().__init__()
        self._is_register_mode = False
        self._server_ready = False
        self._terms_accepted = False
        self._setup_ui()
        self._check_server()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)
        layout.setContentsMargins(60, 40, 60, 40)

        # --- Spacer gorny ---
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # --- Tytul ---
        title = QLabel("\U0001F41F BeSafeFish")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Automatyczne lowienie - Metin2 Eryndos")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(30)

        # --- Formularz (wycentrowany, max 360px) ---
        form_container = QWidget()
        form_container.setMaximumWidth(360)
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(0, 0, 0, 0)

        # Username
        self._username_label = QLabel("Nazwa uzytkownika")
        form_layout.addWidget(self._username_label)
        self._username_input = QLineEdit()
        self._username_input.setPlaceholderText("Wpisz nazwe uzytkownika...")
        self._username_input.setMinimumHeight(38)
        form_layout.addWidget(self._username_input)

        # Email (widoczne tylko w trybie rejestracji)
        self._email_label = QLabel("Email")
        self._email_label.setVisible(False)
        form_layout.addWidget(self._email_label)
        self._email_input = QLineEdit()
        self._email_input.setPlaceholderText("Wpisz adres email...")
        self._email_input.setMinimumHeight(38)
        self._email_input.setVisible(False)
        form_layout.addWidget(self._email_input)

        # Password
        self._password_label = QLabel("Haslo")
        form_layout.addWidget(self._password_label)
        self._password_input = QLineEdit()
        self._password_input.setEchoMode(QLineEdit.Password)
        self._password_input.setPlaceholderText("Wpisz haslo (min. 8 znakow)...")
        self._password_input.setMaxLength(64)
        self._password_input.setMinimumHeight(38)
        form_layout.addWidget(self._password_input)

        # Confirm Password (widoczne tylko w trybie rejestracji)
        self._confirm_label = QLabel("Powtorz haslo")
        self._confirm_label.setVisible(False)
        form_layout.addWidget(self._confirm_label)
        self._confirm_input = QLineEdit()
        self._confirm_input.setEchoMode(QLineEdit.Password)
        self._confirm_input.setPlaceholderText("Powtorz haslo...")
        self._confirm_input.setMaxLength(64)
        self._confirm_input.setMinimumHeight(38)
        self._confirm_input.setVisible(False)
        form_layout.addWidget(self._confirm_input)

        # Regulamin - przycisk (widoczny tylko w trybie rejestracji)
        self._terms_button = QPushButton("Przeczytaj regulamin serwisu")
        self._terms_button.setObjectName("termsButton")
        self._terms_button.setCursor(Qt.PointingHandCursor)
        self._terms_button.setMinimumHeight(36)
        self._terms_button.setVisible(False)
        self._terms_button.clicked.connect(self._open_terms_dialog)
        form_layout.addWidget(self._terms_button)

        # Komunikat bledu / sukcesu
        self._message_label = QLabel("")
        self._message_label.setObjectName("errorLabel")
        self._message_label.setAlignment(Qt.AlignCenter)
        self._message_label.setWordWrap(True)
        form_layout.addWidget(self._message_label)

        # Przycisk akcji (Zaloguj / Zarejestruj)
        self._action_button = QPushButton("Zaloguj sie")
        self._action_button.setObjectName("loginButton")
        self._action_button.setMinimumHeight(42)
        self._action_button.clicked.connect(self._on_action)
        form_layout.addWidget(self._action_button)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        form_layout.addWidget(sep)

        # Przelacznik logowanie/rejestracja
        self._toggle_button = QPushButton("Nie masz konta? Zarejestruj sie")
        self._toggle_button.setObjectName("registerButton")
        self._toggle_button.setCursor(Qt.PointingHandCursor)
        self._toggle_button.clicked.connect(self._toggle_mode)
        form_layout.addWidget(self._toggle_button)

        # Wycentruj formularz
        h_layout = QHBoxLayout()
        h_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        h_layout.addWidget(form_container)
        h_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(h_layout)

        # --- Spacer dolny ---
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Enter = zaloguj
        self._username_input.returnPressed.connect(self._on_action)
        self._email_input.returnPressed.connect(self._on_action)
        self._password_input.returnPressed.connect(self._on_action)
        self._confirm_input.returnPressed.connect(self._on_action)

    def _check_server(self):
        """Sprawdza serwer w tle - nie blokuje GUI."""
        self._action_button.setEnabled(False)
        self._action_button.setText("Laczenie z serwerem...")
        self._show_info("Laczenie z serwerem BeSafeFish... Moze to potrwac do minuty.")

        self._server_thread = _ServerCheckThread()
        self._server_thread.result.connect(self._on_server_check)
        self._server_thread.start()

    def _on_server_check(self, ok: bool, error: str):
        """Callback po sprawdzeniu serwera."""
        self._server_ready = ok
        if ok:
            self._action_button.setEnabled(True)
            self._action_button.setText("Zaloguj sie")
            self._message_label.setText("")
        else:
            self._show_error("Nie mozna polaczyc z serwerem. Sprawdz internet i sprobuj ponownie.")
            self._action_button.setEnabled(True)
            self._action_button.setText("Sprobuj ponownie")
            self._action_button.clicked.disconnect()
            self._action_button.clicked.connect(self._retry_server)

    def _retry_server(self):
        """Ponawia probe polaczenia z serwerem."""
        self._action_button.clicked.disconnect()
        self._action_button.clicked.connect(self._on_action)
        self._check_server()

    def _show_info(self, msg: str):
        self._message_label.setStyleSheet("color: #53a8b6;")
        self._message_label.setText(msg)

    def _toggle_mode(self):
        """Przelacza miedzy trybem logowania a rejestracji."""
        self._is_register_mode = not self._is_register_mode
        self._message_label.setText("")

        if self._is_register_mode:
            self._action_button.setText("Zarejestruj sie")
            self._toggle_button.setText("Masz juz konto? Zaloguj sie")
            self._email_label.setVisible(True)
            self._email_input.setVisible(True)
            self._confirm_label.setVisible(True)
            self._confirm_input.setVisible(True)
            self._terms_button.setVisible(True)
            self._terms_accepted = False
            self._update_terms_button()
        else:
            self._action_button.setText("Zaloguj sie")
            self._toggle_button.setText("Nie masz konta? Zarejestruj sie")
            self._email_label.setVisible(False)
            self._email_input.setVisible(False)
            self._confirm_label.setVisible(False)
            self._confirm_input.setVisible(False)
            self._terms_button.setVisible(False)
            self._terms_accepted = False

    def _open_terms_dialog(self):
        """Otwiera okno z regulaminem."""
        dialog = TermsDialog(self)
        dialog.exec()
        if dialog.was_accepted():
            self._terms_accepted = True
        self._update_terms_button()

    def _update_terms_button(self):
        """Aktualizuje wyglad przycisku regulaminu."""
        if self._terms_accepted:
            self._terms_button.setText("Regulamin zaakceptowany")
            self._terms_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(27, 153, 139, 0.15);
                    border: 1px solid #1b998b;
                    border-radius: 6px;
                    color: #1b998b;
                    font-size: 12px;
                    font-weight: 500;
                }
                QPushButton:hover { background-color: rgba(27, 153, 139, 0.25); }
            """)
        else:
            self._terms_button.setText("Przeczytaj regulamin serwisu")
            self._terms_button.setStyleSheet("")

    def _on_action(self):
        """Obsluguje klikniecie przycisku Zaloguj/Zarejestruj."""
        if not self._server_ready:
            self._show_error("Brak polaczenia z serwerem. Poczekaj lub sprobuj ponownie.")
            return

        username = self._username_input.text().strip()
        password = self._password_input.text()

        if not username or not password:
            self._show_error("Wypelnij wszystkie pola.")
            return

        if self._is_register_mode:
            email = self._email_input.text().strip()
            confirm = self._confirm_input.text()
            if not email:
                self._show_error("Podaj adres email.")
                return
            if password != confirm:
                self._show_error("Hasla sie nie zgadzaja.")
                return
            if not self._terms_accepted:
                self._show_error("Musisz zaakceptowac regulamin serwisu.")
                self._open_terms_dialog()
                return

            ok, msg = register_user(username, email, password)
            if ok:
                self._show_success(msg + " Mozesz sie zalogowac.")
                self._toggle_mode()  # wroc do logowania
                self._username_input.setText(username)
                self._password_input.clear()
            else:
                self._show_error(msg)
        else:
            ok, msg, user_id, subscription = authenticate_user(username, password)
            if ok:
                self.login_success.emit(username, user_id, subscription)
            else:
                self._show_error(msg)

    def _show_error(self, msg: str):
        self._message_label.setStyleSheet("color: #e63946;")
        self._message_label.setText(msg)

    def _show_success(self, msg: str):
        self._message_label.setStyleSheet("color: #1b998b;")
        self._message_label.setText(msg)
