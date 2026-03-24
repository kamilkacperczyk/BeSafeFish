"""
Zakladka Subskrypcja — wyswietla aktualny plan, zuzycie rund, porownanie planow i historie platnosci.

Dane poczatkowe z loginu (bez dodatkowych requestow).
Zuzycie rund aktualizowane lokalnie (+1 z kazda runda bota).
Plany i platnosci ladowane w tle przy pierwszym otwarciu.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSpacerItem, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QProgressBar,
)
from PySide6.QtCore import Qt, QThread, Signal


class _DataLoaderThread(QThread):
    """Laduje plany i platnosci w tle (nie blokuje GUI)."""
    done = Signal(list, list, int, int)  # plans, payments, rounds_used, max_rounds

    def __init__(self, user_id):
        super().__init__()
        self._user_id = user_id

    def run(self):
        from gui.db import get_plans, get_payments, get_daily_usage

        plans_result = get_plans()
        plans = plans_result.get("plans", [])

        payments_result = get_payments(self._user_id)
        payments = payments_result.get("payments", [])

        usage_result = get_daily_usage(self._user_id)
        rounds_used = usage_result.get("rounds_used", 0) if usage_result.get("ok") else 0
        max_rounds = usage_result.get("max_rounds", 0) if usage_result.get("ok") else 0

        self.done.emit(plans, payments, rounds_used, max_rounds)


class SubscriptionTab(QWidget):
    """Zakladka z informacjami o subskrypcji uzytkownika."""

    def __init__(self, user_id: int, subscription: dict):
        super().__init__()
        self._user_id = user_id
        self._subscription = subscription or {}
        self._rounds_used = 0
        self._max_rounds = 0
        self._setup_ui()
        self._load_data_async()

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(20)
        layout.setContentsMargins(16, 16, 16, 16)

        # === HEADER ===
        page_title = QLabel("Twoja subskrypcja")
        page_title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #53a8b6;"
        )
        layout.addWidget(page_title)

        # === AKTUALNY PLAN — karta ===
        self._plan_card = QFrame()
        self._plan_card.setObjectName("subscriptionCard")
        card_layout = QVBoxLayout(self._plan_card)
        card_layout.setSpacing(10)
        card_layout.setContentsMargins(20, 16, 20, 16)

        plan_header_row = QHBoxLayout()
        plan_label = QLabel("Aktualny plan")
        plan_label.setStyleSheet("color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;")
        plan_header_row.addWidget(plan_label)
        plan_header_row.addStretch()

        self._plan_badge = QLabel()
        plan_header_row.addWidget(self._plan_badge)
        card_layout.addLayout(plan_header_row)

        self._plan_name_label = QLabel()
        card_layout.addWidget(self._plan_name_label)

        self._plan_status_label = QLabel()
        self._plan_status_label.setStyleSheet("color: #888; font-size: 12px;")
        card_layout.addWidget(self._plan_status_label)

        card_sep = QFrame()
        card_sep.setFrameShape(QFrame.HLine)
        card_sep.setStyleSheet("background-color: #0f3460; max-height: 1px;")
        card_layout.addWidget(card_sep)

        self._plan_features_label = QLabel()
        self._plan_features_label.setStyleSheet("color: #aaa; font-size: 12px; line-height: 1.6;")
        self._plan_features_label.setWordWrap(True)
        card_layout.addWidget(self._plan_features_label)

        layout.addWidget(self._plan_card)

        # === ZUZYCIE RUND — karta z progress barem ===
        self._usage_card = QFrame()
        self._usage_card.setObjectName("subscriptionCard")
        usage_layout = QVBoxLayout(self._usage_card)
        usage_layout.setSpacing(10)
        usage_layout.setContentsMargins(20, 16, 20, 16)

        usage_header = QLabel("Dzienne zuzycie rund")
        usage_header.setStyleSheet("color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;")
        usage_layout.addWidget(usage_header)

        self._usage_text = QLabel("Ladowanie...")
        self._usage_text.setStyleSheet("font-size: 24px; font-weight: bold; color: #e0e0e0;")
        usage_layout.addWidget(self._usage_text)

        self._usage_bar = QProgressBar()
        self._usage_bar.setMinimum(0)
        self._usage_bar.setMaximum(100)
        self._usage_bar.setTextVisible(False)
        self._usage_bar.setFixedHeight(12)
        self._usage_bar.setStyleSheet(
            "QProgressBar { background-color: #0d1117; border: 1px solid #21262d; border-radius: 6px; }"
            "QProgressBar::chunk { background-color: #1b998b; border-radius: 5px; }"
        )
        self._usage_bar.setVisible(False)
        usage_layout.addWidget(self._usage_bar)

        self._usage_hint = QLabel()
        self._usage_hint.setStyleSheet("color: #666; font-size: 11px;")
        usage_layout.addWidget(self._usage_hint)

        layout.addWidget(self._usage_card)

        # === POROWNANIE PLANOW ===
        plans_header = QLabel("Dostepne plany")
        plans_header.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #53a8b6; margin-top: 4px;"
        )
        layout.addWidget(plans_header)

        self._plans_row = QHBoxLayout()
        self._plans_row.setSpacing(12)
        layout.addLayout(self._plans_row)

        # Placeholder ladowania
        self._plans_loading = QLabel("Ladowanie planow...")
        self._plans_loading.setStyleSheet("color: #555; font-style: italic;")
        self._plans_row.addWidget(self._plans_loading)

        # === HISTORIA PLATNOSCI ===
        payments_header = QLabel("Historia platnosci")
        payments_header.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #53a8b6; margin-top: 4px;"
        )
        layout.addWidget(payments_header)

        self._payments_table = QTableWidget()
        self._payments_table.setColumnCount(4)
        self._payments_table.setHorizontalHeaderLabels(["Data", "Kwota", "Plan", "Status"])
        self._payments_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._payments_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._payments_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._payments_table.setMinimumHeight(120)
        self._payments_table.setMaximumHeight(200)
        self._payments_table.verticalHeader().setVisible(False)
        self._payments_table.setVisible(False)
        layout.addWidget(self._payments_table)

        self._no_payments_label = QLabel("Brak historii platnosci")
        self._no_payments_label.setStyleSheet("color: #555; font-style: italic; padding: 16px;")
        self._no_payments_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._no_payments_label)

        layout.addStretch()

        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Wyswietl dane planu od razu (z loginu, bez API)
        self._update_plan_display()

    def _load_data_async(self):
        """Laduje plany, platnosci i zuzycie rund w tle."""
        self._loader = _DataLoaderThread(self._user_id)
        self._loader.done.connect(self._on_data_loaded)
        self._loader.start()

    def _on_data_loaded(self, plans, payments, rounds_used, max_rounds):
        """Callback z watku — aktualizuje UI."""
        self._rounds_used = rounds_used
        self._max_rounds = max_rounds
        self._update_usage_display()
        self._display_plans(plans)
        self._display_payments(payments)

    def increment_round(self):
        """Wywoływane przez Dashboard po kazdej rundzie bota. Lokalne +1."""
        self._rounds_used += 1
        self._update_usage_display()

    def _update_plan_display(self):
        """Aktualizuje wyswietlanie aktualnego planu (dane z loginu)."""
        sub = self._subscription
        if not sub or not sub.get("has_active"):
            self._plan_name_label.setText("Brak aktywnej subskrypcji")
            self._plan_name_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #e63946;")
            self._plan_badge.setText("NIEAKTYWNY")
            self._plan_badge.setStyleSheet(
                "background-color: #e63946; color: white; font-size: 10px; font-weight: bold; "
                "padding: 2px 10px; border-radius: 10px;"
            )
            self._plan_status_label.setText("Skontaktuj sie z administratorem")
            self._plan_features_label.setText("")
            return

        plan_name = sub.get("plan_name", "Nieznany")
        expires = sub.get("expires_at")

        self._plan_name_label.setText(plan_name)
        self._plan_name_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #1b998b;")

        self._plan_badge.setText("AKTYWNY")
        self._plan_badge.setStyleSheet(
            "background-color: #1b998b; color: white; font-size: 10px; font-weight: bold; "
            "padding: 2px 10px; border-radius: 10px;"
        )

        if expires:
            self._plan_status_label.setText(f"Aktywna do: {expires[:10]}")
        else:
            self._plan_status_label.setText("Aktywna — bez daty wygasniecia")

        features = sub.get("features") or {}
        features_lines = []
        if "max_rounds_per_day" in features:
            val = features["max_rounds_per_day"]
            features_lines.append(f"Rundy dziennie: {'bez limitu' if val == -1 else val}")
        if "max_sessions" in features:
            val = features["max_sessions"]
            features_lines.append(f"Sesje: {'bez limitu' if val == -1 else val}")
        if features.get("priority_support"):
            features_lines.append("Priorytetowe wsparcie")
        if features.get("full_access"):
            features_lines.append("Pelny dostep (Admin)")

        self._plan_features_label.setText("\n".join(features_lines) if features_lines else "")

    def _update_usage_display(self):
        """Aktualizuje progress bar zuzycia rund."""
        if self._max_rounds == -1:
            self._usage_text.setText(f"{self._rounds_used} rund dzisiaj")
            self._usage_bar.setVisible(False)
            self._usage_hint.setText("Twoj plan nie ma limitu dziennych rund")
        elif self._max_rounds > 0:
            remaining = max(0, self._max_rounds - self._rounds_used)
            self._usage_text.setText(f"{self._rounds_used} / {self._max_rounds}")
            self._usage_bar.setMaximum(self._max_rounds)
            self._usage_bar.setValue(self._rounds_used)
            self._usage_bar.setVisible(True)

            pct = self._rounds_used / self._max_rounds
            if pct >= 0.9:
                bar_color = "#e63946"
            elif pct >= 0.7:
                bar_color = "#f4a261"
            else:
                bar_color = "#1b998b"
            self._usage_bar.setStyleSheet(
                f"QProgressBar {{ background-color: #0d1117; border: 1px solid #21262d; border-radius: 6px; }}"
                f"QProgressBar::chunk {{ background-color: {bar_color}; border-radius: 5px; }}"
            )
            self._usage_hint.setText(f"Pozostalo {remaining} rund na dzisiaj")
        else:
            self._usage_text.setText("—")
            self._usage_bar.setVisible(False)
            self._usage_hint.setText("Nie mozna pobrac danych o zuzyciu")

    def _display_plans(self, plans):
        """Wyswietla karty planow."""
        # Usun placeholder
        while self._plans_row.count():
            item = self._plans_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for plan in plans:
            card = self._create_plan_card(plan)
            self._plans_row.addWidget(card)

        if not plans:
            no_plans = QLabel("Nie udalo sie pobrac planow")
            no_plans.setStyleSheet("color: #555;")
            self._plans_row.addWidget(no_plans)

    def _create_plan_card(self, plan: dict) -> QFrame:
        """Tworzy karte planu."""
        is_current = (self._subscription or {}).get("plan_name") == plan["name"]

        card = QFrame()
        card.setObjectName("subscriptionCard")
        if is_current:
            card.setStyleSheet(
                "QFrame#subscriptionCard { background-color: #16213e; border: 2px solid #1b998b; "
                "border-radius: 10px; padding: 16px; }"
            )
        else:
            card.setStyleSheet(
                "QFrame#subscriptionCard { background-color: #16213e; border: 1px solid #0f3460; "
                "border-radius: 10px; padding: 16px; }"
            )

        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)

        if is_current:
            badge = QLabel("AKTUALNY")
            badge.setStyleSheet(
                "background-color: #1b998b; color: white; font-size: 9px; font-weight: bold; "
                "padding: 2px 8px; border-radius: 8px; max-width: 70px;"
            )
            badge.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(badge)

        name = QLabel(plan["name"])
        name.setStyleSheet("color: #e0e0e0; font-size: 18px; font-weight: bold;")
        card_layout.addWidget(name)

        price = plan.get("price", "0.00")
        currency = plan.get("currency", "PLN")

        if float(price) == 0:
            price_text = "Za darmo"
            price_style = "color: #1b998b; font-size: 20px; font-weight: bold;"
        else:
            price_text = f"{price} {currency}/mies."
            price_style = "color: #f4a261; font-size: 20px; font-weight: bold;"

        price_label = QLabel(price_text)
        price_label.setStyleSheet(price_style)
        card_layout.addWidget(price_label)

        desc = plan.get("description", "")
        if desc:
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #888; font-size: 11px;")
            desc_label.setWordWrap(True)
            card_layout.addWidget(desc_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #0f3460; max-height: 1px;")
        card_layout.addWidget(sep)

        features = plan.get("features") or {}
        for key, val in features.items():
            if key == "max_rounds_per_day":
                txt = f"Rundy/dzien: {'bez limitu' if val == -1 else val}"
            elif key == "max_sessions":
                txt = f"Sesje: {'bez limitu' if val == -1 else val}"
            elif key == "priority_support" and val:
                txt = "Priorytetowe wsparcie"
            else:
                continue
            feat_label = QLabel(f"  {txt}")
            feat_label.setStyleSheet("color: #aaa; font-size: 11px;")
            card_layout.addWidget(feat_label)

        if is_current:
            btn = QPushButton("Aktualny plan")
            btn.setEnabled(False)
            btn.setStyleSheet(
                "QPushButton { background-color: #1b998b; color: white; font-size: 11px; "
                "padding: 6px 16px; border-radius: 6px; border: none; }"
                "QPushButton:disabled { background-color: #16413e; color: #1b998b; }"
            )
        elif float(price) > 0:
            btn = QPushButton("Kup Premium")
            btn.setEnabled(False)
            btn.setToolTip("Wkrotce dostepne")
            btn.setStyleSheet(
                "QPushButton { background-color: #f4a261; color: #1a1a2e; font-size: 11px; "
                "font-weight: bold; padding: 6px 16px; border-radius: 6px; border: none; }"
                "QPushButton:disabled { background-color: #3a2a1e; color: #665533; }"
            )
        else:
            btn = None

        if btn:
            card_layout.addWidget(btn)

        card_layout.addStretch()
        return card

    def _display_payments(self, payments):
        """Wyswietla historie platnosci."""
        if not payments:
            self._payments_table.setVisible(False)
            self._no_payments_label.setVisible(True)
            return

        self._payments_table.setVisible(True)
        self._no_payments_label.setVisible(False)
        self._payments_table.setRowCount(len(payments))

        for i, p in enumerate(payments):
            date = (p.get("paid_at") or p.get("created_at", ""))[:10]
            amount = f"{p.get('amount', '0')} {p.get('currency', 'PLN')}"
            plan = p.get("plan_name") or "-"
            status = p.get("status", "-")

            self._payments_table.setItem(i, 0, QTableWidgetItem(date))
            self._payments_table.setItem(i, 1, QTableWidgetItem(amount))
            self._payments_table.setItem(i, 2, QTableWidgetItem(plan))
            self._payments_table.setItem(i, 3, QTableWidgetItem(status))
