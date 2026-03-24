"""
Backend strony BeSafeFish - rejestracja uzytkownikow przez WEB.

Wersja przystosowana do deployu na Render.com (lub inny hosting PaaS).
- Nie importuje gui.db ani innych modulow z post_cnn — jest samodzielny
- Laczy sie z baza bezposrednio przez psycopg2 + DATABASE_URL_ADMIN
- Na Render zmienne srodowiskowe ustawiane w Dashboard (Environment)
- Lokalnie laduje z .env (python-dotenv)

Uruchomienie lokalne:
    python server.py

Deploy (Render/produkcja):
    gunicorn server:app
    Zmienne srodowiskowe: DATABASE_URL_ADMIN, WEB_SYSTEM_USER_ID, GUI_SYSTEM_USER_ID

Migracja na inny hosting:
    1. Skopiuj caly folder website/
    2. Ustaw DATABASE_URL_ADMIN (connection string Supabase Session Pooler)
    3. Ustaw WEB_SYSTEM_USER_ID (domyslnie 7 = Rejestracja_WEB)
    4. Zainstaluj zaleznosci z website/requirements.txt
    5. Uruchom przez gunicorn/waitress/uwsgi
"""

import os

import psycopg2
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Lokalnie laduje .env, na Render zmienne sa w Environment
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL_ADMIN")
WEB_SYSTEM_USER_ID = int(os.getenv("WEB_SYSTEM_USER_ID", "7"))
GUI_SYSTEM_USER_ID = int(os.getenv("GUI_SYSTEM_USER_ID", "5"))

app = Flask(__name__, static_folder=".", static_url_path="")


def _get_connection():
    if not DATABASE_URL:
        raise RuntimeError("Brak DATABASE_URL_ADMIN")
    return psycopg2.connect(DATABASE_URL)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "msg": "Brak danych."}), 400

    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if len(username) < 3:
        return jsonify({"ok": False, "msg": "Nazwa uzytkownika musi miec min. 3 znaki."})
    if not email:
        return jsonify({"ok": False, "msg": "Podaj adres email."})
    if len(password) < 4:
        return jsonify({"ok": False, "msg": "Haslo musi miec min. 4 znaki."})

    source = (data.get("source") or "web").strip().lower()
    system_user_id = GUI_SYSTEM_USER_ID if source == "gui" else WEB_SYSTEM_USER_ID

    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            "SET LOCAL app.current_user_id = %s", (str(system_user_id),)
        )
        cur.execute(
            "SELECT create_user_short(%s, %s, %s, 'user')",
            (username, email, password),
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"ok": True, "msg": "Konto utworzone! Mozesz pobrac aplikacje i sie zalogowac."})
    except psycopg2.errors.UniqueViolation:
        return jsonify({"ok": False, "msg": "Uzytkownik o tej nazwie juz istnieje."})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Blad rejestracji: {e}"}), 500


@app.route("/api/login", methods=["POST"])
def login():
    """Endpoint logowania - uzywany przez aplikacje desktopowa (GUI)."""
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "msg": "Brak danych."}), 400

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"ok": False, "msg": "Wypelnij wszystkie pola."})

    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id FROM users
            WHERE login = %s
              AND password_hash = extensions.crypt(%s, password_hash)
              AND is_active = true
              AND deleted_at IS NULL
            """,
            (username, password),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            return jsonify({"ok": True, "msg": "Zalogowano pomyslnie"})
        else:
            return jsonify({"ok": False, "msg": "Nieprawidlowa nazwa uzytkownika lub haslo."})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Blad logowania: {e}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Sprawdza czy serwer dziala - uzywany przez GUI do init_db."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False}), 500


if __name__ == "__main__":
    print("BeSafeFish WEB - http://localhost:5000")
    app.run(debug=True, port=5000)
