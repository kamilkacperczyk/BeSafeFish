"""
Backend strony BeSafeFish - rejestracja uzytkownikow przez WEB.

Wersja przystosowana do deployu na Render.com (lub inny hosting PaaS).
- Nie importuje gui.db ani innych modulow aplikacji desktop — jest samodzielny
- Laczy sie z baza bezposrednio przez psycopg2 + DATABASE_URL_ADMIN
- Connection pool (ThreadedConnectionPool) — reuzywanie polaczen
- Na Render zmienne srodowiskowe ustawiane w Dashboard (Environment)
- Lokalnie laduje z .env (python-dotenv)

Uruchomienie lokalne:
    python server.py

Deploy (Render/produkcja):
    gunicorn -c gunicorn.conf.py server:app
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
import psycopg2.pool
import psycopg2.extras
from flask import Flask, request, jsonify, g
from dotenv import load_dotenv

# Lokalnie laduje .env, na Render zmienne sa w Environment
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL_ADMIN")
WEB_SYSTEM_USER_ID = int(os.getenv("WEB_SYSTEM_USER_ID", "7"))
GUI_SYSTEM_USER_ID = int(os.getenv("GUI_SYSTEM_USER_ID", "5"))

app = Flask(__name__, static_folder=".", static_url_path="")

# Connection pool — leniwa inicjalizacja (bezpieczne z gunicorn --preload)
_pool = None


def _get_pool():
    """Zwraca pool polaczen (tworzy przy pierwszym uzyciu w workerze)."""
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("Brak DATABASE_URL_ADMIN")
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=4,
            dsn=DATABASE_URL,
        )
    return _pool


@app.before_request
def _before_request():
    """Pobiera polaczenie z poola na poczatku requestu."""
    g.db_conn = _get_pool().getconn()


@app.teardown_request
def _teardown_request(exception):
    """Zwraca polaczenie do poola po zakonczeniu requestu."""
    conn = g.pop("db_conn", None)
    if conn is not None:
        if exception:
            conn.rollback()
        _get_pool().putconn(conn, close=bool(exception))


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
    if len(password) < 8:
        return jsonify({"ok": False, "msg": "Haslo musi miec min. 8 znakow."})
    if len(password) > 64:
        return jsonify({"ok": False, "msg": "Haslo moze miec maks. 64 znaki."})

    source = (data.get("source") or "web").strip().lower()
    system_user_id = GUI_SYSTEM_USER_ID if source == "gui" else WEB_SYSTEM_USER_ID

    try:
        conn = g.db_conn
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
        return jsonify({"ok": True, "msg": "Konto utworzone! Mozesz pobrac aplikacje i sie zalogowac."})
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return jsonify({"ok": False, "msg": "Uzytkownik o tej nazwie juz istnieje."})
    except Exception as e:
        conn.rollback()
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

    xff = request.headers.get("X-Forwarded-For", "")
    ip_address = xff.split(",")[0].strip() if xff else request.remote_addr
    user_agent = request.headers.get("User-Agent", "")

    try:
        conn = g.db_conn
        cur = conn.cursor()

        # Znajdz usera po loginie (potrzebne do login_history nawet przy blednym hasle)
        cur.execute(
            "SELECT id FROM users WHERE login = %s AND deleted_at IS NULL",
            (username,),
        )
        user_row = cur.fetchone()
        found_user_id = user_row[0] if user_row else None

        # Sprawdz haslo
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

        if not row:
            # Nieudane logowanie - zapisz do historii
            failure_reason = "Nieprawidlowe haslo" if found_user_id else "Nieznany login"
            cur.execute(
                """
                INSERT INTO login_history (user_id, success, ip_address, user_agent, failure_reason)
                VALUES (%s, false, %s, %s, %s)
                """,
                (found_user_id, ip_address, user_agent, failure_reason),
            )
            conn.commit()
            cur.close()
            return jsonify({"ok": False, "msg": "Nieprawidlowa nazwa uzytkownika lub haslo."})

        user_id = row[0]

        # Udane logowanie - zapisz do historii
        cur.execute(
            """
            INSERT INTO login_history (user_id, success, ip_address, user_agent)
            VALUES (%s, true, %s, %s)
            """,
            (user_id, ip_address, user_agent),
        )
        conn.commit()

        # Pobierz dane subskrypcji (z lazy expiration)
        cur.execute("SELECT * FROM check_user_subscription(%s)", (user_id,))
        sub_row = cur.fetchone()
        cur.close()

        subscription = None
        if sub_row:
            subscription = {
                "has_active": sub_row[0],
                "plan_name": sub_row[1],
                "features": sub_row[2],
                "expires_at": sub_row[3].isoformat() if sub_row[3] else None,
            }

        return jsonify({
            "ok": True,
            "msg": "Zalogowano pomyslnie",
            "user_id": user_id,
            "subscription": subscription,
        })
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Blad logowania: {e}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Sprawdza czy serwer dziala - uzywany przez GUI do init_db."""
    try:
        cur = g.db_conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False}), 500


@app.route("/api/subscription/<int:user_id>", methods=["GET"])
def get_subscription(user_id):
    """Pobiera aktualna subskrypcje usera (z lazy expiration)."""
    try:
        cur = g.db_conn.cursor()
        cur.execute("SELECT * FROM check_user_subscription(%s)", (user_id,))
        row = cur.fetchone()
        cur.close()

        subscription = None
        if row:
            subscription = {
                "has_active": row[0],
                "plan_name": row[1],
                "features": row[2],
                "expires_at": row[3].isoformat() if row[3] else None,
            }

        return jsonify({"ok": True, "subscription": subscription})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Blad: {e}"}), 500


@app.route("/api/payments/<int:user_id>", methods=["GET"])
def get_payments(user_id):
    """Pobiera historie platnosci usera."""
    try:
        cur = g.db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT p.amount, p.currency, p.status, p.description,
                   p.paid_at, p.created_at, sp.name as plan_name
            FROM payments p
            LEFT JOIN user_subscriptions us ON us.id = p.subscription_id
            LEFT JOIN subscription_plans sp ON sp.id = us.plan_id
            WHERE p.user_id = %s
            ORDER BY p.created_at DESC
            LIMIT 50
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()

        payments = []
        for r in rows:
            payments.append({
                "amount": str(r["amount"]),
                "currency": r["currency"].strip(),
                "status": r["status"],
                "description": r["description"],
                "paid_at": r["paid_at"].isoformat() if r["paid_at"] else None,
                "created_at": r["created_at"].isoformat(),
                "plan_name": r["plan_name"],
            })

        return jsonify({"ok": True, "payments": payments})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Blad: {e}"}), 500


@app.route("/api/usage/<int:user_id>", methods=["GET"])
def get_usage(user_id):
    """Pobiera dzienne zuzycie rund usera (bez inkrementacji)."""
    try:
        cur = g.db_conn.cursor()

        # Aktualne zuzycie
        cur.execute(
            "SELECT rounds_used FROM daily_usage WHERE user_id = %s AND usage_date = CURRENT_DATE",
            (user_id,),
        )
        row = cur.fetchone()
        rounds_used = row[0] if row else 0

        # Limit z subskrypcji
        cur.execute("SELECT * FROM check_user_subscription(%s)", (user_id,))
        sub_row = cur.fetchone()
        cur.close()

        max_rounds = 0
        if sub_row and sub_row[2]:
            max_rounds = sub_row[2].get("max_rounds_per_day", 0)

        return jsonify({
            "ok": True,
            "rounds_used": rounds_used,
            "max_rounds": max_rounds,
        })
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Blad: {e}"}), 500


@app.route("/api/round/use", methods=["POST"])
def use_round():
    """Sprawdza limit rund i inkrementuje licznik zuzycia."""
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "msg": "Brak danych."}), 400

    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"ok": False, "msg": "Brak user_id."}), 400

    try:
        conn = g.db_conn
        cur = conn.cursor()
        cur.execute("SELECT * FROM check_and_increment_rounds(%s)", (user_id,))
        row = cur.fetchone()
        conn.commit()
        cur.close()

        if row:
            return jsonify({
                "ok": True,
                "allowed": row[0],
                "rounds_used": row[1],
                "max_rounds": row[2],
                "msg": row[3],
            })
        return jsonify({"ok": False, "msg": "Brak danych z funkcji."})
    except Exception as e:
        g.db_conn.rollback()
        return jsonify({"ok": False, "msg": f"Blad: {e}"}), 500


@app.route("/api/plans", methods=["GET"])
def get_plans():
    """Pobiera liste dostepnych planow subskrypcyjnych."""
    try:
        cur = g.db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT name, slug, description, price, currency, billing_period, features
            FROM subscription_plans
            WHERE is_active = true
            ORDER BY sort_order
        """)
        rows = cur.fetchall()
        cur.close()

        plans = []
        for r in rows:
            plans.append({
                "name": r["name"],
                "slug": r["slug"],
                "description": r["description"],
                "price": str(r["price"]),
                "currency": r["currency"].strip(),
                "billing_period": r["billing_period"],
                "features": r["features"],
            })

        return jsonify({"ok": True, "plans": plans})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Blad: {e}"}), 500


if __name__ == "__main__":
    print("BeSafeFish WEB - http://localhost:5000")
    app.run(debug=True, port=5000)
