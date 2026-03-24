"""
Baza uzytkownikow BeSafeFish - polaczenie przez API serwera.

Aplikacja desktopowa NIE laczy sie bezposrednio z baza danych.
Zamiast tego wysyla requesty do backendu na Render (server.py),
ktory obsluguje polaczenie z Supabase.

Dzieki temu uzytkownik nie musi konfigurowac .env ani znac
connection stringa do bazy - wystarczy pobrac i uruchomic .exe.
"""

import json
import urllib.request
import urllib.error

API_URL = "https://kosa-h283.onrender.com"


def _api_request(endpoint, data=None, timeout=60):
    """Wysyla request do API serwera BeSafeFish."""
    url = f"{API_URL}{endpoint}"
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
    else:
        req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return {"ok": False, "msg": "Brak polaczenia z serwerem. Sprawdz internet."}
    except Exception as e:
        return {"ok": False, "msg": f"Blad polaczenia: {e}"}


def init_db():
    """Sprawdza czy serwer API dziala. Retry przy cold start Render."""
    import time

    for attempt in range(3):
        result = _api_request("/api/health", timeout=60)
        if result.get("ok"):
            return
        if attempt < 2:
            time.sleep(2)

    raise RuntimeError(
        "Nie mozna polaczyc z serwerem BeSafeFish. Sprawdz polaczenie z internetem."
    )


def register_user(username: str, email: str, password: str) -> tuple:
    """
    Rejestruje nowego uzytkownika przez API serwera.

    Returns:
        (success: bool, message: str)
    """
    if len(username) < 3:
        return False, "Nazwa uzytkownika musi miec min. 3 znaki."
    if len(password) < 8:
        return False, "Haslo musi miec min. 8 znakow."
    if len(password) > 64:
        return False, "Haslo moze miec maks. 64 znaki."
    if not email:
        return False, "Podaj adres email."

    result = _api_request("/api/register", {
        "username": username,
        "email": email,
        "password": password,
        "source": "gui",
    })
    return result.get("ok", False), result.get("msg", "Nieznany blad")


def authenticate_user(username: str, password: str) -> tuple:
    """
    Loguje uzytkownika przez API serwera.

    Returns:
        (success: bool, message: str, user_id: int|None, subscription: dict|None)
    """
    result = _api_request("/api/login", {
        "username": username,
        "password": password,
    })
    return (
        result.get("ok", False),
        result.get("msg", "Nieznany blad"),
        result.get("user_id"),
        result.get("subscription"),
    )


def get_subscription(user_id: int) -> dict:
    """Pobiera aktualna subskrypcje usera."""
    return _api_request(f"/api/subscription/{user_id}")


def get_payments(user_id: int) -> dict:
    """Pobiera historie platnosci usera."""
    return _api_request(f"/api/payments/{user_id}")


def get_plans() -> dict:
    """Pobiera liste dostepnych planow."""
    return _api_request("/api/plans")


def get_daily_usage(user_id: int) -> dict:
    """Pobiera dzienne zuzycie rund (bez inkrementacji)."""
    return _api_request(f"/api/usage/{user_id}")


def use_round(user_id: int) -> dict:
    """Sprawdza limit rund i inkrementuje licznik. Zwraca allowed/rounds_used/max_rounds/msg."""
    return _api_request("/api/round/use", {"user_id": user_id})
