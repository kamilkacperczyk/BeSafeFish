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


def _api_request(endpoint, data=None):
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
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return {"ok": False, "msg": "Brak polaczenia z serwerem. Sprawdz internet."}
    except Exception as e:
        return {"ok": False, "msg": f"Blad polaczenia: {e}"}


def init_db():
    """Sprawdza czy serwer API dziala."""
    result = _api_request("/api/health")
    if not result.get("ok"):
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
    if len(password) < 4:
        return False, "Haslo musi miec min. 4 znaki."
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
        (success: bool, message: str)
    """
    result = _api_request("/api/login", {
        "username": username,
        "password": password,
    })
    return result.get("ok", False), result.get("msg", "Nieznany blad")
