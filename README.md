# BeSafeFish

Asystent computer vision automatyzujacy czynnosci w grze - aplikacja desktopowa + rejestracja przez strone WWW + baza w chmurze (PySide6, Flask API, PostgreSQL, ONNX).

- Strona WWW: https://kosa-h283.onrender.com
- Pobranie .exe: [GitHub Releases](https://github.com/kamilkacperczyk/Kosa/releases/latest)

## Jak dziala

1. Przechwytuje region mini-gry z ekranu (mss)
2. Rozpoznaje stan: bialy okrag (czekaj), czerwony okrag (klikaj), HIT/MISS (ignoruj)
3. Lokalizuje rybke (background subtraction + PatchCNN ONNX)
4. Klika w rybke gdy okrag jest czerwony (pydirectinput)

## Architektura

```
[Uzytkownik]
    |-- [Aplikacja .exe] --HTTP/JSON--> [Flask API (Render.com)] --psycopg2--> [Supabase PostgreSQL]
    |-- [Strona WWW]     --HTTP/JSON--> [Flask API (Render.com)] --psycopg2--> [Supabase PostgreSQL]
```

Aplikacja .exe NIE laczy sie bezposrednio z baza - komunikuje sie przez API serwera.

## Struktura repo

```
versions/
  pre_cnn/              -- Starsza wersja (klasyczne CV, bez CNN)
  post_cnn/             -- Aktualna wersja
    src/                -- Bot: przechwytywanie ekranu, detekcja, klikanie
    cnn/                -- Modele CNN (PatchCNN, FishNet), trening, inferencja
    gui/                -- GUI PySide6 (login, rejestracja, dashboard)
    website/            -- Strona WWW + Flask backend (server.py)
    docs/               -- Dokumentacja (deployment, struktura bazy, architektura CNN)
    SQL/                -- Definicje tabel i funkcji PostgreSQL
    BeSafeFish.spec     -- Konfiguracja PyInstaller
    besafefish.py       -- Entry point GUI
```

## Uruchomienie (deweloper)

```bash
cd versions/post_cnn
pip install -r requirements.txt
py besafefish.py
```

## Budowanie .exe

```bash
cd versions/post_cnn
py -m PyInstaller BeSafeFish.spec --clean -y
```

Wynik: `dist/BeSafeFish/` - spakuj jako .zip i wrzuc do GitHub Releases.

## Baza danych

PostgreSQL na Supabase. 7 tabel: users, subscription_plans, user_subscriptions, payments, daily_usage, login_history, audit_log.

Szczegoly: `versions/post_cnn/docs/struktura-bazy.md`

## Dokumentacja

| Plik | Opis |
|------|------|
| `versions/post_cnn/docs/deployment-i-architektura.md` | Architektura, API, Render, PyInstaller, migracja |
| `versions/post_cnn/docs/struktura-bazy.md` | Diagram zaleznosci tabel, funkcje, triggery |
| `versions/post_cnn/cnn/ARCHITEKTURA_CNN.md` | Architektura sieci CNN, pipeline treningowy |
| `SECURITY.md` | Polityka bezpieczenstwa, pre-commit hook |
