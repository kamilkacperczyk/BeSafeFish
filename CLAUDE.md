# Instrukcje dla AI — projekt (BeSafeFish)

## Dokumentacja projektu

Przed wprowadzaniem zmian przeczytaj odpowiedni plik:

| Plik | Temat | Czytaj gdy... |
|------|-------|---------------|
| `app/docs/deployment-i-architektura.md` | Endpointy API, serwer, gunicorn, Render, PyInstaller | ...pracujesz nad server.py, deploymentem, .exe |
| `app/docs/struktura-bazy.md` | Tabele, funkcje SQL, diagram, przeplywy biznesowe | ...pracujesz nad SQL, dodajesz tabele/funkcje |
| `app/docs/zasady-sql.md` | Konwencje nazewnictwa, checklist zmian w bazie, zapytania diagnostyczne | ...zmieniasz kolumne/tabele/funkcje/enum w SQL |
| `app/docs/regulamin-i-rodo.md` | Regulamin, RODO, checkbox akceptacji, retencja danych | ...dodajesz rejestracje, zbierasz dane osobowe, IP |
| `app/docs/historia-wersji.md` | Ewolucja bota (pre_cnn → post_cnn), co zmienilo sie i dlaczego | ...chcesz zrozumiec rozniace miedzy wariantami bota |
| `app/docs/architektura-i-lekcje.md` | Lekcje z refaktorow (modularnosc, Strategy pattern, hardcoded paths) | ...projektujesz nowa funkcjonalnosc, decydujesz o strukturze |
| `versions/tryb1_rybka_klik/README.md` | Opis trybu rybka-klik i jego wariantow | ...pracujesz nad trybem 1 (lowienie rybki) |
| `versions/tryb1_rybka_klik/post_cnn/README.md` | Szczegoly wariantu post_cnn (pipeline detekcji, PatchCNN) | ...modyfikujesz kod bota z CNN |
| `versions/tryb1_rybka_klik/post_cnn/cnn/ARCHITEKTURA_CNN.md` | Model PatchCNN, trening, inference, ONNX | ...pracujesz nad CNN/modelem |
| `SECURITY.md` | Zasady bezpieczenstwa | ...dotykasz auth, hasel, connection stringow |

**WAZNE**: Po dodaniu tabeli, funkcji SQL lub endpointu API — zaktualizuj odpowiedni plik docs!

---

## Zasady pracy

### Bezpieczenstwo
- NIGDY nie commituj hasel, tokenow, kluczy, connection stringow
- Klient (GUI/.exe/web) NIGDY nie laczy sie bezposrednio z baza — zawsze przez API
- Sekrety: zmienne srodowiskowe lub .env (w .gitignore)

### Konwencje
- Commit message po polsku: `<typ>: <opis>` (feat, fix, refactor, docs, chore)
- NIE dodawaj Co-Authored-By Claude
- Wersja musi byc spojna: GUI footer (dashboard.py), strona (index.html), GitHub Release tag
- `py` zamiast `python` (Windows)

---

## Wypracowane zasady techniczne

### 1. GUI desktopowe (PySide6 / Qt)

1. **Kazda operacja sieciowa w osobnym watku** — nigdy na main thread. Uzytkownik widzi zamrozony ekran.
2. **Stan lokalny zamiast API** gdzie dane zmieniaja sie przewidywalnie (+1, timer). API tylko na start/logowanie.
3. **Reset stanu przy restarcie akcji** — liczniki, labele, flagi wracaja do wartosci poczatkowych.
4. **Health check przy starcie w tle** — pokaz "Laczenie..." + retry, nie bialy ekran.
5. **Feedback wizualny dla kazdej operacji** — uzytkownik musi widziec ze cos sie dzieje.

### 2. Bot / automatyzacja z zewnetrznymi serwisami

6. **Fail-open** — blad API = kontynuuj dzialanie, loguj ostrzezenie. Bot NIE moze sie zatrzymac z powodu bledu sieci.
7. **Krotki timeout per-operacja** (3-5s), dlugi na start/logowanie (30-60s).
8. **Callback pattern** — core bota nie importuje HTTP. Warstwa integracji (worker) obsluguje API i decyduje.
9. **Try/except wokol kazdego callbacka** — wyjatek w callbacku nie moze crashnac glownej petli.

### 3. Flask / backend API

10. **Connection pooling ZAWSZE** — ThreadedConnectionPool (psycopg2) lub odpowiednik. Nigdy connect/close per request.
11. **Polaczenie z poola w request lifecycle** — before_request (pobierz), teardown_request (zwroc, rollback on error).
12. **Gunicorn z gthread** — wielowatkowosc w jednym procesie. workers=N, threads=2, preload.
13. **Timeout adekwatny do hostingu** — darmowy hosting z cold startem: 120s. Produkcja: 30s.

### 4. SQL / bazy danych

14. **Lazy expiration zamiast crona** — sprawdzaj przy kazdym uzyciu, nie scheduluj. Prostsze, nie wymaga infry.
15. **Atomowe operacje** — sprawdzenie + modyfikacja w jednej funkcji/transakcji. Nie SELECT + UPDATE osobno (race condition).
16. **Auto-assign domyslnych wartosci przy rejestracji** — nowy user = darmowy plan/rola od razu.
17. **SECURITY DEFINER + search_path** — ustaw search_path explicite (public, extensions), uzywaj session_user.

### 5. DevOps / release

18. **Duze pliki -> Releases** — nie commituj binarek >100MB. GitHub Releases.
19. **Deploy triggerowany zmianami w rootDir** — commit poza rootDir nie odpali redeployu na Render.
20. **Retry z backoff na start** — przy cold start serwera, GUI robi 3 proby z rosnacym odstepem.
21. **Rollback transakcji przy bledzie** — teardown_request z rollback. Nie zostawiaj otwartych transakcji.
