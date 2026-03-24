# Deployment i architektura - BeSafeFish

## Architektura systemu

```
[Użytkownik]
    |
    |-- [Strona WWW] -----> [Serwer Render.com] -----> [Supabase PostgreSQL]
    |                         (server.py / Flask)        (baza danych)
    |
    |-- [Aplikacja .exe] --> [Serwer Render.com] -----> [Supabase PostgreSQL]
         (gui/db.py)          (te same endpointy)
```

### Przepływ danych
1. Użytkownik wchodzi na stronę WWW lub uruchamia aplikację .exe
2. Rejestracja/logowanie wysyłane jako HTTP POST (JSON) do serwera Render
3. Serwer Render łączy się z bazą Supabase (psycopg2) i zwraca odpowiedź
4. Aplikacja/strona pokazuje wynik

### Dlaczego NIE łączymy się bezpośrednio z bazą?

**Problem z bezpośrednim połączeniem:**
- Connection string (login + hasło do bazy) musiałby być w aplikacji .exe
- Każdy może wyciągnąć go z pliku wykonywalnego (np. strings, dekompilacja)
- Użytkownik musiałby ręcznie konfigurować plik .env z danymi bazy
- Zwykły użytkownik (nie programista) nie potrafi tego zrobić

**Rozwiązanie - API jako pośrednik:**
- Connection string jest TYLKO na serwerze (zmienne środowiskowe Render)
- Aplikacja .exe zna tylko adres URL serwera (publiczny)
- Serwer kontroluje co można zrobić z bazą (tylko login, rejestracja, health check)
- Użytkownik pobiera .exe, uruchamia i działa - zero konfiguracji

## Serwer (Render.com)

### Endpointy API

| Endpoint | Metoda | Opis | Kto używa |
|----------|--------|------|-----------|
| `/` | GET | Strona WWW (pliki statyczne) | Przeglądarka |
| `/api/register` | POST | Rejestracja użytkownika | Strona + GUI .exe |
| `/api/login` | POST | Logowanie użytkownika | GUI .exe |
| `/api/health` | GET | Sprawdzenie czy serwer działa | GUI .exe (init_db) |

### Parametr `source` w rejestracji
Serwer rozróżnia skąd przyszła rejestracja:
- `source: "web"` (domyślne) - rejestracja przez stronę → `created_by = Rejestracja_WEB (id=7)`
- `source: "gui"` - rejestracja przez aplikację → `created_by = Rejestracja_GUI (id=5)`

### Zmienne środowiskowe na Render
| Zmienna | Opis | Przykład |
|---------|------|---------|
| `DATABASE_URL_ADMIN` | Connection string Supabase (SEKRET) | `postgresql://user:pass@host:5432/db` |
| `WEB_SYSTEM_USER_ID` | ID użytkownika systemowego WEB | `7` |
| `GUI_SYSTEM_USER_ID` | ID użytkownika systemowego GUI | `5` |
| `PYTHON_VERSION` | Wersja Pythona na Render | `3.12` |

### Darmowy plan Render - ograniczenia
- **Usypianie**: serwis zasypia po 15 min bezczynności, pierwsze wejście ~30-50s
- **750 godzin/miesiąc**: wystarczy na 1 serwis 24/7
- **512 MB RAM**: wystarczy dla Flask
- **Brak custom domeny z SSL**: tylko `*.onrender.com`
- **Auto-deploy**: każdy push na main przebudowuje stronę

### Co gdy Render padnie?
- Aplikacja .exe nie może się zalogować ani zarejestrować
- Bot (łowienie) działa lokalnie i nie potrzebuje serwera
- Rozwiązanie: migracja na inny hosting (patrz niżej)

## Migracja na inny hosting (z Render)

### Krok po kroku
1. Skopiuj folder `versions/post_cnn/website/` na nowy serwer
2. Zainstaluj zależności: `pip install -r requirements.txt`
3. Ustaw zmienne środowiskowe: `DATABASE_URL_ADMIN`, `WEB_SYSTEM_USER_ID`, `GUI_SYSTEM_USER_ID`
4. Uruchom: `gunicorn server:app` (Linux) lub `waitress-serve server:app` (Windows)
5. **WAŻNE**: zaktualizuj `API_URL` w `gui/db.py` na nowy adres serwera
6. Przebuduj .exe i wrzuć nowy release

### Co wziąć pod uwagę
- Nowy hosting MUSI obsługiwać Python + Flask
- HTTPS jest wymagane (aplikacja wysyła hasła)
- Connection string Supabase się nie zmienia - zmienia się tylko hosting serwera
- Upewnij się że nowy serwer ma dostęp do Supabase (IPv4, poprawny region)

## Budowanie .exe (PyInstaller)

### Jak zbudować
```bash
cd versions/post_cnn
py -m PyInstaller BeSafeFish.spec --clean -y
```

### Co robi .spec
- `onedir` - tworzy folder (nie jeden plik) - mniejszy i szybciej się uruchamia
- `uac_admin=True` - Windows automatycznie wymaga uprawnień Administratora
- `excludes=['torch', 'torchvision', 'matplotlib', 'tkinter']` - wyklucza niepotrzebne biblioteki
- Dołącza: ikonę, model ONNX, pliki CNN

### Co wziąć pod uwagę
- .exe nie zawiera psycopg2 ani python-dotenv (niepotrzebne - łączy się przez API)
- .exe zawiera URL serwera w `gui/db.py` - przy zmianie serwera trzeba przebudować
- Wynikowy .zip waży ~110 MB - za duży na zwykły commit w Git (limit 100 MB)

## GitHub Releases

### Dlaczego Releases zamiast commita
- Git ma limit **100 MB** na pojedynczy plik
- GitHub Releases ma limit **2 GB** per asset
- Release jest powiązany z tagiem (wersją) - łatwo śledzić wersje
- Link do pobrania jest stały: `releases/latest/download/NazwaPliku.zip`

### Jak wrzucić nowy release
```bash
# Usuń stary release
gh release delete v1.0.0 --yes

# Stwórz nowy z plikiem
gh release create v1.0.0 \
  "versions/post_cnn/dist/BeSafeFish.zip#BeSafeFish v1.0 (Windows 11)" \
  --title "BeSafeFish v1.0" \
  --notes "Opis zmian..."
```

### Co wziąć pod uwagę
- Tag (np. `v1.0.0`) musi być unikalny
- Link `releases/latest/download/` zawsze wskazuje na najnowszy release
- Strona WWW linkuje do `releases/latest/download/BeSafeFish.zip` - nie trzeba aktualizować strony przy nowym buildzie
- Przed usunięciem starego release upewnij się że nowy .zip jest gotowy

## Supabase - pgcrypto w schema extensions

Supabase instaluje pgcrypto w schema `extensions`, nie `public`.

**W kodzie Python (zapytania bezpośrednie):**
```sql
-- TAK
SELECT ... WHERE password_hash = extensions.crypt(%s, password_hash)

-- NIE (nie zadziała)
SELECT ... WHERE password_hash = crypt(%s, password_hash)
```

**W funkcjach SQL (SECURITY DEFINER):**
```sql
CREATE FUNCTION ... SECURITY DEFINER SET search_path = public, extensions AS $$
-- tutaj crypt() działa bez prefiksu bo search_path zawiera extensions
```
