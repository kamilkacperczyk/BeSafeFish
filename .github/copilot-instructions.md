# Instrukcje bezpieczeństwa dla GitHub Copilot Agent

## BEZWZGLĘDNE ZAKAZY (NIGDY nie łam tych zasad)

### 1. Ograniczenie do workspace
- Agent operuje WYŁĄCZNIE w obrębie: `C:\Users\REDACTED-USER-PATH\Desktop\Repos\Kosa\`
- **NIGDY** nie czytaj, nie otwieraj, nie przeszukuj plików poza tym katalogiem
- **NIGDY** nie używaj ścieżek typu `~/.ssh/`, `~/.aws/`, `~/.gnupg/`, `~/.config/`, `%USERPROFILE%/.git-credentials`
- **NIGDY** nie nawiguj do katalogów nadrzędnych (`..`, `../../` itp.) w celu dostępu do plików poza workspace

### 2. Zakaz dostępu do wrażliwych danych
Nigdy nie czytaj, nie wyświetlaj, nie kopiuj:
- Kluczy SSH (`id_rsa`, `id_ed25519`, `*.pem`, `*.key`)
- Tokenów API, haseł, sekretów
- Plików `.env` z danymi uwierzytelniającymi
- Git credentials (`~/.git-credentials`, credential store)
- Certyfikatów (`*.crt`, `*.p12`, `*.pfx`)
- Kluczy GPG/PGP
- Cookies, session tokens, plików przeglądarki

### 3. Zakaz komend niebezpiecznych
Nigdy nie uruchamiaj:
- `Get-Content`, `type`, `cat` na plikach poza workspace
- `Get-ChildItem` / `ls` / `dir` na katalogach poza workspace
- `$env:*` aby odczytywać zmienne środowiskowe z sekretami (np. TOKEN, SECRET, PASSWORD, API_KEY)
- Komend sieciowych wysyłających dane na zewnątrz (curl POST z danymi, Invoke-WebRequest z body)
- `git clone` do innych repozytoriów bez wyraźnej prośby użytkownika

### 4. Bezpieczeństwo commitów
Przed każdym `git add` / `git commit`:
- Sprawdź czy żaden plik nie zawiera haseł, tokenów, kluczy
- Sprawdź czy `.gitignore` poprawnie wyklucza pliki wrażliwe
- Nie commituj plików binarnych bez wyraźnej prośby
- Nie commituj plików `.env`, `*.secret`, `*.key`

### 5. Zasada minimalnych uprawnień
- Używaj tylko tych narzędzi które są niezbędne do wykonania zadania
- Nie instaluj pakietów bez wyraźnej potrzeby
- Nie modyfikuj konfiguracji systemowej
- Nie zmieniaj uprawnień plików/katalogów

## DOZWOLONE OPERACJE
- Czytanie/edycja plików w workspace Kosa
- Uruchamianie skryptów Python z workspace
- Instalacja pakietów Python z requirements.txt
- Git operacje na repo Kosa
- Uruchamianie HTTP serwerów na localhost (do testów)
