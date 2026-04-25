# Architektura BeSafeFish - lekcje i wnioski

Zapis dwoch duzych refaktorow z 2026-04-24:
- **Restrukturyzacja repo** (monolit `versions/post_cnn/` -> `app/` + `versions/<tryb>/<wariant>/`)
- **Strategy pattern dla trybow minigry** (Etap 2: wydzielenie `FishClickMode` z `KosaBot`)

Cel dokumentu: nie powtorzyc tych samych bledow przy nastepnych funkcjach.

---

## Najwazniejsza lekcja

**Modularna architektura nie jest "wlasciwoscia ktora dodajesz pozniej".**
Kazda decyzja podejmowana w pierwszych tygodniach projektu zostawia slady,
ktore za 6 miesiecy beda kosztowac dni roboty na refaktor.

Konkretne tego dowody w naszym repo:
- `versions/post_cnn/` byl monolitem (GUI + bot + website + docs + SQL).
  Rozdzielenie zajelo 8 commitow i wykrylo ~45 miejsc z hardcoded sciezkami,
  w tym konfig produkcyjny Render.com.
- `KosaBot.play_fishing_round()` mial 110 linii konkretnej logiki rybki-klik
  zaszytej w glownej petli. Dodanie drugiego trybu wymagalo refaktoru tego
  zanim w ogole zaczelismy pisac drugi tryb.

---

## Konkretne zasady (wynikaja z bolu)

### 1. Rozdzielaj **co sie zmienia osobno** od dnia zero

W BeSafeFish trzy rzeczy zmieniaja sie z roznym tempem:
- **GUI / website / SQL** - rzadko, stabilnie
- **Bot per tryb minigry** - czesto, tryb po trybie
- **Docs / deployment** - rzadko, ale zalezy od reszty

Trzymanie ich razem (`versions/post_cnn/` zawieral wszystkie 3) to bolaczka.

**Zasada:** Jesli widzisz ze A i B zmieniaja sie z innym tempem - **rozdziel
je w pierwszej iteracji**, nie obiecuj "jak bedzie potrzeba to wydzielimy".

### 2. Strategia "snapshot wersji" (kopia folderu) ma swoje miejsce, ale...

Pierwotnie projekt mial podejscie: kazda duza zmiana = kopia calego folderu
(`versions/pre_cnn/` -> `versions/post_cnn/`). To **dobre** dla:
- prostej historii kontrahenta widzac roznice
- mozliwosci rollback'u na konkretna wersje
- zachowywania "wzorca" do porownania

Ale **zle** dla:
- wspolnej infrastruktury (GUI, baza, deploy)
- czesci ktore powinny byc wspoldzielone miedzy snapshotami

**Wniosek:** Snapshot-versions OK dla **kodu specyficznego dla wariantu
bota**. NIE dla GUI, SQL, deploymentu, dokumentacji projektu. Te zostaja
w jednym miejscu, tworza "appke" ponad snapshotami.

### 3. Hardcoded sciezki to czas-bomba

Spotkalem w refaktorze:
- `r'C:\Users\Szpanerski Kaloryfer\Desktop\Repos\Kosa'` w 3 plikach Pythona
  (Repo poprzednio sie nazywalo Kosa - po rename ten kod nie dziala)
- `versions/post_cnn/website` w `render.yaml` (wtedy poprawnie, po
  restrukturyzacji - psul produkcyjny deploy)
- Sciezki w docs (`cd versions/post_cnn` w przykladach)

**Zasada:** Sciezki w kodzie zawsze relatywne wzgledem `__file__`:

```python
# zle
BASE = Path(r'C:\Users\X\Repos\MyProject')

# dobrze
BASE = Path(__file__).resolve().parents[N]   # N = ile poziomow do roota
```

**Bonus:** Sciezki w docs/komentarzach tez zachecam pisac wzgledem **roota
repo**, nie absolutnymi - bo absolutne pokazuja czyje to repo.

### 4. Strategy pattern dla wariantow zachowania - **lekko, nie sztywno**

Kontrakt `FishingMode` w naszym kodzie to **`Protocol`**, NIE `ABC` z
`@abstractmethod`. Roznica:

- **ABC z abstractami:** kazdy tryb MUSI dziedziczyc z `FishingMode` i
  zaimplementowac wszystkie metody. Ale po dziedziczeniu zaczynaja sie
  zaleznosci miedzy klasami, problem MRO, trudniej napisac tryb od zera.

- **Protocol (duck typing):** tryb po prostu **ma** wymagane metody.
  Nie dziedziczy z niczego. Test: czy `bot.mode.play_round()` dziala -
  jak tak, to OK.

W praktyce dla nas: dodanie Trybu 2 (dymek-spacja) to **napisanie
nowego pliku od zera** + dopisanie 2 linii w `bot.py`. Zero zaleznosci
do FishClickMode.

**Zasada:** Im lzejszy kontrakt, tym latwiej dolozyc nowe warianty.
Dzieki temu `Tryb 2` moze miec **calkowicie inna detekcje, akcje,
zaleznosci** - byle wystawial te same 4 metody na zewnatrz.

### 5. **Dependency injection** zamiast self-discovery

Stary `KosaBot` sam tworzyl `ScreenCapture()`, `FishingDetector()`,
`InputSimulator()` w `__init__`. Po wydzieleniu `FishClickMode` te
wspoldzielone narzedzia (capture, input) sa **wstrzykiwane** z `KosaBot`,
a wlasne (FishingDetector, PatchCNN ONNX) tryb laduje sam.

```python
class FishClickMode:
    def __init__(self, capture, input_sim, log_callback, is_running, ...):
        self._capture = capture          # wstrzykniete z KosaBot
        self._input = input_sim          # wstrzykniete z KosaBot
        self.detector = FishingDetector()  # wlasne, tryb-specific
```

Korzysc: testowalnosc (mockujesz capture w testach), dzielenie zasobow
(jeden ScreenCapture dla calego bota, nie 1 per tryb), czystszy podzial
"co tu jest moje, co dzielone".

### 6. **Callable dla flag stanu** zamiast wspoldzielonego obiektu

Tryb potrzebuje wiedziec czy bot ma sie zatrzymac (uzytkownik wcisnal
STOP, klawisz 'q' itp). Pierwsza pokusa: przekazac caly `KosaBot`
do trybu, zeby tryb mogl czytac `bot.running`. Zle - tworzy zaleznosc
cykliczna, tryb zna za duzo o boacie.

Lepiej: przekazac **dwa callable**:
```python
is_running: Callable[[], bool]    # tryb sprawdza: while self._is_running():
request_stop: Callable[[], None]  # tryb prosi: self._request_stop()
```

Tryb nie wie nic o KosaBot - tylko ze ma 2 funkcje do wolania. Latwo
przetestowac (mockujesz callbacky), latwo zmienic implementacje
(np. flagi w bocie -> Event z threading - bez ruszania trybu).

### 7. **YAGNI** wygrywa z "futureproofingiem"

Mialem pokuse zrobic kontrakt obejmujacy wszystkie potencjalne minigry
od razu (`play_continuously()` jako alternatywa dla `play_round()`,
zaawansowane stage'y etc). Ostatecznie: zrobilem 4 metody, tyle ile
**dzis** trzeba.

Powod: nie wiem **jakich** trybow bedziesz dodawac. Refactor pod
**konkretny** przyklad jest 10x lepszy niz refactor pod hipoteze. Bo
widzac konkretny przypadek wiesz czego potrzeba; zgadujac, projektujesz
"elastycznie" cos co potem nie pasuje i tak.

**Zasada:** Buduj pod **najblizsze 1-2 nowe przypadki**, nie pod
"wszystkie mozliwe". Zostaw notatke (`TODO-aplikacja.md`) gdy widzisz
ze cos w przyszlosci moze trzeba zmienic - i wracaj do tego dopiero
gdy bedzie konkret.

---

## Konkretne pulapki techniczne ktore wystapily

### Sparse-checkout blokuje `git add` po rename katalogu
Po `git mv` duzego katalogu (np. `versions/post_cnn/` -> `versions/tryb1_rybka_klik/post_cnn/`)
git probuje aktualizowac sparse-checkout konfiguracje - i przez to **niektore
modyfikacje plikow w nowej lokalizacji NIE wchodza do commita** (cicho
ignorowane, ale `git status` pokazuje "nothing to commit").

**Diagnoza:** `git ls-files <plik>` pokazuje ze plik jest sledzony, ale
`git diff` nic nie pokazuje. Jak takie cos widzisz - sprawdz `git config
core.sparseCheckout`.

**Rozwiazanie:** `git rm --sparse <plik>` lub `git add --sparse <plik>`.
Nawet jak `core.sparseCheckout=false`. Niespojnosc git, ale to dziala.

### Render.yaml NIE jest synchronizowany automatycznie z dashboardem
Plik `render.yaml` w repo ma zastosowanie **tylko przy pierwszym setupie**
serwisu. Pozniej Render trzyma config we **wlasnej bazie danych**
(Dashboard > Settings) i ignoruje zmiany w `render.yaml`.

**Konsekwencja:** Jak zmieniasz `rootDir` w `render.yaml` - zmien tez
recznie w Dashboard, inaczej Render bedzie ciagle uzywal starej sciezki.

### PyInstaller spec - sciezki wzgledem **CWD** uruchomienia, nie pliku .spec
Sciezki w `datas`, `pathex`, `icon` w `BeSafeFish.spec` sa wzgledem
**katalogu z ktorego uruchamiasz `pyinstaller`**, nie wzgledem pliku
`.spec`. Mylace.

W naszym repo: `BeSafeFish.spec` jest w `app/`, ale uruchamiamy go z
**rootu repo** (`py -m PyInstaller app/BeSafeFish.spec`), wiec sciezki
zaczynaja sie `app\\gui\\fish.ico`, `versions\\tryb1_rybka_klik\\...`
itd. Gdyby ktos uruchomil z `app/`, sciezki musialyby byc inne.

### Edit tool a literalne `\uXXXX` w stringach Pythona
Kilka plikow w repo ma stringi typu `"▶ START"` zamiast emoji `▶`.
Edit tool czasem nie znajduje takich stringow gdy w `old_string` jest
wpisany rzeczywisty znak Unicode. Przed edytowaniem - sprawdz czy plik
ma literalne `\u...` czy znaki, dopasuj `old_string` odpowiednio.

### QThread.run() z `try/finally` - nie dublowac emitow w early-return
W `bot_worker.py::BotWorker.run()` blok `finally` zawsze emituje
`status_changed("stopped")` + `finished_signal`. Jak dodajesz `return`
wczesniej (np. brak admina) - **nie wolaj recznie** `finished_signal.emit()`,
inaczej slot odpali sie 2 razy (objawiajace sie dublowanymi liniami w logu).

---

## Pliki ktore warto przejrzec na poczatku nowego projektu

Jak zaczynasz **nowy projekt** od zera, w pierwszych godzinach zaplanuj:

1. **Strukture katalogow** osobno dla:
   - "appki" (entrypoint, GUI, deployment, docs, baza)
   - "logiki domenowej" (kazdy wariant w swoim folderze, jesli przewidujesz wiele)
   - "testow / analiz" (ludzkie testy, eksperymenty - osobno od kodu produkcyjnego)

2. **Kontrakty (Protocols, interfaces)** dla glownych miejsc rozszerzen.
   Niekoniecznie sztywne ABC - protokol z 3-5 metodami wystarczy.

3. **Konfiguracje sciezek** zawsze relatywnie do `__file__`. Hardcoded
   absolutna sciezka = bug.

4. **Ignored files** (`.gitignore`) - przygotuj listy juz na poczatku
   (build artefakty, logi, dane uzytkownika, sekrety).

5. **CI/CD config** - jesli uzywasz Render/Vercel/podobny - zapisz w repo,
   ale pamietaj ze Dashboard hostingu to **inne zrodlo prawdy**.

6. **Docstring na plikach kontraktow** - dlaczego tak, nie inaczej.
   Za 6 miesiecy zapomnisz.

---

## Wnioski "meta"

- **Refactor duzy = duzo malych commitow**, kazdy ze smoke testem. Nie
  jeden gigantyczny "refactor everything".
- **Plan na dysku** (`RESTRUCTURE_PLAN.md`, `ETAP2_PLAN.md`) ratuje przy
  kompresji kontekstu / przelaczeniu sesji. Tymczasowe pliki ktore
  usuwasz po zakonczeniu - ale w trakcie sa kluczowe.
- **Smoke testy lokalne** po kazdym commicie - jeden plik Python,
  10-15 linii, weryfikuje ze import + konstrukcja klas dziala. Nie
  zastapi unit testow, ale lapie 80% regresji w 5 sekund.
- **Snapshot starego stanu** (`old_structure` branch) zanim zaczynasz
  duzy refactor. Tani backup.
