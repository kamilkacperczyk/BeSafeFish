# Plan restrukturyzacji BeSafeFish

**Branch:** `restructure-versions`
**Cel:** rozdzielenie monolitu `versions/post_cnn/` na `app/` (GUI + website + deploy) i `versions/tryb1_rybka_klik/{post_cnn,pre_cnn}/` (kod bota per wariant).
**Status:** W TRAKCIE

Ten plik jest zapisem planu i postepu. Po kompresji kontekstu wczytaj go zanim zaczniesz cokolwiek robic.

---

## Ustalenia z uzytkownikiem

- `app/` w rootcie: `besafefish.py`, `BeSafeFish.spec`, `BeSafeFish.bat`, `gui/`, `website/`, `SQL/`, `docs/`
- `versions/tryb1_rybka_klik/` zawiera `post_cnn/` + `pre_cnn/` (oba warianty botow tego trybu)
- Tryby nazywane od funkcji: `tryb1_rybka_klik`, `tryb2_dymek_spacja` (nie liczbowo)
- Kazdy tryb ma `README.md` (krotki opis co to jest)
- Kazdy wariant w trybie (`post_cnn`, `pre_cnn`) ma wlasny `README.md` (szczegoly techniczne - z obecnego `OPIS_WERSJI.txt`)
- `TODO.txt` **zostaje osobnym plikiem** w `versions/tryb1_rybka_klik/post_cnn/TODO.txt` (nie merge do README) - dopisac na gorze notke ze jest stary i wymaga analizy po restrukturyzacji
- `app/docs/historia-wersji.md` - nowy plik, streszczenie post+pre dla code review
- `tests/` z rootu idzie do `versions/tryb1_rybka_klik/tests/` + nowy `tests/README.md` z podzialem 17 folderow
- `render.yaml` **MUSI** byc zaktualizowany (rootDir) - osobny commit dla latwej kontroli
- Hardcoded sciezka `C:\Users\...\Kosa` w `cnn/fish_collector.py:24` - fix przy okazji
- `start_bot.bat`, `run.py`, `generate_guide_pdf.py` **zostaja w rootcie** - tylko aktualizacja sciezek

---

## Docelowa struktura

```
BeSafeFish/
  app/
    besafefish.py
    BeSafeFish.spec
    BeSafeFish.bat
    gui/
      assets/        (PUSTE na tym branchu, wypelnimy po merge add-new-fishing-games)
      fish.ico
      app.py, dashboard.py, login_screen.py, bot_worker.py, subscription_tab.py, styles.py, db.py, __init__.py
    website/
      server.py, render.yaml, index.html, img/, ...
    SQL/
    docs/
      deployment-i-architektura.md
      struktura-bazy.md
      zasady-sql.md
      regulamin-i-rodo.md
      historia-wersji.md    <- NOWY
  versions/
    tryb1_rybka_klik/
      README.md             <- NOWY
      tests/                <- z rootu
        README.md           <- NOWY
        <17 folderow>
      post_cnn/
        README.md           <- z OPIS_WERSJI.txt
        TODO.txt            <- + notka na gorze
        src/
        cnn/
        requirements.txt
      pre_cnn/
        README.md           <- z OPIS_WERSJI.txt
        src/
        requirements.txt
    tryb2_dymek_spacja/     <- NIE TWORZYMY TERAZ (Etap 3)
  requirements.txt
  README.md                 <- zaktualizowane sciezki
  CLAUDE.md                 <- zaktualizowane sciezki
  SECURITY.md
  .gitignore                <- zaktualizowane sciezki
  run.py, start_bot.bat, generate_guide_pdf.py
  RESTRUCTURE_PLAN.md       <- TEN PLIK - usunac po merge do main
```

---

## Mapowanie zmian (z research raportu agenta)

**Python imports / sys.path:**
- `cnn/fish_collector.py:19` - sys.path.insert - nowa sciezka
- `cnn/fish_collector.py:24` - hardcoded `C:\...\Kosa` - fix na Path(__file__).resolve().parent...
- `cnn/fish_labeler.py:35-36` - sys.path.insert
- `cnn/collect_fish_frames.py:19` - sys.path.insert
- `besafefish.py:18-19` - BASE_DIR (po przeniesieniu do app/ powinien nadal dzialac bez zmian)
- `gui/app.py:15-16` - `from gui.login_screen`, `from gui.dashboard` (bez zmian bo gui/ to podfolder app/)

**PyInstaller `BeSafeFish.spec`:**
- linia 10: `['besafefish.py']` - bez zmian (spec i besafefish.py oba w app/)
- linia 11: `pathex=['.']` - dodac sciezke do `versions/tryb1_rybka_klik/post_cnn` zeby importer znalazl src/, cnn/
- linia 14: `('gui\\fish.ico', 'gui')` - zostaje jak jest bo gui/ w app/
- linia 15-16: `('cnn\\models\\fish_patch_cnn.onnx', 'cnn\\models')` - zmienic na `('..\\versions\\tryb1_rybka_klik\\post_cnn\\cnn\\models\\fish_patch_cnn.onnx', 'cnn\\models')`
- linia 46: `icon=['gui\\fish.ico']` - bez zmian

**.gitignore:**
- linia 41: `!versions/post_cnn/website/img/*.png` -> `!app/website/img/*.png`
- linia 46-48: `versions/post_cnn/cnn/{data,cache,patches}/` -> `versions/tryb1_rybka_klik/post_cnn/cnn/{data,cache,patches}/`
- (jezeli sa) `!versions/post_cnn/cnn/models/*.onnx*` -> `!versions/tryb1_rybka_klik/post_cnn/cnn/models/*.onnx*`

**`render.yaml` (website/render.yaml):**
- `rootDir: versions/post_cnn/website` -> `rootDir: app/website`
- **KRYTYCZNE** - osobny commit

**Docs markdown:**
- `docs/deployment-i-architektura.md` linie 84, 88, 94, 113 - sciezki
- `CLAUDE.md` (root) linie 9-13 - tabela docs
- `README.md` (root) - sprawdzic sciezki
- `generate_guide_pdf.py` linie 357, 1078 - teksty PDF

**Skrypty:**
- `start_bot.bat` - zaktualizowac sciezki do run.py albo nowych entrypointow
- `run.py` - sprawdzic co tam jest

**Website `server.py`:**
- linia 19, 88 - komentarze ze sciezkami

**OPIS_WERSJI.txt (oba):**
- konwersja na README.md, oryginal usuniety
- linie 148-150 w post_cnn/OPIS_WERSJI.txt - hardcoded sciezki (przepisac/uproscic)

---

## Plan commitow

### [ ] Commit 1: chore: usuniecie artefaktow build przed restrukturyzacja
- `rm -rf versions/post_cnn/build/ versions/post_cnn/dist/`
- NIE usuwac __pycache__ ani venv (sa juz w .gitignore)

### [ ] Commit 2: refactor: wydzielenie app/ z versions/post_cnn/
- `mkdir app/`
- przeniesc: `gui/`, `website/`, `SQL/`, `besafefish.py`, `BeSafeFish.spec`, `BeSafeFish.bat` -> `app/`
- zaktualizowac `BeSafeFish.spec` (datas, pathex)
- zaktualizowac `.gitignore` (nowe sciezki - website)
- smoke test: `py app/besafefish.py` musi sie uruchomic (nawet jak bot nie znajdzie importow to przynajmniej GUI ma wstac)

### [ ] Commit 3: refactor: przeniesienie trybu rybka-klik do versions/tryb1_rybka_klik/
- `mkdir -p versions/tryb1_rybka_klik/post_cnn versions/tryb1_rybka_klik/pre_cnn`
- przeniesc: `versions/post_cnn/{src,cnn,requirements.txt,TODO.txt,OPIS_WERSJI.txt}` -> `versions/tryb1_rybka_klik/post_cnn/`
- przeniesc: `versions/pre_cnn/*` -> `versions/tryb1_rybka_klik/pre_cnn/`
- usunac puste: `versions/post_cnn/`, `versions/pre_cnn/`
- zaktualizowac sys.path w: `cnn/fish_collector.py`, `cnn/fish_labeler.py`, `cnn/collect_fish_frames.py`
- fix hardcoded `C:\...\Kosa` w `cnn/fish_collector.py:24`
- zaktualizowac `.gitignore` (sciezki do cnn/data, cnn/cache, cnn/patches)
- zaktualizowac `BeSafeFish.spec` (pathex dodac nowa sciezke, datas cnn\\models)
- smoke test: bot musi sie zaimportowac przy starcie z GUI

### [ ] Commit 4: refactor: tests/ -> versions/tryb1_rybka_klik/tests/
- `git mv tests/ versions/tryb1_rybka_klik/tests/`
- stworzyc `versions/tryb1_rybka_klik/tests/README.md` z podzialem folderow na kategorie:
  * Analizy: analiza_kolorow, analiza_pikseli, analyze_bad_frames, analyze_miss
  * Kalibracja: calibrate, test_kolory, test5_kolory
  * Diagnostyka: diagnostyka, diagnostyka_gra, diagnostyka_live
  * Testy trackingu/detekcji: test8a_tracking, test8b_miss, test8c_hit, test10_clean, test9_long, test_filter
  * Walidacja: walidacja_detektora

### [ ] Commit 5: docs: aktualizacja dokumentacji + README dla trybow
- `app/docs/*.md` - aktualizacja sciezek w tresci
- `app/docs/historia-wersji.md` - NOWY (streszczenie post+pre na 1-2 stronach)
- `versions/tryb1_rybka_klik/README.md` - NOWY (co to jest tryb, jak dziala w skrocie, linki do wariantow)
- `versions/tryb1_rybka_klik/post_cnn/README.md` - z OPIS_WERSJI.txt (formatowanie markdown)
- `versions/tryb1_rybka_klik/pre_cnn/README.md` - z pre_cnn/OPIS_WERSJI.txt
- `versions/tryb1_rybka_klik/post_cnn/TODO.txt` - dodac na **samej gorze pliku** (pierwsze linie, PRZED obecnym naglowkiem `================ TODO - WERSJA POST CNN ===`) notke:
  ```
  ================================================================================
    UWAGA: Ten TODO pochodzi sprzed restrukturyzacji repo (2026-04-24).
    Wymaga analizy — sciezki, numeracja i priorytety moga byc nieaktualne.
    Po zweryfikowaniu — ta sekcja do usuniecia.
  ================================================================================
  
  ```
  Oryginalna tresc zostaje bez zmian pod spodem.
- root `README.md` - zaktualizowane sciezki
- `CLAUDE.md` (root) - zaktualizowana tabela docs
- usunac `OPIS_WERSJI.txt` z obu wariantow (zostaly zamienione na README)

### [ ] Commit 6: fix(deploy): render.yaml rootDir -> app/website
- tylko ta jedna linia
- oddzielny commit dla czystej kontroli

### [ ] Commit 7: chore: aktualizacja start_bot.bat, run.py, generate_guide_pdf.py
- `start_bot.bat` - sciezki
- `run.py` - jesli importuje z post_cnn
- `generate_guide_pdf.py` - teksty PDF

### [ ] Commit 8: chore: usuniecie RESTRUCTURE_PLAN.md po zakonczeniu refactoru
- po wszystkich smoke testach

---

## Smoke testy

**Po Commit 2 (app/):**
- [ ] `py app/besafefish.py` - okno logowania sie pojawia
- [ ] Import `from gui.app import BeSafeFishApp` dziala

**Po Commit 3 (tryb1):**
- [ ] Logowanie -> klik START -> bot moze wystartowac. Blad "nie znaleziono okna Metin2" jest OK (to znaczy ze bot sie zaimportowal)
- [ ] PyInstaller build: `cd app && pyinstaller BeSafeFish.spec` przechodzi bez bledu

**Po Commit 5 (docs):**
- [ ] `git grep versions/post_cnn` w tresci docs nic nie zwraca (wszystko zmienione)

**Koncowy:**
- [ ] `git grep -r "versions/post_cnn"` w calym repo zwraca tylko `RESTRUCTURE_PLAN.md` (ktory usuniemy)
- [ ] `py app/besafefish.py` - pelny flow logowania + klik START z Trybem 1 -> bot startuje (nawet jak blad braku Metin2)

---

## Po merge do main

- Render automatycznie redeployuje strone (nowy rootDir=app/website wykryty)
- Sprawdz online: strona dalej dziala?
- Checkout `add-new-fishing-games`:
  - `git rebase main` - beda konflikty (te same pliki ruszylismy, plus moje zmiany z Etapu 1)
  - Rozwiazac konflikty, force-push
  - Alternatywa: cherry-pick commity z add-new-fishing-games na swiezego brancha

---

## Post-mortem (wypelnic po zakonczeniu)

- Jakie niespodzianki wystapily?
- Ktore commity wymagaly poprawki?
- Ile czasu zajelo?
- Co przydaloby sie zapisac do memory na przyszlosc?
