"""
Konfiguracja aplikacji Kosa.

Wszystkie stale i parametry konfiguracyjne w jednym miejscu.
Pozniej mozna to zaladowac z pliku config.json.
"""

# --- OKNO GRY ---
# Okno gry Metin2 jest zawsze w lewym gornym rogu ekranu.
# Te wartosci trzeba dostosowac do Twojego rozmiaru okna.
# Mozesz je sprawdzic uruchamiajac: python src/screen_capture.py
GAME_WINDOW_X = 0
GAME_WINDOW_Y = 0
GAME_WINDOW_WIDTH = 1358   # szerokosc okna gry w pikselach (1376 - 2*9 ramki)
GAME_WINDOW_HEIGHT = 768   # wysokosc okna gry w pikselach

# --- MINIGRA LOWIENIE ---
# Okienko minigry pojawia sie na srodku okna gry.
# Te wartosci to PRZYBLIZONE pozycje wzgledem okna gry.
# Bedziemy je kalibrowac na zywej grze.
FISHING_BOX_X = 538        # x lewego gornego rogu okienka lowienia (wzgledem okna gry)
FISHING_BOX_Y = 288        # y lewego gornego rogu okienka lowienia (wzgledem okna gry)
FISHING_BOX_WIDTH = 279    # szerokosc okienka lowienia
FISHING_BOX_HEIGHT = 247   # wysokosc okienka lowienia

# --- DETEKCJA KOLORU OKREGU ---
# Analiza pokazala ze roznica miedzy bialym a czerwonym okregiem
# NIE jest w kolorze srodka (oba sa niebieskozielone - tlo wody)
# ale w JASNOSCI konturu okregu:
#   - Bialy okrag: ~869 pikseli z S<40, V>220 (jasny kontur)
#   - Czerwony okrag: ~192 pikseli z S<40, V>220 (ciemniejszy kontur)
# Detekcja polega na liczeniu jasnych/bialych pikseli.

# Progi dla pikseli "prawie bialych" (kontur bialego okregu)
WHITE_BRIGHT_S_MAX = 40     # max saturacja (niska = achromatyczny/bialy)
WHITE_BRIGHT_V_MIN = 220    # min jasnosc (wysoka = bardzo jasny)

# Progi decyzyjne
WHITE_CIRCLE_PIXEL_THRESHOLD = 400  # powyzej = bialy okrag (nie klikaj)
# Ponizej tego progu + aktywna minigra = czerwony okrag (klikaj!)

# Prog aktywnosci minigry - jasne piksele (gray > 200)
# Czerwony okrag: ~906, bialy okrag: ~1556, brak minigry: < 300
FISHING_ACTIVE_BRIGHT_THRESHOLD = 400

# --- KLAWISZE ---
BAIT_KEY = 'f4'            # klawisz robaka (przyneta)
CAST_KEY = 'space'         # klawisz zarzucenia wedki

# --- TIMING ---
SCAN_INTERVAL = 0.03       # co ile sekund robic screenshot (30ms = ~33 FPS)
POST_CLICK_DELAY = 0.05    # opoznienie po kliknieciu (minimalne - szybkie metody pomijaja to)
CAST_DELAY = 1.5           # opoznienie po zarzuceniu wedki (czekamy az minigra sie pojawi)
BAIT_DELAY = 0.5           # opoznienie po uzyciu robaka

# --- OKRAG W MINIGRZE ---
# Okrag lowienia wzgledem okienka lowienia (FISHING_BOX)
# Zmierzone z screenow: center=(140,137), radius=64
CIRCLE_CENTER_X = 140       # x srodka okregu (wzgledem okienka lowienia)
CIRCLE_CENTER_Y = 137       # y srodka okregu (wzgledem okienka lowienia)
CIRCLE_RADIUS = 64          # promien okregu w pikselach
CIRCLE_SAFE_MARGIN = 10     # margines bezpieczenstwa (klikamy max do r-10)
# Bezpieczny promien = 64 - 10 = 54 px od srodka

# --- FILTR NAPISOW HIT/MISS ---
# Napisy HIT/MISS pojawiajace sie po kliknieciu moga byc wykrywane jako rybka.
# Filtr oparty na analizie pikseli: HIT jest zolty (H=15-45, S>80, V>150),
# oba napisy maja jasne piksele (V>200) i wysoki gradient.
# Progi ustalone na podstawie testow 8b/8c:
#   HIT: 2.0% zoltych px, 1.6% jasnych, gradient 71.7
#   OK:  0.0% zoltych px, 0.0% jasnych, gradient 53.3
TEXT_YELLOW_H_MIN = 15      # min H dla zoltego tekstu HIT
TEXT_YELLOW_H_MAX = 45      # max H dla zoltego tekstu HIT
TEXT_YELLOW_S_MIN = 80      # min saturacja
TEXT_YELLOW_V_MIN = 150     # min jasnosc
TEXT_YELLOW_THRESHOLD = 50  # min pikseli zoltych w okregu = napis HIT
TEXT_BRIGHT_V_MIN = 220     # prog jasnosci dla detekcji tekstu (konserwatywny)
TEXT_BRIGHT_THRESHOLD = 50  # min jasnych pikseli w regionie bloba = tekst

# Detekcja tekstu MISS (lavender/jasnofioletowy)
# Analiza 12 przykladow MISS (miss1-12.png) pokazala:
#   - Tekst MISS jest jasno-fioletowy: H=85-180, S=15-110, V=185+
#   - Rybka jest oliwkowa/ciemna: H=20-60, S=150+, V=80-160
#   - Kluczowy wyroznik: saturacja jasnych pikseli (tekst S<120, rybka S>150)
TEXT_MISS_H_MIN = 85        # min H (niebieski-fioletowy zakres)
TEXT_MISS_H_MAX = 180       # max H
TEXT_MISS_S_MAX = 110       # max saturacja (tekst MISS jest wyblakly)
TEXT_MISS_V_MIN = 185       # min jasnosc
TEXT_MISS_THRESHOLD = 130   # min pikseli lavender w okregu = napis MISS (konserwatywny)
TEXT_MISS_SAT_MAX = 120     # max saturacja jasnego px w konturze (tekst vs rybka)
TEXT_MISS_LOW_SAT_RATIO = 0.30  # min proporcja jasnych px z niska saturacja

# --- GAME LOGIC ---
CLICKS_TO_WIN = 3          # ile trafien potrzeba zeby wygrac
BURST_CLICKS = 3           # ile szybkich klikniec przy wejsciu rybki w czerwony
