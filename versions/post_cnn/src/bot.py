"""
Glowna petla bota Kosa — wersja POST CNN.

Obsluguje dwa tryby detekcji:
- CNN (domyslny): FishNet — lekka siec neuronowa, <5ms/klatke, ONNX Runtime
- Klasyczny (fallback): background subtraction + filtry HSV

CNN eliminuje problemy wersji PRE CNN:
- Rozpoznaje napisy HIT/MISS jako osobne klasy (zero false positives)
- Dziala na POJEDYNCZEJ klatce (bez historii/bufora/warmup)
- Stala skutecznosc niezalezna od czasu trwania gry

Cykl zycia jednej rundy:
1. Uzyj robaka (F4)
2. Zarzuc wedke (SPACJA)
3. Czekaj az okienko "Lowienie" sie pojawi
4. Skanuj okrag:
   - Czerwony = kliknij LPM (rybka w srodku)
   - Bialy = czekaj (rybka poza)
   - HIT_TEXT/MISS_TEXT = ignoruj klatke
   - Brak = minigra sie skonczyla
5. Po 3 trafieniach lub koncu czasu -> wracamy do kroku 1

Zabezpieczenia:
- pyautogui.FAILSAFE = True (rusz mysz w lewy gorny rog zeby przerwac)
- Klawisz 'q' w oknie podgladu zamyka bota
- Ctrl+C w terminalu tez zamyka
"""

import time
import sys
import ctypes
import os
import cv2
import numpy as np


def _check_admin():
    """Sprawdza czy bot jest uruchomiony jako Administrator."""
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("BLAD: Bot musi byc uruchomiony jako Administrator!")
        print("Gra Eryndos dziala z uprawnieniami admina - bez tego")
        print("klawisze nie dotra do okna gry (Windows UIPI).")
        print()
        print("Rozwiazanie:")
        print("  1. Otworz PowerShell jako Administrator")
        print("  2. Wklej ponizsze komendy:")
        print()
        print('     $env:PYTHONPATH="C:\\Users\\REDACTED-USER-PATH\\Desktop\\Repos\\Kosa\\versions\\post_cnn"')
        print('     cd "C:\\Users\\REDACTED-USER-PATH\\Desktop\\Repos\\Kosa\\versions\\post_cnn"')
        print('     & "..\\..\\.venv\\Scripts\\python.exe" -m src.bot --debug')
        print()
        sys.exit(1)


from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
from src.input_simulator import InputSimulator
from src.config import (
    SCAN_INTERVAL, CLICKS_TO_WIN,
    CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, CIRCLE_SAFE_MARGIN,
)
import math

# Probuj zaladowac CNN — jesli niedostepne, fallback na klasyczna detekcje
try:
    from cnn.inference import FishNetInference
    HAS_CNN = True
except ImportError:
    HAS_CNN = False

# Probuj zaladowac detektor ksztaltu rybki
try:
    from cnn.fish_shape_detector import FishShapeDetector
    HAS_SHAPE = True
except ImportError:
    HAS_SHAPE = False

# Probuj zaladowac ONNX Runtime (do Patch CNN weryfikacji rybki)
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class KosaBot:
    """Glowny bot automatyzujacy lowienie ryb w Metin2."""

    # Bezpieczny promien — NIGDY nie klikamy dalej niz to od srodka okregu
    SAFE_RADIUS = CIRCLE_RADIUS - CIRCLE_SAFE_MARGIN  # 64 - 10 = 54 px

    # Limit klkniec w to samo miejsce (ochrona przed klikaniem w napis MISS)
    SAME_SPOT_RADIUS = 15      # piksele - jesli klik w tym promieniu = "to samo miejsce"
    SAME_SPOT_MAX_CLICKS = 3   # max klikniec w to samo miejsce

    @staticmethod
    def _clamp_to_circle(x: int, y: int) -> tuple:
        """
        Ogranicza pozycje klikniecia do wnetrza okregu z marginesem.
        Jesli (x,y) jest poza bezpieczna strefa, przesuwa punkt
        na brzeg bezpiecznego okregu w tym samym kierunku.

        Returns:
            (x, y) wewnatrz bezpiecznego okregu
        """
        dx = x - CIRCLE_CENTER_X
        dy = y - CIRCLE_CENTER_Y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= KosaBot.SAFE_RADIUS:
            return (x, y)  # juz w bezpiecznej strefie

        # Przesuń na brzeg bezpiecznego okregu
        scale = KosaBot.SAFE_RADIUS / dist
        new_x = int(CIRCLE_CENTER_X + dx * scale)
        new_y = int(CIRCLE_CENTER_Y + dy * scale)
        return (new_x, new_y)

    def __init__(self, debug: bool = False, use_cnn: bool = True):
        """
        Args:
            debug: jesli True, pokazuje okno podgladu z wizualizacja
            use_cnn: jesli True, uzywa CNN do detekcji (z fallbackiem na klasyczna)
        """
        self.capture = ScreenCapture()
        self.detector = FishingDetector()  # klasyczny — zawsze dostepny jako fallback
        self.input = InputSimulator()
        self.debug = debug
        self.running = False
        self.total_rounds = 0
        self.total_catches = 0

        # CNN detekcja
        self.cnn = None
        if use_cnn and HAS_CNN:
            try:
                self.cnn = FishNetInference()
                print("[BOT] TRYB HYBRYDOWY:")
                print("      CNN → rozpoznawanie stanu (WHITE/RED/INACTIVE/HIT/MISS)")
                print("      Klasyczny → szukanie rybki (background subtraction)")
            except Exception as e:
                print(f"[BOT] CNN niedostepny ({e}) — tryb klasyczny")
        elif use_cnn and not HAS_CNN:
            print("[BOT] CNN nie zainstalowany (brak onnxruntime) — tryb klasyczny")
        else:
            print("[BOT] Tryb klasyczny (CNN wylaczony)")

        # Detektor ksztaltu rybki (fallback gdy bg-sub nie znajdzie)
        self.shape_detector = None
        if HAS_SHAPE:
            try:
                self.shape_detector = FishShapeDetector()
                print("      Shape → fallback detekcji rybki (tlo referencyjne)")
            except Exception as e:
                print(f"[BOT] Shape detector niedostepny ({e})")

        # Patch CNN — weryfikacja kandydatow na rybke (32x32 patch → fish/not_fish)
        self.patch_cnn = None
        if HAS_ORT:
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                patch_model_path = os.path.join(base_dir, "cnn", "models", "fish_patch_cnn.onnx")
                if os.path.exists(patch_model_path):
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.patch_cnn = ort.InferenceSession(
                        patch_model_path, opts,
                        providers=['CPUExecutionProvider']
                    )
                    # Warmup
                    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)
                    self.patch_cnn.run(None, {'patch': dummy})
                    print("      PatchCNN → weryfikacja kandydatow na rybke (ONNX)")
                else:
                    print(f"[BOT] PatchCNN model nie znaleziony: {patch_model_path}")
            except Exception as e:
                print(f"[BOT] PatchCNN niedostepny ({e})")

    # --- Patch CNN: weryfikacja kandydatow na rybke ---
    PATCH_SIZE = 32
    PATCH_HALF = PATCH_SIZE // 2
    PATCH_CNN_THRESHOLD = 0.5  # prob > 0.5 = fish

    def _verify_fish_patch(self, frame, cx: int, cy: int) -> tuple:
        """
        Weryfikuje kandydata na rybke za pomoca Patch CNN.

        Wycina 32x32 patch wokol (cx, cy), przepuszcza przez ONNX model,
        zwraca (is_fish, probability).

        Args:
            frame: klatka minigry (BGR uint8)
            cx, cy: srodek kandydata

        Returns:
            (True/False, prob) — True jesli CNN potwierdza rybke
        """
        if self.patch_cnn is None:
            return (True, 1.0)  # brak CNN = akceptuj wszystko

        h, w = frame.shape[:2]
        half = self.PATCH_HALF

        # Wycinek z paddingiem (jesli blisko krawedzi)
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bot = max(0, y2 - h)

        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(w, x2)
        cy2 = min(h, y2)

        crop = frame[cy1:cy2, cx1:cx2]

        if pad_left or pad_top or pad_right or pad_bot:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bot, pad_left, pad_right,
                                      cv2.BORDER_REFLECT_101)

        if crop.shape[0] != self.PATCH_SIZE or crop.shape[1] != self.PATCH_SIZE:
            crop = cv2.resize(crop, (self.PATCH_SIZE, self.PATCH_SIZE))

        # Preprocess: BGR float32 [0,1], HWC -> CHW, batch dim
        inp = crop.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))  # HWC -> CHW
        inp = inp[np.newaxis, ...]  # (1, 3, 32, 32)

        logit = self.patch_cnn.run(None, {'patch': inp})[0][0]
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        is_fish = prob > self.PATCH_CNN_THRESHOLD

        return (is_fish, prob)

    def _detect_frame(self, frame) -> dict:
        """
        Hybrid detekcja — CNN do rozpoznawania stanu, klasyczny detektor do szukania rybki.

        CNN swietnie rozpoznaje kolor tla (WHITE/RED/INACTIVE/HIT_TEXT/MISS_TEXT),
        ale NIE zna ksztaltu rybki (za malo danych treningowych z pozycjami).

        Klasyczny detektor (background subtraction) dobrze lokalizuje rybke
        (potwierdzone diagnostyka: 1-5px dokladnosc), ale potrzebuje warmup.

        Tryb hybrydowy laczy zalety obu podejsc:
        - CNN → stan gry (pewnosc ~100%, dziala na 1 klatce, bez warmup)
        - Klasyczny → pozycja rybki (background subtraction, 1-5px dokladnosc)

        Returns:
            dict z polami: color, fish_pos, state, state_conf
        """
        if self.cnn is not None:
            # CNN: rozpoznaj stan (kolor tla / napisy HIT/MISS)
            result = self.cnn.predict(frame)
            color_map = {
                'INACTIVE': 'none',
                'WHITE': 'white',
                'RED': 'red',
                'HIT_TEXT': 'none',   # nie klikaj w napis HIT
                'MISS_TEXT': 'none',  # nie klikaj w napis MISS
            }
            color = color_map.get(result['state'], 'none')

            # Klasyczny detektor: znajdz rybke (background subtraction)
            # CNN podaje kolor, klasyczny wie gdzie rybka
            fish_pos = self.detector.find_fish_position(frame, circle_color=color)
            fish_src = "BG-SUB"

            # Fallback: shape detector (tlo referencyjne)
            # Jesli bg-sub nie znalazl rybki, sprobuj detektorem ksztaltu
            if fish_pos is None and self.shape_detector is not None:
                shape_result = self.shape_detector.find_fish_simple(frame)
                if shape_result is not None:
                    fish_pos = shape_result
                    fish_src = "SHAPE"

            return {
                'color': color,
                'fish_pos': fish_pos,
                'fish_src': fish_src,
                'state': result['state'],
                'state_conf': result['state_conf'],
            }
        else:
            # Fallback: w pelni klasyczny detektor
            color = self.detector.detect_circle_color(frame)
            fish_pos = self.detector.find_fish_position(frame, circle_color=color)
            fish_src = "BG-SUB"

            # Shape fallback
            if fish_pos is None and self.shape_detector is not None:
                shape_result = self.shape_detector.find_fish_simple(frame)
                if shape_result is not None:
                    fish_pos = shape_result
                    fish_src = "SHAPE"

            return {
                'color': color,
                'fish_pos': fish_pos,
                'fish_src': fish_src,
                'state': color.upper() if color != 'none' else 'INACTIVE',
                'state_conf': 1.0,
            }

    def wait_for_fishing_minigame(self, timeout: float = 10.0) -> bool:
        """
        Czeka az okienko minigry lowienia sie pojawi.
        Uzywa klasycznego detektora (HSV) — niezawodny, bez CNN.

        Args:
            timeout: maksymalny czas oczekiwania w sekundach

        Returns:
            True jesli minigra sie pojawila, False jesli timeout
        """
        print("[BOT] Czekam na pojawienie sie minigry...")
        start = time.time()
        while time.time() - start < timeout:
            frame = self.capture.grab_fishing_box()
            if self.detector.is_fishing_active(frame):
                print("[BOT] Minigra wykryta!")
                return True
            time.sleep(SCAN_INTERVAL)

        print("[BOT] Timeout - minigra sie nie pojawila.")
        return False

    def play_fishing_round(self) -> bool:
        """
        Gra jedna runde minigry lowienia.

        Strategia: CIAGLE KLIKANIE Z AKTUALNYM SLEDZENIEM
        - W fazie BIALEJ: sledzimy rybke (background subtraction)
        - W fazie CZERWONEJ: klikamy CO KLATKE w aktualna pozycje rybki
        - Background subtraction daje ~1-5px dokladnosc w obu fazach

        Returns:
            True jesli runda sie zakonczyla normalnie,
            False jesli przerwano (np. klawisz 'q')
        """
        print("[BOT] === Rozpoczynam runde lowienia ===")
        click_count = 0
        no_circle_count = 0
        max_no_circle = 15
        was_active = False
        last_fish_pos = None
        click_spots = []  # tracker klikniec w to samo miejsce [(x, y, count)]

        # Reset trackera rybki na poczatku rundy
        self.detector.reset_tracking()

        # Fokusuj okno RAZ na poczatku rundy
        self.input.ensure_focus()

        while self.running:
            frame = self.capture.grab_fishing_box()

            # Klasyczny detektor koloru (HSV) — niezawodnie wykrywa koniec rundy
            # CNN nie nadaje sie do tego: po zamknieciu minigry dalej widzi WHITE/RED
            color = self.detector.detect_circle_color(frame)

            # Szukanie rybki: bg-sub + shape fallback + weryfikacja Patch CNN
            fish_pos = self.detector.find_fish_position(frame, circle_color=color)
            fish_src = "BG-SUB"
            cnn_prob = -1.0  # -1 = nie weryfikowano

            if fish_pos is None and self.shape_detector is not None:
                shape_result = self.shape_detector.find_fish_simple(frame)
                if shape_result is not None:
                    fish_pos = shape_result
                    fish_src = "SHAPE"

            # Weryfikacja Patch CNN — odrzuc false positives (napisy, splash, szum)
            if fish_pos is not None:
                is_fish, cnn_prob = self._verify_fish_patch(frame, fish_pos[0], fish_pos[1])
                if not is_fish:
                    fish_src = f"{fish_src}!CNN"  # odrzucony przez CNN
                    fish_pos = None  # CNN mowi: to nie rybka

            if fish_pos is not None:
                last_fish_pos = fish_pos

            if color == "red":
                was_active = True
                no_circle_count = 0

                # Klikaj w kazdej klatce - pozycja jest aktualna!
                click_target = fish_pos if fish_pos is not None else last_fish_pos

                if click_target is not None:
                    fx, fy = self._clamp_to_circle(click_target[0], click_target[1])

                    # Sprawdz limit klikniec w to samo miejsce
                    spot_blocked = False
                    for i, (sx, sy, cnt) in enumerate(click_spots):
                        dist = math.sqrt((fx - sx)**2 + (fy - sy)**2)
                        if dist <= self.SAME_SPOT_RADIUS:
                            if cnt >= self.SAME_SPOT_MAX_CLICKS:
                                spot_blocked = True
                            else:
                                click_spots[i] = (sx, sy, cnt + 1)
                            break
                    else:
                        click_spots.append((fx, fy, 1))

                    if not spot_blocked:
                        self.input.click_at_fish_fast(fx, fy)
                        click_count += 1
                        if fish_pos is not None:
                            cnn_tag = f" p={cnn_prob:.2f}" if cnn_prob >= 0 else ""
                            src = f"[{fish_src}{cnn_tag}]"
                        else:
                            src = "[LAST]"
                        if click_count % 5 == 1:  # loguj co 5 klikniec
                            print(f"[BOT] Klik #{click_count} w ({fx},{fy}) {src}")

            elif color == "white":
                no_circle_count = 0
                was_active = True
                click_spots = []  # reset spot trackera przy zmianie fazy na biala

            else:
                no_circle_count += 1
                if was_active and no_circle_count >= max_no_circle:
                    print(f"[BOT] Okno minigry sie zamknelo. Klikniec: {click_count}")
                    return True

            # Debug: podglad na zywo
            if self.debug:
                extra = {
                    'fish_src': fish_src,
                    'cnn_prob': cnn_prob,
                }
                if not self._show_debug(frame, color, click_count, last_fish_pos, extra):
                    return False

            time.sleep(SCAN_INTERVAL)

        return False

    def _wait_for_minigame_close(self, timeout: float = 5.0):
        """
        Czeka az okno minigry calkowicie zniknie.
        Upewnia sie ze nie zaczniemy nowej rundy za wczesnie.
        """
        start = time.time()
        while time.time() - start < timeout:
            frame = self.capture.grab_fishing_box()
            if not self.detector.is_fishing_active(frame):
                return  # okno zamkniete
            time.sleep(0.1)
        print("[BOT] Timeout czekania na zamkniecie okna minigry.")

    def _show_debug(self, frame, color, click_count, fish_pos=None, extra=None) -> bool:
        """
        Wyswietla okno debugowe z podgladem.

        Returns:
            False jesli uzytkownik nacisnal 'q' (chce zakonczyc)
        """
        display = frame.copy()

        # Kolor ramki
        if color == "red":
            border = (0, 0, 255)
            text = "KLIKAJ!"
        elif color == "white":
            border = (255, 255, 255)
            text = "Czekaj..."
        else:
            border = (128, 128, 128)
            text = "Brak okregu"

        cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1), border, 3)

        # Rysuj bezpieczny okrag (zielony = strefa klikania)
        cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, (128, 128, 128), 1)
        cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), self.SAFE_RADIUS, (0, 255, 0), 1)

        cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border, 2)
        cv2.putText(display, f"Klikniecia: {click_count}/{CLICKS_TO_WIN}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display, f"Rundy: {self.total_rounds} | Zlow: {self.total_catches}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Rysuj pozycje rybki
        if fish_pos is not None:
            fx, fy = fish_pos
            cv2.circle(display, (fx, fy), 8, (0, 255, 255), 2)  # zolte kolko
            cv2.line(display, (fx - 12, fy), (fx + 12, fy), (0, 255, 255), 1)
            cv2.line(display, (fx, fy - 12), (fx, fy + 12), (0, 255, 255), 1)
            cv2.putText(display, f"Rybka: ({fx},{fy})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(display, "Rybka: ???", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 100), 1)

        # Dodatkowe info (Patch CNN, metoda detekcji)
        if extra:
            fish_src = extra.get('fish_src', '')
            cnn_prob = extra.get('cnn_prob', -1.0)
            method = extra.get('method', '')
            state = extra.get('state', '')
            conf = extra.get('conf', 0)

            parts = []
            if method:
                parts.append(f"[{method}] {state} ({conf:.0%})")
            if fish_src:
                parts.append(f"src:{fish_src}")
            if cnn_prob >= 0:
                parts.append(f"CNN:{cnn_prob:.2f}")
            label = " | ".join(parts) if parts else ""

            if label:
                # Kolor: zielony jesli CNN potwierdza, czerwony jesli odrzuca
                if "!CNN" in fish_src:
                    label_color = (0, 0, 255)  # czerwony — odrzucony
                elif cnn_prob >= 0.5:
                    label_color = (0, 255, 0)  # zielony — potwierdzony
                else:
                    label_color = (200, 200, 0)  # zolty — brak info
                cv2.putText(display, label,
                            (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)

        cv2.imshow("Kosa Bot", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[BOT] Uzytkownik nacisnal 'q' - koncze.")
            self.running = False
            return False
        return True

    def run(self):
        """
        Glowna petla bota. Powtarza cykl lowienia w nieskonczonosc.
        Zatrzymaj przez:
        - Klawisz 'q' w oknie podgladu (tryb debug)
        - Ctrl+C w terminalu
        - Ruszenie myszy w lewy gorny rog (pyautogui failsafe)
        """
        self.running = True
        print("=" * 50)
        print("  KOSA BOT - Automatyczne lowienie ryb")
        print("=" * 50)
        print()

        # --- Znajdz i sfokusuj okno gry ERYNDOS ---
        from src.input_simulator import _find_game_window, _focus_game_window, GAME_WINDOW_TITLE
        win = _find_game_window()
        if win:
            print(f"[BOT] Znaleziono okno gry: \"{win.title}\"")
            print(f"[BOT] Rozmiar: {win.width}x{win.height}, pozycja: ({win.left},{win.top})")
            if _focus_game_window():
                print("[BOT] Okno gry przeniesione na pierwszy plan!")
            else:
                print("[BOT] Nie udalo sie aktywowac okna — przelacz recznie!")
                time.sleep(3)
        else:
            print(f"[BOT] UWAGA: Nie znaleziono okna '{GAME_WINDOW_TITLE}'!")
            print("[BOT] Sprawdz czy gra jest uruchomiona.")
            print("[BOT] Przelacz sie recznie na okno gry w ciagu 5 sekund!")
            time.sleep(5)

        print()
        print("Zabezpieczenia:")
        print("  - Rusz mysz w LEWY GORNY ROG ekranu = natychmiastowe przerwanie")
        print("  - Ctrl+C w terminalu = przerwanie")
        if self.debug:
            print("  - Klawisz 'q' w oknie podgladu = przerwanie")
        print()
        print("Start za 3 sekundy...")
        time.sleep(3)

        try:
            while self.running:
                self.total_rounds += 1
                print(f"\n{'='*40}")
                print(f"[BOT] RUNDA {self.total_rounds}")
                print(f"{'='*40}")

                # 1. Start rundy: robak + wedka
                self.input.start_fishing_round()

                # 2. Czekaj na minigre
                if not self.wait_for_fishing_minigame():
                    print("[BOT] Minigra sie nie pojawila. Probuje ponownie...")
                    # Reset klasycznego detektora — nowa runda, nowe tlo
                    self.detector.reset_tracking()
                    continue

                # 3. Graj runde
                if not self.play_fishing_round():
                    break  # przerwano

                # 4. Czekaj az okno minigry sie zamknie + pauza
                print("[BOT] Czekam az okno minigry calkowicie zniknie...")
                self._wait_for_minigame_close()
                # Reset detektora — nowa runda, nowy model tla
                self.detector.reset_tracking()
                print("[BOT] Pauza 3s przed nastepna runda...")
                time.sleep(3.0)

        except KeyboardInterrupt:
            print("\n[BOT] Przerwano przez Ctrl+C.")
        except Exception as e:
            print(f"\n[BOT] Blad: {e}")
        finally:
            self.running = False
            if self.debug:
                cv2.destroyAllWindows()
            print()
            print(f"[BOT] Podsumowanie:")
            print(f"  Rundy: {self.total_rounds}")
            print(f"  Udane zlowienia: {self.total_catches}")
            print(f"[BOT] Koniec.")


# --- URUCHOMIENIE ---
# Wymagany PowerShell jako Administrator!
# python -m src.bot                  -> tryb normalny z CNN
# python -m src.bot --debug          -> tryb z podgladem + CNN
# python -m src.bot --debug --no-cnn -> tryb klasyczny (bez CNN)
if __name__ == "__main__":
    _check_admin()
    debug_mode = "--debug" in sys.argv
    use_cnn = "--no-cnn" not in sys.argv
    bot = KosaBot(debug=debug_mode, use_cnn=use_cnn)
    bot.run()
