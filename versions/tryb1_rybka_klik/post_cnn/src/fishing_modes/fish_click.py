"""Tryb 1 — Mini-gra lowienie ryb (rybka klik).

Pipeline detekcji (kazda klatka):
1. detect_circle_color() [HSV]      -> stan gry: white/red/none
2. find_fish_position() [bg-sub]    -> pozycja rybki (mediana 15 klatek)
3. find_fish_simple() [shape]       -> fallback: tlo referencyjne + blob
4. _verify_fish_patch() [PatchCNN]  -> weryfikacja 32x32 ONNX (fish/not_fish)

Cykl rundy:
1. KosaBot wola input.start_fishing_round() (F4 + SPACE) — to jest poza tym trybem
2. wait_for_start() czeka az pojawi sie okno minigry
3. play_round() skanuje, klika, konczy gdy okno sie zamyka
4. wait_for_end() czeka az okno calkowicie znikinie

Zabezpieczenia:
- _clamp_to_circle() — klik max 54px od srodka
- Same-spot limiter — max 3 kliki w 15px promieniu (ochrona przed klikaniem MISS)
- PatchCNN verifier — odrzuca false positives (napisy, splash)
- Klawisz 'q' w oknie debug zamyka bota (przez _is_running -> False)

Uwaga: FishNet CNN (self.cnn) jest zaladowany ale nieuzywany w play_round.
Zostawiony dla zgodnosci z poprzednim zachowaniem (patrz historia-wersji.md).
"""

import math
import os
import time

import cv2
import numpy as np

from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
from src.input_simulator import InputSimulator
from src.config import (
    SCAN_INTERVAL, CLICKS_TO_WIN,
    CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, CIRCLE_SAFE_MARGIN,
)

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


class FishClickMode:
    """Tryb 1 — minigra "rybka klik" w Metin2.

    Implementuje kontrakt FishingMode (patrz src/fishing_modes/base.py).
    Konstruktor dostaje wspolne zaleznosci od KosaBot (DI), wlasne (CNN, detector)
    laduje sam.
    """

    name = "Mini-gra lowienie ryb (rybka - klik)"

    # Bezpieczny promien — NIGDY nie klikamy dalej niz to od srodka okregu
    SAFE_RADIUS = CIRCLE_RADIUS - CIRCLE_SAFE_MARGIN  # 64 - 10 = 54 px

    # Limit klikniec w to samo miejsce (ochrona przed klikaniem w napis MISS)
    SAME_SPOT_RADIUS = 15      # piksele
    SAME_SPOT_MAX_CLICKS = 3   # max klikniec w to samo miejsce

    # Patch CNN: wycinek wokol kandydata na rybke
    PATCH_SIZE = 32
    PATCH_HALF = PATCH_SIZE // 2
    PATCH_CNN_THRESHOLD = 0.5  # prob > 0.5 = fish

    def __init__(
        self,
        debug: bool,
        use_cnn: bool,
        log_callback,                # callable(str) -> None
        capture: ScreenCapture,
        input_sim: InputSimulator,
        is_running,                  # callable() -> bool (z KosaBot.running)
        request_stop=None,           # callable() -> None (ustawia KosaBot.running=False)
    ):
        """
        Args:
            debug: czy pokazywac okno podgladu z wizualizacja
            use_cnn: czy ladowac FishNetInference (legacy flag, nieuzywany w play_round)
            log_callback: funkcja logujaca (np. routing do GUI lub print)
            capture: dzielony ScreenCapture z KosaBot
            input_sim: dzielony InputSimulator z KosaBot
            is_running: callable zwracajace stan flagi running na KosaBot
            request_stop: callable proszace KosaBot o zatrzymanie petli rund
                          (np. po wcisnieciu 'q' w oknie debug)
        """
        self.debug = debug
        self._log = log_callback
        self._capture = capture
        self._input = input_sim
        self._is_running = is_running
        self._request_stop = request_stop or (lambda: None)

        # Klasyczny detektor (zawsze, jako baza i fallback dla CNN)
        self.detector = FishingDetector()

        # Liczniki tylko-do-debug-overlay
        self.total_rounds_for_overlay = 0
        self.total_catches_for_overlay = 0

        # CNN detekcja stanu (FishNet) — legacy, zaladowany ale nieuzywany w play_round
        self.cnn = None
        if use_cnn and HAS_CNN:
            try:
                self.cnn = FishNetInference()
                self._log("[BOT] TRYB HYBRYDOWY:")
                self._log("      CNN -> rozpoznawanie stanu (WHITE/RED/INACTIVE/HIT/MISS)")
                self._log("      Klasyczny -> szukanie rybki (background subtraction)")
            except Exception as e:
                self._log(f"[BOT] CNN niedostepny ({e}) - tryb klasyczny")
        elif use_cnn and not HAS_CNN:
            self._log("[BOT] CNN nie zainstalowany (brak onnxruntime) - tryb klasyczny")
        else:
            self._log("[BOT] Tryb klasyczny (CNN wylaczony)")

        # Detektor ksztaltu rybki (fallback gdy bg-sub nie znajdzie)
        self.shape_detector = None
        if HAS_SHAPE:
            try:
                self.shape_detector = FishShapeDetector()
                self._log("      Shape -> fallback detekcji rybki (tlo referencyjne)")
            except Exception as e:
                self._log(f"[BOT] Shape detector niedostepny ({e})")

        # Patch CNN — weryfikacja kandydatow na rybke (32x32 patch -> fish/not_fish)
        self.patch_cnn = None
        if HAS_ORT:
            try:
                # Sciezka do modelu: src/fishing_modes/fish_click.py -> .. -> ..  -> cnn/models/
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
                    self._log("      PatchCNN -> weryfikacja kandydatow na rybke (ONNX)")
                else:
                    self._log(f"[BOT] PatchCNN model nie znaleziony: {patch_model_path}")
            except Exception as e:
                self._log(f"[BOT] PatchCNN niedostepny ({e})")

    # ------------------------------------------------------------------
    # KONTRAKT FishingMode
    # ------------------------------------------------------------------

    def wait_for_start(self, timeout: float = 10.0) -> bool:
        """Czeka az okno minigry sie pojawi (klasyczny detektor HSV)."""
        self._log("[BOT] Czekam na pojawienie sie minigry...")
        start = time.time()
        while time.time() - start < timeout:
            if not self._is_running():
                return False
            frame = self._capture.grab_fishing_box()
            if self.detector.is_fishing_active(frame):
                self._log("[BOT] Minigra wykryta!")
                return True
            time.sleep(SCAN_INTERVAL)
        self._log("[BOT] Timeout - minigra sie nie pojawila.")
        return False

    def play_round(self) -> bool:
        """Gra jedna runde minigry lowienia.

        Strategia: CIAGLE KLIKANIE Z AKTUALNYM SLEDZENIEM
        - W fazie BIALEJ: sledzimy rybke (background subtraction)
        - W fazie CZERWONEJ: klikamy CO KLATKE w aktualna pozycje rybki

        Returns:
            True jesli runda sie zakonczyla normalnie (okno minigry zamknieto)
            False jesli przerwano (klawisz 'q' albo stop bota)
        """
        self._log("[BOT] === Rozpoczynam runde lowienia ===")
        click_count = 0
        no_circle_count = 0
        max_no_circle = 15
        was_active = False
        last_fish_pos = None
        click_spots = []  # tracker klikniec w to samo miejsce [(x, y, count)]

        # Reset trackera rybki na poczatku rundy
        self.detector.reset_tracking()

        # Fokusuj okno RAZ na poczatku rundy
        self._input.ensure_focus()

        while self._is_running():
            frame = self._capture.grab_fishing_box()

            # Klasyczny detektor koloru (HSV) — niezawodnie wykrywa koniec rundy
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
                    fish_src = f"{fish_src}!CNN"
                    fish_pos = None

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
                        self._input.click_at_fish_fast(fx, fy)
                        click_count += 1
                        if fish_pos is not None:
                            cnn_tag = f" p={cnn_prob:.2f}" if cnn_prob >= 0 else ""
                            src = f"[{fish_src}{cnn_tag}]"
                        else:
                            src = "[LAST]"
                        if click_count % 5 == 1:  # loguj co 5 klikniec
                            self._log(f"[BOT] Klik #{click_count} w ({fx},{fy}) {src}")

            elif color == "white":
                no_circle_count = 0
                was_active = True
                click_spots = []  # reset spot trackera przy zmianie fazy na biala

            else:
                no_circle_count += 1
                if was_active and no_circle_count >= max_no_circle:
                    self._log(f"[BOT] Okno minigry sie zamknelo. Klikniec: {click_count}")
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

    def wait_for_end(self, timeout: float = 5.0) -> None:
        """Czeka az okno minigry calkowicie zniknie (przed nastepna runda)."""
        start = time.time()
        while time.time() - start < timeout:
            if not self._is_running():
                return
            frame = self._capture.grab_fishing_box()
            if not self.detector.is_fishing_active(frame):
                # Reset detektora — nowa runda, nowy model tla
                self.detector.reset_tracking()
                return
            time.sleep(0.1)
        self._log("[BOT] Timeout czekania na zamkniecie okna minigry.")
        self.detector.reset_tracking()

    # ------------------------------------------------------------------
    # WEWNETRZNE — pipeline detekcji + helpery
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_to_circle(x: int, y: int) -> tuple:
        """Ogranicza pozycje klikniecia do wnetrza okregu z marginesem.

        Jesli (x,y) jest poza bezpieczna strefa, przesuwa punkt na brzeg
        bezpiecznego okregu w tym samym kierunku.
        """
        dx = x - CIRCLE_CENTER_X
        dy = y - CIRCLE_CENTER_Y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= FishClickMode.SAFE_RADIUS:
            return (x, y)

        scale = FishClickMode.SAFE_RADIUS / dist
        new_x = int(CIRCLE_CENTER_X + dx * scale)
        new_y = int(CIRCLE_CENTER_Y + dy * scale)
        return (new_x, new_y)

    def _verify_fish_patch(self, frame, cx: int, cy: int) -> tuple:
        """Weryfikuje kandydata na rybke za pomoca Patch CNN.

        Wycina 32x32 patch wokol (cx, cy), przepuszcza przez ONNX model,
        zwraca (is_fish, probability).
        """
        if self.patch_cnn is None:
            return (True, 1.0)  # brak CNN = akceptuj wszystko

        h, w = frame.shape[:2]
        half = self.PATCH_HALF

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
        inp = np.transpose(inp, (2, 0, 1))
        inp = inp[np.newaxis, ...]

        logit = self.patch_cnn.run(None, {'patch': inp})[0][0]
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        is_fish = prob > self.PATCH_CNN_THRESHOLD

        return (is_fish, prob)

    def _detect_frame(self, frame) -> dict:
        """Hybrid detekcja CNN+klasyczny.

        DEAD CODE — nie wywolywana w play_round() (patrz historia-wersji.md).
        Zostawione 1:1 zeby refaktor nie zmienial zachowania. Do usuniecia
        w osobnym commicie razem z innymi dead-code.
        """
        if self.cnn is not None:
            result = self.cnn.predict(frame)
            color_map = {
                'INACTIVE': 'none',
                'WHITE': 'white',
                'RED': 'red',
                'HIT_TEXT': 'none',
                'MISS_TEXT': 'none',
            }
            color = color_map.get(result['state'], 'none')

            fish_pos = self.detector.find_fish_position(frame, circle_color=color)
            fish_src = "BG-SUB"

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
            color = self.detector.detect_circle_color(frame)
            fish_pos = self.detector.find_fish_position(frame, circle_color=color)
            fish_src = "BG-SUB"

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

    def _show_debug(self, frame, color, click_count, fish_pos=None, extra=None) -> bool:
        """Wyswietla okno debugowe z podgladem.

        Returns:
            False jesli uzytkownik nacisnal 'q' (chce zakonczyc)
        """
        display = frame.copy()

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

        cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, (128, 128, 128), 1)
        cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), self.SAFE_RADIUS, (0, 255, 0), 1)

        cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border, 2)
        cv2.putText(display, f"Klikniecia: {click_count}/{CLICKS_TO_WIN}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display, f"Rundy: {self.total_rounds_for_overlay} | Zlow: {self.total_catches_for_overlay}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if fish_pos is not None:
            fx, fy = fish_pos
            cv2.circle(display, (fx, fy), 8, (0, 255, 255), 2)
            cv2.line(display, (fx - 12, fy), (fx + 12, fy), (0, 255, 255), 1)
            cv2.line(display, (fx, fy - 12), (fx, fy + 12), (0, 255, 255), 1)
            cv2.putText(display, f"Rybka: ({fx},{fy})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(display, "Rybka: ???", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 100), 1)

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
                if "!CNN" in fish_src:
                    label_color = (0, 0, 255)
                elif cnn_prob >= 0.5:
                    label_color = (0, 255, 0)
                else:
                    label_color = (200, 200, 0)
                cv2.putText(display, label,
                            (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)

        cv2.imshow("Kosa Bot", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self._log("[BOT] Uzytkownik nacisnal 'q' - koncze.")
            # Poprosimy KosaBot o stop (ustawi running=False); play_round() zwroci False.
            self._request_stop()
            return False
        return True
