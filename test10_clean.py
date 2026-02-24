"""
TEST 10 - Czyste klatki do analizy MISS/HIT

Cel: Bot lowi ryby przez 60 sekund. Zapisuje CZYSTE klatki (bez overlay)
zeby uzytkownik mogl wyciagnac przyklady napisow MISS/HIT.

Zapisuje DWA zestawy:
  test10_clean/raw/     - surowe klatki BEZ nakładek (do wycianania MISS/HIT)
  test10_clean/debug/   - klatki z debug overlay (zielony krzyzyk, okrag)

Generuje prosty viewer HTML do przegladania obu zestawow obok siebie.

WYMAGA uruchomienia jako Administrator!
"""

import os
import sys
import time
import csv
import math
import ctypes
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _check_admin():
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("BLAD: Test musi byc uruchomiony jako Administrator!")
        sys.exit(1)


_check_admin()

from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
from src.input_simulator import InputSimulator
from src.config import (
    CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, CIRCLE_SAFE_MARGIN,
    SCAN_INTERVAL,
)

OUTPUT_DIR = "test10_clean"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
DURATION_SEC = 60        # 1 minuta
SAVE_EVERY_N = 3         # Zapisuj co 3-cia klatke (gesciej niz test9)
STATUS_EVERY_SEC = 10
SAFE_RADIUS = CIRCLE_RADIUS - CIRCLE_SAFE_MARGIN

# Limit klkniec w to samo miejsce
SAME_SPOT_RADIUS = 15     # piksele - jesli klik w tym promieniu = "to samo miejsce"
SAME_SPOT_MAX_CLICKS = 3  # max klikniec w to samo miejsce


def clamp_to_circle(x, y):
    dx = x - CIRCLE_CENTER_X
    dy = y - CIRCLE_CENTER_Y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= SAFE_RADIUS:
        return (x, y)
    scale = SAFE_RADIUS / dist
    return (int(CIRCLE_CENTER_X + dx * scale), int(CIRCLE_CENTER_Y + dy * scale))


def draw_debug_frame(frame, frame_nr, color, fish_pos, ts, click_count, round_nr, action=""):
    """Rysuje debug overlay - okrag, bezpieczna strefa, krzyzyk na rybce."""
    display = frame.copy()

    if color == "red":
        circle_color = (0, 0, 255)
    elif color == "white":
        circle_color = (200, 200, 200)
    else:
        circle_color = (80, 80, 80)

    cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, circle_color, 1)
    cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), SAFE_RADIUS, (0, 255, 0), 1)

    if fish_pos is not None:
        fx, fy = fish_pos
        cv2.circle(display, (fx, fy), 6, (0, 255, 0), 2)
        cv2.line(display, (fx - 8, fy), (fx + 8, fy), (0, 255, 0), 1)
        cv2.line(display, (fx, fy - 8), (fx, fy + 8), (0, 255, 0), 1)

    status = "#{} | {:.1f}s | R{} | {} | kliki={}".format(frame_nr, ts, round_nr, color, click_count)
    if fish_pos:
        status += " | fish=({},{})".format(fish_pos[0], fish_pos[1])
    if action:
        status += " | " + action
    cv2.putText(display, status, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    return display


def generate_viewer_html(saved_frames, output_dir):
    """Generuje prosty viewer do porownywania RAW vs DEBUG klatek."""

    js_entries = []
    for f in saved_frames:
        action_str = f.get("action", "").replace("\\", "\\\\").replace('"', '\\"')
        fish_str = "true" if f["has_fish"] else "false"
        fx = str(f["fx"]) if f["fx"] != "" else "null"
        fy = str(f["fy"]) if f["fy"] != "" else "null"
        js_entries.append(
            '    {{file:"{}", num:{}, phase:"{}", ts:{:.2f}, fish:{}, fx:{}, fy:{}, round:{}, action:"{}"}}'.format(
                f["file"], f["nr"], f["color"], f["ts"], fish_str, fx, fy,
                f.get("round", 0), action_str
            )
        )
    frames_js = "[\n" + ",\n".join(js_entries) + "\n]"

    html = r"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>TEST 10 - Czyste klatki (RAW vs DEBUG)</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a2e; color: #eee; font-family: 'Consolas', monospace; padding: 20px; }
h1 { text-align: center; margin-bottom: 10px; }
.subtitle { text-align: center; color: #aaa; margin-bottom: 20px; font-size: 14px; }

.viewer {
    max-width: 1200px; margin: 0 auto; background: #16213e; border-radius: 12px;
    padding: 20px; text-align: center;
}
.frames-row {
    display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin: 15px 0;
}
.frame-col { text-align: center; }
.frame-col h3 { margin-bottom: 8px; font-size: 14px; }
.frame-col h3.raw { color: #4af; }
.frame-col h3.debug { color: #4f4; }
.frame-col img {
    max-width: 100%; border: 3px solid #444; border-radius: 4px;
    image-rendering: pixelated; width: 279px;
}
.frame-col img.red { border-color: #f44; }
.frame-col img.white { border-color: #ddd; }
.frame-col img.none { border-color: #666; }

.info { margin: 10px 0; font-size: 16px; }
.phase { padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.phase.red { background: #a00; }
.phase.white { background: #555; }
.phase.none { background: #333; }

.fish-info { font-size: 14px; color: #aaa; margin: 5px 0; }
.action-info { font-size: 13px; color: #6af; margin: 3px 0; }

.nav {
    display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;
}
.nav button {
    padding: 8px 16px; border: 2px solid #444; background: #1a1a2e; color: #fff;
    border-radius: 6px; cursor: pointer; font-family: inherit; font-size: 13px;
}
.nav button:hover { border-color: #888; }
.nav button.active { border-color: #4af; background: #16213e; }

.frame-nav {
    display: flex; justify-content: center; gap: 20px; margin-top: 15px; align-items: center;
}
.frame-nav button {
    padding: 10px 25px; background: #0f3460; border: 2px solid #4af;
    color: #4af; border-radius: 8px; font-size: 14px; cursor: pointer; font-family: inherit;
}
.frame-nav button:disabled { opacity: 0.3; }
.counter { font-size: 16px; color: #4af; min-width: 100px; }

.mini-grid {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 3px;
    margin: 10px auto; max-width: 1000px;
}
.mini-dot {
    width: 12px; height: 12px; border-radius: 2px; cursor: pointer; border: 1px solid #333;
}
.mini-dot.red { background: #c44; }
.mini-dot.white { background: #888; }
.mini-dot.none { background: #444; }
.mini-dot.current { border: 2px solid #fff; }

.tip {
    background: #0f3460; padding: 12px; border-radius: 8px; margin-bottom: 20px;
    text-align: center; font-size: 13px; line-height: 1.6;
}
.tip b { color: #4af; }
</style>
</head>
<body>

<h1>TEST 10 - Czyste klatki do analizy</h1>
<p class="subtitle">Lewo: SUROWY screen (bez overlay) | Prawo: z debug overlay (zielony krzyzyk)</p>

<div class="tip">
    <b>Cel:</b> Przegladaj klatki i wycinaj przyklady napisow MISS/HIT z surowych screenow.<br>
    Klatki RED to momenty gdy bot klika. Szukaj napisow "MISS" lub "HIT" na surowych klatkach.<br>
    Strzalki <b>← →</b> = nawigacja | Filtry ponizej ograniczaja do fazy
</div>

<div class="nav">
    <button onclick="setFilter('all')" class="active" id="filter-all">Wszystkie</button>
    <button onclick="setFilter('red')" id="filter-red">Tylko RED</button>
    <button onclick="setFilter('white')" id="filter-white">Tylko WHITE</button>
</div>

<div class="mini-grid" id="mini-grid"></div>

<div class="viewer" id="viewer">
    <div class="frames-row">
        <div class="frame-col">
            <h3 class="raw">SUROWY (raw) — do wycinania MISS/HIT</h3>
            <img id="raw-img" src="">
        </div>
        <div class="frame-col">
            <h3 class="debug">DEBUG (overlay) — detekcja bota</h3>
            <img id="debug-img" src="">
        </div>
    </div>
    <div class="info">
        Klatka <span id="frame-num"></span> — Faza: <span id="frame-phase" class="phase"></span>
        — Runda: <span id="frame-round"></span> — Czas: <span id="frame-ts"></span>s
    </div>
    <div class="fish-info" id="fish-info"></div>
    <div class="action-info" id="action-info"></div>
    <div class="frame-nav">
        <button onclick="prevFrame()" id="btn-prev">← Poprzednia</button>
        <span class="counter" id="nav-counter"></span>
        <button onclick="nextFrame()" id="btn-next">Nastepna →</button>
    </div>
</div>

<script>
const frames = %%FRAMES_DATA%%;
let currentFilter = "all";
let filteredFrames = [...frames];
let currentIdx = 0;

function getFiltered() {
    if (currentFilter === "all") return [...frames];
    return frames.filter(function(f) { return f.phase === currentFilter; });
}

function setFilter(f) {
    currentFilter = f;
    filteredFrames = getFiltered();
    currentIdx = 0;
    document.querySelectorAll('.nav button').forEach(function(b) { b.classList.remove('active'); });
    document.getElementById('filter-' + f).classList.add('active');
    renderMiniGrid();
    showFrame();
}

function renderMiniGrid() {
    var grid = document.getElementById('mini-grid');
    grid.innerHTML = '';
    filteredFrames.forEach(function(f, i) {
        var dot = document.createElement('div');
        dot.className = 'mini-dot ' + f.phase;
        if (i === currentIdx) dot.classList.add('current');
        dot.title = '#' + f.num + ' R' + f.round + ' ' + f.phase;
        dot.onclick = function() { currentIdx = i; showFrame(); renderMiniGrid(); };
        grid.appendChild(dot);
    });
}

function showFrame() {
    if (filteredFrames.length === 0) return;
    var f = filteredFrames[currentIdx];
    document.getElementById('raw-img').src = 'raw/' + f.file;
    document.getElementById('raw-img').className = f.phase;
    document.getElementById('debug-img').src = 'debug/' + f.file;
    document.getElementById('debug-img').className = f.phase;
    document.getElementById('frame-num').textContent = '#' + f.num;
    var phaseEl = document.getElementById('frame-phase');
    phaseEl.textContent = f.phase.toUpperCase();
    phaseEl.className = 'phase ' + f.phase;
    document.getElementById('frame-round').textContent = f.round;
    document.getElementById('frame-ts').textContent = f.ts.toFixed(1);

    var fishInfo = document.getElementById('fish-info');
    if (f.fish) {
        fishInfo.textContent = 'Rybka wykryta: (' + f.fx + ', ' + f.fy + ')';
        fishInfo.style.color = '#4f4';
    } else {
        fishInfo.textContent = 'Rybka NIE wykryta';
        fishInfo.style.color = '#f66';
    }

    var actionInfo = document.getElementById('action-info');
    actionInfo.textContent = f.action ? 'Akcja: ' + f.action : '';

    document.getElementById('nav-counter').textContent =
        (currentIdx + 1) + ' / ' + filteredFrames.length;
    document.getElementById('btn-prev').disabled = currentIdx === 0;
    document.getElementById('btn-next').disabled = currentIdx >= filteredFrames.length - 1;
    renderMiniGrid();
}

function prevFrame() { if (currentIdx > 0) { currentIdx--; showFrame(); } }
function nextFrame() { if (currentIdx < filteredFrames.length - 1) { currentIdx++; showFrame(); } }

document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft') prevFrame();
    else if (e.key === 'ArrowRight') nextFrame();
});

setFilter('all');
</script>
</body>
</html>"""

    html_final = html.replace("%%FRAMES_DATA%%", frames_js)
    viewer_path = os.path.join(output_dir, "viewer.html")
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(html_final)
    return viewer_path


def main():
    print("=" * 60)
    print("  TEST 10 - Czyste klatki do analizy MISS/HIT (1 min)")
    print("=" * 60)
    print()
    print("Bot bedzie lowil ryby przez {}s.".format(DURATION_SEC))
    print("Zapisuje SUROWE klatki (bez overlay) + klatki z DEBUG overlay.")
    print("Limit: max {} klkniec w to samo miejsce (r={}px)".format(
        SAME_SPOT_MAX_CLICKS, SAME_SPOT_RADIUS))
    print()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    capture = ScreenCapture()
    detector = FishingDetector()
    inp = InputSimulator()

    log_path = os.path.join(OUTPUT_DIR, "log.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["frame", "timestamp", "color", "active", "fish_x", "fish_y",
                      "action", "click_count", "round_nr", "spot_blocked"])

    saved_frames = []
    frame_nr = 0
    round_nr = 0
    click_count = 0
    no_circle_count = 0
    was_active = False
    last_fish_pos = None
    round_phase = "start"

    # Tracker klikniec w to samo miejsce
    click_spots = []  # lista (x, y, count)

    def check_spot_limit(fx, fy):
        """Sprawdza czy nie kliknelismy za duzo razy w to samo miejsce."""
        for i, (sx, sy, cnt) in enumerate(click_spots):
            dist = math.sqrt((fx - sx)**2 + (fy - sy)**2)
            if dist <= SAME_SPOT_RADIUS:
                if cnt >= SAME_SPOT_MAX_CLICKS:
                    return True  # zablokowane
                click_spots[i] = (sx, sy, cnt + 1)
                return False
        click_spots.append((fx, fy, 1))
        return False

    for i in range(3, 0, -1):
        print("  Start za {}...".format(i))
        time.sleep(1)

    start_time = time.perf_counter()
    last_status_time = start_time
    cooldown_start = 0

    inp.ensure_focus()

    while True:
        now = time.perf_counter()
        elapsed = now - start_time
        if elapsed >= DURATION_SEC:
            break

        if round_phase == "start":
            round_nr += 1
            click_count = 0
            no_circle_count = 0
            was_active = False
            last_fish_pos = None
            click_spots = []  # reset spot trackera na nowa runde
            detector.reset_tracking()
            print("\n[TEST10] === RUNDA {} ({:.0f}s) ===".format(round_nr, elapsed))
            inp.start_fishing_round()
            round_phase = "wait_minigame"
            wait_start = time.perf_counter()
            continue

        if round_phase == "wait_minigame":
            frame = capture.grab_fishing_box()
            if frame is not None and detector.is_fishing_active(frame):
                print("[TEST10] Minigra wykryta!")
                round_phase = "playing"
                continue
            if time.perf_counter() - wait_start > 10.0:
                print("[TEST10] Timeout - probuje ponownie...")
                round_phase = "start"
                continue
            time.sleep(SCAN_INTERVAL)
            continue

        if round_phase == "cooldown":
            if time.perf_counter() - cooldown_start >= 3.0:
                round_phase = "start"
            else:
                time.sleep(0.2)
            continue

        # === PLAYING ===
        frame = capture.grab_fishing_box()
        frame_nr += 1

        if frame is None:
            time.sleep(SCAN_INTERVAL)
            continue

        color = detector.detect_circle_color(frame)
        fish_pos = detector.find_fish_position(frame, circle_color=color)
        if fish_pos is not None:
            last_fish_pos = fish_pos

        action = ""
        spot_blocked = False

        if color == "red":
            was_active = True
            no_circle_count = 0
            click_target = fish_pos if fish_pos is not None else last_fish_pos
            if click_target is not None:
                fx, fy = clamp_to_circle(click_target[0], click_target[1])
                # Sprawdz limit klikniec w to samo miejsce
                if check_spot_limit(fx, fy):
                    spot_blocked = True
                    fresh = "FRESH" if fish_pos else "LAST"
                    action = "BLOCKED({},{})[{}]".format(fx, fy, fresh)
                else:
                    inp.click_at_fish_fast(fx, fy)
                    click_count += 1
                    fresh = "FRESH" if fish_pos else "LAST"
                    action = "CLICK({},{})[{}]".format(fx, fy, fresh)
                    if click_count % 5 == 1:
                        print("[TEST10] Klik #{} w ({},{}) [{}]".format(click_count, fx, fy, fresh))

        elif color == "white":
            no_circle_count = 0
            was_active = True
            action = "WAIT"

        else:
            no_circle_count += 1
            if was_active and no_circle_count >= 15:
                print("[TEST10] Runda {} zakonczona. Kliki: {}".format(round_nr, click_count))
                round_phase = "cooldown"
                cooldown_start = time.perf_counter()
                for _ in range(50):
                    f2 = capture.grab_fishing_box()
                    if f2 is None or not detector.is_fishing_active(f2):
                        break
                    time.sleep(0.1)
                continue

        # Log
        fx_log = fish_pos[0] if fish_pos else ""
        fy_log = fish_pos[1] if fish_pos else ""
        writer.writerow([frame_nr, "{:.3f}".format(elapsed), color, was_active,
                          fx_log, fy_log, action, click_count, round_nr, spot_blocked])

        # Zapisz klatki co N
        if (frame_nr % SAVE_EVERY_N) == 0:
            filename = "frame_{:05d}_{}.png".format(frame_nr, color)

            # RAW — surowy screenshot BEZ nakladek
            cv2.imwrite(os.path.join(RAW_DIR, filename), frame)

            # DEBUG — z overlay
            debug_frame = draw_debug_frame(frame, frame_nr, color, fish_pos, elapsed,
                                            click_count, round_nr, action)
            cv2.imwrite(os.path.join(DEBUG_DIR, filename), debug_frame)

            saved_frames.append({
                "file": filename,
                "nr": frame_nr,
                "ts": round(elapsed, 2),
                "color": color,
                "has_fish": fish_pos is not None,
                "fx": fx_log,
                "fy": fy_log,
                "action": action,
                "round": round_nr,
            })

        # Status
        if now - last_status_time >= STATUS_EVERY_SEC:
            remaining = DURATION_SEC - elapsed
            print("  [{:5.1f}s / {}s] kl={} kliki={} | zostalo {:.0f}s".format(
                elapsed, DURATION_SEC, frame_nr, click_count, remaining))
            last_status_time = now

        time.sleep(SCAN_INTERVAL)

    log_file.close()
    total_elapsed = time.perf_counter() - start_time

    # Policz zablokowane klikniecia
    blocked_count = 0
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["spot_blocked"] == "True":
                blocked_count += 1

    viewer_path = generate_viewer_html(saved_frames, OUTPUT_DIR)

    total = frame_nr
    if total == 0:
        print("\nBrak klatek!")
        return

    fps = total / total_elapsed if total_elapsed > 0 else 0
    print()
    print("=" * 60)
    print("  RAPORT - TEST 10 (czyste klatki, 1 min)")
    print("=" * 60)
    print("  Czas: {:.1f}s, Klatki: {}, FPS: {:.1f}".format(total_elapsed, total, fps))
    print("  Zapisanych klatek: {} (RAW + DEBUG)".format(len(saved_frames)))
    print("  Rundy: {}".format(round_nr))
    print("  Zablokowanych klkniec (spot limit): {}".format(blocked_count))
    print()
    print("  Foldery:")
    print("    RAW:    {}/  (surowe - do wycinania MISS/HIT)".format(RAW_DIR))
    print("    DEBUG:  {}/  (z overlay)".format(DEBUG_DIR))
    print("    Viewer: {}".format(viewer_path))
    print()
    print("  Otworz viewer.html w przegladarce!")
    print("  Uzyj: python -m http.server 8771 w folderze {}".format(OUTPUT_DIR))
    print()
    print("GOTOWE!")


if __name__ == "__main__":
    main()
