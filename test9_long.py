"""
TEST 9 - Pelny test bota (2 minuty)

Cel: Uruchomic bota na 2 minuty - bot sam lowi ryby (klika),
a test nagrywa klatki do pozniejszej oceny przez uzytkownika.
Generuje weryfikacja.html w stylu testow 8a/8b/8c.

Przebieg:
1. Bot lowi ryby przez 120 sekund (z klikaniem!)
2. Co 5-ta klatka zapisywana jako PNG z debug overlay (zielony krzyzyk na rybce)
3. Log przebiegu do test9_long/log.csv
4. Generuje weryfikacja.html z mini-gridem, filtrami, przyciskami OK/ZLE/BRAK/POZA
5. Raport statystyczny na koniec

WYMAGA uruchomienia jako Administrator (pydirectinput + UIPI)!

Uruchomienie:
  Start-Process powershell -Verb RunAs -ArgumentList
    "-NoExit -Command Set-Location '...'; & python test9_long.py"
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
    """Sprawdza czy test jest uruchomiony jako Administrator."""
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("BLAD: Test musi byc uruchomiony jako Administrator!")
        print("Bot musi klikac w okno gry - wymaga uprawnien admina (UIPI).")
        sys.exit(1)


_check_admin()

from src.screen_capture import ScreenCapture
from src.fishing_detector import FishingDetector
from src.input_simulator import InputSimulator
from src.config import (
    CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, CIRCLE_SAFE_MARGIN,
    SCAN_INTERVAL, CLICKS_TO_WIN,
)

OUTPUT_DIR = "test9_long"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
DURATION_SEC = 120       # 2 minuty
SAVE_EVERY_N = 5         # Zapisuj co 5-ta klatke jako obraz
STATUS_EVERY_SEC = 10    # Status co 10 sekund
SAFE_RADIUS = CIRCLE_RADIUS - CIRCLE_SAFE_MARGIN


def clamp_to_circle(x, y):
    """Ogranicza pozycje klikniecia do wnetrza okregu z marginesem."""
    dx = x - CIRCLE_CENTER_X
    dy = y - CIRCLE_CENTER_Y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= SAFE_RADIUS:
        return (x, y)
    scale = SAFE_RADIUS / dist
    return (int(CIRCLE_CENTER_X + dx * scale), int(CIRCLE_CENTER_Y + dy * scale))


def draw_debug_frame(frame, frame_nr, color, fish_pos, ts, click_count, round_nr, action=""):
    """Rysuje debug overlay na klatce - okrag, bezpieczna strefa, krzyzyk na rybce."""
    display = frame.copy()

    # Okrag fazy
    if color == "red":
        circle_color = (0, 0, 255)
    elif color == "white":
        circle_color = (200, 200, 200)
    else:
        circle_color = (80, 80, 80)

    cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), CIRCLE_RADIUS, circle_color, 1)
    cv2.circle(display, (CIRCLE_CENTER_X, CIRCLE_CENTER_Y), SAFE_RADIUS, (0, 255, 0), 1)

    # Zielony krzyzyk na pozycji rybki
    if fish_pos is not None:
        fx, fy = fish_pos
        cv2.circle(display, (fx, fy), 6, (0, 255, 0), 2)
        cv2.line(display, (fx - 8, fy), (fx + 8, fy), (0, 255, 0), 1)
        cv2.line(display, (fx, fy - 8), (fx, fy + 8), (0, 255, 0), 1)

    # Info tekstowe
    status = "#{} | {:.1f}s | R{} | {} | kliki={}".format(frame_nr, ts, round_nr, color, click_count)
    if fish_pos:
        status += " | fish=({},{})".format(fish_pos[0], fish_pos[1])
    else:
        status += " | fish=NONE"
    if action:
        status += " | " + action

    cv2.putText(display, status, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    return display


def generate_weryfikacja_html(saved_frames, output_dir):
    """Generuje weryfikacja.html w stylu testow 8a/8b/8c."""

    # Buduj liste klatek jako JS array
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
    frames_js_array = "[\n" + ",\n".join(js_entries) + "\n]"

    # HTML template - NIE uzywamy f-string zeby uniknac problemow z {{ }}
    html_template = r"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>TEST 9 - Weryfikacja trackingu (pelny bot 2 min)</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a2e; color: #eee; font-family: 'Consolas', monospace; padding: 20px; }
h1 { text-align: center; margin-bottom: 10px; color: #fff; }
.subtitle { text-align: center; color: #aaa; margin-bottom: 20px; font-size: 14px; }

.stats-bar {
    display: flex; justify-content: center; gap: 15px; margin-bottom: 20px;
    background: #16213e; padding: 12px 20px; border-radius: 8px; flex-wrap: wrap;
}
.stat { text-align: center; }
.stat .num { font-size: 22px; font-weight: bold; }
.stat .label { font-size: 11px; color: #888; }
.stat.ok .num { color: #4f4; }
.stat.bad .num { color: #f44; }
.stat.misstext .num { color: #f6a; }
.stat.hittext .num { color: #fa4; }
.stat.missing .num { color: #fa0; }
.stat.outside .num { color: #48f; }
.stat.skip .num { color: #888; }
.stat.total .num { color: #4af; }
.stat.pct .num { color: #ff0; font-size: 28px; }

.instructions {
    background: #0f3460; padding: 15px; border-radius: 8px; margin-bottom: 20px;
    text-align: center; line-height: 1.8;
}
.instructions b { color: #4f4; }
.instructions .red { color: #f66; }
.instructions .yellow { color: #fa0; }
.instructions .blue { color: #48f; }

.nav { display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
.nav button {
    padding: 8px 16px; border: 2px solid #444; background: #1a1a2e; color: #fff;
    border-radius: 6px; cursor: pointer; font-family: inherit; font-size: 13px;
}
.nav button:hover { border-color: #888; }
.nav button.active { border-color: #4af; background: #16213e; }

.frame-viewer {
    max-width: 800px; margin: 0 auto; background: #16213e; border-radius: 12px;
    padding: 20px; text-align: center;
}
.frame-viewer img {
    max-width: 100%; border: 3px solid #444; border-radius: 4px;
    image-rendering: pixelated;
}
.frame-viewer img.red { border-color: #f44; }
.frame-viewer img.white { border-color: #ddd; }
.frame-viewer img.none { border-color: #666; }

.frame-info { margin: 10px 0; font-size: 16px; }
.frame-info .phase { padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.frame-info .phase.red { background: #a00; }
.frame-info .phase.white { background: #555; }
.frame-info .phase.none { background: #333; }

.frame-fish-info { font-size: 14px; color: #aaa; margin: 5px 0; }
.frame-action-info { font-size: 13px; color: #6af; margin: 3px 0; }

.review-buttons {
    display: flex; justify-content: center; gap: 10px; margin: 15px 0; flex-wrap: wrap;
}
.review-buttons button {
    padding: 12px 24px; border: 3px solid; border-radius: 10px;
    font-size: 15px; font-weight: bold; cursor: pointer; font-family: inherit;
    transition: transform 0.1s;
}
.review-buttons button:hover { transform: scale(1.05); }
.review-buttons button:active { transform: scale(0.95); }

.btn-ok { background: #1a3a1a; border-color: #4f4; color: #4f4; }
.btn-ok.selected { background: #2a5a2a; box-shadow: 0 0 15px #4f4; }

.btn-bad { background: #3a1a1a; border-color: #f44; color: #f44; }
.btn-bad.selected { background: #5a2a2a; box-shadow: 0 0 15px #f44; }

.btn-misstext { background: #3a1a2a; border-color: #f6a; color: #f6a; }
.btn-misstext.selected { background: #5a2a3a; box-shadow: 0 0 15px #f6a; }

.btn-hittext { background: #3a2a1a; border-color: #fa4; color: #fa4; }
.btn-hittext.selected { background: #5a3a2a; box-shadow: 0 0 15px #fa4; }

.btn-missing { background: #3a3a1a; border-color: #fa0; color: #fa0; }
.btn-missing.selected { background: #5a5a2a; box-shadow: 0 0 15px #fa0; }

.btn-outside { background: #1a2a3a; border-color: #48f; color: #48f; }
.btn-outside.selected { background: #2a3a5a; box-shadow: 0 0 15px #48f; }

.btn-skip { background: #2a2a2a; border-color: #666; color: #888; }

.frame-nav {
    display: flex; justify-content: center; gap: 20px; margin-top: 15px; align-items: center;
}
.frame-nav button {
    padding: 10px 25px; background: #0f3460; border: 2px solid #4af;
    color: #4af; border-radius: 8px; font-size: 14px; cursor: pointer; font-family: inherit;
}
.frame-nav button:hover { background: #16213e; }
.frame-nav button:disabled { opacity: 0.3; cursor: default; }
.frame-counter { font-size: 16px; color: #4af; min-width: 100px; }

.progress-bar {
    width: 100%; height: 8px; background: #333; border-radius: 4px; margin: 10px 0;
    overflow: hidden;
}
.progress-bar .fill { height: 100%; background: #4af; transition: width 0.3s; border-radius: 4px; }

.summary-section {
    display: none; max-width: 800px; margin: 20px auto; background: #16213e;
    border-radius: 12px; padding: 30px; text-align: center;
}
.summary-section h2 { color: #4af; margin-bottom: 20px; }
.summary-table { margin: 0 auto; text-align: left; }
.summary-table td { padding: 5px 15px; font-size: 16px; }
.export-btn {
    margin-top: 20px; padding: 12px 30px; background: #0f3460; border: 2px solid #4af;
    color: #4af; border-radius: 8px; font-size: 14px; cursor: pointer; font-family: inherit;
}

.mini-grid {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 4px;
    margin: 10px auto; max-width: 800px;
}
.mini-dot {
    width: 14px; height: 14px; border-radius: 3px; cursor: pointer;
    border: 1px solid #333;
}
.mini-dot.current { border: 2px solid #fff; }
.mini-dot.pending { background: #333; }
.mini-dot.ok { background: #4f4; }
.mini-dot.bad { background: #f44; }
.mini-dot.misstext { background: #f6a; }
.mini-dot.hittext { background: #fa4; }
.mini-dot.missing { background: #fa0; }
.mini-dot.outside { background: #48f; }
.mini-dot.skipped { background: #555; }

.shortcut { font-size: 11px; color: #666; margin-top: 3px; }
</style>
</head>
<body>

<h1>TEST 9 - Weryfikacja trackingu (pelny bot, 2 min)</h1>
<p class="subtitle">Bot sam lowil ryby przez 2 minuty. Sprawdz czy zielony krzyzyk (tracking rybki) byl w dobrym miejscu.</p>

<div class="instructions">
    Dla kazdej klatki RED ocen detekcje rybki:<br>
    <b>OK</b> (klawisz 1) = zielony krzyzyk jest NA rybce<br>
    <span class="red">ZLE</span> (klawisz 2) = krzyzyk jest w ZLYM miejscu (cos innego niz tekst)<br>
    <span style="color:#f6a">MISS txt</span> (klawisz 5) = krzyzyk celuje w napis MISS zamiast w rybke<br>
    <span style="color:#fa4">HIT txt</span> (klawisz 6) = krzyzyk celuje w napis HIT zamiast w rybke<br>
    <span class="yellow">BRAK</span> (klawisz 3) = nie ma krzyzyka, a rybka JEST widoczna w okregu<br>
    <span class="blue">POZA</span> (klawisz 4) = rybka jest POZA okregiem lub niewidoczna = brak detekcji jest OK<br>
    SKIP (klawisz 0/S) = pomijasz
</div>

<div class="stats-bar">
    <div class="stat pct"><div class="num" id="pct-display">--%</div><div class="label">TRAFNOSC</div></div>
    <div class="stat ok"><div class="num" id="ok-count">0</div><div class="label">OK</div></div>
    <div class="stat bad"><div class="num" id="bad-count">0</div><div class="label">ZLE</div></div>
    <div class="stat misstext"><div class="num" id="misstext-count">0</div><div class="label">MISS txt</div></div>
    <div class="stat hittext"><div class="num" id="hittext-count">0</div><div class="label">HIT txt</div></div>
    <div class="stat missing"><div class="num" id="missing-count">0</div><div class="label">BRAK</div></div>
    <div class="stat outside"><div class="num" id="outside-count">0</div><div class="label">POZA</div></div>
    <div class="stat skip"><div class="num" id="skip-count">0</div><div class="label">SKIP</div></div>
    <div class="stat total"><div class="num" id="reviewed-count">0</div><div class="label">OCENIONE</div></div>
</div>

<div class="nav">
    <button onclick="setFilter('all')" class="active" id="filter-all">Wszystkie</button>
    <button onclick="setFilter('red')" id="filter-red">Tylko RED</button>
    <button onclick="setFilter('white')" id="filter-white">Tylko WHITE</button>
    <button onclick="setFilter('none')" id="filter-none">Tylko NONE</button>
</div>

<div class="mini-grid" id="mini-grid"></div>
<div class="progress-bar"><div class="fill" id="progress-fill"></div></div>

<div class="frame-viewer" id="viewer">
    <img id="frame-img" src="">
    <div class="frame-info">
        Klatka <span id="frame-num"></span> — Faza: <span id="frame-phase" class="phase"></span>
        — Runda: <span id="frame-round"></span>
    </div>
    <div class="frame-fish-info" id="fish-info"></div>
    <div class="frame-action-info" id="action-info"></div>

    <div class="review-buttons" id="review-btns">
        <button class="btn-ok" onclick="review('ok')">OK<div class="shortcut">klawisz 1</div></button>
        <button class="btn-bad" onclick="review('bad')">ZLE<div class="shortcut">klawisz 2</div></button>
        <button class="btn-misstext" onclick="review('misstext')">MISS txt<div class="shortcut">klawisz 5</div></button>
        <button class="btn-hittext" onclick="review('hittext')">HIT txt<div class="shortcut">klawisz 6</div></button>
        <button class="btn-missing" onclick="review('missing')">? BRAK<div class="shortcut">klawisz 3</div></button>
        <button class="btn-outside" onclick="review('outside')">POZA<div class="shortcut">klawisz 4</div></button>
        <button class="btn-skip" onclick="review('skip')">SKIP<div class="shortcut">klawisz 0</div></button>
    </div>

    <div class="frame-nav">
        <button onclick="prevFrame()" id="btn-prev">Poprzednia</button>
        <span class="frame-counter" id="nav-counter"></span>
        <button onclick="nextFrame()" id="btn-next">Nastepna</button>
    </div>
</div>

<div class="summary-section" id="summary">
    <h2>PODSUMOWANIE</h2>
    <div id="summary-content"></div>
    <button class="export-btn" onclick="exportResults()">Kopiuj wyniki do schowka</button>
</div>

<script>
const frames = %%FRAMES_DATA%%;

let reviews = {};
let currentFilter = "all";
let filteredFrames = [...frames];
let currentIdx = 0;

function getFiltered() {
    if (currentFilter === "all") return [...frames];
    return frames.filter(f => f.phase === currentFilter);
}

function setFilter(f) {
    currentFilter = f;
    filteredFrames = getFiltered();
    currentIdx = 0;
    document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
    document.getElementById('filter-' + f).classList.add('active');
    renderMiniGrid();
    showFrame();
}

function renderMiniGrid() {
    const grid = document.getElementById('mini-grid');
    grid.innerHTML = '';
    filteredFrames.forEach((f, i) => {
        const dot = document.createElement('div');
        dot.className = 'mini-dot';
        const r = reviews[f.file];
        if (r === 'ok') dot.classList.add('ok');
        else if (r === 'bad') dot.classList.add('bad');
        else if (r === 'misstext') dot.classList.add('misstext');
        else if (r === 'hittext') dot.classList.add('hittext');
        else if (r === 'missing') dot.classList.add('missing');
        else if (r === 'outside') dot.classList.add('outside');
        else if (r === 'skip') dot.classList.add('skipped');
        else dot.classList.add('pending');
        if (i === currentIdx) dot.classList.add('current');
        dot.title = '#' + f.num + ' (R' + f.round + ' ' + f.phase + ') ' + (r || 'pending');
        dot.onclick = function() { currentIdx = i; showFrame(); renderMiniGrid(); };
        grid.appendChild(dot);
    });
}

function showFrame() {
    if (filteredFrames.length === 0) return;
    const f = filteredFrames[currentIdx];
    document.getElementById('frame-img').src = 'frames/' + f.file;
    document.getElementById('frame-img').className = f.phase;
    document.getElementById('frame-num').textContent = '#' + f.num;
    var phaseEl = document.getElementById('frame-phase');
    phaseEl.textContent = f.phase.toUpperCase();
    phaseEl.className = 'phase ' + f.phase;
    document.getElementById('frame-round').textContent = f.round;

    var fishInfo = document.getElementById('fish-info');
    if (f.fish) {
        fishInfo.textContent = 'Rybka wykryta: (' + f.fx + ', ' + f.fy + ') — szukaj zielonego krzyzyka na obrazku';
        fishInfo.style.color = '#4f4';
    } else {
        fishInfo.textContent = 'Rybka NIE wykryta — brak krzyzyka na obrazku';
        fishInfo.style.color = '#f66';
    }

    var actionInfo = document.getElementById('action-info');
    if (f.action) {
        actionInfo.textContent = 'Akcja bota: ' + f.action;
    } else {
        actionInfo.textContent = '';
    }

    document.querySelectorAll('.review-buttons button').forEach(function(b) { b.classList.remove('selected'); });
    var r = reviews[f.file];
    if (r === 'ok') document.querySelector('.btn-ok').classList.add('selected');
    if (r === 'bad') document.querySelector('.btn-bad').classList.add('selected');
    if (r === 'misstext') document.querySelector('.btn-misstext').classList.add('selected');
    if (r === 'hittext') document.querySelector('.btn-hittext').classList.add('selected');
    if (r === 'missing') document.querySelector('.btn-missing').classList.add('selected');
    if (r === 'outside') document.querySelector('.btn-outside').classList.add('selected');

    document.getElementById('nav-counter').textContent =
        (currentIdx + 1) + ' / ' + filteredFrames.length;
    document.getElementById('btn-prev').disabled = currentIdx === 0;
    document.getElementById('btn-next').disabled = currentIdx >= filteredFrames.length - 1;

    updateProgress();
}

function review(verdict) {
    var f = filteredFrames[currentIdx];
    reviews[f.file] = verdict;
    updateStats();
    renderMiniGrid();

    if (currentIdx < filteredFrames.length - 1) {
        currentIdx++;
        showFrame();
        renderMiniGrid();
    } else {
        showFrame();
        checkComplete();
    }
}

function updateStats() {
    var ok = 0, bad = 0, misstext = 0, hittext = 0, missing = 0, outside = 0, skip = 0;
    Object.values(reviews).forEach(function(v) {
        if (v === 'ok') ok++;
        else if (v === 'bad') bad++;
        else if (v === 'misstext') misstext++;
        else if (v === 'hittext') hittext++;
        else if (v === 'missing') missing++;
        else if (v === 'outside') outside++;
        else if (v === 'skip') skip++;
    });
    document.getElementById('ok-count').textContent = ok;
    document.getElementById('bad-count').textContent = bad;
    document.getElementById('misstext-count').textContent = misstext;
    document.getElementById('hittext-count').textContent = hittext;
    document.getElementById('missing-count').textContent = missing;
    document.getElementById('outside-count').textContent = outside;
    document.getElementById('skip-count').textContent = skip;
    document.getElementById('reviewed-count').textContent = ok + bad + misstext + hittext + missing + outside + skip;

    var good = ok + outside;
    var judged = ok + bad + misstext + hittext + missing + outside;
    var pct = judged > 0 ? Math.round(100 * good / judged) : '--';
    document.getElementById('pct-display').textContent = pct + '%';
}

function updateProgress() {
    var total = filteredFrames.length;
    var done = filteredFrames.filter(function(f) { return reviews[f.file]; }).length;
    document.getElementById('progress-fill').style.width = (100 * done / total) + '%';
}

function prevFrame() {
    if (currentIdx > 0) { currentIdx--; showFrame(); renderMiniGrid(); }
}
function nextFrame() {
    if (currentIdx < filteredFrames.length - 1) { currentIdx++; showFrame(); renderMiniGrid(); }
}

function checkComplete() {
    var redFrames = frames.filter(function(f) { return f.phase === 'red'; });
    var allReviewed = redFrames.every(function(f) { return reviews[f.file]; });
    if (allReviewed) showSummary();
}

function showSummary() {
    var redFrames = frames.filter(function(f) { return f.phase === 'red'; });
    var ok = 0, bad = 0, misstext = 0, hittext = 0, missing = 0, outside = 0;
    redFrames.forEach(function(f) {
        var v = reviews[f.file];
        if (v === 'ok') ok++;
        else if (v === 'bad') bad++;
        else if (v === 'misstext') misstext++;
        else if (v === 'hittext') hittext++;
        else if (v === 'missing') missing++;
        else if (v === 'outside') outside++;
    });
    var good = ok + outside;
    var judged = ok + bad + misstext + hittext + missing + outside;
    var pct = judged > 0 ? (100 * good / judged).toFixed(1) : 0;
    var textErrors = misstext + hittext;

    document.getElementById('summary').style.display = 'block';
    document.getElementById('summary-content').innerHTML =
        '<table class="summary-table">' +
        '<tr><td style="color:#4f4">OK (dobrze wykryl):</td><td><b>' + ok + '</b></td></tr>' +
        '<tr><td style="color:#f44">ZLE (zle wykryl):</td><td><b>' + bad + '</b></td></tr>' +
        '<tr><td style="color:#f6a">MISS txt (celuje w MISS):</td><td><b>' + misstext + '</b></td></tr>' +
        '<tr><td style="color:#fa4">HIT txt (celuje w HIT):</td><td><b>' + hittext + '</b></td></tr>' +
        '<tr><td style="color:#fa0">BRAK (nie wykryl a powinien):</td><td><b>' + missing + '</b></td></tr>' +
        '<tr><td style="color:#48f">POZA (rybka poza okregiem):</td><td><b>' + outside + '</b></td></tr>' +
        '<tr><td colspan="2"><hr style="border-color:#444"></td></tr>' +
        '<tr><td>Ocenione klatki RED:</td><td><b>' + judged + '</b></td></tr>' +
        '<tr><td style="color:#f6a">Bledy z powodu napisow HIT/MISS:</td><td><b>' + textErrors + '</b></td></tr>' +
        '<tr><td style="color:#ff0;font-size:20px">TRAFNOSC (OK+POZA):</td><td style="font-size:20px;color:#ff0"><b>' + pct + '%</b></td></tr>' +
        '</table>' +
        '<p style="margin-top:15px;color:#888">Skopiuj wyniki i wklej mi w chacie!</p>';
    document.getElementById('summary').scrollIntoView({behavior:'smooth'});
}

function exportResults() {
    var redFrames = frames.filter(function(f) { return f.phase === 'red'; });
    var ok = 0, bad = 0, misstext = 0, hittext = 0, missing = 0, outside = 0;
    var lines = ['TEST 9 - Weryfikacja trackingu pelny bot 2 min (ocena uzytkownika)', ''];
    redFrames.forEach(function(f) {
        var v = reviews[f.file] || 'skip';
        if (v === 'ok') ok++;
        else if (v === 'bad') bad++;
        else if (v === 'misstext') misstext++;
        else if (v === 'hittext') hittext++;
        else if (v === 'missing') missing++;
        else if (v === 'outside') outside++;
        lines.push('Klatka #' + f.num + ' (R' + f.round + '): ' + v.toUpperCase());
    });
    var good = ok + outside;
    var judged = ok + bad + misstext + hittext + missing + outside;
    var pct = judged > 0 ? (100 * good / judged).toFixed(1) : 0;
    var textErrors = misstext + hittext;
    lines.push('', 'OK: ' + ok + ', ZLE: ' + bad + ', MISS_TXT: ' + misstext + ', HIT_TXT: ' + hittext + ', BRAK: ' + missing + ', POZA: ' + outside,
               'Bledy z napisow HIT/MISS: ' + textErrors,
               'TRAFNOSC (OK+POZA): ' + pct + '%');

    navigator.clipboard.writeText(lines.join('\n')).then(function() {
        alert('Skopiowano do schowka! Wklej w chacie.');
    }).catch(function() {
        var ta = document.createElement('textarea');
        ta.value = lines.join('\n');
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        alert('Skopiowano! Wklej w chacie.');
    });
}

document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft') prevFrame();
    else if (e.key === 'ArrowRight') nextFrame();
    else if (e.key === '1') review('ok');
    else if (e.key === '2') review('bad');
    else if (e.key === '3') review('missing');
    else if (e.key === '4') review('outside');
    else if (e.key === '5') review('misstext');
    else if (e.key === '6') review('hittext');
    else if (e.key === '0' || e.key === 's') review('skip');
});

// Auto-skip non-RED frames
function autoSkipNonRed() {
    frames.forEach(function(f) {
        if (f.phase !== 'red') reviews[f.file] = 'skip';
    });
    updateStats();
}

autoSkipNonRed();
setFilter('red');
</script>

</body>
</html>"""

    # Wstaw dane klatek zamiast placeholdera
    html_final = html_template.replace("%%FRAMES_DATA%%", frames_js_array)

    viewer_path = os.path.join(output_dir, "weryfikacja.html")
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(html_final)
    return viewer_path


def main():
    print("=" * 60)
    print("  TEST 9 - Pelny test bota (2 minuty)")
    print("=" * 60)
    print()
    print("Bot bedzie:")
    print("  1. Sam zakladal robaka (F4) i zarzucal wedke (SPACJA)")
    print("  2. Klikal w rybke w fazie czerwonej")
    print("  3. Nagrywal klatki z debug overlay do oceny")
    print()
    print("Czas trwania: {}s ({} min)".format(DURATION_SEC, DURATION_SEC // 60))
    print("Wyniki zapisze do: {}/".format(OUTPUT_DIR))
    print()

    os.makedirs(FRAMES_DIR, exist_ok=True)

    capture = ScreenCapture()
    detector = FishingDetector()
    inp = InputSimulator()

    # CSV log
    log_path = os.path.join(OUTPUT_DIR, "log.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["frame", "timestamp", "color", "active", "fish_x", "fish_y",
                      "action", "click_count", "round_nr"])

    # Statystyki
    stats = {
        "total": 0,
        "red": 0, "white": 0, "none": 0,
        "red_detected": 0, "white_detected": 0, "detected_total": 0,
        "clicks": 0, "rounds": 0,
    }
    saved_frames = []
    frame_nr = 0
    round_nr = 0
    click_count = 0
    no_circle_count = 0
    was_active = False
    last_fish_pos = None
    round_phase = "start"  # start / wait_minigame / playing / cooldown

    # Odliczanie
    for i in range(3, 0, -1):
        print("  Start za {}...".format(i))
        time.sleep(1)
    print()

    start_time = time.perf_counter()
    last_status_time = start_time
    cooldown_start = 0

    # Fokusuj okno gry
    inp.ensure_focus()

    while True:
        now = time.perf_counter()
        elapsed = now - start_time
        if elapsed >= DURATION_SEC:
            break

        # === FAZA: START NOWEJ RUNDY ===
        if round_phase == "start":
            round_nr += 1
            click_count = 0
            no_circle_count = 0
            was_active = False
            last_fish_pos = None
            detector.reset_tracking()

            print("\n[TEST9] === RUNDA {} ({:.0f}s) ===".format(round_nr, elapsed))
            inp.start_fishing_round()
            round_phase = "wait_minigame"
            wait_start = time.perf_counter()
            continue

        # === FAZA: CZEKAJ NA MINIGRE ===
        if round_phase == "wait_minigame":
            frame = capture.grab_fishing_box()
            if frame is not None and detector.is_fishing_active(frame):
                print("[TEST9] Minigra wykryta!")
                round_phase = "playing"
                continue
            if time.perf_counter() - wait_start > 10.0:
                print("[TEST9] Timeout - minigra sie nie pojawila. Probuje ponownie...")
                round_phase = "start"
                continue
            time.sleep(SCAN_INTERVAL)
            continue

        # === FAZA: COOLDOWN PO RUNDZIE ===
        if round_phase == "cooldown":
            if time.perf_counter() - cooldown_start >= 3.0:
                round_phase = "start"
            else:
                time.sleep(0.2)
            continue

        # === FAZA: GRA ===
        frame = capture.grab_fishing_box()
        frame_nr += 1

        if frame is None:
            writer.writerow([frame_nr, "{:.3f}".format(elapsed), "error", False, "", "", "", click_count, round_nr])
            time.sleep(SCAN_INTERVAL)
            continue

        color = detector.detect_circle_color(frame)
        fish_pos = detector.find_fish_position(frame, circle_color=color)
        if fish_pos is not None:
            last_fish_pos = fish_pos

        action = ""

        if color == "red":
            was_active = True
            no_circle_count = 0

            click_target = fish_pos if fish_pos is not None else last_fish_pos
            if click_target is not None:
                fx, fy = clamp_to_circle(click_target[0], click_target[1])
                inp.click_at_fish_fast(fx, fy)
                click_count += 1
                stats["clicks"] += 1
                fresh = "FRESH" if fish_pos else "LAST"
                action = "CLICK({},{})[{}]".format(fx, fy, fresh)
                if click_count % 5 == 1:
                    print("[TEST9] Klik #{} w ({},{}) [{}]".format(click_count, fx, fy, fresh))

        elif color == "white":
            no_circle_count = 0
            was_active = True
            action = "WAIT"

        else:
            no_circle_count += 1
            if was_active and no_circle_count >= 15:
                print("[TEST9] Runda {} zakonczona. Kliki: {}".format(round_nr, click_count))
                stats["rounds"] += 1
                round_phase = "cooldown"
                cooldown_start = time.perf_counter()
                # Czekaj az okno zniknie
                for _ in range(50):
                    f2 = capture.grab_fishing_box()
                    if f2 is None or not detector.is_fishing_active(f2):
                        break
                    time.sleep(0.1)
                continue

        # Statystyki
        stats["total"] += 1
        if color == "red":
            stats["red"] += 1
            if fish_pos:
                stats["red_detected"] += 1
        elif color == "white":
            stats["white"] += 1
            if fish_pos:
                stats["white_detected"] += 1
        else:
            stats["none"] += 1
        if fish_pos:
            stats["detected_total"] += 1

        # Log CSV
        fx_log = fish_pos[0] if fish_pos else ""
        fy_log = fish_pos[1] if fish_pos else ""
        writer.writerow([frame_nr, "{:.3f}".format(elapsed), color, was_active, fx_log, fy_log,
                          action, click_count, round_nr])

        # Zapisz klatke co N
        if (frame_nr % SAVE_EVERY_N) == 0:
            display = draw_debug_frame(frame, frame_nr, color, fish_pos, elapsed,
                                        click_count, round_nr, action)
            filename = "frame_{:05d}_{}.png".format(frame_nr, color)
            cv2.imwrite(os.path.join(FRAMES_DIR, filename), display)
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

        # Status co N sekund
        if now - last_status_time >= STATUS_EVERY_SEC:
            remaining = DURATION_SEC - elapsed
            det_pct = 100 * stats["detected_total"] / stats["total"] if stats["total"] else 0
            print("  [{:5.1f}s / {}s] kl={} det={:.1f}% | rundy={} kliki={} | zostalo {:.0f}s".format(
                elapsed, DURATION_SEC, frame_nr, det_pct, stats['rounds'], stats['clicks'], remaining))
            last_status_time = now

        time.sleep(SCAN_INTERVAL)

    log_file.close()
    total_elapsed = time.perf_counter() - start_time

    # Generuj HTML weryfikacja
    viewer_path = generate_weryfikacja_html(saved_frames, OUTPUT_DIR)

    # RAPORT
    total = stats["total"]
    if total == 0:
        print("\nBrak klatek! Cos poszlo nie tak.")
        return

    fps = total / total_elapsed if total_elapsed > 0 else 0
    red_det_pct = 100 * stats["red_detected"] / stats["red"] if stats["red"] else 0
    white_det_pct = 100 * stats["white_detected"] / stats["white"] if stats["white"] else 0
    total_det_pct = 100 * stats["detected_total"] / total

    print()
    print("=" * 60)
    print("  RAPORT - TEST 9 (pelny bot 2 min)")
    print("=" * 60)
    print("  Czas nagrywania:     {:.1f}s".format(total_elapsed))
    print("  Klatki lacznie:      {}".format(total))
    print("  FPS:                 {:.1f}".format(fps))
    print("  Zapisanych klatek:   {}".format(len(saved_frames)))
    print("  Rundy lowienia:      {}".format(stats['rounds']))
    print("  Klikniecia lacznie:  {}".format(stats['clicks']))
    print()
    print("  Fazy:")
    print("    RED:   {:5d} ({:.0f}%)".format(stats['red'], 100 * stats['red'] / total))
    print("    WHITE: {:5d} ({:.0f}%)".format(stats['white'], 100 * stats['white'] / total))
    print("    NONE:  {:5d} ({:.0f}%)".format(stats['none'], 100 * stats['none'] / total))
    print()
    print("  Detekcja rybki:")
    print("    W RED:   {:5d}/{:5d} ({:.1f}%)".format(stats['red_detected'], stats['red'], red_det_pct))
    print("    W WHITE: {:5d}/{:5d} ({:.1f}%)".format(stats['white_detected'], stats['white'], white_det_pct))
    print("    TOTAL:   {:5d}/{:5d} ({:.1f}%)".format(stats['detected_total'], total, total_det_pct))

    # Zapisz raport do pliku
    report_path = os.path.join(OUTPUT_DIR, "raport.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TEST 9 - Pelny bot 2 min ({})\n".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write("=" * 60 + "\n\n")
        f.write("Czas: {:.1f}s, Klatki: {}, FPS: {:.1f}\n".format(total_elapsed, total, fps))
        f.write("Zapisanych klatek: {}\n".format(len(saved_frames)))
        f.write("Rundy: {}, Klikniec: {}\n\n".format(stats['rounds'], stats['clicks']))
        f.write("Fazy: RED={}, WHITE={}, NONE={}\n".format(stats['red'], stats['white'], stats['none']))
        f.write("Detekcja w RED: {}/{} ({:.1f}%)\n".format(stats['red_detected'], stats['red'], red_det_pct))
        f.write("Detekcja w WHITE: {}/{} ({:.1f}%)\n".format(stats['white_detected'], stats['white'], white_det_pct))
        f.write("Detekcja TOTAL: {}/{} ({:.1f}%)\n".format(stats['detected_total'], total, total_det_pct))

    print("\n  Log CSV:        {}".format(log_path))
    print("  Raport:         {}".format(report_path))
    print("  Klatki:         {}/ ({} plikow)".format(FRAMES_DIR, len(saved_frames)))
    print("  Weryfikacja:    {}".format(viewer_path))
    print()
    print("  Otworz weryfikacja.html w przegladarce aby ocenic klatki!")
    print("  (mozesz tez uzyc: python -m http.server 8770 w folderze {})".format(OUTPUT_DIR))
    print()
    print("GOTOWE!")


if __name__ == "__main__":
    main()
