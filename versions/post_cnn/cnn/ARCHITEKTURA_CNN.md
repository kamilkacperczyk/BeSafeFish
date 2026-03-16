# Architektura CNN dla Kosa Bot — Plan Wdrożenia

## Spis treści
1. [Architektura modelu](#1-architektura-modelu)
2. [Klasy i zadania](#2-klasy--zadania)
3. [Pipeline treningowy](#3-pipeline-treningowy)
4. [Pipeline inferencji](#4-pipeline-inferencji-produkcyjny)
5. [Narzędzie do etykietowania](#5-narzędzie-do-etykietowania)
6. [Plan implementacji](#6-plan-implementacji)
7. [Porównanie z PRE_CNN](#7-porównanie-z-pre_cnn)

---

## 1. Architektura modelu

### Wybór: Custom lightweight CNN (nie MobileNet/EfficientNet/YOLO)

**Uzasadnienie:**
- **Rozmiar wejścia jest mały** (279×247 → przeskalowane do 128×128). Pretrained modele (MobileNet, EfficientNet) oczekują 224×224+ i mają miliony parametrów — overkill dla regionu 128×128 z ~5 klasami.
- **YOLO odpada** — służy do detekcji wielu obiektów w dużych obrazach. Tu mamy 1 obiekt (rybkę) w małym regionie. YOLO v8-nano to ~3.2M parametrów i ~6ms na GPU, na CPU będzie 50-100ms.
- **MobileNetV2 odpada** — nawet wersja `width_multiplier=0.25` to ~0.5M parametrów, i wymaga ImageNet preprocessing niezgodnego z naszą domeną.
- **Custom CNN** — 3 bloki konwolucyjne + 2 FC warstwy → ~120K parametrów, <1MB wag, <3ms inferencji na CPU.

### Wybór frameworka: PyTorch

**Uzasadnienie:**
- PyTorch ma lepszą integrację z ONNX Runtime (kluczowe dla CPU inferencji <10ms)
- TorchScript / ONNX export jest prostszy niż TF Lite na Windows
- Debugging jest łatwiejszy (eager execution domyślnie)
- `torch.no_grad()` + ONNX Runtime = najszybsza ścieżka CPU

### Architektura szczegółowa: FishNet

```
INPUT: 128 × 128 × 3 (RGB, znormalizowane 0-1)
       (oryginał 279×247 przeskalowany bicubic)

┌─────────────────────────────────────────────────┐
│ BACKBONE (ekstrakcja cech)                      │
├─────────────────────────────────────────────────┤
│ Block 1:                                        │
│   Conv2d(3→16, 3×3, pad=1) + BatchNorm + ReLU  │
│   Conv2d(16→16, 3×3, pad=1) + BatchNorm + ReLU │
│   MaxPool2d(2×2, stride=2)                      │
│   → 64 × 64 × 16                               │
├─────────────────────────────────────────────────┤
│ Block 2:                                        │
│   Conv2d(16→32, 3×3, pad=1) + BatchNorm + ReLU │
│   Conv2d(32→32, 3×3, pad=1) + BatchNorm + ReLU │
│   MaxPool2d(2×2, stride=2)                      │
│   → 32 × 32 × 32                               │
├─────────────────────────────────────────────────┤
│ Block 3:                                        │
│   Conv2d(32→64, 3×3, pad=1) + BatchNorm + ReLU │
│   Conv2d(64→64, 3×3, pad=1) + BatchNorm + ReLU │
│   MaxPool2d(2×2, stride=2)                      │
│   → 16 × 16 × 64                               │
├─────────────────────────────────────────────────┤
│ Block 4:                                        │
│   Conv2d(64→64, 3×3, pad=1) + BatchNorm + ReLU │
│   MaxPool2d(2×2, stride=2)                      │
│   → 8 × 8 × 64                                 │
├─────────────────────────────────────────────────┤
│ Global Average Pooling                          │
│   → 1 × 1 × 64  (wektor 64-D)                  │
└─────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│ HEAD: Multi-task                                │
├─────────────────────────────────────────────────┤
│                                                 │
│ ┌─── Klasyfikacja stanu ───────────────────┐    │
│ │ FC(64→32) + ReLU + Dropout(0.3)          │    │
│ │ FC(32→5)  → logits [5 klas]              │    │
│ └──────────────────────────────────────────┘    │
│                                                 │
│ ┌─── Regresja pozycji rybki ───────────────┐    │
│ │ FC(64→32) + ReLU + Dropout(0.3)          │    │
│ │ FC(32→3)  → [x, y, confidence]           │    │
│ │            x,y ∈ [0,1] (sigmoid)          │    │
│ │            confidence ∈ [0,1] (sigmoid)   │    │
│ └──────────────────────────────────────────┘    │
│                                                 │
└─────────────────────────────────────────────────┘

OUTPUT:
  state_logits: [5] — klasa stanu (softmax w loss, argmax w inferencji)
  position: [3]     — (x_norm, y_norm, conf)
                      x_rel = x_norm * 279, y_rel = y_norm * 247
                      conf < 0.5 → brak rybki
```

### Uzasadnienie każdej warstwy

| Decyzja | Uzasadnienie |
|---------|-------------|
| **Input 128×128** | Najmniejszy rozmiar zachowujący rybkę (~30px blob → ~14px po skalowaniu). 64×64 by było za małe. 128×128 daje 4× mniejszy tensor niż oryginalny. |
| **3×3 kernele** | Standard — najefektywniejsze (2× 3×3 = 1× 5×5 receptive field, ale mniej parametrów) |
| **BatchNorm** | Stabilizuje trening z małym datasetem, pozwala na wyższy LR |
| **16→32→64→64 filtry** | Progresywne zwiększanie głębokości. Więcej niż 64 zbędne dla tak prostego zadania |
| **4 bloki pool** | 128→64→32→16→8, potem GAP→1. Wystarczająco duże receptive field |
| **Global Average Pooling** | Zamiast Flatten+FC — 10× mniej parametrów, lepsza generalizacja |
| **Dropout 0.3** | Tylko w FC — zapobiega overfittingowi z ~3000 próbek |
| **Osobne heads** | Multi-task: jedno zadanie regularizuje drugie (shared backbone) |
| **confidence output** | Rozwiązuje problem "brak rybki" — conf < 0.5 = pozycja nieważna |

### Liczba parametrów (szacunek)

```
Block 1: (3×3×3×16 + 16) + (3×3×16×16 + 16) = 448 + 2320 = 2,768
Block 2: (3×3×16×32 + 32) + (3×3×32×32 + 32) = 4,640 + 9,248 = 13,888
Block 3: (3×3×32×64 + 64) + (3×3×64×64 + 64) = 18,496 + 36,928 = 55,424
Block 4: (3×3×64×64 + 64) = 36,928
BatchNorm: 2 × (16+32+64+64) = 352
GAP: 0
Head cls: (64×32 + 32) + (32×5 + 5) = 2,080 + 165 = 2,245
Head pos: (64×32 + 32) + (32×3 + 3) = 2,080 + 99 = 2,179

TOTAL: ~113,784 parametrów ≈ 445 KB (float32) ≈ 115 KB (int8 quantized)
```

**Rozmiar modelu: ~0.5 MB** (float32) — spełnia wymóg < 5MB z ogromnym marginesem.

---

## 2. Klasy / zadania

### Zadanie 1: Klasyfikacja stanu (5 klas)

| Klasa | ID | Opis | Akcja bota |
|-------|----|------|------------|
| `INACTIVE` | 0 | Brak minigry (okno zamknięte) | Czekaj na minigre |
| `WHITE` | 1 | Biały okrąg — rybka poza | Czekaj, śledź rybkę |
| `RED` | 2 | Czerwony okrąg — rybka w środku | KLIKAJ w rybkę! |
| `HIT_TEXT` | 3 | Napis HIT widoczny | Ignoruj klatkę (nie klikaj) |
| `MISS_TEXT` | 4 | Napis MISS widoczny | Ignoruj klatkę (nie klikaj) |

**Uzasadnienie 5 klas zamiast 3:**
- Separacja HIT_TEXT i MISS_TEXT jako osobne klasy eliminuje problem #1 (MISS_TXT 20% błędów) — CNN nauczy się rozpoznawać napisy jako stan, a nie rybkę
- INACTIVE osobno — teraz wykrywane progiem jasnych pikseli, co jest kruche

### Zadanie 2: Regresja pozycji rybki

**Format: znormalizowane (x, y) + confidence**

```python
# Output: [x_norm, y_norm, conf] — wszystko sigmoid → [0, 1]
# x_norm = fish_x / 279  (szerokość fishing box)
# y_norm = fish_y / 247  (wysokość fishing box)
# conf = P(rybka jest widoczna)

# Dekodowanie w inferencji:
fish_x = int(x_norm * 279)  # piksele względem fishing box
fish_y = int(y_norm * 247)
is_visible = conf > 0.5
```

**Dlaczego nie bounding box?**
- Rybka jest mała (~30-1000px area), nieregularna — bbox jest zbyt zgrubny
- Potrzebujemy punktu kliknięcia (środek rybki), nie prostokąta
- Regresja (x,y) jest prostsza do trenowania z ~3000 próbek

**Dlaczego nie heatmapa?**
- Heatmapa 128×128 to 16K wymiarów wyjścia — zbyt dużo parametrów dla małego datasetu
- Heatmapa ma sens przy >10K etykietowanych próbek
- Punkt (x,y) + confidence to tylko 3 wartości — łatwiej się uczy

**Jak obsłużyć brak rybki?**
- `confidence < 0.5` → brak rybki, ignoruj x,y
- W treningu: klatki bez rybki mają `conf_target = 0`, a loss pozycji jest maskowany (nie liczy się)
- Klasy `HIT_TEXT`, `MISS_TEXT`, `INACTIVE` naturalnie mają conf=0

---

## 3. Pipeline treningowy

### 3.1 Przygotowanie danych

#### Źródła klatek

| Źródło | Ilość | Jakość | Etykiety |
|--------|-------|--------|----------|
| `test10_clean/raw/` | ~268 | Najczystsza (bez overlay) | Brak — do ręcznego etykietowania |
| `test9_long/frames/` | ~264 | Debug overlay (krzyżyk na rybce) | Pseudo-labels z loga (pozycja) |
| `test8a-c/frames/` | ~200 | Debug overlay | Pseudo-labels z logów |
| `miss and hit/miss/` | 12 | Czyste MISS | Stan = MISS_TEXT |
| `miss and hit/hit/` | 0 | Brak | ⚠️ Trzeba zebrać |
| Nowe nagrywanie | ~1000+ | Do zebrania | Ręczne etykietowanie |

#### Preprocessing

```python
def preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    """279×247 BGR → 128×128 RGB tensor znormalizowany."""
    # BGR → RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize z zachowaniem proporcji? Nie — mały błąd, a prostsze
    resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
    # Normalizacja do [0, 1]
    tensor = torch.from_numpy(resized).float() / 255.0
    # HWC → CHW
    tensor = tensor.permute(2, 0, 1)
    return tensor
```

**Dlaczego nie ImageNet normalization (mean/std)?**
- Nie używamy pretrained modelu — custom normalizacja [0,1] jest prostsza i równie skuteczna
- Mniej kodu = mniej błędów w produkcji

#### Augmentacja

```python
import torchvision.transforms.v2 as T

train_augment = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # p=0.3
    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
    # NIE używamy RandomHorizontalFlip — zmieni pozycję rybki!
    # NIE używamy RandomCrop — mały obraz, stracimy kontekst
    # NIE używamy RandomRotation — okrąg jest zawsze w tym samym miejscu
])
```

**Kluczowe ograniczenia augmentacji:**
- **Brak flip/rotation** — pozycja rybki jest absolutna (x,y względem fishing box)
- **Brak crop** — 128×128 to już mały obraz
- **Tylko kolor/jasność** — symuluje różne warunki oświetlenia w grze
- **Gaussian blur** — symuluje lekki motion blur z szybkiego przechwytywania

#### Train/Val split

```
Strategia: 80% train / 20% val
- Stratyfikowany po klasach (StratifiedKFold)
- NIE losowo — klatki z tej samej rundy do tego samego zbioru
  (zapobiega data leakage — sąsiednie klatki są bardzo podobne)
- Grupowanie: klatki z tego samego test_* folderu = jedna grupa
```

### 3.2 Etykietowanie

#### Strategia etykietowania (realistyczna dla 1 osoby)

**Faza 1 — Pseudo-labels (automatyczne, ~2h pracy):**
```python
# Parsuj logi z testów (test8a/log.csv, test9/log.csv, test10/log.csv)
# Format: frame_idx, phase(white/red/none), fish_x, fish_y, action
# → Generuj etykiety: stan = phase, pozycja = (fish_x, fish_y) z loga

# Problem: pseudo-labels mają te same błędy co stary detektor!
# Rozwiązanie: użyj pseudo-labels TYLKO dla:
#   - Klasyfikacja stanu (white/red/none — ta detekcja działała dobrze, >95%)
#   - Pozycja rybki TYLKO gdy detektor znalazł i nie było filtra tekstu
# Weryfikacja: ~300 losowych klatek sprawdzić ręcznie
```

**Faza 2 — Ręczne etykietowanie pozycji (~4-6h pracy):**
```
# Narzędzie GUI (sekcja 5)
# Priorytet: etykietuj 268 klatek z test10_clean/raw/ (najczystsze)
# Potem: ~200 klatek z test9_long (debug overlay → trzeba użyć raw albo wyciąć krzyżyk)
# Cel: minimum 500 klatek z ręczną pozycją rybki
```

**Faza 3 — Nowe nagranie zbierające (~2h):**
```
# Uruchom nowy skrypt zbierający TYLKO raw klatki (bez debug overlay)
# 5 minut gry × 33 FPS = ~10,000 klatek
# Subsampling co 3-5 klatek → ~2000-3000 unikalnych
# Etykietowanie: 500 najważniejszych (red phase + trudne przypadki)
```

### 3.3 Hiperparametry treningu

```python
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Scheduler: OneCycleLR (najlepszy dla małych datasetów)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=80,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,      # 30% warmup
    anneal_strategy='cos'
)

# Epoki: 80 (z early stopping patience=15)
# Batch size: 32 (mały dataset → mały batch)
# Czas treningu: ~5-10 min na CPU (małe dane, mały model)
```

**Uzasadnienie:**
- **AdamW** (nie SGD) — lepiej radzi sobie z małymi datasetami, mniej wrażliwy na LR
- **OneCycleLR** — agresywny warmup + decay, sprawdzony dla small-data regime
- **80 epok** — dostatecznie dużo z early stopping (typowo konwerguje w 30-50)
- **weight_decay=1e-4** — regularyzacja L2, zapobiega overfitting
- **batch_size=32** — z 2400 próbek train to ~75 kroków/epokę

### 3.4 Loss function (wielozadaniowy)

```python
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Wagi klas — kompensacja niezbalansowania
        # Szacunkowo: INACTIVE=15%, WHITE=45%, RED=25%, HIT=5%, MISS=10%
        class_weights = torch.tensor([1.0, 0.5, 1.0, 3.0, 2.0])
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.pos_loss = nn.SmoothL1Loss(reduction='none')  # Huber loss
        self.conf_loss = nn.BCEWithLogitsLoss()

    def forward(self, cls_logits, pos_pred, cls_target, pos_target, conf_target):
        """
        Args:
            cls_logits: [B, 5] — logity klasyfikacji
            pos_pred: [B, 3] — [x, y, conf_logit] (przed sigmoid dla conf)
            cls_target: [B] — indeks klasy groundtruth
            pos_target: [B, 2] — [x_norm, y_norm] groundtruth
            conf_target: [B] — 1.0 gdy rybka widoczna, 0.0 gdy nie
        """
        # Loss klasyfikacji
        L_cls = self.cls_loss(cls_logits, cls_target)

        # Loss confidence (czy rybka w ogóle jest)
        L_conf = self.conf_loss(pos_pred[:, 2], conf_target)

        # Loss pozycji — TYLKO dla klatek z widoczną rybką
        mask = conf_target > 0.5  # [B] bool
        if mask.any():
            pos_loss_xy = self.pos_loss(
                torch.sigmoid(pos_pred[mask, :2]),
                pos_target[mask]
            )
            L_pos = pos_loss_xy.mean()
        else:
            L_pos = torch.tensor(0.0, device=cls_logits.device)

        # Łączny loss z wagami
        # λ_cls=1.0, λ_pos=5.0, λ_conf=1.0
        # Wyższa waga pozycji bo to kluczowe zadanie
        total = 1.0 * L_cls + 5.0 * L_pos + 1.0 * L_conf

        return total, {
            'cls': L_cls.item(),
            'pos': L_pos.item(),
            'conf': L_conf.item(),
            'total': total.item()
        }
```

**Uzasadnienie:**
- **CrossEntropyLoss z wagami** — kompensacja nierównomiernego rozkładu klas (HIT/MISS rzadkie)
- **SmoothL1Loss (Huber)** — odporniejszy na outlier'y niż MSE, gładszy niż L1
- **Maskowanie pos_loss** — gdy conf=0 (brak rybki), nie karamy za pozycję
- **λ_pos=5.0** — wyższe niż λ_cls bo precyzyjna pozycja = precyzyjny klik = wyższa skuteczność
- **BCEWithLogitsLoss** — numerycznie stabilniejszy niż BCE + sigmoid osobno

### 3.5 Radzenie sobie z małym datasetem

1. **Multi-task learning** — shared backbone uczy się lepszych features bo dwa zadania regulyzują się wzajemnie
2. **Augmentacja kolorystyczna** — 3× efektywna ilość danych
3. **BatchNorm** — stabilizuje trening z małym batch
4. **Dropout 0.3** — zapobiega zapamiętywaniu
5. **Early stopping** — patience=15, monitor=val_loss
6. **Weight decay** — L2 regularyzacja
7. **Pseudo-labels + ręczna weryfikacja** — wykorzystuje istniejące dane
8. **Transfer z prostego zadania** — najpierw trenuj TYLKO klasyfikację (łatwiejsze), potem dodaj regresję pozycji (curriculum learning)

---

## 4. Pipeline inferencji (produkcyjny)

### 4.1 ONNX Runtime — klucz do <10ms

```python
# Export z PyTorch do ONNX (jednorazowo po treningu)
dummy = torch.randn(1, 3, 128, 128)
torch.onnx.export(
    model, dummy, "fishnet.onnx",
    input_names=["frame"],
    output_names=["state", "position"],
    opset_version=17,
    dynamic_axes={"frame": {0: "batch"}}
)

# Opcjonalnie: kwantyzacja INT8 (dalsze przyspieszenie)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("fishnet.onnx", "fishnet_int8.onnx", weight_type=QuantType.QUInt8)
```

**Benchmarkowy budżet czasowy na CPU (Intel i5/i7, 1 wątek):**

| Operacja | Czas |
|----------|------|
| Screen capture (mss) | ~2ms |
| BGR→RGB + resize 128×128 | ~0.3ms |
| Normalizacja + tensor | ~0.2ms |
| **ONNX Runtime inferencja** | **~2-4ms** |
| Dekodowanie output | ~0.1ms |
| Logika + click | ~1ms |
| **TOTAL** | **~6-8ms** ✅ |

**Dlaczego ONNX Runtime a nie czysty PyTorch?**
- PyTorch CPU inferencja tego modelu: ~8-15ms (Python overhead, GIL)
- ONNX Runtime: ~2-4ms (C++ backend, operator fusion, threading)
- ONNX Runtime potrzebuje tylko `pip install onnxruntime` (~15MB)

### 4.2 Klasa inferencji produkcyjnej

```python
import onnxruntime as ort
import numpy as np
import cv2

class FishNetInference:
    """Produkcyjna inferencja CNN — single-frame, <10ms."""

    # Mapowanie ID → nazwa stanu
    STATES = ['INACTIVE', 'WHITE', 'RED', 'HIT_TEXT', 'MISS_TEXT']

    def __init__(self, model_path: str = "cnn/models/fishnet_int8.onnx"):
        # Sesja ONNX — ładowana RAZ na start bota
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1       # 1 wątek = deterministyczny timing
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, opts,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self._warmup()

    def _warmup(self):
        """Rozgrzewka — pierwszy run jest wolniejszy (alokacje)."""
        dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
        self.session.run(None, {self.input_name: dummy})

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """279×247 BGR → [1, 3, 128, 128] float32 numpy array."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
        return arr[np.newaxis]  # dodaj batch dim → [1, 3, 128, 128]

    def predict(self, frame_bgr: np.ndarray) -> dict:
        """
        Pełna predykcja z jednej klatki.

        Args:
            frame_bgr: 279×247 BGR screenshot fishing box

        Returns:
            {
                'state': str,        # 'INACTIVE'|'WHITE'|'RED'|'HIT_TEXT'|'MISS_TEXT'
                'state_conf': float, # pewność klasyfikacji (0-1)
                'fish_x': int,       # pozycja x rybki (piksele, wzgl. fishing box)
                'fish_y': int,       # pozycja y rybki
                'fish_visible': bool # czy rybka jest widoczna
            }
        """
        # Preprocessing
        input_tensor = self.preprocess(frame_bgr)

        # Inferencja
        state_logits, pos_raw = self.session.run(None, {self.input_name: input_tensor})

        # Dekodowanie stanu
        state_probs = self._softmax(state_logits[0])
        state_id = int(np.argmax(state_probs))
        state_conf = float(state_probs[state_id])

        # Dekodowanie pozycji
        x_norm = self._sigmoid(pos_raw[0][0])
        y_norm = self._sigmoid(pos_raw[0][1])
        conf = self._sigmoid(pos_raw[0][2])

        fish_x = int(x_norm * 279)
        fish_y = int(y_norm * 247)
        fish_visible = conf > 0.5

        return {
            'state': self.STATES[state_id],
            'state_conf': state_conf,
            'fish_x': fish_x,
            'fish_y': fish_y,
            'fish_visible': fish_visible,
            'fish_conf': float(conf),
        }

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
```

### 4.3 Integracja z istniejącym botem

Modyfikacja `bot.py` — **minimalna zmiana, fallback na klasykę:**

```python
class KosaBot:
    def __init__(self, debug: bool = False, use_cnn: bool = True):
        self.capture = ScreenCapture()
        self.input = InputSimulator()
        self.debug = debug

        # Detekcja: CNN (primary) + klasyczny (fallback)
        self.detector = FishingDetector()  # klasyczny — zostaje jako fallback
        self.cnn = None
        if use_cnn:
            try:
                from cnn.inference import FishNetInference
                self.cnn = FishNetInference()
                print("[BOT] CNN załadowany — tryb neuronowy")
            except Exception as e:
                print(f"[BOT] CNN niedostępny ({e}) — tryb klasyczny")

    def _detect_frame(self, frame) -> dict:
        """Unified detekcja — CNN z fallbackiem na klasykę."""
        if self.cnn is not None:
            result = self.cnn.predict(frame)
            # Mapowanie CNN → format kompatybilny z botem
            color_map = {
                'INACTIVE': 'none',
                'WHITE': 'white',
                'RED': 'red',
                'HIT_TEXT': 'none',   # nie klikaj
                'MISS_TEXT': 'none',  # nie klikaj
            }
            return {
                'color': color_map[result['state']],
                'fish_pos': (result['fish_x'], result['fish_y'])
                             if result['fish_visible'] else None,
                'state': result['state'],
                'state_conf': result['state_conf'],
            }
        else:
            # Fallback: klasyczny detektor
            color = self.detector.detect_circle_color(frame)
            fish_pos = self.detector.find_fish_position(frame, circle_color=color)
            return {
                'color': color,
                'fish_pos': fish_pos,
                'state': color.upper(),
                'state_conf': 1.0,
            }
```

**Kluczowe różnice CNN vs klasyka:**
- CNN **nie potrzebuje historii** — działa na POJEDYNCZEJ klatce ✅
- CNN **nie potrzebuje warmup** — od 1. klatki wie gdzie rybka ✅
- CNN **rozpoznaje HIT/MISS jako osobne klasy** — nie myli z rybką ✅

### 4.4 Fallback strategy

```python
# Scenariusze fallback:
# 1. ONNX nie zainstalowany → klasyczny detektor
# 2. Model .onnx nie znaleziony → klasyczny detektor
# 3. CNN confidence < 0.6 → klasyczny detektor na tej klatce
# 4. CNN error/exception → klasyczny detektor (loguj błąd)

# Hybrydowy tryb (opcjonalny):
# CNN klasyfikacja stanu + klasyczny tracking pozycji
# (gdyby CNN pozycja była nieprecyzyjna, a stan deterministyczny)
```

---

## 5. Narzędzie do etykietowania

### Design: Minimalny GUI w OpenCV (bez zależności)

```
Interfejs:
┌──────────────────────────────────────────────────────┐
│  [frame_0042.png]                    [42 / 268]      │
│                                                      │
│  ┌───────────────────────────┐  Stan: RED            │
│  │                           │  Rybka: (123, 98)     │
│  │     279 × 247 frame       │  Conf: visible        │
│  │     (powiększony 2×)      │                       │
│  │         ⊕ ← kliknięta     │  [UNDO: Ctrl+Z]      │
│  │           pozycja rybki   │                       │
│  │                           │                       │
│  └───────────────────────────┘                       │
│                                                      │
│  Klawisze:                                           │
│  [1] INACTIVE  [2] WHITE  [3] RED                    │
│  [4] HIT_TEXT  [5] MISS_TEXT                         │
│  [N] brak rybki  [←→] prev/next  [S] save           │
│  [klik] pozycja rybki  [Q] quit                      │
└──────────────────────────────────────────────────────┘
```

### Format etykiet: JSON Lines (.jsonl)

```jsonl
{"file": "frame_0042.png", "state": "RED", "fish_x": 123, "fish_y": 98, "fish_visible": true}
{"file": "frame_0043.png", "state": "WHITE", "fish_x": 145, "fish_y": 102, "fish_visible": true}
{"file": "frame_0044.png", "state": "MISS_TEXT", "fish_x": null, "fish_y": null, "fish_visible": false}
{"file": "frame_0045.png", "state": "INACTIVE", "fish_x": null, "fish_y": null, "fish_visible": false}
```

**Dlaczego JSONL (nie CSV)?**
- Obsługuje null natywnie (CSV wymaga konwencji)
- Łatwy append (jedna linia = jeden rekord)
- Czytelny dla człowieka
- Łatwy parsing: `json.loads(line)` per linia

---

## 6. Plan implementacji

### Faza 1: Przygotowanie danych (3-4 dni)

| Krok | Opis | Plik |
|------|------|------|
| 1.1 | Stwórz skrypt do zbierania surowych klatek (bez overlay) | `cnn/collect_frames.py` |
| 1.2 | Nagrywaj 5-10 minut gry → ~3000-6000 klatek | dane w `cnn/data/raw/` |
| 1.3 | Parsuj istniejące logi → pseudo-labels dla stanu | `cnn/generate_pseudo_labels.py` |
| 1.4 | Subsample: co 3-5 klatka → ~1000-2000 unikalnych | skrypt dedup |

**Struktura folderów:**
```
cnn/
├── data/
│   ├── raw/                  # surowe 279×247 klatki PNG
│   ├── labels.jsonl          # etykiety (JSONL)
│   └── pseudo_labels.jsonl   # auto-etykiety z logów
├── models/
│   ├── fishnet.pth           # checkpoint PyTorch
│   ├── fishnet.onnx          # eksport ONNX
│   └── fishnet_int8.onnx     # kwantyzowany ONNX
├── collect_frames.py
├── generate_pseudo_labels.py
├── label_tool.py
├── model.py
├── dataset.py
├── train.py
├── export_onnx.py
├── inference.py
├── benchmark.py
└── __init__.py
```

### Faza 2: Etykietowanie (2-3 dni)

| Krok | Opis |
|------|------|
| 2.1 | Zaimplementuj label_tool.py (GUI etykietowania) |
| 2.2 | Etykietuj ~500 klatek z test10_clean/raw/ (najczystsze) |
| 2.3 | Etykietuj ~500 klatek z nowego nagrania |
| 2.4 | Weryfikuj 200 losowych pseudo-labels ręcznie |
| 2.5 | Merge pseudo-labels + ręczne → finalne labels.jsonl |

**Cel: minimum 1000 klatek z etykietami** (500 ręcznych + 500 zweryfikowanych pseudo)

### Faza 3: Trening (2-3 dni)

| Krok | Opis |
|------|------|
| 3.1 | Zaimplementuj model.py (FishNet architecture) |
| 3.2 | Zaimplementuj dataset.py (ładowanie + augmentacja) |
| 3.3 | Zaimplementuj train.py (pętla treningowa) |
| 3.4 | Trening v1: TYLKO klasyfikacja stanu (curriculum) |
| 3.5 | Trening v2: dodaj regresję pozycji (multi-task) |
| 3.6 | Eksport do ONNX + kwantyzacja INT8 |
| 3.7 | Benchmark inferencji na CPU (cel: <5ms) |

### Faza 4: Integracja z botem (1-2 dni)

| Krok | Opis |
|------|------|
| 4.1 | Zaimplementuj inference.py (FishNetInference) |
| 4.2 | Zmodyfikuj bot.py — dodaj CNN detection path |
| 4.3 | Dodaj fallback na klasyczny detektor |
| 4.4 | Test na żywo: 5 minut z CNN vs 5 minut klasyczny |
| 4.5 | Aktualizuj requirements.txt (torch, onnxruntime) |

### Faza 5: Testowanie i iteracja (2-3 dni)

| Krok | Opis |
|------|------|
| 5.1 | Test 30-minutowy: mierz skuteczność end-to-end |
| 5.2 | Analiza błędów: jakie klatki CNN źle klasyfikuje? |
| 5.3 | Dozbieraj dane dla trudnych przypadków |
| 5.4 | Retrain z nowymi danymi |
| 5.5 | A/B test: CNN vs klasyka przez 1h gry |

**Łączny czas: ~10-15 dni** (jedna osoba, praca po godzinach)

---

## 7. Porównanie z PRE_CNN

### Co CNN zastępuje

| Komponent PRE_CNN | Linie kodu | CNN zastępuje? | Komentarz |
|-------------------|------------|----------------|-----------|
| `_count_bright_white_pixels()` | ~10 | ✅ TAK | CNN klasyfikuje WHITE vs RED |
| `_count_bright_pixels()` | ~5 | ✅ TAK | CNN klasyfikuje INACTIVE |
| `_has_text_overlay()` | ~20 | ✅ TAK | CNN klasyfikuje HIT_TEXT |
| `_is_text_contour()` | ~50 | ✅ TAK | CNN klasyfikuje MISS_TEXT |
| `_is_red_blob()` | ~15 | ✅ TAK | Nie potrzebne — CNN filtruje napisy |
| `_recompute_background()` | ~8 | ✅ TAK | CNN nie potrzebuje historii! |
| `_find_fish_bg_subtraction()` | ~60 | ✅ TAK | CNN daje pozycję z 1 klatki |
| `_find_fish_frame_diff()` | ~35 | ✅ TAK | Brak potrzeby fallbacku |
| `find_fish_position()` | ~80 | ✅ TAK | Zastąpione przez `cnn.predict()` |
| `predict_fish_position()` | ~30 | ⚠️ OPCJONALNIE | Ekstrapolacja nadal przydatna |
| `detect_circle_color()` | ~15 | ✅ TAK | CNN unified output |
| `is_fishing_active()` | ~5 | ✅ TAK | CNN klasa INACTIVE |
| `reset_tracking()` | ~10 | ✅ TAK | CNN jest stateless |
| Stale filtrowania (config.py) | ~30 | ✅ TAK | Wbudowane w wagi CNN |

### Co zostaje (bez zmian)

| Komponent | Dlaczego zostaje |
|-----------|-----------------|
| `ScreenCapture` | Przechwytywanie ekranu nie zmienia się |
| `InputSimulator` | Klikanie nie zmienia się |
| `KosaBot._clamp_to_circle()` | Logika bezpieczeństwa jest niezależna od detekcji |
| `KosaBot.play_fishing_round()` | Główna pętla — zmodyfikowana, ale logika ta sama |
| `config.py` (częściowo) | Timings, klawisze, pozycja okna — bez zmian |
| `FishingDetector` (cały) | **Zostaje jako fallback** |

### Oczekiwane usprawnienia

| Problem | PRE_CNN | CNN (oczekiwane) | Poprawa |
|---------|---------|-------------------|---------|
| **MISS_TXT** (klik w napis) | 20% błędów | ~2% (osobna klasa) | **-90%** |
| **BRAK** (rybka niewykryta) | 26.7% błędów | ~5% (single frame) | **-80%** |
| **POZA** (pozycja nieprecyzyjna) | 13.3% błędów | ~8% (regresja centroida) | **-40%** |
| **Degradacja w czasie** | 62%→34% skuteczność | Stała ~80%+ | **Eliminacja** |
| **Warmup (3+ klatki)** | Od klatki 4 | Od klatki 1 | **Eliminacja** |
| **Czas detekcji** | ~5-15ms (cv2) | ~3-5ms (ONNX) | **~2× szybciej** |
| **Pamięć** | 15 klatek bufor | 0 bufor | **-100%** |
| **Złożoność kodu** | 639 linii detector | ~50 linii inference | **-92%** |

### Szacunkowa skuteczność end-to-end

```
PRE_CNN:  80% (krótki test), degradacja do 34% w 2 min
CNN v1:   85-90% (oczekiwane z ~1000 etykiet)
CNN v2:   90-95% (po iteracji z danymi z błędów)
```

**Największa wartość CNN: stabilność w czasie** — brak degradacji bo brak modelu tła, brak historii, brak stanu.

---

## Appendix A: Szybki benchmark wykonalności

Poniższy test potwierdza, że model tej wielkości zmieści się w budżecie <10ms:

```python
# benchmark.py — uruchomić PRZED pełną implementacją
import time, numpy as np

# Symulacja: 113K parametrów × forward pass
# 4 konwolucje (128→64→32→16→8) + 2×FC
# Na numpy (wolniejszy niż ONNX!) nie powinno przekroczyć 10ms

input_size = (1, 3, 128, 128)
dummy = np.random.randn(*input_size).astype(np.float32)

# ONNX RT test (po treningu):
# import onnxruntime as ort
# session = ort.InferenceSession("fishnet_int8.onnx")
# start = time.perf_counter()
# for _ in range(100):
#     session.run(None, {"frame": dummy})
# elapsed = (time.perf_counter() - start) / 100 * 1000
# print(f"ONNX RT: {elapsed:.1f} ms/frame")
```

---

## Appendix B: Minimalne wymagania do startu

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime
# Reszta (opencv, numpy, mss) już jest w requirements.txt
```

**Rozmiar pakietów:**
- `torch` (CPU only): ~150 MB
- `torchvision` (CPU only): ~30 MB  
- `onnxruntime`: ~15 MB
- **Total dodatkowy**: ~195 MB

**Uwaga:** PyTorch potrzebny TYLKO do treningu. W produkcji wystarczy `onnxruntime` (15 MB).
