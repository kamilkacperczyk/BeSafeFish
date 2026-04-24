"""
Trening FishPatchCNN — klasyfikator patch 32x32: fish vs not_fish.

Uzywa TYLKO zweryfikowanych patchy (verified=True) z labels.json.
Augmentacja w PyTorch (flip, rotate, color jitter) balansuje klasy.

Architektura:
  3x Conv2d(3x3) + BN + ReLU + MaxPool → FC → sigmoid
  Input: 32x32x3 BGR
  Output: 1 (fish probability)

Export: PyTorch .pth + ONNX .onnx
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PATCHES_DIR = SCRIPT_DIR / 'patches'
FISH_DIR = PATCHES_DIR / 'fish'
NOT_FISH_DIR = PATCHES_DIR / 'not_fish'
LABELS_FILE = PATCHES_DIR / 'labels.json'
MODEL_DIR = SCRIPT_DIR / 'models'

PATCH_SIZE = 32


# --- Dataset ---
class FishPatchDataset(Dataset):
    """Dataset z zweryfikowanych patchy."""

    def __init__(self, samples, augment=False):
        """
        Args:
            samples: list of (path, label) — label 1=fish, 0=not_fish
            augment: czy stosowac augmentacje
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Fallback: czarny patch
            img = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)

        # Augmentacja
        if self.augment:
            # Random flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)  # horizontal
            if random.random() > 0.5:
                img = cv2.flip(img, 0)  # vertical

            # Random rotation (0, 90, 180, 270)
            rot = random.choice([0, 1, 2, 3])
            if rot == 1:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 2:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rot == 3:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Random brightness
            beta = random.randint(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)

            # Random small shift (±3px)
            dx, dy = random.randint(-3, 3), random.randint(-3, 3)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (PATCH_SIZE, PATCH_SIZE),
                                 borderMode=cv2.BORDER_REFLECT_101)

        # BGR -> tensor [C, H, W], float32, normalized 0-1
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        return tensor, torch.tensor(label, dtype=torch.float32)


# --- Model ---
class FishPatchCNN(nn.Module):
    """
    Maly CNN do klasyfikacji patchy 32x32.

    3 bloki conv: 3->16->32->64, kazdy z BN + ReLU + MaxPool(2)
    Po 3 poolach: 32/8 = 4x4 feature map
    FC: 64*4*4=1024 -> 64 -> 1
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32x3 -> 16x16x16
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16x16x16 -> 8x8x32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 8x8x32 -> 4x4x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)


def load_verified_samples():
    """Laduje tylko zweryfikowane patche."""
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    fish_samples = []
    neg_samples = []

    for l in labels:
        if l.get('verified') is not True:
            continue

        cls = l['class']
        if cls == 'fish':
            path = FISH_DIR / l['file']
        else:
            path = NOT_FISH_DIR / l['file']

        if not path.exists():
            continue

        if cls == 'fish':
            fish_samples.append((path, 1))
        else:
            neg_samples.append((path, 0))

    return fish_samples, neg_samples


def train():
    """Trenuj FishPatchCNN."""
    print("=== Trening FishPatchCNN ===\n")

    fish_samples, neg_samples = load_verified_samples()
    print(f"Zweryfikowane: fish={len(fish_samples)}, not_fish={len(neg_samples)}")

    if len(fish_samples) < 5:
        print("BLAD: Za malo zweryfikowanych patchy fish!")
        return

    # --- Balansowanie klas przez oversampling fish ---
    # Oversampluj fish zeby mialy tyle samo co neg
    oversample_factor = max(1, len(neg_samples) // len(fish_samples))
    fish_oversampled = fish_samples * oversample_factor
    # Docinamy do dokladnego rozmiaru neg
    while len(fish_oversampled) < len(neg_samples):
        fish_oversampled.append(random.choice(fish_samples))
    fish_oversampled = fish_oversampled[:len(neg_samples)]

    all_samples = fish_oversampled + neg_samples
    random.shuffle(all_samples)

    print(f"Po oversamplingu: fish={len(fish_oversampled)}, neg={len(neg_samples)}")
    print(f"Razem: {len(all_samples)}")

    # --- Split: 85% train, 15% val ---
    split = int(0.85 * len(all_samples))
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]

    # Upewnij sie ze w val sa oba klasy
    val_fish = sum(1 for _, l in val_samples if l == 1)
    val_neg = sum(1 for _, l in val_samples if l == 0)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} (fish={val_fish}, neg={val_neg})")

    train_ds = FishPatchDataset(train_samples, augment=True)
    val_ds = FishPatchDataset(val_samples, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # --- Model ---
    device = torch.device('cpu')
    model = FishPatchCNN().to(device)

    # Policz parametry
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {params:,} parametrow")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # --- Trening ---
    epochs = 50
    best_val_acc = 0
    best_state = None

    print(f"\nTrening: {epochs} epok\n")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for imgs, labels_batch in train_loader:
            imgs = imgs.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels_batch).sum().item()
            train_total += imgs.size(0)

        scheduler.step()

        # Val
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_tp = 0  # true positive (fish predicted as fish)
        val_fp = 0  # false positive
        val_fn = 0  # false negative

        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                imgs = imgs.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels_batch).sum().item()
                val_total += imgs.size(0)

                val_tp += ((preds == 1) & (labels_batch == 1)).sum().item()
                val_fp += ((preds == 1) & (labels_batch == 0)).sum().item()
                val_fn += ((preds == 0) & (labels_batch == 1)).sum().item()

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        precision = val_tp / max(val_tp + val_fp, 1)
        recall = val_tp / max(val_tp + val_fn, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: loss={train_loss/train_total:.4f} acc={train_acc:.1%} | "
                  f"Val: acc={val_acc:.1%} prec={precision:.1%} rec={recall:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    # --- Zapisz model ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Wczytaj najlepszy
    if best_state:
        model.load_state_dict(best_state)

    pth_path = MODEL_DIR / 'fish_patch_cnn.pth'
    torch.save(model.state_dict(), pth_path)
    print(f"\nNajlepszy val acc: {best_val_acc:.1%}")
    print(f"Model zapisany: {pth_path} ({pth_path.stat().st_size / 1024:.1f} KB)")

    # --- Export ONNX ---
    onnx_path = MODEL_DIR / 'fish_patch_cnn.onnx'
    model.eval()
    dummy = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=['patch'],
        output_names=['logit'],
        dynamic_axes={'patch': {0: 'batch'}, 'logit': {0: 'batch'}},
        opset_version=11,
    )
    print(f"ONNX zapisany: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # --- Test inference ONNX ---
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        dummy_np = dummy.numpy()

        import time
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            result = sess.run(None, {'patch': dummy_np})
            times.append((time.perf_counter() - t0) * 1000)

        logit = result[0][0]
        prob = 1.0 / (1.0 + np.exp(-logit))
        print(f"\nONNX inference test: logit={logit:.3f}, prob={prob:.3f}")
        print(f"ONNX speed: mean={np.mean(times):.2f}ms, "
              f"median={np.median(times):.2f}ms")
    except ImportError:
        print("onnxruntime nie zainstalowany — pomijam test ONNX")

    print("\nGotowe! Nastepny krok: python -m cnn.train_patch_cnn test")


def test():
    """Test modelu na wszystkich zweryfikowanych patchach."""
    print("=== Test FishPatchCNN ===\n")

    fish_samples, neg_samples = load_verified_samples()
    all_samples = fish_samples + neg_samples
    print(f"Testuje na: fish={len(fish_samples)}, neg={len(neg_samples)}")

    # ONNX
    onnx_path = MODEL_DIR / 'fish_patch_cnn.onnx'
    if not onnx_path.exists():
        print(f"BLAD: Brak modelu {onnx_path}")
        return

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))

    tp, fp, tn, fn = 0, 0, 0, 0
    errors = []

    for path, label in all_samples:
        img = cv2.imread(str(path))
        if img is None:
            continue

        # Preprocess
        inp = img.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))  # HWC -> CHW
        inp = inp[np.newaxis, ...]  # batch dim

        logit = sess.run(None, {'patch': inp})[0][0]
        prob = 1.0 / (1.0 + np.exp(-logit))
        pred = 1 if prob > 0.5 else 0

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
            errors.append(('FP', path.name, prob))
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
            errors.append(('FN', path.name, prob))

    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    print(f"\nWyniki:")
    print(f"  Accuracy:  {acc:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.3f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    if errors:
        print(f"\nBledy ({len(errors)}):")
        for err_type, name, prob in errors[:20]:
            print(f"  {err_type}: {name[:60]} (prob={prob:.3f})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    else:
        print(f"Uzycie: python -m cnn.train_patch_cnn [train|test]")
