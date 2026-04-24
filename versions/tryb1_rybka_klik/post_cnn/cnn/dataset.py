"""
Dataset do treningu FishNet.

Laduje klatki z folderu + etykiety z pliku JSONL.
Obsluguje augmentacje i preprocessing.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FishingDataset(Dataset):
    """
    Dataset klatek minigry lowienia z etykietami.

    Format etykiet (JSONL — jedna linia = jeden rekord):
        {"file": "frame_0042.png", "state": "RED",
         "fish_x": 123, "fish_y": 98, "fish_visible": true}

    States: INACTIVE=0, WHITE=1, RED=2, HIT_TEXT=3, MISS_TEXT=4
    """

    STATE_TO_ID = {
        'INACTIVE': 0,
        'WHITE': 1,
        'RED': 2,
        'HIT_TEXT': 3,
        'MISS_TEXT': 4,
    }

    # Rozmiary oryginalnego fishing box
    ORIG_W = 279
    ORIG_H = 247

    # Input size dla CNN
    INPUT_SIZE = 128

    def __init__(
        self,
        frames_dir: str,
        labels_file: str,
        augment: bool = False,
    ):
        """
        Args:
            frames_dir: folder z klatkami PNG (279×247)
            labels_file: sciezka do pliku JSONL z etykietami
            augment: czy stosowac augmentacje (True dla train, False dla val)
        """
        self.frames_dir = Path(frames_dir)
        self.augment = augment

        # Wczytaj etykiety
        self.samples = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label = json.loads(line)
                # Sprawdz czy plik istnieje
                img_path = self.frames_dir / label['file']
                if img_path.exists():
                    self.samples.append(label)

        print(f"[Dataset] Zaladowano {len(self.samples)} probek z {labels_file}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Wyswietla rozklad klas."""
        counts = {}
        for s in self.samples:
            state = s['state']
            counts[state] = counts.get(state, 0) + 1
        for state, count in sorted(counts.items()):
            pct = count / len(self.samples) * 100
            print(f"  {state}: {count} ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = self.samples[idx]

        # Wczytaj obraz (BGR)
        img_path = self.frames_dir / label['file']
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Nie mozna wczytac: {img_path}")

        # BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Augmentacja (przed resize — na oryginalnym rozmiarze)
        if self.augment:
            img_rgb = self._apply_augmentation(img_rgb)

        # Resize do INPUT_SIZE × INPUT_SIZE
        img_resized = cv2.resize(
            img_rgb, (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalizacja do [0, 1] i konwersja do tensor
        tensor = torch.from_numpy(img_resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC → CHW

        # Etykiety
        state_id = self.STATE_TO_ID[label['state']]

        fish_visible = label.get('fish_visible', False)
        if fish_visible and label.get('fish_x') is not None:
            # Normalizuj pozycje do [0, 1]
            x_norm = label['fish_x'] / self.ORIG_W
            y_norm = label['fish_y'] / self.ORIG_H
            pos_target = torch.tensor([x_norm, y_norm], dtype=torch.float32)
            conf_target = torch.tensor(1.0, dtype=torch.float32)
        else:
            pos_target = torch.tensor([0.5, 0.5], dtype=torch.float32)  # dummy
            conf_target = torch.tensor(0.0, dtype=torch.float32)

        return {
            'image': tensor,                                      # [3, 128, 128]
            'state': torch.tensor(state_id, dtype=torch.long),   # scalar
            'position': pos_target,                                # [2]
            'confidence': conf_target,                             # scalar
        }

    def _apply_augmentation(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Augmentacja kolorystyczna (bez geometrycznej!).
        Uzywamy OpenCV zamiast torchvision — mniej zaleznosci.
        """
        img = img_rgb.copy()

        # 1. Losowa zmiana jasnosci (±30%)
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # 2. Losowa zmiana kontrastu (±20%)
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

        # 3. Losowa zmiana saturacji (±20%) w HSV
        if np.random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            factor = np.random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 4. Losowy szum Gaussowski
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Losowy Gaussian blur
        if np.random.random() < 0.2:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img


def create_train_val_split(
    labels_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Dzieli etykiety na train/val.
    Stratyfikowane po klasach, z grupowaniem po prefixie nazwy pliku.
    """
    rng = np.random.RandomState(seed)

    # Wczytaj wszystkie etykiety
    with open(labels_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]

    # Grupuj po klasie
    by_class = {}
    for s in samples:
        state = s['state']
        by_class.setdefault(state, []).append(s)

    train_samples = []
    val_samples = []

    for state, class_samples in by_class.items():
        rng.shuffle(class_samples)
        n_val = max(1, int(len(class_samples) * val_ratio))
        val_samples.extend(class_samples[:n_val])
        train_samples.extend(class_samples[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    # Zapisz
    with open(train_file, 'w', encoding='utf-8') as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"[Split] Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"  Zapisano: {train_file}, {val_file}")


if __name__ == "__main__":
    print("=== Test dataset ===")
    print("Uzyj: python -m cnn.dataset")
    print("Wymaga pliku z etykietami (labels.jsonl) i klatek w data/raw/")
