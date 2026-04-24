"""
FishNet — lekka CNN do detekcji stanu minigry i pozycji rybki.

Architektura: 4 bloki konwolucyjne + GAP + 2 heads (klasyfikacja + regresja).
~113K parametrów, <0.5MB wag, <5ms inferencja ONNX na CPU.

Input:  [B, 3, 128, 128] — RGB znormalizowane [0, 1]
Output: state_logits [B, 5], position [B, 3] (x, y, conf — raw, przed sigmoid)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FishNet(nn.Module):
    """
    Lightweight multi-task CNN for fishing minigame.

    Tasks:
        1. State classification (5 classes):
           INACTIVE=0, WHITE=1, RED=2, HIT_TEXT=3, MISS_TEXT=4
        2. Fish position regression:
           (x_norm, y_norm, confidence) — all sigmoid-activated

    Architecture:
        - 4 conv blocks: 3→16→32→64→64 channels
        - MaxPool 2×2 after each block: 128→64→32→16→8
        - Global Average Pooling → 64-D vector
        - Two FC heads with Dropout(0.3)
    """

    # Nazwy klas — stałe
    STATE_NAMES = ['INACTIVE', 'WHITE', 'RED', 'HIT_TEXT', 'MISS_TEXT']
    NUM_STATES = 5

    def __init__(self):
        super().__init__()

        # Backbone: 4 bloki konwolucyjne z MaxPooling
        self.backbone = nn.Sequential(
            # Block 1: 3→16, 128→64
            ConvBlock(3, 16),
            ConvBlock(16, 16),
            nn.MaxPool2d(2, 2),

            # Block 2: 16→32, 64→32
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2),

            # Block 3: 32→64, 32→16
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),

            # Block 4: 64→64, 16→8
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling → [B, 64, 1, 1] → [B, 64]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Head 1: klasyfikacja stanu (5 klas)
        self.head_state = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, self.NUM_STATES),
        )

        # Head 2: regresja pozycji rybki (x, y, confidence)
        self.head_position = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 3),  # x_raw, y_raw, conf_raw (przed sigmoid)
        )

        # Inicjalizacja wag
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization — najlepsza dla ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: [B, 3, 128, 128] — RGB input, normalized to [0, 1]

        Returns:
            state_logits: [B, 5] — raw logits (użyj softmax/argmax w inferencji)
            position_raw: [B, 3] — (x_raw, y_raw, conf_raw) przed sigmoid
        """
        # Backbone
        features = self.backbone(x)          # [B, 64, 8, 8]
        features = self.gap(features)        # [B, 64, 1, 1]
        features = features.flatten(1)       # [B, 64]

        # Multi-task heads
        state_logits = self.head_state(features)     # [B, 5]
        position_raw = self.head_position(features)  # [B, 3]

        return state_logits, position_raw

    def predict(self, x: torch.Tensor) -> dict:
        """
        Convenience method — predykcja z dekodowaniem.

        Args:
            x: [1, 3, 128, 128] — pojedyncza klatka

        Returns:
            dict z polami: state, state_conf, fish_x, fish_y, fish_visible
        """
        self.eval()
        with torch.no_grad():
            state_logits, pos_raw = self(x)

        # Dekoduj stan
        state_probs = torch.softmax(state_logits[0], dim=0)
        state_id = int(torch.argmax(state_probs))
        state_conf = float(state_probs[state_id])

        # Dekoduj pozycję
        pos_sigmoid = torch.sigmoid(pos_raw[0])
        x_norm = float(pos_sigmoid[0])
        y_norm = float(pos_sigmoid[1])
        conf = float(pos_sigmoid[2])

        return {
            'state': self.STATE_NAMES[state_id],
            'state_id': state_id,
            'state_conf': state_conf,
            'fish_x': int(x_norm * 279),   # piksele wzgl. fishing box
            'fish_y': int(y_norm * 247),
            'fish_visible': conf > 0.5,
            'fish_conf': conf,
        }


def count_parameters(model: nn.Module) -> int:
    """Liczy parametry modelu."""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Szacuje rozmiar modelu w MB (float32)."""
    params = count_parameters(model)
    return params * 4 / (1024 * 1024)


if __name__ == "__main__":
    # Szybki test architektury
    model = FishNet()
    print(f"FishNet — parametry: {count_parameters(model):,}")
    print(f"FishNet — rozmiar:   {model_size_mb(model):.2f} MB")
    print()

    # Test forward pass
    dummy = torch.randn(2, 3, 128, 128)
    state, pos = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"State:  {state.shape}  (logits)")
    print(f"Pos:    {pos.shape}    (x_raw, y_raw, conf_raw)")
    print()

    # Test predict
    single = torch.randn(1, 3, 128, 128)
    result = model.predict(single)
    print(f"Predict: {result}")
