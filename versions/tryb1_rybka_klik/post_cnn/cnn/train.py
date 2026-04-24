"""
Petla treningowa FishNet.

Trening multi-task: klasyfikacja stanu + regresja pozycji rybki.
Obsluguje:
- Early stopping (patience=15)
- Curriculum learning (opcjonalnie: najpierw sam state, potem multi-task)
- Logowanie do CSV
- Zapis najlepszego modelu
"""

import os
import sys
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn.model import FishNet, count_parameters, model_size_mb
from cnn.dataset import FishingDataset, create_train_val_split


class MultiTaskLoss(nn.Module):
    """
    Wielozadaniowy loss: klasyfikacja stanu + regresja pozycji.

    L = lambda_cls * CrossEntropy(state)
      + lambda_pos * SmoothL1(position)   [maskowany: tylko gdy rybka widoczna]
      + lambda_conf * BCE(confidence)
    """

    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None:
            self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # Domyslne wagi kompensujace niezbalansowanie klas
            # INACTIVE=15%, WHITE=45%, RED=25%, HIT=5%, MISS=10%
            weights = torch.tensor([1.0, 0.5, 1.0, 3.0, 2.0])
            self.cls_loss = nn.CrossEntropyLoss(weight=weights)

        self.pos_loss = nn.SmoothL1Loss(reduction='none')
        self.conf_loss = nn.BCEWithLogitsLoss()

    def forward(self, cls_logits, pos_pred, cls_target, pos_target, conf_target):
        """
        Args:
            cls_logits: [B, 5]
            pos_pred:   [B, 3] — (x_raw, y_raw, conf_raw) przed sigmoid
            cls_target: [B] — long
            pos_target: [B, 2] — (x_norm, y_norm)
            conf_target: [B] — float (1.0 = rybka widoczna)
        """
        # Klasyfikacja
        L_cls = self.cls_loss(cls_logits, cls_target)

        # Confidence
        L_conf = self.conf_loss(pos_pred[:, 2], conf_target)

        # Pozycja — maskowana (tylko gdy rybka widoczna)
        mask = conf_target > 0.5
        if mask.any():
            pred_xy = torch.sigmoid(pos_pred[mask, :2])
            L_pos = self.pos_loss(pred_xy, pos_target[mask]).mean()
        else:
            L_pos = torch.tensor(0.0, device=cls_logits.device)

        # Laczny loss
        total = 1.0 * L_cls + 5.0 * L_pos + 1.0 * L_conf

        return total, {
            'cls': L_cls.item(),
            'pos': L_pos.item(),
            'conf': L_conf.item(),
            'total': total.item(),
        }


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """Trenuje model przez jedną epoke."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pos_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch['image'].to(device)
        states = batch['state'].to(device)
        positions = batch['position'].to(device)
        confidences = batch['confidence'].to(device)

        optimizer.zero_grad()

        cls_logits, pos_raw = model(images)
        loss, loss_dict = criterion(cls_logits, pos_raw, states, positions, confidences)

        loss.backward()
        # Gradient clipping — zapobiega eksplozji gradientow z malym datasetem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # OneCycleLR wymaga step() po kazdym batchu
        if scheduler is not None:
            scheduler.step()

        total_loss += loss_dict['total'] * images.size(0)
        total_cls_loss += loss_dict['cls'] * images.size(0)
        total_pos_loss += loss_dict['pos'] * images.size(0)

        # Accuracy
        preds = cls_logits.argmax(dim=1)
        correct += (preds == states).sum().item()
        total += images.size(0)

    n = max(total, 1)
    return {
        'loss': total_loss / n,
        'cls_loss': total_cls_loss / n,
        'pos_loss': total_pos_loss / n,
        'accuracy': correct / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Walidacja modelu."""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pos_loss = 0.0
    correct = 0
    total = 0

    # Metryki pozycji
    pos_errors = []

    for batch in loader:
        images = batch['image'].to(device)
        states = batch['state'].to(device)
        positions = batch['position'].to(device)
        confidences = batch['confidence'].to(device)

        cls_logits, pos_raw = model(images)
        loss, loss_dict = criterion(cls_logits, pos_raw, states, positions, confidences)

        total_loss += loss_dict['total'] * images.size(0)
        total_cls_loss += loss_dict['cls'] * images.size(0)
        total_pos_loss += loss_dict['pos'] * images.size(0)

        preds = cls_logits.argmax(dim=1)
        correct += (preds == states).sum().item()
        total += images.size(0)

        # Blad pozycji (piksele)
        mask = confidences > 0.5
        if mask.any():
            pred_xy = torch.sigmoid(pos_raw[mask, :2])
            errors_px = torch.abs(pred_xy - positions[mask])
            errors_px[:, 0] *= 279  # x_norm → piksele
            errors_px[:, 1] *= 247  # y_norm → piksele
            pos_errors.extend(errors_px.cpu().numpy().tolist())

    n = max(total, 1)
    result = {
        'loss': total_loss / n,
        'cls_loss': total_cls_loss / n,
        'pos_loss': total_pos_loss / n,
        'accuracy': correct / n,
    }

    if pos_errors:
        import numpy as np
        errors = np.array(pos_errors)
        result['pos_error_x_mean'] = float(errors[:, 0].mean())
        result['pos_error_y_mean'] = float(errors[:, 1].mean())
        result['pos_error_px_mean'] = float(np.sqrt((errors ** 2).sum(axis=1)).mean())

    return result


def train(
    frames_dir: str = "cnn/data/raw",
    labels_file: str = "cnn/data/labels.jsonl",
    output_dir: str = "cnn/models",
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "cpu",
):
    """
    Pelny pipeline treningowy.

    Args:
        frames_dir: folder z klatkami PNG
        labels_file: plik JSONL z etykietami
        output_dir: folder na modele
        epochs: max epok
        batch_size: rozmiar batcha
        lr: learning rate
        patience: early stopping patience
        device: 'cpu' lub 'cuda'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = Path(frames_dir).parent

    # --- Split train/val ---
    train_labels = str(data_dir / "train_labels.jsonl")
    val_labels = str(data_dir / "val_labels.jsonl")
    create_train_val_split(labels_file, train_labels, val_labels)

    # --- Datasets ---
    train_dataset = FishingDataset(frames_dir, train_labels, augment=True)
    val_dataset = FishingDataset(frames_dir, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # --- Model ---
    model = FishNet().to(device)
    print(f"\n[Train] FishNet: {count_parameters(model):,} parametrow, "
          f"{model_size_mb(model):.2f} MB")

    # --- Loss, optimizer, scheduler ---
    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
    )

    # --- Training loop ---
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    log_path = output_path / "training_log.csv"
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
        'val_pos_error_px', 'lr', 'time_s',
    ])

    print(f"\n[Train] Start treningu: {epochs} epok, batch={batch_size}, lr={lr}")
    print(f"[Train] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"{'='*70}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train (scheduler step per batch wewnatrz)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        current_lr = optimizer.param_groups[0]['lr']

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        # Logging
        pos_err = val_metrics.get('pos_error_px_mean', -1)
        log_writer.writerow([
            epoch,
            f"{train_metrics['loss']:.4f}",
            f"{train_metrics['accuracy']:.4f}",
            f"{val_metrics['loss']:.4f}",
            f"{val_metrics['accuracy']:.4f}",
            f"{pos_err:.1f}" if pos_err >= 0 else "N/A",
            f"{current_lr:.6f}",
            f"{elapsed:.1f}",
        ])
        log_file.flush()

        # Print
        pos_str = f", pos_err={pos_err:.1f}px" if pos_err >= 0 else ""
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f}"
            f"{pos_str} | "
            f"lr={current_lr:.5f} | {elapsed:.1f}s"
        )

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            no_improve = 0

            # Zapisz najlepszy model
            best_path = output_path / "fishnet_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_metrics['accuracy'],
            }, best_path)
            print(f"  → Zapisano najlepszy model (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Train] Early stopping po {patience} epokach bez poprawy.")
                print(f"[Train] Najlepszy model: epoch {best_epoch}, "
                      f"val_loss={best_val_loss:.4f}")
                break

    log_file.close()

    # Zapisz ostatni model
    last_path = output_path / "fishnet_last.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
    }, last_path)

    print(f"\n{'='*70}")
    print(f"[Train] Trening zakonczony.")
    print(f"  Najlepszy model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"  Zapisano: {best_path}")
    print(f"  Log: {log_path}")

    return str(best_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trening FishNet CNN")
    parser.add_argument("--frames", default="cnn/data/frames", help="Folder z klatkami PNG")
    parser.add_argument("--labels", default="cnn/data/labels.jsonl", help="Plik JSONL z etykietami")
    parser.add_argument("--output", default="cnn/models", help="Folder na modele")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--prepare", action="store_true", help="Zbierz dane przed treningiem")
    parser.add_argument("--export", action="store_true", help="Eksportuj do ONNX po treningu")
    args = parser.parse_args()

    if args.prepare:
        from cnn.prepare_data import prepare_data
        labels_file, frames_dir, count = prepare_data()
        if count < 10:
            print("[ABORT] Za malo danych do treningu!")
            sys.exit(1)
        args.labels = labels_file
        args.frames = frames_dir

    best_model = train(
        frames_dir=args.frames,
        labels_file=args.labels,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    if args.export and best_model:
        from cnn.export_onnx import export_to_onnx
        export_to_onnx(checkpoint_path=best_model)
