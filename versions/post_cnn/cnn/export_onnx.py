"""
Eksport modelu PyTorch → ONNX + opcjonalna kwantyzacja INT8.

Uruchomienie:
    python -m cnn.export_onnx
    python -m cnn.export_onnx --checkpoint cnn/models/fishnet_best.pth
    python -m cnn.export_onnx --no-quantize
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np

from cnn.model import FishNet


def export_to_onnx(
    checkpoint_path: str = "cnn/models/fishnet_best.pth",
    output_path: str = "cnn/models/fishnet.onnx",
    quantize: bool = True,
):
    """
    Eksportuje model PyTorch do ONNX (+ opcjonalna kwantyzacja INT8).

    Args:
        checkpoint_path: sciezka do checkpointu .pth
        output_path: sciezka wyjsciowa .onnx
        quantize: czy kwantyzowac do INT8
    """
    # Laduj model
    model = FishNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[Export] Model zaladowany z {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Val loss: {checkpoint.get('val_loss', '?')}, "
          f"Val acc: {checkpoint.get('val_accuracy', '?')}")

    # Dummy input
    dummy = torch.randn(1, 3, 128, 128)

    # Eksport ONNX
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["frame"],
        output_names=["state_logits", "position_raw"],
        opset_version=18,
        dynamic_axes={
            "frame": {0: "batch"},
            "state_logits": {0: "batch"},
            "position_raw": {0: "batch"},
        },
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"[Export] ONNX zapisany: {output_path} ({file_size:.1f} KB)")

    # Kwantyzacja INT8
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quant_path = output_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(
                output_path,
                quant_path,
                weight_type=QuantType.QUInt8,
            )
            quant_size = os.path.getsize(quant_path) / 1024
            print(f"[Export] INT8 ONNX: {quant_path} ({quant_size:.1f} KB)")
            print(f"  Kompresja: {file_size/quant_size:.1f}×")
        except ImportError:
            print("[Export] onnxruntime.quantization niedostepne — pominiety INT8")

    # Weryfikacja
    _verify_onnx(output_path, model)


def _verify_onnx(onnx_path: str, torch_model: torch.nn.Module):
    """Porownuje output ONNX z PyTorch — upewnia sie ze eksport jest poprawny."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Verify] onnxruntime nie zainstalowany — pominiety test")
        return

    # PyTorch output
    dummy = torch.randn(1, 3, 128, 128)
    torch_model.eval()
    with torch.no_grad():
        pt_state, pt_pos = torch_model(dummy)

    # ONNX output
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_state, ort_pos = session.run(None, {"frame": dummy.numpy()})

    # Porownaj
    state_diff = np.abs(pt_state.numpy() - ort_state).max()
    pos_diff = np.abs(pt_pos.numpy() - ort_pos).max()

    print(f"\n[Verify] Max roznica PyTorch vs ONNX:")
    print(f"  State logits: {state_diff:.6f}")
    print(f"  Position:     {pos_diff:.6f}")

    ok = state_diff < 1e-4 and pos_diff < 1e-4
    print(f"  Status: {'OK' if ok else 'UWAGA — roznice ponad prog!'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eksport FishNet do ONNX")
    parser.add_argument(
        "--checkpoint", default="cnn/models/fishnet_best.pth",
        help="Checkpoint PyTorch (.pth)"
    )
    parser.add_argument(
        "--output", default="cnn/models/fishnet.onnx",
        help="Plik wyjsciowy ONNX"
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Pominij kwantyzacje INT8"
    )
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        quantize=not args.no_quantize,
    )
