"""
Benchmark inferencji FishNet na CPU.

Mierzy czas preprocessing + inferencja ONNX.
Uruchom po eksporcie modelu: python -m cnn.benchmark
"""

import time
import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def benchmark_preprocessing(n_iterations: int = 1000):
    """Benchmark samego preprocessingu (BGR→RGB, resize, normalize)."""
    # Symulowany input: 279×247 BGR
    dummy_bgr = np.random.randint(0, 256, (247, 279, 3), dtype=np.uint8)

    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()

        # Preprocessing (identyczny jak w FishNetInference.preprocess)
        rgb = cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis]

        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    print(f"Preprocessing ({n_iterations} iteracji):")
    print(f"  Mean:   {times.mean():.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  P99:    {np.percentile(times, 99):.3f} ms")


def benchmark_onnx_inference(model_path: str, n_iterations: int = 500):
    """Benchmark inferencji ONNX Runtime."""
    if not HAS_ONNX:
        print("ONNX Runtime nie zainstalowany!")
        return

    import os
    if not os.path.exists(model_path):
        print(f"Model nie znaleziony: {model_path}")
        return

    # Sesja ONNX
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path, opts,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
    for _ in range(20):
        session.run(None, {input_name: dummy})

    # Benchmark — TYLKO inferencja
    times_inf = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times_inf.append((t1 - t0) * 1000)

    times_inf = np.array(times_inf)
    print(f"\nONNX Inferencja — {model_path} ({n_iterations} iteracji):")
    print(f"  Mean:   {times_inf.mean():.3f} ms")
    print(f"  Median: {np.median(times_inf):.3f} ms")
    print(f"  P95:    {np.percentile(times_inf, 95):.3f} ms")
    print(f"  P99:    {np.percentile(times_inf, 99):.3f} ms")
    print(f"  Min:    {times_inf.min():.3f} ms")
    print(f"  Max:    {times_inf.max():.3f} ms")

    # Benchmark — end-to-end (preprocess + inferencja + decode)
    dummy_bgr = np.random.randint(0, 256, (247, 279, 3), dtype=np.uint8)
    times_e2e = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()

        # Preprocess
        rgb = cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis]

        # Inferencja
        state_logits, pos_raw = session.run(None, {input_name: arr})

        # Decode
        probs = np.exp(state_logits[0] - state_logits[0].max())
        probs /= probs.sum()
        state_id = int(np.argmax(probs))
        x_norm = 1.0 / (1.0 + np.exp(-pos_raw[0][0]))
        y_norm = 1.0 / (1.0 + np.exp(-pos_raw[0][1]))
        conf = 1.0 / (1.0 + np.exp(-pos_raw[0][2]))

        t1 = time.perf_counter()
        times_e2e.append((t1 - t0) * 1000)

    times_e2e = np.array(times_e2e)
    print(f"\nEnd-to-end (preprocess + infer + decode) ({n_iterations} iteracji):")
    print(f"  Mean:   {times_e2e.mean():.3f} ms")
    print(f"  Median: {np.median(times_e2e):.3f} ms")
    print(f"  P95:    {np.percentile(times_e2e, 95):.3f} ms")
    print(f"  P99:    {np.percentile(times_e2e, 99):.3f} ms")

    budget = 10.0
    ok = np.percentile(times_e2e, 99) < budget
    print(f"\n  Budzet {budget:.0f}ms (P99): {'✓ OK' if ok else '✗ PRZEKROCZONY'}")


if __name__ == "__main__":
    print("=" * 60)
    print("  FishNet Benchmark — CPU Inference")
    print("=" * 60)

    benchmark_preprocessing()

    import os
    models = [
        "cnn/models/fishnet.onnx",
        "cnn/models/fishnet_int8.onnx",
    ]
    for m in models:
        if os.path.exists(m):
            benchmark_onnx_inference(m)
        else:
            print(f"\n[skip] {m} — nie znaleziony")

    print("\n" + "=" * 60)
