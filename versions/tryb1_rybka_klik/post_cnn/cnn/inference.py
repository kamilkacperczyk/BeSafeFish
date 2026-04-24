"""
Produkcyjna inferencja FishNet przez ONNX Runtime.

Klasa FishNetInference — loadowanie modelu raz na start,
predykcja <5ms na CPU, zero zależności od PyTorch.
"""

import os
import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class FishNetInference:
    """
    Produkcyjna inferencja CNN — single-frame, <10ms na CPU.

    Wymaga: pip install onnxruntime
    Model: fishnet_int8.onnx (lub fishnet.onnx)

    Uzycie:
        cnn = FishNetInference("cnn/models/fishnet_int8.onnx")
        result = cnn.predict(frame_bgr_279x247)
        # result = {'state': 'RED', 'fish_x': 123, 'fish_y': 98, ...}
    """

    STATES = ['INACTIVE', 'WHITE', 'RED', 'HIT_TEXT', 'MISS_TEXT']
    INPUT_SIZE = 128
    ORIG_W = 279
    ORIG_H = 247

    def __init__(self, model_path: str = None):
        """
        Laduje model ONNX. Wywolaj RAZ na start bota.

        Args:
            model_path: sciezka do pliku .onnx
                        Domyslnie szuka: cnn/models/fishnet_int8.onnx
                        Fallback: cnn/models/fishnet.onnx
        """
        if not HAS_ONNX:
            raise ImportError(
                "ONNX Runtime nie zainstalowany. "
                "Zainstaluj: pip install onnxruntime"
            )

        if model_path is None:
            # Szukaj modelu wzgledem katalogu tego pliku
            base = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(base, "models", "fishnet_int8.onnx"),
                os.path.join(base, "models", "fishnet.onnx"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break
            if model_path is None:
                raise FileNotFoundError(
                    f"Model ONNX nie znaleziony. Szukano w: {candidates}"
                )

        # Sesja ONNX — zoptymalizowana dla single-thread CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, opts,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

        # Warmup — pierwszy run alokuje pamiec (wolniejszy)
        self._warmup()

        print(f"[CNN] Model zaladowany: {model_path}")

    def _warmup(self):
        """Rozgrzewka — pierwszy inference jest wolniejszy."""
        dummy = np.random.randn(1, 3, self.INPUT_SIZE, self.INPUT_SIZE).astype(np.float32)
        self.session.run(None, {self.input_name: dummy})

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocessing: 279×247 BGR → [1, 3, 128, 128] float32.

        Args:
            frame_bgr: screenshot fishing box (279×247, BGR, uint8)

        Returns:
            numpy array [1, 3, 128, 128] float32, znormalizowany [0, 1]
        """
        # BGR → RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Resize
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        # Normalizacja + format
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
        return arr[np.newaxis]  # [1, 3, 128, 128]

    def predict(self, frame_bgr: np.ndarray) -> dict:
        """
        Pelna predykcja z jednej klatki.

        Args:
            frame_bgr: 279×247 BGR screenshot fishing box

        Returns:
            {
                'state': str,         # 'INACTIVE'|'WHITE'|'RED'|'HIT_TEXT'|'MISS_TEXT'
                'state_id': int,      # 0-4
                'state_conf': float,  # pewnosc klasyfikacji (0-1)
                'fish_x': int,        # pozycja x rybki (piksele wzgl. fishing box)
                'fish_y': int,        # pozycja y rybki
                'fish_visible': bool, # czy rybka jest widoczna (conf > 0.5)
                'fish_conf': float,   # pewnosc obecnosci rybki (0-1)
            }
        """
        # Preprocessing
        input_tensor = self.preprocess(frame_bgr)

        # Inferencja ONNX
        outputs = self.session.run(None, {self.input_name: input_tensor})
        state_logits = outputs[0][0]  # [5]
        pos_raw = outputs[1][0]       # [3]

        # Dekoduj stan
        state_probs = _softmax(state_logits)
        state_id = int(np.argmax(state_probs))
        state_conf = float(state_probs[state_id])

        # Dekoduj pozycje
        x_norm = _sigmoid(pos_raw[0])
        y_norm = _sigmoid(pos_raw[1])
        conf = _sigmoid(pos_raw[2])

        return {
            'state': self.STATES[state_id],
            'state_id': state_id,
            'state_conf': state_conf,
            'fish_x': int(x_norm * self.ORIG_W),
            'fish_y': int(y_norm * self.ORIG_H),
            'fish_visible': conf > 0.5,
            'fish_conf': float(conf),
        }

    def predict_raw(self, frame_bgr: np.ndarray):
        """
        Surowe outputy (bez dekodowania) — do debugowania.

        Returns:
            (state_logits, pos_raw) — numpy arrays
        """
        input_tensor = self.preprocess(frame_bgr)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0][0], outputs[1][0]


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerycznie stabilny softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _sigmoid(x: float) -> float:
    """Sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    import time

    if not HAS_ONNX:
        print("ONNX Runtime nie zainstalowany!")
        print("pip install onnxruntime")
        exit(1)

    # Benchmark
    print("=== Benchmark FishNet Inference ===")

    try:
        cnn = FishNetInference()
    except FileNotFoundError as e:
        print(f"Model nie znaleziony: {e}")
        print("Najpierw wytrenuj i wyeksportuj model (python -m cnn.export_onnx)")
        exit(1)

    # Symuluj input
    dummy_frame = np.random.randint(0, 256, (247, 279, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        cnn.predict(dummy_frame)

    # Benchmark
    N = 100
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        result = cnn.predict(dummy_frame)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times_arr = np.array(times)
    print(f"\nWyniki ({N} iteracji):")
    print(f"  Mean:   {times_arr.mean():.2f} ms")
    print(f"  Median: {np.median(times_arr):.2f} ms")
    print(f"  P95:    {np.percentile(times_arr, 95):.2f} ms")
    print(f"  P99:    {np.percentile(times_arr, 99):.2f} ms")
    print(f"  Min:    {times_arr.min():.2f} ms")
    print(f"  Max:    {times_arr.max():.2f} ms")
    print()
    print(f"Przykladowy output: {result}")
    budget = 10
    ok = times_arr.mean() < budget
    print(f"\nBudzet {budget}ms: {'OK' if ok else 'PRZEKROCZONY'}")
