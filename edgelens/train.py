import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def quantize(w):
    scale = np.maximum(np.max(np.abs(w), axis=0) / 127, 1e-12).astype(np.float32)
    return np.clip(np.rint(w / scale), -127, 127).astype(np.int8), scale


def predict(x, weights, biases):
    return np.maximum(x @ weights[0] + biases[0], 0) @ weights[1] + biases[1]


def bootstrap_difference(a, b):
    rng = np.random.default_rng(24)
    delta = b.astype(float) - a.astype(float)
    return np.quantile(
        [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)],
        [0.025, 0.975],
    ).tolist()


def run():
    data = load_digits()
    x = data.data.astype(np.float32) / 16
    y = data.target
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=17, stratify=y
    )
    train, val = train_test_split(
        train, test_size=0.2, random_state=18, stratify=y[train]
    )
    model = MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=220, random_state=17, alpha=0.001
    )
    model.fit(x[train], y[train])
    weights = [w.astype(np.float32) for w in model.coefs_]
    biases = [b.astype(np.float32) for b in model.intercepts_]
    encoded = [quantize(w) for w in weights]
    restored = [q.astype(np.float32) * s for q, s in encoded]
    np.savez(
        ROOT / "artifacts/fp32.npz",
        w0=weights[0],
        w1=weights[1],
        b0=biases[0],
        b1=biases[1],
    )
    np.savez(
        ROOT / "artifacts/int8.npz",
        q0=encoded[0][0],
        q1=encoded[1][0],
        s0=encoded[0][1],
        s1=encoded[1][1],
        b0=biases[0],
        b1=biases[1],
    )
    result = {}
    predictions = {}
    for name, ws in [("fp32", weights), ("int8_weight_simulation", restored)]:
        pred = predict(x[test], ws, biases).argmax(axis=1)
        predictions[name] = pred
        for _ in range(50):
            predict(x[test[:1]], ws, biases)
        timings = []
        for i in range(500):
            sample = x[test[i % len(test) : i % len(test) + 1]]
            start = time.perf_counter_ns()
            predict(sample, ws, biases)
            timings.append((time.perf_counter_ns() - start) / 1e6)
        result[name] = dict(
            test_accuracy=float(accuracy_score(y[test], pred)),
            validation_accuracy=float(
                accuracy_score(y[val], predict(x[val], ws, biases).argmax(axis=1))
            ),
            confusion_matrix=confusion_matrix(y[test], pred).tolist(),
            host_numpy_p50_ms=float(np.percentile(timings, 50)),
            host_numpy_p95_ms=float(np.percentile(timings, 95)),
            apple_device_latency_ms=None,
            apple_process_memory_bytes=None,
        )
    result["paired_accuracy_delta_95ci"] = bootstrap_difference(
        predictions["fp32"] == y[test], predictions["int8_weight_simulation"] == y[test]
    )
    result["tensor_payload_bytes"] = dict(
        fp32=sum(a.nbytes for a in weights + biases),
        int8=sum(q.nbytes + s.nbytes for q, s in encoded)
        + sum(b.nbytes for b in biases),
    )
    result["protocol"] = dict(
        seed=17,
        train=len(train),
        validation=len(val),
        test=len(test),
        dataset="scikit-learn bundled UCI optical recognition digits, 8x8, 10 classes",
        fingerprint=hashlib.sha256(data.data.tobytes() + y.tobytes()).hexdigest(),
        platform=platform.platform(),
        numpy=np.__version__,
        timing="warm single-sample NumPy CPU forward; 50 warmups, 500 measurements; excludes preprocessing",
        caveat="INT8 weights dequantized for NumPy inference: compression error simulation, not integer compute or Core ML speed.",
    )
    result["validation_gate_passed"] = (
        result["int8_weight_simulation"]["validation_accuracy"]
        >= result["fp32"]["validation_accuracy"] - 0.01
    )
    (ROOT / "reports/host.json").write_text(json.dumps(result, indent=2))
    (ROOT / "artifacts/test_samples.json").write_text(
        json.dumps(
            dict(
                samples=[dict(input=x[i].tolist(), label=int(y[i])) for i in test],
                dataset_sha256=result["protocol"]["fingerprint"],
            )
        )
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["fp32", "int8_weight_simulation"]
            },
            indent=2,
        )
    )
    print(
        "Test accuracy:",
        {k: result[k]["test_accuracy"] for k in ["fp32", "int8_weight_simulation"]},
    )


if __name__ == "__main__":
    run()
