import numpy as np
from edgelens.train import quantize, predict, bootstrap_difference


def test_zero_channels():
    q, s = quantize(np.zeros((4, 3), dtype=np.float32))
    assert np.isfinite(s).all() and (s > 0).all() and not q.any()


def test_error_bound():
    w = np.random.default_rng(2).normal(size=(64, 10)).astype(np.float32)
    q, s = quantize(w)
    assert q.dtype == np.int8
    assert np.all(np.abs(w - q * s) <= s / 2 + 1e-6)


def test_forward_contract():
    scores = predict(
        np.ones((2, 64)),
        [np.zeros((64, 64)), np.zeros((64, 10))],
        [np.zeros(64), np.arange(10)],
    )
    assert scores.shape == (2, 10) and (scores.argmax(axis=1) == 9).all()


def test_paired_interval():
    a = np.array([True, False, True])
    assert bootstrap_difference(a, a) == [0.0, 0.0]


def test_published_artifacts_replay():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    samples = json.loads((root / "artifacts/test_samples.json").read_text())["samples"]
    report = json.loads((root / "reports/host.json").read_text())
    x = np.array([s["input"] for s in samples], dtype=np.float32)
    y = np.array([s["label"] for s in samples])
    arrays = np.load(root / "artifacts/fp32.npz", allow_pickle=False)
    pred = predict(
        x, [arrays["w0"], arrays["w1"]], [arrays["b0"], arrays["b1"]]
    ).argmax(axis=1)
    assert float(np.mean(pred == y)) == report["fp32"]["test_accuracy"]
    assert len(samples) == report["protocol"]["test"]
    assert report["fp32"]["apple_device_latency_ms"] is None
    assert report["int8_weight_simulation"]["apple_process_memory_bytes"] is None
