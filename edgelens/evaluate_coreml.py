"""macOS-only correctness gate for actual exported Core ML models."""

import json
from pathlib import Path
import coremltools as ct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    samples = json.loads((ROOT / "artifacts/test_samples.json").read_text())["samples"]
    labels = np.array([s["label"] for s in samples])
    results = {}
    for name in ["FP32", "FP16", "INT8"]:
        model = ct.models.MLModel(
            str(ROOT / f"apple/Sources/EdgeLens/Resources/{name}.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        predictions = []
        for sample in samples:
            output = model.predict(
                {"pixels": np.array(sample["input"], dtype=np.float32).reshape(1, 64)}
            )
            predictions.append(int(np.asarray(output["logits"]).argmax()))
        results[name] = {
            "accuracy": float(np.mean(np.array(predictions) == labels)),
            "predictions": predictions,
        }
    results["int8_test_gate_passed"] = (
        results["INT8"]["accuracy"] >= results["FP32"]["accuracy"] - 0.01
    )
    (ROOT / "reports/coreml_accuracy.json").write_text(json.dumps(results, indent=2))
    print({k: v["accuracy"] for k, v in results.items() if isinstance(v, dict)})
    if not results["int8_test_gate_passed"]:
        raise SystemExit(
            "Core ML compression failed the preregistered 1 percentage point quality gate. Do not promote; keep the result."
        )


if __name__ == "__main__":
    main()
