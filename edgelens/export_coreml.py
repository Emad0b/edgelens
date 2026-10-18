"""Run on macOS: export actual Core ML FP32/FP16/INT8-weight ML Programs."""

import json
import shutil
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpLinearQuantizerConfig,
    linear_quantize_weights,
)

ROOT = Path(__file__).resolve().parents[1]


class Network(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.hidden = torch.nn.Linear(64, 64)
        self.output = torch.nn.Linear(64, 10)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(a["w0"].T.copy()))
            self.hidden.bias.copy_(torch.from_numpy(a["b0"]))
            self.output.weight.copy_(torch.from_numpy(a["w1"].T.copy()))
            self.output.bias.copy_(torch.from_numpy(a["b1"]))

    def forward(self, pixels):
        return self.output(torch.relu(self.hidden(pixels)))


def main():
    a = np.load(ROOT / "artifacts/fp32.npz", allow_pickle=False)
    traced = torch.jit.trace(Network(a).eval(), torch.zeros(1, 64))
    resources = ROOT / "apple/Sources/EdgeLens/Resources"
    sizes = {}
    for name, precision in [
        ("FP32", ct.precision.FLOAT32),
        ("FP16", ct.precision.FLOAT16),
    ]:
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="pixels", shape=(1, 64))],
            outputs=[ct.TensorType(name="logits")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=precision,
        )
        variants = [(name, model)]
        if name == "FP16":
            config = OptimizationConfig(
                global_config=OpLinearQuantizerConfig(
                    mode="linear_symmetric", dtype="int8", weight_threshold=0
                )
            )
            variants.append(("INT8", linear_quantize_weights(model, config=config)))
        for label, variant in variants:
            destination = resources / f"{label}.mlpackage"
            variant.save(str(destination))
            sizes[label] = sum(
                p.stat().st_size for p in destination.rglob("*") if p.is_file()
            )
    shutil.copy2(ROOT / "artifacts/test_samples.json", resources / "test_samples.json")
    (ROOT / "reports/coreml_packages.json").write_text(json.dumps(sizes, indent=2))
    print("Actual package bytes:", sizes)


if __name__ == "__main__":
    main()
