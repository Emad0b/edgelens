# Benchmark evidence

`host.json` records a real Windows/NumPy run. Numerical loss after symmetric per-channel INT8 weight reconstruction is measured on the same held-out examples as FP32. The paired bootstrap reports the compressed-minus-original accuracy difference.

Tensor payload size counts weight, bias and scale arrays; it excludes archive headers and Core ML program metadata. Actual exported Core ML package size is a separate metric. The host INT8 evaluation reconstructs FP32 weights, so no speedup or memory reduction during inference can be inferred from this experiment.

Apple results are intentionally absent until the Swift benchmark runs on a Mac. The report uses sampled total process RSS rather than labelling it model-only memory. Repeated fresh-process runs and Instruments measurements are needed to support performance claims.

A release decision needs both acceptable accuracy and measured latency/memory on the target hardware. Passing the Python validation gate alone is insufficient. Reject a compressed candidate if it regresses the agreed accuracy limit or fails a device budget; preserve evidence of rejection rather than retuning on the held-out test.
