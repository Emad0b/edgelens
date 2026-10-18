# EdgeLens

A reproducible compression experiment and native macOS Core ML benchmark app. Compare a small digit classifier before and after weight compression without confusing file size, inference speed and numerical accuracy.

## Verification status

The Python experiment and unit tests run on Windows. Core ML conversion, Swift compilation and Apple-device measurements require macOS and are **not verified by this Windows run**. Missing device results remain null, never simulated. This is a compact benchmarking reference project, not a claim of production-scale vision performance.

## Local experiment

Python 3.12 recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
python -m edgelens.train
python -m pytest -q
```

Training produces FP32 weights, an INT8 storage representation, held-out inputs, confusion matrices, paired bootstrap accuracy intervals and host timing results in `reports/host.json`. The INT8 NumPy evaluation dequantizes weights to FP32: its timing is **not** an integer-inference benchmark.

## Run the Apple app

On a Mac with macOS 14+ and Xcode command-line tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
# Use the committed weights, or reproduce them first:
# OPENBLAS_NUM_THREADS=1 python -m edgelens.train
python -m edgelens.export_coreml
cd apple
swift run -c release EdgeLens
```

The exporter generates actual FP32, FP16 and INT8-weight Core ML ML Programs. Swift loads these without generated model classes. Select CPU-only or allow all compute units, run the benchmark and export its JSON report. Results include held-out accuracy, p50/p95 latency, model load time, sampled process memory, device model, OS, thermal state and dataset fingerprint. Package sizes are recorded separately in `reports/coreml_packages.json`.

The app is a macOS SwiftUI executable. An iPhone app and signed App Store distribution are outside this initial implementation.

## Protocol

- Bundled UCI optical digit data: 1,797 images, 64 grayscale features, ten classes. Inputs divided by 16 identically in Python and Swift.
- Stratified fixed splits: 20% test, then 20% of the remaining data for validation. Architecture and training settings are fixed; no test-based selection.
- Two-layer 64-hidden-unit ReLU MLP; symmetric per-output-channel INT8 weight simulation. Biases and scales remain floating point.
- A validation gate permits at most one percentage point of simulated accuracy loss. Core ML compression uses its own implementation and must be evaluated separately.
- Device runs use 50 warmups and 500 sequential timed predictions; accuracy evaluated once over every held-out sample. Compilation and input preparation excluded; model loading reported separately.
- Memory is sampled whole-process RSS, including the app and prepared inputs. It is neither isolated model memory nor a guaranteed peak. Fixed variant order can bias results through thermal state and allocator caching.

For a defensible application demo, run each variant in a fresh process, repeat at least five times, alternate order, keep thermal conditions stable, compare CPU-only with all-compute policy, and use Xcode Instruments to investigate allocations and actual compute placement. Allowed compute units do not prove Neural Engine execution. Package compression does not guarantee faster inference.

## Evidence and limitations

See [host results](reports/host.json), [research notes](reports/README.md), [training code](edgelens/train.py), [Core ML exporter](edgelens/export_coreml.py) and [Swift app](apple/Sources/EdgeLens/EdgeLens.swift).

The dataset is small and recognises preprocessed 8x8 digits, not arbitrary phone photos. Random stratified splits do not establish writer-independent generalisation. The paired bootstrap conditions on one trained model and one test sample; it does not include training-seed uncertainty. No energy/battery savings are claimed. A useful next experiment is a larger image dataset with writer/source-disjoint splits and hardware measurements.

## References and licensing

Code: MIT. Dataset: scikit-learn's bundled copy of UCI Optical Recognition of Handwritten Digits, E. Alpaydin and C. Kaynak; consult the [UCI source](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) and [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) for provenance and applicable data terms. Model weights and test samples derive from that dataset.

Core ML implementation follows Apple's [weight quantization guide](https://github.com/apple/coremltools/blob/main/docs-guides/source/opt-quantization-api.md). Use Apple's [Xcode performance tools](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode) to validate hardware placement. The included macOS CI workflow has not yet run remotely.

The measured run reached the fixed 220-iteration training limit with a convergence warning. This is retained as a limitation; no additional training was selected using test performance.

