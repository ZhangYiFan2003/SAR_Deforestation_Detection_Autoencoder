# SAR Deforestation Detection Autoencoder

## Project Overview

This research project detects micro-deforestation anomalies from Sentinel-1
Synthetic Aperture Radar (SAR) imagery with a convolutional autoencoder. Each
model sample contains aligned VV and VH backscatter channels. The core workflow
is:

```text
VV/VH data preparation
  -> autoencoder training
  -> reconstruction error
  -> clustering and spatial post-processing
  -> GIS vector output and fixed-grid evaluation
```

The repository is designed for a single-machine research and batch workflow. It
does not include the original Sentinel-1 dataset and is not intended to be a
production microservice or full MLOps platform.

## Historical Internship Project

The original prototype was developed during an IRD internship from September
2024 to February 2025. The preserved `report/` directory and final internship
PDF contain historical figures, annotations, GIS artifacts, notebooks, and
project context. They are retained as historical material and are not rewritten
as though the later engineering work existed during the internship.

## Current Architecture

- `pipeline/transforms/`: one serializable `SARTransform` contract for channel
  validation, CHW conversion, float32 conversion, normalization, and optional
  clamping.
- `pipeline/datasets/`: immutable raw-input preprocessing, semantic VV/VH
  pairing, validated atomic GeoTIFF output, manifests, quarantine, datasets,
  and configurable DataLoaders.
- `pipeline/models/`: legacy-compatible AE/VAE wrappers and configurable
  experimental ablations. The default AE remains the historical 512-feature,
  tanh, p4+p3, unscaled-attention baseline.
- `pipeline/anomaly_detection/`: reconstruction-error analysis, explicit
  transductive/inductive detector metadata, GIS vectorization, and inference
  completeness reports.
- `pipeline/evaluation/`: prediction-independent `EvaluationGrid`, raster
  alignment, and TP/FP/FN/TN, precision, recall, F1, and IoU calculation.
- `pipeline/experiments/`: isolated run directories, resolved configuration,
  runtime metadata, checkpoints, logs, and metrics.
- `tests/`: CPU unit, integration, production-wiring, and adversarial regression
  tests using small synthetic GeoTIFF fixtures.

## 2026 Engineering Refactor

The early research prototype was audited again in 2026. This later refactor
focused on engineering correctness and reproducibility rather than changing the
historical experimental claims. It introduced:

- consistent training and inference preprocessing;
- a fixed evaluation universe that cannot be selected by prediction extent;
- semantic one-to-one VV/VH pairing instead of positional `zip`;
- immutable raw data, recoverable rejection, manifests, and atomic outputs;
- explicit SUCCESS/PARTIAL/FAILED inference completeness;
- isolated run and Optuna trial artifacts;
- structured recovery checkpoints with legacy bare-state-dict compatibility;
- deterministic seed controls, metadata, logging, lightweight profiling;
- CPU automated tests and GitHub Actions CI.

Experimental decoder activation, FPN skip, attention, and inductive detector
variants remain opt-in. They are not enabled by default and no performance
improvement is claimed without a real-data ablation.

## Historical Results

The final internship report recorded the following values:

| Metric | Historical report value |
|---|---:|
| Precision | 0.8284 |
| Recall | 0.6865 |
| F1 | 0.7508 |
| IoU | 0.6011 |

These are values recorded in the historical final internship report. They are
not a current verified benchmark. The repository does not contain all original
data and checkpoints required to reproduce these exact historical metrics, and
the corrected fixed-grid evaluation may produce different results when the
original artifacts become available.

## Installation

Python 3.11 is used by CI. A CPU-oriented development installation is:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-test.txt
```

For a CPU-only PyTorch wheel, install PyTorch from its CPU index before the
requirements command, as shown in `.github/workflows/ci.yml`.

Geospatial Python packages may require platform-specific binary wheels. The
versions in `requirements.txt` match the tested project environment; they were
not broadly upgraded as part of finalization.

## Usage

Inspect the actual training/testing contract:

```bash
python train.py --help
```

Training and testing require local directories of aligned two-channel
GeoTIFFs. Testing an existing model requires an explicit checkpoint:

```bash
python train.py --test --checkpoint path/to/checkpoint.ckpt --no-cuda
```

The fixed-grid vector evaluation entry is:

```bash
python scripts/evaluate.py --help
python scripts/evaluate.py --config evaluation.json --output metrics.json
```

`evaluation.json` declares the CRS, AOI bounds, resolution, and paths to ground
truth, prediction, and an optional forest mask. See
`docs/refactoring/07-post-refactor-verification.md` and the evaluation tests for
the exact contract.

## Testing

Run the complete CPU suite with:

```bash
python -m pytest -q
```

The repository includes unit, integration, production-wiring, and adversarial
regression tests. The exact test count may grow; the finalization pass completed
with 89 passing tests. CI also compiles the executable Python tree before
running the suite.

## Repository Layout

```text
config/                         CLI configuration
pipeline/                       data, models, training, inference, evaluation
scripts/                        evaluation and experiment entry points
tests/                          CPU automated tests
docs/interview-preparation/     audited project explanations
docs/refactoring/               refactor, migration, and verification records
report/                         preserved historical internship artifacts
.github/workflows/ci.yml        fresh-checkout CPU CI
train.py                        main train/test entry
```

## Repository Limitations

- The real Sentinel-1 training, validation, and test datasets are not included.
- The historical trained checkpoint may not be available.
- Upstream CuSum/pyroSAR preprocessing remains external to the tested CLI
  workflow; its historical notebook contains workstation-specific paths.
- Historical metrics cannot currently be reproduced end to end.
- CUDA throughput, pinned-memory overlap, and GPU memory behavior require
  hardware-specific profiling and are not claimed by the CPU tests.

The post-refactor evidence and remaining gaps are documented in
`docs/refactoring/07-post-refactor-verification.md` and
`docs/refactoring/08-test-gap-analysis.md`.
