# Model Ablation Plan

These changes are EXPERIMENTAL, not correctness fixes. Defaults preserve the
historical baseline:

| Dimension | Legacy default | Optional variants |
|---|---|---|
| Output activation | `legacy_tanh` | `sigmoid`, `linear` |
| FPN decoder skips | `p4+p3` | `none`, `p4` |
| Attention score | `legacy` unscaled | `scaled` by sqrt(query channels) |
| Detector | `transductive_gmm` | `validation_gmm`, `fixed_threshold` |

The unused p2 branch remains in the legacy encoder so old state dicts and compute
graphs are comparable. `none`/`p4` change decoder use but not the stored legacy
parameters. A future clean architecture may remove p2 only under a new explicit
variant and checkpoint family.

`ReconstructionErrorDetector` exposes the detector protocols programmatically.
They are not yet a training CLI switch because this snapshot has no calibration
dataset contract; exposing an unwired option would be misleading.

`scripts/run_ablation.py` generates a plan with identical seed, epoch budget,
dataset manifest, preprocessing, and evaluation config. Required observations
are validation reconstruction objective, normal/anomaly error distributions,
P/R/F1/IoU, parameter count, and runtime. The runner produces no synthetic
benchmark values. A real experiment must also freeze the same spatial split and
fixed EvaluationGrid.

Transductive and inductive results must be reported separately. A validation
GMM is fit on calibration errors and frozen before test; a fixed threshold is
also inductive. A transductive GMM fits the test-time error distribution and
must not be described as an untouched frozen detector.
