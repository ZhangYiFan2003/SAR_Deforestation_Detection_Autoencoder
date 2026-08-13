# Test gap analysis

## Closed in the adversarial pass

The original 26-test suite had material behavior gaps. This pass closed the
following with production wiring or fault injection:

- Dataset parity now covers single-image, five-image, histogram, and large-area
  model calls, not only `_load_and_preprocess_image`.
- Evaluation now has a CLI/file/CRS entry test and all requested fixed-AOI empty,
  outside, partial, boundary, and zero-denominator cases.
- Pairing uses repository-real full and fused tile patterns and adversarial
  duplicate, ambiguity, suffix, date, region, tile, and order cases.
- Atomic output covers existing targets, write/validation/replace exceptions,
  temp cleanup, truncated outputs, and production rewrite.
- Manifest/raw tests inject corrupt TIFF, NaN, all-zero, shape mismatch, missing
  counterpart, fused/tile write failure, and manifest failure.
- Inference completeness covers missing/corrupt/model/vectorization/merge and
  duplicate-tile failures plus SUCCESS/PARTIAL/FAILED truth tables.
- Checkpoint tests perform AE forward in both formats and restore recovery state.
- Lifecycle, resolved run paths, parent-Git rejection, two Optuna trials,
  best-trial coherence, raw metric/checkpoint identity, and DataLoader matrices
  are now guarded.

## Still requiring external evidence

| Gap | Why synthetic CI is insufficient | Required evidence |
|---|---|---|
| historical checkpoint compatibility | no `best_model.pth` is supplied | load the actual file and run CPU/GPU forward |
| real evaluation metrics | historical GIS artifacts are preserved, but the original dataset/checkpoint and a fully declared evaluation protocol are unavailable | declare the fixed AOI and artifact roles, run `scripts/evaluate.py`, archive new metrics separately |
| naming universe | only formats evidenced by source/report references were tested | sample manifest from every upstream export job |
| full preprocessing | pyroSAR/Sentinel products are unavailable | immutable small raw fixture with expected geospatial outputs |
| GPU behavior | CPU cannot establish asynchronous overlap or memory use | pinned-memory CUDA profile with synchronization points |
| full Optuna | synthetic trials verify contracts, not learning behavior | two minimal real-data trials under equal seed/budget |
| crash durability | injected Python exceptions do not simulate power loss | process-kill/power-failure test on the target filesystem |
| detector effectiveness | identity/synthetic errors do not represent forest/anomaly distributions | frozen calibration/test split and fixed-grid evaluation |

No coverage percentage is claimed: the tests target failure credibility and
production wiring rather than maximizing a line-coverage number.
