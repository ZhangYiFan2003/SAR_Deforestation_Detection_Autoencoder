# Correctness Fixes

## DEFINITE FIX: one preprocessing contract

`SARTransformConfig` and `SARTransform` are now the only implementation for
shape/channel validation, float32 conversion, `[-15, -3]` normalization, and
optional clipping. Dataset, single-image, five-image, histogram, and large-area
inference paths reuse it. Historical defaults remain two channels, float32,
`min=-15`, `max=-3`, and `clamp=false`.

Structured checkpoints embed the preprocessing config. Inference rejects a
checkpoint whose embedded preprocessing differs from the resolved inference
config. Historical bare state-dict checkpoints still load, but cannot supply
metadata; the configured legacy defaults are therefore used explicitly.

## DEFINITE FIX: fixed evaluation universe

The new `pipeline/evaluation` package requires an `EvaluationGrid` defined from
CRS, AOI bounds, resolution, transform, dimensions, and optional valid mask.
Ground truth, prediction, and forest masks are rasterized/aligned only after the
grid exists. Prediction extent can no longer define the evaluation denominator.

Metrics include TP, FP, FN, TN, precision, recall, F1, and IoU. Evaluation JSON
can retain the grid/config and all input artifact paths. Historical report
metrics were not copied into tests or changed.

## DEFINITE FIX: semantic VV/VH join

`VVFileKey` parses region, acquisition date/time, tile identifier, and the
polarization-independent product name. Pairing is a one-to-one key join and
fails on missing counterparts, duplicate keys, ambiguous names, wrong
polarization, or mismatched dates. Equal directory counts are not accepted as
evidence of correct pairing.

## DEFINITE FIX: objective/checkpoint agreement

Optuna uses `best_validation_objective` for trial return values, recorded
metrics, and `best.ckpt` selection. The VAE now returns the total validation
objective that EarlyStopping monitors, rather than returning reconstruction loss
while selecting on total loss.

For non-Optuna legacy runs, moving-average selection remains the default for
historical comparison. `--selection-strategy best_validation` enables the new
raw-objective contract explicitly.
