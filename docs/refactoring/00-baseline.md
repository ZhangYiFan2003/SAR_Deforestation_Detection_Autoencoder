# Refactoring Baseline

## Scope and source baseline

This refactoring starts from the repository snapshot named
`SAR_Deforestation_Detection_Autoencoder-6d7ee247f4dd6d2d2e83e83227605c83c0cfd528`.
The prior audit identifies GitHub `main` commit
`6d7ee247f4dd6d2d2e83e83227605c83c0cfd528` as the intended source baseline.

The local snapshot is not an independent Git repository: it is contained in an
uncommitted outer repository rooted at `D:\Code`. Consequently, the local
checkout cannot independently prove the commit SHA and logical batch commits
cannot be created safely without changing unrelated repository state. The
directory name and audited documents are the available provenance evidence.

## Current architecture

The project is a single-machine research/batch workflow:

1. `pipeline/datasets/preprocessing/split_data.py` pairs VV and VH GeoTIFFs,
   validates basic geospatial alignment, fuses them into two bands, and writes
   non-overlapping full-size tiles.
2. `ProcessedForestDataset` reads materialized TIFF tiles and normalizes values
   with the historical `[-15, -3]` min/max range.
3. `train.py` constructs an AE or VAE wrapper. Each wrapper owns its model,
   optimizer, scheduler, early stopping object, data loaders, and TensorBoard
   writer.
4. The selected model is trained as a normal-forest reconstruction model.
5. `pipeline/anomaly_detection` calculates pixel reconstruction error and uses
   KMeans or a two-component GMM plus connected-component filtering for single
   image, five-image, and two-date large-area workflows.
6. Large-area results are vectorized to Shapefile. Historical notebook-based
   evaluation rasterizes GIS artifacts and reports overlap metrics.

This architecture should remain a local research workflow. A service platform,
database, queue, or distributed scheduler is outside the intended scope.

## Currently runnable commands

Source syntax compilation succeeds with the available Python 3.8 interpreter:

```powershell
D:\Python\Python38\python.exe -m compileall -q config pipeline train.py
```

The intended commands in the original project are:

```text
python train.py --help
python train.py --train
python train.py --test
```

They are not currently runnable in this environment because neither the system
Python 3.8 environment nor the bundled Python 3.12 runtime contains PyTorch and
the geospatial/test dependency set. The system `python` launcher also points at
an unavailable Windows Store alias in the sandbox session.

## Known blockers

- No submitted Sentinel-1 dataset or dataset manifest is available, so the
  historical full-data workflow and report metrics cannot be reproduced.
- No historical `best_model.pth` is present in this repository snapshot.
- The local snapshot has no usable independent Git history.
- Project runtime dependencies are not installed in the available interpreters.
- The original requirements pin CUDA-specific PyTorch wheels without declaring
  the required package index, which is unsuitable for minimal CPU CI.
- The complete upstream CuSum/pyroSAR preprocessing is not contained in this
  repository.

## Known correctness bugs

- Training/validation/test use `[-15, -3]` normalization, but five-image and
  large-area inference bypass or change that transform (train-serving skew).
- Evaluation notebook geometry is derived from prediction extent, so ground
  truth outside prediction extent can be removed before FN counting.
- VV and VH are paired by independently sorting two lists and using positional
  `zip`; equal counts do not guarantee semantic pairing.
- NaN and all-zero cleanup directly deletes inputs and silently continues on
  decode failures.
- Fused and tile outputs are written directly to final paths. Existing paths are
  trusted without validating readability, shape, channels, or provenance.
- Large-area inference can skip failed tiles and still return a success-like
  result without a completeness report.
- `--test` is `store_true` with `default=True`, so it cannot be disabled normally.
- `train.py` instantiates both AE and VAE plus another data-loader wrapper,
  duplicating models, optimizers, loaders, writers, and possible GPU allocation.
- Optuna trials share checkpoint/log paths and mutable arguments. The returned
  final validation loss is not guaranteed to identify the checkpoint selected
  by early stopping.
- Checkpoints contain only a state dict, without preprocessing, optimizer,
  scheduler, seed, config, epoch, or best-validation metadata.

## Behavior to preserve

- Historical AE preprocessing defaults: two channels, float32 tensor,
  min/max normalization using `min=-15`, `max=-3`, and no clipping by default.
- AE latent size 512 for legacy checkpoints and the default legacy architecture.
- Summed squared reconstruction loss and current anomaly-score definition unless
  an explicitly named experimental variant is selected.
- Legacy decoder `tanh`, p4+p3 FPN skips, unscaled attention, and transductive GMM
  remain available and default for historical comparisons.
- Spatially materialized `2x256x256` TIFF input and CPU-capable batch workflow.
- Historical reports and their metrics remain unchanged and are treated as
  report-only evidence, not regression-test constants.

## Intentionally experimental behavior

The following are not definite fixes and must only be introduced as named,
non-default variants with recorded configuration: sigmoid/linear decoder output,
alternative FPN skip sets, removal/use of p2, scaled attention, and an inductive
validation-fitted anomaly detector. No performance claim is valid until these
variants are run on the same split, preprocessing, seed, budget, and fixed
evaluation grid.
