# Migration Notes

## Legacy behavior retained

- AE latent size 512, summed reconstruction SSE, `tanh`, p4+p3 skips, unscaled
  attention, and transductive GMM remain the default baseline.
- Preprocessing remains `[-15, -3]` min/max, float32, two channels, without
  clipping by default.
- Historical documents and metrics are unchanged.
- Bare historical model state dicts remain loadable.

## New behavior

- Training and inference share `SARTransform`; checkpoint preprocessing is
  checked when metadata exists.
- Evaluation requires a fixed grid independent of predictions.
- VV/VH pairing is semantic and fail-fast.
- Invalid data is quarantined/manifested; artifacts commit atomically.
- Runs/trials use isolated artifact directories and structured checkpoints.
- Large-area inference returns explicit completeness status.

## Breaking changes

- `--train` and/or `--test` must be explicit. Testing without training requires
  `--checkpoint`.
- Train/validation/test directories are CLI-configured; personal
  Executable Python/CLI paths no longer depend on `/home/yifan/...`. The
  historical CuSum notebook remains byte-semantically unchanged and still
  contains its original workstation paths; it is not part of the automated
  production call path.
- New checkpoints are dictionaries rather than bare state dicts. Use the
  compatibility loader instead of passing them directly to
  `model.load_state_dict`.
- VV/VH names must contain one region, acquisition, and polarization token.
  Ambiguous legacy names must be renamed or mapped explicitly before ingestion.

## Commands

```powershell
python train.py --help
python train.py --train --train-dir data/train/processed `
  --validation-dir data/validation/processed --test-dir data/test/processed
python train.py --test --checkpoint runs/<run_id>/checkpoints/best.ckpt `
  --train-dir data/train/processed --validation-dir data/validation/processed `
  --test-dir data/test/processed
python -m pytest -q
python scripts/run_ablation.py --dataset-manifest <manifest.jsonl> `
  --evaluation-config <evaluation.json>
```

No full-data metric has been recomputed in this refactoring because the dataset
and historical checkpoint are absent.
