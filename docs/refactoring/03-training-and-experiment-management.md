# Training and Experiment Management

## Lifecycle and configuration

The CLI creates one `ProcessedForestDataLoader` and only the selected AE/VAE
wrapper. Optuna does not pre-create an unused wrapper. The flow is now:

```text
resolved args -> run context -> data -> selected model -> train -> evaluate
```

`--test` defaults to false and requires an explicit checkpoint when it is run
without training. `--embedding-size` is documented as VAE-only; legacy AE
remains fixed at 512 for checkpoint compatibility. Previously ignored suffix,
tile-size, and dataset-path arguments now control their consumers.

## Run isolation and metadata

Each normal run creates:

```text
runs/<run_id>/
  config.json
  metadata.json
  checkpoints/{best,last,epoch_*.ckpt}
  logs/
  metrics.json
```

Optuna trial directories are `trial_000`, `trial_001`, etc. `best_trial.json`
references the trial ID, params, metric, and checkpoint. Metadata records UTC
time, available Git SHA, model, dataset manifest, preprocessing, resolved args,
Python/PyTorch/device, seed, and deterministic-mode choice.

Python, NumPy, PyTorch, and CUDA seeds are set. Deterministic algorithms are
opt-in because they may reduce performance; the code does not claim default
bitwise reproducibility.

Normal-run metrics distinguish the raw best validation value from the validation
value that selected the checkpoint. This matters when the legacy moving-average
strategy is intentionally retained. Optuna overrides selection to raw
`best_validation`, so its objective and checkpoint are identical by contract.

## Checkpoints

Structured training checkpoints contain model, optimizer, scheduler, epoch,
best validation, resolved/preprocessing config, seeds, and EarlyStopping state.
Atomic `last.ckpt` supports recovery semantics. The loader also accepts the
historical `best_model.pth` bare state dict and labels it
`legacy_state_dict`.

## Performance and logging

DataLoader worker, pinned memory, prefetch, persistent worker, and non-blocking
copy controls are explicit. `--profile` records data wait, H2D, forward,
backward, optimizer time, and samples/sec. These are measurements, not claims of
improvement. TensorBoard remains, while standard logging includes run/stage,
epoch, objective, duration, file, and error type where relevant.
