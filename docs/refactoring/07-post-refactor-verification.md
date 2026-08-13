# Post-refactor adversarial verification

## 1. Verification scope

This pass compared the working tree with a separate, read-only checkout of
commit `6d7ee247f4dd6d2d2e83e83227605c83c0cfd528`. The active directory was never
reset, cleaned, or checked out. Generated caches, `.venv`, `.tmp`,
`.pytest_cache`, and `__pycache__` were excluded.

The verification followed production entries rather than treating a new helper
as proof. It exercised CLI parsing, `ProcessedForestDataset`, all model-input
paths in `AnomalyDetectionPipeline`, preprocessing orchestration, the vector
evaluation CLI, checkpoint round trips, the training entry, and Optuna
objectives. No historical metric was used as a test oracle and no model
ablation was enabled.

## 2. Change inventory and original fixes verified

| Class | Original bug | New implementation | Production call path | Regression proof |
|---|---|---|---|---|
| DEFINITE FIX | Dataset normalized but temporal, histogram, and large-area inference could use raw values | `SARTransform` owns shape/channel validation, CHW conversion, float32, min/max, and clamp | `train.py -> ProcessedForestDataLoader -> ProcessedForestDataset`; `test_model -> AnomalyDetectionPipeline -> single/five/histogram/large-area` | The same georeferenced TIFF reaches a capture model through every entry and is equal with `torch.testing.assert_close`; restoring raw inference would fail |
| DEFINITE FIX | Prediction extent could define the metric universe and erase FN | immutable `EvaluationGrid`, rasterization after grid creation, file entry with CRS reprojection | `scripts/evaluate.py -> evaluate_vector_files -> evaluate_geometries -> evaluate_arrays` | empty prediction, one-of-two GT, smaller/outside prediction, boundary, missing/different CRS, empty GT, and CLI tests; the old notebook approach would fail the fixed-AOI cases |
| DEFINITE FIX | `sorted(VV)`, `sorted(VH)`, `zip` silently mismatched products | semantic `VVFileKey` one-to-one join | `fuse_and_split_images -> pair_vv_vh_files` | real full-scene and fused-tile names, ordering, duplicate VV/VH, missing, region/tile/date/product mismatch, ambiguity; old zip behavior would fail |
| DEFINITE FIX | cleanup deleted data and silently continued | quarantine/rejected records and immutable raw fusion inputs | legacy cleanup wrappers and `fuse_and_split_images` | corrupt, NaN, all-zero, shape, pairing, write, and manifest-failure injection verify filesystem/status agreement |
| DEFINITE FIX | direct writes and `exists()` accepted partial files | validate, write `.tmp`, close, validate, `os.replace` | fused/tile production writer and checkpoint writer | absent/existing target, write/validation/replace failures, temp cleanup, and truncated target rewrite; old direct write would fail |
| DEFINITE FIX | large-area errors could still look successful | `InferenceRunReport` with counts, fatal errors, detector metadata | `test_model -> generate_large_change_map` | 100/100, 97/100, zero processed, missing, corrupt/model/vectorization/merge failure, and pre-existing output truth table |
| DEFINITE FIX | `--test` could not be disabled and AE latent option was misleading | explicit `--train`/`--test`; embedding documented VAE-only | `train.py -> parse_arguments` | action and invalid DataLoader contract tests plus `train.py --help` |
| DEFINITE FIX | bare and structured checkpoints had incompatible contracts | versioned loader with bare-state fallback and recovery state | `test_model -> load_checkpoint`; training `EarlyStopping`/epoch/last checkpoints | real AE forward for both formats and optimizer/scheduler/EarlyStopping/epoch/config restoration |
| DEFINITE FIX | Optuna returned last validation while another metric selected the checkpoint | exact raw best-validation selection plus runtime equality assertion | `train_model -> objective -> wrapper.test -> EarlyStopping` | a sub-delta improvement must update checkpoint; two isolated lightweight trials and coherent `best_trial.json` |
| DEFINITE FIX | existing output was trusted by path only | `validate_geotiff` before reuse | preprocessing loop | zero-byte existing tile is rewritten and validated |
| REFACTOR | duplicate AE/VAE/DataLoader/writer lifecycle | selected wrapper only; anomaly pipeline no longer creates an unused writer | `train.main` | lifecycle test asserts one data object, one selected wrapper, one writer close |
| REFACTOR | mutable/shared run outputs and weak provenance | run/trial directories and resolved artifact paths | `create_run_context`, Optuna objective | two-trial isolation and resolved config tests; parent Git SHA is explicitly rejected |
| EXPERIMENTAL | activation/skip/attention alternatives and inductive detector | configurable variants/interfaces | opt-in only | default-contract test proves 512, tanh, p4+p3, unscaled attention, summed SSE remain default |
| TEST/DOCUMENTATION | no reliable automated safety net | CPU synthetic tests, CI, refactoring docs | CI/local validation | 26 pre-pass tests became 89 after adversarial coverage |

## 3. Newly discovered regressions and fixes

The following correctness regressions were found and fixed in this verification
pass:

- **P1: preprocessing/inference filename disconnect.** Default preprocessing
  emitted `tile_...`, while large-area inference only matched historical
  `622_975..._fused.tif`. Defaults now preserve the production filename
  contract.
- **P1: fixed-grid evaluation was infrastructure-only.** No production entry
  called it. `scripts/evaluate.py` now resolves a declared protocol, reads and
  reprojects vector files, and writes metrics plus provenance.
- **P1: realistic fused-tile semantic keys failed.** `256_512` was parsed as a
  second region and adjacent `VV_VH` was not detected as ambiguous. Region is
  anchored to the product prefix and polarization tokens use non-consuming
  boundaries.
- **P1: manifest/filesystem states could diverge.** Quarantine rolls back a move
  if the manifest cannot commit; corrupt inputs and move/write failures receive
  the correct FAILED stage; reject markers are atomic and removed on manifest
  failure.
- **P1: fused/tile write failures could be absent from, or mislabeled in, the
  manifest.** Both are now `write_error:<type>`; accepted outputs use SUCCESS.
- **P1: vectorization/merge failures and duplicate tile IDs could be reported
  as success or silently overwrite.** They now produce failed tile or fatal run
  state. Missing CRS fails explicitly.
- **P1: Optuna objective and checkpoint could differ for improvements smaller
  than EarlyStopping `delta`.** `best_validation` uses the exact raw minimum,
  stores the selected epoch, and the objective asserts equality before return.
- **P1: structured recovery did not restore EarlyStopping.** Loader recovery now
  restores its state in addition to model, optimizer, scheduler, epoch, and
  config.
- **P1: `config.json` was written before run paths were resolved and Git SHA
  could come from a parent repository.** Resolved artifact paths are persisted;
  a parent repository SHA is no longer accepted.
- **P1: missing inference checkpoint called `sys.exit()` without a failure
  code.** It now raises `FileNotFoundError`.
- **P1: DataLoader values could be invalid or silently ignored after automatic
  CPU worker resolution.** Negative workers, non-positive prefetch, persistent
  workers with zero workers, and non-blocking without pinned memory fail fast.
- **REFACTOR: duplicate unused anomaly `SummaryWriter`.** Removed; the selected
  model wrapper remains the single TensorBoard writer owner.

## 4. Test quality findings

| Test/area | Claim protected | Fails against old behavior? | Production wiring? | Remaining adversarial gap |
|---|---|---:|---:|---|
| transform production wiring | every actual input entry shares tensor semantics | Yes | Yes | real very-large tiled dataset and CUDA transfer |
| evaluation CLI adversarial set | prediction cannot select AOI | Yes | Yes | real forest-mask nodata conventions and antimeridian/geographic AOIs |
| semantic pairing | names, not ordering/count, determine joins | Yes | Yes | additional upstream provider naming families not present in repository evidence |
| atomic/preprocessing fault injection | final artifacts and statuses remain credible on failure | Yes | Yes | power loss between filesystem operations and network filesystems |
| inference completeness | artifact existence cannot override incomplete work | Yes | Yes | multi-thousand tile memory/runtime behavior |
| checkpoint forward/recovery | old inference and new recovery formats load usable state | Yes | Yes | no submitted historical `best_model.pth` was available byte-for-byte |
| lifecycle | only selected wrapper/data/writer is created | Yes | Yes | full training allocation was not profiled on GPU |
| Optuna isolation/metric | args, paths, metrics, checkpoint, and best-trial reference agree | Yes | Yes | synthetic wrappers replace expensive full AE trials |
| DataLoader matrix | legal CPU worker/pin/persistent combinations run | Yes | Yes | Windows `spawn` was exercised locally; CUDA pinned-transfer overlap was not |
| legacy model default | experimental variants cannot drift into default | N/A (future drift guard) | constructor/CLI | numerical parity with an unavailable historical checkpoint |

The original 26 tests mostly proved new helpers and happy paths. In particular,
they did not prove the preprocessing filename contract, evaluation CLI wiring,
all inference paths, failed-write cleanup, realistic filenames, complete
training recovery, or Optuna metric identity. Those gaps now have production
entry or failure-injection tests.

## 5. Production-wiring findings

Current model-input call graph:

```text
train / validation / test
  -> ProcessedForestDataset.__getitem__
  -> SARTransform.read
  -> selected AE/VAE

single-image inference
  -> test_loader.dataset / test_loader
  -> ProcessedForestDataset.__getitem__
  -> SARTransform.read
  -> model

five-image temporal and clustering
  -> _compute_all_pixel_losses
  -> _load_and_preprocess_image
  -> shared SARTransform
  -> model

histogram
  -> shared SARTransform.read
  -> model

large-area inference
  -> shared SARTransform.read
  -> model
  -> completeness report
```

No model-input path found in `pipeline/`, `scripts/`, or `train.py` performs an
independent normalization. The unrelated percentile-analysis helper reads TIFFs
but does not create model input.

Object lifecycle is:

```text
parse args -> seed -> create run -> one data owner -> selected wrapper
  -> its model/optimizer/scheduler/EarlyStopping/SummaryWriter
  -> train and/or test -> close writer
```

## 6. Compatibility findings

- Synthetic historical bare AE state dict: loads, enters `eval()`, and forwards
  `1x2x256x256` to the same output shape.
- Structured checkpoint: model, optimizer, scheduler, EarlyStopping, epoch,
  resolved config, and preprocessing metadata restore in CPU tests.
- A submitted historical `best_model.pth` is absent, so byte-for-byte production
  checkpoint compatibility remains partially verified.
- Legacy model defaults are 512 latent features, `legacy_tanh`, p4+p3 skips,
  unscaled attention, and summed SSE. Experimental variants remain opt-in.
- Transductive GMM result metadata states detector type, TRANS-DUCTIVE protocol,
  test fit split, and fit scope. Inductive detector interfaces state validation
  fit/frozen scope but are not enabled by default.

## 7. CI readiness

The workflow uses a fresh Ubuntu checkout, Python 3.11, CPU PyTorch, explicit
geospatial/test dependencies, compileall, and pytest. It contains no local
`.venv`, Windows absolute path, private data, Sentinel download, or GPU step.
The equivalent compile and test steps pass locally. No claim is made that a
remote GitHub Actions run has completed.

## 8. Remaining risks and unverified items

- **Resolved during repository finalization:** the verification input tree
  lacked the license, final PDF, and 143 of 146 files under `report/`. The final
  Git synchronization used remote `main` as the immutable source for historical
  material and restored/preserved the complete report, annotation, figure, GIS,
  notebook, PDF, and LICENSE set without regenerating or editing it.
- Full Sentinel-1 preprocessing, pyroSAR/CuSum, real fixed-AOI re-evaluation,
  full AE/VAE training, true Optuna training, real historical checkpoint load,
  CUDA/non-blocking overlap, GPU memory, and throughput require external data or
  hardware.
- The historical `cusum_preprocessing.ipynb` still contains `/home/yifan/...`
  paths. Its parsed notebook JSON is identical to the baseline and it was not
  modified in order to preserve historical material; it is not wired into the
  tested CLI pipeline.
- The single-image KMeans and synthetic GMM tests emit degenerate-cluster
  warnings for identity reconstructions. Production failure status remains
  correct, but detector quality cannot be established from synthetic data.
- “Training recovery” is verified at loader/state level; there is still no
  end-user `--resume` training command. It should not be described as a complete
  interrupted-run resume workflow.
- Manifest append uses flush/fsync and rolls back caught write failures. It is
  not a transactional database and cannot promise crash-atomic multi-artifact
  commits after sudden power loss.

Verification-time recommendation was **NEEDS MORE FIXES** because the supplied
working directory was incomplete. That repository-integrity blocker was later
resolved by the final remote-preservation sync. The remaining limitations are
the unavailable real Sentinel-1 data, historical checkpoint, GPU validation,
and end-to-end reproduction inputs. No model architecture redesign or benchmark
was performed.
