# Data Pipeline Hardening

## Raw, processed, and rejected boundaries

The preprocessing code treats VV/VH source paths as immutable. Invalid pairs or
images are represented under `rejected/` and in `manifest.jsonl`; source VV/VH
files are not removed. Legacy cleanup functions now move already-materialized
invalid TIFFs to a recoverable quarantine instead of deleting them.

Each manifest record contains sample ID, source VV/VH, output path, status,
reject reason, preprocessing version, shape, UTC timestamp, and optional SHA-256
checksum. Decode failures, NaN, all-zero, shape, and alignment failures are
explicit statuses/reasons.

## Atomic artifacts and idempotency

Fused images, tiles, and torch checkpoints are written to a same-directory
`.tmp`, closed/flushed, validated, and committed with `os.replace`. Existing
GeoTIFFs are reused only after readability, expected dimensions, and expected
channel count are verified.

## Completeness

Large-area inference returns `InferenceRunReport`, containing expected,
processed, failed, missing-pair, and output tile counts plus failed file names.
Its status is `SUCCESS`, `PARTIAL`, or `FAILED`; a 97/100 run is no longer
indistinguishable from 100/100.
