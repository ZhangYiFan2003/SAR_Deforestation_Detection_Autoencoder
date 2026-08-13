"""Recoverable validation/quarantine for already materialized TIFF files."""

import os
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile

from .manifest import ManifestRecord, ManifestWriter


def quarantine_invalid_tiffs(
    processed_dir,
    rejected_dir,
    manifest_path,
    predicate: Callable[[np.ndarray], bool],
    reason: str,
) -> int:
    processed_dir, rejected_dir = Path(processed_dir), Path(rejected_dir)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(Path(manifest_path))
    rejected = 0
    for source in sorted(processed_dir.glob("*.tif")):
        try:
            image = tifffile.imread(source)
        except Exception as exc:
            manifest.append(
                ManifestRecord(
                    sample_id=source.stem,
                    source_vv=str(source),
                    source_vh="",
                    output_file=None,
                    status="FAILED",
                    reject_reason=f"decode_error:{type(exc).__name__}",
                )
            )
            continue
        if not predicate(image):
            continue

        target = rejected_dir / source.name
        if target.exists():
            manifest.append(
                ManifestRecord(
                    sample_id=source.stem,
                    source_vv=str(source),
                    source_vh="",
                    output_file=None,
                    status="FAILED",
                    reject_reason="quarantine_error:FileExistsError",
                    shape=tuple(image.shape),
                )
            )
            continue
        try:
            os.replace(source, target)
            try:
                manifest.append(
                    ManifestRecord(
                        sample_id=source.stem,
                        source_vv=str(source),
                        source_vh="",
                        output_file=str(target),
                        status="REJECTED",
                        reject_reason=reason,
                        shape=tuple(image.shape),
                    )
                )
            except Exception:
                # Restore the pre-call filesystem state if the audit record cannot commit.
                os.replace(target, source)
                raise
            rejected += 1
        except Exception as exc:
            # A move failure leaves the source in place; record the real stage.
            if source.exists():
                manifest.append(
                    ManifestRecord(
                        sample_id=source.stem,
                        source_vv=str(source),
                        source_vh="",
                        output_file=None,
                        status="FAILED",
                        reject_reason=f"quarantine_error:{type(exc).__name__}",
                        shape=tuple(image.shape),
                    )
                )
            else:
                # Never hide a failure that left the artifact in an unknown
                # location and could not be represented in the manifest.
                raise
    return rejected
