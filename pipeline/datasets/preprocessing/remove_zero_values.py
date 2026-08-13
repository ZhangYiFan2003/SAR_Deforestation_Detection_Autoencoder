"""Backward-compatible zero cleanup that quarantines instead of deleting."""

from pathlib import Path

import numpy as np

from .quarantine import quarantine_invalid_tiffs


def remove_invalid_tif_files(tiff_dir, rejected_dir=None, manifest_path=None):
    tiff_dir = Path(tiff_dir)
    return quarantine_invalid_tiffs(
        tiff_dir,
        Path(rejected_dir or tiff_dir.parent / "rejected"),
        Path(manifest_path or tiff_dir.parent / "manifest.jsonl"),
        lambda image: bool(np.min(image) == 0 and np.max(image) == 0),
        "all_zero",
    )
