"""Backward-compatible NaN cleanup that quarantines instead of deleting."""

from pathlib import Path

import numpy as np

from .quarantine import quarantine_invalid_tiffs


def check_and_remove_nan_images(directory, rejected_dir=None, manifest_path=None):
    directory = Path(directory)
    return quarantine_invalid_tiffs(
        directory,
        Path(rejected_dir or directory.parent / "rejected"),
        Path(manifest_path or directory.parent / "manifest.jsonl"),
        lambda image: bool(np.isnan(image).any()),
        "nan",
    )
