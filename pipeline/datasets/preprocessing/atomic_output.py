"""Validated atomic GeoTIFF output helpers."""

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import rasterio


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Commit a small artifact without exposing a partial final path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def validate_geotiff(
    path: Path,
    *,
    expected_shape: Optional[Tuple[int, int]] = None,
    expected_channels: Optional[int] = None,
) -> bool:
    try:
        with rasterio.open(path) as source:
            if expected_shape and (source.height, source.width) != expected_shape:
                return False
            if expected_channels and source.count != expected_channels:
                return False
            sample = source.read(1, window=((0, min(1, source.height)), (0, min(1, source.width))))
            return sample.size == 1
    except Exception:
        return False


def atomic_write_geotiff(path: Path, data: np.ndarray, metadata: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    output_meta = dict(metadata)
    output_meta["driver"] = "GTiff"
    try:
        with rasterio.open(temporary, "w", **output_meta) as destination:
            destination.write(data)
        if not validate_geotiff(
            temporary,
            expected_shape=(data.shape[-2], data.shape[-1]),
            expected_channels=data.shape[0],
        ):
            raise IOError(f"Atomic GeoTIFF validation failed for {temporary}")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
