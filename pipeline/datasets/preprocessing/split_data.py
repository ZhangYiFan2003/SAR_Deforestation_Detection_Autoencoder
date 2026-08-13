"""Fuse semantically paired VV/VH images and write validated tiles atomically."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from .atomic_output import atomic_write_bytes, atomic_write_geotiff, validate_geotiff
from .manifest import ManifestRecord, ManifestWriter
from .pairing import PairingError, pair_vv_vh_files


LOGGER = logging.getLogger(__name__)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject(
    rejected_dir: Path,
    manifest: ManifestWriter,
    *,
    sample_id: str,
    vv_path: Path,
    vh_path: Path,
    reason: str,
    status: str = "REJECTED",
    shape=None,
) -> None:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    marker = rejected_dir / f"{sample_id}.json"
    atomic_write_bytes(
        marker,
        json.dumps(
            {
                "sample_id": sample_id,
                "source_vv": str(vv_path),
                "source_vh": str(vh_path),
                "reason": reason,
                "raw_inputs_modified": False,
            },
            indent=2,
        ).encode("utf-8"),
    )
    try:
        manifest.append(
            ManifestRecord(
                sample_id=sample_id,
                source_vv=str(vv_path),
                source_vh=str(vh_path),
                output_file=None,
                status=status,
                reject_reason=reason,
                shape=shape,
            )
        )
    except Exception:
        marker.unlink(missing_ok=True)
        raise


def fuse_and_split_images(
    vv_dir,
    vh_dir,
    fused_dir,
    tiles_dir,
    tile_size=256,
    prefix_fused="",
    prefix_tile="",
    *,
    rejected_dir=None,
    manifest_path=None,
):
    """Process immutable raw inputs into reproducible fused/tiled outputs.

    Rejected samples are represented by recoverable reason records; raw inputs
    are never deleted or moved.
    """
    vv_dir, vh_dir = Path(vv_dir), Path(vh_dir)
    fused_dir, tiles_dir = Path(fused_dir), Path(tiles_dir)
    rejected_dir = Path(rejected_dir or tiles_dir.parent / "rejected")
    manifest = ManifestWriter(Path(manifest_path or tiles_dir.parent / "manifest.jsonl"))
    fused_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    vv_files = sorted(vv_dir.glob("*.tif"))
    vh_files = sorted(vh_dir.glob("*.tif"))
    try:
        pairs = pair_vv_vh_files(vv_files, vh_files)
    except PairingError as exc:
        manifest.append(
            ManifestRecord(
                sample_id="pairing",
                source_vv=str(vv_dir),
                source_vh=str(vh_dir),
                output_file=None,
                status="FAILED",
                reject_reason=f"pairing_error:{exc}",
            )
        )
        raise

    for key, vv_path, vh_path in pairs:
        sample_id = f"{key.region}_{key.acquisition}_{key.tile}"
        fused_filename = f"{prefix_fused + '_' if prefix_fused else ''}{vv_path.stem}.tif"
        fused_path = fused_dir / fused_filename
        stage = "decode"
        try:
            with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                if vv_src.crs != vh_src.crs or vv_src.transform != vh_src.transform:
                    _reject(
                        rejected_dir,
                        manifest,
                        sample_id=sample_id,
                        vv_path=vv_path,
                        vh_path=vh_path,
                        reason="alignment_error",
                    )
                    continue
                if (vv_src.width, vv_src.height) != (vh_src.width, vh_src.height):
                    _reject(
                        rejected_dir,
                        manifest,
                        sample_id=sample_id,
                        vv_path=vv_path,
                        vh_path=vh_path,
                        reason="shape_error",
                    )
                    continue
                vv_data, vh_data = vv_src.read(1), vh_src.read(1)
                fused_data = np.stack([vv_data, vh_data], axis=0)
                if np.isnan(fused_data).any():
                    _reject(
                        rejected_dir,
                        manifest,
                        sample_id=sample_id,
                        vv_path=vv_path,
                        vh_path=vh_path,
                        reason="nan",
                        shape=fused_data.shape,
                    )
                    continue
                if not np.any(fused_data):
                    _reject(
                        rejected_dir,
                        manifest,
                        sample_id=sample_id,
                        vv_path=vv_path,
                        vh_path=vh_path,
                        reason="all_zero",
                        shape=fused_data.shape,
                    )
                    continue
                fused_meta = vv_src.meta.copy()
                fused_meta.update(count=2, dtype=fused_data.dtype)
                stage = "write"
                if not validate_geotiff(
                    fused_path,
                    expected_shape=(vv_src.height, vv_src.width),
                    expected_channels=2,
                ):
                    atomic_write_geotiff(fused_path, fused_data, fused_meta)
        except Exception as exc:
            LOGGER.exception("Failed during %s stage for sample %s", stage, sample_id)
            _reject(
                rejected_dir,
                manifest,
                sample_id=sample_id,
                vv_path=vv_path,
                vh_path=vh_path,
                reason=f"{stage}_error:{type(exc).__name__}",
                status="FAILED",
            )
            continue

        with rasterio.open(fused_path) as fused_src:
            rows = fused_src.height // tile_size
            columns = fused_src.width // tile_size
            for tile_row in range(rows):
                for tile_col in range(columns):
                    row_off, col_off = tile_row * tile_size, tile_col * tile_size
                    window = Window(col_off, row_off, tile_size, tile_size)
                    tile_data = fused_src.read(window=window)
                    tile_name = (
                        f"{prefix_tile + '_' if prefix_tile else ''}"
                        f"{vv_path.stem}_{row_off}_{col_off}_fused.tif"
                    )
                    tile_path = tiles_dir / tile_name
                    tile_meta = fused_src.meta.copy()
                    tile_meta.update(
                        height=tile_size,
                        width=tile_size,
                        transform=window_transform(window, fused_src.transform),
                    )
                    tile_sample_id = f"{sample_id}_{row_off}_{col_off}"
                    try:
                        if not validate_geotiff(
                            tile_path,
                            expected_shape=(tile_size, tile_size),
                            expected_channels=2,
                        ):
                            atomic_write_geotiff(tile_path, tile_data, tile_meta)
                        manifest.append(
                            ManifestRecord(
                                sample_id=tile_sample_id,
                                source_vv=str(vv_path),
                                source_vh=str(vh_path),
                                output_file=str(tile_path),
                                status="SUCCESS",
                                reject_reason=None,
                                shape=tile_data.shape,
                                checksum=_checksum(tile_path),
                            )
                        )
                    except Exception as exc:
                        manifest.append(
                            ManifestRecord(
                                sample_id=tile_sample_id,
                                source_vv=str(vv_path),
                                source_vh=str(vh_path),
                                output_file=str(tile_path) if tile_path.exists() else None,
                                status="FAILED",
                                reject_reason=f"write_error:{type(exc).__name__}",
                                shape=tile_data.shape,
                            )
                        )
                        raise
    return Path(manifest.path)
