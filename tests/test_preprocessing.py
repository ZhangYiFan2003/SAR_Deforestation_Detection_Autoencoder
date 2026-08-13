import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from conftest import write_single_band
from pipeline.datasets.preprocessing import atomic_output
from pipeline.datasets.preprocessing.atomic_output import atomic_write_geotiff, validate_geotiff
from pipeline.datasets.preprocessing.manifest import ManifestWriter
from pipeline.datasets.preprocessing.quarantine import quarantine_invalid_tiffs
from pipeline.datasets.preprocessing import quarantine as quarantine_module
from pipeline.datasets.preprocessing import split_data as split_module
from pipeline.datasets.preprocessing.split_data import fuse_and_split_images


def test_atomic_output(tmp_path):
    path = tmp_path / "output.tif"
    data = np.ones((2, 4, 4), dtype=np.float32)
    metadata = {
        "driver": "GTiff",
        "width": 4,
        "height": 4,
        "count": 2,
        "dtype": "float32",
    }
    atomic_write_geotiff(path, data, metadata)
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    with rasterio.open(path) as source:
        assert source.read().shape == (2, 4, 4)


def _geotiff_metadata():
    return {"driver": "GTiff", "width": 4, "height": 4, "count": 2, "dtype": "float32"}


def test_atomic_output_replaces_existing_valid_target(tmp_path):
    path = tmp_path / "output.tif"
    atomic_write_geotiff(path, np.zeros((2, 4, 4), dtype=np.float32), _geotiff_metadata())
    atomic_write_geotiff(path, np.ones((2, 4, 4), dtype=np.float32), _geotiff_metadata())
    with rasterio.open(path) as source:
        assert np.all(source.read() == 1)
    assert not path.with_name(path.name + ".tmp").exists()


@pytest.mark.parametrize("failure_stage", ["write", "validation", "replace"])
def test_atomic_output_failure_preserves_existing_target_and_cleans_temp(
    tmp_path, monkeypatch, failure_stage
):
    path = tmp_path / "output.tif"
    old = np.full((2, 4, 4), 7, dtype=np.float32)
    atomic_write_geotiff(path, old, _geotiff_metadata())
    metadata = _geotiff_metadata()
    data = np.ones((2, 4, 4), dtype=np.float32)
    if failure_stage == "write":
        metadata["count"] = 1
    elif failure_stage == "validation":
        monkeypatch.setattr(atomic_output, "validate_geotiff", lambda *args, **kwargs: False)
    else:
        monkeypatch.setattr(atomic_output.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("replace")))
    with pytest.raises(Exception):
        atomic_write_geotiff(path, data, metadata)
    assert not path.with_name(path.name + ".tmp").exists()
    with rasterio.open(path) as source:
        np.testing.assert_array_equal(source.read(), old)


def test_truncated_existing_target_is_not_valid(tmp_path):
    path = tmp_path / "output.tif"
    path.write_bytes(b"")
    assert not validate_geotiff(path, expected_shape=(4, 4), expected_channels=2)


def test_manifest_reject_path_preserves_raw(tmp_path):
    processed, rejected = tmp_path / "processed", tmp_path / "rejected"
    processed.mkdir()
    source = processed / "bad.tif"
    write_single_band(source, np.zeros((4, 4), dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    count = quarantine_invalid_tiffs(
        processed,
        rejected,
        manifest,
        lambda image: not np.any(image),
        "all_zero",
    )
    assert count == 1
    assert not source.exists()
    assert (rejected / source.name).exists()
    record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "REJECTED"
    assert record["reject_reason"] == "all_zero"


def test_quarantine_decode_failure_is_failed_and_source_remains(tmp_path):
    processed, rejected = tmp_path / "processed", tmp_path / "rejected"
    processed.mkdir()
    source = processed / "corrupt.tif"
    source.write_bytes(b"not a tiff")
    manifest = tmp_path / "manifest.jsonl"
    assert quarantine_invalid_tiffs(processed, rejected, manifest, lambda _: True, "bad") == 0
    record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert source.exists() and not (rejected / source.name).exists()
    assert record["status"] == "FAILED" and record["reject_reason"].startswith("decode_error:")


def test_quarantine_manifest_failure_rolls_back_move(tmp_path, monkeypatch):
    processed, rejected = tmp_path / "processed", tmp_path / "rejected"
    processed.mkdir()
    source = processed / "bad.tif"
    write_single_band(source, np.zeros((4, 4), dtype=np.float32))
    monkeypatch.setattr(ManifestWriter, "append", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("manifest")))
    with pytest.raises(OSError, match="manifest"):
        quarantine_invalid_tiffs(processed, rejected, tmp_path / "manifest.jsonl", lambda _: True, "all_zero")
    assert source.exists() and not (rejected / source.name).exists()


def test_quarantine_rollback_failure_is_not_silenced(tmp_path, monkeypatch):
    processed, rejected = tmp_path / "processed", tmp_path / "rejected"
    processed.mkdir()
    source = processed / "bad.tif"
    write_single_band(source, np.zeros((4, 4), dtype=np.float32))
    monkeypatch.setattr(ManifestWriter, "append", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("manifest")))
    real_replace = quarantine_module.os.replace
    calls = 0
    def fail_rollback(origin, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("rollback")
        return real_replace(origin, target)
    monkeypatch.setattr(quarantine_module.os, "replace", fail_rollback)
    with pytest.raises(OSError):
        quarantine_invalid_tiffs(processed, rejected, tmp_path / "manifest.jsonl", lambda _: True, "all_zero")
    assert (rejected / source.name).exists()


def test_fuse_tile_manifest_and_raw_immutable(tmp_path):
    vv, vh = tmp_path / "raw" / "VV", tmp_path / "raw" / "VH"
    fused, tiles = tmp_path / "processed" / "fused", tmp_path / "processed" / "tiles"
    vv.mkdir(parents=True)
    vh.mkdir(parents=True)
    vv_file = vv / "622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db.tif"
    vh_file = vh / "622_975_S1A__IW___D_20220721T120000_VH_gamma0-rtc_db.tif"
    write_single_band(vv_file, np.ones((8, 8), dtype=np.float32))
    write_single_band(vh_file, np.full((8, 8), 2, dtype=np.float32))
    manifest = fuse_and_split_images(vv, vh, fused, tiles, tile_size=8)
    assert vv_file.exists() and vh_file.exists()
    assert len(list(tiles.glob("*.tif"))) == 1
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert records[0]["status"] == "SUCCESS"
    assert records[0]["shape"] == [2, 8, 8]


def test_default_preprocessing_output_matches_inference_filename_contract(tmp_path):
    vv, vh = tmp_path / "raw" / "VV", tmp_path / "raw" / "VH"
    fused, tiles = tmp_path / "processed" / "fused", tmp_path / "processed" / "tiles"
    vv.mkdir(parents=True)
    vh.mkdir(parents=True)
    stem = "622_975_S1A__IW___D_20220721T120000"
    vv_file = vv / f"{stem}_VV_gamma0-rtc_db.tif"
    vh_file = vh / f"{stem}_VH_gamma0-rtc_db.tif"
    write_single_band(vv_file, np.ones((8, 8), dtype=np.float32))
    write_single_band(vh_file, np.ones((8, 8), dtype=np.float32))

    fuse_and_split_images(vv, vh, fused, tiles, tile_size=8)

    output = next(tiles.glob("*.tif"))
    assert output.name == f"{stem}_VV_gamma0-rtc_db_0_0_fused.tif"
    assert output.match("622_975_S1A__IW___D_*_VV_gamma0-rtc_db_0_0_fused.tif")


@pytest.mark.parametrize(
    ("vv_data", "vh_data", "reason"),
    [
        (np.full((8, 8), np.nan, dtype=np.float32), np.ones((8, 8), dtype=np.float32), "nan"),
        (np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32), "all_zero"),
    ],
)
def test_rejected_fusion_preserves_raw_and_matches_manifest(tmp_path, vv_data, vh_data, reason):
    vv, vh = tmp_path / "raw" / "VV", tmp_path / "raw" / "VH"
    vv.mkdir(parents=True); vh.mkdir(parents=True)
    vv_file = vv / "622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db.tif"
    vh_file = vh / "622_975_S1A__IW___D_20220721T120000_VH_gamma0-rtc_db.tif"
    write_single_band(vv_file, vv_data); write_single_band(vh_file, vh_data)
    manifest = fuse_and_split_images(vv, vh, tmp_path / "fused", tmp_path / "tiles", tile_size=8)
    record = json.loads(manifest.read_text().splitlines()[0])
    assert vv_file.exists() and vh_file.exists()
    assert record["status"] == "REJECTED" and record["reject_reason"] == reason
    assert (tmp_path / "rejected" / f"622_975_20220721T120000_full.json").exists()


def test_shape_mismatch_is_rejected_without_modifying_raw(tmp_path):
    from rasterio.transform import from_origin

    vv, vh = tmp_path / "raw" / "VV", tmp_path / "raw" / "VH"
    vv.mkdir(parents=True); vh.mkdir(parents=True)
    vv_file = vv / "622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db.tif"
    vh_file = vh / "622_975_S1A__IW___D_20220721T120000_VH_gamma0-rtc_db.tif"
    transform = from_origin(0, 80, 10, 10)
    write_single_band(vv_file, np.ones((8, 8), dtype=np.float32), transform=transform)
    write_single_band(vh_file, np.ones((4, 4), dtype=np.float32), transform=transform)
    manifest = fuse_and_split_images(vv, vh, tmp_path / "fused", tmp_path / "tiles", tile_size=4)
    record = json.loads(manifest.read_text().splitlines()[0])
    assert record["status"] == "REJECTED" and record["reject_reason"] == "shape_error"
    assert vv_file.exists() and vh_file.exists()


def test_missing_pair_is_failed_in_manifest(tmp_path):
    vv, vh = tmp_path / "raw" / "VV", tmp_path / "raw" / "VH"
    vv.mkdir(parents=True); vh.mkdir(parents=True)
    write_single_band(
        vv / "622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db.tif",
        np.ones((8, 8), dtype=np.float32),
    )
    manifest_path = tmp_path / "manifest.jsonl"
    with pytest.raises(Exception, match="missing VH"):
        fuse_and_split_images(vv, vh, tmp_path / "fused", tmp_path / "tiles", manifest_path=manifest_path)
    record = json.loads(manifest_path.read_text().splitlines()[0])
    assert record["status"] == "FAILED" and record["reject_reason"].startswith("pairing_error:")


def test_tile_write_failure_records_failed_and_preserves_raw(tmp_path, monkeypatch):
    vv, vh = tmp_path / "raw" / "VV", tmp_path / "raw" / "VH"
    vv.mkdir(parents=True); vh.mkdir(parents=True)
    vv_file = vv / "622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db.tif"
    vh_file = vh / "622_975_S1A__IW___D_20220721T120000_VH_gamma0-rtc_db.tif"
    write_single_band(vv_file, np.ones((8, 8), dtype=np.float32))
    write_single_band(vh_file, np.ones((8, 8), dtype=np.float32))
    real_write = split_module.atomic_write_geotiff
    def fail_tiles(path, data, metadata):
        if "tiles" in str(path):
            raise OSError("injected tile write")
        return real_write(path, data, metadata)
    monkeypatch.setattr(split_module, "atomic_write_geotiff", fail_tiles)
    manifest_path = tmp_path / "manifest.jsonl"
    with pytest.raises(OSError, match="injected"):
        fuse_and_split_images(
            vv, vh, tmp_path / "fused", tmp_path / "tiles", tile_size=8, manifest_path=manifest_path
        )
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert records[-1]["status"] == "FAILED"
    assert records[-1]["reject_reason"] == "write_error:OSError"
    assert vv_file.exists() and vh_file.exists()


def test_fused_write_failure_is_not_mislabeled_decode_error(tmp_path, monkeypatch):
    vv, vh = tmp_path / "raw" / "VV", tmp_path / "raw" / "VH"
    vv.mkdir(parents=True); vh.mkdir(parents=True)
    vv_file = vv / "622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db.tif"
    vh_file = vh / "622_975_S1A__IW___D_20220721T120000_VH_gamma0-rtc_db.tif"
    write_single_band(vv_file, np.ones((8, 8), dtype=np.float32))
    write_single_band(vh_file, np.ones((8, 8), dtype=np.float32))
    monkeypatch.setattr(
        split_module,
        "atomic_write_geotiff",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected fused write")),
    )
    manifest_path = tmp_path / "manifest.jsonl"
    fuse_and_split_images(
        vv, vh, tmp_path / "fused", tmp_path / "tiles", tile_size=8, manifest_path=manifest_path
    )
    record = json.loads(manifest_path.read_text().splitlines()[0])
    assert record["status"] == "FAILED"
    assert record["reject_reason"] == "write_error:OSError"
    assert vv_file.exists() and vh_file.exists()


def test_truncated_existing_tile_is_rewritten_not_skipped(tmp_path):
    vv, vh = tmp_path / "raw" / "VV", tmp_path / "raw" / "VH"
    fused, tiles = tmp_path / "fused", tmp_path / "tiles"
    vv.mkdir(parents=True); vh.mkdir(parents=True); tiles.mkdir()
    stem = "622_975_S1A__IW___D_20220721T120000"
    write_single_band(vv / f"{stem}_VV_gamma0-rtc_db.tif", np.ones((8, 8), dtype=np.float32))
    write_single_band(vh / f"{stem}_VH_gamma0-rtc_db.tif", np.ones((8, 8), dtype=np.float32))
    tile = tiles / f"{stem}_VV_gamma0-rtc_db_0_0_fused.tif"
    tile.write_bytes(b"")
    fuse_and_split_images(vv, vh, fused, tiles, tile_size=8)
    assert validate_geotiff(tile, expected_shape=(8, 8), expected_channels=2)
