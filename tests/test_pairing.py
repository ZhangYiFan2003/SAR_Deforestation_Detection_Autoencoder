from pathlib import Path

import pytest

from pipeline.datasets.preprocessing.pairing import PairingError, pair_vv_vh_files, parse_vv_file_key


def name(date, pol, tile="0_0"):
    return Path(f"622_975_S1A__IW___D_{date}T120000_{pol}_gamma0-rtc_db_{tile}.tif")


def test_normal_pairing():
    pairs = pair_vv_vh_files([name("20220721", "VV")], [name("20220721", "VH")])
    assert len(pairs) == 1
    assert pairs[0][0].acquisition == "20220721T120000"


def test_same_count_mismatch_fails():
    with pytest.raises(PairingError, match="missing"):
        pair_vv_vh_files(
            [name("20220721", "VV"), name("20220722", "VV")],
            [name("20220721", "VH"), name("20220723", "VH")],
        )


def test_missing_vh_fails():
    with pytest.raises(PairingError, match="missing VH"):
        pair_vv_vh_files([name("20220721", "VV")], [])


def test_duplicate_key_fails():
    with pytest.raises(PairingError, match="Duplicate VV"):
        pair_vv_vh_files([name("20220721", "VV"), name("20220721", "VV")], [])


def test_duplicate_vh_key_fails():
    with pytest.raises(PairingError, match="Duplicate VH"):
        pair_vv_vh_files([], [name("20220721", "VH"), name("20220721", "VH")])


def test_wrong_date_cannot_pair():
    with pytest.raises(PairingError):
        pair_vv_vh_files([name("20220721", "VV")], [name("20220722", "VH")])


def test_real_full_scene_and_fused_tile_patterns_pair_without_polarization_in_key():
    full_vv = Path("622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db.tif")
    full_vh = Path("622_975_S1A__IW___D_20220721T120000_VH_gamma0-rtc_db.tif")
    tile_vv = Path("622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db_256_512_fused.tif")
    tile_vh = Path("622_975_S1A__IW___D_20220721T120000_VH_gamma0-rtc_db_256_512_fused.tif")
    full = pair_vv_vh_files([full_vv], [full_vh])[0][0]
    tile = pair_vv_vh_files([tile_vv], [tile_vh])[0][0]
    assert full.tile == "full"
    assert tile.tile == "256_512"
    assert "vv" not in full.product and "vh" not in full.product


def test_region_and_tile_are_part_of_semantic_key():
    with pytest.raises(PairingError, match="missing"):
        pair_vv_vh_files(
            [name("20220721", "VV", "0_0")],
            [Path("623_976_S1A__IW___D_20220721T120000_VH_gamma0-rtc_db_0_1.tif")],
        )


def test_ordering_does_not_control_pairing():
    vv = [name("20220722", "VV"), name("20220721", "VV")]
    vh = [name("20220721", "VH"), name("20220722", "VH")]
    pairs = pair_vv_vh_files(vv, vh)
    assert [key.acquisition for key, _, _ in pairs] == ["20220721T120000", "20220722T120000"]


@pytest.mark.parametrize(
    "filename",
    [
        "622_975_S1A_20220721T120000_VV_VH_gamma0.tif",
        "scene_20220721T120000_VV_gamma0.tif",
        "622_975_S1A_VV_gamma0.tif",
        "622_975_S1A_20220721T120000_gamma0.tif",
    ],
)
def test_ambiguous_or_incomplete_names_fail_fast(filename):
    with pytest.raises(PairingError):
        parse_vv_file_key(filename)


def test_unexpected_suffix_cannot_silently_pair():
    vv = name("20220721", "VV")
    vh = Path(str(name("20220721", "VH")).replace(".tif", "_different_product.tif"))
    with pytest.raises(PairingError, match="missing"):
        pair_vv_vh_files([vv], [vh])
