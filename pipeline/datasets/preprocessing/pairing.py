"""Semantic one-to-one VV/VH file pairing."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


class PairingError(ValueError):
    pass


_POLARIZATION = re.compile(r"(?i)(?<![A-Za-z0-9])(?P<pol>VV|VH)(?![A-Za-z0-9])")
_REGION = re.compile(r"^(?P<region>\d{3}_\d{3})(?:[_-]|$)")
_ACQUISITION = re.compile(r"(?P<date>\d{8})(?:T(?P<time>\d{6}))?")
_TILE = re.compile(r"(?i)(?:^|[_-])(?:tile[_-]?)?(?P<row>\d+)[_-](?P<col>\d+)(?:[_-]|$)")


@dataclass(frozen=True, order=True)
class VVFileKey:
    region: str
    acquisition: str
    tile: str
    product: str


def parse_vv_file_key(path: Union[str, Path]) -> Tuple[VVFileKey, str]:
    name = Path(path).name
    matches = list(_POLARIZATION.finditer(Path(name).stem))
    if len(matches) != 1:
        raise PairingError(
            f"Ambiguous polarization in {name!r}: expected exactly one VV or VH token"
        )
    polarization = matches[0].group("pol").upper()

    region_matches = {match.group("region") for match in _REGION.finditer(name)}
    date_matches = {
        match.group("date") + ("T" + match.group("time") if match.group("time") else "")
        for match in _ACQUISITION.finditer(name)
    }
    if len(region_matches) != 1:
        raise PairingError(f"Ambiguous or missing region in {name!r}")
    if len(date_matches) != 1:
        raise PairingError(f"Ambiguous or missing acquisition date in {name!r}")

    stem_without_pol = _POLARIZATION.sub("POL", Path(name).stem)
    tile_matches = list(_TILE.finditer(stem_without_pol))
    tile = "full"
    if tile_matches:
        # The region also looks like two integers; use the last distinct match.
        row, col = tile_matches[-1].group("row"), tile_matches[-1].group("col")
        candidate = f"{row}_{col}"
        if candidate != next(iter(region_matches)):
            tile = candidate

    product = stem_without_pol.lower()
    key = VVFileKey(
        region=next(iter(region_matches)),
        acquisition=next(iter(date_matches)),
        tile=tile,
        product=product,
    )
    return key, polarization


def pair_vv_vh_files(
    vv_files: Iterable[Union[str, Path]],
    vh_files: Iterable[Union[str, Path]],
) -> List[Tuple[VVFileKey, Path, Path]]:
    indexes: Dict[str, Dict[VVFileKey, Path]] = {"VV": {}, "VH": {}}
    for expected, files in (("VV", vv_files), ("VH", vh_files)):
        for item in files:
            path = Path(item)
            key, actual = parse_vv_file_key(path)
            if actual != expected:
                raise PairingError(
                    f"File {path.name!r} is {actual}, but it was provided as {expected}"
                )
            if key in indexes[expected]:
                raise PairingError(
                    f"Duplicate {expected} key {key}: {indexes[expected][key]} and {path}"
                )
            indexes[expected][key] = path

    vv_keys, vh_keys = set(indexes["VV"]), set(indexes["VH"])
    missing_vh = sorted(vv_keys - vh_keys)
    missing_vv = sorted(vh_keys - vv_keys)
    if missing_vh or missing_vv:
        details = []
        if missing_vh:
            details.append(f"missing VH for {missing_vh}")
        if missing_vv:
            details.append(f"missing VV for {missing_vv}")
        raise PairingError("VV/VH semantic join failed: " + "; ".join(details))
    return [(key, indexes["VV"][key], indexes["VH"][key]) for key in sorted(vv_keys)]
