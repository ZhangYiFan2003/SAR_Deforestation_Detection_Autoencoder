from pathlib import Path

import matplotlib
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


matplotlib.use("Agg", force=True)


@pytest.fixture
def sar_array():
    values = np.linspace(-15.0, -3.0, 2 * 8 * 8, dtype=np.float32)
    return values.reshape(2, 8, 8)


@pytest.fixture
def sar_tiff(tmp_path: Path, sar_array):
    path = tmp_path / "622_975_S1A__IW___D_20220721T000000_VV_gamma0-rtc_db_0_0_fused.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=2,
        dtype="float32",
        crs="EPSG:32621",
        transform=from_origin(0, 80, 10, 10),
    ) as destination:
        destination.write(sar_array)
    return path


def write_single_band(path: Path, data: np.ndarray, transform=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=str(data.dtype),
        crs="EPSG:32621",
        transform=transform or from_origin(0, data.shape[0] * 10, 10, 10),
    ) as destination:
        destination.write(data, 1)


def write_sar(path: Path, data: np.ndarray):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs="EPSG:32621",
        transform=from_origin(0, data.shape[1] * 10, 10, 10),
    ) as destination:
        destination.write(data)
