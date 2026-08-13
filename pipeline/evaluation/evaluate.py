"""Evaluation entry points that require a fixed grid before predictions."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import numpy as np
import geopandas as gpd

from .evaluation_grid import EvaluationGrid
from .metrics import BinaryMetrics, binary_metrics


@dataclass(frozen=True)
class EvaluationInputs:
    ground_truth: str
    prediction: str
    forest_mask: Optional[str] = None


def evaluate_arrays(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    grid: EvaluationGrid,
    forest_mask: Optional[np.ndarray] = None,
) -> BinaryMetrics:
    if ground_truth.shape != grid.shape or prediction.shape != grid.shape:
        raise ValueError("Ground truth and prediction must already be aligned to the fixed grid")
    valid = np.ones(grid.shape, dtype=bool)
    if grid.valid_mask is not None:
        valid &= grid.valid_mask.astype(bool)
    if forest_mask is not None:
        if forest_mask.shape != grid.shape:
            raise ValueError("forest_mask must match the fixed grid")
        valid &= forest_mask.astype(bool)
    return binary_metrics(ground_truth, prediction, valid)


def evaluate_geometries(
    ground_truth_geometries: Iterable[Any],
    prediction_geometries: Iterable[Any],
    grid: EvaluationGrid,
    *,
    forest_geometries: Optional[Iterable[Any]] = None,
    all_touched: bool = False,
) -> BinaryMetrics:
    """Rasterize geometries already expressed in ``grid.crs``."""
    ground_truth = grid.rasterize(ground_truth_geometries, all_touched=all_touched)
    prediction = grid.rasterize(prediction_geometries, all_touched=all_touched)
    forest = (
        grid.rasterize(forest_geometries, all_touched=all_touched)
        if forest_geometries is not None
        else None
    )
    return evaluate_arrays(ground_truth, prediction, grid, forest)


def _read_vector_in_grid_crs(path: Union[str, Path], grid: EvaluationGrid) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Vector artifact has no CRS: {path}")
    return frame.to_crs(grid.crs)


def evaluate_vector_files(
    *,
    ground_truth_path: Union[str, Path],
    prediction_path: Union[str, Path],
    grid: EvaluationGrid,
    forest_mask_path: Optional[Union[str, Path]] = None,
    all_touched: bool = False,
) -> BinaryMetrics:
    """Production file entry: read, reproject, align, and evaluate vector artifacts."""
    ground_truth = _read_vector_in_grid_crs(ground_truth_path, grid)
    prediction = _read_vector_in_grid_crs(prediction_path, grid)
    forest = (
        _read_vector_in_grid_crs(forest_mask_path, grid)
        if forest_mask_path is not None
        else None
    )
    return evaluate_geometries(
        ground_truth.geometry,
        prediction.geometry,
        grid,
        forest_geometries=forest.geometry if forest is not None else None,
        all_touched=all_touched,
    )


def write_evaluation_result(
    path: Union[str, Path],
    *,
    grid: EvaluationGrid,
    inputs: EvaluationInputs,
    metrics: BinaryMetrics,
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "evaluation_grid": grid.to_dict(),
        "inputs": asdict(inputs),
        "config": dict(config or {}),
        "metrics": metrics.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
