import numpy as np
from shapely.geometry import box

from pipeline.evaluation import EvaluationGrid, binary_metrics, evaluate_geometries, evaluate_vector_files


def test_metric_formulas():
    ground_truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    metrics = binary_metrics(ground_truth, prediction)
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (1, 1, 1, 1)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.iou == 1 / 3


def test_fixed_aoi_does_not_drop_false_negative():
    grid = EvaluationGrid.from_bounds(
        crs="EPSG:32621", bounds=(0, 0, 10, 10), resolution=1
    )
    first_gt = box(1, 1, 3, 3)
    second_gt = box(7, 7, 9, 9)
    prediction_only_first = box(1, 1, 3, 3)
    metrics = evaluate_geometries(
        [first_gt, second_gt], [prediction_only_first], grid
    )
    assert metrics.tp == 4
    assert metrics.fn == 4
    assert metrics.recall == 0.5


def test_valid_mask_controls_tn_universe():
    grid = EvaluationGrid.from_bounds(
        crs="EPSG:32621",
        bounds=(0, 0, 2, 2),
        resolution=1,
        valid_mask=np.array([[1, 1], [0, 0]], dtype=bool),
    )
    metrics = evaluate_geometries([], [], grid)
    assert metrics.tn == 2


def test_empty_prediction_counts_all_ground_truth_as_false_negative():
    grid = EvaluationGrid.from_bounds(crs="EPSG:32621", bounds=(0, 0, 4, 4), resolution=1)
    metrics = evaluate_geometries([box(1, 1, 3, 3)], [], grid)
    assert metrics.tp == 0 and metrics.fn == 4 and metrics.recall == 0.0


def test_prediction_outside_aoi_neither_expands_grid_nor_adds_fp():
    grid = EvaluationGrid.from_bounds(crs="EPSG:32621", bounds=(0, 0, 4, 4), resolution=1)
    metrics = evaluate_geometries([], [box(10, 10, 12, 12)], grid)
    assert metrics.fp == 0 and metrics.tn == 16
    assert grid.shape == (4, 4) and grid.bounds == (0, 0, 4, 4)


def test_empty_gt_and_prediction_use_explicit_zero_denominator_semantics():
    metrics = binary_metrics(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))
    assert metrics.precision == metrics.recall == metrics.f1 == metrics.iou == 0.0


def test_boundary_touching_polygon_is_clipped_to_fixed_aoi():
    grid = EvaluationGrid.from_bounds(crs="EPSG:32621", bounds=(0, 0, 2, 2), resolution=1)
    metrics = evaluate_geometries([box(-1, 0, 1, 1)], [box(-1, 0, 1, 1)], grid)
    assert metrics.tp == 1 and metrics.fp == 0 and metrics.fn == 0


def test_vector_file_entry_reprojects_to_grid_crs(tmp_path):
    import geopandas as gpd

    gt = tmp_path / "gt.geojson"
    pred = tmp_path / "pred.geojson"
    source_geometry = gpd.GeoSeries([box(0, 0, 0.00001, 0.00001)], crs="EPSG:4326")
    gpd.GeoDataFrame(geometry=source_geometry).to_file(gt, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=source_geometry).to_file(pred, driver="GeoJSON")
    projected = source_geometry.to_crs("EPSG:3857").total_bounds
    grid = EvaluationGrid.from_bounds(
        crs="EPSG:3857", bounds=(0, 0, 2, 2), resolution=1
    )
    metrics = evaluate_vector_files(
        ground_truth_path=gt, prediction_path=pred, grid=grid, all_touched=True
    )
    assert metrics.tp > 0 and projected[2] > 1


def test_vector_file_entry_rejects_missing_crs(tmp_path, monkeypatch):
    import geopandas as gpd
    import pytest

    monkeypatch.setattr(gpd, "read_file", lambda path: gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]))
    grid = EvaluationGrid.from_bounds(crs="EPSG:32621", bounds=(0, 0, 2, 2), resolution=1)
    with pytest.raises(ValueError, match="no CRS"):
        evaluate_vector_files(ground_truth_path="gt", prediction_path="pred", grid=grid)


def test_evaluation_cli_uses_declared_grid_not_prediction_extent(tmp_path):
    import geopandas as gpd
    import json
    from scripts.evaluate import main

    gt = tmp_path / "gt.geojson"
    pred = tmp_path / "pred.geojson"
    gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3), box(7, 7, 9, 9)], crs="EPSG:32621").to_file(gt)
    gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3)], crs="EPSG:32621").to_file(pred)
    config = tmp_path / "evaluation.json"
    config.write_text(json.dumps({
        "grid": {"crs": "EPSG:32621", "bounds": [0, 0, 10, 10], "resolution": 1},
        "ground_truth": gt.name,
        "prediction": pred.name,
    }), encoding="utf-8")
    output = tmp_path / "metrics.json"
    main(["--config", str(config), "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evaluation_grid"]["width"] == 10
    assert payload["metrics"]["fn"] == 4
    assert payload["metrics"]["recall"] == 0.5
