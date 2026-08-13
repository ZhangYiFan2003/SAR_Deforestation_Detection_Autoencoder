import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import write_sar
from pipeline.anomaly_detection.anomaly_detection_pipeline import AnomalyDetectionPipeline
from pipeline.datasets.data_loader import ProcessedForestDataLoader
from pipeline.datasets.preprocessing.atomic_output import atomic_write_geotiff
from pipeline.transforms import SARTransformConfig
from pipeline.utils.hyperparameter_optimize import optuna_optimization
from pipeline.experiments.run_context import _git_sha, create_run_context
from pipeline.train import train_pipeline


def _loader_args(tmp_path, **overrides):
    values = {
        "cuda": False,
        "non_blocking": False,
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "train_dir": str(tmp_path / "train"),
        "validation_dir": str(tmp_path / "validation"),
        "test_dir": str(tmp_path / "test"),
        "batch_size": 1,
        "min_value": -15.0,
        "max_value": -3.0,
        "clamp_input": False,
        "expected_channels": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("num_workers", "pin_memory", "persistent_workers"),
    [(0, False, False), (0, True, False), (1, False, False), (1, True, True)],
)
def test_dataloader_configuration_matrix_cpu(
    tmp_path, sar_array, num_workers, pin_memory, persistent_workers
):
    import tifffile

    for split in ("train", "validation", "test"):
        directory = tmp_path / split
        directory.mkdir()
        tifffile.imwrite(directory / "sample.tif", sar_array)
    data = ProcessedForestDataLoader(
        _loader_args(
            tmp_path,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    )
    loader = data.test_loader
    batch = next(iter(loader))
    assert batch.shape == (1, 2, 8, 8)
    assert loader.num_workers == num_workers
    assert loader.pin_memory is pin_memory
    assert loader.prefetch_factor == (None if num_workers == 0 else 2)
    assert loader.persistent_workers is persistent_workers


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_workers": -1},
        {"num_workers": 0, "persistent_workers": True},
        {"num_workers": 1, "prefetch_factor": 0},
        {"pin_memory": False, "non_blocking": True},
    ],
)
def test_dataloader_rejects_invalid_runtime_combinations(tmp_path, overrides):
    with pytest.raises(ValueError):
        ProcessedForestDataLoader(_loader_args(tmp_path, **overrides))


class _IdentityModel(torch.nn.Module):
    def forward(self, value):
        return value


def _large_area_pipeline(tmp_path, sar_array):
    from pipeline.datasets.data_loader import ProcessedForestDataset

    for date in ("20220720", "20220721"):
        write_sar(
            tmp_path / f"622_975_S1A__IW___D_{date}T000000_VV_gamma0-rtc_db_0_0_fused.tif",
            sar_array,
        )
    dataset = ProcessedForestDataset(tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    args = SimpleNamespace(results_path=str(tmp_path / "results"), test_dir=str(tmp_path))
    return AnomalyDetectionPipeline(_IdentityModel(), loader, loader, loader, torch.device("cpu"), args)


def test_large_area_vectorization_failure_is_not_success(tmp_path, sar_array, monkeypatch):
    pipeline = _large_area_pipeline(tmp_path, sar_array)
    monkeypatch.setattr(
        pipeline,
        "_vectorize_difference_tile",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("vectorization")),
    )
    report = pipeline.generate_large_change_map(
        "20220721", "20220720", image_dir=tmp_path, tile_size=8, min_size=1
    )
    assert report.status == "FAILED"
    assert report.processed_tiles == 0 and report.failed_tiles == 1


def test_large_area_model_failure_is_failed(tmp_path, sar_array):
    pipeline = _large_area_pipeline(tmp_path, sar_array)
    pipeline.model = torch.nn.Linear(1, 1)
    report = pipeline.generate_large_change_map(
        "20220721", "20220720", image_dir=tmp_path, tile_size=8, min_size=1
    )
    assert report.status == "FAILED" and report.failed_tiles == 1


def test_large_area_missing_pair_is_failed(tmp_path, sar_array):
    pipeline = _large_area_pipeline(tmp_path, sar_array)
    (tmp_path / "622_975_S1A__IW___D_20220720T000000_VV_gamma0-rtc_db_0_0_fused.tif").unlink()
    report = pipeline.generate_large_change_map(
        "20220721", "20220720", image_dir=tmp_path, tile_size=8, min_size=1
    )
    assert report.status == "FAILED"


def test_large_area_duplicate_tile_fails_instead_of_overwriting(tmp_path, sar_array):
    pipeline = _large_area_pipeline(tmp_path, sar_array)
    write_sar(
        tmp_path / "622_975_S1A__IW___D_20220721T120000_VV_gamma0-rtc_db_0_0_fused.tif",
        sar_array,
    )
    report = pipeline.generate_large_change_map(
        "20220721", "20220720", image_dir=tmp_path, tile_size=8, min_size=1
    )
    assert report.status == "FAILED"
    assert report.fatal_errors == ["duplicate_target_tile:0,0"]


def test_vectorization_rejects_source_without_crs(tmp_path):
    data = np.ones((2, 4, 4), dtype=np.float32)
    source = tmp_path / "no_crs.tif"
    atomic_write_geotiff(
        source,
        data,
        {"driver": "GTiff", "width": 4, "height": 4, "count": 2, "dtype": "float32"},
    )
    pipeline = SimpleNamespace(args=SimpleNamespace(results_path=str(tmp_path)))
    from pipeline.anomaly_detection.anomaly_detection import AnomalyDetection
    with pytest.raises(ValueError, match="without CRS"):
        AnomalyDetection._vectorize_difference_tile(
            pipeline, np.ones((4, 4), dtype=np.uint8), source, 0, 0
        )


def test_partial_inference_counts_corrupt_tile_without_losing_successful_tile(tmp_path, sar_array):
    pipeline = _large_area_pipeline(tmp_path, sar_array)
    for date in ("20220720", "20220721"):
        path = tmp_path / f"622_975_S1A__IW___D_{date}T000000_VV_gamma0-rtc_db_0_8_fused.tif"
        write_sar(path, sar_array)
    corrupt = tmp_path / "622_975_S1A__IW___D_20220721T000000_VV_gamma0-rtc_db_0_8_fused.tif"
    corrupt.write_bytes(b"corrupt")
    report = pipeline.generate_large_change_map(
        "20220721", "20220720", image_dir=tmp_path, tile_size=8, min_size=1
    )
    assert report.status == "PARTIAL"
    assert report.expected_tiles == 2 and report.processed_tiles == 1 and report.failed_tiles == 1


def test_merge_write_failure_forces_failed_status(tmp_path, sar_array, monkeypatch):
    from shapely.geometry import box
    import geopandas as gpd

    pipeline = _large_area_pipeline(tmp_path, sar_array)
    monkeypatch.setattr(
        pipeline, "_vectorize_difference_tile", lambda *args: ([box(0, 0, 1, 1)], "EPSG:32621")
    )
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("merge")))
    report = pipeline.generate_large_change_map(
        "20220721", "20220720", image_dir=tmp_path, tile_size=8, min_size=1
    )
    assert report.processed_tiles == 1
    assert report.status == "FAILED" and report.fatal_errors == ["merge_output:OSError"]


class _FakeTrial:
    def __init__(self, number):
        self.number = number
        self.params = {}
        self.user_attrs = {}

    def suggest_float(self, name, low, high, step=None):
        value = low + self.number * step
        self.params[name] = value
        return value

    def report(self, value, step):
        self.reported = (value, step)

    def should_prune(self):
        return False

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class _FakeData:
    transform_config = SARTransformConfig()


class _FakeWriter:
    def close(self):
        pass


class _FakeEarlyStopping:
    def __init__(self):
        self.best_validation = None


class _FakeWrapper:
    def __init__(self, args, data):
        self.args = args
        self.writer = _FakeWriter()
        self.early_stopping = _FakeEarlyStopping()

    def train(self, epoch):
        pass

    def test(self, epoch):
        metric = self.args.lr
        self.early_stopping.best_validation = metric
        checkpoint = Path(self.args.best_checkpoint)
        checkpoint.write_text(f"trial={self.args.run_id};metric={metric}", encoding="utf-8")
        return False, metric


def test_two_optuna_trials_isolate_args_config_checkpoints_logs_and_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(optuna_optimization, "ProcessedForestDataLoader", lambda args: _FakeData())
    base = SimpleNamespace(
        run_id="study",
        selection_strategy="legacy_moving_average",
        lr=9.0,
        weight_decay=9.0,
        epochs=1,
    )
    trials = [_FakeTrial(0), _FakeTrial(1)]
    values = [
        optuna_optimization.objective(trial, deepcopy(base), _FakeWrapper, tmp_path)
        for trial in trials
    ]
    assert base.lr == 9.0 and base.selection_strategy == "legacy_moving_average"
    configs = [json.loads((tmp_path / f"trial_{i:03d}" / "config.json").read_text()) for i in range(2)]
    assert configs[0]["lr"] != configs[1]["lr"]
    checkpoints = [Path(trial.user_attrs["best_checkpoint"]) for trial in trials]
    assert checkpoints[0] != checkpoints[1] and all(path.is_file() for path in checkpoints)
    assert (tmp_path / "trial_000" / "logs") != (tmp_path / "trial_001" / "logs")
    best_index = int(np.argmin(values))
    assert f"trial_{best_index:03d}" in str(checkpoints[best_index])
    for index in range(2):
        metrics = json.loads((tmp_path / f"trial_{index:03d}" / "metrics.json").read_text())
        assert metrics["best_validation"] == metrics["selected_checkpoint_validation"] == values[index]


def test_best_trial_manifest_references_one_coherent_trial(tmp_path, monkeypatch):
    best_trial = SimpleNamespace(
        number=1,
        user_attrs={"best_checkpoint": str(tmp_path / "trial_001" / "checkpoints" / "best.ckpt")},
    )
    class Study:
        def __init__(self):
            self.best_trial = best_trial
            self.best_params = {"lr": 0.0002}
            self.best_value = 0.2
        def optimize(self, function, n_trials):
            assert n_trials == 10
    monkeypatch.setattr(train_pipeline.optuna, "create_study", lambda **kwargs: Study())
    args = SimpleNamespace(use_optuna=True, run_id="study", results_path=str(tmp_path), seed=42)
    best = train_pipeline.train_model(args, None, object)
    saved = json.loads((tmp_path / "best_trial.json").read_text())
    assert saved == best
    assert saved["best_trial_id"] == 1
    assert saved["best_params"] == {"lr": 0.0002}
    assert saved["best_metric"] == 0.2
    assert "trial_001" in saved["best_checkpoint"]


def test_git_metadata_does_not_inherit_parent_repository(tmp_path, monkeypatch):
    called = False
    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
    monkeypatch.setattr("pipeline.experiments.run_context.subprocess.run", unexpected)
    assert _git_sha(tmp_path) is None
    assert called is False


def test_run_config_records_resolved_artifact_paths(tmp_path):
    args = SimpleNamespace(
        results_path=str(tmp_path),
        run_id="resolved",
        model="AE",
        dataset_manifest=None,
        dataset_manifest_version="test",
        min_value=-15.0,
        max_value=-3.0,
        clamp_input=False,
        expected_channels=2,
        cuda=False,
        seed=42,
        deterministic=False,
    )
    run = create_run_context(args)
    config = json.loads(run.config_path.read_text())
    metadata = json.loads(run.metadata_path.read_text())
    assert config["results_path"] == str(run.root)
    assert config["checkpoint_dir"] == str(run.checkpoints)
    assert config["log_dir"] == str(run.logs)
    assert config["best_checkpoint"] == str(run.best_checkpoint)
    assert metadata["checkpoint"] == str(run.best_checkpoint)
    assert metadata["metrics"] == str(run.metrics_path)


@pytest.mark.parametrize(("selected", "other"), [("AE", "VAE"), ("VAE", "AE")])
def test_train_main_instantiates_only_selected_wrapper(tmp_path, monkeypatch, selected, other):
    import train

    calls = {"data": 0, "AE": 0, "VAE": 0, "closed": 0}
    args = SimpleNamespace(
        no_cuda=True,
        seed=42,
        deterministic=False,
        model=selected,
        use_optuna=False,
        train=True,
        test=False,
    )
    run = SimpleNamespace(
        run_id="run",
        root=tmp_path,
        checkpoints=tmp_path / "checkpoints",
        logs=tmp_path / "logs",
        best_checkpoint=tmp_path / "checkpoints" / "best.ckpt",
    )
    class Writer:
        def close(self):
            calls["closed"] += 1
    class Wrapper:
        def __init__(self, args, data):
            calls[args.model] += 1
            self.writer = Writer()
    monkeypatch.setattr(train, "parse_arguments", lambda argv: args)
    monkeypatch.setattr(train, "create_run_context", lambda args: run)
    monkeypatch.setattr(
        train,
        "ProcessedForestDataLoader",
        lambda args: calls.__setitem__("data", calls["data"] + 1) or object(),
    )
    monkeypatch.setattr(train, "AE", Wrapper)
    monkeypatch.setattr(train, "VAE", Wrapper)
    monkeypatch.setattr(train, "train_model", lambda *args: None)
    train.main([])
    assert calls == {"data": 1, selected: 1, other: 0, "closed": 1}
