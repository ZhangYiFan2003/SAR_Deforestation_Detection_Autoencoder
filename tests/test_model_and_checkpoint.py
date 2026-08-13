from types import SimpleNamespace

import numpy as np
import pytest
import torch
from config.parse_args import parse_arguments

from pipeline.anomaly_detection.anomaly_detection import AnomalyDetection
from pipeline.anomaly_detection.detectors import ReconstructionErrorDetector
from pipeline.datasets.data_loader import ProcessedForestDataset
from pipeline.models.autoencoder import AE_Network
from pipeline.utils.checkpointing import load_checkpoint, save_checkpoint
from pipeline.utils.early_stop.early_stopping import EarlyStopping
from pipeline.test.test_pipeline import test_model as run_test_model


def model_args(**overrides):
    values = {
        "output_activation": "legacy_tanh",
        "fpn_skips": "p4+p3",
        "attention_variant": "legacy",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_connected_component_filtering():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[0, 0] = 1
    image[3:5, 3:5] = 1
    result = AnomalyDetection()._filter_small_components(image, min_size=4)
    assert result.sum() == 4
    assert result[0, 0] == 0


def test_geotiff_dataset_model_reconstruction_shape(tmp_path):
    import tifffile

    image = np.linspace(-15, -3, 2 * 256 * 256, dtype=np.float32).reshape(2, 256, 256)
    tifffile.imwrite(tmp_path / "model_input.tif", image)
    tensor = ProcessedForestDataset(tmp_path)[0].unsqueeze(0)
    model = AE_Network(model_args()).eval()
    with torch.no_grad():
        reconstruction = model(tensor)
    assert reconstruction.shape == tensor.shape


def test_legacy_checkpoint_loading(tmp_path):
    original = torch.nn.Linear(2, 1)
    path = tmp_path / "best_model.pth"
    torch.save(original.state_dict(), path)
    restored = torch.nn.Linear(2, 1)
    metadata = load_checkpoint(path, restored)
    assert metadata["checkpoint_type"] == "legacy_state_dict"
    for expected, actual in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_structured_checkpoint_has_preprocessing(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "best.ckpt"
    save_checkpoint(
        path,
        model,
        preprocessing_config={"min_value": -15, "max_value": -3},
    )
    metadata = load_checkpoint(path, torch.nn.Linear(2, 1))
    assert metadata["preprocessing_config"]["max_value"] == -3


def test_legacy_and_structured_ae_checkpoints_forward_256(tmp_path):
    source = AE_Network(model_args()).eval()
    input_tensor = torch.zeros((1, 2, 256, 256), dtype=torch.float32)
    path = tmp_path / "model.ckpt"
    torch.save(source.state_dict(), path)
    legacy = AE_Network(model_args()).eval()
    metadata = load_checkpoint(path, legacy)
    with torch.no_grad():
        output = legacy(input_tensor)
    assert metadata["checkpoint_type"] == "legacy_state_dict"
    assert output.shape == input_tensor.shape

    save_checkpoint(path, legacy, preprocessing_config={"min_value": -15, "max_value": -3})
    structured = AE_Network(model_args()).eval()
    metadata = load_checkpoint(path, structured)
    with torch.no_grad():
        output = structured(input_tensor)
    assert metadata["checkpoint_type"] == "inference"
    assert output.shape == input_tensor.shape


def test_structured_training_checkpoint_restores_optimizer_scheduler_and_early_stop(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    early = EarlyStopping(strategy="best_validation", path=tmp_path / "unused.ckpt")
    early.best_score = -0.25
    early.best_validation = 0.25
    early.counter = 2
    path = tmp_path / "training.ckpt"
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=7,
        best_validation=0.25,
        resolved_config={"lr": 0.02},
        early_stopping_state=early.state_dict(),
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1.0)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=5, gamma=0.1)
    restored_early = EarlyStopping(path=tmp_path / "other.ckpt")
    metadata = load_checkpoint(
        path,
        restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        early_stopping=restored_early,
    )
    assert metadata["epoch"] == 7 and metadata["resolved_config"]["lr"] == 0.02
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_early.counter == 2 and restored_early.best_validation == 0.25
    torch.testing.assert_close(restored_model(torch.ones(1, 2)), model(torch.ones(1, 2)))


def test_experimental_variants_keep_legacy_default():
    legacy = AE_Network(model_args())
    sigmoid = AE_Network(model_args(output_activation="sigmoid", fpn_skips="none"))
    assert isinstance(legacy.decoder.final[-1], torch.nn.Tanh)
    assert isinstance(sigmoid.decoder.final[-1], torch.nn.Sigmoid)


def test_cli_default_is_exact_legacy_ae_contract():
    args = parse_arguments(["--train"])
    model = AE_Network(args)
    assert args.model == "AE"
    assert args.output_activation == "legacy_tanh"
    assert args.fpn_skips == "p4+p3"
    assert args.attention_variant == "legacy"
    assert model.encoder.fc.out_features == 512
    assert model.decoder.input_dim == 512
    assert isinstance(model.decoder.final[-1], torch.nn.Tanh)
    assert model.encoder.attention.scaled is False
    assert model.decoder.attention_decoder.scaled is False
    expected_sse = float(2 * 256 * 256)
    assert AE_Network is not None
    from pipeline.models.autoencoder import AE
    assert AE.loss_function(None, torch.zeros(1, 2, 256, 256), torch.ones(1, 2, 256, 256)).item() == expected_sse


def test_detector_protocols_are_explicit():
    detector = ReconstructionErrorDetector(mode="fixed_threshold", threshold=0.5)
    result = detector.fit(np.array([0.1, 0.9])).predict(np.array([[0.4, 0.6]]))
    assert detector.protocol == "INDUCTIVE"
    np.testing.assert_array_equal(result, np.array([[0, 1]], dtype=np.uint8))
    assert ReconstructionErrorDetector(mode="transductive_gmm").protocol == "TRANSDUCTIVE"
    assert detector.metadata() == {
        "detector_type": "fixed_threshold",
        "protocol": "INDUCTIVE",
        "fit_split": "validation",
        "fit_scope": "frozen_calibration_parameters",
    }
    assert ReconstructionErrorDetector(mode="transductive_gmm").metadata()["fit_split"] == "test"


def test_best_validation_checkpoint_uses_exact_raw_minimum_and_epoch(tmp_path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "best.ckpt"
    early = EarlyStopping(delta=0.001, strategy="best_validation", path=path)
    early(1.0, model, epoch=1)
    early(0.9995, model, epoch=2)
    metadata = load_checkpoint(path, torch.nn.Linear(1, 1))
    assert early.best_validation == 0.9995
    assert metadata["best_validation"] == 0.9995
    assert metadata["epoch"] == 2


def test_missing_checkpoint_is_a_real_failure(tmp_path):
    args = SimpleNamespace(checkpoint=str(tmp_path / "missing.ckpt"), best_checkpoint="unused")
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        run_test_model(args, SimpleNamespace(), SimpleNamespace())
