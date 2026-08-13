import numpy as np
import pytest
import torch

from pipeline.anomaly_detection.anomaly_detection import AnomalyDetection
from pipeline.datasets.data_loader import ProcessedForestDataset
from pipeline.transforms import SARTransform, SARTransformConfig
from pipeline.anomaly_detection.anomaly_detection_pipeline import AnomalyDetectionPipeline
from conftest import write_sar


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.detach().cpu().clone())
        return value


def test_normalization_and_dtype():
    array = np.array([[[-15.0, -9.0, -3.0]], [[-15.0, -9.0, -3.0]]])
    tensor = SARTransform()(array)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor[0, 0], torch.tensor([0.0, 0.5, 1.0]))


def test_clamp_is_explicit():
    array = np.array(
        [[[-18.0, 0.0, -9.0]], [[-18.0, 0.0, -9.0]]]
    )
    unclamped = SARTransform()(array)
    clamped = SARTransform(SARTransformConfig(clamp=True))(array)
    assert unclamped.min() < 0 and unclamped.max() > 1
    assert clamped.min() == 0 and clamped.max() == 1


@pytest.mark.parametrize("shape", [(8, 8), (1, 8, 8), (3, 8, 8)])
def test_shape_channel_validation(shape):
    with pytest.raises(ValueError):
        SARTransform()(np.zeros(shape, dtype=np.float32))


def test_dataset_chw_conversion(tmp_path, sar_array):
    import tifffile

    tifffile.imwrite(tmp_path / "hwc.tif", np.transpose(sar_array, (1, 2, 0)))
    result = ProcessedForestDataset(tmp_path)[0]
    assert result.shape == (2, 8, 8)


def test_train_inference_transform_consistency(sar_tiff):
    dataset = ProcessedForestDataset(sar_tiff.parent)
    train_tensor = dataset[0]
    detector = AnomalyDetection()
    detector.sar_transform = dataset.sar_transform
    inference_tensor, _ = detector._load_and_preprocess_image(sar_tiff, "cpu")
    assert train_tensor.shape == inference_tensor.squeeze(0).shape
    assert train_tensor.dtype == inference_tensor.dtype
    torch.testing.assert_close(train_tensor, inference_tensor.squeeze(0))


def test_all_production_inference_loaders_use_dataset_transform(tmp_path, sar_array):
    from types import SimpleNamespace

    names = [
        "622_975_S1A__IW___D_20220719T000000_VV_gamma0-rtc_db_0_0_fused.tif",
        "622_975_S1A__IW___D_20220720T000000_VV_gamma0-rtc_db_0_0_fused.tif",
        "622_975_S1A__IW___D_20220721T000000_VV_gamma0-rtc_db_0_0_fused.tif",
        "622_975_S1A__IW___D_20220722T000000_VV_gamma0-rtc_db_0_0_fused.tif",
        "622_975_S1A__IW___D_20220723T000000_VV_gamma0-rtc_db_0_0_fused.tif",
    ]
    for name in names:
        write_sar(tmp_path / name, sar_array)
    dataset = ProcessedForestDataset(tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = CaptureModel()
    args = SimpleNamespace(log_dir=str(tmp_path / "logs"), results_path=str(tmp_path), test_dir=str(tmp_path))
    pipeline = AnomalyDetectionPipeline(model, loader, loader, loader, torch.device("cpu"), args)
    expected = dataset[names.index(names[2])]
    direct, _ = pipeline._load_and_preprocess_image(tmp_path / names[2], "cpu")
    torch.testing.assert_close(direct.squeeze(0), expected)

    pipeline._compute_all_pixel_losses(
        [str(tmp_path / names[2])], transform=None, device="cpu"
    )
    torch.testing.assert_close(model.inputs[-1].squeeze(0), expected)

    pipeline.reconstruct_and_analyze_images(image_index=names.index(names[2]))
    torch.testing.assert_close(model.inputs[-1].squeeze(0), expected)

    pipeline.plot_pixel_error_histogram(image_dir=tmp_path, num_bins=4)
    for captured in model.inputs[-5:]:
        assert captured.dtype == expected.dtype
        torch.testing.assert_close(captured.squeeze(0), expected)

    report = pipeline.generate_large_change_map(
        target_date="20220721",
        prev_date="20220720",
        image_dir=tmp_path,
        tile_size=8,
        min_size=1,
    )
    assert report.status == "SUCCESS"
    for captured in model.inputs[-2:]:
        assert captured.dtype == expected.dtype
        torch.testing.assert_close(captured.squeeze(0), expected)
