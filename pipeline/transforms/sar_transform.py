"""Single source of truth for SAR tensor preprocessing."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import tifffile
import torch


@dataclass(frozen=True)
class SARTransformConfig:
    """Serializable preprocessing contract used by training and inference."""

    min_value: float = -15.0
    max_value: float = -3.0
    clamp: bool = False
    expected_channels: int = 2
    dtype: str = "float32"
    version: str = "sar-minmax-v1"

    def __post_init__(self) -> None:
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value")
        if self.expected_channels <= 0:
            raise ValueError("expected_channels must be positive")
        if self.dtype != "float32":
            raise ValueError("Only float32 tensors are currently supported")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SARTransformConfig":
        return cls(**values)


class SARTransform:
    """Validate, channel-normalize, scale, clamp, and tensorize SAR arrays."""

    def __init__(self, config: SARTransformConfig = SARTransformConfig()):
        self.config = config

    def read(self, path: Union[str, Path]) -> torch.Tensor:
        path = Path(path)
        try:
            array = tifffile.imread(path)
        except Exception as exc:
            raise ValueError(f"Unable to decode TIFF {path}: {exc}") from exc
        return self(array, source=str(path))

    def __call__(self, array: np.ndarray, source: str = "<array>") -> torch.Tensor:
        chw = self._to_chw(np.asarray(array), source)
        if not np.issubdtype(chw.dtype, np.number):
            raise ValueError(f"Expected numeric SAR data in {source}, got {chw.dtype}")
        if not np.isfinite(chw).all():
            raise ValueError(f"Non-finite SAR values found in {source}")

        # Convert before arithmetic so integer and float64 TIFFs share one contract.
        normalized = chw.astype(np.float32, copy=False)
        normalized = (normalized - self.config.min_value) / (
            self.config.max_value - self.config.min_value
        )
        if self.config.clamp:
            normalized = np.clip(normalized, 0.0, 1.0)
        normalized = np.ascontiguousarray(normalized, dtype=np.float32)
        return torch.from_numpy(normalized)

    def _to_chw(self, array: np.ndarray, source: str) -> np.ndarray:
        if array.ndim != 3:
            raise ValueError(
                f"Expected a 3D two-channel SAR image in {source}, got shape {array.shape}"
            )

        channels = self.config.expected_channels
        first_is_channels = array.shape[0] == channels
        last_is_channels = array.shape[-1] == channels
        if first_is_channels and last_is_channels:
            raise ValueError(
                f"Ambiguous channel axis in {source}: shape {array.shape} has "
                f"{channels} channels on both first and last axes"
            )
        if first_is_channels:
            return array
        if last_is_channels:
            return np.transpose(array, (2, 0, 1))
        raise ValueError(
            f"Expected {channels} channels in {source}, got shape {array.shape}"
        )
