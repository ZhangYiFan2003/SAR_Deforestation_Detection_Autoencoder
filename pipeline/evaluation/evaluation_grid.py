"""Prediction-independent evaluation grid definition."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import from_origin


@dataclass(frozen=True)
class EvaluationGrid:
    crs: str
    bounds: Tuple[float, float, float, float]
    resolution: Tuple[float, float]
    transform: Affine
    width: int
    height: int
    valid_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Evaluation grid width and height must be positive")
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError("Evaluation resolution must be positive")
        if self.valid_mask is not None and self.valid_mask.shape != self.shape:
            raise ValueError(
                f"valid_mask shape {self.valid_mask.shape} does not match grid {self.shape}"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @classmethod
    def from_bounds(
        cls,
        *,
        crs: str,
        bounds: Tuple[float, float, float, float],
        resolution: float,
        valid_mask: Optional[np.ndarray] = None,
    ) -> "EvaluationGrid":
        left, bottom, right, top = bounds
        if right <= left or top <= bottom:
            raise ValueError(f"Invalid evaluation bounds: {bounds}")
        width_float = (right - left) / resolution
        height_float = (top - bottom) / resolution
        width = int(round(width_float))
        height = int(round(height_float))
        if not np.isclose(width_float, width) or not np.isclose(height_float, height):
            raise ValueError("Bounds must be an integer multiple of resolution")
        return cls(
            crs=str(crs),
            bounds=bounds,
            resolution=(resolution, resolution),
            transform=from_origin(left, top, resolution, resolution),
            width=width,
            height=height,
            valid_mask=valid_mask,
        )

    def rasterize(self, geometries: Iterable[Any], all_touched: bool = False) -> np.ndarray:
        shapes = [(geometry, 1) for geometry in geometries if geometry is not None]
        if not shapes:
            return np.zeros(self.shape, dtype=np.uint8)
        return rasterize(
            shapes,
            out_shape=self.shape,
            transform=self.transform,
            fill=0,
            default_value=1,
            all_touched=all_touched,
            dtype="uint8",
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["transform"] = tuple(self.transform)
        result["valid_mask"] = "provided" if self.valid_mask is not None else None
        return result
