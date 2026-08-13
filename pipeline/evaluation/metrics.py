"""Binary spatial evaluation metrics."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    tp: int
    fp: int
    fn: int
    tn: Optional[int]
    precision: float
    recall: float
    f1: float
    iou: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> BinaryMetrics:
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"ground_truth shape {ground_truth.shape} != prediction shape {prediction.shape}"
        )
    if valid_mask is None:
        valid = np.ones(ground_truth.shape, dtype=bool)
    else:
        if valid_mask.shape != ground_truth.shape:
            raise ValueError("valid_mask must match metric array shape")
        valid = valid_mask.astype(bool)

    gt = ground_truth.astype(bool)[valid]
    pred = prediction.astype(bool)[valid]
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = _safe_ratio(tp, tp + fp + fn)
    return BinaryMetrics(tp, fp, fn, tn, precision, recall, f1, iou)
