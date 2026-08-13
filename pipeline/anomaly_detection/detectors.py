"""Explicit transductive and inductive reconstruction-error detector contracts."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class ReconstructionErrorDetector:
    mode: str = "transductive_gmm"
    threshold: Optional[float] = None
    random_state: int = 0
    model: Optional[GaussianMixture] = None
    anomaly_cluster: Optional[int] = None
    fit_split: Optional[str] = None
    fit_scope: Optional[str] = None

    def __post_init__(self) -> None:
        if self.fit_split is None:
            self.fit_split = "test" if self.mode == "transductive_gmm" else "validation"
        if self.fit_scope is None:
            self.fit_scope = (
                "test_time_errors"
                if self.mode == "transductive_gmm"
                else "frozen_calibration_parameters"
            )

    def fit(self, calibration_errors: np.ndarray) -> "ReconstructionErrorDetector":
        errors = np.asarray(calibration_errors).reshape(-1, 1)
        if self.mode in ("transductive_gmm", "validation_gmm"):
            self.model = GaussianMixture(n_components=2, random_state=self.random_state)
            self.model.fit(errors)
            self.anomaly_cluster = int(np.argmax(self.model.means_.ravel()))
        elif self.mode == "fixed_threshold":
            if self.threshold is None:
                raise ValueError("fixed_threshold mode requires threshold")
        else:
            raise ValueError(f"Unsupported detector mode: {self.mode}")
        return self

    def predict(self, errors: np.ndarray) -> np.ndarray:
        values = np.asarray(errors)
        if self.mode == "fixed_threshold":
            if self.threshold is None:
                raise RuntimeError("Detector is not fitted/configured")
            return (values >= self.threshold).astype(np.uint8)
        if self.model is None or self.anomaly_cluster is None:
            raise RuntimeError("GMM detector must be fitted before prediction")
        labels = self.model.predict(values.reshape(-1, 1))
        return (labels.reshape(values.shape) == self.anomaly_cluster).astype(np.uint8)

    @property
    def protocol(self) -> str:
        return "TRANSDUCTIVE" if self.mode == "transductive_gmm" else "INDUCTIVE"

    def metadata(self) -> Dict[str, str]:
        return {
            "detector_type": self.mode,
            "protocol": self.protocol,
            "fit_split": str(self.fit_split),
            "fit_scope": str(self.fit_scope),
        }
