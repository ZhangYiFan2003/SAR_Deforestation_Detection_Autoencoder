"""Lightweight stage timing for the local training workflow."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict


@dataclass
class StepProfiler:
    enabled: bool = False
    totals: Dict[str, float] = field(default_factory=dict)
    samples: int = 0

    def start(self) -> float:
        return perf_counter()

    def add(self, stage: str, started: float) -> None:
        if self.enabled:
            self.totals[stage] = self.totals.get(stage, 0.0) + perf_counter() - started

    def report(self) -> Dict[str, float]:
        result = dict(self.totals)
        total = sum(self.totals.values())
        result["samples"] = self.samples
        result["samples_per_second"] = self.samples / total if total else 0.0
        return result
