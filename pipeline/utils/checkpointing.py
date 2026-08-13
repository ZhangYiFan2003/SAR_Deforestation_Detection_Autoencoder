"""Checkpoint IO with atomic writes and legacy state-dict compatibility."""

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch


CHECKPOINT_FORMAT_VERSION = 1


def build_checkpoint(
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    best_validation: Optional[float] = None,
    resolved_config: Optional[Mapping[str, Any]] = None,
    preprocessing_config: Optional[Mapping[str, Any]] = None,
    seed_info: Optional[Mapping[str, Any]] = None,
    early_stopping_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_type": "training" if optimizer is not None else "inference",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_validation": best_validation,
        "resolved_config": dict(resolved_config or {}),
        "preprocessing_config": dict(preprocessing_config or {}),
        "seed_info": dict(seed_info or {}),
        "early_stopping_state": dict(early_stopping_state or {}),
    }


def atomic_torch_save(payload: Any, target: Union[str, Path]) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def save_checkpoint(target: Union[str, Path], model: torch.nn.Module, **metadata: Any) -> Path:
    return atomic_torch_save(build_checkpoint(model, **metadata), target)


def load_checkpoint(
    source: Union[str, Path],
    model: torch.nn.Module,
    *,
    map_location: Any = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a structured checkpoint or a legacy bare model state dict."""
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location=map_location)

    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported checkpoint payload in {source}")
    if "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if early_stopping is not None and payload.get("early_stopping_state"):
            early_stopping.load_state_dict(payload["early_stopping_state"])
        return dict(payload)
    if payload and all(isinstance(key, str) for key in payload):
        model.load_state_dict(payload)
        return {
            "format_version": 0,
            "checkpoint_type": "legacy_state_dict",
            "preprocessing_config": {},
        }
    raise ValueError(f"Unsupported checkpoint mapping in {source}")
