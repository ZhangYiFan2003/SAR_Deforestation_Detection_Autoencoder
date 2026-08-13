"""Minimal isolated run artifact layout and metadata capture."""

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import torch


def _git_sha(project_dir: Path) -> Optional[str]:
    project_dir = project_dir.resolve()
    if not (project_dir / ".git").exists():
        # Do not accidentally report a parent workspace repository's SHA.
        return None
    try:
        root = subprocess.run(
            ["git", "-c", f"safe.directory={project_dir}", "rev-parse", "--show-toplevel"],
            cwd=str(project_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        if Path(root.stdout.strip()).resolve() != project_dir:
            return None
        result = subprocess.run(
            ["git", "-c", f"safe.directory={project_dir}", "rev-parse", "HEAD"],
            cwd=str(project_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True)
class RunContext:
    run_id: str
    root: Path
    checkpoints: Path
    logs: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints / "best.ckpt"

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints / "last.ckpt"

    def write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_run_context(args, *, run_root=None, run_id=None) -> RunContext:
    project_dir = Path(__file__).resolve().parents[2]
    root_base = Path(run_root or args.results_path)
    resolved_id = run_id or args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    )
    root = root_base / resolved_id
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {root}")
    checkpoints, logs = root / "checkpoints", root / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    context = RunContext(
        run_id=resolved_id,
        root=root,
        checkpoints=checkpoints,
        logs=logs,
        config_path=root / "config.json",
        metadata_path=root / "metadata.json",
        metrics_path=root / "metrics.json",
    )
    resolved_config = vars(args).copy()
    resolved_config.update(
        {
            "run_id": resolved_id,
            "results_path": str(root),
            "checkpoint_dir": str(checkpoints),
            "log_dir": str(logs),
            "best_checkpoint": str(context.best_checkpoint),
        }
    )
    context.write_json(context.config_path, resolved_config)
    context.write_json(
        context.metadata_path,
        {
            "run_id": resolved_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_sha(project_dir),
            "model": args.model,
            "dataset_manifest_path": getattr(args, "dataset_manifest", None),
            "dataset_manifest_version": args.dataset_manifest_version,
            "preprocessing_config": {
                "min_value": args.min_value,
                "max_value": args.max_value,
                "clamp": args.clamp_input,
                "expected_channels": args.expected_channels,
                "dtype": "float32",
                "version": "sar-minmax-v1",
            },
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "device": "cuda" if args.cuda else "cpu",
            "seed": args.seed,
            "deterministic": args.deterministic,
            "checkpoint": str(context.best_checkpoint),
            "metrics": str(context.metrics_path),
        },
    )
    return context
