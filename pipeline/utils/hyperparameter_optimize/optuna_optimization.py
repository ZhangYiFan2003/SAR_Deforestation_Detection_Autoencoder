"""Optuna objective with per-trial artifact isolation."""

import json
import math
from pathlib import Path

import torch
import optuna

from pipeline.datasets.data_loader import ProcessedForestDataLoader


def objective(trial, args, wrapper_class, study_root):
    # Optuna contract: raw best validation objective selects both trial and checkpoint.
    args.selection_strategy = "best_validation"
    args.lr = trial.suggest_float("lr", 1e-4, 5e-4, step=1e-4)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-5, step=1e-6)
    trial_root = Path(study_root) / f"trial_{trial.number:03d}"
    args.run_id = f"{args.run_id}_trial_{trial.number:03d}"
    args.results_path = str(trial_root)
    args.checkpoint_dir = str(trial_root / "checkpoints")
    args.log_dir = str(trial_root / "logs")
    args.best_checkpoint = str(Path(args.checkpoint_dir) / "best.ckpt")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=False)
    Path(args.log_dir).mkdir(parents=True, exist_ok=False)
    Path(trial_root, "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )

    data = ProcessedForestDataLoader(args)
    autoencoder = wrapper_class(args, data=data)
    best_validation = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            autoencoder.train(epoch)
            should_stop, validation = autoencoder.test(epoch)
            best_validation = min(best_validation, validation)
            trial.report(best_validation, step=epoch)
            should_prune = trial.should_prune()
            if should_stop or should_prune:
                if should_prune:
                    raise optuna.TrialPruned()
                break
    finally:
        autoencoder.writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_checkpoint_validation = autoencoder.early_stopping.best_validation
    if selected_checkpoint_validation is None or not math.isclose(
        selected_checkpoint_validation, best_validation, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError(
            "Optuna objective/checkpoint metric mismatch: "
            f"objective={best_validation}, checkpoint={selected_checkpoint_validation}"
        )
    if not Path(args.best_checkpoint).is_file():
        raise RuntimeError(f"Best checkpoint was not materialized: {args.best_checkpoint}")

    Path(trial_root, "metrics.json").write_text(
        json.dumps(
            {
                "optimization_metric": "best_validation_objective",
                "best_validation": best_validation,
                "selected_checkpoint_validation": selected_checkpoint_validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    trial.set_user_attr("best_checkpoint", args.best_checkpoint)
    trial.set_user_attr("best_validation", best_validation)
    return best_validation
