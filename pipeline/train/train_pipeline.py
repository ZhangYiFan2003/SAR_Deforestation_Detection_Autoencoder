import json
import logging
from time import perf_counter
from copy import deepcopy
from pathlib import Path

import optuna

from pipeline.utils.checkpointing import save_checkpoint
from pipeline.utils.hyperparameter_optimize.optuna_optimization import objective


LOGGER = logging.getLogger(__name__)


def _checkpoint_metadata(args, autoencoder, epoch, validation):
    return {
        "optimizer": autoencoder.optimizer,
        "scheduler": autoencoder.scheduler,
        "epoch": epoch,
        "best_validation": validation,
        "resolved_config": vars(args),
        "preprocessing_config": autoencoder.data.transform_config.to_dict(),
        "seed_info": {"python": args.seed, "numpy": args.seed, "torch": args.seed},
        "early_stopping_state": autoencoder.early_stopping.state_dict(),
    }


def train_model(args, autoencoder, wrapper_class):
    if args.use_optuna:
        study_id = args.run_id
        study_root = Path(args.results_path)
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            lambda trial: objective(trial, deepcopy(args), wrapper_class, study_root),
            n_trials=10,
        )
        best = {
            "best_trial_id": study.best_trial.number,
            "best_params": study.best_params,
            "best_metric": study.best_value,
            "best_checkpoint": study.best_trial.user_attrs["best_checkpoint"],
            "optimization_metric": "best_validation_objective",
        }
        (study_root / "best_trial.json").write_text(
            json.dumps(best, indent=2, sort_keys=True), encoding="utf-8"
        )
        LOGGER.info("Optuna study=%s best_trial=%s metric=%s", study_id, study.best_trial.number, study.best_value)
        return best

    best_validation = float("inf")
    last_epoch = 0
    try:
        LOGGER.info("run_id=%s stage=train device=%s", args.run_id, "cuda" if args.cuda else "cpu")
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            epoch_started = perf_counter()
            autoencoder.train(epoch)
            should_stop, validation = autoencoder.test(epoch)
            LOGGER.info(
                "run_id=%s stage=validation epoch=%s objective=%s duration_seconds=%.3f",
                args.run_id,
                epoch,
                validation,
                perf_counter() - epoch_started,
            )
            best_validation = min(best_validation, validation)
            save_checkpoint(
                Path(args.checkpoint_dir) / f"epoch_{epoch:04d}.ckpt",
                autoencoder.model,
                **_checkpoint_metadata(args, autoencoder, epoch, validation),
            )
            if should_stop:
                LOGGER.info("run_id=%s stage=train early_stop_epoch=%s", args.run_id, epoch)
                break
    finally:
        save_checkpoint(
            Path(args.checkpoint_dir) / "last.ckpt",
            autoencoder.model,
            **_checkpoint_metadata(args, autoencoder, last_epoch, best_validation),
        )

    metrics = {
        "selection_strategy": args.selection_strategy,
        "best_raw_validation": best_validation,
        "selected_checkpoint_validation": autoencoder.early_stopping.best_validation,
        "last_epoch": last_epoch,
        "best_checkpoint": args.best_checkpoint,
        "profile": autoencoder.profiler.report() if hasattr(autoencoder, "profiler") else {},
    }
    if args.profile and args.cuda:
        metrics["profile"].update(
            {
                "gpu_peak_allocated_bytes": __import__('torch').cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": __import__('torch').cuda.max_memory_reserved(),
            }
        )
    Path(args.results_path, "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics
