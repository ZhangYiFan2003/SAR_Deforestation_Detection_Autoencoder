"""Generate reproducible ablation run configurations without fabricating results."""

import argparse
import json
from itertools import product
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/ablation_plan.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--evaluation-config", required=True)
    args = parser.parse_args()

    variants = []
    for activation, skips, attention in product(
        ("legacy_tanh", "sigmoid", "linear"),
        ("none", "p4", "p4+p3"),
        ("legacy", "scaled"),
    ):
        variants.append(
            {
                "output_activation": activation,
                "fpn_skips": skips,
                "attention_variant": attention,
                "seed": args.seed,
                "epochs": args.epochs,
                "dataset_manifest": args.dataset_manifest,
                "evaluation_config": args.evaluation_config,
                "required_metrics": [
                    "validation_reconstruction_loss",
                    "normal_reconstruction_distribution",
                    "anomaly_reconstruction_distribution",
                    "precision",
                    "recall",
                    "f1",
                    "iou",
                    "parameters",
                    "runtime",
                ],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"variants": variants}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
