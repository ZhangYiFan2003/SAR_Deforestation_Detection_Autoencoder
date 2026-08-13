"""Evaluate vector predictions on a prediction-independent grid."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evaluation import EvaluationGrid, EvaluationInputs, evaluate_vector_files
from pipeline.evaluation.evaluate import write_evaluation_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Evaluation protocol JSON")
    parser.add_argument("--output", required=True, help="Metrics/provenance JSON")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    grid_config = config["grid"]
    grid = EvaluationGrid.from_bounds(
        crs=grid_config["crs"],
        bounds=tuple(grid_config["bounds"]),
        resolution=float(grid_config["resolution"]),
    )
    def artifact_path(value):
        if value is None:
            return None
        path = Path(value)
        return str(path if path.is_absolute() else config_path.parent / path)

    inputs = EvaluationInputs(
        ground_truth=artifact_path(config["ground_truth"]),
        prediction=artifact_path(config["prediction"]),
        forest_mask=artifact_path(config.get("forest_mask")),
    )
    metrics = evaluate_vector_files(
        ground_truth_path=inputs.ground_truth,
        prediction_path=inputs.prediction,
        forest_mask_path=inputs.forest_mask,
        grid=grid,
        all_touched=bool(config.get("all_touched", False)),
    )
    write_evaluation_result(
        args.output,
        grid=grid,
        inputs=inputs,
        metrics=metrics,
        config={"protocol_path": str(config_path), "all_touched": config.get("all_touched", False)},
    )
    return metrics


if __name__ == "__main__":
    main()
