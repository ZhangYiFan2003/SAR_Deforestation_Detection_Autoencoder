import pytest

from config.parse_args import parse_arguments
from pipeline.anomaly_detection.reports import InferenceRunReport


def test_cli_actions_are_explicit():
    with pytest.raises(SystemExit):
        parse_arguments([])
    args = parse_arguments(["--train"])
    assert args.train is True and args.test is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--train", "--num-workers", "-1"],
        ["--train", "--prefetch-factor", "0"],
        ["--train", "--num-workers", "0", "--persistent-workers"],
        ["--train", "--non-blocking", "--no-pin-memory"],
    ],
)
def test_cli_rejects_invalid_dataloader_contracts(argv):
    with pytest.raises(SystemExit):
        parse_arguments(argv)


def test_inference_report_status():
    assert InferenceRunReport().status == "FAILED"
    complete = InferenceRunReport(expected_tiles=2, processed_tiles=2)
    assert complete.status == "SUCCESS"
    partial = InferenceRunReport(expected_tiles=2, processed_tiles=1, failed_tiles=1)
    assert partial.status == "PARTIAL"
    assert InferenceRunReport(expected_tiles=100, processed_tiles=97, failed_tiles=3).status == "PARTIAL"
    assert InferenceRunReport(expected_tiles=1, processed_tiles=1, missing_pairs=1).status == "PARTIAL"
    assert InferenceRunReport(expected_tiles=1, processed_tiles=1, fatal_errors=["write"]).status == "FAILED"
    # A pre-existing output must not override incomplete accounting.
    assert InferenceRunReport(
        expected_tiles=2, processed_tiles=1, failed_tiles=1, output_artifact="exists.shp"
    ).status == "PARTIAL"
