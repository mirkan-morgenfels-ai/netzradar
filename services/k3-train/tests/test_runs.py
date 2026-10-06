from pathlib import Path

from contract import validate_run_log
from k3_train.jsonio import read_json, round_floats
from k3_train.pipeline import BaselineOutput
from k3_train.runs import run_log_name, write_run_logs


def test_run_log_name() -> None:
    assert run_log_name("2026-01-01", "synthetic", "zscore") == "2026-01-01_synthetic_zscore.json"


def test_every_baseline_run_writes_a_complete_log(
    small_output: BaselineOutput, tmp_path: Path
) -> None:
    paths = write_run_logs(small_output, tmp_path)
    assert [path.name for path in paths] == [
        "2026-01-01_synthetic_zscore.json",
        "2026-01-01_synthetic_iforest.json",
    ]
    for path, run in zip(paths, small_output.runs, strict=True):
        log = read_json(path)
        assert validate_run_log(log) == []
        assert log["seed"] == 42
        assert log["date"] == "2026-01-01"
        assert log["dataset"]["name"] == "synthetic"
        assert log["split"]["kind"] == "temporal"
        assert log["hyperparameters"] == round_floats(run["hyperparameters"])
        assert log["prAuc"] == round(run["prAuc"], 4)
        assert log["prCurve"] == round_floats(run["prCurve"])


def test_hyperparameters_are_recorded(small_output: BaselineOutput) -> None:
    zscore, iforest = small_output.runs
    assert zscore["hyperparameters"]["madScale"] == 1.4826
    assert zscore["hyperparameters"]["fitOn"] == "train"
    assert iforest["hyperparameters"]["nEstimators"] == 200
    assert iforest["hyperparameters"]["contamination"] == 0.02
    assert iforest["hyperparameters"]["randomState"] == 42
    assert iforest["hyperparameters"]["features"][-4:] == [
        "g_in_degree",
        "g_out_degree",
        "g_betweenness",
        "g_eigenvector",
    ]
