from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from k3_train.gnn_results import (
    GnnOutput,
    check_compatible,
    load_gnn,
    save_gnn,
    score_gnn_method,
    with_seed_spread,
)
from k3_train.pipeline import BaselineOutput, metrics_payload, ordered_runs

SPREAD = {"seeds": [42, 43], "reportedSeed": 42, "prAuc": [0.5, 0.4], "mean": 0.45}


def gnn_run(method: str, validation: float) -> dict[str, Any]:
    return {
        "method": method,
        "displayName": method.upper(),
        "featureSet": "local",
        "seed": 42,
        "date": "2026-01-01",
        "hyperparameters": {"validationPrAuc": validation, "environment": {"torch": "x"}},
        "prAuc": 0.5,
        "precisionAtRecall50": 0.25,
        "recallAtPrecision50": 0.0,
        "accuracy": 0.75,
        "accuracyThreshold": "oberste 2 %",
        "accuracyFlagged": 1,
        "accuracyTruePositives": 0,
        "prCurve": [{"recall": 1.0, "precision": 0.5}],
    }


def fake_output(runs: list[dict[str, Any]]) -> GnnOutput:
    return GnnOutput(
        dataset={"name": "synthetic", "nodes": 3},
        split={"kind": "temporal"},
        evaluation={"testPositives": 1},
        seed=42,
        runs=runs,
        seed_spread={run["method"]: SPREAD for run in runs},
        scores=pd.DataFrame(
            {
                "node_id": ["a", "b", "c"],
                **{f"score_{run['method']}": [0.5, -1.25, 3.0] for run in runs},
            }
        ),
    )


def test_seed_spread_goes_before_the_environment() -> None:
    hyperparameters = {"validationPrAuc": 0.3, "search": [], "environment": {"torch": "x"}}
    result = with_seed_spread(hyperparameters, SPREAD)
    assert list(result) == ["validationPrAuc", "search", "seedSpread", "environment"]
    assert list(with_seed_spread({"a": 1}, SPREAD)) == ["a", "seedSpread"]
    assert "seedSpread" not in hyperparameters


def test_seed_spread_only_in_the_run_log() -> None:
    output = fake_output([gnn_run("gcn", 0.3)])
    log = output.run_logs()[0]
    assert log["hyperparameters"]["seedSpread"] == SPREAD
    assert "seedSpread" not in output.runs[0]["hyperparameters"]


def test_score_gnn_method_uses_the_validation_and_ignores_the_mlp() -> None:
    assert score_gnn_method([gnn_run("gcn", 0.3), gnn_run("graphsage", 0.4)]) == "graphsage"
    assert score_gnn_method([gnn_run("graphsage", 0.4), gnn_run("gcn", 0.4)]) == "gcn"
    assert score_gnn_method([gnn_run("gcn", 0.2), gnn_run("mlp", 0.9)]) == "gcn"
    assert score_gnn_method([gnn_run("mlp", 0.9)]) is None
    assert score_gnn_method([]) is None


def test_score_gnn_method_compares_the_published_four_digit_values() -> None:
    assert score_gnn_method([gnn_run("gcn", 0.73641), gnn_run("graphsage", 0.73644)]) == "gcn"
    assert score_gnn_method(
        [gnn_run("gcn", 0.73636), gnn_run("graphsage", 0.7364225418944983)]
    ) == ("gcn")
    assert score_gnn_method([gnn_run("gcn", 0.7364), gnn_run("graphsage", 0.7365)]) == "graphsage"
    assert score_gnn_method([gnn_run("gcn", 0.73644), gnn_run("graphsage", 0.73646)]) == (
        "graphsage"
    )


def test_incompatible_outputs_are_rejected() -> None:
    gnn = fake_output([gnn_run("gcn", 0.3)])
    baseline = BaselineOutput(
        dataset=gnn.dataset,
        split=gnn.split,
        evaluation=gnn.evaluation,
        seed=42,
        runs=[],
        scores=pd.DataFrame(),
        measures=pd.DataFrame(),
    )
    check_compatible(baseline, gnn)
    baseline.seed = 7
    with pytest.raises(ValueError, match="seed"):
        check_compatible(baseline, gnn)


def test_save_and_load(tmp_path: Path) -> None:
    output = fake_output([gnn_run("gcn", 0.3), gnn_run("mlp", 0.1)])
    save_gnn(output, tmp_path)
    loaded = load_gnn(tmp_path)
    assert loaded.summary() == output.summary()
    pd.testing.assert_frame_equal(loaded.scores, output.scores)
    with pytest.raises(FileNotFoundError):
        load_gnn(tmp_path / "missing")


def test_runs_are_ordered_like_the_methods() -> None:
    runs = [{"method": "mlp"}, {"method": "zscore"}, {"method": "gcn"}]
    assert [run["method"] for run in ordered_runs(runs)] == ["zscore", "gcn", "mlp"]
    with pytest.raises(ValueError, match="doppelt"):
        ordered_runs([{"method": "gcn"}, {"method": "gcn"}])
    with pytest.raises(ValueError, match="Unbekannte"):
        ordered_runs([{"method": "gat"}])
    baseline = BaselineOutput(
        dataset={},
        split={},
        evaluation={},
        seed=42,
        runs=[{"method": "iforest"}, {"method": "zscore"}],
        scores=pd.DataFrame(),
        measures=pd.DataFrame(),
    )
    payload = metrics_payload(baseline, "", [{"method": "graphsage"}])
    assert [run["method"] for run in payload["runs"]] == ["zscore", "iforest", "graphsage"]
    assert payload["schemaVersion"] == 3
