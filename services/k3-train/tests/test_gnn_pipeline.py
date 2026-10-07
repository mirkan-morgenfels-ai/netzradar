from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

SKIP_REASON = "Extra gnn fehlt (torch, torch_geometric): uv sync --extra gnn"
pytest.importorskip("torch", reason=SKIP_REASON)
pytest.importorskip("torch_geometric", reason=SKIP_REASON)

from contract import (
    expected_score_gnn_method,
    validate_edges,
    validate_metrics,
    validate_nodes,
    validate_run_log,
)
from k3_train import cli, gnn_pipeline
from k3_train.export import ExportBundle, build_exports, write_exports
from k3_train.features import FEATURE_SETS, feature_matrix
from k3_train.gnn import GnnConfig
from k3_train.gnn_pipeline import run_gnn
from k3_train.gnn_results import GnnOutput, load_gnn, save_gnn, score_column
from k3_train.jsonio import dumps, read_json
from k3_train.load import Dataset
from k3_train.pipeline import BaselineOutput
from k3_train.runs import write_run_logs
from k3_train.split import split_for
from k3_train.synth import SMALL_CONFIG
from k3_train.training import fit_scaler, training_parts, transform

RUN_DATE = "2026-01-01"
GENERATED_AT = "2026-01-01T00:00:00Z"
CONFIG = GnnConfig(max_epochs=30, patience=10, spread_seeds=(42, 43))


@pytest.fixture(scope="module")
def small_gnn(small_dataset: Dataset, small_output: BaselineOutput) -> GnnOutput:
    return run_gnn(small_dataset, small_output.measures, RUN_DATE, config=CONFIG)


@pytest.fixture(scope="module")
def gnn_bundle(
    small_dataset: Dataset, small_output: BaselineOutput, small_gnn: GnnOutput
) -> ExportBundle:
    return build_exports(small_dataset, small_output, GENERATED_AT, gnn=small_gnn)


def parts_of(dataset: Dataset) -> dict[str, np.ndarray]:
    return training_parts(dataset.nodes, split_for(dataset.name, dataset.time_steps))


def test_smoke_run_follows_the_contract(
    small_gnn: GnnOutput, gnn_bundle: ExportBundle, tmp_path: Path
) -> None:
    assert [run["method"] for run in small_gnn.runs] == ["gcn", "graphsage", "mlp"]
    assert [run["displayName"] for run in small_gnn.runs] == [
        "GCN (2 Schichten)",
        "GraphSAGE (2 Schichten)",
        "MLP ohne Kanten (2 Schichten)",
    ]
    assert validate_metrics(gnn_bundle.metrics) == []
    assert [run["method"] for run in gnn_bundle.metrics["runs"]] == [
        "zscore",
        "iforest",
        "gcn",
        "graphsage",
        "mlp",
    ]
    for run in small_gnn.runs:
        for key in ("prAuc", "precisionAtRecall50", "recallAtPrecision50", "accuracy"):
            assert 0.0 <= run[key] <= 1.0
        assert 1 <= run["hyperparameters"]["selectedEpoch"] <= 30
        assert run["seed"] == 42
        assert run["date"] == RUN_DATE
        assert run["hyperparameters"]["selectionSteps"] == {"from": 1, "to": 6}
        assert run["hyperparameters"]["validationSteps"] == {"from": 7, "to": 8}
        assert run["hyperparameters"]["finalFitSteps"] == {"from": 1, "to": 8}
        assert len(run["hyperparameters"]["search"]) == 4
    for method in ("gcn", "graphsage", "mlp"):
        scores = small_gnn.scores[score_column(method)].to_numpy()
        assert scores.shape == (960,)
        assert np.isfinite(scores).all()
    paths = write_run_logs(small_gnn, tmp_path)
    assert [path.name for path in paths] == [
        "2026-01-01_synthetic_gcn.json",
        "2026-01-01_synthetic_graphsage.json",
        "2026-01-01_synthetic_mlp.json",
    ]
    for path in paths:
        log = read_json(path)
        assert validate_run_log(log) == []
        assert log["hyperparameters"]["seedSpread"]["seeds"] == [42, 43]
        assert log["hyperparameters"]["seedSpread"]["prAuc"][0] == log["prAuc"]
    assert all("seedSpread" not in run["hyperparameters"] for run in gnn_bundle.metrics["runs"])


def test_class_weights_come_from_the_loss_labels(small_gnn: GnnOutput) -> None:
    for run in small_gnn.runs:
        hyperparameters = run["hyperparameters"]
        for entry in hyperparameters["search"]:
            expected = 4.24 if entry["positiveWeightRule"] == "trainRatio" else 20.0
            assert entry["positiveWeight"] == pytest.approx(expected)
        final = 142 / 37 if hyperparameters["positiveWeightRule"] == "trainRatio" else 20.0
        assert hyperparameters["positiveWeight"] == pytest.approx(final)


def test_selection_follows_the_validation(small_gnn: GnnOutput) -> None:
    for run in small_gnn.runs:
        search = run["hyperparameters"]["search"]
        values = [round(entry["validationPrAuc"], 4) for entry in search]
        chosen = values.index(max(values))
        assert [entry["selected"] for entry in search] == [
            index == chosen for index in range(len(search))
        ]
        assert run["featureSet"] == search[chosen]["featureSet"]
        assert run["hyperparameters"]["selectedEpoch"] == search[chosen]["bestEpoch"]


def test_test_nodes_change_neither_selection_nor_training(
    small_dataset: Dataset, small_output: BaselineOutput, small_gnn: GnnOutput
) -> None:
    parts = parts_of(small_dataset)
    nodes = small_dataset.nodes.copy()
    columns = small_dataset.feature_columns
    nodes.loc[parts["test"], columns] = nodes.loc[parts["test"], columns] * 1000 + 50
    nodes.loc[parts["test"], "label"] = "illicit"
    changed = run_gnn(
        replace(small_dataset, nodes=nodes), small_output.measures, RUN_DATE, config=CONFIG
    )
    train = parts["train"]
    for before, after in zip(small_gnn.runs, changed.runs, strict=True):
        assert after["featureSet"] == before["featureSet"]
        for key in ("search", "selectedEpoch", "positiveWeight", "validationPrAuc", "scaling"):
            assert after["hyperparameters"][key] == before["hyperparameters"][key]
        column = score_column(before["method"])
        np.testing.assert_array_equal(
            changed.scores[column].to_numpy()[train], small_gnn.scores[column].to_numpy()[train]
        )
        assert not np.array_equal(
            changed.scores[column].to_numpy()[parts["test"]],
            small_gnn.scores[column].to_numpy()[parts["test"]],
        )


def test_selection_scaling_comes_from_the_fit_steps_only(
    small_dataset: Dataset, small_output: BaselineOutput
) -> None:
    parts = parts_of(small_dataset)
    prepared = gnn_pipeline.prepare(small_dataset, small_output.measures, CONFIG)
    nodes = small_dataset.nodes.copy()
    columns = small_dataset.feature_columns
    validation = parts["validation"]
    nodes.loc[validation, columns] = nodes.loc[validation, columns] * 1000 + 50
    shifted = gnn_pipeline.prepare(
        replace(small_dataset, nodes=nodes), small_output.measures, CONFIG
    )
    for feature_set in FEATURE_SETS:
        values, _ = feature_matrix(small_dataset.nodes, small_output.measures, feature_set)
        selection = prepared.selection_graphs[feature_set]
        final = prepared.final_graphs[feature_set]
        fit_only = fit_scaler(values, parts["fit"])
        whole_train = fit_scaler(values, parts["train"])
        np.testing.assert_array_equal(selection.scaler.center, fit_only.center)
        np.testing.assert_array_equal(selection.scaler.scale, fit_only.scale)
        np.testing.assert_array_equal(selection.graph.x.numpy(), transform(values, fit_only))
        np.testing.assert_array_equal(final.scaler.center, whole_train.center)
        np.testing.assert_array_equal(final.scaler.scale, whole_train.scale)
        np.testing.assert_array_equal(final.graph.x.numpy(), transform(values, whole_train))
        moved = shifted.selection_graphs[feature_set].scaler
        np.testing.assert_array_equal(moved.center, selection.scaler.center)
        np.testing.assert_array_equal(moved.scale, selection.scaler.scale)
        assert not np.array_equal(
            shifted.final_graphs[feature_set].scaler.center, final.scaler.center
        )


def test_loss_never_sees_unknown_validation_or_test_labels(
    small_dataset: Dataset, small_output: BaselineOutput, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    original = gnn_pipeline.train_model

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append({"index": np.asarray(args[2]), "validation": kwargs.get("validation")})
        return original(*args, **kwargs)

    monkeypatch.setattr(gnn_pipeline, "train_model", recording)
    run_gnn(
        small_dataset,
        small_output.measures,
        RUN_DATE,
        config=GnnConfig(max_epochs=3, patience=3, spread_seeds=(42,)),
    )
    parts = parts_of(small_dataset)
    labels = small_dataset.nodes["label"].to_numpy()
    assert len(calls) == 3 * (4 + 1)
    for call in calls:
        index = call["index"]
        assert (labels[index] != "unknown").all()
        assert not parts["test"][index].any()
        if call["validation"] is None:
            assert parts["train"][index].all()
        else:
            assert parts["fit"][index].all()
            assert not parts["validation"][index].any()
            assert parts["validation"][call["validation"][0]].all()


def test_gnn_run_is_deterministic(
    small_dataset: Dataset, small_output: BaselineOutput, small_gnn: GnnOutput
) -> None:
    again = run_gnn(small_dataset, small_output.measures, RUN_DATE, config=CONFIG)
    assert dumps(again.summary()) == dumps(small_gnn.summary())
    pd.testing.assert_frame_equal(again.scores, small_gnn.scores)


def test_export_fills_score_gnn(
    gnn_bundle: ExportBundle, small_gnn: GnnOutput, tmp_path: Path
) -> None:
    assert validate_nodes(gnn_bundle.nodes) == []
    method = gnn_bundle.nodes["scoreGnnMethod"]
    assert method == expected_score_gnn_method(gnn_bundle.metrics)
    assert method in ("gcn", "graphsage")
    scores = small_gnn.scores.set_index("node_id")[score_column(method)]
    for node in gnn_bundle.nodes["nodes"]:
        assert node["scoreGnn"] == round(float(scores.at[node["id"]]), 4)
    assert (
        validate_edges(gnn_bundle.edges, {node["id"] for node in gnn_bundle.nodes["nodes"]}) == []
    )
    first = write_exports(gnn_bundle, tmp_path / "first")
    second = write_exports(gnn_bundle, tmp_path / "second")
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()
        assert left.read_bytes().startswith(b'{\n  "schemaVersion": 3,')


def test_export_rejects_gnn_results_of_other_data(
    small_dataset: Dataset, small_output: BaselineOutput, small_gnn: GnnOutput
) -> None:
    stale = replace(small_gnn, dataset={**small_gnn.dataset, "nodes": 1})
    with pytest.raises(ValueError, match="passen nicht zur Baseline"):
        build_exports(small_dataset, small_output, GENERATED_AT, gnn=stale)


def test_gnn_results_survive_saving(small_gnn: GnnOutput, tmp_path: Path) -> None:
    save_gnn(small_gnn, tmp_path)
    loaded = load_gnn(tmp_path)
    assert dumps(loaded.summary()) == dumps(small_gnn.summary())
    pd.testing.assert_frame_equal(loaded.scores, small_gnn.scores)


def test_cli_runs_data_baseline_gnn_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "runs"
    export_dir = tmp_path / "k3"
    monkeypatch.setattr(cli, "SynthConfig", lambda: SMALL_CONFIG)
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "EXPORT_DIR", export_dir)
    data_dir = str(tmp_path / "data")
    assert cli.main(["data", "--synth", "--data-dir", data_dir]) == 0
    assert cli.main(["baseline", "--data-dir", data_dir]) == 0
    assert cli.main(["gnn", "--data-dir", data_dir, "--max-epochs", "2"]) == 0
    assert cli.main(["export", "--data-dir", data_dir]) == 0
    metrics = read_json(export_dir / "metrics.json")
    nodes = read_json(export_dir / "nodes.json")
    assert validate_metrics(metrics) == []
    assert validate_nodes(nodes) == []
    assert nodes["scoreGnnMethod"] in ("gcn", "graphsage")
    assert len(list(runs_dir.glob("*_synthetic_*.json"))) == 5
    assert all(run["hyperparameters"].get("maxEpochs", 2) == 2 for run in metrics["runs"][2:])
