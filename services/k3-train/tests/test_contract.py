import copy
from collections.abc import Callable
from typing import Any

import pytest

from contract import validate_metrics, validate_nodes, validate_run_log
from k3_train.boosting import HgbConfig
from k3_train.export import ExportBundle, build_exports
from k3_train.jsonio import round_floats
from k3_train.load import Dataset
from k3_train.pipeline import BaselineOutput, run_baseline

LIGHT = HgbConfig(max_iter=25, spread_seeds=(42, 43))


@pytest.fixture(scope="module")
def light_output(small_dataset: Dataset) -> BaselineOutput:
    return run_baseline(small_dataset, run_date="2026-01-01", hgb_config=LIGHT)


@pytest.fixture(scope="module")
def light_bundle(small_dataset: Dataset, light_output: BaselineOutput) -> ExportBundle:
    return build_exports(small_dataset, light_output, generated_at="2026-01-01T00:00:00Z")


def broken_metrics(bundle: ExportBundle, change: Callable[[dict[str, Any]], None]) -> list[str]:
    metrics = copy.deepcopy(bundle.metrics)
    change(metrics["runs"][2]["hyperparameters"])
    return validate_metrics(metrics)


def test_valid_files_pass(light_bundle: ExportBundle, light_output: BaselineOutput) -> None:
    assert validate_metrics(light_bundle.metrics) == []
    assert validate_nodes(light_bundle.nodes) == []
    for log in light_output.run_logs():
        assert validate_run_log(round_floats(log)) == []


def test_hgb_search_needs_exactly_one_first_best_candidate(light_bundle: ExportBundle) -> None:
    def two_selected(hyperparameters: dict[str, Any]) -> None:
        for entry in hyperparameters["search"]:
            entry["selected"] = True

    def later_selected(hyperparameters: dict[str, Any]) -> None:
        for entry in hyperparameters["search"]:
            entry["selected"] = False
            entry["validationPrAuc"] = 0.5
        hyperparameters["search"][-1]["selected"] = True

    assert any("genau ein" in error for error in broken_metrics(light_bundle, two_selected))
    assert any("erste beste" in error for error in broken_metrics(light_bundle, later_selected))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("earlyStopping", True, "earlyStopping"),
        ("score", "probabilityIllicit", "score"),
        ("randomState", 7, "randomState"),
        ("selectedIteration", 26, "selectedIteration"),
        ("maxLeafNodes", 99, "maxLeafNodes"),
        ("finalFitSteps", {"from": 1, "to": 7}, "finalFitSteps"),
    ],
)
def test_hgb_fixed_values_and_selection_are_checked(
    light_bundle: ExportBundle, key: str, value: Any, message: str
) -> None:
    def change(hyperparameters: dict[str, Any]) -> None:
        hyperparameters[key] = value

    assert any(message in error for error in broken_metrics(light_bundle, change))


def test_seed_spread_must_belong_to_the_run(light_bundle: ExportBundle) -> None:
    def other_seed(hyperparameters: dict[str, Any]) -> None:
        hyperparameters["seedSpread"]["reportedSeed"] = 43

    def other_value(hyperparameters: dict[str, Any]) -> None:
        hyperparameters["seedSpread"]["prAuc"][0] = 0.0
        hyperparameters["seedSpread"]["min"] = 0.0

    def missing(hyperparameters: dict[str, Any]) -> None:
        del hyperparameters["seedSpread"]

    assert any("reportedSeed" in error for error in broken_metrics(light_bundle, other_seed))
    assert any("berichteten Seeds" in error for error in broken_metrics(light_bundle, other_value))
    assert any("Schlüssel" in error for error in broken_metrics(light_bundle, missing))


def test_feature_count_must_match_the_feature_set(light_bundle: ExportBundle) -> None:
    metrics = copy.deepcopy(light_bundle.metrics)
    run = metrics["runs"][2]
    run["featureSet"] = "local+graph" if run["featureSet"] == "local" else "local"
    errors = validate_metrics(metrics)
    assert any("features: Zahl" in error for error in errors)


def test_runs_must_follow_the_method_order(light_bundle: ExportBundle) -> None:
    metrics = copy.deepcopy(light_bundle.metrics)
    metrics["runs"][1], metrics["runs"][2] = metrics["runs"][2], metrics["runs"][1]
    assert "metrics.runs: Reihenfolge wie METHODS" in validate_metrics(metrics)


def test_nodes_without_gnn_have_no_gnn_fields(light_bundle: ExportBundle) -> None:
    nodes = copy.deepcopy(light_bundle.nodes)
    nodes["nodes"][0]["seedRankGnn"] = 1
    nodes["nodes"][0]["hopGnn"] = 0
    errors = validate_nodes(nodes)
    assert "nodes[0].seedRankGnn: null erwartet" in errors
    assert "nodes[0].hopGnn: null erwartet" in errors
    field = copy.deepcopy(light_bundle.nodes)
    field["selection"]["scoreFieldGnn"] = "scoreGnn"
    assert "selection.scoreFieldGnn" in validate_nodes(field)


def test_every_node_needs_a_distance_and_matching_ranks(light_bundle: ExportBundle) -> None:
    nodes = copy.deepcopy(light_bundle.nodes)
    neighbour = next(index for index, node in enumerate(nodes["nodes"]) if node["hop"] == 1)
    nodes["nodes"][neighbour]["hop"] = None
    assert f"nodes[{neighbour}]: weder hop noch hopGnn" in validate_nodes(nodes)
    ranked = copy.deepcopy(light_bundle.nodes)
    ranked["nodes"][neighbour]["seedRank"] = 3
    errors = validate_nodes(ranked)
    assert f"nodes[{neighbour}]: seedRank und hop" in errors
    assert "nodes: seedRank doppelt" in errors
