from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from k3_train.boosting import HgbConfig
from k3_train.load import Dataset
from k3_train.pipeline import (
    METHODS,
    SCORE_COLUMNS,
    UNSUPERVISED_SCORE_COLUMNS,
    BaselineOutput,
    run_baseline,
)
from k3_train.split import split_for, split_masks

RUN_DATE = "2026-01-01"
LIGHT = HgbConfig(max_iter=25, spread_seeds=(42,))
GOLDEN_EVALUATION = {
    "testPositives": 15,
    "testNegatives": 62,
    "prevalence": 0.1948,
    "randomPrAucExpected": 0.2364,
    "randomPrAucQ95": 0.3531,
}
GOLDEN_RUNS = {
    "zscore": {
        "prAuc": 0.3986,
        "precisionAtRecall50": 0.2157,
        "recallAtPrecision50": 0.4,
        "accuracy": 0.8312,
        "accuracyFlagged": 2,
        "accuracyTruePositives": 2,
    },
    "iforest": {
        "prAuc": 0.3803,
        "precisionAtRecall50": 0.2308,
        "recallAtPrecision50": 0.2667,
        "accuracy": 0.8312,
        "accuracyFlagged": 2,
        "accuracyTruePositives": 2,
    },
    "hgb": {
        "prAuc": 0.2098,
        "precisionAtRecall50": 0.2245,
        "recallAtPrecision50": 0.0,
        "accuracy": 0.7792,
        "accuracyFlagged": 2,
        "accuracyTruePositives": 0,
    },
}
GOLDEN_HGB_SELECTION = {
    "maxLeafNodes": 7,
    "positiveWeightRule": "fixed",
    "positiveWeight": 1.0,
    "selectedIteration": 7,
    "validationPrAuc": 0.3444,
}


def masks_of(dataset: Dataset) -> dict[str, pd.Series]:
    return split_masks(dataset.nodes, split_for(dataset.name, dataset.time_steps))


@pytest.fixture(scope="module")
def light_output(small_dataset: Dataset) -> BaselineOutput:
    return run_baseline(small_dataset, run_date=RUN_DATE, hgb_config=LIGHT)


def test_small_network_matches_golden_values(small_output: BaselineOutput) -> None:
    for key, value in GOLDEN_EVALUATION.items():
        assert small_output.evaluation[key] == pytest.approx(value, abs=1e-4)
    assert [run["method"] for run in small_output.runs] == list(GOLDEN_RUNS)
    for run in small_output.runs:
        for key, value in GOLDEN_RUNS[run["method"]].items():
            assert run[key] == pytest.approx(value, abs=1e-4)
    hgb = small_output.runs[2]
    assert hgb["featureSet"] == "local+graph"
    for key, value in GOLDEN_HGB_SELECTION.items():
        assert hgb["hyperparameters"][key] == pytest.approx(value, abs=1e-4)


def test_baseline_runs_follow_the_method_order(small_output: BaselineOutput) -> None:
    methods = [run["method"] for run in small_output.runs]
    assert methods == ["zscore", "iforest", "hgb"]
    assert methods == sorted(methods, key=METHODS.index)
    assert METHODS == ("zscore", "iforest", "hgb", "gcn", "graphsage", "mlp")
    assert list(small_output.scores.columns) == ["node_id", *SCORE_COLUMNS]


def test_test_features_do_not_change_training_scores(
    small_dataset: Dataset, light_output: BaselineOutput
) -> None:
    masks = masks_of(small_dataset)
    nodes = small_dataset.nodes.copy()
    columns = small_dataset.feature_columns
    nodes.loc[masks["test"], columns] = nodes.loc[masks["test"], columns] * 1000 + 50
    changed = run_baseline(replace(small_dataset, nodes=nodes), run_date=RUN_DATE, hgb_config=LIGHT)
    train = masks["train"].to_numpy()
    for column in SCORE_COLUMNS:
        np.testing.assert_array_equal(
            changed.scores[column].to_numpy()[train], light_output.scores[column].to_numpy()[train]
        )
    assert changed.runs[0]["hyperparameters"] == light_output.runs[0]["hyperparameters"]
    for key in ("search", "selectedIteration", "maxLeafNodes", "positiveWeight"):
        assert (
            changed.runs[2]["hyperparameters"][key] == light_output.runs[2]["hyperparameters"][key]
        )


def test_labels_do_not_change_any_unsupervised_score(
    small_dataset: Dataset, light_output: BaselineOutput
) -> None:
    nodes = small_dataset.nodes.copy()
    nodes["label"] = np.random.default_rng(0).permutation(nodes["label"].to_numpy())
    assert (nodes["label"] != small_dataset.nodes["label"]).any()
    permuted = run_baseline(
        replace(small_dataset, nodes=nodes), run_date=RUN_DATE, hgb_config=LIGHT
    )
    pd.testing.assert_frame_equal(
        permuted.scores[["node_id", *UNSUPERVISED_SCORE_COLUMNS]],
        light_output.scores[["node_id", *UNSUPERVISED_SCORE_COLUMNS]],
    )
    pd.testing.assert_frame_equal(permuted.measures, light_output.measures)
    assert not np.array_equal(
        permuted.scores["score_hgb"].to_numpy(), light_output.scores["score_hgb"].to_numpy()
    )


def test_runs_record_the_extra_settings(small_output: BaselineOutput) -> None:
    zscore, iforest, hgb = small_output.runs
    assert zscore["hyperparameters"]["zeroMadFeatures"] == ["f_round_amount", "f_change_output"]
    assert iforest["hyperparameters"]["graphMeasures"]["sampledSteps"] == []
    assert iforest["hyperparameters"]["graphMeasures"]["eigenvectorFallbackSteps"] == []
    assert hgb["hyperparameters"]["maxIter"] == 300
    assert hgb["hyperparameters"]["seedSpread"]["seeds"] == [42, 43, 44, 45, 46]
    assert small_output.dataset["homophily"]["labelledEdges"] == sum(
        small_output.dataset["homophily"][key]
        for key in ("illicitIllicitEdges", "licitLicitEdges", "illicitLicitEdges")
    )
