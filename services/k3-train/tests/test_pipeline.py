from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from k3_train.load import Dataset
from k3_train.pipeline import SCORE_COLUMNS, BaselineOutput, run_baseline
from k3_train.split import split_for, split_masks

RUN_DATE = "2026-01-01"
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
}


def masks_of(dataset: Dataset) -> dict[str, pd.Series]:
    return split_masks(dataset.nodes, split_for(dataset.name, dataset.time_steps))


def test_small_network_matches_golden_values(small_output: BaselineOutput) -> None:
    for key, value in GOLDEN_EVALUATION.items():
        assert small_output.evaluation[key] == pytest.approx(value, abs=1e-4)
    assert [run["method"] for run in small_output.runs] == list(GOLDEN_RUNS)
    for run in small_output.runs:
        for key, value in GOLDEN_RUNS[run["method"]].items():
            assert run[key] == pytest.approx(value, abs=1e-4)


def test_test_features_do_not_change_training_scores(
    small_dataset: Dataset, small_output: BaselineOutput
) -> None:
    masks = masks_of(small_dataset)
    nodes = small_dataset.nodes.copy()
    columns = small_dataset.feature_columns
    nodes.loc[masks["test"], columns] = nodes.loc[masks["test"], columns] * 1000 + 50
    changed = run_baseline(replace(small_dataset, nodes=nodes), run_date=RUN_DATE)
    train = masks["train"].to_numpy()
    for column in SCORE_COLUMNS:
        np.testing.assert_array_equal(
            changed.scores[column].to_numpy()[train], small_output.scores[column].to_numpy()[train]
        )
    assert changed.runs[0]["hyperparameters"] == small_output.runs[0]["hyperparameters"]


def test_labels_do_not_change_any_score(
    small_dataset: Dataset, small_output: BaselineOutput
) -> None:
    nodes = small_dataset.nodes.copy()
    nodes["label"] = np.random.default_rng(0).permutation(nodes["label"].to_numpy())
    assert (nodes["label"] != small_dataset.nodes["label"]).any()
    permuted = run_baseline(replace(small_dataset, nodes=nodes), run_date=RUN_DATE)
    pd.testing.assert_frame_equal(permuted.scores, small_output.scores)
    pd.testing.assert_frame_equal(permuted.measures, small_output.measures)


def test_runs_record_the_extra_settings(small_output: BaselineOutput) -> None:
    zscore, iforest = small_output.runs
    assert zscore["hyperparameters"]["zeroMadFeatures"] == ["f_round_amount", "f_change_output"]
    assert iforest["hyperparameters"]["graphMeasures"]["sampledSteps"] == []
    assert iforest["hyperparameters"]["graphMeasures"]["eigenvectorFallbackSteps"] == []
    assert small_output.dataset["homophily"]["labelledEdges"] == sum(
        small_output.dataset["homophily"][key]
        for key in ("illicitIllicitEdges", "licitLicitEdges", "illicitLicitEdges")
    )
