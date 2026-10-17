from typing import Any

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info

from k3_train import boosting
from k3_train.boosting import (
    HGB_SEARCH_SPACE,
    HgbCandidate,
    HgbConfig,
    HgbResult,
    best_iteration,
    fit_model,
    log_odds,
    run_hgb,
    sample_weights,
    validation_curve,
)
from k3_train.jsonio import dumps, round_floats
from k3_train.load import Dataset
from k3_train.metrics import evaluation_target, pr_auc
from k3_train.pipeline import BaselineOutput
from k3_train.split import split_for, split_masks
from k3_train.training import FIXED, TRAIN_RATIO, training_parts

LIGHT = HgbConfig(max_iter=25, spread_seeds=(42, 43))


def prepared(dataset: Dataset) -> tuple[Any, np.ndarray, np.ndarray]:
    split = split_for(dataset.name, dataset.time_steps)
    evaluated, y = evaluation_target(
        dataset.nodes["label"], split_masks(dataset.nodes, split)["test"]
    )
    return split, evaluated, y


def run_light(dataset: Dataset, measures: pd.DataFrame) -> HgbResult:
    split, evaluated, y = prepared(dataset)
    return run_hgb(dataset.nodes, measures, split, evaluated, y, 42, LIGHT)


@pytest.fixture(scope="module")
def light(small_dataset: Dataset, small_output: BaselineOutput) -> HgbResult:
    return run_light(small_dataset, small_output.measures)


def test_search_space_has_eight_candidates_in_a_fixed_order() -> None:
    assert [
        (c.feature_set, c.weight_rule, c.fixed_weight, c.max_leaf_nodes) for c in HGB_SEARCH_SPACE
    ] == [
        ("local", TRAIN_RATIO, None, 7),
        ("local", TRAIN_RATIO, None, 31),
        ("local", FIXED, 1.0, 7),
        ("local", FIXED, 1.0, 31),
        ("local+graph", TRAIN_RATIO, None, 7),
        ("local+graph", TRAIN_RATIO, None, 31),
        ("local+graph", FIXED, 1.0, 7),
        ("local+graph", FIXED, 1.0, 31),
    ]
    assert HgbConfig().search_space == HGB_SEARCH_SPACE
    assert HgbConfig().max_iter == 300
    assert HgbConfig().learning_rate == 0.1


def test_candidate_weights_by_hand() -> None:
    targets = np.array([1, 0, 0, 0, 0])
    assert HgbCandidate("local", TRAIN_RATIO, 7).weight(targets) == 4.0
    assert HgbCandidate("local", FIXED, 7, 1.0).weight(targets) == 1.0
    assert sample_weights(targets, 4.0).tolist() == [4.0, 1.0, 1.0, 1.0, 1.0]
    assert sample_weights(targets, 1.0).tolist() == [1.0] * 5


def test_best_iteration_takes_the_first_maximum() -> None:
    assert best_iteration([0.1, 0.3, 0.3, 0.2]) == (2, 0.3)
    assert best_iteration([0.5]) == (1, 0.5)
    assert best_iteration([0.2, 0.2]) == (1, 0.2)
    with pytest.raises(ValueError, match="Leere"):
        best_iteration([])


def test_score_is_the_log_odds_of_illicit() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(200, 3))
    targets = (values[:, 0] + rng.normal(scale=0.5, size=200) > 1.0).astype(np.int64)
    index = np.arange(200)
    model = fit_model(values, index, targets, 2.0, 7, 15, LIGHT, 42)
    scores = log_odds(model, values)
    probability = model.predict_proba(values)[:, 1]
    np.testing.assert_allclose(1.0 / (1.0 + np.exp(-scores)), probability, rtol=1e-12)
    curve = validation_curve(model, values, targets)
    assert len(curve) == 15
    assert curve[-1] == pytest.approx(pr_auc(targets, scores))
    assert curve == [pr_auc(targets, stage) for stage in model.staged_decision_function(values)]


def test_class_weights_and_steps_on_the_small_network(light: HgbResult) -> None:
    hyperparameters = light.hyperparameters
    for entry in hyperparameters["search"]:
        expected = 4.24 if entry["positiveWeightRule"] == TRAIN_RATIO else 1.0
        assert entry["positiveWeight"] == pytest.approx(expected)
        assert 1 <= entry["bestIteration"] <= 25
    final = 142 / 37 if hyperparameters["positiveWeightRule"] == TRAIN_RATIO else 1.0
    assert hyperparameters["positiveWeight"] == pytest.approx(final)
    assert hyperparameters["selectionSteps"] == {"from": 1, "to": 6}
    assert hyperparameters["validationSteps"] == {"from": 7, "to": 8}
    assert hyperparameters["finalFitSteps"] == {"from": 1, "to": 8}
    assert hyperparameters["maxIter"] == 25
    assert hyperparameters["randomState"] == 42
    assert hyperparameters["environment"]["threads"] == 1
    assert list(hyperparameters)[-2:] == ["seedSpread", "environment"]


def test_selection_follows_the_validation(light: HgbResult) -> None:
    search = light.hyperparameters["search"]
    values = [round(entry["validationPrAuc"], 4) for entry in search]
    chosen = values.index(max(values))
    assert [entry["selected"] for entry in search] == [
        index == chosen for index in range(len(search))
    ]
    selected = search[chosen]
    assert light.feature_set == selected["featureSet"]
    assert light.hyperparameters["selectedIteration"] == selected["bestIteration"]
    assert light.hyperparameters["maxLeafNodes"] == selected["maxLeafNodes"]
    assert light.hyperparameters["validationPrAuc"] == selected["validationPrAuc"]
    assert light.hyperparameters["positiveWeightRule"] == selected["positiveWeightRule"]


def test_seed_spread_without_random_component(light: HgbResult) -> None:
    spread = light.hyperparameters["seedSpread"]
    assert spread["seeds"] == [42, 43]
    assert spread["reportedSeed"] == 42
    assert spread["prAuc"][0] == light.metrics["prAuc"]
    assert spread["min"] == spread["max"] == spread["mean"]


def test_hgb_run_is_deterministic(
    small_dataset: Dataset, small_output: BaselineOutput, light: HgbResult
) -> None:
    again = run_light(small_dataset, small_output.measures)
    assert dumps(round_floats(again.hyperparameters)) == dumps(round_floats(light.hyperparameters))
    np.testing.assert_array_equal(again.scores, light.scores)


def test_fits_use_one_openmp_thread_and_only_labelled_training_rows(
    small_dataset: Dataset, small_output: BaselineOutput, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    original = boosting.fit_model
    original_fit = boosting.HistGradientBoostingClassifier.fit

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append({"index": np.asarray(args[1]), "max_iter": args[5]})
        return original(*args, **kwargs)

    threads: list[int] = []

    def counting_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        threads.extend(
            info["num_threads"] for info in threadpool_info() if info["user_api"] == "openmp"
        )
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(boosting, "fit_model", recording)
    monkeypatch.setattr(boosting.HistGradientBoostingClassifier, "fit", counting_fit)
    result = run_light(small_dataset, small_output.measures)
    parts = training_parts(small_dataset.nodes, split_for("synthetic", small_dataset.time_steps))
    labels = small_dataset.nodes["label"].to_numpy()
    assert len(calls) == len(HGB_SEARCH_SPACE) + len(LIGHT.spread_seeds)
    assert threads
    assert set(threads) == {1}
    for call in calls[: len(HGB_SEARCH_SPACE)]:
        assert call["max_iter"] == 25
        assert parts["fit"][call["index"]].all()
        assert not parts["validation"][call["index"]].any()
    for call in calls[len(HGB_SEARCH_SPACE) :]:
        assert call["max_iter"] == result.hyperparameters["selectedIteration"]
        assert parts["train"][call["index"]].all()
        np.testing.assert_array_equal(
            call["index"], np.flatnonzero(parts["train"] & (labels != "unknown"))
        )
    for call in calls:
        assert (labels[call["index"]] != "unknown").all()
        assert not parts["test"][call["index"]].any()


def test_test_nodes_change_neither_selection_nor_training(
    small_dataset: Dataset, small_output: BaselineOutput, light: HgbResult
) -> None:
    parts = training_parts(small_dataset.nodes, split_for("synthetic", small_dataset.time_steps))
    test = parts["test"]
    nodes = small_dataset.nodes.copy()
    columns = small_dataset.feature_columns
    nodes.loc[test, columns] = nodes.loc[test, columns] * 1000 + 50
    nodes.loc[test, "label"] = "illicit"
    split, evaluated, y = prepared(small_dataset)
    changed = run_hgb(nodes, small_output.measures, split, evaluated, y, 42, LIGHT)
    for key in (
        "search",
        "selectedIteration",
        "maxLeafNodes",
        "positiveWeight",
        "positiveWeightRule",
        "validationPrAuc",
        "features",
    ):
        assert changed.hyperparameters[key] == light.hyperparameters[key]
    assert changed.feature_set == light.feature_set
    np.testing.assert_array_equal(changed.scores[~test], light.scores[~test])
    assert not np.array_equal(changed.scores[test], light.scores[test])
