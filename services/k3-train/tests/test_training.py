import numpy as np
import pandas as pd
import pytest

from k3_train.load import Dataset
from k3_train.split import SYNTHETIC_SPLIT, SplitError, split_for
from k3_train.training import (
    FIXED,
    SEARCH_SPACE,
    TRAIN_RATIO,
    Candidate,
    EarlyStopping,
    check_fit_validation_edges,
    fit_scaler,
    fit_validation_edges,
    loss_targets,
    positive_weight,
    scaler_hyperparameters,
    select_candidate,
    training_parts,
    transform,
)


def parts_of(dataset: Dataset) -> dict[str, np.ndarray]:
    return training_parts(dataset.nodes, split_for(dataset.name, dataset.time_steps))


def label_counts(dataset: Dataset, mask: np.ndarray) -> dict[str, int]:
    counts = dataset.nodes.loc[mask, "label"].value_counts()
    return {label: int(counts.get(label, 0)) for label in ("illicit", "licit", "unknown")}


def test_scaler_hand_example() -> None:
    values = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0], [100.0, 0.0]])
    scaler = fit_scaler(values, np.ones(5, dtype=bool))
    assert scaler.center.tolist() == [3.0, 0.0]
    assert scaler.scale[0] == pytest.approx(1.4826)
    assert scaler.scale[1] == pytest.approx(0.4)
    assert scaler.rules == ("mad", "std")
    scaled = transform(values, scaler)
    assert scaled[4, 0] == 10.0
    assert scaled[0, 0] == pytest.approx(-1.348982, abs=1e-6)
    assert scaled[2, 0] == 0.0
    assert scaled[2, 1] == pytest.approx(2.5)
    assert scaled[0, 1] == 0.0


def test_constant_feature_keeps_unit_scale() -> None:
    values = np.array([[5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 1.0], [5.0, 4.0, 0.0]])
    scaler = fit_scaler(values, np.ones(4, dtype=bool))
    assert scaler.rules == ("one", "mad", "std")
    assert scaler.scale[0] == 1.0
    assert transform(np.array([[7.0, 2.5, 0.0]]), scaler)[0, 0] == 2.0
    hyperparameters = scaler_hyperparameters(
        ["f_constant", "f_spread", "f_binary"], scaler, "train"
    )
    assert hyperparameters["zeroMadFeatures"] == ["f_constant", "f_binary"]
    assert hyperparameters["unitScaleFeatures"] == ["f_constant"]
    assert hyperparameters["clip"] == 10.0
    assert hyperparameters["fitOn"] == "train"
    with pytest.raises(ValueError, match="Länge"):
        scaler_hyperparameters(["f_constant"], scaler, "train")


def test_scaling_uses_only_training_rows() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 3))
    train = np.arange(40) < 30
    scaler = fit_scaler(values, train)
    shifted = values.copy()
    shifted[~train] = shifted[~train] * 1000.0 + 50.0
    shifted_scaler = fit_scaler(shifted, train)
    np.testing.assert_array_equal(shifted_scaler.center, scaler.center)
    np.testing.assert_array_equal(shifted_scaler.scale, scaler.scale)
    np.testing.assert_array_equal(
        transform(shifted, shifted_scaler)[train], transform(values, scaler)[train]
    )
    assert np.abs(transform(shifted, shifted_scaler)[~train]).max() == 10.0


def test_positive_weight_hand_example() -> None:
    labels = ["illicit", "licit", "licit", "licit", "unknown", "licit", "illicit", "licit"]
    train = np.array([True] * 6 + [False] * 2)
    index, targets = loss_targets(labels, train)
    assert index.tolist() == [0, 1, 2, 3, 5]
    assert targets.tolist() == [1, 0, 0, 0, 0]
    assert positive_weight(targets) == 4.0
    changed = [*labels[:6], "licit", "illicit"]
    assert positive_weight(loss_targets(changed, train)[1]) == 4.0
    with pytest.raises(ValueError, match="illicit"):
        positive_weight([0, 0, 0])
    with pytest.raises(ValueError, match="licit"):
        positive_weight([1, 1])


def test_candidate_weights() -> None:
    targets = np.array([1, 0, 0, 0, 0])
    assert Candidate("local", TRAIN_RATIO).weight(targets) == 4.0
    assert Candidate("local", FIXED, 20.0).weight(targets) == 20.0
    with pytest.raises(ValueError, match="Gewichtsregel"):
        Candidate("local", "balanced").weight(targets)
    assert [(c.feature_set, c.weight_rule, c.fixed_weight) for c in SEARCH_SPACE] == [
        ("local", TRAIN_RATIO, None),
        ("local", FIXED, 20.0),
        ("local+graph", TRAIN_RATIO, None),
        ("local+graph", FIXED, 20.0),
    ]


def test_training_parts_on_the_default_network(default_dataset: Dataset) -> None:
    parts = parts_of(default_dataset)
    assert not (parts["fit"] & parts["validation"]).any()
    np.testing.assert_array_equal(parts["fit"] | parts["validation"], parts["train"])
    assert not (parts["train"] & parts["test"]).any()
    assert {name: int(mask.sum()) for name, mask in parts.items()} == {
        "fit": 6800,
        "validation": 1600,
        "train": 8400,
        "test": 3600,
    }
    assert label_counts(default_dataset, parts["fit"]) == {
        "illicit": 176,
        "licit": 1468,
        "unknown": 5156,
    }
    assert label_counts(default_dataset, parts["validation"]) == {
        "illicit": 21,
        "licit": 362,
        "unknown": 1217,
    }
    assert label_counts(default_dataset, parts["train"]) == {
        "illicit": 197,
        "licit": 1830,
        "unknown": 6373,
    }
    assert label_counts(default_dataset, parts["test"]) == {
        "illicit": 95,
        "licit": 757,
        "unknown": 2748,
    }
    labels = default_dataset.nodes["label"]
    fit_weight = positive_weight(loss_targets(labels, parts["fit"])[1])
    train_weight = positive_weight(loss_targets(labels, parts["train"])[1])
    assert fit_weight == pytest.approx(1468 / 176)
    assert train_weight == pytest.approx(1830 / 197)
    assert round(fit_weight, 4) == 8.3409
    assert round(train_weight, 4) == 9.2893


def test_class_weights_on_the_small_network(small_dataset: Dataset) -> None:
    parts = parts_of(small_dataset)
    labels = small_dataset.nodes["label"]
    assert label_counts(small_dataset, parts["fit"])["illicit"] == 25
    assert label_counts(small_dataset, parts["fit"])["licit"] == 106
    assert positive_weight(loss_targets(labels, parts["fit"])[1]) == pytest.approx(4.24)
    train_weight = positive_weight(loss_targets(labels, parts["train"])[1])
    assert train_weight == pytest.approx(142 / 37)
    assert round(train_weight, 4) == 3.8378


def test_loss_set_contains_only_labelled_nodes_of_the_phase(small_dataset: Dataset) -> None:
    parts = parts_of(small_dataset)
    labels = small_dataset.nodes["label"].to_numpy()
    for phase in ("fit", "train"):
        index, targets = loss_targets(labels, parts[phase])
        assert parts[phase][index].all()
        assert not parts["test"][index].any()
        assert (labels[index] != "unknown").all()
        np.testing.assert_array_equal(targets, (labels[index] == "illicit").astype(np.int64))
        assert len(index) == int((parts[phase] & (labels != "unknown")).sum())
    fit_index = loss_targets(labels, parts["fit"])[0]
    assert not parts["validation"][fit_index].any()


def test_edge_between_fit_and_validation_is_detected() -> None:
    nodes = pd.DataFrame(
        {"node_id": ["a", "b", "c", "d"], "time_step": [17, 18, 18, 22], "label": ["licit"] * 4}
    )
    parts = training_parts(nodes, SYNTHETIC_SPLIT)
    inside = pd.DataFrame({"source": ["b"], "target": ["c"]})
    assert check_fit_validation_edges(nodes, inside, parts) == 0
    forward = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    assert fit_validation_edges(nodes, forward, parts["fit"], parts["validation"]) == 1
    with pytest.raises(SplitError, match="Fit- und Validierungsknoten"):
        check_fit_validation_edges(nodes, forward, parts)
    backward = pd.DataFrame({"source": ["c", "b"], "target": ["a", "a"]})
    assert fit_validation_edges(nodes, backward, parts["fit"], parts["validation"]) == 2


def test_default_network_has_no_edge_between_fit_and_validation(
    default_dataset: Dataset,
) -> None:
    assert (
        check_fit_validation_edges(
            default_dataset.nodes, default_dataset.edges, parts_of(default_dataset)
        )
        == 0
    )


def test_select_candidate_prefers_the_first_on_ties() -> None:
    assert select_candidate([0.5, 0.7, 0.7, 0.1]) == 1
    assert select_candidate([0.3, 0.3]) == 0
    assert select_candidate([0.2]) == 0
    assert select_candidate([0.73641, 0.73644]) == 0
    assert select_candidate([0.7364, 0.73646, 0.7365]) == 1
    with pytest.raises(ValueError, match="Keine Kandidaten"):
        select_candidate([])


def test_early_stopping_keeps_the_earliest_best_epoch() -> None:
    tracker = EarlyStopping(patience=3)
    stops = []
    for epoch, value in enumerate([0.1, 0.3, 0.3, 0.2, 0.25, 0.4], start=1):
        tracker.update(epoch, value)
        stops.append(tracker.stop)
        if tracker.stop:
            break
    assert tracker.best_epoch == 2
    assert tracker.best_value == 0.3
    assert tracker.last_epoch == 5
    assert stops == [False, False, False, False, True]
