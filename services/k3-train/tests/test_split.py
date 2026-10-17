from pathlib import Path

import pandas as pd
import pytest

from k3_train.ibm_aml import load_ibm_aml
from k3_train.load import Dataset, load_elliptic
from k3_train.split import (
    ELLIPTIC_SPLIT,
    SYNTHETIC_SPLIT,
    SplitError,
    StepRange,
    TemporalSplit,
    check_split,
    cross_split_edges,
    ibm_aml_split,
    split_for,
    split_masks,
    steps_on_both_sides,
    validate_split,
)


def assert_temporal_split_holds(dataset: Dataset, split: TemporalSplit) -> None:
    validate_split(split)
    masks = split_masks(dataset.nodes, split)
    steps = dataset.nodes["time_step"]
    train_steps = set(steps[masks["train"]])
    validation_steps = set(steps[masks["validation"]])
    test_steps = set(steps[masks["test"]])
    assert train_steps
    assert test_steps
    assert max(train_steps) < min(test_steps)
    assert validation_steps <= train_steps
    assert not train_steps & test_steps
    assert not (masks["train"] & masks["test"]).any()
    assert steps_on_both_sides(dataset.nodes, split) == []
    assert cross_split_edges(dataset.nodes, dataset.edges, split) == 0
    assert check_split(dataset.nodes, dataset.edges, split) == 0


def test_synthetic_default_split(default_dataset: Dataset) -> None:
    split = split_for("synthetic", default_dataset.time_steps)
    assert split == SYNTHETIC_SPLIT
    assert split.as_dict() == {
        "kind": "temporal",
        "train": {"from": 1, "to": 21},
        "validation": {"from": 18, "to": 21},
        "test": {"from": 22, "to": 30},
    }
    assert_temporal_split_holds(default_dataset, split)


def test_small_synthetic_split(small_dataset: Dataset) -> None:
    split = split_for("synthetic", small_dataset.time_steps)
    assert split == TemporalSplit(StepRange(1, 8), StepRange(7, 8), StepRange(9, 12))
    assert_temporal_split_holds(small_dataset, split)


def test_elliptic_split(elliptic_mini_dir: Path) -> None:
    dataset = load_elliptic(elliptic_mini_dir, aggregated_count=1)
    split = split_for("elliptic", dataset.time_steps)
    assert split == ELLIPTIC_SPLIT
    assert (split.train.start, split.train.end) == (1, 34)
    assert (split.validation.start, split.validation.end) == (30, 34)
    assert (split.test.start, split.test.end) == (35, 49)
    assert_temporal_split_holds(dataset, split)


def test_ibm_aml_split_follows_sixty_twenty_twenty_by_days(ibm_aml_mini_dir: Path) -> None:
    dataset = load_ibm_aml(ibm_aml_mini_dir)
    split = split_for("ibm-aml", dataset.time_steps)
    assert split == TemporalSplit(StepRange(1, 8), StepRange(7, 8), StepRange(9, 10))
    assert_temporal_split_holds(dataset, split)


def test_ibm_aml_split_for_other_lengths() -> None:
    assert ibm_aml_split(16) == TemporalSplit(
        StepRange(1, 13), StepRange(11, 13), StepRange(14, 16)
    )
    assert ibm_aml_split(97) == TemporalSplit(
        StepRange(1, 78), StepRange(59, 78), StepRange(79, 97)
    )
    assert ibm_aml_split(4) == TemporalSplit(StepRange(1, 3), StepRange(3, 3), StepRange(4, 4))
    assert ibm_aml_split(5) == TemporalSplit(StepRange(1, 4), StepRange(4, 4), StepRange(5, 5))
    for steps in (4, 5, 10, 16, 97):
        validate_split(ibm_aml_split(steps))
    with pytest.raises(SplitError, match="mindestens 4 Tage"):
        ibm_aml_split(3)


def test_overlapping_split_is_rejected() -> None:
    faulty = TemporalSplit(StepRange(1, 22), StepRange(18, 21), StepRange(22, 30))
    with pytest.raises(SplitError, match="Trainingsschritt"):
        validate_split(faulty)


def test_validation_outside_training_is_rejected() -> None:
    faulty = TemporalSplit(StepRange(1, 21), StepRange(20, 23), StepRange(24, 30))
    with pytest.raises(SplitError, match="Validierungsteil"):
        validate_split(faulty)


def test_edge_between_train_and_test_is_detected() -> None:
    nodes = pd.DataFrame({"node_id": ["a", "b", "c"], "time_step": [1, 2, 2]})
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    split = TemporalSplit(StepRange(1, 1), StepRange(1, 1), StepRange(2, 2))
    assert cross_split_edges(nodes, edges, split) == 1
    with pytest.raises(SplitError, match="Kanten"):
        check_split(nodes, edges, split)


def test_edge_from_test_to_train_is_detected() -> None:
    nodes = pd.DataFrame({"node_id": ["a", "b", "c"], "time_step": [1, 2, 2]})
    split = TemporalSplit(StepRange(1, 1), StepRange(1, 1), StepRange(2, 2))
    backwards = pd.DataFrame({"source": ["b", "b"], "target": ["a", "c"]})
    assert cross_split_edges(nodes, backwards, split) == 1
    with pytest.raises(SplitError, match="Kanten"):
        check_split(nodes, backwards, split)
    both = pd.DataFrame({"source": ["a", "b"], "target": ["b", "a"]})
    assert cross_split_edges(nodes, both, split) == 2


def test_time_step_on_both_sides_is_detected() -> None:
    nodes = pd.DataFrame({"node_id": ["a", "b", "c"], "time_step": [1, 2, 3]})
    overlapping = TemporalSplit(StepRange(1, 2), StepRange(1, 2), StepRange(2, 3))
    assert steps_on_both_sides(nodes, overlapping) == [2]
