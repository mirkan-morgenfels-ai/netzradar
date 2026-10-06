from dataclasses import dataclass

import pandas as pd

SYNTHETIC_REFERENCE_STEPS = 30


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class StepRange:
    start: int
    end: int

    def contains(self, step: int) -> bool:
        return self.start <= step <= self.end

    def as_dict(self) -> dict[str, int]:
        return {"from": self.start, "to": self.end}


@dataclass(frozen=True)
class TemporalSplit:
    train: StepRange
    validation: StepRange
    test: StepRange

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "temporal",
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "test": self.test.as_dict(),
        }


SYNTHETIC_SPLIT = TemporalSplit(
    train=StepRange(1, 21), validation=StepRange(18, 21), test=StepRange(22, 30)
)
ELLIPTIC_SPLIT = TemporalSplit(
    train=StepRange(1, 34), validation=StepRange(30, 34), test=StepRange(35, 49)
)


def synthetic_split(time_steps: int) -> TemporalSplit:
    if time_steps == SYNTHETIC_REFERENCE_STEPS:
        return SYNTHETIC_SPLIT
    if time_steps < 3:
        raise SplitError("Fuer einen zeitlichen Split sind mindestens 3 Zeitschritte noetig")
    train_end = min(time_steps - 1, max(1, round(0.7 * time_steps)))
    validation_length = max(1, round(time_steps * 4 / SYNTHETIC_REFERENCE_STEPS))
    validation_start = max(1, train_end - validation_length + 1)
    return TemporalSplit(
        train=StepRange(1, train_end),
        validation=StepRange(validation_start, train_end),
        test=StepRange(train_end + 1, time_steps),
    )


def split_for(dataset: str, time_steps: int) -> TemporalSplit:
    if dataset == "synthetic":
        return synthetic_split(time_steps)
    if dataset == "elliptic":
        return ELLIPTIC_SPLIT
    raise SplitError(f"Kein Split fuer Datensatz {dataset} definiert")


def validate_split(split: TemporalSplit) -> None:
    for name, steps in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if steps.start < 1 or steps.start > steps.end:
            raise SplitError(f"Ungueltiger Bereich fuer {name}: {steps.start}-{steps.end}")
    if split.train.end >= split.test.start:
        raise SplitError("Der letzte Trainingsschritt muss vor dem ersten Testschritt liegen")
    if split.validation.start < split.train.start or split.validation.end > split.train.end:
        raise SplitError("Der Validierungsteil muss im Trainingszeitraum liegen")


def split_masks(nodes: pd.DataFrame, split: TemporalSplit) -> dict[str, pd.Series]:
    steps = nodes["time_step"]
    return {
        "train": steps.between(split.train.start, split.train.end),
        "validation": steps.between(split.validation.start, split.validation.end),
        "test": steps.between(split.test.start, split.test.end),
    }


def steps_on_both_sides(nodes: pd.DataFrame, split: TemporalSplit) -> list[int]:
    masks = split_masks(nodes, split)
    train_steps = set(nodes.loc[masks["train"], "time_step"])
    test_steps = set(nodes.loc[masks["test"], "time_step"])
    return sorted(int(step) for step in train_steps & test_steps)


def cross_split_edges(nodes: pd.DataFrame, edges: pd.DataFrame, split: TemporalSplit) -> int:
    masks = split_masks(nodes, split)
    train_ids = set(nodes.loc[masks["train"], "node_id"])
    test_ids = set(nodes.loc[masks["test"], "node_id"])
    source_train = edges["source"].isin(train_ids)
    source_test = edges["source"].isin(test_ids)
    target_train = edges["target"].isin(train_ids)
    target_test = edges["target"].isin(test_ids)
    return int(((source_train & target_test) | (source_test & target_train)).sum())


def check_split(nodes: pd.DataFrame, edges: pd.DataFrame, split: TemporalSplit) -> int:
    validate_split(split)
    shared = steps_on_both_sides(nodes, split)
    if shared:
        raise SplitError(f"Zeitschritte auf beiden Seiten: {shared}")
    crossing = cross_split_edges(nodes, edges, split)
    if crossing:
        raise SplitError(f"{crossing} Kanten verbinden Trainings- und Testknoten")
    masks = split_masks(nodes, split)
    if not masks["train"].any() or not masks["test"].any():
        raise SplitError("Trainings- oder Testteil ist leer")
    return crossing
