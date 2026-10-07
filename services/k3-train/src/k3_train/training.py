import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from k3_train.baseline import MAD_SCALE
from k3_train.jsonio import FLOAT_DIGITS
from k3_train.metrics import EXCLUDED_LABEL, POSITIVE_LABEL
from k3_train.split import SplitError, TemporalSplit, split_masks

CLIP = 10.0
TRAIN_RATIO = "trainRatio"
FIXED = "fixed"
FIXED_POSITIVE_WEIGHT = 20.0
SCALE_RULES = ("mad", "std", "one")


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray
    rules: tuple[str, ...]
    clip: float


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    weight_rule: str
    fixed_weight: float | None = None

    def weight(self, targets: Any) -> float:
        if self.weight_rule == TRAIN_RATIO:
            return positive_weight(targets)
        if self.weight_rule == FIXED and self.fixed_weight is not None:
            return float(self.fixed_weight)
        raise ValueError(f"Unbekannte Gewichtsregel: {self.weight_rule}")


SEARCH_SPACE = (
    Candidate("local", TRAIN_RATIO),
    Candidate("local", FIXED, FIXED_POSITIVE_WEIGHT),
    Candidate("local+graph", TRAIN_RATIO),
    Candidate("local+graph", FIXED, FIXED_POSITIVE_WEIGHT),
)


@dataclass
class EarlyStopping:
    patience: int
    best_epoch: int = 0
    best_value: float = -math.inf
    last_epoch: int = 0

    def update(self, epoch: int, value: float) -> bool:
        self.last_epoch = epoch
        if value > self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            return True
        return False

    @property
    def stop(self) -> bool:
        return self.last_epoch - self.best_epoch >= self.patience


def training_parts(nodes: pd.DataFrame, split: TemporalSplit) -> dict[str, np.ndarray]:
    masks = split_masks(nodes, split)
    train = masks["train"].to_numpy()
    validation = masks["validation"].to_numpy()
    return {
        "fit": train & ~validation,
        "validation": validation,
        "train": train,
        "test": masks["test"].to_numpy(),
    }


def loss_targets(labels: Any, mask: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(str)
    mask = np.asarray(mask, dtype=bool)
    if labels.shape != mask.shape:
        raise ValueError("Labels und Maske haben unterschiedliche Länge")
    index = np.flatnonzero(mask & (labels != EXCLUDED_LABEL)).astype(np.int64)
    targets = (labels[index] == POSITIVE_LABEL).astype(np.int64)
    return index, targets


def positive_weight(targets: Any) -> float:
    targets = np.asarray(targets)
    positives = int((targets == 1).sum())
    negatives = int((targets == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            "Das Klassengewicht braucht illicit- und licit-Knoten in der Verlustmenge "
            f"(gefunden: {positives} illicit, {negatives} licit)"
        )
    return negatives / positives


def fit_scaler(values: Any, mask: Any, clip: float = CLIP) -> RobustScaler:
    rows = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("Erwartet eine nicht leere Matrix der Trainingszeilen")
    center = np.median(rows, axis=0)
    mad = np.median(np.abs(rows - center), axis=0)
    spread = rows.std(axis=0)
    scale = np.where(mad > 0, MAD_SCALE * mad, np.where(spread > 0, spread, 1.0))
    rules = tuple(
        SCALE_RULES[0] if m > 0 else SCALE_RULES[1] if s > 0 else SCALE_RULES[2]
        for m, s in zip(mad, spread, strict=True)
    )
    return RobustScaler(center=center, scale=scale, rules=rules, clip=clip)


def transform(values: Any, scaler: RobustScaler) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.clip((values - scaler.center) / scaler.scale, -scaler.clip, scaler.clip)


def scaler_hyperparameters(
    features: list[str], scaler: RobustScaler, fit_on: str
) -> dict[str, Any]:
    if len(features) != len(scaler.rules):
        raise ValueError("Merkmalsnamen und Skalierung haben unterschiedliche Länge")
    return {
        "center": "median",
        "scale": "mad",
        "madScale": MAD_SCALE,
        "fallback": list(SCALE_RULES[1:]),
        "clip": scaler.clip,
        "fitOn": fit_on,
        "zeroMadFeatures": [
            name for name, rule in zip(features, scaler.rules, strict=True) if rule != "mad"
        ],
        "unitScaleFeatures": [
            name for name, rule in zip(features, scaler.rules, strict=True) if rule == "one"
        ],
    }


def select_candidate(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("Keine Kandidaten")
    published = [round(float(value), FLOAT_DIGITS) for value in values]
    best = 0
    for index, value in enumerate(published):
        if value > published[best]:
            best = index
    return best


def fit_validation_edges(
    nodes: pd.DataFrame, edges: pd.DataFrame, fit: Any, validation: Any
) -> int:
    ids = nodes["node_id"].to_numpy()
    fit_ids = set(ids[np.asarray(fit, dtype=bool)])
    validation_ids = set(ids[np.asarray(validation, dtype=bool)])
    source_fit = edges["source"].isin(fit_ids)
    source_validation = edges["source"].isin(validation_ids)
    target_fit = edges["target"].isin(fit_ids)
    target_validation = edges["target"].isin(validation_ids)
    crossing = (source_fit & target_validation) | (source_validation & target_fit)
    return int(crossing.sum())


def check_fit_validation_edges(
    nodes: pd.DataFrame, edges: pd.DataFrame, parts: dict[str, np.ndarray]
) -> int:
    crossing = fit_validation_edges(nodes, edges, parts["fit"], parts["validation"])
    if crossing:
        raise SplitError(f"{crossing} Kanten verbinden Fit- und Validierungsknoten")
    return crossing
