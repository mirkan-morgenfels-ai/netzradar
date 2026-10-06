from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

MAD_SCALE = 1.4826
IFOREST_TREES = 200
IFOREST_CONTAMINATION = 0.02


@dataclass(frozen=True)
class RobustStats:
    median: np.ndarray
    mad: np.ndarray


@dataclass(frozen=True)
class MethodSpec:
    method: str
    display_name: str
    feature_set: str


ZSCORE = MethodSpec(
    method="zscore",
    display_name="Robuste Z-Scores (Einzelmerkmale)",
    feature_set="local",
)
IFOREST = MethodSpec(
    method="iforest",
    display_name="Isolation Forest (Einzelmerkmale + Graphmaße)",
    feature_set="local+graph",
)
BASELINE_METHODS = (ZSCORE, IFOREST)


def fit_robust_stats(values: np.ndarray) -> RobustStats:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Erwartet eine nicht leere Matrix")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    return RobustStats(median=median, mad=mad)


def robust_z(values: np.ndarray, stats: RobustStats) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = MAD_SCALE * stats.mad
    usable = scale > 0
    z = np.zeros_like(values)
    z[:, usable] = (values[:, usable] - stats.median[usable]) / scale[usable]
    return z


def zscore_scores(values: np.ndarray, stats: RobustStats) -> np.ndarray:
    return np.abs(robust_z(values, stats)).max(axis=1)


def zscore_for_split(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, RobustStats]:
    stats = fit_robust_stats(values[np.asarray(train_mask, dtype=bool)])
    return zscore_scores(values, stats), stats


def fit_isolation_forest(
    train_values: np.ndarray,
    seed: int,
    n_estimators: int = IFOREST_TREES,
    contamination: float = IFOREST_CONTAMINATION,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(np.asarray(train_values, dtype=np.float64))
    return model


def isolation_forest_scores(model: IsolationForest, values: np.ndarray) -> np.ndarray:
    return -model.score_samples(np.asarray(values, dtype=np.float64))


def iforest_for_split(values: np.ndarray, train_mask: np.ndarray, seed: int) -> np.ndarray:
    model = fit_isolation_forest(values[np.asarray(train_mask, dtype=bool)], seed=seed)
    return isolation_forest_scores(model, values)


def zero_mad_features(features: list[str], stats: RobustStats) -> list[str]:
    if len(features) != stats.mad.size:
        raise ValueError("Merkmalsnamen und Statistik haben unterschiedliche Länge")
    return [name for name, mad in zip(features, stats.mad, strict=True) if mad == 0]


def zscore_hyperparameters(features: list[str], stats: RobustStats) -> dict[str, Any]:
    return {
        "center": "median",
        "scale": "mad",
        "madScale": MAD_SCALE,
        "madZeroContribution": 0,
        "aggregation": "maxAbs",
        "fitOn": "train",
        "features": features,
        "zeroMadFeatures": zero_mad_features(features, stats),
    }


def iforest_hyperparameters(
    features: list[str], seed: int, graph_measures: dict[str, Any]
) -> dict[str, Any]:
    return {
        "nEstimators": IFOREST_TREES,
        "contamination": IFOREST_CONTAMINATION,
        "maxSamples": "auto",
        "maxFeatures": 1.0,
        "bootstrap": False,
        "randomState": seed,
        "score": "-score_samples",
        "fitOn": "train",
        "labelsUsed": False,
        "features": features,
        "graphMeasures": graph_measures,
    }
