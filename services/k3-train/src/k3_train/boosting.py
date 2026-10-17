import platform
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from k3_train.features import feature_matrix
from k3_train.metrics import evaluate_scores, pr_auc, seed_spread_summary
from k3_train.split import TemporalSplit
from k3_train.training import (
    FIXED,
    TRAIN_RATIO,
    Candidate,
    loss_targets,
    select_candidate,
    training_parts,
)

HGB_METHOD = "hgb"
HGB_DISPLAY_NAME = "Gradient Boosting (ohne Kanten)"
ESTIMATOR = "HistGradientBoostingClassifier"
LOSS = "log_loss"
SCORE_TEXT = "logOddsIllicit"
SELECTION_METRIC = "validationPrAuc"
UNWEIGHTED = 1.0
LEAF_OPTIONS = (7, 31)
THREADS = 1


@dataclass(frozen=True)
class HgbCandidate:
    feature_set: str
    weight_rule: str
    max_leaf_nodes: int
    fixed_weight: float | None = None

    def weight(self, targets: Any) -> float:
        return Candidate(self.feature_set, self.weight_rule, self.fixed_weight).weight(targets)


HGB_SEARCH_SPACE = tuple(
    HgbCandidate(feature_set, rule, leaves, fixed)
    for feature_set in ("local", "local+graph")
    for rule, fixed in ((TRAIN_RATIO, None), (FIXED, UNWEIGHTED))
    for leaves in LEAF_OPTIONS
)


@dataclass(frozen=True)
class HgbConfig:
    learning_rate: float = 0.1
    max_iter: int = 300
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_bins: int = 255
    spread_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    search_space: tuple[HgbCandidate, ...] = HGB_SEARCH_SPACE


@dataclass(frozen=True)
class HgbSearchResult:
    candidate: HgbCandidate
    positive_weight: float
    validation_pr_auc: float
    best_iteration: int

    def as_dict(self, selected: bool) -> dict[str, Any]:
        return {
            "featureSet": self.candidate.feature_set,
            "positiveWeightRule": self.candidate.weight_rule,
            "positiveWeight": self.positive_weight,
            "maxLeafNodes": self.candidate.max_leaf_nodes,
            "validationPrAuc": self.validation_pr_auc,
            "bestIteration": self.best_iteration,
            "selected": selected,
        }


@dataclass(frozen=True)
class HgbResult:
    feature_set: str
    hyperparameters: dict[str, Any]
    scores: np.ndarray
    metrics: dict[str, Any]


def sample_weights(targets: Any, weight: float) -> np.ndarray:
    targets = np.asarray(targets)
    return np.where(targets == 1, float(weight), 1.0)


def fit_model(
    values: np.ndarray,
    index: np.ndarray,
    targets: np.ndarray,
    weight: float,
    max_leaf_nodes: int,
    max_iter: int,
    config: HgbConfig,
    seed: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        loss=LOSS,
        learning_rate=config.learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,
        max_features=1.0,
        early_stopping=False,
        random_state=seed,
    )
    with threadpool_limits(limits=THREADS, user_api="openmp"):
        model.fit(values[index], targets, sample_weight=sample_weights(targets, weight))
    return model


def log_odds(model: HistGradientBoostingClassifier, values: np.ndarray) -> np.ndarray:
    with threadpool_limits(limits=THREADS, user_api="openmp"):
        return np.asarray(model.decision_function(values), dtype=np.float64)


def validation_curve(
    model: HistGradientBoostingClassifier, values: np.ndarray, targets: np.ndarray
) -> list[float]:
    with threadpool_limits(limits=THREADS, user_api="openmp"):
        return [
            pr_auc(targets, np.asarray(scores, dtype=np.float64))
            for scores in model.staged_decision_function(values)
        ]


def best_iteration(curve: Sequence[float]) -> tuple[int, float]:
    if not curve:
        raise ValueError("Leere Validierungskurve")
    best = 0
    for index, value in enumerate(curve):
        if value > curve[best]:
            best = index
    return best + 1, float(curve[best])


def environment() -> dict[str, Any]:
    return {
        "scikitLearn": sklearn.__version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "threads": THREADS,
    }


def search_candidates(
    matrices: dict[str, np.ndarray],
    fit: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    config: HgbConfig,
    seed: int,
) -> list[HgbSearchResult]:
    results = []
    for candidate in config.search_space:
        values = matrices[candidate.feature_set]
        weight = candidate.weight(fit[1])
        model = fit_model(
            values,
            fit[0],
            fit[1],
            weight,
            candidate.max_leaf_nodes,
            config.max_iter,
            config,
            seed,
        )
        iteration, value = best_iteration(
            validation_curve(model, values[validation[0]], validation[1])
        )
        results.append(
            HgbSearchResult(
                candidate=candidate,
                positive_weight=weight,
                validation_pr_auc=value,
                best_iteration=iteration,
            )
        )
    return results


def hgb_hyperparameters(
    config: HgbConfig,
    search: list[HgbSearchResult],
    chosen: int,
    features: list[str],
    weight: float,
    split: TemporalSplit,
    seed: int,
    spread: dict[str, Any],
) -> dict[str, Any]:
    selected = search[chosen]
    return {
        "estimator": ESTIMATOR,
        "loss": LOSS,
        "score": SCORE_TEXT,
        "learningRate": config.learning_rate,
        "maxIter": config.max_iter,
        "maxLeafNodes": selected.candidate.max_leaf_nodes,
        "minSamplesLeaf": config.min_samples_leaf,
        "l2Regularization": config.l2_regularization,
        "maxBins": config.max_bins,
        "maxFeatures": 1.0,
        "earlyStopping": False,
        "randomState": seed,
        "positiveWeight": weight,
        "positiveWeightRule": selected.candidate.weight_rule,
        "edges": "none",
        "features": features,
        "selectedIteration": selected.best_iteration,
        "selectionMetric": SELECTION_METRIC,
        "validationPrAuc": selected.validation_pr_auc,
        "selectionSteps": {"from": split.train.start, "to": split.validation.start - 1},
        "validationSteps": split.validation.as_dict(),
        "finalFitSteps": split.train.as_dict(),
        "search": [result.as_dict(index == chosen) for index, result in enumerate(search)],
        "seedSpread": spread,
        "environment": environment(),
    }


def run_hgb(
    nodes: pd.DataFrame,
    measures: pd.DataFrame,
    split: TemporalSplit,
    evaluated: np.ndarray,
    y: np.ndarray,
    seed: int,
    config: HgbConfig | None = None,
) -> HgbResult:
    config = config or HgbConfig()
    parts = training_parts(nodes, split)
    labels = nodes["label"].to_numpy()
    fit = loss_targets(labels, parts["fit"])
    validation = loss_targets(labels, parts["validation"])
    train = loss_targets(labels, parts["train"])
    matrices: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}
    for feature_set in sorted({candidate.feature_set for candidate in config.search_space}):
        matrices[feature_set], names[feature_set] = feature_matrix(nodes, measures, feature_set)
    search = search_candidates(matrices, fit, validation, config, seed)
    chosen = select_candidate([result.validation_pr_auc for result in search])
    selected = search[chosen]
    values = matrices[selected.candidate.feature_set]
    weight = selected.candidate.weight(train[1])

    def final_scores(model_seed: int) -> np.ndarray:
        model = fit_model(
            values,
            train[0],
            train[1],
            weight,
            selected.candidate.max_leaf_nodes,
            selected.best_iteration,
            config,
            model_seed,
        )
        return log_odds(model, values)

    scores = final_scores(seed)
    metrics = evaluate_scores(y, scores[evaluated])
    spread_values = [
        metrics["prAuc"] if other == seed else pr_auc(y, final_scores(other)[evaluated])
        for other in config.spread_seeds
    ]
    spread = seed_spread_summary(config.spread_seeds, seed, spread_values)
    return HgbResult(
        feature_set=selected.candidate.feature_set,
        hyperparameters=hgb_hyperparameters(
            config,
            search,
            chosen,
            names[selected.candidate.feature_set],
            weight,
            split,
            seed,
            spread,
        ),
        scores=scores,
        metrics=metrics,
    )
