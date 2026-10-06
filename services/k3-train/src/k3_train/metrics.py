import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

POSITIVE_LABEL = "illicit"
NEGATIVE_LABEL = "licit"
EXCLUDED_LABEL = "unknown"
TOP_SHARE = 0.02
TARGET_LEVEL = 0.5
CURVE_POINTS = 101
RANDOM_PERMUTATIONS = 10_000
RANDOM_QUANTILE = 0.95
RANDOM_CHUNK = 1000
ACCURACY_THRESHOLD_TEXT = (
    "oberste 2 % der Test-Scores (gelabelte Testknoten) gelten als auffällig, "
    "Gleichstände an der Schwelle eingeschlossen"
)


def _validated(y_true: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.size != s.size:
        raise ValueError("Labels und Scores muessen gleich lange Vektoren sein")
    if y.size == 0:
        raise ValueError("Keine Beobachtungen")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels muessen 0/1 oder bool sein")
    if not np.isfinite(s).all():
        raise ValueError("Scores muessen endlich sein")
    y = y.astype(bool)
    if not y.any():
        raise ValueError("Ohne positive Beobachtung ist PR-AUC nicht definiert")
    return y, s


def pr_auc(y_true: Any, scores: Any) -> float:
    y, s = _validated(y_true, scores)
    return float(average_precision_score(y, s))


def _operating_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(y, s)
    return precision[:-1], recall[:-1]


def precision_at_recall(y_true: Any, scores: Any, min_recall: float = TARGET_LEVEL) -> float:
    y, s = _validated(y_true, scores)
    precision, recall = _operating_points(y, s)
    reachable = recall >= min_recall
    return float(precision[reachable].max()) if reachable.any() else 0.0


def recall_at_precision(y_true: Any, scores: Any, min_precision: float = TARGET_LEVEL) -> float:
    y, s = _validated(y_true, scores)
    precision, recall = _operating_points(y, s)
    reachable = precision >= min_precision
    return float(recall[reachable].max()) if reachable.any() else 0.0


def top_share_flags(scores: Any, share: float = TOP_SHARE) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    count = max(1, math.ceil(round(share * s.size, 9)))
    threshold = np.sort(s)[::-1][count - 1]
    return s >= threshold


def accuracy_at_top_share(y_true: Any, scores: Any, share: float = TOP_SHARE) -> float:
    y, s = _validated(y_true, scores)
    return float(np.mean(top_share_flags(s, share) == y))


def top_share_counts(y_true: Any, scores: Any, share: float = TOP_SHARE) -> tuple[int, int]:
    y, s = _validated(y_true, scores)
    flags = top_share_flags(s, share)
    return int(flags.sum()), int((flags & y).sum())


def expected_random_pr_auc(positives: int, total: int) -> float:
    if not 1 <= positives <= total:
        raise ValueError("Erwartet 1 <= positive Beobachtungen <= Gesamtzahl")
    if total == 1:
        return 1.0
    ranks = np.arange(1, total + 1, dtype=np.float64)
    hit_rate = 1.0 + (ranks - 1.0) * (positives - 1) / (total - 1)
    return float(np.sum(hit_rate / ranks) / total)


def ranked_average_precision(ranked_labels: Any) -> np.ndarray:
    labels = np.atleast_2d(np.asarray(ranked_labels)).astype(bool)
    positives = labels.sum(axis=1)
    if (positives == 0).any():
        raise ValueError("Ohne positive Beobachtung ist PR-AUC nicht definiert")
    ranks = np.arange(1, labels.shape[1] + 1)
    precision = np.cumsum(labels, axis=1) / ranks
    return (precision * labels).sum(axis=1) / positives


def random_ranking_pr_aucs(
    y_true: Any, permutations: int = RANDOM_PERMUTATIONS, seed: int = 0
) -> np.ndarray:
    y = np.asarray(y_true).astype(bool)
    if y.ndim != 1 or not y.any():
        raise ValueError("Ohne positive Beobachtung ist PR-AUC nicht definiert")
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, permutations, RANDOM_CHUNK):
        size = min(RANDOM_CHUNK, permutations - start)
        shuffled = rng.permuted(np.tile(y, (size, 1)), axis=1)
        values.append(ranked_average_precision(shuffled))
    return np.concatenate(values)


def all_negative_accuracy(y_true: Any) -> float:
    y = np.asarray(y_true).astype(bool)
    return float(np.mean(~y))


def prevalence(y_true: Any) -> float:
    y = np.asarray(y_true).astype(bool)
    return float(np.mean(y))


def thin_indices(length: int, max_points: int = CURVE_POINTS) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.round(np.linspace(0, length - 1, max_points)).astype(np.int64))


def pr_curve_points(
    y_true: Any, scores: Any, max_points: int = CURVE_POINTS
) -> list[dict[str, float]]:
    y, s = _validated(y_true, scores)
    precision, recall, _ = precision_recall_curve(y, s, drop_intermediate=True)
    precision = precision[:-1][::-1]
    recall = recall[:-1][::-1]
    return [
        {"recall": float(recall[index]), "precision": float(precision[index])}
        for index in thin_indices(recall.size, max_points)
    ]


def evaluation_target(labels: pd.Series, test_mask: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    evaluated = (test_mask & (labels != EXCLUDED_LABEL)).to_numpy()
    y = (labels[evaluated] == POSITIVE_LABEL).to_numpy()
    return evaluated, y


def evaluation_summary(
    y_true: Any, seed: int, permutations: int = RANDOM_PERMUTATIONS
) -> dict[str, Any]:
    y = np.asarray(y_true).astype(bool)
    positives = int(y.sum())
    random_values = random_ranking_pr_aucs(y, permutations=permutations, seed=seed)
    return {
        "positiveLabel": POSITIVE_LABEL,
        "excludedLabel": EXCLUDED_LABEL,
        "testPositives": positives,
        "testNegatives": int(y.size - positives),
        "prevalence": prevalence(y),
        "allLicitAccuracy": all_negative_accuracy(y),
        "randomPrAucExpected": expected_random_pr_auc(positives, int(y.size)),
        "randomPrAucQ95": float(np.quantile(random_values, RANDOM_QUANTILE)),
        "randomPermutations": permutations,
    }


def evaluate_scores(y_true: Any, scores: Any) -> dict[str, Any]:
    flagged, true_positives = top_share_counts(y_true, scores)
    return {
        "prAuc": pr_auc(y_true, scores),
        "precisionAtRecall50": precision_at_recall(y_true, scores),
        "recallAtPrecision50": recall_at_precision(y_true, scores),
        "accuracy": accuracy_at_top_share(y_true, scores),
        "accuracyThreshold": ACCURACY_THRESHOLD_TEXT,
        "accuracyFlagged": flagged,
        "accuracyTruePositives": true_positives,
        "prCurve": pr_curve_points(y_true, scores),
    }
