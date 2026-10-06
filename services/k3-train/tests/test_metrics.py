import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from k3_train.metrics import (
    accuracy_at_top_share,
    all_negative_accuracy,
    evaluate_scores,
    evaluation_summary,
    evaluation_target,
    expected_random_pr_auc,
    pr_auc,
    pr_curve_points,
    precision_at_recall,
    prevalence,
    random_ranking_pr_aucs,
    ranked_average_precision,
    recall_at_precision,
    thin_indices,
    top_share_counts,
    top_share_flags,
)

Y_HAND = [1, 0, 1, 0]
S_HAND = [0.9, 0.8, 0.7, 0.1]
Y_WEAK = [0, 0, 1, 0, 0]
S_WEAK = [0.9, 0.8, 0.3, 0.2, 0.1]
Y_TIED = [1, 0, 1, 0]
S_TIED = [0.9, 0.9, 0.5, 0.1]


def test_pr_auc_hand_example() -> None:
    assert pr_auc(Y_HAND, S_HAND) == pytest.approx(0.5 * 1.0 + 0.5 * (2.0 / 3.0))
    assert pr_auc(Y_HAND, S_HAND) == pytest.approx(0.8333, abs=1e-4)


def test_precision_at_recall_hand_example() -> None:
    assert precision_at_recall(Y_HAND, S_HAND) == pytest.approx(1.0)


def test_recall_at_precision_hand_example() -> None:
    assert recall_at_precision(Y_HAND, S_HAND) == pytest.approx(1.0)


def test_tied_scores_form_one_operating_point_by_hand() -> None:
    assert pr_auc(Y_TIED, S_TIED) == pytest.approx(0.5 * 0.5 + 0.5 * (2.0 / 3.0))
    assert pr_auc(Y_TIED, S_TIED) == pytest.approx(0.5833, abs=1e-4)
    assert precision_at_recall(Y_TIED, S_TIED) == pytest.approx(2.0 / 3.0)
    assert recall_at_precision(Y_TIED, S_TIED) == pytest.approx(1.0)


def test_unreachable_precision_gives_zero_recall() -> None:
    assert recall_at_precision(Y_WEAK, S_WEAK) == 0.0
    assert precision_at_recall(Y_WEAK, S_WEAK) == pytest.approx(1.0 / 3.0)
    assert pr_auc(Y_WEAK, S_WEAK) == pytest.approx(1.0 / 3.0)


def test_precision_at_recall_picks_best_threshold() -> None:
    y = [1, 0, 0, 1, 1, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert precision_at_recall(y, s) == pytest.approx(3.0 / 5.0)
    assert recall_at_precision(y, s) == pytest.approx(1.0)
    assert pr_auc(y, s) == pytest.approx((1.0 / 3.0) * 1.0 + (1.0 / 3.0) * 0.5 + (1.0 / 3.0) * 0.6)


def test_accuracy_of_top_two_percent_by_hand() -> None:
    scores = np.arange(100, 0, -1, dtype=float)
    y = np.zeros(100, dtype=int)
    y[[0, 50, 99]] = 1
    flags = top_share_flags(scores)
    assert flags.sum() == 2
    assert flags[:2].all()
    assert accuracy_at_top_share(y, scores) == pytest.approx(0.97)


def test_top_share_includes_ties_at_threshold() -> None:
    scores = np.zeros(50)
    scores[[3, 7]] = 1.0
    assert top_share_flags(scores).sum() == 2


def test_flagged_nodes_and_hits_by_hand() -> None:
    scores = np.arange(100, 0, -1, dtype=float)
    y = np.zeros(100, dtype=int)
    y[[0, 50, 99]] = 1
    assert top_share_counts(y, scores) == (2, 1)


def test_ties_at_threshold_flag_more_nodes_by_hand() -> None:
    scores = np.zeros(100)
    scores[0] = 5.0
    scores[[1, 2, 3]] = 4.0
    y = np.zeros(100, dtype=int)
    y[[0, 2, 60]] = 1
    assert top_share_counts(y, scores) == (4, 2)
    assert accuracy_at_top_share(y, scores) == pytest.approx((100 - 2 - 1) / 100)


def test_all_licit_accuracy_and_prevalence_by_hand() -> None:
    y = [1, 0, 0, 0]
    assert all_negative_accuracy(y) == pytest.approx(0.75)
    assert prevalence(y) == pytest.approx(0.25)
    summary = evaluation_summary([1, 0, 0, 0, 0], seed=42, permutations=2000)
    assert summary == {
        "positiveLabel": "illicit",
        "excludedLabel": "unknown",
        "testPositives": 1,
        "testNegatives": 4,
        "prevalence": pytest.approx(0.2),
        "allLicitAccuracy": pytest.approx(0.8),
        "randomPrAucExpected": pytest.approx(137.0 / 300.0),
        "randomPrAucQ95": pytest.approx(1.0),
        "randomPermutations": 2000,
    }


def test_prevalence_equals_pr_auc_of_constant_score() -> None:
    y = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert pr_auc(y, np.zeros(10)) == pytest.approx(prevalence(y))


def test_expected_pr_auc_of_random_ranking_by_hand() -> None:
    assert expected_random_pr_auc(1, 1) == 1.0
    assert expected_random_pr_auc(1, 2) == pytest.approx((1.0 + 0.5) / 2.0)
    assert expected_random_pr_auc(2, 3) == pytest.approx((1.0 + 5.0 / 6.0 + 7.0 / 12.0) / 3.0)
    assert expected_random_pr_auc(1, 5) == pytest.approx(137.0 / 300.0)
    assert expected_random_pr_auc(95, 852) == pytest.approx(0.1181, abs=1e-4)
    assert expected_random_pr_auc(95, 852) > prevalence([1] * 95 + [0] * 757)


def test_ranked_average_precision_by_hand() -> None:
    values = ranked_average_precision([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    assert values.tolist() == pytest.approx([(1.0 + 2.0 / 3.0) / 2.0, 0.5, (1.0 / 3.0 + 0.5) / 2.0])
    assert values[0] == pytest.approx(average_precision_score(Y_HAND, S_HAND))


def test_random_rankings_match_the_expected_value() -> None:
    values = random_ranking_pr_aucs([1, 0, 1], permutations=30_000, seed=1)
    assert set(np.round(values, 4)) == {1.0, 0.8333, 0.5833}
    assert values.mean() == pytest.approx(expected_random_pr_auc(2, 3), abs=0.01)
    again = random_ranking_pr_aucs([1, 0, 1], permutations=30_000, seed=1)
    np.testing.assert_array_equal(values, again)


def test_unknown_labels_are_excluded_from_evaluation() -> None:
    labels = pd.Series(["illicit", "unknown", "licit", "licit", "illicit"])
    test_mask = pd.Series([True, True, True, False, False])
    evaluated, y = evaluation_target(labels, test_mask)
    assert evaluated.tolist() == [True, False, True, False, False]
    assert y.tolist() == [True, False]


def test_hand_curve_starts_at_first_operating_point_and_ends_at_full_recall() -> None:
    points = pr_curve_points(Y_HAND, S_HAND)
    assert points == [
        {"recall": 0.5, "precision": 1.0},
        {"recall": 0.5, "precision": 0.5},
        {"recall": 1.0, "precision": pytest.approx(2.0 / 3.0)},
        {"recall": 1.0, "precision": 0.5},
    ]


def test_curve_has_no_artificial_end_point() -> None:
    points = pr_curve_points(Y_WEAK, S_WEAK)
    assert points[0] == {"recall": 0.0, "precision": 0.0}
    assert {"recall": 0.0, "precision": 1.0} not in points


def test_long_curve_is_thinned_to_at_most_101_points() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.1
    scores = rng.random(5000) + y * 0.3
    points = pr_curve_points(y, scores)
    assert 2 <= len(points) <= 101
    recalls = [point["recall"] for point in points]
    assert recalls == sorted(recalls)
    assert 0.0 <= recalls[0] <= 0.01
    assert recalls[-1] == 1.0


def test_thin_indices_keep_both_ends() -> None:
    assert thin_indices(5).tolist() == [0, 1, 2, 3, 4]
    indices = thin_indices(1000)
    assert len(indices) == 101
    assert indices[0] == 0
    assert indices[-1] == 999


def test_evaluate_scores_has_contract_keys() -> None:
    result = evaluate_scores(Y_HAND, S_HAND)
    assert list(result) == [
        "prAuc",
        "precisionAtRecall50",
        "recallAtPrecision50",
        "accuracy",
        "accuracyThreshold",
        "accuracyFlagged",
        "accuracyTruePositives",
        "prCurve",
    ]
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["accuracyFlagged"] == 1
    assert result["accuracyTruePositives"] == 1


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        pr_auc([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="gleich lange"):
        pr_auc([1, 0], [0.1])
    with pytest.raises(ValueError, match="0/1"):
        pr_auc(["illicit", "licit"], [0.1, 0.2])
