import numpy as np
import pytest

from k3_train.baseline import (
    MAD_SCALE,
    fit_isolation_forest,
    fit_robust_stats,
    iforest_for_split,
    isolation_forest_scores,
    robust_z,
    zero_mad_features,
    zscore_for_split,
    zscore_hyperparameters,
    zscore_scores,
)


def test_robust_z_with_outlier_by_hand() -> None:
    train = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    stats = fit_robust_stats(train)
    assert stats.median.tolist() == [3.0]
    assert stats.mad.tolist() == [1.0]
    z = robust_z(train, stats)[:, 0]
    assert z[4] == pytest.approx(65.4256, abs=1e-4)
    assert z[0] == pytest.approx(-1.3490, abs=1e-4)
    assert z[2] == 0.0
    assert MAD_SCALE == 1.4826


def test_feature_with_zero_mad_contributes_nothing() -> None:
    train = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0], [9.0, 5.0]])
    stats = fit_robust_stats(train)
    assert stats.mad.tolist() == [0.0, 1.0]
    probe = np.array([[1000.0, 3.0], [5.0, 6.0]])
    z = robust_z(probe, stats)
    assert z[0].tolist() == [0.0, 0.0]
    scores = zscore_scores(probe, stats)
    assert scores[0] == 0.0
    assert scores[1] == pytest.approx(2.0235, abs=1e-4)


def test_zero_mad_features_are_named_in_the_hyperparameters() -> None:
    train = np.array([[5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 1.0], [9.0, 4.0, 0.0]])
    stats = fit_robust_stats(train)
    assert stats.mad.tolist() == [0.0, 1.0, 0.0]
    names = ["f_constant", "f_spread", "f_binary"]
    assert zero_mad_features(names, stats) == ["f_constant", "f_binary"]
    hyperparameters = zscore_hyperparameters(names, stats)
    assert hyperparameters["features"] == names
    assert hyperparameters["zeroMadFeatures"] == ["f_constant", "f_binary"]
    with pytest.raises(ValueError, match="Länge"):
        zero_mad_features(names[:2], stats)


def test_score_is_maximum_absolute_z() -> None:
    train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    stats = fit_robust_stats(train)
    scores = zscore_scores(np.array([[3.0, -4.0]]), stats)
    assert scores[0] == pytest.approx(5.0 / 1.4826)


def test_zscore_normalisation_uses_only_training_rows() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 3))
    train = np.arange(40) < 30
    scores, stats = zscore_for_split(values, train)
    reference = fit_robust_stats(values[train])
    np.testing.assert_array_equal(stats.median, reference.median)
    np.testing.assert_array_equal(stats.mad, reference.mad)
    shifted = values.copy()
    shifted[~train] = shifted[~train] * 1000.0 + 50.0
    shifted_scores, shifted_stats = zscore_for_split(shifted, train)
    np.testing.assert_array_equal(shifted_stats.median, stats.median)
    np.testing.assert_array_equal(shifted_stats.mad, stats.mad)
    np.testing.assert_array_equal(shifted_scores[train], scores[train])


def test_isolation_forest_is_deterministic_for_equal_seed() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(300, 4))
    train = np.arange(300) < 200
    first = iforest_for_split(values, train, seed=42)
    second = iforest_for_split(values, train, seed=42)
    other = iforest_for_split(values, train, seed=7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_isolation_forest_is_fitted_on_training_rows_only() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(300, 4))
    train = np.arange(300) < 200
    scores = iforest_for_split(values, train, seed=42)
    changed = values.copy()
    changed[~train] = 1e6
    changed_scores = iforest_for_split(changed, train, seed=42)
    np.testing.assert_array_equal(scores[train], changed_scores[train])


def test_isolation_forest_score_is_negative_score_samples() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(100, 2))
    model = fit_isolation_forest(values, seed=42)
    assert model.n_estimators == 200
    assert model.contamination == 0.02
    np.testing.assert_array_equal(
        isolation_forest_scores(model, values), -model.score_samples(values)
    )
