"""Pure-NumPy correctness tests for baselines/aggregation_rules.py --
run unconditionally, no torch/EC backend needed.
"""
import numpy as np
import pytest

from baselines.aggregation_rules import (
    AGGREGATION_RULES, coordinate_median, krum, mean_aggregate, multi_krum, trimmed_mean,
)

DIM = 12


def _honest_deltas(n, seed=0):
    rng = np.random.default_rng(seed)
    center = rng.normal(0, 1.0, size=DIM)
    return [center + rng.normal(0, 0.05, size=DIM) for _ in range(n)]


def test_mean_aggregate_matches_numpy_mean():
    deltas = _honest_deltas(5)
    got = mean_aggregate(deltas)
    want = np.mean(np.stack(deltas), axis=0)
    assert np.allclose(got, want)


def test_mean_aggregate_empty_raises():
    with pytest.raises(ValueError):
        mean_aggregate([])


def test_mean_aggregate_dragged_by_single_outlier():
    """The defense this file exists to contrast against everything
    else: one huge delta among many honest ones moves the plain mean
    arbitrarily far.
    """
    deltas = _honest_deltas(9)
    huge = np.full(DIM, 1e6)
    deltas.append(huge)
    got = mean_aggregate(deltas)
    assert np.max(np.abs(got)) > 1e4, "plain mean should be dragged hugely by the one outlier"


def test_coordinate_median_robust_to_one_outlier():
    deltas = _honest_deltas(9)
    honest_mean = mean_aggregate(deltas)
    huge = np.full(DIM, 1e6)
    deltas_with_outlier = deltas + [huge]
    got = coordinate_median(deltas_with_outlier)
    # median of 10 honest-ish values + 1 huge one stays close to the honest cluster
    assert np.allclose(got, honest_mean, atol=1.0), (
        f"coordinate median should be near the honest cluster even with one huge outlier, "
        f"got max abs diff {np.max(np.abs(got - honest_mean))}"
    )


def test_coordinate_median_empty_raises():
    with pytest.raises(ValueError):
        coordinate_median([])


def test_trimmed_mean_drops_tails():
    # 10 clients: 8 honest clustered near 0, 2 wild outliers (one very
    # negative, one very positive) -- trim_fraction=0.2 drops exactly
    # 2 (one from each tail) per coordinate, which are exactly the two
    # outliers since they sit at the extremes on every coordinate.
    rng = np.random.default_rng(1)
    honest = [rng.normal(0, 0.01, size=DIM) for _ in range(8)]
    deltas = honest + [np.full(DIM, -1000.0), np.full(DIM, 1000.0)]
    got = trimmed_mean(deltas, trim_fraction=0.2)
    assert np.max(np.abs(got)) < 1.0, (
        f"trimmed mean should have dropped both outliers, got max abs {np.max(np.abs(got))}"
    )


def test_trimmed_mean_zero_trim_equals_mean():
    deltas = _honest_deltas(5)
    assert np.allclose(trimmed_mean(deltas, trim_fraction=0.0), mean_aggregate(deltas))


def test_trimmed_mean_invalid_fraction_raises():
    with pytest.raises(ValueError):
        trimmed_mean(_honest_deltas(5), trim_fraction=0.5)
    with pytest.raises(ValueError):
        trimmed_mean(_honest_deltas(5), trim_fraction=-0.1)


def test_trimmed_mean_empty_raises():
    with pytest.raises(ValueError):
        trimmed_mean([])


def test_krum_selects_an_honest_client_not_the_outlier():
    deltas = _honest_deltas(9)
    outlier_idx = len(deltas)
    deltas.append(np.full(DIM, 1e6))
    selected, idx = krum(deltas, n_byzantine=1)
    assert idx != outlier_idx, "Krum selected the obvious outlier -- defense failure"
    assert np.max(np.abs(selected)) < 10, "Krum's selection should be one of the honest, clustered deltas"


def test_krum_insufficient_clients_raises():
    with pytest.raises(ValueError):
        krum(_honest_deltas(4), n_byzantine=1)  # needs n >= 2*1+3 = 5


def test_multi_krum_averages_m_honest_selections():
    deltas = _honest_deltas(9)
    deltas.append(np.full(DIM, 1e6))  # 1 outlier among 10
    got = multi_krum(deltas, n_byzantine=1, m=5)
    assert np.max(np.abs(got)) < 10, "Multi-Krum's average should stay near the honest cluster"


def test_multi_krum_m_too_large_raises():
    with pytest.raises(ValueError):
        multi_krum(_honest_deltas(5), n_byzantine=1, m=5)  # m > n - n_byz = 4


def test_all_registered_rules_runnable_on_same_input():
    deltas = _honest_deltas(9)
    deltas.append(np.full(DIM, 50.0))  # mild outlier, keeps Krum's distance math well-conditioned
    for name, fn in AGGREGATION_RULES.items():
        out = fn(deltas, 1)
        assert out.shape == (DIM,), f"{name} returned wrong shape: {out.shape}"
        assert np.all(np.isfinite(out)), f"{name} returned non-finite values"
