"""PURE-MATH tests for fl/norm_rule.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fl.norm_rule import compute_threshold, flag_anomalous, bootstrap_from_server_root, update_threshold_ema


def test_threshold_flags_obvious_outliers_not_the_benign_bulk():
    rng = np.random.default_rng(0)
    benign = rng.normal(loc=5.0, scale=0.5, size=200)
    attackers = np.array([50.0, 60.0, 45.0])  # a1_scaling-style blown-up norms
    norms = np.concatenate([benign, attackers])

    threshold = compute_threshold(norms, k=3.0)
    flags = flag_anomalous(norms, threshold)

    assert flags[-3:].all(), "all 3 injected outliers must be flagged"
    false_positive_rate = flags[:200].mean()
    assert false_positive_rate < 0.05, f"too many benign clients flagged: {false_positive_rate}"


def test_threshold_robust_to_large_minority_coalition():
    """Up to just under half malicious should not be able to drag the
    median far enough to hide themselves -- the whole point of using
    median/MAD instead of mean/std.
    """
    rng = np.random.default_rng(1)
    benign = rng.normal(loc=5.0, scale=0.5, size=26)   # 26 honest
    coalition = np.full(24, 40.0)                       # 24 colluding attackers (48%)
    norms = np.concatenate([benign, coalition])

    threshold = compute_threshold(norms, k=3.0)
    # median should still track the benign cluster, not the coalition
    assert abs(threshold.median - 5.0) < 1.0, f"median dragged to {threshold.median}"
    flags = flag_anomalous(norms, threshold)
    assert flags[26:].all(), "the colluding coalition must still be flagged"


def test_degenerate_identical_norms_does_not_explode():
    norms = np.full(50, 7.0)
    threshold = compute_threshold(norms)
    assert threshold.mad > 0  # floor kicked in
    assert threshold.B_t > threshold.median


def test_bootstrap_and_ema_update_chain():
    rng = np.random.default_rng(2)
    root_norms = rng.normal(5.0, 0.3, size=100)
    t0 = bootstrap_from_server_root(root_norms)

    round1_norms = rng.normal(4.0, 0.3, size=50)  # norms shrinking as training converges
    t1 = update_threshold_ema(t0, round1_norms, alpha=0.3)

    # blended median should sit between the two round medians
    assert min(t0.median, 4.0) - 0.5 <= t1.median <= max(t0.median, 4.0) + 0.5


def test_empty_norms_raises():
    try:
        compute_threshold(np.array([]))
        assert False, "should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_threshold_flags_obvious_outliers_not_the_benign_bulk()
    print("test_threshold_flags_obvious_outliers_not_the_benign_bulk PASSED")
    test_threshold_robust_to_large_minority_coalition()
    print("test_threshold_robust_to_large_minority_coalition PASSED")
    test_degenerate_identical_norms_does_not_explode()
    print("test_degenerate_identical_norms_does_not_explode PASSED")
    test_bootstrap_and_ema_update_chain()
    print("test_bootstrap_and_ema_update_chain PASSED")
    test_empty_norms_raises()
    print("test_empty_norms_raises PASSED")
    print("\nALL norm_rule TESTS PASSED")
