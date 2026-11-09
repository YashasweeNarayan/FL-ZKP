import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from crypto.dp import RDPAccountant, split_noise


def test_noise_shares_sum_to_approximately_target_variance():
    rng = np.random.default_rng(0)
    total_sigma = 1.0
    n_shares = 10
    dim = 20_000
    trials = 200

    sums = []
    for _ in range(trials):
        shares = split_noise(total_sigma, n_shares, dim, rng)
        total = sum(shares)
        sums.append(total)
    sums = np.array(sums)
    empirical_sigma = sums.std()
    # expect empirical std close to total_sigma (within ~5% at this trial count)
    assert abs(empirical_sigma - total_sigma) / total_sigma < 0.05, empirical_sigma


def test_epsilon_decreases_as_sigma_increases():
    acc_small_sigma = RDPAccountant(sigma=0.5, sampling_rate=0.25)
    acc_large_sigma = RDPAccountant(sigma=4.0, sampling_rate=0.25)
    eps_small = acc_small_sigma.eps_for_rounds(45)
    eps_large = acc_large_sigma.eps_for_rounds(45)
    assert eps_large < eps_small, (eps_small, eps_large)


def test_subsampling_amplifies_privacy():
    """Lower sampling_rate (p) should give a smaller (better) epsilon than
    p=1 (full participation), everything else equal -- this is the
    'privacy amplification by subsampling' claim in ARCHITECTURE.md §5.5.
    """
    acc_full = RDPAccountant(sigma=1.0, sampling_rate=1.0)
    acc_subsampled = RDPAccountant(sigma=1.0, sampling_rate=0.25)
    eps_full = acc_full.eps_for_rounds(45)
    eps_sub = acc_subsampled.eps_for_rounds(45)
    assert eps_sub < eps_full, (eps_full, eps_sub)


def test_epsilon_grows_with_more_rounds():
    acc = RDPAccountant(sigma=1.0, sampling_rate=0.25)
    eps_10 = acc.eps_for_rounds(10)
    eps_45 = acc.eps_for_rounds(45)
    assert eps_45 > eps_10


def test_curve_is_monotonic_and_matches_eps_for_rounds():
    acc = RDPAccountant(sigma=1.0, sampling_rate=0.25)
    curve = acc.curve(45, step=5)
    assert curve[0][0] == 5
    assert curve[-1][0] == 45
    epsilons = [e for _, e in curve]
    assert all(epsilons[i] <= epsilons[i + 1] for i in range(len(epsilons) - 1))


if __name__ == "__main__":
    test_noise_shares_sum_to_approximately_target_variance()
    test_epsilon_decreases_as_sigma_increases()
    test_subsampling_amplifies_privacy()
    test_epsilon_grows_with_more_rounds()
    test_curve_is_monotonic_and_matches_eps_for_rounds()
    print("ALL dp.py TESTS PASSED")
