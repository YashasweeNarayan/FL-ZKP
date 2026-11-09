"""PURE-MATH tests for fl/evidence.py -- the §6.1 accumulator and its
certified false-accusation bound (ARCHITECTURE.md's Theorem). Synthetic
vectors only, no torch.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fl.evidence import (
    cosine_distance, per_round_evidence, EvidenceAccumulator,
    calibrate_theta,
)

DIM = 64


def test_cosine_distance_identical_vectors_is_zero():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(cosine_distance(v, v)) < 1e-9


def test_cosine_distance_opposite_vectors_is_two():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_distance(v, -v) - 2.0) < 1e-9


def test_cosine_distance_handles_zero_vector():
    v = np.zeros(5)
    ref = np.ones(5)
    assert cosine_distance(v, ref) == 1.0


def test_accumulator_sums_correctly_per_pseudonym():
    acc = EvidenceAccumulator(epoch=0)
    acc.record("pseudoA", 0.5)
    acc.record("pseudoA", 0.7)
    acc.record("pseudoB", 10.0)
    assert abs(acc.total_for("pseudoA") - 1.2) < 1e-9
    assert abs(acc.total_for("pseudoB") - 10.0) < 1e-9
    assert acc.total_for("unseen_pseudonym") == 0.0


def test_accumulator_flags_sorted_descending():
    acc = EvidenceAccumulator(epoch=0)
    acc.record("low", 3.0)
    acc.record("high", 9.0)
    acc.record("mid", 6.0)
    acc.record("under_threshold", 1.0)
    flagged = acc.flagged(theta=2.0)
    assert flagged == ["high", "mid", "low"]


def make_root_set(rng, n=200, dim=DIM, center=None):
    center = center if center is not None else np.zeros(dim)
    return [center + rng.normal(0, 1.0, size=dim) for _ in range(n)]


def test_calibration_bounds_honest_false_accusation_rate():
    """THE guardrail test for ARCHITECTURE.md's Theorem: calibrating
    theta on honest root-set data, then evaluating it against FRESH
    simulated honest epochs (not reused in calibration), should keep
    the empirical open-rate close to the target alpha -- not
    exploding due to non-IID-style legitimate variation.
    """
    rng = np.random.default_rng(0)
    median_ref = np.zeros(DIM)
    root_vectors = make_root_set(rng, n=300)
    loss_deltas = rng.normal(0, 0.01, size=300)
    B_t = 10.0
    alpha = 0.05
    rounds_per_epoch = 25
    sampling_rate = 0.25

    cert = calibrate_theta(
        root_vectors, median_ref, B_t, loss_deltas,
        rounds_per_epoch, sampling_rate, alpha=alpha,
        n_permutations=3000, rng=rng,
    )
    assert abs(cert.empirical_fp_rate_at_theta - alpha) < 0.02

    # now evaluate on a FRESH batch of simulated honest epochs (held
    # out from calibration) and confirm the open-rate still tracks alpha
    n_eval_epochs = 2000
    opens = 0
    for _ in range(n_eval_epochs):
        n_rounds = max(rng.binomial(rounds_per_epoch, sampling_rate), 1)
        acc = EvidenceAccumulator(epoch=1)
        for t in range(n_rounds):
            q = median_ref + rng.normal(0, 1.0, size=DIM)
            norm_q = float(np.linalg.norm(q))
            loss_delta = float(rng.normal(0, 0.01))
            ev_t = per_round_evidence(q, median_ref, norm_q, B_t, loss_delta)
            acc.record("honest_client", ev_t)
        if acc.total_for("honest_client") > cert.theta:
            opens += 1
    empirical_rate = opens / n_eval_epochs
    assert empirical_rate < alpha + 0.03, (
        f"false-accusation rate {empirical_rate} exceeds alpha={alpha} budget by too much"
    )


def test_malicious_client_with_persistently_higher_norm_gets_flagged():
    """A1-style scaling attacker: consistently inflated update norm
    every round should accumulate evidence well past theta within one
    epoch, while an honest client with the same median-cosine behavior
    stays under it.
    """
    rng = np.random.default_rng(1)
    median_ref = np.zeros(DIM)
    root_vectors = make_root_set(rng, n=300)
    loss_deltas = rng.normal(0, 0.01, size=300)
    B_t = 10.0
    rounds_per_epoch = 25
    sampling_rate = 0.25

    cert = calibrate_theta(
        root_vectors, median_ref, B_t, loss_deltas,
        rounds_per_epoch, sampling_rate, alpha=0.05,
        n_permutations=2000, rng=rng,
    )

    n_rounds = 10  # within one epoch, p*R>=5 constraint satisfied
    honest_acc = EvidenceAccumulator(epoch=2)
    malicious_acc = EvidenceAccumulator(epoch=2)
    for t in range(n_rounds):
        q_honest = median_ref + rng.normal(0, 1.0, size=DIM)
        ev_honest = per_round_evidence(q_honest, median_ref, float(np.linalg.norm(q_honest)), B_t, float(rng.normal(0, 0.01)))
        honest_acc.record("honest", ev_honest)

        q_malicious = median_ref + rng.normal(0, 1.0, size=DIM)
        scaled_norm = B_t * 5.0  # a1_scaling: norm blown up 5x the bound
        ev_malicious = per_round_evidence(q_malicious, median_ref, scaled_norm, B_t, float(rng.normal(0.05, 0.01)))
        malicious_acc.record("malicious", ev_malicious)

    assert honest_acc.total_for("honest") < cert.theta
    assert malicious_acc.total_for("malicious") > cert.theta


if __name__ == "__main__":
    test_cosine_distance_identical_vectors_is_zero()
    print("test_cosine_distance_identical_vectors_is_zero PASSED")
    test_cosine_distance_opposite_vectors_is_two()
    print("test_cosine_distance_opposite_vectors_is_two PASSED")
    test_cosine_distance_handles_zero_vector()
    print("test_cosine_distance_handles_zero_vector PASSED")
    test_accumulator_sums_correctly_per_pseudonym()
    print("test_accumulator_sums_correctly_per_pseudonym PASSED")
    test_accumulator_flags_sorted_descending()
    print("test_accumulator_flags_sorted_descending PASSED")
    test_calibration_bounds_honest_false_accusation_rate()
    print("test_calibration_bounds_honest_false_accusation_rate PASSED")
    test_malicious_client_with_persistently_higher_norm_gets_flagged()
    print("test_malicious_client_with_persistently_higher_norm_gets_flagged PASSED")
    print("\nALL evidence TESTS PASSED")
