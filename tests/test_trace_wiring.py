"""PURE-MATH tests for fl/trace_wiring.py -- the real-per-round-pipeline
evidence wiring (epoch rollover + per-round recording + theta-crossing
detection). No torch, no EC backend: pseudonyms here are plain strings
standing in for clsag.py's P_k^e point bytes (any hashable works, same
as fl/evidence.py::EvidenceAccumulator itself requires).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fl.trace_wiring import EpochEvidenceState


def make_state(rng=None, n_permutations=1000):
    rng = rng if rng is not None else np.random.default_rng(0)
    root_norms = np.abs(rng.normal(5.0, 1.0, size=300))
    return EpochEvidenceState(
        epoch_rounds=25, sampling_rate=0.25, server_root_norms=root_norms,
        alpha=0.05, n_permutations=n_permutations, rng=rng,
    )


def test_epoch_starts_unset_until_first_record_round():
    state = make_state()
    assert state.epoch is None
    state.record_round(t_round=0, pseudonym_norm_pairs=[("p1", 5.0)], B_t=10.0)
    assert state.epoch == 0


def test_same_epoch_does_not_recalibrate_theta():
    state = make_state()
    state.record_round(t_round=0, pseudonym_norm_pairs=[("p1", 5.0)], B_t=10.0)
    theta_after_round0 = state.theta
    state.record_round(t_round=1, pseudonym_norm_pairs=[("p1", 5.0)], B_t=10.0)
    assert state.theta == theta_after_round0


def test_epoch_rollover_resets_accumulated_totals():
    state = make_state()
    # round 0..24 all fall in epoch 0 (epoch_rounds=25)
    state.record_round(t_round=0, pseudonym_norm_pairs=[("p1", 1000.0)], B_t=10.0)
    assert state.total_for("p1") > 0.0
    # round 25 crosses into epoch 1 -- fresh accumulator
    state.record_round(t_round=25, pseudonym_norm_pairs=[], B_t=10.0)
    assert state.epoch == 1
    assert state.total_for("p1") == 0.0


def test_honest_pseudonym_stays_unflagged_across_a_full_epoch():
    """Deliberately deterministic, unlike the calibration-rate tests
    above: calibrate_theta_from_norms only certifies Pr[open|honest] <=
    alpha, so asserting a SINGLE random honest epoch never flags would
    itself have a ~5% chance of a false failure. "Honest" here mirrors
    the real pipeline's dummy-padding design (every client is in
    eligible_this_round every round, but a round a client's VRF
    self-sampling didn't select gets an honestly-zero dummy delta, so
    norm_q~0 that round) -- ~sampling_rate=0.25 of the 25 rounds carry
    a real norm well inside the root set's own distribution, the rest
    are dummy zeros, matching what calibrate_theta_from_norms's own
    Binomial(rounds_per_epoch, sampling_rate) participation model
    assumes. The only question this test asks is "does normal
    behavior clear theta with room to spare", not "does alpha hold
    exactly" (that's the calibration-rate tests' job, over thousands
    of simulated epochs).
    """
    rng = np.random.default_rng(7)
    state = make_state(rng=rng)
    sampled_rounds = {0, 4, 8, 12, 16, 20}  # 6 of 25 ~= sampling_rate 0.25
    newly_flagged_ever = []
    for t in range(25):
        norm_q = 5.0 if t in sampled_rounds else 0.0
        newly_flagged_ever += state.record_round(t_round=t, pseudonym_norm_pairs=[("honest", norm_q)], B_t=10.0)
    assert newly_flagged_ever == []


def test_persistently_scaled_pseudonym_gets_flagged_and_only_once():
    rng = np.random.default_rng(8)
    state = make_state(rng=rng)
    B_t = 10.0
    all_newly_flagged = []
    for t in range(10):
        all_newly_flagged += state.record_round(
            t_round=t, pseudonym_norm_pairs=[("malicious", B_t * 5.0)], B_t=B_t,
        )
    assert all_newly_flagged.count("malicious") == 1  # flagged exactly once, the round it first crossed theta

    # further rounds of the SAME bad behavior must not re-flag it
    more_flags = []
    for t in range(10, 15):
        more_flags += state.record_round(
            t_round=t, pseudonym_norm_pairs=[("malicious", B_t * 5.0)], B_t=B_t,
        )
    assert "malicious" not in more_flags


def test_certificate_matches_theta_and_is_none_before_first_round():
    state = make_state()
    assert state.certificate is None
    state.record_round(t_round=0, pseudonym_norm_pairs=[("p1", 5.0)], B_t=10.0)
    assert state.certificate is not None
    assert state.certificate.theta == state.theta


def test_rejected_clients_still_contribute_evidence():
    """A client this round's protocol rejected (e.g. V3) must still be
    passed into record_round -- evidence tracks behavior independent of
    this round's accept/reject outcome, and excluding rejected clients
    would stop tracking exactly who evidence accumulation targets.
    This test only confirms the API accepts and accumulates for ANY
    pseudonym handed to it, accepted or not -- round_loop.py's own
    wiring is responsible for actually including rejected clients.
    """
    state = make_state()
    state.record_round(t_round=0, pseudonym_norm_pairs=[("rejected_client", 999.0)], B_t=10.0)
    assert state.total_for("rejected_client") > 0.0


if __name__ == "__main__":
    test_epoch_starts_unset_until_first_record_round()
    print("test_epoch_starts_unset_until_first_record_round PASSED")
    test_same_epoch_does_not_recalibrate_theta()
    print("test_same_epoch_does_not_recalibrate_theta PASSED")
    test_epoch_rollover_resets_accumulated_totals()
    print("test_epoch_rollover_resets_accumulated_totals PASSED")
    test_honest_pseudonym_stays_unflagged_across_a_full_epoch()
    print("test_honest_pseudonym_stays_unflagged_across_a_full_epoch PASSED")
    test_persistently_scaled_pseudonym_gets_flagged_and_only_once()
    print("test_persistently_scaled_pseudonym_gets_flagged_and_only_once PASSED")
    test_certificate_matches_theta_and_is_none_before_first_round()
    print("test_certificate_matches_theta_and_is_none_before_first_round PASSED")
    test_rejected_clients_still_contribute_evidence()
    print("test_rejected_clients_still_contribute_evidence PASSED")
    print("\nALL trace_wiring TESTS PASSED")
