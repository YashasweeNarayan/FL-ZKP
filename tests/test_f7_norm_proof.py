"""FIX.md F7 (V3 norm-check redesign) -- crypto/norm_proof.py and its
wiring into fl/server.py::check_V3_norm_bound.

Runs under the test-only pure-Python ristretto255 backend
(JANUS_ALLOW_PURE_PYTHON_GROUP=1), same as tests/test_ristretto_pure.py.
Skips cleanly if that env var is unset (no real rbcl in this sandbox,
see crypto/ristretto.py).
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest

if os.environ.get("JANUS_ALLOW_PURE_PYTHON_GROUP") != "1":
    pytest.skip(
        "needs JANUS_ALLOW_PURE_PYTHON_GROUP=1 (test-only ristretto255 backend); "
        "see crypto/_ristretto_py.py",
        allow_module_level=True,
    )

from crypto import norm_proof as NP
from crypto.rangeproof import prove_range, verify_range, recompose_commitment
from crypto.pedersen import commit as pedersen_commit, commit_sum, verify_opening
from crypto.ristretto import Scalar, backend_name
from crypto.sketch import int_sketch_norm_sq_bound


def test_backend_is_test_only():
    assert backend_name() == "pure-python (TEST ONLY)"


# --------------------------------------------------------------------
# F7c: square proof
# --------------------------------------------------------------------

def test_square_proof_honest():
    y = Scalar.from_int(37)
    r_y, r_c = Scalar.random(), Scalar.random()
    C_y = pedersen_commit(y, r_y)
    C_c = pedersen_commit(Scalar.from_int(37 * 37), r_c)
    pf = NP.prove_square(y, r_y, r_c, b"tx")
    assert NP.verify_square(pf, C_y.point, C_c.point, b"tx")


def test_square_proof_rejects_wrong_square():
    y = Scalar.from_int(37)
    r_y, r_c = Scalar.random(), Scalar.random()
    C_y = pedersen_commit(y, r_y)
    pf = NP.prove_square(y, r_y, r_c, b"tx")
    C_c_bad = pedersen_commit(Scalar.from_int(37 * 37 + 1), r_c)
    assert not NP.verify_square(pf, C_y.point, C_c_bad.point, b"tx")


def test_square_proof_rejects_wrong_base_value():
    y = Scalar.from_int(37)
    r_y, r_c = Scalar.random(), Scalar.random()
    C_c = pedersen_commit(Scalar.from_int(37 * 37), r_c)
    pf = NP.prove_square(y, r_y, r_c, b"tx")
    C_y_bad = pedersen_commit(Scalar.from_int(38), r_y)
    assert not NP.verify_square(pf, C_y_bad.point, C_c.point, b"tx")


# --------------------------------------------------------------------
# F7b/c: full per-coordinate sketch commitment
# --------------------------------------------------------------------

Y_VEC = [5, -3, 0, 12, -7]


def test_sketch_commitment_honest():
    sc, y_bl, sq_bl = NP.commit_sketch(Y_VEC, b"client0:round0")
    assert sc.v == sum(v * v for v in Y_VEC)
    assert NP.verify_sketch_commitment(sc, b"client0:round0", norm_bound_sq=10 ** 9)


def test_sketch_commitment_rejects_over_bound():
    sc, _, _ = NP.commit_sketch(Y_VEC, b"client0:round0")
    assert not NP.verify_sketch_commitment(sc, b"client0:round0", norm_bound_sq=10)


def test_sketch_commitment_rejects_swapped_proofs():
    sc, _, _ = NP.commit_sketch(Y_VEC, b"client0:round0")
    sc_bad = copy.deepcopy(sc)
    sc_bad.square_proofs[0], sc_bad.square_proofs[1] = sc_bad.square_proofs[1], sc_bad.square_proofs[0]
    assert not NP.verify_sketch_commitment(sc_bad, b"client0:round0", norm_bound_sq=10 ** 9)


def test_sketch_commitment_rejects_mismatched_v():
    sc, _, _ = NP.commit_sketch(Y_VEC, b"client0:round0")
    sc_bad = copy.deepcopy(sc)
    sc_bad.v = sc.v + 1
    assert not NP.verify_sketch_commitment(sc_bad, b"client0:round0", norm_bound_sq=10 ** 9)


def test_sketch_commitment_wrong_transcript_rejected():
    sc, _, _ = NP.commit_sketch(Y_VEC, b"client0:round0")
    assert not NP.verify_sketch_commitment(sc, b"client0:round1", norm_bound_sq=10 ** 9)


# --------------------------------------------------------------------
# F7e: cross-client aggregate check
# --------------------------------------------------------------------

def test_aggregate_sketch_check_honest_and_isolates_bad_coordinate():
    y1, y2 = [1, 2, 3], [4, -2, 1]
    sc1, y_bl1, _ = NP.commit_sketch(y1, b"c1")
    sc2, y_bl2, _ = NP.commit_sketch(y2, b"c2")
    true_agg = [a + b for a, b in zip(y1, y2)]
    sum_bl = [y_bl1[i] + y_bl2[i] for i in range(3)]

    ok, bad = NP.verify_aggregate_sketch([sc1.y_commitments, sc2.y_commitments], sum_bl, true_agg)
    assert ok and bad == []

    ok2, bad2 = NP.verify_aggregate_sketch([sc1.y_commitments, sc2.y_commitments], sum_bl, [9, 9, 9])
    assert not ok2 and bad2 == [0, 1, 2]

    bad_agg = list(true_agg)
    bad_agg[1] += 1
    ok3, bad3 = NP.verify_aggregate_sketch([sc1.y_commitments, sc2.y_commitments], sum_bl, bad_agg)
    assert not ok3 and bad3 == [1]


# --------------------------------------------------------------------
# F7d: hidden wraparound test (binding + range proof, no cleartext
# test value or blinding ever produced on the wire)
# --------------------------------------------------------------------

MODEL_DIM, DENSITY, QUANT_BITS = 1000, 1.0 / 3.0, 16


def _wraparound_round(y_vec, m, seed=b"seed", n_tests=4):
    window = NP.wraparound_window(m, MODEL_DIM, DENSITY, QUANT_BITS)
    weights_list = NP.subset_sum_challenge(seed, m, n_tests=n_tests)
    n_bits = max(8, int(np.ceil(np.log2(max(2 * window + 2, 2)))))
    return window, weights_list, n_bits


def test_wraparound_window_is_tight_against_pmask():
    """The whole point of deriving the window from quant_bits instead
    of P_MASK//2 (see wraparound_window's docstring): a P_MASK-sized
    shift in a single coordinate must dwarf the honest window, not be
    swamped by it."""
    m = 32
    window = NP.wraparound_window(m, MODEL_DIM, DENSITY, QUANT_BITS)
    P_MASK = 2 ** 61 - 1
    assert window < P_MASK / 1000  # window is many orders of magnitude below one P_MASK step


def test_wraparound_binding_honest_roundtrip():
    sc, y_bl, _ = NP.commit_sketch(Y_VEC, b"c0")
    m = len(Y_VEC)
    window, weights_list, n_bits = _wraparound_round(Y_VEC, m)
    for j, w in enumerate(weights_list):
        tx = b"c0:wrap:" + j.to_bytes(4, "little")
        s = sum(wi * yi for wi, yi in zip(w, Y_VEC))
        shifted = s + window
        assert 0 <= shifted < (1 << n_bits)
        rp, agg_bl = prove_range(shifted, n_bits)
        bp = NP.prove_wraparound_binding(Y_VEC, y_bl, w, window, tx, recompose_commitment(rp), agg_bl)
        assert verify_range(rp)
        C_j = recompose_commitment(rp)
        assert NP.verify_wraparound_binding(C_j, bp, w, sc.y_commitments, window, tx)


def test_wraparound_binding_rejects_mismatched_weights():
    sc, y_bl, _ = NP.commit_sketch(Y_VEC, b"c0")
    m = len(Y_VEC)
    window, weights_list, n_bits = _wraparound_round(Y_VEC, m)
    w0, w1 = weights_list[0], weights_list[1]
    s = sum(wi * yi for wi, yi in zip(w0, Y_VEC))
    shifted = min(max(s + window, 0), (1 << n_bits) - 1)
    rp, agg_bl = prove_range(shifted, n_bits)
    bp = NP.prove_wraparound_binding(Y_VEC, y_bl, w0, window, b"tx", recompose_commitment(rp), agg_bl)
    assert not NP.verify_wraparound_binding(recompose_commitment(rp), bp, w1, sc.y_commitments, window, b"tx")


def test_wraparound_catches_pmask_sized_shift():
    """A client that sketches a wrapped representative for one
    coordinate (q_j +-> q_j + P_MASK) must land outside the honest
    window on at least one of several random +-1 tests."""
    m = 5
    P_MASK = 2 ** 61 - 1
    weights_list = NP.subset_sum_challenge(b"seed-round0", m, n_tests=8)
    window = NP.wraparound_window(m, MODEL_DIM, DENSITY, QUANT_BITS)
    y_wrapped = list(Y_VEC)
    y_wrapped[2] += P_MASK
    outside = sum(
        1 for w in weights_list
        if abs(sum(wi * yi for wi, yi in zip(w, y_wrapped))) > window
    )
    assert outside >= 1


# --------------------------------------------------------------------
# F7f: group-testing blame localization
# --------------------------------------------------------------------

def test_localize_fault_narrows_to_bad_clients():
    clients = list(range(20))
    bad_set = {7, 13}

    def check(group):
        return not (set(group) & bad_set)

    res = NP.localize_fault(clients, coordinate=0, seed=b"roundseed",
                             check_group_opens_correctly=check, min_group_size=2)
    flat = set()
    for g in res.suspect_groups:
        flat |= set(g)
    assert bad_set <= flat
    assert len(flat) < len(clients)


def test_localize_fault_no_fault_means_no_suspects():
    clients = list(range(20))

    def check(group):
        return True  # never fails

    res = NP.localize_fault(clients, coordinate=0, seed=b"roundseed",
                             check_group_opens_correctly=check, min_group_size=2)
    assert res.suspect_groups == []


# --------------------------------------------------------------------
# Integration: fl/server.py::check_V3_norm_bound against a hand-built
# submission (mirrors what fl/client.py::round_step_phase2 produces,
# without needing torch for this test).
# --------------------------------------------------------------------

def _build_phase2(y_vec, commitment_point, sketch_seed, sketch_dim, n_tests=8):
    transcript = b"norm_proof:" + commitment_point.b
    sc, y_bl, _ = NP.commit_sketch(y_vec, transcript)
    weights_list = NP.subset_sum_challenge(seed=sketch_seed, m=sketch_dim, n_tests=n_tests)
    window = NP.wraparound_window(sketch_dim, MODEL_DIM, DENSITY, QUANT_BITS)
    n_bits = max(8, int(np.ceil(np.log2(max(2 * window + 2, 2)))))
    bindings, rps = [], []
    for j, w in enumerate(weights_list):
        tx = transcript + b":wrap:" + j.to_bytes(4, "little")
        s = sum(wi * yi for wi, yi in zip(w, y_vec))
        shifted = min(max(s + window, 0), (1 << n_bits) - 1)
        rp, agg_bl = prove_range(shifted, n_bits)
        bp = NP.prove_wraparound_binding(y_vec, y_bl, w, window, tx, recompose_commitment(rp), agg_bl)
        bindings.append(bp)
        rps.append(rp)
    from types import SimpleNamespace
    return SimpleNamespace(sketch_commitment=sc, wraparound_bindings=bindings, wraparound_range_proofs=rps)


def test_server_check_V3_accepts_honest_in_bound_submission():
    import fl.server as server
    commitment_point = pedersen_commit(Scalar.from_int(999), Scalar.random()).point
    sketch_seed = b"sigma_t_round1"
    sketch_dim = 32
    rng = np.random.default_rng(3)
    y_vec = [int(v) for v in rng.integers(-5, 5, size=sketch_dim)]
    phase2 = _build_phase2(y_vec, commitment_point, sketch_seed, sketch_dim)
    B_t = 500.0
    assert server.check_V3_norm_bound(phase2, B_t, commitment_point, sketch_dim, MODEL_DIM,
                                       sketch_seed, QUANT_BITS, DENSITY, 8)


def test_server_check_V3_rejects_over_bound_submission():
    import fl.server as server
    commitment_point = pedersen_commit(Scalar.from_int(12345), Scalar.random()).point
    sketch_seed = b"sigma_t_round0"
    sketch_dim = 32
    rng = np.random.default_rng(0)
    y_vec = [int(v) for v in rng.integers(-50, 50, size=sketch_dim)]
    phase2 = _build_phase2(y_vec, commitment_point, sketch_seed, sketch_dim)
    B_t = 50.0  # too tight for this y_vec
    assert not server.check_V3_norm_bound(phase2, B_t, commitment_point, sketch_dim, MODEL_DIM,
                                           sketch_seed, QUANT_BITS, DENSITY, 8)


def test_server_check_V3_rejects_mismatched_sketch_seed():
    """A server using the wrong sigma_t derives different public
    weights than the client proved against -- must reject, not just
    happen to disagree with an attacker."""
    import fl.server as server
    commitment_point = pedersen_commit(Scalar.from_int(999), Scalar.random()).point
    sketch_seed = b"sigma_t_round1"
    sketch_dim = 32
    rng = np.random.default_rng(3)
    y_vec = [int(v) for v in rng.integers(-5, 5, size=sketch_dim)]
    phase2 = _build_phase2(y_vec, commitment_point, sketch_seed, sketch_dim)
    assert not server.check_V3_norm_bound(phase2, 500.0, commitment_point, sketch_dim, MODEL_DIM,
                                           b"WRONG_SEED", QUANT_BITS, DENSITY, 8)


def test_server_check_V3_rejects_swapped_wraparound_proof():
    import fl.server as server
    commitment_point = pedersen_commit(Scalar.from_int(999), Scalar.random()).point
    sketch_seed = b"sigma_t_round1"
    sketch_dim = 32
    rng = np.random.default_rng(3)
    y_vec = [int(v) for v in rng.integers(-5, 5, size=sketch_dim)]
    phase2 = _build_phase2(y_vec, commitment_point, sketch_seed, sketch_dim)
    phase2.wraparound_range_proofs[0], phase2.wraparound_range_proofs[1] = (
        phase2.wraparound_range_proofs[1], phase2.wraparound_range_proofs[0]
    )
    assert not server.check_V3_norm_bound(phase2, 500.0, commitment_point, sketch_dim, MODEL_DIM,
                                           sketch_seed, QUANT_BITS, DENSITY, 8)
