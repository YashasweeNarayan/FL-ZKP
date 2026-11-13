"""End-to-end `fl/server.py::run_round` -- V1, V2, V3 (F7's committed
sketch), V5, and F7e/F7f, on real crypto, with no torch.

This is the first test in the repo to execute `run_round` at all. Every
check inside it was previously only exercised by its own unit test or
by a Colab run of the full training loop, which meant the WIRING in
`run_round` (argument order, field names, the new F7e abort branch) was
written-but-unrun -- the exact pattern that produced several of the
bugs in FIX.md.

Torch is avoided by building the submissions directly rather than
through `round_step_phase1`/`round_step_phase2` (those call
`local_train_delta`/`TorchSketchProjector`), while using the real
primitives for everything `run_round` actually verifies: real CLSAG
ring signatures over the real F7a message, real Pedersen commitments,
the real NumPy integer sketch, real square proofs, real masking. The
NumPy `SketchProjector` is passed to `run_round` as `sketch_projector`
so both sides agree (see `_true_aggregate_sketch`'s docstring).

Runs under the test-only pure-Python ristretto255 backend.
"""
from __future__ import annotations

import hashlib
import itertools
import os

import numpy as np
import pytest

# Gate on whether ANY group backend is live, not on which one. With rbcl
# installed (Colab) these run against real libsodium; in a network-less
# sandbox they run against the pure-Python backend when
# JANUS_ALLOW_PURE_PYTHON_GROUP=1 selects it; with neither they skip.
#
# Gating on the env var alone -- as the first version of these files did --
# meant the whole F7 suite SKIPPED on Colab, which is the one place it most
# needs to run: the pure-Python backend proves the algebra, only rbcl proves
# the bindings.
from crypto.ristretto import backend_name as _backend_name

if _backend_name() is None:
    pytest.skip(
        "no ristretto255 backend available: install rbcl, or set "
        "JANUS_ALLOW_PURE_PYTHON_GROUP=1 to use the test-only pure-Python "
        "backend (crypto/_ristretto_py.py)",
        allow_module_level=True,
    )

from types import SimpleNamespace

from crypto import norm_proof as NP
from crypto.clsag import sign as clsag_sign
from crypto.elgamal_threshold import encrypt as elgamal_encrypt
from crypto.pedersen import commit as pedersen_commit
from crypto.rangeproof import prove_range, recompose_commitment
from crypto.r_cons import prove_identity_binding
from crypto.ristretto import Point, Scalar, scalar_mul_base
from crypto.secagg import (
    P_MASK, aggregate_masked, pairwise_mask, pairwise_round_seed, self_mask,
    self_secret_bytes, new_self_mask_secret, shamir_share_self_secret, unmask_aggregate,
)
from crypto.sketch import SketchProjector, int_sketch_norm_sq_bound
from crypto.vector_commit import weighted_digest
from fl.client import ClientSubmissionPhase1, ClientSubmissionPhase2
from fl.server import UnmaskingMaterial, run_round

# k_min is 8, so the accepted set must reach 8 or run_round aborts.
N_CLIENTS = 8
MODEL_DIM = 48
SKETCH_M = 8
SHARE_T = 3
QUANT_BITS = 16
DENSITY = 1.0 / 3.0
N_TESTS = 4  # fewer than the default 8, deliberately: proves n_tests is really threaded through rather than agreeing by coincidence
RHO_T = b"rho_t_e2e"
RHO_E = b"rho_e_e2e"
SIGMA_T = b"sigma_t_e2e"
VRF_ALPHA = b"vrf_alpha_e2e"
B_T = 1.0e7


def _build_env(cheater: int | None = None):
    rng = np.random.default_rng(11)
    projector = SketchProjector(seed=SIGMA_T, m=SKETCH_M, d=MODEL_DIM)

    sks = [Scalar.random() for _ in range(N_CLIENTS)]
    ring = [scalar_mul_base(sk) for sk in sks]
    pk_trace = Point.hash_to_point(b"e2e-pk-trace")

    secrets = {}
    for a, b in itertools.combinations(range(N_CLIENTS), 2):
        s = hashlib.sha256(f"e2e-pair-{a}-{b}".encode()).digest()
        secrets.setdefault(a, {})[b] = s
        secrets.setdefault(b, {})[a] = s

    q_true = {k: rng.integers(-300, 300, size=MODEL_DIM).astype(np.int64) for k in range(N_CLIENTS)}
    if cheater is not None:
        q_true[cheater] = np.full(MODEL_DIM, 10 ** 6, dtype=np.int64)  # A1-scale update, hidden behind a lying sketch

    self_secrets = {k: new_self_mask_secret() for k in range(N_CLIENTS)}
    shares_held = {
        k: shamir_share_self_secret(self_secrets[k], SHARE_T, list(range(1, N_CLIENTS + 1)))
        for k in range(N_CLIENTS)
    }

    submissions, phase2s, masked_vectors = [], {}, {}
    for k in range(N_CLIENTS):
        # ---- phase 1, mirroring round_step_phase1 steps 3-7
        masked = q_true[k] % P_MASK
        for j in range(N_CLIENTS):
            if j == k:
                continue
            sign = 1 if k < j else -1
            masked = (masked + pairwise_mask(secrets[k][j], RHO_T, MODEL_DIM, sign)) % P_MASK
        masked = (masked + self_mask(self_secret_bytes(self_secrets[k]), RHO_T, MODEL_DIM)) % P_MASK
        masked_vectors[k] = masked

        blinding = Scalar.random()
        commitment = pedersen_commit(Scalar.from_int(weighted_digest(masked, seed=RHO_T)), blinding)
        escrow = elgamal_encrypt(ring[k], pk_trace)
        masked_hash = hashlib.sha256(b"janus-f7a-masked:" + masked.tobytes()).digest()
        message = commitment.point.b + escrow.c1.b + escrow.c2.b + masked_hash
        sig = clsag_sign(sks[k], k, ring, message, RHO_T, RHO_E)
        submissions.append((k, ClientSubmissionPhase1(
            commitment_point=commitment.point,
            escrow_ciphertext=escrow,
            key_image_round=sig.key_image_round,
            key_image_epoch=sig.key_image_epoch,
            ring_signature=sig,
            masked_vector_hash=masked_hash,
            was_sampled=True,
            _secret_blinding=blinding,
        )))

        # ---- phase 2, mirroring round_step_phase2
        sketched = np.zeros(MODEL_DIM, dtype=np.int64) if (cheater is not None and k == cheater) else q_true[k]
        y = [int(v) for v in projector.project_int(sketched)]
        transcript = b"norm_proof:" + commitment.point.b
        sc, y_bl, _ = NP.commit_sketch(y, transcript)

        weights_list = NP.subset_sum_challenge(seed=SIGMA_T, m=SKETCH_M, n_tests=N_TESTS)
        window = NP.wraparound_window(SKETCH_M, MODEL_DIM, DENSITY, QUANT_BITS)
        n_bits = max(8, int(np.ceil(np.log2(max(2 * window + 2, 2)))))
        bindings, rps = [], []
        for j, w in enumerate(weights_list):
            tx = transcript + b":wrap:" + j.to_bytes(4, "little")
            s = sum(wi * yi for wi, yi in zip(w, y))
            shifted = min(max(s + window, 0), (1 << n_bits) - 1)
            rp, agg_bl = prove_range(shifted, n_bits)
            bindings.append(NP.prove_wraparound_binding(y, y_bl, w, window, tx, recompose_commitment(rp), agg_bl))
            rps.append(rp)

        seeds = {j: pairwise_round_seed(secrets[k][j], RHO_T) for j in range(N_CLIENTS) if j != k}
        phase2s[k] = ClientSubmissionPhase2(
            sketch_commitment=sc,
            wraparound_bindings=bindings,
            wraparound_range_proofs=rps,
            identity_binding_proof=prove_identity_binding(
                sks[k], RHO_T, RHO_E, VRF_ALPHA, b"r_cons:" + commitment.point.b
            ),
            masked_y_blindings=NP.mask_blinding_vector(y_bl, k, seeds),
        )

    def unmask_provider(accepted_idx, excluded_idx):
        shares = {k: [shares_held[k][h + 1] for h in accepted_idx] for k in accepted_idx}
        seeds = {}
        for k in excluded_idx:
            for j in accepted_idx:
                seeds[(j, k)] = pairwise_round_seed(secrets[j][k], RHO_T)
        return UnmaskingMaterial(shares, seeds, SHARE_T)

    def blame_material_provider(group):
        group = list(group)
        if len(group) < SHARE_T:
            return None
        outside = [k for k in range(N_CLIENTS) if k not in set(group)]
        straddling = {}
        for j in group:
            for k in outside:
                straddling[(j, k)] = pairwise_round_seed(secrets[j][k], RHO_T)
        partial = aggregate_masked(masked_vectors[k] for k in group)
        shares = {k: [shares_held[k][h + 1] for h in group] for k in group}
        try:
            partial_q = unmask_aggregate(partial, group, shares, SHARE_T, straddling, RHO_T, MODEL_DIM)
        except ValueError:
            return None
        return partial_q, straddling

    return SimpleNamespace(
        submissions=submissions, phase2s=phase2s, masked_vectors=masked_vectors,
        q_true=q_true, projector=projector, unmask_provider=unmask_provider,
        blame_material_provider=blame_material_provider,
    )


def _run(env, **kw):
    return run_round(
        env.submissions, env.phase2s, RHO_T, RHO_E, VRF_ALPHA, B_T, N_CLIENTS, MODEL_DIM,
        unmask_provider=env.unmask_provider,
        masked_vector_provider=lambda k: env.masked_vectors[k],
        sketch_dim=SKETCH_M, sigma_t=SIGMA_T, quant_bits=QUANT_BITS, density=DENSITY,
        n_tests=N_TESTS, sketch_projector=env.projector,
        **kw,
    )


def test_honest_round_accepts_everyone_and_passes_every_check():
    env = _build_env()
    out = _run(env, blame_material_provider=env.blame_material_provider)
    assert not out.aborted, f"{out.abort_reason} / {out.rejected_reasons}"
    assert sorted(out.accepted_client_indices) == list(range(N_CLIENTS))
    assert out.rejected_reasons == {}
    # F7e ran and passed
    assert out.aggregate_sketch_check_passed is True
    assert out.aggregate_sketch_bad_coordinates == []
    assert out.aggregate_sketch_suspect_groups == []


def test_honest_round_unmasks_to_the_true_aggregate():
    env = _build_env()
    out = _run(env)
    expected = aggregate_masked(env.q_true[k] for k in out.accepted_client_indices)
    assert np.array_equal(out.aggregate_quantized, expected)


def test_v5_opening_matches_the_commitment_product():
    from verifier.audit import independent_v5_check
    env = _build_env()
    out = _run(env)
    by_index = dict(env.submissions)
    commitments = [by_index[i].commitment_point for i in out.accepted_client_indices]
    audit = independent_v5_check(commitments, out.published_sum_value, out.published_sum_blinding)
    assert audit.v5_passed, audit.detail


def test_v1_rejects_a_tampered_masked_vector_hash():
    """F7a's whole point: the signature now covers the masked vector,
    so swapping the vector (or its hash) invalidates V1."""
    env = _build_env()
    idx, sub = env.submissions[2]
    env.submissions[2] = (idx, ClientSubmissionPhase1(
        commitment_point=sub.commitment_point,
        escrow_ciphertext=sub.escrow_ciphertext,
        key_image_round=sub.key_image_round,
        key_image_epoch=sub.key_image_epoch,
        ring_signature=sub.ring_signature,
        masked_vector_hash=hashlib.sha256(b"a different vector").digest(),
        was_sampled=sub.was_sampled,
        _secret_blinding=sub._secret_blinding,
    ))
    out = _run(env)
    # one client drops below k_min=8, so the round aborts -- but it must
    # abort for V1, not for anything else
    assert "V1_ring_signature_failed" in out.rejected_reasons.get(2, [])


def test_v3_rejects_an_over_bound_norm():
    env = _build_env()
    out = run_round(
        env.submissions, env.phase2s, RHO_T, RHO_E, VRF_ALPHA, 1e-6, N_CLIENTS, MODEL_DIM,
        unmask_provider=env.unmask_provider,
        masked_vector_provider=lambda k: env.masked_vectors[k],
        sketch_dim=SKETCH_M, sigma_t=SIGMA_T, quant_bits=QUANT_BITS, density=DENSITY,
        n_tests=N_TESTS, sketch_projector=env.projector,
    )
    assert out.aborted
    assert out.rejected_reasons, "nobody was rejected at all"
    for reasons in out.rejected_reasons.values():
        assert any("V3_norm_bound_exceeded" in s for s in reasons), reasons


def test_f7e_aborts_the_round_on_a_lying_sketch_and_localizes_it():
    """The cheat that passes V1, V2, V3, V4 and V5: an A1-scale update
    masked and committed consistently, with the sketch of a zero vector
    committed alongside it. Only F7e sees it."""
    env = _build_env(cheater=5)
    out = _run(env, blame_material_provider=env.blame_material_provider)
    assert out.aborted
    assert out.aggregate_sketch_check_passed is False
    assert "F7e aggregate sketch check failed" in out.abort_reason
    assert out.aggregate_sketch_bad_coordinates
    flat = set()
    for g in out.aggregate_sketch_suspect_groups:
        flat |= set(g)
    assert 5 in flat, f"cheater must be localized, got {out.aggregate_sketch_suspect_groups} / {out.abort_reason}"
    assert len(flat) < N_CLIENTS
    # the round did NOT apply an update
    assert out.aggregated_delta is None


def test_f7e_failure_without_a_blame_provider_is_unattributed_but_still_aborts():
    env = _build_env(cheater=5)
    out = _run(env)
    assert out.aborted
    assert out.aggregate_sketch_check_passed is False
    assert out.aggregate_sketch_suspect_groups == []
    assert "not localized" in out.abort_reason


def test_f7e_not_run_is_reported_as_none_not_failure():
    """A pre-F7e submission (no masked blinding vector) must leave the
    round running with the check marked un-run, not abort it as though
    cheating had been detected."""
    env = _build_env()
    env.phase2s[4].masked_y_blindings = None
    out = _run(env)
    assert not out.aborted, out.abort_reason
    assert out.aggregate_sketch_check_passed is None
