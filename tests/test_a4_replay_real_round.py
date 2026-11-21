"""A4-replay (free-riding via resubmitting a prior round's signed
update) inside a REAL round, not just the primitive-level isolated
check in `attacks/a4_freeride.py::run_attack_replay`.

Same gap `RUN_ATTACKS.ipynb` cell 22's own markdown names for A4-
replay specifically (distinct from A4-false-claim, see that cell's
reasoning on why the false-claim variant has no real-round analog --
its "lie" cannot even be constructed inside the honest submission
format, since the server recomputes the VRF outcome itself from
public data).

Builds the round directly against `fl/server.py::run_round`, same
torch-free, 100%-real-crypto pattern as `tests/test_round_end_to_end.py`
and `tests/test_a3_sybil_real_round.py`: 8 genuinely distinct, honest
clients submit fresh, correctly-signed updates for round t (rho_t_new);
a 9th REAL client's submission is NOT a fresh round-t submission at
all -- it is the EXACT ClientSubmissionPhase1/Phase2 object that same
client genuinely produced for round t-1 (rho_t_old), resubmitted
verbatim, unmodified, at round t.

The ring signature's key image and Fiat-Shamir transcript are built
over rho_t_old's hash-to-point base (`crypto/clsag.py`'s
`verify_with_scopes` recomputes `H_p(rho_t)` from whatever rho_t is
passed to it) -- re-verifying that same signature against round t's
DIFFERENT rho_t_new fails the closing equation, which is exactly V1
(`check_V1_ring_signature`). This is a genuine, no-shortcuts replay:
the resubmitted object is bit-for-bit what the honest client itself
produced one round earlier, not a hand-edited forgery.

Expected result: the replayed client is rejected with
"V1_ring_signature_failed"; the round is NOT aborted (8 fresh honest
clients already meet k_min=8); every genuinely fresh client is
accepted normally, unaffected by the replay attempt.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest

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
    P_MASK, pairwise_mask, pairwise_round_seed, self_mask,
    self_secret_bytes, new_self_mask_secret, shamir_share_self_secret,
)
from crypto.sketch import SketchProjector
from crypto.vector_commit import weighted_digest
from fl.client import ClientSubmissionPhase1, ClientSubmissionPhase2
from fl.server import UnmaskingMaterial, run_round

N_CLIENTS = 9          # clients 0-7 fresh/honest at round t, client 8 replays round t-1
REPLAYER_IDX = 8
MODEL_DIM = 48
SKETCH_M = 8
SHARE_T = 3
QUANT_BITS = 16
DENSITY = 1.0 / 3.0
N_TESTS = 4
RHO_T_OLD = b"rho_t_a4_replay_ROUND_PREV"   # round t-1's nonce
RHO_T_NEW = b"rho_t_a4_replay_ROUND_CURR"   # round t's nonce -- what run_round actually verifies against
RHO_E = b"rho_e_a4_replay"
SIGMA_T = b"sigma_t_a4_replay"
VRF_ALPHA = b"vrf_alpha_a4_replay"
B_T = 1.0e7  # generous -- this test is about V1, not V3
PK_TRACE = Point.hash_to_point(b"a4-replay-pk-trace")


def _one_client_submission(k, sk, ring, secrets, self_secrets, blinding_self_secrets,
                            n_participants, q_k, projector, rho_t: bytes):
    """Mirrors round_step_phase1/round_step_phase2's real steps, same
    as test_a3_sybil_real_round.py's helper, parameterized on rho_t so
    this can build the SAME client's submission under two DIFFERENT
    round nonces (the genuine round t-1 one, and what an honest round
    t submission would look like, for comparison).
    """
    masked = q_k % P_MASK
    for j in range(n_participants):
        if j == k:
            continue
        sign = 1 if k < j else -1
        masked = (masked + pairwise_mask(secrets[k][j], rho_t, MODEL_DIM, sign)) % P_MASK
    masked = (masked + self_mask(self_secret_bytes(self_secrets[k]), rho_t, MODEL_DIM)) % P_MASK

    blinding = Scalar.random()
    commitment = pedersen_commit(Scalar.from_int(weighted_digest(masked, seed=rho_t)), blinding)
    escrow = elgamal_encrypt(ring[k], PK_TRACE)
    masked_hash = hashlib.sha256(b"janus-f7a-masked:" + rho_t + masked.tobytes()).digest()
    message = commitment.point.b + escrow.c1.b + escrow.c2.b + masked_hash
    sig = clsag_sign(sk, k, ring, message, rho_t, RHO_E)

    phase1 = ClientSubmissionPhase1(
        commitment_point=commitment.point,
        escrow_ciphertext=escrow,
        key_image_round=sig.key_image_round,
        key_image_epoch=sig.key_image_epoch,
        ring_signature=sig,
        masked_vector_hash=masked_hash,
        was_sampled=True,
        _secret_blinding=blinding,
    )

    y = [int(v) for v in projector.project_int(q_k)]
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
    seeds = {j: pairwise_round_seed(secrets[k][j], rho_t) for j in range(n_participants) if j != k}
    phase2 = ClientSubmissionPhase2(
        sketch_commitment=sc,
        wraparound_bindings=bindings,
        wraparound_range_proofs=rps,
        identity_binding_proof=prove_identity_binding(sk, rho_t, RHO_E, VRF_ALPHA, b"r_cons:" + commitment.point.b),
        masked_y_blindings=NP.mask_blinding_vector(
            y_bl, k, seeds, self_secret_bytes(blinding_self_secrets[k]), rho_t,
        ),
    )
    return phase1, phase2, masked


def _build_replay_env():
    rng = np.random.default_rng(41)
    projector = SketchProjector(seed=SIGMA_T, m=SKETCH_M, d=MODEL_DIM)

    sks = [Scalar.random() for _ in range(N_CLIENTS)]
    ring = [scalar_mul_base(sk) for sk in sks]

    secrets = {}
    for a, b in itertools.combinations(range(N_CLIENTS), 2):
        s = hashlib.sha256(f"a4-pair-{a}-{b}".encode()).digest()
        secrets.setdefault(a, {})[b] = s
        secrets.setdefault(b, {})[a] = s

    q_true = {k: rng.integers(-300, 300, size=MODEL_DIM).astype(np.int64) for k in range(N_CLIENTS)}

    self_secrets = {k: new_self_mask_secret() for k in range(N_CLIENTS)}
    blinding_self_secrets = {k: new_self_mask_secret() for k in range(N_CLIENTS)}

    submissions, phase2s, masked_vectors = [], {}, {}

    # Clients 0-7: genuinely fresh, honestly-constructed round-t
    # submissions, signed against RHO_T_NEW (this round's real nonce).
    for k in range(REPLAYER_IDX):
        p1, p2, masked = _one_client_submission(
            k, sks[k], ring, secrets, self_secrets, blinding_self_secrets,
            N_CLIENTS, q_true[k], projector, rho_t=RHO_T_NEW,
        )
        submissions.append((k, p1))
        phase2s[k] = p2
        masked_vectors[k] = masked

    # Client 8: this is the GENUINE submission that client made for
    # round t-1 (signed against RHO_T_OLD) -- not a forgery, not
    # hand-edited. The attack is resubmitting it, verbatim, as if it
    # were a round-t submission.
    p1_replayed, p2_replayed, masked_replayed = _one_client_submission(
        REPLAYER_IDX, sks[REPLAYER_IDX], ring, secrets, self_secrets, blinding_self_secrets,
        N_CLIENTS, q_true[REPLAYER_IDX], projector, rho_t=RHO_T_OLD,
    )
    submissions.append((REPLAYER_IDX, p1_replayed))
    phase2s[REPLAYER_IDX] = p2_replayed
    masked_vectors[REPLAYER_IDX] = masked_replayed

    def unmask_provider(accepted_idx, excluded_idx):
        # Only clients 0-7 ever have self-mask shares in THIS round's
        # sharing (client 8's round t-1 shares were for a different
        # round's holder set in a real deployment) -- but since this
        # test's replayer is always rejected before unmasking, the
        # unmask step never needs to resolve client 8's shares at all.
        # Build shares only for the honest 0-7 to make that assumption
        # explicit and fail loudly if it's ever violated.
        held = {
            k: shamir_share_self_secret(self_secrets[k], SHARE_T, list(range(1, N_CLIENTS + 1)))
            for k in range(REPLAYER_IDX)
        }
        bheld = {
            k: shamir_share_self_secret(blinding_self_secrets[k], SHARE_T, list(range(1, N_CLIENTS + 1)))
            for k in range(REPLAYER_IDX)
        }
        shares = {k: [held[k][h + 1] for h in accepted_idx] for k in accepted_idx}
        blinding_shares = {k: [bheld[k][h + 1] for h in accepted_idx] for k in accepted_idx}
        seeds = {}
        for k in excluded_idx:
            for j in accepted_idx:
                seeds[(j, k)] = pairwise_round_seed(secrets[j][k], RHO_T_NEW)
        return UnmaskingMaterial(shares, seeds, SHARE_T, blinding_shares)

    return SimpleNamespace(
        submissions=submissions, phase2s=phase2s, masked_vectors=masked_vectors,
        projector=projector, unmask_provider=unmask_provider,
    )


def _run(env):
    return run_round(
        env.submissions, env.phase2s, RHO_T_NEW, RHO_E, VRF_ALPHA, B_T,
        N_CLIENTS, MODEL_DIM,
        unmask_provider=env.unmask_provider,
        masked_vector_provider=lambda k: env.masked_vectors[k],
        sketch_dim=SKETCH_M, sigma_t=SIGMA_T, quant_bits=QUANT_BITS, density=DENSITY,
        n_tests=N_TESTS, sketch_projector=env.projector,
    )


def test_replayed_submission_rejected_fresh_clients_unaffected():
    env = _build_replay_env()
    outcome = _run(env)

    assert not outcome.aborted, (
        f"round should NOT abort -- 8 fresh honest clients already meet k_min=8 "
        f"even with the replayer rejected. abort_reason={outcome.abort_reason}"
    )

    accepted_indices = set(outcome.accepted_client_indices)
    assert REPLAYER_IDX not in accepted_indices, (
        "the replayed round-(t-1) submission was ACCEPTED at round t -- V1's "
        "rho_t binding failed to catch a verbatim replay, a real defense failure"
    )
    assert "V1_ring_signature_failed" in outcome.rejected_reasons[REPLAYER_IDX], (
        f"replayer was rejected, but not for V1 -- actual reasons: "
        f"{outcome.rejected_reasons[REPLAYER_IDX]}"
    )
    for k in range(REPLAYER_IDX):
        assert k in accepted_indices, f"fresh honest client {k} was rejected -- {outcome.rejected_reasons.get(k)}"


def test_same_client_fresh_round_t_submission_would_have_been_accepted():
    """Control: confirms the replayer's rejection above is specifically
    about REUSING round t-1's submission, not some unrelated bug in
    this test's harness that would have rejected client 8 regardless
    of which round it signed against. Builds client 8's HONEST round-t
    submission (same secrets, same data, signed against RHO_T_NEW like
    everyone else) in an otherwise-identical round and confirms it is
    accepted normally.
    """
    # Simplest, clearest control: rebuild a fresh env from scratch with
    # ALL 9 clients honestly submitting against RHO_T_NEW (no replay at
    # all), then confirm client 8 is accepted normally there.
    rng2 = np.random.default_rng(41)
    projector2 = SketchProjector(seed=SIGMA_T, m=SKETCH_M, d=MODEL_DIM)
    sks = [Scalar.random() for _ in range(N_CLIENTS)]
    ring = [scalar_mul_base(sk) for sk in sks]
    secrets = {}
    for a, b in itertools.combinations(range(N_CLIENTS), 2):
        s = hashlib.sha256(f"a4-control-pair-{a}-{b}".encode()).digest()
        secrets.setdefault(a, {})[b] = s
        secrets.setdefault(b, {})[a] = s
    q_true = {k: rng2.integers(-300, 300, size=MODEL_DIM).astype(np.int64) for k in range(N_CLIENTS)}
    self_secrets = {k: new_self_mask_secret() for k in range(N_CLIENTS)}
    blinding_self_secrets = {k: new_self_mask_secret() for k in range(N_CLIENTS)}

    submissions, phase2s, masked_vectors = [], {}, {}
    for k in range(N_CLIENTS):
        p1, p2, masked = _one_client_submission(
            k, sks[k], ring, secrets, self_secrets, blinding_self_secrets,
            N_CLIENTS, q_true[k], projector2, rho_t=RHO_T_NEW,
        )
        submissions.append((k, p1))
        phase2s[k] = p2
        masked_vectors[k] = masked

    def unmask_provider(accepted_idx, excluded_idx):
        held = {
            k: shamir_share_self_secret(self_secrets[k], SHARE_T, list(range(1, N_CLIENTS + 1)))
            for k in range(N_CLIENTS)
        }
        bheld = {
            k: shamir_share_self_secret(blinding_self_secrets[k], SHARE_T, list(range(1, N_CLIENTS + 1)))
            for k in range(N_CLIENTS)
        }
        shares = {k: [held[k][h + 1] for h in accepted_idx] for k in accepted_idx}
        blinding_shares = {k: [bheld[k][h + 1] for h in accepted_idx] for k in accepted_idx}
        seeds = {}
        for k in excluded_idx:
            for j in accepted_idx:
                seeds[(j, k)] = pairwise_round_seed(secrets[j][k], RHO_T_NEW)
        return UnmaskingMaterial(shares, seeds, SHARE_T, blinding_shares)

    outcome = run_round(
        submissions, phase2s, RHO_T_NEW, RHO_E, VRF_ALPHA, B_T, N_CLIENTS, MODEL_DIM,
        unmask_provider=unmask_provider,
        masked_vector_provider=lambda k: masked_vectors[k],
        sketch_dim=SKETCH_M, sigma_t=SIGMA_T, quant_bits=QUANT_BITS, density=DENSITY,
        n_tests=N_TESTS, sketch_projector=projector2,
    )
    assert not outcome.aborted, outcome.abort_reason
    assert REPLAYER_IDX in set(outcome.accepted_client_indices), (
        "control failed: client 8's FRESH round-t submission should be accepted "
        "normally -- if this fails, the main test's rejection isn't proving what "
        "it claims to prove"
    )


def test_replay_result_matches_primitive_level_a4_test():
    """Cross-check against attacks/a4_freeride.py's own isolated claim."""
    from attacks.a4_freeride import run_attack_replay

    sk = Scalar.random()
    ring = [scalar_mul_base(Scalar.random()) for _ in range(5)] + [scalar_mul_base(sk)]
    pk_index = len(ring) - 1
    result = run_attack_replay(sk, pk_index, ring, RHO_T_OLD, RHO_E, RHO_T_NEW, b"old-message")
    assert result.defense_triggered, result.detail
