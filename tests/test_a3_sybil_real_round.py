"""A3 (Sybil / multi-submission) inside a REAL round, not just the
primitive-level isolated check in `attacks/a3_sybil.py`.

Same gap `RUN_ATTACKS.ipynb` cell 22's own markdown already names:
"A3 (Sybil) and A4-replay are also genuinely per-round attacks and
belong here too in a future pass." This is that pass, for A3.

Unlike A1 (`tests/test_round_end_to_end.py`'s F7e/F11-style gates,
`fl/round_loop.py`'s `cheater_client_idx`/`cheater_force_delta`), A3
cannot be expressed through `run_training_loop`'s per-client dict
structure: `phase1_by_client`/`phase2_by_client` are keyed one entry
per client index per round, so a genuine SECOND submission from the
SAME client in the SAME round has no slot to live in without
restructuring that orchestration -- which is deliberately NOT done
here, to avoid touching a 900-line, already-fragile file for a
change this test doesn't need. Instead, this builds the round
directly against `fl/server.py::run_round` (same low-level entry
point `tests/test_round_end_to_end.py` already uses, torch-free, 100%
real crypto -- CLSAG, Pedersen, the real integer sketch, real square
proofs, real masking): 8 genuinely distinct, honest clients submit
once each, and a 9th REAL client (same signing key, same ring
position, same round nonce rho_t) submits TWICE -- an honest
submission and a Sybil duplicate with different crafted content --
both fed into the SAME `submissions` list `run_round` verifies.

Key image I_t = sk * H_p(rho_t) depends only on the signer's secret
key and the round nonce, not on which message was signed or which
dict key this test's own bookkeeping uses for the duplicate entry --
so both of client 8's submissions carry the IDENTICAL key image,
which is exactly what V2 (`check_V2_no_double_submission`) checks
for. The bookkeeping key for the duplicate ("8_sybil") is a harness
convenience so `phase2s`/`submissions` don't collide on dict-equality
of the real client index; it has no cryptographic meaning and is not
read by `run_round` for anything except its own dict lookup.

Expected result: exactly ONE of client 8's two submissions is
accepted; the other is rejected with "V2_double_submission" in its
reasons; the round is NOT aborted (9 total submissions, k_min=8 is
already met by clients 0-7 alone); every genuinely honest client
(0-7) is accepted normally, unaffected by the attack.
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

N_HONEST = 8          # clients 0-7, k_min exactly, each submits once
SYBIL_IDX = 8         # the 9th REAL client, ring position 8 -- submits TWICE
MODEL_DIM = 48
SKETCH_M = 8
SHARE_T = 3
QUANT_BITS = 16
DENSITY = 1.0 / 3.0
N_TESTS = 4
RHO_T = b"rho_t_a3_real_round"
RHO_E = b"rho_e_a3_real_round"
SIGMA_T = b"sigma_t_a3_real_round"
VRF_ALPHA = b"vrf_alpha_a3_real_round"
B_T = 1.0e7  # generous bound -- this test is about V2, not V3


def _one_client_submission(k, sk, ring, secrets, self_secrets, blinding_self_secrets,
                            shares_held, blinding_shares_held, n_participants, q_k,
                            projector, message_tag: bytes):
    """Builds one real, honestly-constructed (phase1, phase2) pair for
    client k, mirroring round_step_phase1/round_step_phase2's real
    steps exactly as tests/test_round_end_to_end.py's _build_env does
    -- the only new parameter is message_tag, which lets the SAME
    signer (same sk, same ring position k, same rho_t -- so the same
    key image) produce two DIFFERENT, independently valid submissions
    (different masked vector / different message) for the Sybil
    duplicate.
    """
    masked = q_k % P_MASK
    for j in range(n_participants):
        if j == k:
            continue
        sign = 1 if k < j else -1
        masked = (masked + pairwise_mask(secrets[k][j], RHO_T, MODEL_DIM, sign)) % P_MASK
    masked = (masked + self_mask(self_secret_bytes(self_secrets[k]), RHO_T, MODEL_DIM)) % P_MASK

    blinding = Scalar.random()
    commitment = pedersen_commit(Scalar.from_int(weighted_digest(masked, seed=RHO_T)), blinding)
    pk_trace = Point.hash_to_point(b"a3-real-round-pk-trace")
    escrow = elgamal_encrypt(ring[k], pk_trace)
    masked_hash = hashlib.sha256(message_tag + masked.tobytes()).digest()
    message = commitment.point.b + escrow.c1.b + escrow.c2.b + masked_hash
    sig = clsag_sign(sk, k, ring, message, RHO_T, RHO_E)

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
    seeds = {j: pairwise_round_seed(secrets[k][j], RHO_T) for j in range(n_participants) if j != k}
    phase2 = ClientSubmissionPhase2(
        sketch_commitment=sc,
        wraparound_bindings=bindings,
        wraparound_range_proofs=rps,
        identity_binding_proof=prove_identity_binding(sk, RHO_T, RHO_E, VRF_ALPHA, b"r_cons:" + commitment.point.b),
        masked_y_blindings=NP.mask_blinding_vector(
            y_bl, k, seeds, self_secret_bytes(blinding_self_secrets[k]), RHO_T,
        ),
    )
    return phase1, phase2, masked


def _build_sybil_env():
    n_participants = N_HONEST + 1  # 0..7 honest + 8 (the real Sybil signer)
    rng = np.random.default_rng(23)
    projector = SketchProjector(seed=SIGMA_T, m=SKETCH_M, d=MODEL_DIM)

    sks = [Scalar.random() for _ in range(n_participants)]
    ring = [scalar_mul_base(sk) for sk in sks]

    secrets = {}
    for a, b in itertools.combinations(range(n_participants), 2):
        s = hashlib.sha256(f"a3-pair-{a}-{b}".encode()).digest()
        secrets.setdefault(a, {})[b] = s
        secrets.setdefault(b, {})[a] = s

    q_true = {k: rng.integers(-300, 300, size=MODEL_DIM).astype(np.int64) for k in range(n_participants)}
    # the Sybil's SECOND submission uses a DIFFERENT crafted vector --
    # the attack's whole point is trying to get an extra, different
    # contribution counted under cover of the same anonymous ring key.
    q_sybil_duplicate = rng.integers(-300, 300, size=MODEL_DIM).astype(np.int64)

    self_secrets = {k: new_self_mask_secret() for k in range(n_participants)}
    shares_held = {
        k: shamir_share_self_secret(self_secrets[k], SHARE_T, list(range(1, n_participants + 1)))
        for k in range(n_participants)
    }
    blinding_self_secrets = {k: new_self_mask_secret() for k in range(n_participants)}
    blinding_shares_held = {
        k: shamir_share_self_secret(blinding_self_secrets[k], SHARE_T, list(range(1, n_participants + 1)))
        for k in range(n_participants)
    }

    submissions, phase2s, masked_vectors = [], {}, {}

    # clients 0..7: one honest submission each.
    for k in range(N_HONEST):
        p1, p2, masked = _one_client_submission(
            k, sks[k], ring, secrets, self_secrets, blinding_self_secrets,
            shares_held, blinding_shares_held, n_participants, q_true[k],
            projector, message_tag=b"janus-f7a-masked:",
        )
        submissions.append((k, p1))
        phase2s[k] = p2
        masked_vectors[k] = masked

    # client 8 (the real Sybil signer): TWO submissions this round,
    # same (sk, ring position, rho_t) -> same key image, different
    # content. "8_sybil" is a harness-only bookkeeping key (see module
    # docstring) -- cryptographically this is the SAME client 8.
    p1_honest, p2_honest, masked_honest = _one_client_submission(
        SYBIL_IDX, sks[SYBIL_IDX], ring, secrets, self_secrets, blinding_self_secrets,
        shares_held, blinding_shares_held, n_participants, q_true[SYBIL_IDX],
        projector, message_tag=b"janus-f7a-masked:genuine:",
    )
    p1_sybil, p2_sybil, masked_sybil = _one_client_submission(
        SYBIL_IDX, sks[SYBIL_IDX], ring, secrets, self_secrets, blinding_self_secrets,
        shares_held, blinding_shares_held, n_participants, q_sybil_duplicate,
        projector, message_tag=b"janus-f7a-masked:sybil-duplicate:",
    )
    assert p1_honest.key_image_round.b == p1_sybil.key_image_round.b, (
        "sanity check failed: both of client 8's submissions must share the SAME "
        "key image (same sk, same ring position, same rho_t) -- if this assertion "
        "fails the attack scenario itself is mis-constructed, not a defense result"
    )

    submissions.append((SYBIL_IDX, p1_honest))
    phase2s[SYBIL_IDX] = p2_honest
    masked_vectors[SYBIL_IDX] = masked_honest

    submissions.append(("8_sybil", p1_sybil))
    phase2s["8_sybil"] = p2_sybil
    masked_vectors["8_sybil"] = masked_sybil

    def unmask_provider(accepted_idx, excluded_idx):
        shares = {k: [shares_held[k][h + 1] for h in accepted_idx] for k in accepted_idx}
        blinding_shares = {k: [blinding_shares_held[k][h + 1] for h in accepted_idx] for k in accepted_idx}
        seeds = {}
        for k in excluded_idx:
            for j in accepted_idx:
                seeds[(j, k)] = pairwise_round_seed(secrets[j][k], RHO_T)
        return UnmaskingMaterial(shares, seeds, SHARE_T, blinding_shares)

    return SimpleNamespace(
        submissions=submissions, phase2s=phase2s, masked_vectors=masked_vectors,
        projector=projector, unmask_provider=unmask_provider, n_participants=n_participants,
    )


def _run(env):
    return run_round(
        env.submissions, env.phase2s, RHO_T, RHO_E, VRF_ALPHA, B_T,
        env.n_participants, MODEL_DIM,
        unmask_provider=env.unmask_provider,
        masked_vector_provider=lambda k: env.masked_vectors[k],
        sketch_dim=SKETCH_M, sigma_t=SIGMA_T, quant_bits=QUANT_BITS, density=DENSITY,
        n_tests=N_TESTS, sketch_projector=env.projector,
    )


def test_sybil_duplicate_rejected_honest_submission_unaffected():
    env = _build_sybil_env()
    outcome = _run(env)

    assert not outcome.aborted, (
        f"round should NOT abort -- 9 submissions, k_min=8 reachable even with "
        f"exactly one of client 8's two rejected. abort_reason={outcome.abort_reason}"
    )

    accepted_indices = set(outcome.accepted_client_indices)
    # exactly one of {8, "8_sybil"} accepted, never both, never neither.
    client_8_variants_accepted = accepted_indices & {SYBIL_IDX, "8_sybil"}
    assert len(client_8_variants_accepted) == 1, (
        f"expected exactly ONE of client 8's two submissions accepted, got "
        f"{client_8_variants_accepted} -- V2 either let both through (double-counts "
        f"one client's contribution) or rejected both (over-rejection, a false positive "
        f"against the honest half of this same client's activity)"
    )

    # the rejected one must be rejected SPECIFICALLY for the double-submission
    # check, not some unrelated accident (e.g. a V1/V3/V4 failure that would
    # mean this test's own honest-submission construction has a bug).
    rejected_variant = ({SYBIL_IDX, "8_sybil"} - client_8_variants_accepted).pop()
    assert "V2_double_submission" in outcome.rejected_reasons[rejected_variant], (
        f"client 8's duplicate was rejected, but not for V2 -- actual reasons: "
        f"{outcome.rejected_reasons[rejected_variant]}"
    )

    # every genuinely honest client (0-7) must be accepted, completely
    # unaffected by client 8 trying to submit twice.
    for k in range(N_HONEST):
        assert k in accepted_indices, f"honest client {k} was rejected -- {outcome.rejected_reasons.get(k)}"


def test_sybil_attack_result_matches_primitive_level_a3_test():
    """Cross-check against attacks/a3_sybil.py's own isolated claim --
    both must agree that exactly 1 of N submissions survives the
    key-image dedup, confirming the real-round result isn't an
    artifact of this test's own harness.
    """
    from attacks.a3_sybil import run_attack

    sk = Scalar.random()
    ring = [scalar_mul_base(Scalar.random()) for _ in range(5)] + [scalar_mul_base(sk)]
    pk_index = len(ring) - 1
    result = run_attack(sk, pk_index, ring, RHO_T, RHO_E, n_submissions=5)
    assert result.defense_triggered, result.detail
