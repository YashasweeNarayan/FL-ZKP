"""Server-side round orchestration (ARCHITECTURE.md §4: B_t derivation,
checks V1-V4, k_min gate, aggregation, V5 publication). TORCH-DEPENDENT
only for the final Dequantize + weight update step; every verification
check is pure crypto-call wiring over already-tested primitives. Not
run in this sandbox -- validate on Colab per STATUS.md.

Deliberately mirrors ARCHITECTURE.md's round protocol pseudocode
structure 1:1 (same step order, same check names V1-V5) so a reviewer
can line this file up against the spec directly instead of having to
reverse-engineer the mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crypto.clsag import verify_with_scopes, same_round_same_signer
from crypto.rangeproof import verify_range, verify_recomposition, recompose_commitment
from crypto.r_cons import verify_identity_binding
from crypto.pedersen import commit_sum, verify_opening, commit as pedersen_commit, Commitment
from crypto.ristretto import Scalar
from crypto.secagg import aggregate_masked, unmask_aggregate, P_MASK
from crypto.vector_commit import weighted_digest
from crypto.norm_proof import (
    verify_sketch_commitment, subset_sum_challenge, wraparound_window, verify_wraparound_binding,
    verify_aggregate_sketch, localize_fault,
)
from crypto.sketch import int_sketch_norm_sq_bound

from .norm_rule import compute_threshold, NormThreshold
from .client import dequantize

K_MIN = 8  # ARCHITECTURE.md §4: "|A_real| >= k_min = 8 else ABORT the round"


@dataclass
class VerificationResult:
    accepted: bool
    reasons_rejected: list = field(default_factory=list)


def derive_B_t(previous_round_committed_norms: np.ndarray, kappa: float = 1.0) -> NormThreshold:
    """C5: 'B_t is NOT chosen by S: B_t = median{committed norms posted
    on BB at round t-1} * kappa, recomputable by anyone from BB' --
    this is exactly fl/norm_rule.py's compute_threshold with k folded
    into kappa; kept as a thin named wrapper so server.py's call sites
    read the same way ARCHITECTURE.md's pseudocode does.
    """
    return compute_threshold(previous_round_committed_norms, k=kappa)


def check_V1_ring_signature(submission, rho_t: bytes, rho_e: bytes) -> bool:
    """FIX.md F7a: the signed message now includes a hash of the
    client's masked vector (`submission.masked_vector_hash`), so this
    check (unchanged in mechanism -- still CLSAG verification) now
    ALSO attests that `masked_vector_hash` genuinely belongs with this
    c_k/signature pair. A submission from before F7a (no
    `masked_vector_hash`, the dataclass default b"") will fail here,
    which is correct: it cannot be re-verified against the new message
    shape, not silently accepted under the old one.
    """
    sig = submission.ring_signature
    message = (
        submission.commitment_point.b
        + submission.escrow_ciphertext.c1.b
        + submission.escrow_ciphertext.c2.b
        + submission.masked_vector_hash
    )
    return verify_with_scopes(sig, message, rho_t, rho_e)


def check_V2_no_double_submission(submission, seen_key_images_this_round: set) -> bool:
    """I_k^t not in seen(rho_t) -- the key image is itself hashable
    (its `.b` bytes), so 'seen' is tracked by that byte string.
    """
    tag = submission.key_image_round.b
    return tag not in seen_key_images_this_round


def check_V3_norm_bound(phase2, B_t: float, commitment_point, sketch_dim: int, model_dim: int,
                         sketch_seed: bytes, quant_bits: int = 16, density: float = 1.0 / 3.0,
                         n_tests: int = 8) -> bool:
    """FIX.md F7 (V3 redesign): the server NEVER sees a cleartext
    sketch or any individual sketch coordinate now. Three independent
    things have to hold:

    1. `phase2.sketch_commitment` verifies on its own terms
       (`crypto.norm_proof.verify_sketch_commitment`): every
       per-coordinate square proof is valid, and the one PUBLIC scalar
       v = <y,y> it reveals is within the norm bound B_t implies
       (`int_sketch_norm_sq_bound`). This replaces the old cleartext
       `np.linalg.norm(phase2.sketch)` call -- same check, hidden input.
    2. Each of the `n_tests` wraparound bindings verifies against
       `phase2.sketch_commitment.y_commitments` under this round's
       PUBLICLY-derived weights and window (`subset_sum_challenge`,
       `wraparound_window` -- both pure functions of public round/config
       data, recomputed here rather than trusted from the client).
    3. Each corresponding range proof is well-formed (catches a client
       claiming an out-of-window flooded sum).

    `commitment_point` (c_k) and `sketch_seed` (sigma_t) must be the
    SAME values `round_step_phase2` used to build its transcript
    (`b"norm_proof:" + phase1.commitment_point.b`) and its weights
    (`subset_sum_challenge(seed=sketch_seed, ...)`) -- a mismatch on
    either fails every honest client, not just a malicious one.
    """
    transcript = b"norm_proof:" + commitment_point.b
    norm_bound_sq = int_sketch_norm_sq_bound(B_t, sketch_dim, density)
    if not verify_sketch_commitment(phase2.sketch_commitment, transcript, norm_bound_sq):
        return False
    weights_list = subset_sum_challenge(seed=sketch_seed, m=sketch_dim, n_tests=n_tests)
    window = wraparound_window(sketch_dim, model_dim, density, quant_bits)
    if not (len(phase2.wraparound_range_proofs) == len(phase2.wraparound_bindings) == len(weights_list)):
        return False
    for j, w in enumerate(weights_list):
        rp = phase2.wraparound_range_proofs[j]
        bp = phase2.wraparound_bindings[j]
        if not verify_range(rp):
            return False
        tx = transcript + b":wrap:" + j.to_bytes(4, "little")
        C_j = recompose_commitment(rp)
        if not verify_wraparound_binding(C_j, bp, w, phase2.sketch_commitment.y_commitments, window, tx):
            return False
    return True


def check_V4_consistency(phase2, rho_t: bytes, rho_e: bytes, vrf_alpha: bytes, commitment_point) -> bool:
    transcript = b"r_cons:" + commitment_point.b
    return verify_identity_binding(phase2.identity_binding_proof, rho_t, rho_e, vrf_alpha, transcript)


def verify_submission(submission, phase2, rho_t: bytes, rho_e: bytes, vrf_alpha: bytes, B_t: float,
                       seen_key_images_this_round: set, sketch_dim: int, model_dim: int,
                       sigma_t: bytes, quant_bits: int = 16, density: float = 1.0 / 3.0) -> VerificationResult:
    reasons = []
    if not check_V1_ring_signature(submission, rho_t, rho_e):
        reasons.append("V1_ring_signature_failed")
    if not check_V2_no_double_submission(submission, seen_key_images_this_round):
        reasons.append("V2_double_submission")
    if not check_V3_norm_bound(phase2, B_t, submission.commitment_point, sketch_dim, model_dim,
                                sketch_seed=sigma_t, quant_bits=quant_bits, density=density):
        # Diagnostic detail, not just the bare check name: v is PUBLIC
        # (unlike the hidden y itself) so it is safe and useful to log,
        # same spirit as the old cleartext-norm diagnostic this replaces.
        norm_bound_sq = int_sketch_norm_sq_bound(B_t, sketch_dim, density)
        reasons.append(
            f"V3_norm_bound_exceeded (v={phase2.sketch_commitment.v}, "
            f"norm_bound_sq={norm_bound_sq}, B_t={B_t:.3f})"
        )
    if not check_V4_consistency(phase2, rho_t, rho_e, vrf_alpha, submission.commitment_point):
        reasons.append("V4_consistency_failed")
    return VerificationResult(accepted=(len(reasons) == 0), reasons_rejected=reasons)


@dataclass
class RoundOutcome:
    aggregated_delta: "object" = None      # torch tensor, dequantized mean delta
    accepted_client_indices: list = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""
    v5_check_passed: bool = None
    sum_commitment_check: "object" = None
    published_sum_value: int = None       # exact sum of each accepted client's own weighted_digest(masked_k, rho_t) -- what the server publishes for V5; may exceed P_MASK (see run_round's comment on why it must not be re-reduced mod P_MASK)
    published_sum_blinding: "object" = None  # Scalar: sum of revealed r_k's for the accepted set
    rejected_reasons: dict = field(default_factory=dict)  # client_index -> list[str], e.g. ["V3_norm_bound_exceeded"]
    aggregate_quantized: "object" = None  # np.int64 in Z_{P_MASK}: sum of the accepted clients' q_k AFTER unmasking -- the protocol's public output, before dequantization


@dataclass
class UnmaskingMaterial:
    """What the accepted clients release once the accepted set is fixed
    (crypto/secagg.py's unmasking round). Collected by the caller
    through `unmask_provider`, because WHICH material is needed depends
    on the accepted set this function computes.
    """
    self_mask_shares: dict       # {accepted k: [ShamirShare, ...]}
    excluded_pair_seeds: dict    # {(accepted j, excluded k): round_seed bytes}
    share_threshold: int


def run_round(
    submissions: list,     # list[(client_index, ClientSubmissionPhase1)]
    phase2s: dict,          # client_index -> ClientSubmissionPhase2
    rho_t: bytes,
    rho_e: bytes,
    vrf_alpha: bytes,
    B_t: float,
    n_clients: int,
    model_dim: int,
    unmask_provider,        # callable(accepted_indices, excluded_indices) -> UnmaskingMaterial
    masked_vector_provider=None,  # callable(client_index) -> x_k; see the streaming note below (FIX.md F5)
    sketch_dim: int = None,       # FIX.md F7: sketch_m -- required, kept keyword-with-None so old
    sigma_t: bytes = None,        # positional call sites fail loudly instead of silently mis-seeding V3
    quant_bits: int = 16,
    density: float = 1.0 / 3.0,
) -> RoundOutcome:
    if sketch_dim is None or sigma_t is None:
        raise ValueError(
            "run_round needs sketch_dim and sigma_t (FIX.md F7: check_V3_norm_bound "
            "verifies the committed sketch, which needs the round's sketch dimension "
            "and seed to recompute the same public weights/window the client proved against)"
        )
    # Default provider: read the vector off the submission itself, the
    # pre-F5 behaviour. `fl/round_loop.py` passes a provider backed by
    # the round's vector store instead, so the submissions carry no
    # full-length arrays at all.
    if masked_vector_provider is None:
        by_index = {idx: sub for idx, sub in submissions}

        def masked_vector_provider(idx):
            vec = by_index[idx]._secret_masked_vector
            if vec is None:
                raise ValueError(
                    f"client {idx}: no masked vector on the submission and no "
                    f"masked_vector_provider given -- phase 1 stored it in a "
                    f"vector store, so run_round needs that store's provider"
                )
            return vec
    seen_key_images: set = set()
    accepted = []
    rejected_reasons = {}
    for client_index, submission in submissions:
        phase2 = phase2s[client_index]
        result = verify_submission(submission, phase2, rho_t, rho_e, vrf_alpha, B_t, seen_key_images,
                                    sketch_dim=sketch_dim, model_dim=model_dim, sigma_t=sigma_t,
                                    quant_bits=quant_bits, density=density)
        if result.accepted:
            seen_key_images.add(submission.key_image_round.b)
            accepted.append((client_index, submission))
        else:
            rejected_reasons[client_index] = result.reasons_rejected
        # rejected submissions are logged (transcript, verdict) to BB in
        # the real pipeline -- audit logging is verifier/audit.py's job,
        # not re-implemented here to avoid duplicating that concern.
        # rejected_reasons above is the minimal in-memory version of
        # that, so a round loop's caller can actually see WHY a
        # submission failed instead of just a bare accepted/rejected count.

    if len(accepted) < K_MIN:
        return RoundOutcome(
            aborted=True,
            abort_reason=f"only {len(accepted)} accepted, k_min={K_MIN}",
            rejected_reasons=rejected_reasons,
        )

    # FIX.md F5: the masked vectors are STREAMED, one client at a time,
    # and the two things computed from them -- the coordinate-wise sum
    # and each client's own digest -- are produced in the SAME pass, so
    # each vector is fetched once and dropped before the next is
    # fetched. Peak here is the accumulator plus one vector, i.e. O(d),
    # independent of how many clients were accepted. Holding the list
    # was 10.6 GB at 30 clients and crashed the runtime on 2026-11-12.
    #
    # This is also what a real server does: process each submission as
    # it arrives, accumulate, discard. `masked_vector_provider(idx)` is
    # the seam -- in this simulation it reads from the round's vector
    # store; in a deployment it would be the arriving message.
    #
    # The digest MUST be computed here from the vector the server
    # actually received, not taken from a value the client supplied:
    # recomputing it is precisely what binds x_k to the commitment c_k
    # that V5 checks. A client-supplied digest would always match its
    # own c_k by construction, so V5 would pass for any x_k whatsoever.
    aggregate_masked_vector = None
    published_sum_value = 0
    for client_index, _ in accepted:
        x_k = masked_vector_provider(client_index)
        aggregate_masked_vector = (
            np.array(x_k, dtype=np.int64) % P_MASK if aggregate_masked_vector is None
            else (aggregate_masked_vector + x_k) % P_MASK
        )
        published_sum_value += weighted_digest(x_k, seed=rho_t)
        del x_k

    # V5: anyone can recompute Pi c_k and compare against the server's
    # published aggregate-opening. `expected` is the LHS (product of the
    # publicly posted per-client commitments); `published_sum_value` /
    # `published_sum_blinding` are what the server PUBLISHES as the
    # claimed opening of that product. Independent verification (not
    # trusting that the server computed this honestly) is
    # verifier/audit.py's independent_v5_check, deliberately a separate
    # implementation -- see that module's docstring.
    #
    # published_sum_value is the EXACT sum of each accepted client's
    # own digest (sum(weighted_digest(masked_k, rho_t) for k in A)),
    # NOT weighted_digest(aggregate_masked_vector, rho_t) -- those two
    # are only CONGRUENT mod P_MASK, not equal as integers, and
    # Pedersen's binding property needs exact equality. Pi(c_k) commits
    # to the EC-group sum of each client's digest d_k as a plain
    # integer (EC/Pedersen addition never reduces mod P_MASK, only mod
    # the much larger group order L, and N*P_MASK << L for any
    # realistic client count, so no wraparound happens there). But
    # `aggregate_masked_vector` is already reduced mod P_MASK
    # COORDINATE-WISE by the streaming accumulation above (same
    # reduction `aggregate_masked` applies) -- required for SecAgg's
    # masks to cancel correctly -- so digesting it afterward reduces
    # the *weighted sum of already-wrapped coordinates*, which differs
    # from the exact sum(d_k) by a multiple of P_MASK as soon as more
    # than one client's raw digest would have summed past P_MASK (near
    # certain with >=2 clients, each d_k ~uniform over [0, P_MASK)).
    # Caught live on Colab: V3 finally passed (8/8 accepted) once the
    # earlier norm-bound bugs were fixed, only for V5 to fail every
    # round with exactly this "does not open to Pi(c_k)" mismatch.
    commitments = [Commitment(sub.commitment_point) for _, sub in accepted]
    expected = commit_sum(commitments)

    # Clients reveal r_k only AFTER acceptance (see client.py's
    # reveal_blinding_after_acceptance) -- this function assumes the
    # caller already collected those and attached them, since the
    # reveal is an out-of-band step gated on this function's own output
    # in a real deployment (chicken-and-egg only in sequence diagrams,
    # not in the actual control flow: accept first, THEN ask for r_k).
    published_sum_blinding = None
    for _, sub in accepted:
        r_k = sub._secret_blinding
        published_sum_blinding = r_k if published_sum_blinding is None else published_sum_blinding + r_k

    # Unmasking round. `aggregate_masked_vector` is NOT the aggregate:
    # it still carries every accepted client's self-mask, plus the
    # pairwise masks accepted clients share with excluded ones (those
    # only cancel when BOTH partners are summed). Until 2026-11-13 this
    # function dequantized it directly, which handed uniformly random
    # values (~1e12 after dequantization) to apply_update every round.
    # V5 above is unaffected and stays on the MASKED vectors: it checks
    # that what was summed is what was committed, not what it decodes to.
    accepted_indices = [idx for idx, _ in accepted]
    accepted_set = set(accepted_indices)
    excluded_indices = [idx for idx, _ in submissions if idx not in accepted_set]
    material = unmask_provider(accepted_indices, excluded_indices)
    try:
        aggregate_quantized = unmask_aggregate(
            aggregate_masked_vector, accepted_indices, material.self_mask_shares,
            material.share_threshold, material.excluded_pair_seeds, rho_t, model_dim,
        )
    except ValueError as e:
        return RoundOutcome(aborted=True, abort_reason=f"unmasking failed: {e}", rejected_reasons=rejected_reasons)

    dequantized_mean_delta = dequantize(aggregate_quantized, bits=16) / max(len(accepted), 1)

    return RoundOutcome(
        aggregated_delta=dequantized_mean_delta,
        accepted_client_indices=accepted_indices,
        aborted=False,
        aggregate_quantized=aggregate_quantized,
        v5_check_passed=None,  # filled in by verifier/audit.py's independent_v5_check once posted to BB
        sum_commitment_check=expected,
        published_sum_value=published_sum_value,
        published_sum_blinding=published_sum_blinding,
        rejected_reasons=rejected_reasons,
    )


def apply_update(w_t, aggregated_delta):
    """w_{t+1} = w_t + Dequantize(sum q_k)/|A_real| -- already mean-
    divided by run_round, so this is a plain add. TORCH-DEPENDENT: w_t
    is the flat tensor from fl.models.flatten_state_dict.
    """
    import torch

    return w_t + torch.from_numpy(aggregated_delta).float()
