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
from crypto.rangeproof import verify_range, verify_recomposition
from crypto.r_cons import verify_identity_binding
from crypto.pedersen import commit_sum, verify_opening, commit as pedersen_commit, Commitment
from crypto.ristretto import Scalar
from crypto.secagg import aggregate_masked, unmask_aggregate, P_MASK
from crypto.vector_commit import weighted_digest

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
    sig = submission.ring_signature
    message = submission.commitment_point.b + submission.escrow_ciphertext.c1.b + submission.escrow_ciphertext.c2.b
    return verify_with_scopes(sig, message, rho_t, rho_e)


def check_V2_no_double_submission(submission, seen_key_images_this_round: set) -> bool:
    """I_k^t not in seen(rho_t) -- the key image is itself hashable
    (its `.b` bytes), so 'seen' is tracked by that byte string.
    """
    tag = submission.key_image_round.b
    return tag not in seen_key_images_this_round


def check_V3_norm_bound(phase2, B_t: float) -> bool:
    """Checks the client's range proof is well-formed AND is bound to
    the ACTUAL norm of its publicly-sent sketch `phase2.sketch` --
    not, as an earlier version of this function did, to `c_k`
    (`submission.commitment_point`). `c_k` is a Pedersen commitment to
    `weighted_digest(masked, rho_t)` with an independent random
    blinding (see fl/client.py step 5) -- a different value AND a
    different blinding than `prove_range`'s (norm_estimate,
    agg_blinding). Checking recomposition against `c_k` therefore
    compared two unrelated commitments and could only ever succeed by
    astronomical coincidence, rejecting every honest client every
    round (caught via live Colab testing: "0 accepted" for all 5-8
    clients regardless of model).

    The server independently recomputes the claimed norm from
    `phase2.sketch` (sent in cleartext in this reference
    implementation -- crypto/sketch.py and crypto/rangeproof.py are
    both explicitly flagged elsewhere as correctness references, not
    a production norm-hiding Bulletproofs-over-vector-commitment
    construction) and re-derives the SAME commitment the client's
    `prove_range` call is tied to via `agg_blinding`, so recomposition
    is a real check again: it fails if the client's range proof does
    not actually decompose to the sketch's own norm.
    """
    if not verify_range(phase2.range_proof):
        return False
    n_bits = phase2.range_proof.n_bits
    norm_estimate = int(np.linalg.norm(phase2.sketch))
    if norm_estimate < 0 or norm_estimate >= (1 << n_bits):
        # Genuinely outside what this proof can attest to. fl/client.py's
        # round_step_phase2 clamps its OWN prove_range call to avoid a
        # crash when norm_estimate >= 2**n_bits; rejecting that case here
        # (rather than independently re-clamping to match) prevents a
        # client whose real norm exceeds the provable range from having
        # it silently capped-and-accepted.
        return False
    claimed_commitment = pedersen_commit(Scalar.from_int(norm_estimate), phase2.range_proof_aggregate_blinding)
    return verify_recomposition(phase2.range_proof, claimed_commitment)


def check_V4_consistency(phase2, rho_t: bytes, rho_e: bytes, vrf_alpha: bytes, commitment_point) -> bool:
    transcript = b"r_cons:" + commitment_point.b
    return verify_identity_binding(phase2.identity_binding_proof, rho_t, rho_e, vrf_alpha, transcript)


def verify_submission(submission, phase2, rho_t: bytes, rho_e: bytes, vrf_alpha: bytes, B_t: float, seen_key_images_this_round: set) -> VerificationResult:
    reasons = []
    if not check_V1_ring_signature(submission, rho_t, rho_e):
        reasons.append("V1_ring_signature_failed")
    if not check_V2_no_double_submission(submission, seen_key_images_this_round):
        reasons.append("V2_double_submission")
    if not check_V3_norm_bound(phase2, B_t):
        # Diagnostic detail, not just the bare check name -- computed
        # independently here (cheap, read-only) purely for the
        # rejected_reasons message, so a future "every client fails
        # V3" report is immediately diagnosable instead of requiring
        # another round of guessing: which of range-well-formed /
        # out-of-provable-range / recomposition-mismatch it was, and
        # the actual numbers involved.
        range_ok = verify_range(phase2.range_proof)
        n_bits = phase2.range_proof.n_bits
        norm_estimate = int(np.linalg.norm(phase2.sketch))
        reasons.append(
            f"V3_norm_bound_exceeded (range_proof_valid={range_ok}, "
            f"norm_estimate={norm_estimate}, B_t={B_t:.3f}, n_bits={n_bits}, "
            f"provable_cap={(1 << n_bits) - 1})"
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
) -> RoundOutcome:
    seen_key_images: set = set()
    accepted = []
    rejected_reasons = {}
    for client_index, submission in submissions:
        phase2 = phase2s[client_index]
        result = verify_submission(submission, phase2, rho_t, rho_e, vrf_alpha, B_t, seen_key_images)
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

    # Clients reveal r_k only AFTER acceptance (see client.py's
    # reveal_blinding_after_acceptance) -- this function assumes the
    # caller already collected those and attached them, since the
    # reveal is an out-of-band step gated on this function's own output
    # in a real deployment (chicken-and-egg only in sequence diagrams,
    # not in the actual control flow: accept first, THEN ask for r_k).
    masked_vectors = [sub._secret_masked_vector for _, sub in accepted]
    aggregate_masked_vector = aggregate_masked(masked_vectors)

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
    # COORDINATE-WISE by `aggregate_masked` -- required for SecAgg's
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
    published_sum_value = sum(weighted_digest(sub._secret_masked_vector, seed=rho_t) for _, sub in accepted)
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
