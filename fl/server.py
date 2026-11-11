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
from crypto.pedersen import commit_sum, verify_opening, Commitment
from crypto.secagg import aggregate_masked, P_MASK
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


def check_V3_norm_bound(phase2, commitment_point, B_t: float) -> bool:
    if not verify_range(phase2.range_proof):
        return False
    claimed_commitment = Commitment(commitment_point)
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
    if not check_V3_norm_bound(phase2, submission.commitment_point, B_t):
        reasons.append("V3_norm_bound_exceeded")
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
    published_sum_value: int = None       # weighted_digest(aggregate_masked_vector, rho_t) -- what the server publishes for V5
    published_sum_blinding: "object" = None  # Scalar: sum of revealed r_k's for the accepted set


def run_round(
    submissions: list,     # list[(client_index, ClientSubmissionPhase1)]
    phase2s: dict,          # client_index -> ClientSubmissionPhase2
    rho_t: bytes,
    rho_e: bytes,
    vrf_alpha: bytes,
    B_t: float,
    n_clients: int,
    model_dim: int,
) -> RoundOutcome:
    seen_key_images: set = set()
    accepted = []
    for client_index, submission in submissions:
        phase2 = phase2s[client_index]
        result = verify_submission(submission, phase2, rho_t, rho_e, vrf_alpha, B_t, seen_key_images)
        if result.accepted:
            seen_key_images.add(submission.key_image_round.b)
            accepted.append((client_index, submission))
        # rejected submissions are logged (transcript, verdict) to BB in
        # the real pipeline -- audit logging is verifier/audit.py's job,
        # not re-implemented here to avoid duplicating that concern.

    if len(accepted) < K_MIN:
        return RoundOutcome(aborted=True, abort_reason=f"only {len(accepted)} accepted, k_min={K_MIN}")

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
    # claimed opening of that product -- using the identical
    # weighted_digest(..., seed=rho_t) scheme each client used in
    # client.py's step 5, so the two sides are computed the same way
    # (see crypto/vector_commit.py for why a plain coordinate sum isn't
    # sound here, and why this one still is). Independent verification
    # (not trusting that the server computed this honestly) is
    # verifier/audit.py's independent_v5_check, deliberately a separate
    # implementation -- see that module's docstring.
    commitments = [Commitment(sub.commitment_point) for _, sub in accepted]
    expected = commit_sum(commitments)
    published_sum_value = weighted_digest(aggregate_masked_vector, seed=rho_t)
    published_sum_blinding = None
    for _, sub in accepted:
        r_k = sub._secret_blinding
        published_sum_blinding = r_k if published_sum_blinding is None else published_sum_blinding + r_k

    mean_masked = aggregate_masked_vector.astype(np.float64) / max(len(accepted), 1)
    dequantized_mean_delta = dequantize(aggregate_masked_vector, bits=16) / max(len(accepted), 1)

    return RoundOutcome(
        aggregated_delta=dequantized_mean_delta,
        accepted_client_indices=[idx for idx, _ in accepted],
        aborted=False,
        v5_check_passed=None,  # filled in by verifier/audit.py's independent_v5_check once posted to BB
        sum_commitment_check=expected,
        published_sum_value=published_sum_value,
        published_sum_blinding=published_sum_blinding,
    )


def apply_update(w_t, aggregated_delta):
    """w_{t+1} = w_t + Dequantize(sum q_k)/|A_real| -- already mean-
    divided by run_round, so this is a plain add. TORCH-DEPENDENT: w_t
    is the flat tensor from fl.models.flatten_state_dict.
    """
    import torch

    return w_t + torch.from_numpy(aggregated_delta).float()
