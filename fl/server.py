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
    verify_aggregate_sketch, localize_fault, reconstruct_blinding_sum,
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
                       sigma_t: bytes, quant_bits: int = 16, density: float = 1.0 / 3.0,
                       n_tests: int = 8) -> VerificationResult:
    reasons = []
    if not check_V1_ring_signature(submission, rho_t, rho_e):
        reasons.append("V1_ring_signature_failed")
    if not check_V2_no_double_submission(submission, seen_key_images_this_round):
        reasons.append("V2_double_submission")
    if not check_V3_norm_bound(phase2, B_t, submission.commitment_point, sketch_dim, model_dim,
                                sketch_seed=sigma_t, quant_bits=quant_bits, density=density,
                                n_tests=n_tests):
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
    aggregate_sketch_check_passed: "object" = None  # FIX.md F7e: True = sum of committed sketches matched the aggregate's own sketch; False = mismatch (round aborted); None = check could not be RUN (missing material -- a liveness gap, NOT evidence of cheating)
    aggregate_sketch_bad_coordinates: list = field(default_factory=list)  # F7e: public coordinate indices that mismatched (coordinates of a public sketch, not client identities)
    aggregate_sketch_suspect_groups: list = field(default_factory=list)   # F7f: disjoint client-index groups a failed check was localized to, each >= min_group_size; empty when not localized
    aggregate_quantized: "object" = None  # np.int64 in Z_{P_MASK}: sum of the accepted clients' q_k AFTER unmasking -- the protocol's public output, before dequantization


def _true_aggregate_sketch(aggregate_quantized, sketch_dim: int, model_dim: int, sigma_t: bytes,
                            sketch_projector=None) -> list:
    """S(sum_k q_k) as exact integers, from the post-unmask aggregate.

    Two things that are easy to get wrong here:

    1. The aggregate must be CENTERED before sketching, for the same
       reason each client centers its own q_k (`fl/client.py`'s
       round_step_phase2 docstring): `unmask_aggregate` returns a
       Z_{P_MASK}-reduced vector, so any negative coordinate sits near
       P_MASK. `center_mod_p_mask(sum_k q_k)` equals `sum_k
       center_mod_p_mask(q_k)` exactly as long as the true signed sum
       stays inside (-P_MASK/2, P_MASK/2] -- true by a wide margin for
       honest clients (n * 2^(quant_bits-1) is ~2^21 at n=50 against
       ~2^60), and its failure is precisely the wraparound this check
       exists to catch, so the equality is an assumption about HONEST
       behaviour only, not about the adversary.
    2. The projector must be the SAME CLASS the clients used.
       `TorchSketchProjector` and `SketchProjector` generate different
       sign matrices from the same seed (torch's RNG stream is not
       NumPy's -- stated outright in crypto/sketch.py), so mixing them
       makes every coordinate mismatch. `sketch_projector` is injectable
       only so a test can pass the NumPy reference for both sides; a
       real round leaves it None and gets the torch one, matching
       `round_step_phase2`.
    """
    from .client import center_mod_p_mask
    if sketch_projector is None:
        from crypto.sketch import TorchSketchProjector
        sketch_projector = TorchSketchProjector(seed=sigma_t, m=sketch_dim, d=model_dim)
    centered = center_mod_p_mask(np.asarray(aggregate_quantized, dtype=np.int64))
    return [int(v) for v in sketch_projector.project_int(centered)]


def _check_F7e_aggregate_sketch(phase2s: dict, accepted_indices: list, aggregate_quantized,
                                 excluded_pair_seeds: dict, sketch_dim: int, model_dim: int,
                                 sigma_t: bytes, rho_t: bytes,
                                 blinding_self_mask_shares: dict, share_threshold: int,
                                 sketch_projector=None,
                                 blame_material_provider=None):
    """Returns (ok, bad_coordinates, suspect_groups, detail).

    `ok` is None (not False) when the check could not be RUN at all --
    a submission missing its masked blinding vector, or a blinding-sum
    reconstruction that raised because released seeds were incomplete.
    That is a liveness failure, not evidence of cheating, and must not
    be reported as a failed integrity check: the caller treats None as
    "not checked this round" and proceeds, exactly as it did before F7e
    existed. Only a reconstruction that SUCCEEDS and then disagrees with
    the committed sketches returns False.
    """
    if not accepted_indices:
        return None, [], [], "no accepted clients"
    masked_by_client = {}
    for k in accepted_indices:
        row = getattr(phase2s[k], "masked_y_blindings", None)
        if row is None:
            return None, [], [], f"client {k} published no masked blinding vector (pre-F7e submission?)"
        masked_by_client[k] = row
    try:
        sum_blindings = reconstruct_blinding_sum(
            masked_by_client, accepted_indices, excluded_pair_seeds, sketch_dim,
            blinding_self_mask_shares, share_threshold, rho_t,
        )
    except (ValueError, KeyError) as e:
        return None, [], [], f"blinding-sum reconstruction unavailable: {e}"

    true_sketch = _true_aggregate_sketch(aggregate_quantized, sketch_dim, model_dim, sigma_t, sketch_projector)
    y_commitments_by_client = [phase2s[k].sketch_commitment.y_commitments for k in accepted_indices]
    ok, bad = verify_aggregate_sketch(y_commitments_by_client, sum_blindings, true_sketch)
    if ok:
        return True, [], [], "all coordinates matched"

    # ---- F7f: localize.
    #
    # The subgroup test has to compare a subgroup's committed sketch sum
    # against that subgroup's OWN true partial aggregate -- and for a
    # strict subset the server cannot get that from `aggregate_quantized`,
    # which is only the whole accepted set. It can get it another way:
    # once the accepted set's self-mask shares are released (they already
    # are -- that is how `unmask_aggregate` works at all), the only thing
    # still hiding an individual q_k is its pairwise masks, so a
    # subgroup's partial aggregate becomes computable as soon as the
    # clients in that subgroup release their pair seeds about the
    # accepted clients OUTSIDE it. That is `unmask_aggregate` run on the
    # subgroup, and it is what `blame_material_provider` supplies.
    #
    # THAT IS THE PRIVACY COST, and it is the real reason
    # `min_group_size` exists: every bisection step hands the server one
    # more partial aggregate, so bisecting all the way down would hand it
    # one client's update in the clear. `localize_fault` stops at
    # min_group_size, so the smallest thing the server ever learns is the
    # summed update of that many clients -- ARCHITECTURE.md §6.3's
    # accountability-anonymity budget, spent explicitly, and only on a
    # round that has already failed an integrity check.
    #
    # Without a provider there is nothing to bisect with, so the failure
    # stays unattributed and SAYS so, rather than looking investigated.
    if blame_material_provider is None:
        return False, bad, [], (
            f"{len(bad)} mismatched coordinate(s), first={bad[:4]}; not localized "
            f"(no blame_material_provider -- F7f needs the accepted clients in a "
            f"subgroup to release pair seeds about those outside it, see "
            f"fl/client.py::release_straddling_pair_seeds)"
        )

    # One coordinate is enough to implicate a group; bisecting on every
    # mismatched coordinate would spend more privacy budget for no extra
    # attribution.
    coordinate = bad[0]

    def _group_opens_correctly(group: list) -> bool:
        group = list(group)
        material = blame_material_provider(group)
        if material is None:
            return False  # cannot verify this subgroup -> leave it suspect; never clear a group on missing evidence
        partial_q, straddling_seeds, group_blinding_self_mask_shares = material
        try:
            sub_blindings = reconstruct_blinding_sum(
                masked_by_client, group, straddling_seeds, sketch_dim,
                group_blinding_self_mask_shares, share_threshold, rho_t,
            )
        except (ValueError, KeyError):
            return False
        sub_true = _true_aggregate_sketch(partial_q, sketch_dim, model_dim, sigma_t, sketch_projector)
        sub_commitments = [[phase2s[k].sketch_commitment.y_commitments[coordinate]] for k in group]
        sub_ok, _ = verify_aggregate_sketch(
            sub_commitments, [sub_blindings[coordinate]], [sub_true[coordinate]],
        )
        return sub_ok

    blame = localize_fault(list(accepted_indices), coordinate, sigma_t, _group_opens_correctly)
    return False, bad, blame.suspect_groups, (
        f"{len(bad)} mismatched coordinate(s), first={bad[:4]}; localized on coordinate "
        f"{coordinate} to {len(blame.suspect_groups)} suspect group(s) in "
        f"{blame.rounds_used} bisection step(s)"
    )


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
    blinding_self_mask_shares: dict = field(default_factory=dict)  # FIX.md F7e dropout resilience: {accepted k: [ShamirShare, ...]} for the SEPARATE blinding self-mask secret


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
    n_tests: int = 8,             # FIX.md F7d: MUST match round_step_phase2's own n_tests, or the server derives a different number of public weight vectors than the client proved against and every honest client fails V3. Threaded explicitly rather than left as two independent hardcoded 8s.
    sketch_projector=None,        # FIX.md F7e: MUST be the same projector class the clients used (see _true_aggregate_sketch); None -> TorchSketchProjector, matching round_step_phase2
    blame_material_provider=None,  # FIX.md F7f: callable(group) -> (partial_q, straddling_pair_seeds, group_blinding_self_mask_shares) or None. Omit to abort without attribution on an F7e failure.
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
                                    quant_bits=quant_bits, density=density, n_tests=n_tests)
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
    # FIX.md F7f: `excluded_indices` must cover every OTHER configured
    # client (0..n_clients-1), not just the ones that showed up in
    # `submissions`. A client's masked vector carries pairwise mask terms
    # against every partner it has a pairwise secret for -- ALL n_clients-1
    # others, computed once up front, regardless of whether a given
    # partner submitted anything this round (e.g. sitting out a
    # suspect-group quarantine). Deriving this from `submissions` alone
    # would silently leave those uncancelled pairwise terms in the
    # aggregate whenever `submissions` is a strict subset of the full
    # client set -- `n_clients` (previously an unused parameter here) is
    # what makes that distinction available.
    excluded_indices = [idx for idx in range(n_clients) if idx not in accepted_set]
    material = unmask_provider(accepted_indices, excluded_indices)
    try:
        aggregate_quantized = unmask_aggregate(
            aggregate_masked_vector, accepted_indices, material.self_mask_shares,
            material.share_threshold, material.excluded_pair_seeds, rho_t, model_dim,
        )
    except ValueError as e:
        return RoundOutcome(aborted=True, abort_reason=f"unmasking failed: {e}", rejected_reasons=rejected_reasons)

    # ---- F7e: does the sum of the accepted clients' COMMITTED sketches
    # equal the sketch of the aggregate that was actually unmasked?
    #
    # V3 (per client) only constrains each sketch's own internal
    # algebra and its norm; nothing there forces client k's sketch to be
    # the sketch of the vector k actually contributed. This check does,
    # in aggregate: sum_k y_k must equal S(sum_k q_k). A client that
    # sketched a different (e.g. P_MASK-wrapped) representative than it
    # masked shows up here as a per-coordinate mismatch.
    #
    # Runs only after unmasking, because the true aggregate is the input
    # (the F1/F5 dependency the original F7 plan already flagged).
    sketch_ok, bad_coords, blame_groups, sketch_detail = _check_F7e_aggregate_sketch(
        phase2s, accepted_indices, aggregate_quantized, material.excluded_pair_seeds,
        sketch_dim, model_dim, sigma_t, rho_t,
        material.blinding_self_mask_shares, material.share_threshold,
        sketch_projector, blame_material_provider,
    )
    if sketch_ok is False:
        # ABORT rather than apply: an aggregate that fails this check is
        # not the sum of what was committed, which is the same class of
        # failure V5 aborts on. Unlike the pre-F7f behaviour, the abort
        # now carries attribution (`blame_groups`) when a seed provider
        # was supplied, so a repeat offender can be excluded instead of
        # DoS-ing every round anonymously. Acting on that is the
        # caller's policy decision, deliberately not made here.
        return RoundOutcome(
            aborted=True,
            abort_reason=f"F7e aggregate sketch check failed: {sketch_detail}",
            rejected_reasons=rejected_reasons,
            accepted_client_indices=accepted_indices,
            aggregate_quantized=aggregate_quantized,
            sum_commitment_check=expected,
            published_sum_value=published_sum_value,
            published_sum_blinding=published_sum_blinding,
            aggregate_sketch_check_passed=False,
            aggregate_sketch_bad_coordinates=bad_coords,
            aggregate_sketch_suspect_groups=blame_groups,
        )

    dequantized_mean_delta = dequantize(aggregate_quantized, bits=16) / max(len(accepted), 1)

    return RoundOutcome(
        aggregate_sketch_check_passed=sketch_ok,
        aggregate_sketch_bad_coordinates=bad_coords,
        aggregate_sketch_suspect_groups=blame_groups,
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
