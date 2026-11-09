"""A3: Sybil / multi-submission under anonymity (ARCHITECTURE.md §10).
One client submits 5 updates in the same round, hoping ring-signature
anonymity hides that they're all from the same signer. Defense under
test: V2 -- the per-ROUND key image I_k^t is deterministic given
(sk_k, rho_t), so submitting twice in the same round produces the
SAME key image twice, which crypto.clsag.same_round_same_signer (and
server.py's check_V2_no_double_submission) detects even though the
ring signatures themselves give no other link between the two
submissions. "This attack exists only because of ring signatures -- it
is what justifies linkability" (ARCHITECTURE.md). EC-DEPENDENT.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import AttackResult
from crypto.clsag import sign as clsag_sign, verify_with_scopes, same_round_same_signer
from fl.server import check_V2_no_double_submission


@dataclass
class _FakeSubmission:
    """Minimal stand-in exposing only what check_V2_no_double_submission
    reads (submission.key_image_round) -- avoids constructing a full
    fl.client.ClientSubmissionPhase1 (which needs the whole phase-1
    pipeline) just to exercise the V2 dedup check in isolation.
    """
    key_image_round: object


def run_attack(sk, pk_index: int, ring: list, rho_t: bytes, rho_e: bytes, n_submissions: int = 5):
    """Same signer crafts n_submissions distinct messages (e.g.
    different, cherry-picked fake updates) but must reuse the same
    (sk, rho_t) for every one of them within this round -- that's what
    makes I_k^t identical across all n_submissions, by construction of
    crypto/clsag.py's key image (I_t = sk*H_p(rho_t), independent of
    the message).
    """
    sigs = []
    for i in range(n_submissions):
        message = f"fake-update-{i}".encode()
        sig = clsag_sign(sk, pk_index, ring, message, rho_t, rho_e)
        sigs.append((message, sig))

    # sanity: every signature is individually VALID (ring signatures
    # don't care how many times a key is used) -- the attack is caught
    # at the LINKING layer (V2), not at signature verification (V1).
    all_individually_valid = all(verify_with_scopes(sig, msg, rho_t, rho_e) for msg, sig in sigs)

    key_images = [sig.key_image_round for _, sig in sigs]
    all_same_key_image = all(same_round_same_signer(key_images[0], ki) for ki in key_images[1:])

    seen = set()
    accepted_count = 0
    for _, sig in sigs:
        if check_V2_no_double_submission(_FakeSubmission(sig.key_image_round), seen):
            seen.add(sig.key_image_round.b)
            accepted_count += 1

    defense_triggered = (accepted_count == 1) and all_individually_valid and all_same_key_image
    return AttackResult(
        "A3_sybil", defense_triggered=defense_triggered, defense_name="V2_key_image_dedup",
        detail=f"{n_submissions} submissions, {accepted_count} accepted (expect exactly 1), "
               f"all_valid_sigs={all_individually_valid}, same_key_image={all_same_key_image}",
    )
