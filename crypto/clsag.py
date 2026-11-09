"""CLSAG-style linkable ring signature with DUAL-SCOPE tags (ARCHITECTURE.md
§3 P4, §5.1).

EC-DEPENDENT — see ristretto.py's header for the Colab-validation note.

Standard CLSAG (Goodell-Noether) proves "I know sk for some pk in the
ring" with ONE key image I = sk*H_p(scope). JANUS-FL needs TWO
independently-scoped tags from the SAME signature (§5.1's fix for the
accumulate-vs-unlink contradiction, D1 in REVIEW.md):
    I_k^t = sk * H_p(rho_t)   -- per-round, for double-submission detection
    P_k^e = sk * H_p(rho_e)   -- per-epoch, for evidence accumulation

Construction: an AOS/LSAG-style ring proof-of-knowledge where EVERY step
of the ring equation carries three parallel terms (ring-membership,
round-tag, epoch-tag) built from the SAME per-step scalars. That's what
forces I_t and I_e to share the real signer's sk -- not just "look
plausible independently" -- without a separate proof.

Reference implementation: O(n) scalar mults per sign/verify, one
Fiat-Shamir hash per ring member. Real CLSAG batches this more
efficiently; that's a performance gap only (flagged in STATUS.md), not
a security one -- the anonymity guarantee (computational
indistinguishability of the signer within the ring) is the same.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ristretto import Point, Scalar, fiat_shamir_challenge, scalar_mul_base


def h_p(scope: bytes) -> Point:
    return Point.hash_to_point(scope, domain=b"janus-fl-clsag-tag")


@dataclass
class RingSignature:
    ring: list[Point]          # public keys, fixed public order
    challenges: list[Scalar]   # c_0 .. c_{n-1}
    responses: list[Scalar]    # r_0 .. r_{n-1}
    key_image_round: Point     # I_k^t
    key_image_epoch: Point     # P_k^e


def sign(sk: Scalar, pk_index: int, ring: list[Point], message: bytes,
         rho_t: bytes, rho_e: bytes) -> RingSignature:
    n = len(ring)
    base_t, base_e = h_p(rho_t), h_p(rho_e)
    I_t = base_t.scalar_mul(sk)
    I_e = base_e.scalar_mul(sk)
    transcript = message + I_t.b + I_e.b

    challenges: list[Scalar | None] = [None] * n
    responses: list[Scalar | None] = [None] * n
    for j in range(n):
        if j != pk_index:
            responses[j] = Scalar.random()

    alpha = Scalar.random()
    idx = (pk_index + 1) % n
    challenges[idx] = fiat_shamir_challenge(
        transcript, scalar_mul_base(alpha).b, base_t.scalar_mul(alpha).b, base_e.scalar_mul(alpha).b
    )

    j = idx
    while j != pk_index:
        L_j = scalar_mul_base(responses[j]) + ring[j].scalar_mul(challenges[j])
        Rt_j = base_t.scalar_mul(responses[j]) + I_t.scalar_mul(challenges[j])
        Re_j = base_e.scalar_mul(responses[j]) + I_e.scalar_mul(challenges[j])
        c_next = fiat_shamir_challenge(transcript, L_j.b, Rt_j.b, Re_j.b)
        j = (j + 1) % n
        challenges[j] = c_next

    # challenges[pk_index] was set on the last loop iteration (when the
    # previous j wrapped forward into pk_index). Close the ring:
    responses[pk_index] = alpha - challenges[pk_index] * sk

    return RingSignature(ring, challenges, responses, I_t, I_e)


def verify_with_scopes(sig: RingSignature, message: bytes, rho_t: bytes, rho_e: bytes) -> bool:
    n = len(sig.ring)
    if n == 0 or len(sig.challenges) != n or len(sig.responses) != n:
        return False
    base_t, base_e = h_p(rho_t), h_p(rho_e)
    transcript = message + sig.key_image_round.b + sig.key_image_epoch.b

    for j in range(n):
        L_j = scalar_mul_base(sig.responses[j]) + sig.ring[j].scalar_mul(sig.challenges[j])
        Rt_j = base_t.scalar_mul(sig.responses[j]) + sig.key_image_round.scalar_mul(sig.challenges[j])
        Re_j = base_e.scalar_mul(sig.responses[j]) + sig.key_image_epoch.scalar_mul(sig.challenges[j])
        c_next = fiat_shamir_challenge(transcript, L_j.b, Rt_j.b, Re_j.b)
        expected = sig.challenges[(j + 1) % n]
        if c_next.b != expected.b:
            return False
    return True


def same_round_same_signer(tag_a: Point, tag_b: Point) -> bool:
    """V2 (ARCHITECTURE.md §4): two submissions with the same I_k^t in
    the same round came from the same signer -- the double-submission
    detector.
    """
    return tag_a == tag_b
