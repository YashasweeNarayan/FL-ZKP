"""R_cons: the compound consistency relation (ARCHITECTURE.md §5.4).
EC-DEPENDENT.

Purpose (the "identity-laundering" fix): without R_cons, nothing stops
a malicious client from using key A to produce its ring signature /
round tag / epoch pseudonym, but key B to decide (via the VRF) whether
it was sampled this round, or key C's escrow ciphertext for the
identity-tracing committee. That decouples "who signed" from "who can
be traced" from "who was legitimately sampled" -- exactly the gap a
server-side or coalition attacker would exploit to frame an innocent
key image, or to have a sampled contribution attributed to an escrow
that doesn't open to the real signer. R_cons closes it by requiring
ONE Sigma-protocol AND-proof that the SAME secret key sk underlies
every one of these artifacts simultaneously:

    pk        = sk * G                     (long-term identity key)
    I_t       = sk * H_p(rho_t)             (round key image, clsag.py)
    P_e       = sk * H_p(rho_e)             (epoch pseudonym, clsag.py)
    gamma     = sk * H_p(alpha)             (VRF point, vrf.py)

all four reuse crypto.schnorr_nizk.and_prove/and_verify directly (same
"same secret, many bases" shape already used by elgamal_threshold.py's
partial-decryption proof). The identity-escrow binding is checked at
TRACE time (see bind_escrow_to_identity below), not inside the AND
proof itself, because the committee never sees sk or pk in the clear
during normal operation -- only at a successful trace opening,
consistent with ARCHITECTURE.md's anonymity-accountability budget.

The VRF-sampling disjunction ("b=0 => q=nu", i.e. an unsampled client
still submits a cover-traffic commitment to pure DP noise rather than a
real update, so the server cannot distinguish sampled from unsampled
submissions by submission pattern alone) is exposed here as a generic
two-branch OR statement (`prove_sampling_disjunction` /
`verify_sampling_disjunction`) over whatever two (point, base) pairs
the caller supplies. This is intentionally left generic rather than
hard-coded to a specific noise-commitment encoding: the exact
noise-cover commitment construction is finalized alongside the DP
mechanism in fl/ (Task #10, not yet built), and committing to a
specific encoding here before that code exists would risk silently
mismatching it. What IS fully specified and tested here is the
identity-binding AND-proof, which is the actual fix for the
identity-laundering gap -- see tests/test_r_cons.py (Colab-side).
"""
from __future__ import annotations

from dataclasses import dataclass

from .ristretto import Point, Scalar, scalar_mul_base
from .schnorr_nizk import (
    AndProof, OrProof, and_prove, and_verify,
    or_prove_branch0_true, or_prove_branch1_true, or_verify,
)
from .clsag import h_p as ring_h_p
from .vrf import h_p as vrf_h_p


@dataclass
class RConsIdentityProof:
    and_proof: AndProof
    pk: Point
    key_image_round: Point
    key_image_epoch: Point
    vrf_gamma: Point


def prove_identity_binding(sk: Scalar, rho_t: bytes, rho_e: bytes, vrf_alpha: bytes, transcript: bytes) -> RConsIdentityProof:
    """The core R_cons proof: one AND-composition binding sk across
    the identity key, both clsag.py scope tags, and the VRF point.
    """
    base_g = Point.base()
    base_t = ring_h_p(rho_t)
    base_e = ring_h_p(rho_e)
    base_vrf = vrf_h_p(vrf_alpha)

    bases = [base_g, base_t, base_e, base_vrf]
    proof = and_prove(sk, bases, transcript)

    pk = scalar_mul_base(sk)
    I_t = base_t.scalar_mul(sk)
    P_e = base_e.scalar_mul(sk)
    gamma = base_vrf.scalar_mul(sk)
    return RConsIdentityProof(proof, pk, I_t, P_e, gamma)


def verify_identity_binding(proof: RConsIdentityProof, rho_t: bytes, rho_e: bytes, vrf_alpha: bytes, transcript: bytes) -> bool:
    base_g = Point.base()
    base_t = ring_h_p(rho_t)
    base_e = ring_h_p(rho_e)
    base_vrf = vrf_h_p(vrf_alpha)

    bases = [base_g, base_t, base_e, base_vrf]
    points = [proof.pk, proof.key_image_round, proof.key_image_epoch, proof.vrf_gamma]
    return and_verify(proof.and_proof, points, bases, transcript)


def bind_escrow_to_identity(opened_plaintext_point: Point, proof: RConsIdentityProof) -> bool:
    """Trace-time check (ARCHITECTURE.md §6.2's opening protocol): once
    the committee cooperatively decrypts a client's escrow ciphertext
    (via elgamal_threshold.combine_partials) to recover the plaintext
    identity point, it must equal THIS proof's pk -- otherwise the
    escrow was for a different key than the one that actually produced
    the signed artifacts, and the opening is invalid evidence.
    """
    return opened_plaintext_point == proof.pk


# ---------------------------------------------------------------------------
# Generic two-branch OR statement for the sampling disjunction. See the
# module docstring: the exact (point, base) encoding of "real update"
# vs "noise cover" is supplied by the fl/ layer once it exists.
# ---------------------------------------------------------------------------

def prove_sampling_disjunction(which_branch_true: int, x: Scalar, P0: Point, base0: Point, P1: Point, base1: Point, transcript: bytes) -> OrProof:
    if which_branch_true == 0:
        return or_prove_branch0_true(x, base0, P1, base1, transcript)
    elif which_branch_true == 1:
        return or_prove_branch1_true(x, base1, P0, base0, transcript)
    raise ValueError("which_branch_true must be 0 or 1")


def verify_sampling_disjunction(proof: OrProof, P0: Point, base0: Point, P1: Point, base1: Point, transcript: bytes) -> bool:
    return or_verify(proof, P0, base0, P1, base1, transcript)
