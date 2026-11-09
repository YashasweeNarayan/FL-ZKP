"""EC-based VRF for anonymous self-sampling (ARCHITECTURE.md §5.5, P12).

EC-DEPENDENT — see ristretto.py's header for the Colab-validation note.

A simplified ECVRF (same family as RFC 9381, specialized to
ristretto255): the client computes
    gamma = sk * H_p(alpha)          (alpha = round nonce rho_t)
    beta  = H(gamma)                  -- the VRF output, used for the
                                          Bernoulli(p) sampling decision
    pi    = Chaum-Pedersen proof that gamma = sk * H_p(alpha), i.e. that
            log_G(pk) == log_{H_p(alpha)}(gamma)
Why this matters: the server cannot choose who's sampled (alpha is
public, sk is the client's own), and cannot even verify beta without
pi -- but ANYONE holding pi and pk can verify beta was computed
honestly from pk and alpha, which is what the b=0 => q=nu disjunction
in r_cons.py relies on to stop a sampled client from claiming to be a
dummy, and vice versa.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .ristretto import Point, Scalar, scalar_mul_base
from .schnorr_nizk import AndProof, and_prove, and_verify


@dataclass
class VrfOutput:
    gamma: Point
    beta: bytes          # H(gamma), the pseudorandom output
    proof: AndProof       # proves gamma = sk * H_p(alpha) consistent with pk = sk*G


def h_p(alpha: bytes) -> Point:
    return Point.hash_to_point(alpha, domain=b"janus-fl-vrf")


def evaluate(sk: Scalar, pk: Point, alpha: bytes) -> VrfOutput:
    base_alpha = h_p(alpha)
    gamma = base_alpha.scalar_mul(sk)
    beta = hashlib.sha256(gamma.b).digest()
    transcript = b"vrf:" + alpha
    proof = and_prove(sk, bases=[Point.base(), base_alpha], transcript=transcript)
    return VrfOutput(gamma, beta, proof)


def verify(output: VrfOutput, pk: Point, alpha: bytes) -> bool:
    base_alpha = h_p(alpha)
    transcript = b"vrf:" + alpha
    if not and_verify(output.proof, points=[pk, output.gamma], bases=[Point.base(), base_alpha],
                       transcript=transcript):
        return False
    return hashlib.sha256(output.gamma.b).digest() == output.beta


def is_sampled(beta: bytes, p: float, bit_length: int = 256) -> bool:
    """Deterministic Bernoulli(p) from the VRF output -- same beta always
    gives the same sampling decision for a given round, and nobody can
    predict it without the secret key (R_vrf, ARCHITECTURE.md §5.5).
    """
    threshold = int(p * (2 ** bit_length))
    value = int.from_bytes(beta[: bit_length // 8], "big")
    return value < threshold
