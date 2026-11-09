"""Threshold ElGamal identity escrow (ARCHITECTURE.md §1.2, §3 P5/P6).

EC-DEPENDENT — see ristretto.py's header for the Colab-validation note.

Encrypts a client's long-term public key pk_k under the committee's
shared PK_trace. Opening NEVER reconstructs sk_trace (that would burn
the escrow forever, per §1.2) — instead, t=7 of the 10 key-holders each
compute a partial decryption of ONE targeted ciphertext, with a
Chaum-Pedersen proof that the partial is correct, and the partials
combine via Lagrange interpolation to reveal exactly one plaintext.
"""
from __future__ import annotations

from dataclasses import dataclass

from .field import L, ShamirShare, lagrange_coeffs
from .ristretto import Point, Scalar, fiat_shamir_challenge, scalar_mul_base
from .schnorr_nizk import and_prove, and_verify


@dataclass(frozen=True)
class ElGamalCiphertext:
    """tau_k = (ephemeral*G, plaintext_point + ephemeral*PK_trace)"""
    c1: Point
    c2: Point


def encrypt(plaintext_point: Point, pk_trace: Point, ephemeral: Scalar | None = None) -> ElGamalCiphertext:
    e = ephemeral if ephemeral is not None else Scalar.random()
    return ElGamalCiphertext(c1=scalar_mul_base(e), c2=plaintext_point + pk_trace.scalar_mul(e))


@dataclass(frozen=True)
class PartialDecryption:
    holder_id: int
    d_i: Point          # share_i * c1
    proof: object        # Chaum-Pedersen proof that dlog(d_i)/dlog(c1) == dlog(pk_share_i)/dlog(G)


def compute_partial(ct: ElGamalCiphertext, share: ShamirShare, pk_share_i: Point, transcript: bytes):
    """Key-holder i's contribution to opening ct. The Chaum-Pedersen proof
    binds d_i to the SAME share value that was committed to in the DKG
    (dkg.py's Feldman commitments) — that binding is what makes a
    corrupted/garbage partial detectable, i.e. what makes opening
    non-frameable (T8 in ARCHITECTURE.md §8).
    """
    s_i = Scalar.from_int(share.value)
    d_i = ct.c1.scalar_mul(s_i)
    # Chaum-Pedersen: prove log_G(pk_share_i) == log_{c1}(d_i), both equal s_i.
    proof = and_prove(s_i, bases=[Point.base(), ct.c1], transcript=transcript)
    return PartialDecryption(share.index, d_i, proof)


def verify_partial(pd: PartialDecryption, ct: ElGamalCiphertext, pk_share_i: Point, transcript: bytes) -> bool:
    return and_verify(pd.proof, points=[pk_share_i, pd.d_i], bases=[Point.base(), ct.c1], transcript=transcript)


def combine_partials(ct: ElGamalCiphertext, partials: list[PartialDecryption]) -> Point:
    """Lagrange-combine t verified partials to recover the plaintext
    point. sk_trace is never assembled -- only this combination, which
    only works for the ONE ciphertext `ct` the partials were computed
    against, is ever produced.
    """
    xs = [p.holder_id for p in partials]
    lambdas = lagrange_coeffs(xs, L)
    acc = None
    for lam, p in zip(lambdas, partials):
        term = p.d_i.scalar_mul(Scalar.from_int(lam))
        acc = term if acc is None else acc + term
    return ct.c2 - acc
