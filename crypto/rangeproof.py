"""Sigma-protocol bit-decomposition range proof -- the MVP's R_norm
realization (ARCHITECTURE.md §5.2, §9's sketched-norm-proof lemma uses
THIS as its per-coordinate-group primitive). EC-DEPENDENT.

*** Part of the same 2026-11-09 scope decision as schnorr_nizk.py ***
True Bulletproofs give an O(log n) range proof over n bits. This
bit-by-bit Sigma-protocol version is O(n) -- real zero-knowledge, same
security definition, just bigger proofs -- which is exactly why
sketch.py's JL projection (making the proven quantity a small fixed-size
sketch, not the full d=23.5M-dim gradient) is load-bearing: it keeps n
small enough that O(n) range proofs are actually practical for this
MVP. circuits/*.circom + real Bulletproofs stay the camera-ready upgrade
(see STATUS.md).

Construction: to prove a committed value v is in [0, 2^n_bits) without
revealing v, decompose v = sum(b_i * 2^i), commit to each bit
separately as C_i = b_i*G + r_i*H, prove each C_i opens to 0 OR 1 (via
schnorr_nizk's OR-composition), and prove the bit commitments correctly
recompose into the original commitment C_v (an algebraic linear check
on the blinding factors, not a separate proof -- see
verify_recomposition).
"""
from __future__ import annotations

from dataclasses import dataclass

from .pedersen import H as PEDERSEN_H, Commitment, commit
from .ristretto import Point, Scalar, scalar_mul_base
from .schnorr_nizk import OrProof, or_prove_branch0_true, or_prove_branch1_true, or_verify

G = Point.base()


@dataclass
class BitProof:
    bit_commitment: Point   # C_i = b_i*G + r_i*H
    or_proof: OrProof


@dataclass
class RangeProof:
    bit_proofs: list[BitProof]
    n_bits: int


def _bit_or_statement(bit_commitment: Point):
    """branch0: bit=0 => bit_commitment = r*H.  branch1: bit=1 =>
    (bit_commitment - G) = r*H. Both branches share base H -- only the
    public point differs -- which is exactly the shape or_verify expects.
    """
    P0, base0 = bit_commitment, PEDERSEN_H
    P1, base1 = bit_commitment - G, PEDERSEN_H
    return P0, base0, P1, base1


def prove_bit(bit: int, blinding: Scalar, transcript: bytes) -> BitProof:
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    bit_commitment = commit(Scalar.from_int(bit), blinding).point
    P0, base0, P1, base1 = _bit_or_statement(bit_commitment)
    if bit == 0:
        proof = or_prove_branch0_true(blinding, base0, P1, base1, transcript)
    else:
        proof = or_prove_branch1_true(blinding, base1, P0, base0, transcript)
    return BitProof(bit_commitment, proof)


def verify_bit(proof: BitProof, transcript: bytes) -> bool:
    P0, base0, P1, base1 = _bit_or_statement(proof.bit_commitment)
    return or_verify(proof.or_proof, P0, base0, P1, base1, transcript)


def prove_range(value: int, n_bits: int) -> tuple[RangeProof, Scalar]:
    """Decompose `value` into n_bits bits, prove each is 0/1, and
    return (proof, aggregate_blinding) where aggregate_blinding is the
    blinding factor the AGGREGATE commitment sum(2^i * C_i) uses --
    callers combine this with their own commitment to `value` (see
    verify_recomposition) to tie the range proof to a specific, already
    public Pedersen commitment to `value`.
    """
    if value < 0 or value >= (1 << n_bits):
        raise ValueError(f"value {value} out of range [0, 2^{n_bits})")
    bit_proofs = []
    blindings = []
    for i in range(n_bits):
        bit = (value >> i) & 1
        r_i = Scalar.random()
        blindings.append(r_i)
        transcript = f"rangeproof-bit-{i}".encode()
        bit_proofs.append(prove_bit(bit, r_i, transcript))
    # aggregate blinding factor for sum(2^i * C_i) = value*G + (sum 2^i*r_i)*H
    agg_blinding_int = 0
    for i, r_i in enumerate(blindings):
        agg_blinding_int = (agg_blinding_int + (1 << i) * int.from_bytes(r_i.b, "little"))
    from .field import L
    agg_blinding = Scalar.from_int(agg_blinding_int % L)
    return RangeProof(bit_proofs, n_bits), agg_blinding


def verify_range(proof: RangeProof) -> bool:
    for i, bp in enumerate(proof.bit_proofs):
        transcript = f"rangeproof-bit-{i}".encode()
        if not verify_bit(bp, transcript):
            return False
    return True


def recompose_commitment(proof: RangeProof) -> Point:
    """sum(2^i * C_i) -- should equal value*G + agg_blinding*H for the
    SAME (value, agg_blinding) prove_range returned. This is the public
    recomposition check tying the per-bit proofs to one aggregate value.
    """
    acc = None
    for i, bp in enumerate(proof.bit_proofs):
        term = bp.bit_commitment.scalar_mul(Scalar.from_int(1 << i))
        acc = term if acc is None else acc + term
    return acc


def verify_recomposition(proof: RangeProof, claimed_commitment: Commitment) -> bool:
    """Checks that the external commitment a caller holds for `value`
    (e.g. one already published as part of R_cons / the update
    commitment c_k) is exactly the recomposition of this range proof's
    bit commitments -- this is what binds "this value is in-range" to
    "this value is the SAME value committed to elsewhere", per
    ARCHITECTURE.md §5.2.
    """
    return recompose_commitment(proof) == claimed_commitment.point
