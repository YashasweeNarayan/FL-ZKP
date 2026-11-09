"""Compound Sigma-protocol NIZKs — the MVP's proof system.

EC-DEPENDENT — see ristretto.py's header for the Colab-validation note.

*** SCOPE DECISION, made 2026-11-09 under the 3-day deadline ***
ARCHITECTURE.md §3.1/§8 specifies Bulletproofs (range proofs) and PLONK
(R_cons) as the camera-ready proof system. Both need toolchains this
build cannot risk on a 3-day clock: Bulletproofs needs a Rust+maturin
build (flagged in REVIEW.md as the single biggest schedule risk, E2/R17);
PLONK needs circom+snarkjs (Node toolchain, circuit compilation, a
universal-SRS download). Neither is needed for correctness — only for
proof *size and speed* at scale.

What's here instead: every relation in R_cons and R_norm is a
conjunction (AND) or disjunction (OR) of discrete-log statements over
ristretto255 — exactly the class Sigma-protocols handle natively,
without a general-purpose circuit. These are real, textbook
zero-knowledge proofs (Schnorr '91 / Cramer-Damgard-Schoenmakers '94
AND/OR composition, Fiat-Shamir NIZK transform) — not a toy stand-in.
The honest tradeoff: proofs are O(witnesses) field elements, not O(log)
like Bulletproofs/PLONK, so this doesn't carry the camera-ready paper's
efficiency numbers. circuits/*.circom is kept as the documented upgrade
path for that pass. See STATUS.md.

This module provides the general AND/OR Sigma-protocol toolkit;
r_cons.py and rangeproof.py build the two specific JANUS-FL relations
on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ristretto import Point, Scalar, fiat_shamir_challenge, scalar_mul_base


# ---------------------------------------------------------------------------
# Core building block: Schnorr proof of knowledge of x s.t. P = x*Base
# ---------------------------------------------------------------------------

@dataclass
class SchnorrProof:
    commitment: Point   # k*Base
    response: Scalar    # k + c*x


def schnorr_prove(x: Scalar, base: Point, transcript: bytes, k: Scalar | None = None) -> SchnorrProof:
    k = k if k is not None else Scalar.random()
    R = base.scalar_mul(k)
    c = fiat_shamir_challenge(transcript, base.b, R.b)
    s = k + c * x
    return SchnorrProof(R, s)


def schnorr_verify(proof: SchnorrProof, P: Point, base: Point, transcript: bytes) -> bool:
    c = fiat_shamir_challenge(transcript, base.b, proof.commitment.b)
    lhs = base.scalar_mul(proof.response)
    rhs = proof.commitment + P.scalar_mul(c)
    return lhs == rhs


# ---------------------------------------------------------------------------
# AND-composition: prove the SAME secret x satisfies P_i = x*Base_i for
# every (P_i, Base_i) pair, using ONE shared challenge (this is exactly
# what makes R_cons's "repeated sk across five lines" (ARCHITECTURE.md
# §5.4) sound: one witness, one challenge, many public checks).
# ---------------------------------------------------------------------------

@dataclass
class AndProof:
    commitments: list[Point]   # R_i = k*Base_i, same k for every i
    response: Scalar           # k + c*x


def and_prove(x: Scalar, bases: list[Point], transcript: bytes, k: Scalar | None = None) -> AndProof:
    k = k if k is not None else Scalar.random()
    Rs = [base.scalar_mul(k) for base in bases]
    c = fiat_shamir_challenge(transcript, *[b.b for b in bases], *[r.b for r in Rs])
    s = k + c * x
    return AndProof(Rs, s)


def and_verify(proof: AndProof, points: list[Point], bases: list[Point], transcript: bytes) -> bool:
    if not (len(proof.commitments) == len(points) == len(bases)):
        return False
    c = fiat_shamir_challenge(transcript, *[b.b for b in bases], *[r.b for r in proof.commitments])
    for R, P, base in zip(proof.commitments, points, bases):
        if base.scalar_mul(proof.response) != R + P.scalar_mul(c):
            return False
    return True


# ---------------------------------------------------------------------------
# OR-composition (CDS94): prove P0 = x*Base0 OR P1 = x1*Base1, without
# revealing which. Used for R_cons's sampling disjunction
# (b=0 => q=nu) in rangeproof.py / r_cons.py: "I know an opening of c_k
# as a real update" OR "I know an opening of c_k as pure DP noise",
# without revealing which branch is true (that's exactly what hides
# the VRF-sampling bit from the server per §5.5).
# ---------------------------------------------------------------------------

@dataclass
class OrProof:
    R0: Point
    R1: Point
    s0: Scalar
    s1: Scalar
    c0: Scalar   # the simulated challenge for the FALSE branch is revealed;
                 # the real branch's challenge is derived as c_total - c_other


def or_prove_branch0_true(x0: Scalar, base0: Point, P1: Point, base1: Point, transcript: bytes) -> OrProof:
    """Caller knows x0 with P0 = x0*base0 (branch 0 true); branch 1 is
    simulated without knowing its witness.
    """
    # Simulate branch 1: pick random response s1 and random fake challenge c1,
    # solve for R1 = s1*base1 - c1*P1.
    s1 = Scalar.random()
    c1 = Scalar.random()
    R1 = base1.scalar_mul(s1) - P1.scalar_mul(c1)

    # Real branch 0: commit with fresh randomness k0.
    k0 = Scalar.random()
    R0 = base0.scalar_mul(k0)

    c_total = fiat_shamir_challenge(transcript, base0.b, base1.b, R0.b, R1.b)
    c0 = c_total - c1
    s0 = k0 + c0 * x0
    return OrProof(R0, R1, s0, s1, c0)


def or_verify(proof: OrProof, P0: Point, base0: Point, P1: Point, base1: Point, transcript: bytes) -> bool:
    c_total = fiat_shamir_challenge(transcript, base0.b, base1.b, proof.R0.b, proof.R1.b)
    c1 = c_total - proof.c0
    ok0 = base0.scalar_mul(proof.s0) == proof.R0 + P0.scalar_mul(proof.c0)
    ok1 = base1.scalar_mul(proof.s1) == proof.R1 + P1.scalar_mul(c1)
    return ok0 and ok1


def or_prove_branch1_true(x1: Scalar, base1: Point, P0: Point, base0: Point, transcript: bytes) -> OrProof:
    """Mirror of or_prove_branch0_true for when branch 1 is the true one.
    NOT implemented by swapping arguments into or_prove_branch0_true and
    relabeling the result -- the Fiat-Shamir challenge is computed over
    (base0.b, base1.b, R0.b, R1.b) in that FIXED order inside
    fiat_shamir_challenge, so a swapped call hashes a different byte
    string and produces a proof or_verify (which always hashes in
    canonical branch-0-then-branch-1 order) would reject. This function
    simulates branch 0 and proves branch 1 directly in canonical order
    instead, so or_verify works unmodified regardless of which branch
    was actually true -- used by rangeproof.py's bit proofs, where the
    prover needs to hide which branch (bit=0 vs bit=1) is real.
    """
    s0 = Scalar.random()
    c0 = Scalar.random()
    R0 = base0.scalar_mul(s0) - P0.scalar_mul(c0)

    k1 = Scalar.random()
    R1 = base1.scalar_mul(k1)

    c_total = fiat_shamir_challenge(transcript, base0.b, base1.b, R0.b, R1.b)
    c1 = c_total - c0
    s1 = k1 + c1 * x1
    return OrProof(R0, R1, s0, s1, c0)
