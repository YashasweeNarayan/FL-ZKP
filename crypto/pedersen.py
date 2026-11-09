"""Pedersen vector commitments (ARCHITECTURE.md §3, P2).

EC-DEPENDENT — see ristretto.py's header for the Colab-validation note.

Com(v; r) = v*G + r*H, with H a second generator derived by hashing G
to a point so nobody (including us) knows log_G(H) — the standard
"nothing-up-my-sleeve" construction that makes the commitment
*binding* under the discrete-log assumption and *hiding* given r is
random. Homomorphic: Com(a;r) + Com(b;s) = Com(a+b; r+s), which is
exactly what check V5 (§4) and the double-masking sum in secagg.py
rely on.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ristretto import Point, Scalar, scalar_mul_base


H = Point.hash_to_point(b"pedersen-second-generator", domain=b"janus-fl-pedersen")


@dataclass(frozen=True)
class Commitment:
    point: Point

    def __add__(self, other: "Commitment") -> "Commitment":
        return Commitment(self.point + other.point)

    def __eq__(self, other) -> bool:
        return isinstance(other, Commitment) and self.point == other.point


def commit(value: Scalar, blinding: Scalar) -> Commitment:
    return Commitment(scalar_mul_base(value) + H.scalar_mul(blinding))


def commit_sum(commitments: list[Commitment]) -> Commitment:
    """Pi c_k — product-of-commitments == commitment-of-sum, by homomorphism.
    This is the left-hand side the server can always recompute from BB
    without touching any client's value or blinding factor.
    """
    acc = commitments[0]
    for c in commitments[1:]:
        acc = acc + c
    return acc


def verify_opening(c: Commitment, value: Scalar, blinding: Scalar) -> bool:
    """Checks c == Com(value; blinding). Used by:
    - check V5 (verifier/audit.py): the server reveals (sum_value,
      sum_blinding) after acceptance; anyone recomputes Com and compares
      to Pi c_k from BB.
    - tests only, to sanity-check the homomorphism before Colab.
    """
    return c == commit(value, blinding)
