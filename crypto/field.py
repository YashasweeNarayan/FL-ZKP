"""Prime-field arithmetic for Shamir sharing.

PURE-MATH MODULE — verified in-sandbox, see tests/test_field.py.

This is scalar arithmetic mod the ristretto255 group order `L`. It is
deliberately separate from ristretto.py: Shamir share *construction and
reconstruction* only ever needs field arithmetic over Z_L, never group
(point) operations. Keeping it separate means the polynomial math is
testable without pynacl, which is exactly what this sandbox needed.

L is the standard ristretto255 / Curve25519 scalar field order, so shares
produced here are valid exponents for ristretto group elements once
pedersen.py / elgamal_threshold.py (Colab-side) lift them into the group.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

# ristretto255 scalar field order (same as Ed25519's L).
L = 2**252 + 27742317777372353535851937790883648493


def rand_scalar() -> int:
    """Uniform random element of Z_L, via rejection sampling."""
    nbytes = (L.bit_length() + 7) // 8 + 1
    while True:
        x = int.from_bytes(secrets.token_bytes(nbytes), "little") % (2 ** (8 * nbytes))
        if x < L:
            return x


def modinv(a: int, m: int = L) -> int:
    return pow(a % m, -1, m)


def poly_eval(coeffs: list[int], x: int, m: int = L) -> int:
    """Evaluate coeffs[0] + coeffs[1]*x + ... + coeffs[k-1]*x^(k-1) mod m."""
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % m
    return acc


def lagrange_coeffs(xs: list[int], m: int = L) -> list[int]:
    """Lagrange coefficients lambda_i such that sum(lambda_i * f(x_i)) = f(0).

    Standard Shamir-at-zero reconstruction. xs must be distinct mod m.
    """
    coeffs = []
    for i, xi in enumerate(xs):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i == j:
                continue
            num = (num * (-xj)) % m
            den = (den * (xi - xj)) % m
        coeffs.append((num * modinv(den, m)) % m)
    return coeffs


@dataclass
class ShamirShare:
    index: int   # x-coordinate, 1-indexed (party id), never 0
    value: int   # f(index) mod L


class ShamirPolynomial:
    """A degree-(t-1) polynomial over Z_L with a chosen constant term.

    secret=None picks a random constant term (ordinary sharing).
    secret=0 is used by proactive resharing (§6.4): the sub-shares sum
    to the original secret's shares without ever re-exposing the secret.
    """

    def __init__(self, t: int, secret: int | None = None, m: int = L):
        self.t = t
        self.m = m
        c0 = secret if secret is not None else rand_scalar()
        self.coeffs = [c0 % m] + [rand_scalar() for _ in range(t - 1)]

    def share_for(self, index: int) -> ShamirShare:
        if index == 0:
            raise ValueError("index 0 is reserved for the secret itself")
        return ShamirShare(index, poly_eval(self.coeffs, index, self.m))

    def shares_for(self, indices: list[int]) -> list[ShamirShare]:
        return [self.share_for(i) for i in indices]

    @property
    def secret(self) -> int:
        return self.coeffs[0]


def reconstruct(shares: list[ShamirShare], m: int = L) -> int:
    """Full reconstruction — f(0). Used ONLY in tests on a throwaway secret.

    Per ARCHITECTURE.md §1.2, sk_trace is NEVER reconstructed in the
    actual protocol: production tracing uses threshold decryption
    (elgamal_threshold.py, Colab-side), which combines partial
    decryptions without this function ever touching sk_trace.
    """
    xs = [s.index for s in shares]
    lambdas = lagrange_coeffs(xs, m)
    return sum(l * s.value for l, s in zip(lambdas, shares)) % m
