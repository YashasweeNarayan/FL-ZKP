"""Verifiable proactive resharing (ARCHITECTURE.md §6.4) — EC wrapper
around crypto/shamir.py's already-tested reshare() field math.

EC-DEPENDENT — see ristretto.py's header for the Colab-validation note.
The algebraic invariant (new shares reconstruct the SAME secret) is
already verified in tests/test_shamir.py, which ran in this sandbox.
What's added here is Feldman verifiability for the sub-shares exchanged
during resharing, so a cheating outgoing member is caught exactly like
in the initial DKG (dkg.py).
"""
from __future__ import annotations

from .field import L, ShamirPolynomial, ShamirShare, lagrange_coeffs
from .ristretto import Point, Scalar, scalar_mul_base
from .dkg import verify_sub_share


def reshare_verifiable(
    old_shares: dict[int, ShamirShare],
    t: int,
    new_member_ids: list[int],
) -> tuple[dict[int, ShamirShare], dict[int, list[Point]]]:
    """Same transformation as crypto.shamir.reshare, but each cooperating
    outgoing member also publishes Feldman commitments for its
    zero-constant-term sub-polynomial, so every incoming member can run
    verify_sub_share() before accepting its new share. Returns
    (new_shares, {outgoing_member_id: feldman_commitments}).
    """
    cooperating = list(old_shares.values())[:t]
    xs = [s.index for s in cooperating]
    lambdas = lagrange_coeffs(xs, L)

    sub_shares_per_new_member: dict[int, int] = {j: 0 for j in new_member_ids}
    commitments_by_member: dict[int, list[Point]] = {}

    for lam, old_share in zip(lambdas, cooperating):
        weighted = (lam * old_share.value) % L
        sub_poly = ShamirPolynomial(t, secret=weighted)
        commitments_by_member[old_share.index] = [
            scalar_mul_base(Scalar.from_int(c)) for c in sub_poly.coeffs
        ]
        for j in new_member_ids:
            sub_shares_per_new_member[j] = (
                sub_shares_per_new_member[j] + sub_poly.share_for(j).value
            ) % L

    new_shares = {j: ShamirShare(j, v) for j, v in sub_shares_per_new_member.items()}
    return new_shares, commitments_by_member


def verify_resharing_sub_contribution(new_member_id: int, sub_share_value: int,
                                       commitments: list[Point]) -> bool:
    return verify_sub_share(ShamirShare(new_member_id, sub_share_value), commitments)
