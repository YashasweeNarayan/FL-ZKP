"""Pedersen DKG with Feldman verification (ARCHITECTURE.md §3 P7).

EC-DEPENDENT — see ristretto.py's header for the Colab-validation note.

No trusted dealer: PK_trace = g^sk_trace emerges as the sum of n_c
independently-chosen sub-secrets' public commitments, and sk_trace
itself is never assembled by anyone (same non-reconstruction principle
as elgamal_threshold.py). Each key-holder i:
  1. picks a random sub-secret f_i(0), Shamir-shares it (degree t-1)
     across all n_c holders,
  2. publishes Feldman commitments to its own polynomial's coefficients
     (so every other holder can verify the sub-share it receives),
  3. the holder's own final share is the sum of sub-shares received
     from everyone (including itself).
PK_trace = sum_i (f_i(0) * G) = sum of each holder's published
commitment to their own constant term.
"""
from __future__ import annotations

from dataclasses import dataclass

from .field import L, ShamirPolynomial, ShamirShare
from .ristretto import Point, Scalar, scalar_mul_base


@dataclass
class DkgContribution:
    holder_id: int
    feldman_commitments: list[Point]     # C_j = coeff_j * G, j=0..t-1
    sub_shares: dict[int, ShamirShare]   # sub_shares[j] = this holder's sub-poly evaluated at j


def generate_contribution(holder_id: int, t: int, member_ids: list[int]) -> DkgContribution:
    poly = ShamirPolynomial(t)  # random constant term = this holder's sub-secret
    commitments = [scalar_mul_base(Scalar.from_int(c)) for c in poly.coeffs]
    sub_shares = {j: poly.share_for(j) for j in member_ids}
    return DkgContribution(holder_id, commitments, sub_shares)


def verify_sub_share(share: ShamirShare, commitments: list[Point]) -> bool:
    """Feldman check: g^{share.value} == product_j (C_j)^{x^j}, where x
    is the recipient's index. If this fails for a sub-share from holder
    i, holder i is provably cheating and gets excluded before any
    aggregate share is finalized -- this is R_share (§5.3)'s "honest
    sharing" half, realized directly as a public verification rather
    than a separate ZK proof, which is possible because Feldman
    commitments are already public group elements.
    """
    x = share.index
    lhs = scalar_mul_base(Scalar.from_int(share.value))
    rhs = commitments[0]
    x_pow = 1
    for j in range(1, len(commitments)):
        x_pow = (x_pow * x) % L
        rhs = rhs + commitments[j].scalar_mul(Scalar.from_int(x_pow))
    return lhs == rhs


def finalize(contributions: list[DkgContribution], member_ids: list[int]) -> tuple[Point, dict[int, ShamirShare]]:
    """Combine verified contributions into (PK_trace, {member_id: final_share}).
    Caller must have already run verify_sub_share on every (contribution,
    sub_share) pair and discarded any contribution that failed.

    Returns ShamirShare objects (not plain ints) so that a post-DKG
    final share and a post-reshare share (see reshare.py's
    reshare_verifiable, which already returns dict[int, ShamirShare])
    are the SAME type and can both be handed to
    elgamal_threshold.py's compute_partial without the caller having
    to remember which pipeline produced which share.
    """
    pk_trace = contributions[0].feldman_commitments[0]
    for contrib in contributions[1:]:
        pk_trace = pk_trace + contrib.feldman_commitments[0]

    raw_values = {j: 0 for j in member_ids}
    for contrib in contributions:
        for j in member_ids:
            raw_values[j] = (raw_values[j] + contrib.sub_shares[j].value) % L
    final_shares = {j: ShamirShare(index=j, value=raw_values[j]) for j in member_ids}
    return pk_trace, final_shares


def pk_share_for(holder_id: int, contributions: list[DkgContribution]) -> Point:
    """The public commitment to holder `holder_id`'s own FINAL share
    (sum of sub-shares it received) — used by elgamal_threshold.py's
    verify_partial to check a partial decryption without learning the
    share itself: pk_share_i = sum_i'( C_{i'}[0] ... ) evaluated the same
    way as the Feldman check, aggregated across all contributions at x=holder_id.
    """
    acc = None
    for contrib in contributions:
        x = holder_id
        term = contrib.feldman_commitments[0]
        x_pow = 1
        for j in range(1, len(contrib.feldman_commitments)):
            x_pow = (x_pow * x) % L
            term = term + contrib.feldman_commitments[j].scalar_mul(Scalar.from_int(x_pow))
        acc = term if acc is None else acc + term
    return acc
