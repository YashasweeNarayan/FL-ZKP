"""Shamir VSS, committee roster math, and proactive resharing — the field-math half.

PURE-MATH MODULE — verified in-sandbox, see tests/test_shamir.py.

Feldman VSS normally commits each polynomial coefficient as g^{c_i} (a
group element) so shares are *verifiable* against public commitments.
That commitment step needs ristretto group exponentiation and lives in
dkg.py (Colab-side). What's here is everything that is pure Z_L
arithmetic: share generation, threshold reconstruction (test-only, see
field.py docstring), and the resharing transformation that
committee.py / reshare.py build on. Splitting it this way means the
*protocol logic* — did resharing preserve the secret? does t-of-n_c
actually reconstruct and (t-1)-of-n_c actually fail? — is fully
testable without any EC library.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .field import L, ShamirPolynomial, ShamirShare, lagrange_coeffs, reconstruct


@dataclass
class CommitteeRoster:
    """Roster state for one epoch. Pure bookkeeping — no crypto here;
    committee.py (Colab-side) adds the signature/consent verification
    around calls to `propose` / `install`.
    """
    epoch: int
    members: list[int]          # client ids holding a VSS share this epoch
    t: int
    installed: bool = True


def split_secret(secret: int, t: int, n_c: int, member_ids: list[int]) -> dict[int, ShamirShare]:
    """(t, n_c) Shamir split of `secret` across member_ids. DKG use only —
    in the real DKG (dkg.py) no single party ever holds the pre-split
    `secret`; it only ever exists as the sum of n_c parties' independent
    sub-secrets. This function is the per-party primitive that the DKG
    composes.
    """
    if len(member_ids) != n_c:
        raise ValueError("member_ids must have exactly n_c entries")
    poly = ShamirPolynomial(t, secret=secret)
    return {i: poly.share_for(i) for i in member_ids}


def reshare(old_shares: dict[int, ShamirShare], t: int, new_member_ids: list[int]) -> dict[int, ShamirShare]:
    """Proactive resharing (ARCHITECTURE.md §6.4): t of the OUTGOING
    members each draw a fresh degree-(t-1) polynomial with ZERO constant
    term and distribute sub-shares to the incoming roster; the sum of
    sub-shares a new member receives is their new share of the SAME
    secret. Here we simulate the "t outgoing members cooperate" step by
    taking exactly `t` of the provided old_shares as the cooperating set.

    Returns the new members' shares. The invariant this buys: summing
    any `t` of the *new* shares with correct Lagrange weights reproduces
    the same secret as the old shares did — without the secret ever
    being assembled in this function or anywhere else.
    """
    cooperating = list(old_shares.values())[:t]
    xs = [s.index for s in cooperating]
    lambdas = lagrange_coeffs(xs, L)

    # Each cooperating member i contributes a zero-constant-term
    # sub-sharing of (lambda_i * old_share_i) to every new member.
    sub_shares_per_new_member: dict[int, int] = {j: 0 for j in new_member_ids}
    for lam, old_share in zip(lambdas, cooperating):
        weighted_secret_contribution = (lam * old_share.value) % L
        sub_poly = ShamirPolynomial(t, secret=weighted_secret_contribution)
        for j in new_member_ids:
            sub_shares_per_new_member[j] = (
                sub_shares_per_new_member[j] + sub_poly.share_for(j).value
            ) % L

    return {j: ShamirShare(j, v) for j, v in sub_shares_per_new_member.items()}


def verify_threshold_property(secret: int, t: int, n_c: int, member_ids: list[int]) -> bool:
    """Sanity check used only by tests: any t distinct shares reconstruct
    the secret exactly; feeding Lagrange only t-1 shares plus a decoy
    point (an invalid, under-determined use of the interpolation) must
    NOT silently recover the same secret.
    """
    shares = split_secret(secret, t, n_c, member_ids)
    all_shares = list(shares.values())

    ok_t = reconstruct(all_shares[:t]) == secret % L

    # t-1 real shares is an under-determined system; interpolating through
    # them plus an arbitrary extra point (x=0 is reserved for the secret,
    # so use a point far outside the roster) must NOT reconstruct `secret`
    # unless by astronomical coincidence.
    decoy = ShamirShare(10_000_000, 1)  # garbage point, not a real share
    wrong = reconstruct(all_shares[: t - 1] + [decoy])
    ok_insufficient = wrong != secret % L

    return ok_t and ok_insufficient
