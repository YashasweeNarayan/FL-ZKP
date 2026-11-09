import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.field import L, rand_scalar, reconstruct
from crypto.shamir import split_secret, reshare, verify_threshold_property


def test_split_and_reconstruct_7_of_10():
    secret = rand_scalar()
    member_ids = list(range(1, 11))  # 10 key holders
    shares = split_secret(secret, t=7, n_c=10, member_ids=member_ids)
    assert len(shares) == 10

    any_seven = list(shares.values())[:7]
    assert reconstruct(any_seven) == secret % L

    any_other_seven = list(shares.values())[3:10]
    assert reconstruct(any_other_seven) == secret % L


def test_threshold_property_holds_for_the_real_committee_params():
    secret = rand_scalar()
    assert verify_threshold_property(secret, t=7, n_c=10, member_ids=list(range(1, 11)))


def test_reshare_preserves_the_secret_pk_trace_invariant():
    """This is the test that matters most for §6.4: PK_trace must be
    invariant across a committee rotation. We can't check PK_trace itself
    here (that's g^sk_trace, an EC operation) but we CAN check the
    algebraic invariant that makes PK_trace invariant possible: the new
    roster's shares must reconstruct to the IDENTICAL secret as the old
    roster's shares did.
    """
    secret = rand_scalar()
    old_members = list(range(1, 11))        # committee epoch e
    new_members = list(range(101, 111))      # committee epoch e+1, disjoint ids

    old_shares = split_secret(secret, t=7, n_c=10, member_ids=old_members)
    new_shares = reshare(old_shares, t=7, new_member_ids=new_members)

    assert len(new_shares) == 10
    any_seven_new = list(new_shares.values())[:7]
    assert reconstruct(any_seven_new) == secret % L, (
        "reshare() must preserve the secret exactly -- this is the "
        "invariant that lets old escrow ciphertexts stay openable "
        "after a committee rotation without re-encryption"
    )


def test_old_shares_do_not_combine_with_new_shares():
    """Forward security: a retired member's OLD share must not combine
    with a mix of NEW shares to reconstruct anything meaningful. Since
    old and new rosters use disjoint x-coordinates (ids), Lagrange
    interpolation over a mixed set is simply the wrong set of points for
    either polynomial and will not reconstruct the secret.
    """
    secret = rand_scalar()
    old_members = list(range(1, 11))
    new_members = list(range(101, 111))
    old_shares = split_secret(secret, t=7, n_c=10, member_ids=old_members)
    new_shares = reshare(old_shares, t=7, new_member_ids=new_members)

    mixed = list(old_shares.values())[:3] + list(new_shares.values())[:4]
    assert reconstruct(mixed) != secret % L


if __name__ == "__main__":
    test_split_and_reconstruct_7_of_10()
    test_threshold_property_holds_for_the_real_committee_params()
    test_reshare_preserves_the_secret_pk_trace_invariant()
    test_old_shares_do_not_combine_with_new_shares()
    print("ALL shamir.py TESTS PASSED")
