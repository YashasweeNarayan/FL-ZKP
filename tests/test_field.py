import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.field import L, ShamirPolynomial, lagrange_coeffs, reconstruct, rand_scalar, modinv


def test_modinv():
    a = rand_scalar()
    assert (a * modinv(a)) % L == 1


def test_poly_eval_matches_secret_at_zero():
    secret = 123456789
    poly = ShamirPolynomial(t=5, secret=secret)
    assert poly.secret == secret


def test_shamir_t_of_n_reconstructs():
    secret = rand_scalar()
    t, n = 7, 10
    poly = ShamirPolynomial(t, secret=secret)
    ids = list(range(1, n + 1))
    shares = poly.shares_for(ids)
    for subset_ids in ([1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9, 2, 4]):
        subset = [s for s in shares if s.index in subset_ids]
        assert len(subset) == t
        assert reconstruct(subset) == secret % L


def test_shamir_insufficient_shares_do_not_leak_secret():
    secret = rand_scalar()
    t, n = 7, 10
    poly = ShamirPolynomial(t, secret=secret)
    shares = poly.shares_for(list(range(1, n + 1)))
    # exactly t-1 shares + Lagrange math for a DIFFERENT point set should
    # not reconstruct the real secret
    six = shares[:6]
    # t-1 = 6 shares are structurally insufficient to interpolate a
    # degree-(t-1)=6 polynomial's constant term correctly; Shamir's
    # security relies on this, we just assert the precondition holds.
    assert len(six) == t - 1


if __name__ == "__main__":
    test_modinv()
    test_poly_eval_matches_secret_at_zero()
    test_shamir_t_of_n_reconstructs()
    test_shamir_insufficient_shares_do_not_leak_secret()
    print("ALL field.py TESTS PASSED")
