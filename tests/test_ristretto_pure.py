"""Pure-Python tests for crypto/ristretto.py's wide-scalar reduction.

This covers ONLY `_wide_scalar_reduce` and `Scalar.from_int` -- the two
pieces of crypto/ristretto.py that do NOT call into `rbcl` and are
therefore actually runnable in this network-less sandbox. Everything
else in that module (group addition, scalar multiplication, hash-to-
point, ...) needs `rbcl`'s libsodium bindings and is EC-DEPENDENT,
UNTESTED HERE -- see notebooks/00_Setup_and_Crypto_Tests.ipynb and
FIX.md F8.

Why this piece exists at all: rbcl's `crypto_core_ristretto255_scalar_
reduce` wrapper only accepts a 32-byte buffer (checked directly against
its source), but this module needs to reduce a 64-byte wide hash/int
(the whole point of `_NONREDUCEDSCALARBYTES` existing in libsodium's C
API). So the reduction mod L is done directly in Python instead of
routed through rbcl -- this file is what actually exercises that math.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.ristretto import _wide_scalar_reduce, _GROUP_ORDER_L, Scalar, _SCALAR_ZERO

L = _GROUP_ORDER_L


def test_group_order_matches_rfc8032():
    # The standard Ed25519/ristretto255 scalar group order, as published
    # in RFC 8032 ("Edwards-Curve Digital Signature Algorithm"), section 5.1:
    # L = 2^252 + 27742317777372353535851937790883648493.
    assert L == 2**252 + 27742317777372353535851937790883648493
    assert L.bit_length() == 253


def test_reduce_of_zero_is_zero():
    assert _wide_scalar_reduce(b"\x00" * 64) == b"\x00" * 32


def test_reduce_below_L_is_identity():
    # Any value already < L, zero-extended to 64 bytes, must come back
    # unchanged in its 32-byte encoding -- reduction mod L of x < L is x.
    x = 123456789
    wide = x.to_bytes(32, "little") + b"\x00" * 32
    assert _wide_scalar_reduce(wide) == x.to_bytes(32, "little")


def test_reduce_of_L_itself_is_zero():
    wide = L.to_bytes(64, "little")
    assert _wide_scalar_reduce(wide) == b"\x00" * 32


def test_reduce_of_L_plus_one_is_one():
    wide = (L + 1).to_bytes(64, "little")
    assert _wide_scalar_reduce(wide) == (1).to_bytes(32, "little")


def test_reduce_matches_manual_mod_for_random_wide_values():
    import random
    rng = random.Random(0)
    for _ in range(200):
        x = rng.getrandbits(512)
        wide = x.to_bytes(64, "little")
        expected = (x % L).to_bytes(32, "little")
        assert _wide_scalar_reduce(wide) == expected


def test_reduce_output_is_always_32_bytes():
    import random
    rng = random.Random(1)
    for _ in range(50):
        x = rng.getrandbits(512)
        assert len(_wide_scalar_reduce(x.to_bytes(64, "little"))) == 32


def test_scalar_from_int_zero_matches_sentinel():
    # crypto/ristretto.py short-circuits scalar_mul / scalar_mul_base on
    # this exact byte string (_SCALAR_ZERO) to avoid libsodium's reject-
    # zero-scalar behavior -- Scalar.from_int(0) MUST produce it exactly,
    # or that short-circuit silently never fires.
    assert Scalar.from_int(0).b == _SCALAR_ZERO
    assert Scalar.from_int(0).b == b"\x00" * 32


def test_scalar_from_int_one():
    assert Scalar.from_int(1).b == (1).to_bytes(32, "little")


def test_scalar_from_int_reduces_values_at_and_above_L():
    assert Scalar.from_int(L).b == b"\x00" * 32
    assert Scalar.from_int(L + 5).b == (5).to_bytes(32, "little")


def test_scalar_from_int_handles_huge_ints_beyond_512_bits():
    # from_int's docstring/code claims it also reduces ints too big for a
    # 512-bit buffer by pre-reducing mod 2**512 before the 64-byte encode.
    huge = (L * 7919) + 42  # arbitrary multiple of L, offset by 42, well within 512 bits actually --
    # force something genuinely > 2**512 to exercise that branch:
    huge = (2**520) + 42
    result = Scalar.from_int(huge)
    expected = (huge % (2**512) % L).to_bytes(32, "little")
    assert result.b == expected


if __name__ == "__main__":
    test_group_order_matches_rfc8032()
    test_reduce_of_zero_is_zero()
    test_reduce_below_L_is_identity()
    test_reduce_of_L_itself_is_zero()
    test_reduce_of_L_plus_one_is_one()
    test_reduce_matches_manual_mod_for_random_wide_values()
    test_reduce_output_is_always_32_bytes()
    test_scalar_from_int_zero_matches_sentinel()
    test_scalar_from_int_one()
    test_scalar_from_int_reduces_values_at_and_above_L()
    test_scalar_from_int_handles_huge_ints_beyond_512_bits()
    print("ALL ristretto.py PURE-MATH TESTS PASSED (rbcl-dependent group ops still untested here)")
