"""FIX.md F3: pairwise_mask_cached must be indistinguishable from
pairwise_mask (correctness), while only expanding each pair's magnitude
vector once when a cache is supplied (the actual saving).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.secagg import P_MASK, pairwise_mask, pairwise_mask_cached

DIM = 4096
ROUND_NONCE = b"round-cache-test"


def test_cache_none_matches_direct_call():
    secret = os.urandom(32)
    for sign in (1, -1):
        direct = pairwise_mask(secret, ROUND_NONCE, DIM, sign)
        via_none = pairwise_mask_cached(secret, ROUND_NONCE, DIM, sign, None, ("x", "y"))
        np.testing.assert_array_equal(direct, via_none)


def test_cached_result_matches_uncached_for_both_signs():
    secret = os.urandom(32)
    cache = {}
    v_plus = pairwise_mask_cached(secret, ROUND_NONCE, DIM, 1, cache, (0, 1))
    v_minus = pairwise_mask_cached(secret, ROUND_NONCE, DIM, -1, cache, (0, 1))
    np.testing.assert_array_equal(v_plus, pairwise_mask(secret, ROUND_NONCE, DIM, 1))
    np.testing.assert_array_equal(v_minus, pairwise_mask(secret, ROUND_NONCE, DIM, -1))
    # and they must still cancel when added, exactly like the uncached pair
    assert np.array_equal((v_plus + v_minus) % P_MASK, np.zeros(DIM, dtype=np.int64))


def test_second_party_reuses_the_cached_expansion_not_a_fresh_one():
    """The whole point of F3: a second call with the same pair_key must
    not re-derive the magnitude from the PRG -- it must reuse exactly
    the array object the first call stored."""
    secret = os.urandom(32)
    cache = {}
    pairwise_mask_cached(secret, ROUND_NONCE, DIM, 1, cache, (2, 5))
    stored = cache[(2, 5)]
    # Call again with a DIFFERENT (wrong) secret but the same pair_key:
    # if F3 is wired correctly, the cache hit means the wrong secret is
    # never consulted and the result still matches the original secret.
    wrong_secret = os.urandom(32)
    v = pairwise_mask_cached(wrong_secret, ROUND_NONCE, DIM, -1, cache, (2, 5))
    np.testing.assert_array_equal(v, (-1 * stored) % P_MASK)
    np.testing.assert_array_equal(v, pairwise_mask(secret, ROUND_NONCE, DIM, -1))


def test_different_pair_keys_do_not_collide_in_one_cache():
    cache = {}
    sA = os.urandom(32)
    sB = os.urandom(32)
    a = pairwise_mask_cached(sA, ROUND_NONCE, DIM, 1, cache, (0, 1))
    b = pairwise_mask_cached(sB, ROUND_NONCE, DIM, 1, cache, (0, 2))
    assert not np.array_equal(a, b)
    assert len(cache) == 2


def test_cache_is_per_round_not_cross_round():
    """A fresh cache per round (as fl/round_loop.py creates) must not
    leak a stale vector from a different round_nonce -- this is a
    usage contract, not something pairwise_mask_cached enforces itself,
    so this test documents it: a cache built for round_nonce A, reused
    against round_nonce B's secret, silently returns round A's masks.
    """
    secret = os.urandom(32)
    cache_round_a = {}
    pairwise_mask_cached(secret, b"round-A", DIM, 1, cache_round_a, (0, 1))
    stale = pairwise_mask_cached(secret, b"round-B", DIM, 1, cache_round_a, (0, 1))
    fresh = pairwise_mask(secret, b"round-B", DIM, 1)
    assert not np.array_equal(stale, fresh), (
        "a cache must be rebuilt every round (fl/round_loop.py does this) -- "
        "this failure mode is exactly why"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "PASSED")
