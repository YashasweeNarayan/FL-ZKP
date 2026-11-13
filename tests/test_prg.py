"""FIX.md F2: the AES-CTR expander and the exact uint64 weighted sum."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.prg import P_MASK, derive_stream_key, expand_field_stream, iter_field_stream
from crypto.vector_commit import _dot_mod_p, _weight_stream, weighted_digest


def test_stream_is_deterministic_in_range_and_right_length():
    k = derive_stream_key(b"t", b"seed")
    a = expand_field_stream(k, 100_003)
    b = expand_field_stream(k, 100_003)
    assert a.shape == (100_003,) and a.dtype == np.int64
    assert np.array_equal(a, b)
    assert int(a.min()) >= 0 and int(a.max()) < P_MASK


def test_stream_prefix_property_and_chunk_independence():
    # A shorter request is a prefix of a longer one, and the chunk size
    # used to pull the stream does not change its contents.
    k = derive_stream_key(b"t", b"seed")
    long = expand_field_stream(k, 3_000_000)   # spans several internal chunks
    assert np.array_equal(expand_field_stream(k, 1234), long[:1234])
    pieces = np.concatenate(list(iter_field_stream(k, 3_000_000, chunk=77_777)))
    assert np.array_equal(pieces, long)


def test_key_derivation_separates_domains_and_part_boundaries():
    assert derive_stream_key(b"a", b"x") != derive_stream_key(b"b", b"x")
    assert derive_stream_key(b"d", b"ab", b"c") != derive_stream_key(b"d", b"a", b"bc")
    a = expand_field_stream(derive_stream_key(b"d", b"s1"), 1000)
    b = expand_field_stream(derive_stream_key(b"d", b"s2"), 1000)
    assert not np.array_equal(a, b)


def test_stream_looks_uniform():
    v = expand_field_stream(derive_stream_key(b"t", b"u"), 400_000)
    mean = float(v.mean()) / P_MASK
    assert abs(mean - 0.5) < 0.005
    top_bits = (v >> 58).astype(np.int64)           # 3 top bits -> 8 buckets
    counts = np.bincount(top_bits, minlength=8)
    assert counts.min() > 0.97 * 50_000 and counts.max() < 1.03 * 50_000


def test_dot_mod_p_matches_big_integer_arithmetic():
    rng = np.random.default_rng(0)
    for n in (1, 7, 5000):
        w = rng.integers(0, P_MASK, size=n, dtype=np.int64)
        v = rng.integers(0, P_MASK, size=n, dtype=np.int64)
        want = sum(int(a) * int(b) for a, b in zip(w, v)) % P_MASK
        assert _dot_mod_p(w, v) == want
    # extreme values: every entry at the top of the range
    w = np.full(4096, P_MASK - 1, dtype=np.int64)
    assert _dot_mod_p(w, w) == (4096 * (P_MASK - 1) ** 2) % P_MASK
    z = np.zeros(10, dtype=np.int64)
    assert _dot_mod_p(z, w[:10]) == 0


def test_weighted_digest_matches_reference_across_chunk_boundary():
    rng = np.random.default_rng(1)
    d = (1 << 18) * 2 + 321                      # more than two digest chunks
    vec = rng.integers(0, P_MASK, size=d, dtype=np.int64)
    w = _weight_stream(b"rho", d)
    want = sum(int(a) * int(b) for a, b in zip(w, vec)) % P_MASK
    assert weighted_digest(vec, b"rho") == want
    # object-dtype input takes the big-integer path and must agree
    assert weighted_digest(vec[:5000].astype(object), b"rho") == weighted_digest(vec[:5000], b"rho")


def test_weighted_digest_is_linear_mod_p():
    rng = np.random.default_rng(2)
    u = rng.integers(0, P_MASK, size=50_000, dtype=np.int64)
    v = rng.integers(0, P_MASK, size=50_000, dtype=np.int64)
    s = ((u.astype(object) + v.astype(object)) % P_MASK).astype(np.int64)
    assert (weighted_digest(u, b"r") + weighted_digest(v, b"r")) % P_MASK == weighted_digest(s, b"r")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "PASSED")
