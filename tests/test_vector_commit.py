"""Pure-math tests for crypto/vector_commit.py -- no EC dependency, runs
in this sandbox. Validates the two properties the V5 fix actually
depends on: linearity (so the Pedersen homomorphism lines up) and
soundness against coordinate-sum collisions (the exact flaw being
fixed).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from crypto.vector_commit import weighted_digest, P_MASK, _weight_stream


def test_weight_stream_is_deterministic_given_same_seed():
    w1 = _weight_stream(b"round-7", 500)
    w2 = _weight_stream(b"round-7", 500)
    assert np.array_equal(w1, w2)
    w3 = _weight_stream(b"round-8", 500)
    assert not np.array_equal(w1, w3)


def test_weight_stream_values_in_range():
    w = _weight_stream(b"seed", 2000)
    assert w.min() >= 0
    assert w.max() < P_MASK


def test_digest_is_linear_under_mod_p_mask_addition():
    rng = np.random.default_rng(42)
    dim = 1000
    u = rng.integers(0, P_MASK, size=dim, dtype=np.int64)
    v = rng.integers(0, P_MASK, size=dim, dtype=np.int64)
    seed = b"test-round-rho-t"

    digest_u = weighted_digest(u, seed)
    digest_v = weighted_digest(v, seed)
    combined = (u.astype(object) + v.astype(object)) % P_MASK
    combined = combined.astype(np.int64)
    digest_combined = weighted_digest(combined, seed)

    assert (digest_u + digest_v) % P_MASK == digest_combined


def test_digest_linearity_holds_across_many_clients():
    """Mirrors the actual aggregation shape: secagg.py's aggregate_masked
    sums N client vectors coordinatewise mod P_MASK; the sum of each
    client's individual digest should equal the digest of that same
    aggregate, exactly the property fl/server.py's V5 check needs.
    """
    rng = np.random.default_rng(7)
    dim = 300
    n_clients = 12
    seed = b"rho-t-for-this-round"

    client_vectors = [rng.integers(0, P_MASK, size=dim, dtype=np.int64) for _ in range(n_clients)]
    aggregate = np.zeros(dim, dtype=object)
    for v in client_vectors:
        aggregate = (aggregate + v.astype(object)) % P_MASK
    aggregate = aggregate.astype(np.int64)

    sum_of_digests = 0
    for v in client_vectors:
        sum_of_digests = (sum_of_digests + weighted_digest(v, seed)) % P_MASK

    digest_of_aggregate = weighted_digest(aggregate, seed)
    assert sum_of_digests == digest_of_aggregate


def test_digest_catches_a_coordinate_sum_preserving_tamper():
    """THE actual flaw being fixed: a plain coordinate-sum digest cannot
    tell two vectors apart if an attacker moves some amount from one
    coordinate to another (sum unchanged). The weighted digest must.
    """
    rng = np.random.default_rng(99)
    dim = 500
    seed = b"rho-t-tamper-test"

    original = rng.integers(1, P_MASK // 2, size=dim, dtype=np.int64)
    tampered = original.copy()
    # move 1000 units from coordinate 0 to coordinate 1 -- plain sum is IDENTICAL
    move = 1000
    tampered[0] = (int(tampered[0]) - move) % P_MASK
    tampered[1] = (int(tampered[1]) + move) % P_MASK

    assert int(original.astype(object).sum() % P_MASK) == int(tampered.astype(object).sum() % P_MASK), (
        "test setup bug: the whole point is these should have an identical coordinate sum"
    )

    digest_original = weighted_digest(original, seed)
    digest_tampered = weighted_digest(tampered, seed)
    assert digest_original != digest_tampered, (
        "weighted digest failed to catch a coordinate-sum-preserving tamper -- "
        "this is exactly the attack the old placeholder digest was vulnerable to"
    )


def test_digest_different_seed_gives_different_digest_with_overwhelming_probability():
    rng = np.random.default_rng(3)
    vec = rng.integers(0, P_MASK, size=200, dtype=np.int64)
    d1 = weighted_digest(vec, b"rho-t-round-1")
    d2 = weighted_digest(vec, b"rho-t-round-2")
    assert d1 != d2


if __name__ == "__main__":
    test_weight_stream_is_deterministic_given_same_seed()
    test_weight_stream_values_in_range()
    test_digest_is_linear_under_mod_p_mask_addition()
    test_digest_linearity_holds_across_many_clients()
    test_digest_catches_a_coordinate_sum_preserving_tamper()
    test_digest_different_seed_gives_different_digest_with_overwhelming_probability()
    print("ALL tests passed: crypto/vector_commit.py")
