"""PURE-MATH tests for the unmasking round (crypto/secagg.py's
`unmask_aggregate`, `ShareReleasePolicy`, `pairwise_round_seed`).

Why this file exists: until 2026-11-13 the round loop summed the
accepted clients' masked vectors and dequantized the result directly.
Pairwise masks cancel in that sum; self-masks do not, and neither do
the pairwise masks an accepted client shares with a rejected one. The
"aggregate" applied to the model was therefore uniformly random.
test_sum_without_unmasking_is_not_the_aggregate pins that failure down
so it cannot come back unnoticed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.secagg import (
    P_MASK, ShareReleasePolicy, aggregate_masked, new_self_mask_secret, pairwise_mask,
    pairwise_round_seed, self_mask, self_secret_bytes, shamir_share_self_secret, unmask_aggregate,
)

DIM = 1500
RHO = b"round-3"


def _setup(n, t, seed=0):
    rng = np.random.default_rng(seed)
    q = [np.mod(rng.integers(-5000, 5000, size=DIM), P_MASK).astype(np.int64) for _ in range(n)]
    pair = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair[(i, j)] = pair[(j, i)] = os.urandom(32)
    self_secrets = [new_self_mask_secret() for _ in range(n)]
    holder_ids = list(range(1, n + 1))
    shares = [shamir_share_self_secret(s, t, holder_ids) for s in self_secrets]   # shares[k][holder_id]
    masked = []
    for i in range(n):
        x = q[i].copy()
        for j in range(n):
            if j != i:
                x = (x + pairwise_mask(pair[(i, j)], RHO, DIM, 1 if i < j else -1)) % P_MASK
        masked.append((x + self_mask(self_secret_bytes(self_secrets[i]), RHO, DIM)) % P_MASK)
    return q, pair, shares, masked


def _release(accepted, excluded, pair, shares, t):
    """What the accepted (alive) clients hand over once the accepted set is fixed."""
    policy = ShareReleasePolicy()
    self_shares = {}
    for k in accepted:
        policy.release_self_share(k)
        self_shares[k] = [shares[k][h + 1] for h in accepted[:t]]
    seeds = {}
    for k in excluded:
        policy.release_pairwise_seed(k)
        for j in accepted:
            seeds[(j, k)] = pairwise_round_seed(pair[(j, k)], RHO)
    return self_shares, seeds


def _true_sum(q, idx):
    return aggregate_masked([q[i] for i in idx])


def test_sum_without_unmasking_is_not_the_aggregate():
    q, pair, shares, masked = _setup(n=6, t=4)
    assert not np.array_equal(aggregate_masked(masked), _true_sum(q, range(6)))


def test_all_accepted_recovers_exact_sum():
    n, t = 6, 4
    q, pair, shares, masked = _setup(n, t)
    accepted = list(range(n))
    self_shares, seeds = _release(accepted, [], pair, shares, t)
    out = unmask_aggregate(aggregate_masked(masked), accepted, self_shares, t, seeds, RHO, DIM)
    assert np.array_equal(out, _true_sum(q, accepted))


def test_rejected_clients_are_removed_exactly():
    n, t = 9, 5
    q, pair, shares, masked = _setup(n, t, seed=1)
    excluded = [2, 7]
    accepted = [i for i in range(n) if i not in excluded]
    self_shares, seeds = _release(accepted, excluded, pair, shares, t)
    out = unmask_aggregate(aggregate_masked([masked[i] for i in accepted]), accepted, self_shares, t, seeds, RHO, DIM)
    assert np.array_equal(out, _true_sum(q, accepted))


def test_excluded_pair_masks_matter():
    """Self-masks alone are not enough once anyone is excluded."""
    n, t = 6, 4
    q, pair, shares, masked = _setup(n, t, seed=2)
    accepted = [0, 1, 2, 3, 4]
    self_shares, _ = _release(accepted, [5], pair, shares, t)
    out = unmask_aggregate(aggregate_masked([masked[i] for i in accepted]), accepted, self_shares, t, {}, RHO, DIM)
    assert not np.array_equal(out, _true_sum(q, accepted))


def test_too_few_shares_raises_instead_of_corrupting():
    n, t = 6, 4
    q, pair, shares, masked = _setup(n, t, seed=3)
    accepted = list(range(n))
    self_shares, seeds = _release(accepted, [], pair, shares, t)
    self_shares[0] = self_shares[0][: t - 1]
    try:
        unmask_aggregate(aggregate_masked(masked), accepted, self_shares, t, seeds, RHO, DIM)
    except ValueError:
        return
    raise AssertionError("unmasking with t-1 shares must fail loudly")


def test_policy_never_releases_both_kinds_about_one_client():
    """The double-masking guarantee (R5): a server that asks for BOTH a
    client's self-mask shares and its pairwise seeds is refused, in
    either order -- with both it could strip that client's vector bare.
    """
    for first, second in (("release_self_share", "release_pairwise_seed"),
                          ("release_pairwise_seed", "release_self_share")):
        policy = ShareReleasePolicy()
        getattr(policy, first)(4)
        try:
            getattr(policy, second)(4)
        except PermissionError:
            continue
        raise AssertionError(f"{first} then {second} about the same client must be refused")


def test_round_seed_is_round_specific():
    secret = os.urandom(32)
    assert pairwise_round_seed(secret, b"round-1") != pairwise_round_seed(secret, b"round-2")
    a = pairwise_mask(secret, b"round-1", 64, 1)
    b = pairwise_mask(secret, b"round-2", 64, 1)
    assert not np.array_equal(a, b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} PASSED")
    print("\nALL unmasking TESTS PASSED")
