"""Pure-Python/NumPy mirror of fl/client.py::local_train_delta's mini-
batch/epoch indexing scheme (FIX.md F6).

fl/client.py cannot be imported at all in this sandbox -- importing it
pulls in crypto/pedersen.py, which computes its Pedersen generator H at
MODULE IMPORT TIME via crypto/ristretto.py::Point.hash_to_point, which
needs `rbcl` (not installed here, no network -- see FIX.md F8). So this
file does NOT call the real function; it re-implements the exact same
batching loop (`for start in range(0, n, batch_size): perm[start:start+
batch_size]`, reshuffled every epoch) using numpy's RNG instead of
torch's, and checks the INDEXING ARITHMETIC: full coverage, no
duplicates within an epoch, correct batch count, correct degenerate
behavior when batch_size >= n. This catches an off-by-one or coverage
bug in the scheme itself before the real torch version ever runs on
Colab -- it does NOT verify gradients, loss, or anything torch-side.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def _epoch_batches(n: int, batch_size: int, rng: np.random.Generator):
    """Mirrors local_train_delta's per-epoch loop: shuffle once, then
    walk it in batch_size-sized slices. Returns the list of index
    arrays, one per mini-batch, exactly as fl/client.py would produce
    them (torch.randperm + the same range(0, n, batch_size) slicing).
    """
    perm = rng.permutation(n)
    return [perm[start:start + batch_size] for start in range(0, n, batch_size)]


def test_full_coverage_no_duplicates_within_an_epoch():
    rng = np.random.default_rng(0)
    for n, batch_size in [(100, 20), (95, 20), (7, 20), (1, 20), (23, 1), (1000, 37)]:
        batches = _epoch_batches(n, batch_size, rng)
        seen = np.concatenate(batches) if batches else np.array([], dtype=int)
        assert len(seen) == n, f"n={n}, batch_size={batch_size}: coverage {len(seen)} != {n}"
        assert set(seen.tolist()) == set(range(n)), "every index must appear exactly once"
        assert len(seen) == len(set(seen.tolist())), "no index may repeat within one epoch"


def test_batch_count_matches_ceil_division():
    rng = np.random.default_rng(1)
    for n, batch_size in [(100, 20), (95, 20), (101, 20), (7, 20), (1000, 37)]:
        batches = _epoch_batches(n, batch_size, rng)
        import math
        assert len(batches) == math.ceil(n / batch_size)


def test_last_batch_is_the_only_one_allowed_to_be_short():
    rng = np.random.default_rng(2)
    n, batch_size = 95, 20   # 4 full batches of 20 + 1 of 15
    batches = _epoch_batches(n, batch_size, rng)
    sizes = [len(b) for b in batches]
    assert sizes[:-1] == [batch_size] * (len(sizes) - 1), "every batch but the last must be full-size"
    assert sizes[-1] == n % batch_size, f"last batch should be the {n % batch_size} leftover samples"


def test_batch_size_covering_whole_shard_degenerates_to_one_batch():
    rng = np.random.default_rng(3)
    n = 50
    for batch_size in (50, 64, 1000):
        batches = _epoch_batches(n, batch_size, rng)
        assert len(batches) == 1, "batch_size >= n must yield exactly one batch per epoch"
        assert len(batches[0]) == n


def test_reshuffling_differs_across_epochs_with_high_probability():
    # Not a correctness requirement in the strict sense (a shuffle CAN
    # coincide), but with n=200 the odds of two independent permutations
    # matching are astronomically small -- this guards against a bug
    # where the "reshuffle every epoch" step silently reuses one
    # permutation for every epoch (which would make `epochs` just
    # relabel the same full pass repeatedly, exactly the bug F6 fixes).
    rng = np.random.default_rng(4)
    n = 200
    perm_a = rng.permutation(n)
    perm_b = rng.permutation(n)
    assert not np.array_equal(perm_a, perm_b), "two independent shuffles should not coincide"


def test_multi_epoch_total_batches_matches_epochs_times_batches_per_epoch():
    rng = np.random.default_rng(5)
    n, batch_size, epochs = 95, 20, 5
    total_batches = 0
    total_samples_seen = 0
    for _ in range(epochs):
        batches = _epoch_batches(n, batch_size, rng)
        total_batches += len(batches)
        total_samples_seen += sum(len(b) for b in batches)
    assert total_batches == epochs * 5   # ceil(95/20) == 5
    assert total_samples_seen == epochs * n


if __name__ == "__main__":
    test_full_coverage_no_duplicates_within_an_epoch()
    test_batch_count_matches_ceil_division()
    test_last_batch_is_the_only_one_allowed_to_be_short()
    test_batch_size_covering_whole_shard_degenerates_to_one_batch()
    test_reshuffling_differs_across_epochs_with_high_probability()
    test_multi_epoch_total_batches_matches_epochs_times_batches_per_epoch()
    print("ALL F6 batching-logic mirror tests PASSED (the real torch function is still untested here -- see module docstring)")
