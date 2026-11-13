"""FIX.md F5: the per-round vector store, and the streaming aggregation
that replaces holding every client's vector in RAM.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.secagg import P_MASK, aggregate_masked
from crypto.vector_commit import weighted_digest
from fl.vector_store import RoundVectorStore, suggested_batch_size

D = 2048


def _vec(seed, d=D):
    return np.mod(np.random.default_rng(seed).integers(-5000, 5000, d), P_MASK).astype(np.int64)


# -- storage round-trip ------------------------------------------------

def test_resident_path_round_trips_exactly():
    with RoundVectorStore(budget_bytes=10 * 1024 ** 2) as store:
        v = _vec(0)
        store.put(("q", 0), v)
        np.testing.assert_array_equal(store.get(("q", 0)), v)
        assert store.stats()["spilled_arrays"] == 0


def test_spilled_path_round_trips_exactly():
    """budget 0 forces every array to disk -- the path that matters at
    50 clients, and the one that would otherwise never run in a small
    test."""
    with RoundVectorStore(budget_bytes=0) as store:
        vectors = {("x", k): _vec(k) for k in range(5)}
        for key, v in vectors.items():
            store.put(key, v)
        assert store.stats()["spilled_arrays"] == 5
        assert store.stats()["resident_arrays"] == 0
        for key, v in vectors.items():
            np.testing.assert_array_equal(store.get(key), v)
        # re-reading gives the same values again, not a consumed stream
        for key, v in vectors.items():
            np.testing.assert_array_equal(store.get(key), v)


def test_mixed_residency_keeps_peak_under_budget():
    one = _vec(0).nbytes
    with RoundVectorStore(budget_bytes=3 * one + 7) as store:
        originals = {}
        for k in range(10):
            v = _vec(k)
            originals[k] = v
            store.put(("q", k), v)
        st = store.stats()
        assert st["resident_arrays"] == 3, st
        assert st["spilled_arrays"] == 7, st
        assert st["peak_resident_bytes"] <= st["budget_bytes"]
        # every value survives, whichever side of the budget it fell on
        for k, v in originals.items():
            np.testing.assert_array_equal(store.get(("q", k)), v)


def test_returned_arrays_are_read_only_on_both_paths():
    """A resident get() hands back the stored array itself while a
    spilled get() hands back a fresh load. If callers could mutate, a
    bug would appear only below the budget -- so neither is writable."""
    with RoundVectorStore(budget_bytes=_vec(0).nbytes) as store:
        store.put(("q", 0), _vec(0))   # resident
        store.put(("q", 1), _vec(1))   # spilled
        for key in (("q", 0), ("q", 1)):
            arr = store.get(key)
            assert not arr.flags.writeable
            try:
                arr[0] = 123
                assert False, f"{key} should not be writable"
            except ValueError:
                pass


# -- lifecycle ---------------------------------------------------------

def test_close_removes_spill_files_and_is_idempotent():
    store = RoundVectorStore(budget_bytes=0)
    store.put(("x", 0), _vec(0))
    spill_dir = store.stats()["dir"]
    assert spill_dir is not None and os.path.isdir(spill_dir)
    assert os.listdir(spill_dir)
    store.close()
    assert not os.path.exists(spill_dir)
    store.close()  # must not raise


def test_use_after_close_raises():
    store = RoundVectorStore(budget_bytes=0)
    store.put(("x", 0), _vec(0))
    store.close()
    for call in (lambda: store.put(("x", 1), _vec(1)), lambda: store.get(("x", 0))):
        try:
            call()
            assert False, "expected RuntimeError after close"
        except RuntimeError:
            pass


def test_duplicate_put_and_missing_get_raise():
    with RoundVectorStore(budget_bytes=10 * 1024 ** 2) as store:
        store.put(("q", 0), _vec(0))
        try:
            store.put(("q", 0), _vec(1))
            assert False, "expected KeyError on duplicate put"
        except KeyError:
            pass
        try:
            store.get(("q", 99))
            assert False, "expected KeyError on missing key"
        except KeyError:
            pass


def test_drop_frees_resident_bytes_and_spill_files():
    one = _vec(0).nbytes
    with RoundVectorStore(budget_bytes=10 * one) as store:
        store.put(("q", 0), _vec(0))
        store.drop(("q", 0))
        assert store.stats()["resident_arrays"] == 0
        # dropping a key that was never stored is a no-op, not an error
        store.drop(("q", 42))
    with RoundVectorStore(budget_bytes=0) as store:
        store.put(("x", 0), _vec(0))
        path = os.path.join(store.stats()["dir"], "x_0.npy")
        assert os.path.exists(path)
        store.drop(("x", 0))
        assert not os.path.exists(path)


# -- the actual memory claim -------------------------------------------

def test_peak_residency_does_not_grow_with_client_count():
    """The F5 claim in one assertion: the same budget holds the same
    peak whether 8 clients or 200 are stored."""
    budget = 4 * _vec(0).nbytes
    peaks = []
    for n_clients in (8, 30, 50, 200):
        with RoundVectorStore(budget_bytes=budget) as store:
            for k in range(n_clients):
                store.put(("q", k), _vec(k))
                store.put(("x", k), _vec(1000 + k))
            peaks.append(store.stats()["peak_resident_bytes"])
    assert len(set(peaks)) == 1, f"peak RAM varied with client count: {peaks}"
    assert peaks[0] <= budget


def test_suggested_batch_size_is_independent_of_client_count():
    d = 11_208_318
    b = suggested_batch_size(d, 1024 ** 3)
    assert b >= 1
    assert b == suggested_batch_size(d, 1024 ** 3)          # no hidden n dependence
    assert suggested_batch_size(d, 0) == 1                   # never zero
    assert suggested_batch_size(d, 10 ** 15) <= 16           # capped
    assert suggested_batch_size(2 * d, 1024 ** 3) <= b       # bigger model, smaller batch


# -- streaming equals the old list-based computation --------------------

def test_aggregate_masked_accepts_a_generator():
    vectors = [_vec(k) for k in range(6)]
    from_list = aggregate_masked(vectors)
    from_gen = aggregate_masked(v for v in vectors)
    np.testing.assert_array_equal(from_list, from_gen)
    try:
        aggregate_masked(iter([]))
        assert False, "expected ValueError on an empty iterable"
    except ValueError:
        pass


def test_aggregate_masked_streams_from_the_store_unchanged():
    vectors = {k: _vec(k) for k in range(7)}
    reference = aggregate_masked(list(vectors.values()))
    with RoundVectorStore(budget_bytes=0) as store:   # force disk
        for k, v in vectors.items():
            store.put(("x", k), v)
        streamed = aggregate_masked(store.get(("x", k)) for k in vectors)
    np.testing.assert_array_equal(reference, streamed)


def test_fused_streaming_pass_matches_separate_list_passes():
    """fl/server.py now computes the masked sum and the per-client
    digest sum in ONE pass over streamed vectors. Both must equal what
    the two separate list comprehensions produced before."""
    rho = b"round-stream"
    vectors = {k: _vec(k) for k in range(6)}
    want_sum = aggregate_masked(list(vectors.values()))
    want_digest = sum(weighted_digest(v, seed=rho) for v in vectors.values())

    with RoundVectorStore(budget_bytes=0) as store:
        for k, v in vectors.items():
            store.put(("x", k), v)
        got_sum = None
        got_digest = 0
        for k in vectors:
            x_k = store.get(("x", k))
            got_sum = (np.array(x_k, dtype=np.int64) % P_MASK if got_sum is None
                       else (got_sum + x_k) % P_MASK)
            got_digest += weighted_digest(x_k, seed=rho)
            del x_k

    np.testing.assert_array_equal(want_sum, got_sum)
    assert want_digest == got_digest
    # the digest sum is the EXACT integer sum, deliberately not reduced
    # mod P_MASK (fl/server.py explains why V5 needs it that way)
    assert want_digest >= 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "PASSED")
