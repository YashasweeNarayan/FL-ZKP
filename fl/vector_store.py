"""Per-round storage for the two O(model_dim) arrays each client produces.

FIX.md F5. The round loop used to hold every client's `q_k` and
`x_k = q_k + m_k` in RAM simultaneously, for the whole round, because
both are produced in phase 1 but consumed later:

    q_k  -- phase 2's sketch, and the simulation's own aggregate check
    x_k  -- the server's masked-vector sum and its per-client digest

Two int64 arrays per client is `2 * n * model_dim * 8` bytes:
2.8 GB at 8 clients and ResNet-50, 10.6 GB at 30 (the 2026-11-12
crash), 17.6 GB at 50. That is a hard ceiling on client count, not a
tuning problem.

Nothing in the protocol needs them resident together. Every consumer
reads them ONE CLIENT AT A TIME (phase 2 reads a small batch; see
`fl/client.py::sketch_phase1_batch`), so the arrays only have to be
*retrievable*, not *resident*. This store keeps what fits in a byte
budget and spills the rest to `.npy` files, so peak RAM is set by the
budget rather than by the number of clients.

What this does NOT change: the order of protocol messages. Phase 1
still completes for every client before sigma_t is revealed and phase 2
begins (ARCHITECTURE.md line 194: "S publishes sigma_t ... AFTER all
c_k are on BB"). Where an array is held is not a protocol fact; a naive
"fuse phase 1 and 2 per client" restructure WOULD have broken that
ordering, which is why this takes the storage route instead.

Faithfulness note for the paper: a real server holds one submission at
a time and a running accumulator, and a real client holds only its own
vectors. Streaming is what a real deployment does; holding all n was
the artifact of running every party in one process.

Arrays come back read-only. The resident path returns the stored array
itself while the spilled path returns a fresh load, so a caller that
mutated a returned array would corrupt data at small client counts and
not at large ones -- the worst kind of bug to find later. Making every
return read-only turns that into an immediate, loud failure instead.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

# Two separate pools, deliberately not one knob shared between them:
#
#   DEFAULT_BUDGET_BYTES       arrays this store keeps resident for the
#                              whole round
#   DEFAULT_SKETCH_BATCH_BYTES the transient block phase 2 builds to
#                              sketch several clients at once
#
# They can be live at the same time, so a round's peak for client
# vectors is about the sum, ~3 GiB at these defaults -- comfortable
# beside torch, the model and the data on Colab's ~12.7 GB. Sharing one
# number between them would have understated the peak by the batch.
DEFAULT_BUDGET_BYTES = 2 * 1024 ** 3        # 2 GiB resident across the round
DEFAULT_SKETCH_BATCH_BYTES = 1024 ** 3      # 1 GiB transient inside phase 2


class RoundVectorStore:
    """Holds per-client arrays for one round; call `close()` when done.

    `budget_bytes` is the ceiling on RAM used by arrays held here.
    Arrays are kept resident first-come-first-served until it is
    reached, then spilled. `budget_bytes=0` forces every array to
    disk, which is what the tests use to exercise the spill path.
    """

    def __init__(self, budget_bytes: int = DEFAULT_BUDGET_BYTES, dirpath: str | None = None):
        self._budget = max(0, int(budget_bytes))
        self._resident: dict = {}
        self._spilled: dict = {}
        self._resident_bytes = 0
        self._peak_resident_bytes = 0
        self._dir = dirpath
        self._owns_dir = dirpath is None
        self._spills = 0
        self._bytes_written = 0
        self._bytes_read = 0
        self._closed = False

    # -- internals ---------------------------------------------------

    def _ensure_dir(self) -> str:
        if self._dir is None:
            self._dir = tempfile.mkdtemp(prefix="janus-fl-round-")
        os.makedirs(self._dir, exist_ok=True)
        return self._dir

    @staticmethod
    def _filename(key) -> str:
        if isinstance(key, tuple):
            return "_".join(str(p) for p in key) + ".npy"
        return f"{key}.npy"

    def _check_open(self):
        if self._closed:
            raise RuntimeError("RoundVectorStore is closed; it holds one round only")

    # -- api ---------------------------------------------------------

    def put(self, key, arr: np.ndarray) -> None:
        self._check_open()
        if key in self._resident or key in self._spilled:
            raise KeyError(f"{key!r} already stored; each client writes each array once per round")
        arr = np.asarray(arr)
        if self._resident_bytes + arr.nbytes <= self._budget:
            arr.flags.writeable = False
            self._resident[key] = arr
            self._resident_bytes += arr.nbytes
            self._peak_resident_bytes = max(self._peak_resident_bytes, self._resident_bytes)
        else:
            path = os.path.join(self._ensure_dir(), self._filename(key))
            # Let an OSError (disk full) propagate: a silently dropped
            # vector would corrupt the aggregate with no error, which is
            # exactly the failure mode F1 was.
            np.save(path, arr, allow_pickle=False)
            self._spilled[key] = path
            self._spills += 1
            self._bytes_written += arr.nbytes

    def get(self, key) -> np.ndarray:
        """Returns the array for `key`, read-only. Spilled keys are
        re-read on every call -- callers stream, they do not hold."""
        self._check_open()
        if key in self._resident:
            return self._resident[key]
        path = self._spilled.get(key)
        if path is None:
            raise KeyError(f"{key!r} not in this round's store")
        arr = np.load(path, allow_pickle=False)
        arr.flags.writeable = False
        self._bytes_read += arr.nbytes
        return arr

    def drop(self, key) -> None:
        """Releases one array early. Safe to call for a key never stored."""
        arr = self._resident.pop(key, None)
        if arr is not None:
            self._resident_bytes -= arr.nbytes
        path = self._spilled.pop(key, None)
        if path is not None and os.path.exists(path):
            os.remove(path)

    def close(self) -> None:
        """Drops everything and removes the spill directory. Idempotent;
        call it from a `finally` so an aborted round does not leave
        gigabytes behind across a 45-round sweep."""
        if self._closed:
            return
        self._resident.clear()
        self._resident_bytes = 0
        if self._dir is not None and os.path.isdir(self._dir):
            if self._owns_dir:
                shutil.rmtree(self._dir, ignore_errors=True)
            else:
                for path in self._spilled.values():
                    if os.path.exists(path):
                        os.remove(path)
        self._spilled.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def stats(self) -> dict:
        """What actually happened -- reported per round so the memory
        claim is measured rather than asserted."""
        return {
            "budget_bytes": self._budget,
            "peak_resident_bytes": self._peak_resident_bytes,
            "resident_arrays": len(self._resident),
            "spilled_arrays": len(self._spilled),
            "spills": self._spills,
            "bytes_written": self._bytes_written,
            "bytes_read": self._bytes_read,
            "dir": self._dir,
        }


def suggested_batch_size(model_dim: int, budget_bytes: int, bytes_per_element: int = 12) -> int:
    """How many clients phase 2 can sketch together inside `budget_bytes`.

    Batching the sketch across clients is FIX.md F4's saving (the sign
    chunks are identical for every client in a round, so generating
    them once per batch rather than once per client is most of the
    win). Batching across ALL n clients would reintroduce exactly the
    O(n * model_dim) residency F5 removes, so the batch is sized from a
    byte budget instead of from n: the result is independent of client
    count, and batch size 8 already captures 7/8 of the maximum saving.

    `bytes_per_element` is the transient cost per client per element in
    `sketch_phase1_batch`: 4 for its float32 row in the stack, plus 8
    for the int64 vector being converted, which is freed before the
    next client is read.
    """
    per_client = max(1, model_dim * bytes_per_element)
    return max(1, min(16, int(budget_bytes // per_client)))
