"""Johnson-Lindenstrauss sketching for the scaled norm proof (ARCHITECTURE.md §9).

PURE-MATH MODULE — verified in-sandbox, see tests/test_sketch.py.

The whole point of this module: proving ||q||_2 <= B over a sketch
Pi(q) in R^m is cheap for a Bulletproofs-style range proof (m ~ 8192)
where proving it over q in R^d directly is not (d ~ 23.5M for ResNet-50).
This file only does the numpy-side projection and the empirical
distortion check; the actual range *proof* over the sketch is
EC-dependent and lives in rangeproof.py (Colab-side).
"""
from __future__ import annotations

import numpy as np


def seed_commitment(seed: bytes) -> bytes:
    """h_sigma = H(sigma_t): the server commits to the projection seed
    BEFORE clients reveal their update commitments, so the sketch matrix
    is chosen independently of any client's q_k (ARCHITECTURE.md §9's
    soundness lemma needs exactly this ordering).
    """
    import hashlib
    return hashlib.sha256(seed).digest()


def make_sketch_matrix(seed: bytes, m: int, d: int, density: float = 1.0 / 3.0) -> "ScipyOrDenseSketch":
    """Sparse Achlioptas-style +-1/sqrt(m) projection, seeded deterministically
    so every party (prover, verifier, auditor) derives the identical matrix
    from the public seed.

    For d in the millions, a dense (m, d) float matrix is impractical to
    materialize (23.5M * 8192 floats is ~770 GB). We never materialize it:
    SketchProjector below generates columns on the fly from a PRG keyed on
    `seed`, so memory is O(m) regardless of d.
    """
    return SketchProjector(seed, m, d, density)


class SketchProjector:
    """Streaming sparse JL projector: Pi(q)_i = sum_j s_{ij} * q_j / sqrt(m*density),
    where s_{ij} in {-1, 0, +1} drawn with P(0) = 1 - density.

    Applying Pi to q costs O(m*d) random draws total and O(m) memory per
    chunk (no (m,d) matrix is ever held at once). Implemented as a
    chunked, seeded-PRG pass over q so it also works when q doesn't fit
    comfortably with a naive matmul.

    *** PERFORMANCE NOTE, found by actually running this in-sandbox ***
    This NumPy/CPU reference is for CORRECTNESS testing only (see
    tests/test_sketch.py), at small d. Timed here: d=500,000, m=8192
    takes ~20s on CPU. At ResNet-50's real d=23.5M that extrapolates to
    ~16 minutes PER CLIENT PER ROUND just for sketching -- before any
    proving. That is not viable, on CPU or even naively on GPU with a
    materialized matrix.
    The real client.py (Colab/GPU) implementation MUST do this with a
    torch tensor on GPU using torch.Generator for the seeded signs and a
    batched matmul (or, better, scipy.sparse / torch.sparse for the
    true-sparse case), which is 2-3 orders of magnitude faster than
    this. Flagged explicitly in STATUS.md -- do not port this class
    verbatim into the GPU training loop.
    """

    def __init__(self, seed: bytes, m: int, d: int, density: float = 1.0 / 3.0):
        self.seed = seed
        self.m = m
        self.d = d
        self.density = density
        self.scale = 1.0 / np.sqrt(m * density)

    def project(self, q: np.ndarray, chunk: int = 4_000) -> np.ndarray:
        """Chunked so an (m, chunk) sign matrix (int8) stays small even
        for m=8192 — e.g. chunk=4000 is 8192*4000 = 32.8 MB, vs. the
        12+ GB a naive (m, d) float64 materialization would need for
        ResNet-50's d=23.5M. This is the actual memory fix for the
        "never materialize (m, d)" promise in the module docstring —
        caught by running this test suite in-sandbox before it ever
        reached Colab or a GPU run.
        """
        if q.shape[0] != self.d:
            raise ValueError(f"expected d={self.d}, got {q.shape[0]}")
        out = np.zeros(self.m, dtype=np.float64)
        q32 = q.astype(np.float32, copy=False)
        for start in range(0, self.d, chunk):
            end = min(start + chunk, self.d)
            width = end - start
            sub_seed = int.from_bytes(
                seed_commitment(self.seed + start.to_bytes(8, "little"))[:8], "little"
            )
            sub_rng = np.random.default_rng(sub_seed)
            # Fast path: rng.choice(..., p=[...]) is pathologically slow
            # for large arrays (internal cumulative-sum/searchsorted per
            # draw). Threshold-on-uniform is the standard vectorized
            # equivalent and is ~100x faster at this size -- caught here
            # by actually timing it, not by assuming numpy is fast.
            u = sub_rng.random(size=(self.m, width), dtype=np.float32)
            signs = np.zeros((self.m, width), dtype=np.float32)
            signs[u < self.density / 2] = -1.0
            signs[u > 1.0 - self.density / 2] = 1.0
            out += signs @ q32[start:end]
        return out * self.scale


def empirical_jl_distortion(q: np.ndarray, seed: bytes, m: int, trials: int = 1) -> float:
    """||Pi(q)||_2 / ||q||_2 — should be close to 1.0 within the JL
    guarantee's epsilon for a good m. This is exactly what
    ROADMAP.md Stage 1's sketch.py test checks against real gradient
    vectors on Colab; here we only verify the MACHINERY is correct on a
    synthetic vector (ground truth norm is known exactly).
    """
    d = q.shape[0]
    proj = SketchProjector(seed, m, d)
    y = proj.project(q)
    return float(np.linalg.norm(y) / np.linalg.norm(q))


class TorchSketchProjector:
    """GPU/torch reimplementation of `SketchProjector`, per this
    module's own long-standing instruction not to port the NumPy class
    verbatim into the training loop. SAME algorithm (streaming
    Achlioptas-style +-1/0/+1 sketch, seeded per-chunk exactly like the
    NumPy reference's `seed_commitment(seed + start_bytes)` scheme, so
    the construction is equally auditable), reimplemented with torch
    tensors and `torch.Generator` so chunks run as batched GPU matmuls
    instead of a NumPy `rng.random` + Python-level loop.

    NOT bit-identical to `SketchProjector` for the same seed (torch's
    RNG stream differs from NumPy's) -- this does NOT matter for
    correctness here: nothing in this repo recomputes a client's
    sketch independently to cross-check it bit-for-bit. `fl/server.py`
    trusts `phase2.sketch` as sent (verified only via the V3 range
    proof tying it to a specific committed norm, fixed earlier this
    session) and never re-derives the projection itself; the sketch
    is purely a client-side computation whose output is consumed as
    data. What DOES matter, and what this class preserves: (1)
    determinism given the same seed (so a client's own repeated calls,
    or any future auditing need, are reproducible), and (2) a valid JL
    embedding that approximately preserves the L2 norm, which V3's
    norm bound actually depends on.

    *** WRITTEN, NOT RUN -- see STATUS.md ***. torch isn't installed
    in the sandbox this was written in (same constraint as every other
    torch-dependent module in this repo), so this has been reviewed
    carefully against the NumPy reference's semantics and against
    documented torch API behavior, but not executed. Validate on Colab
    (tests/test_sketch_gpu.py) before trusting it for a real run,
    exactly like every other 🟡 module in STATUS.md.
    """

    def __init__(self, seed: bytes, m: int, d: int, density: float = 1.0 / 3.0, device: str = None):
        import torch
        self.torch = torch
        self.seed = seed
        self.m = m
        self.d = d
        self.density = density
        self.scale = 1.0 / np.sqrt(m * density)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Computed ONCE here, from the device's fixed total memory, and
        # reused for every project() call on this instance -- see
        # _compute_default_chunk's docstring for why this must NOT be
        # recomputed per-call from currently-free memory.
        self._default_chunk_size = self._compute_default_chunk()

    def _compute_default_chunk(self) -> int:
        """Pick a chunk size from the device's fixed TOTAL memory, not
        currently-free memory. This distinction matters for correctness,
        not just performance: chunk size determines where the (m, d)
        sign matrix gets split, and each chunk's RNG is seeded from its
        own start offset (`seed_commitment(self.seed + start_bytes)`).
        Two instances with the *same* seed must produce the *same*
        chunk boundaries to produce the *same* output -- that's exactly
        what test_torch_sketch_is_deterministic_given_seed checks, and
        it failed when this was computed from `torch.cuda.mem_get_info()`
        (currently-free memory drifts between calls -- allocator
        caching, fragmentation -- so two instances built moments apart
        picked different chunk sizes and produced completely different
        sign matrices from the identical seed; not a performance
        regression, an actual reproducibility bug). `total_memory` is a
        static property of the GPU itself, so it's the same every time
        for every instance on this device, restoring determinism.

        A fixed 200_000 on CUDA regardless of device (the original
        default) assumed ~16GB of headroom for the transient (m, chunk)
        `u` and `signs` float32 tensors (~2 * m * chunk * 4 bytes
        together); a real T4 only exposes ~14.6GB usable, and confirmed
        OOM'd. Sizing from 25% of total_memory instead adapts to the
        actual GPU without depending on transient allocator state.
        CPU has no `mem_get_info`/`get_device_properties`, so it keeps a
        fixed 20_000 (~1.3GB transient, safe on typical RAM).
        """
        torch = self.torch
        if self.device != "cuda":
            return 20_000
        device_idx = torch.cuda.current_device()
        total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        safe_bytes = total_bytes * 0.25
        chunk = int(safe_bytes / (2 * self.m * 4))
        return max(2_000, min(200_000, chunk))

    def project(self, q, chunk: int = None):
        """Same chunking discipline as `SketchProjector.project` (never
        materialize the full (m, d) sign matrix at once), but on CUDA a
        larger chunk than the NumPy reference is generally affordable
        and faster (fewer, larger kernel launches) -- GPU memory and
        matmul throughput favor it over the NumPy reference's
        chunk=4_000, which was sized for CPU cache/memory, not GPU
        occupancy.

        Leaving `chunk=None` (the default) uses `self._default_chunk_size`,
        computed once at construction time from the device's fixed total
        memory (see `_compute_default_chunk`'s docstring for why it must
        be fixed per-instance rather than recomputed from live free
        memory -- that was an actual determinism bug, not just a sizing
        one). On CPU it's a fixed 20_000 (~1.3GB transient memory, safe
        on typical RAM -- the earlier fixed 200_000 default meant ~13GB
        of transient CPU RAM per chunk, which thrashed/hung rather than
        erroring on a GPU-less runtime; that was the earlier apparent
        "hang" in test_torch_sketch_timing_at_real_scale, not an
        infinite loop).

        Even sizing from total device memory is a prediction, not a
        guarantee (other processes, fragmentation), so an actual OOM
        mid-call still triggers one retry at a quarter the chunk size,
        after freeing the cache. This retry path is explicitly a
        degraded-conditions fallback and does NOT preserve the
        determinism guarantee above (a run that hits it will not match
        a run that didn't) -- acceptable because it only triggers under
        genuine external memory pressure, which already makes bit-exact
        reproducibility impossible to promise.

        Accepts a NumPy array or a torch tensor for `q`; always
        returns a NumPy array, so callers (`fl/client.py`,
        `fl/server.py`'s `np.linalg.norm(phase2.sketch)`,
        `verifier/audit.py`) need no changes to consume either
        projector's output interchangeably.
        """
        torch = self.torch
        if chunk is None:
            chunk = self._default_chunk_size
        if q.shape[0] != self.d:
            raise ValueError(f"expected d={self.d}, got {q.shape[0]}")
        if isinstance(q, np.ndarray):
            q_t = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        else:
            q_t = q.to(device=self.device, dtype=torch.float32)

        try:
            return self._project_chunked(q_t, chunk)
        except torch.cuda.OutOfMemoryError:
            if chunk <= 2_000:
                raise
            torch.cuda.empty_cache()
            return self._project_chunked(q_t, max(2_000, chunk // 4))

    def _project_chunked(self, q_t, chunk: int):
        torch = self.torch
        out = torch.zeros(self.m, dtype=torch.float32, device=self.device)
        for start in range(0, self.d, chunk):
            end = min(start + chunk, self.d)
            width = end - start
            sub_seed = int.from_bytes(
                seed_commitment(self.seed + start.to_bytes(8, "little"))[:8], "little"
            )
            # torch.Generator's seed is a signed 64-bit int internally;
            # masking to 63 bits keeps this valid across torch versions
            # without relying on unsigned-seed support.
            gen = torch.Generator(device=self.device)
            gen.manual_seed(sub_seed & ((1 << 63) - 1))
            u = torch.rand((self.m, width), generator=gen, device=self.device, dtype=torch.float32)
            signs = torch.zeros((self.m, width), dtype=torch.float32, device=self.device)
            signs[u < self.density / 2] = -1.0
            signs[u > 1.0 - self.density / 2] = 1.0
            out += signs @ q_t[start:end]
        return (out * self.scale).cpu().numpy()
