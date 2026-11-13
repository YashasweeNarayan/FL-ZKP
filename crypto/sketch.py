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


# ---------------------------------------------------------------------------
# Exact INTEGER sketch (FIX.md F7b).
#
# The float sketch above is enough to *estimate* a norm. The F7 norm
# proof needs more: the client commits to its sketch y = S q, and the
# verifier later checks that the committed sketches add up to the sketch
# of the aggregate, sum_k y_k == S (sum_k q_k). That is an equation, not
# an estimate, so S q has to be computed exactly, over the integers,
# with the unscaled sign matrix S in {-1, 0, +1}^(m x d), and every
# party has to derive the SAME S.
#
# Two consequences for how S is generated:
#   * the columns are generated in fixed blocks of INT_SKETCH_BLOCK,
#     each seeded from its own start offset. The block width is a
#     constant, NOT derived from device memory as the float projector's
#     chunk is: there, two GPUs with different memory split the matrix
#     at different places and produce different matrices from one seed.
#   * nothing may silently change the block layout (the float path's
#     out-of-memory retry at a quarter chunk does exactly that).
#
# The norm it estimates: E ||S q||^2 = m * density * ||q||^2, so
# ||q|| ~ sqrt(||S q||^2 / (m * density)) -- `int_sketch_norm` below.
# ---------------------------------------------------------------------------

INT_SKETCH_BLOCK = 16384   # columns per independently seeded block. FIXED: part of the definition of S.
_LIMB_BITS = 8             # torch path: 16384 * 255 < 2^24, so float32 sums of one block are exact


def _int_block_seed(seed: bytes, start: int) -> int:
    import hashlib
    h = hashlib.sha256(b"janus-int-sketch:" + seed + start.to_bytes(8, "little")).digest()
    return int.from_bytes(h[:8], "little")


def int_sketch_norm(norm_sq: int, m: int, density: float = 1.0 / 3.0) -> float:
    """||q||_2 estimate from v = ||S q||_2^2 (an exact integer)."""
    return float(np.sqrt(norm_sq / (m * density)))


def int_sketch_norm_sq_bound(B: float, m: int, density: float = 1.0 / 3.0) -> int:
    """Largest integer v with int_sketch_norm(v) <= B, so the server's
    acceptance test is an exact integer comparison on the public v."""
    return int(np.floor(B * B * m * density))


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

    def _int_signs(self, start: int, width: int) -> np.ndarray:
        """Block `start` of the integer sign matrix, shape (m, width),
        float64 entries in {-1, 0, +1}."""
        rng = np.random.default_rng(_int_block_seed(self.seed, start))
        u = rng.random(size=(self.m, width), dtype=np.float32)
        signs = np.zeros((self.m, width), dtype=np.float64)
        signs[u < self.density / 2] = -1.0
        signs[u > 1.0 - self.density / 2] = 1.0
        return signs

    def project_int_batch(self, qs) -> np.ndarray:
        """EXACT integer sketch (FIX.md F7b): rows of `qs` (int64, signed
        and centered -- NOT reduced mod P_MASK) -> S q as int64, shape
        (n, m). Uses the unscaled sign matrix and fixed blocks, see the
        note above INT_SKETCH_BLOCK.

        float64 BLAS is used for speed and is exact here: one block's
        partial sums are bounded by INT_SKETCH_BLOCK * max|q|, which is
        checked to stay below 2^53.
        """
        qs = np.asarray(qs)
        if qs.ndim == 1:
            qs = qs[None, :]
        if qs.shape[1] != self.d:
            raise ValueError(f"expected d={self.d}, got {qs.shape[1]}")
        if not np.issubdtype(qs.dtype, np.integer):
            raise TypeError("project_int_batch needs an integer array (the centered quantized update)")
        max_abs = int(np.max(np.abs(qs))) if qs.size else 0
        if INT_SKETCH_BLOCK * max_abs >= 2 ** 53:
            raise OverflowError(
                f"max |q| = {max_abs} is too large for an exact float64 block sum; "
                f"an update this large is far outside any norm bound"
            )
        out = np.zeros((qs.shape[0], self.m), dtype=np.int64)
        for start in range(0, self.d, INT_SKETCH_BLOCK):
            end = min(start + INT_SKETCH_BLOCK, self.d)
            signs = self._int_signs(start, end - start)
            part = qs[:, start:end].astype(np.float64) @ signs.T
            out += np.rint(part).astype(np.int64)
        return out

    def project_int(self, q) -> np.ndarray:
        return self.project_int_batch(np.asarray(q)[None, :])[0]

    def project_batch(self, qs: list, chunk: int = 4_000) -> list:
        """Projects several vectors under this SAME seed in one pass
        over the sign chunks, instead of calling `project()` once per
        vector (FIX.md F4: that would regenerate every chunk's signs
        once per vector even though they are identical). Verified
        against calling `project()` individually in
        tests/test_sketch.py.
        """
        n = len(qs)
        if n == 0:
            return []
        # A caller that already built an (n, d) float32 block -- as
        # fl/client.py::sketch_phase1_batch does, to bound peak memory
        # (FIX.md F5) -- hands it straight through, instead of paying
        # for a second full copy here.
        if isinstance(qs, np.ndarray) and qs.ndim == 2 and qs.dtype == np.float32:
            if qs.shape[1] != self.d:
                raise ValueError(f"expected d={self.d}, got {qs.shape[1]}")
            qs32 = qs
        else:
            for q in qs:
                if q.shape[0] != self.d:
                    raise ValueError(f"expected d={self.d}, got {q.shape[0]}")
            qs32 = np.stack([q.astype(np.float32, copy=False) for q in qs], axis=0)  # (n, d)
        out = np.zeros((n, self.m), dtype=np.float64)
        for start in range(0, self.d, chunk):
            end = min(start + chunk, self.d)
            width = end - start
            sub_seed = int.from_bytes(
                seed_commitment(self.seed + start.to_bytes(8, "little"))[:8], "little"
            )
            sub_rng = np.random.default_rng(sub_seed)
            u = sub_rng.random(size=(self.m, width), dtype=np.float32)
            signs = np.zeros((self.m, width), dtype=np.float32)
            signs[u < self.density / 2] = -1.0
            signs[u > 1.0 - self.density / 2] = 1.0
            out += qs32[:, start:end] @ signs.T  # (n, width) @ (width, m) -> (n, m)
        out *= self.scale
        return [out[i] for i in range(n)]


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

    def project_batch(self, qs: list, chunk: int = None) -> list:
        """Torch counterpart of `SketchProjector.project_batch`
        (FIX.md F4): generates each chunk's sign matrix once and
        applies it to every vector in `qs` in the same pass, instead
        of once per vector. Same chunking, seeding and OOM-retry
        discipline as `project()`; returns one NumPy array per input,
        in input order.

        *** WRITTEN, NOT RUN -- same caveat as the rest of this class
        (see class docstring and STATUS.md): torch is not installed in
        the sandbox this was written in. Reviewed carefully against
        `project()`'s already-reviewed-but-unrun implementation; the
        NumPy counterpart (same algorithm, same chunk/seed scheme) IS
        sandbox-tested against calling `project()` per vector, in
        tests/test_sketch.py. Validate this method on Colab before
        trusting it, exactly like `project()` itself.
        """
        torch = self.torch
        n = len(qs)
        if n == 0:
            return []
        if chunk is None:
            chunk = self._default_chunk_size
        # Pre-stacked (n, d) block from sketch_phase1_batch: move it
        # across in one go rather than stacking a second copy.
        if isinstance(qs, np.ndarray) and qs.ndim == 2:
            if qs.shape[1] != self.d:
                raise ValueError(f"expected d={self.d}, got {qs.shape[1]}")
            qs_t = torch.as_tensor(qs, dtype=torch.float32, device=self.device)
        else:
            for q in qs:
                if q.shape[0] != self.d:
                    raise ValueError(f"expected d={self.d}, got {q.shape[0]}")
            qs_t = torch.stack([
                torch.as_tensor(q, dtype=torch.float32, device=self.device) if isinstance(q, np.ndarray)
                else q.to(device=self.device, dtype=torch.float32)
                for q in qs
            ], dim=0)  # (n, d)
        try:
            out = self._project_chunked_batch(qs_t, chunk)
        except torch.cuda.OutOfMemoryError:
            if chunk <= 2_000:
                raise
            torch.cuda.empty_cache()
            out = self._project_chunked_batch(qs_t, max(2_000, chunk // 4))
        return [out[i].cpu().numpy() for i in range(n)]

    def _project_chunked_batch(self, qs_t, chunk: int):
        torch = self.torch
        n = qs_t.shape[0]
        out = torch.zeros((n, self.m), dtype=torch.float32, device=self.device)
        for start in range(0, self.d, chunk):
            end = min(start + chunk, self.d)
            width = end - start
            sub_seed = int.from_bytes(
                seed_commitment(self.seed + start.to_bytes(8, "little"))[:8], "little"
            )
            gen = torch.Generator(device=self.device)
            gen.manual_seed(sub_seed & ((1 << 63) - 1))
            u = torch.rand((self.m, width), generator=gen, device=self.device, dtype=torch.float32)
            signs = torch.zeros((self.m, width), dtype=torch.float32, device=self.device)
            signs[u < self.density / 2] = -1.0
            signs[u > 1.0 - self.density / 2] = 1.0
            out += qs_t[:, start:end] @ signs.T  # (n, width) @ (width, m) -> (n, m)
        return out * self.scale

    # -- exact integer sketch (FIX.md F7b) ------------------------------

    def project_int_batch(self, qs) -> np.ndarray:
        """Torch counterpart of `SketchProjector.project_int_batch`:
        rows of `qs` (int64, signed, centered) -> S q exactly, as a
        NumPy int64 array of shape (n, m).

        GPU matmul is float32, which is exact only up to 2^24, and
        float64 is not an option (a T4 runs it at ~1/32 the speed). So
        each block of q is split into base-256 limbs: one block's limb
        sum is at most 16384 * 255 < 2^24, hence exact in float32; the
        limbs are recombined in int64. An update with |q| < 2^23 costs
        three matmuls per block instead of one.

        The block layout and seeding are fixed (INT_SKETCH_BLOCK), so
        this matrix does not depend on GPU memory and there is no
        out-of-memory retry that could change it. It is still generated
        by torch's RNG, so it is NOT the NumPy projector's matrix, and a
        CUDA generator does not reproduce a CPU one: every party to one
        round must use the same projector class on the same device type.
        That holds in this single-process simulation; a deployment with
        mixed hardware needs a portable generator (see FIX.md F7,
        "known limits").

        *** WRITTEN, NOT RUN in the sandbox (no torch). The NumPy method
        with the same block scheme is tested in tests/test_sketch.py;
        tests/test_sketch_gpu.py checks this one against exact integer
        arithmetic on Colab. ***
        """
        torch = self.torch
        qs = np.asarray(qs)
        if qs.ndim == 1:
            qs = qs[None, :]
        if qs.shape[1] != self.d:
            raise ValueError(f"expected d={self.d}, got {qs.shape[1]}")
        if not np.issubdtype(qs.dtype, np.integer):
            raise TypeError("project_int_batch needs an integer array (the centered quantized update)")
        n = qs.shape[0]
        max_abs = int(np.max(np.abs(qs))) if qs.size else 0
        # top limb is signed and must fit in [-128, 127]
        n_limbs = 1
        while max_abs >= (1 << (_LIMB_BITS * n_limbs - 1)):
            n_limbs += 1
        if _LIMB_BITS * n_limbs > 48:
            raise OverflowError(
                f"max |q| = {max_abs} needs {n_limbs} limbs; an update this large "
                f"is far outside any norm bound"
            )
        mask = (1 << _LIMB_BITS) - 1
        out = torch.zeros((n, self.m), dtype=torch.int64, device=self.device)
        for start in range(0, self.d, INT_SKETCH_BLOCK):
            end = min(start + INT_SKETCH_BLOCK, self.d)
            width = end - start
            gen = torch.Generator(device=self.device)
            gen.manual_seed(_int_block_seed(self.seed, start) & ((1 << 63) - 1))
            u = torch.rand((self.m, width), generator=gen, device=self.device, dtype=torch.float32)
            signs_t = torch.zeros((width, self.m), dtype=torch.float32, device=self.device)
            u = u.T
            signs_t[u < self.density / 2] = -1.0
            signs_t[u > 1.0 - self.density / 2] = 1.0
            del u
            qb = torch.as_tensor(np.ascontiguousarray(qs[:, start:end]), dtype=torch.int64, device=self.device)
            for limb_idx in range(n_limbs):
                shifted = qb >> (_LIMB_BITS * limb_idx)          # arithmetic shift: floor division
                limb = shifted if limb_idx == n_limbs - 1 else (shifted & mask)
                part = limb.to(torch.float32) @ signs_t           # (n, width) @ (width, m), exact
                out += torch.round(part).to(torch.int64) << (_LIMB_BITS * limb_idx)
            del qb, signs_t
        return out.cpu().numpy()

    def project_int(self, q) -> np.ndarray:
        return self.project_int_batch(np.asarray(q)[None, :])[0]
