"""Tests for crypto/sketch.py's TorchSketchProjector -- the GPU/torch
reimplementation STATUS.md TODO #3 called for, since SketchProjector
(the NumPy reference) is correctness-only and extrapolates to ~16
min/client/round at ResNet-50's real d~23.5M (see sketch.py's own
PERFORMANCE NOTE).

*** NOT RUNNABLE IN THIS SANDBOX -- torch isn't installed here (same
constraint as every other torch-dependent module in this repo, see
STATUS.md). This file is WRITTEN, NOT RUN: reviewed carefully against
torch's documented API and against SketchProjector's own already-
tested semantics, but not executed. Run on Colab before trusting it.

The module-level import below is guarded so this file can still be
collected/read in an environment without torch; run it with
`python3 tests/test_sketch_gpu.py` on Colab (or wherever torch is
installed) to actually execute these.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

from crypto.sketch import SketchProjector


def _skip_if_no_torch():
    if not _HAVE_TORCH:
        print("SKIPPED (torch not installed in this environment)")
        return True
    return False


def test_torch_sketch_is_deterministic_given_seed():
    if _skip_if_no_torch():
        return
    from crypto.sketch import TorchSketchProjector
    rng = np.random.default_rng(1)
    d, m = 100_000, 4096
    q = rng.normal(0, 1, size=d).astype(np.float64)
    seed = b"deterministic-seed"

    p1 = TorchSketchProjector(seed, m, d)
    p2 = TorchSketchProjector(seed, m, d)
    y1 = p1.project(q)
    y2 = p2.project(q)
    np.testing.assert_allclose(y1, y2, rtol=0, atol=0)


def test_torch_sketch_different_seeds_give_different_projections():
    if _skip_if_no_torch():
        return
    from crypto.sketch import TorchSketchProjector
    rng = np.random.default_rng(2)
    d, m = 100_000, 4096
    q = rng.normal(0, 1, size=d).astype(np.float64)

    p1 = TorchSketchProjector(b"seed-A", m, d)
    p2 = TorchSketchProjector(b"seed-B", m, d)
    y1, y2 = p1.project(q), p2.project(q)
    assert not np.allclose(y1, y2)


def test_torch_sketch_preserves_norm_within_jl_epsilon():
    if _skip_if_no_torch():
        return
    from crypto.sketch import TorchSketchProjector
    rng = np.random.default_rng(0)
    d, m = 20_000, 2048
    q = rng.normal(0, 1, size=d).astype(np.float64)

    proj = TorchSketchProjector(b"janus-fl-test-seed-0001", m, d)
    y = proj.project(q)
    ratio = float(np.linalg.norm(y) / np.linalg.norm(q))
    # Same generous JL slack as tests/test_sketch.py's equivalent
    # NumPy-reference test -- same algorithm, same guarantee, just a
    # different (torch) RNG stream so not expected to match the NumPy
    # reference bit-for-bit or even give the exact same ratio.
    assert 0.7 < ratio < 1.3, f"distortion ratio {ratio} far outside expected JL range"


def test_torch_and_numpy_projectors_agree_in_expectation():
    """NOT a bit-exact match (different RNG streams by design -- see
    TorchSketchProjector's own docstring on why that's fine here).
    This instead checks the two implementations behave like the SAME
    algorithm statistically: both should land within the JL epsilon
    band for the same input, confirming the torch rewrite didn't
    silently change the sketch's actual math (e.g. a wrong density,
    a missing scale factor, a sign-threshold bug).
    """
    if _skip_if_no_torch():
        return
    from crypto.sketch import TorchSketchProjector
    rng = np.random.default_rng(7)
    d, m = 20_000, 2048
    q = rng.normal(0, 1, size=d).astype(np.float64)
    seed = b"cross-check-seed"

    numpy_ratio = float(np.linalg.norm(SketchProjector(seed, m, d).project(q)) / np.linalg.norm(q))
    torch_ratio = float(np.linalg.norm(TorchSketchProjector(seed, m, d).project(q)) / np.linalg.norm(q))
    print(f"  numpy ratio={numpy_ratio:.4f}  torch ratio={torch_ratio:.4f}")
    assert 0.7 < numpy_ratio < 1.3
    assert 0.7 < torch_ratio < 1.3


def test_torch_sketch_timing_at_real_scale():
    """Not a hard assertion (GPU availability/speed varies) -- prints
    timing at ResNet-50's real d~23.5M so this can be directly
    compared against sketch.py's documented NumPy baseline (~16
    min/client/round extrapolated) once run on Colab. The whole point
    of TODO #3 was to make this number land in SECONDS, not minutes.
    """
    if _skip_if_no_torch():
        return
    from crypto.sketch import TorchSketchProjector
    import time
    rng = np.random.default_rng(3)
    m = 8192
    for d in (500_000, 5_000_000, 23_500_000):
        q = rng.normal(0, 1, size=d).astype(np.float64)
        proj = TorchSketchProjector(b"timing-seed", m, d)
        t0 = time.time()
        proj.project(q)
        dt = time.time() - t0
        device = proj.device
        print(f"  d={d:>11,}  project() wall time = {dt:.2f}s  (device={device})")


if __name__ == "__main__":
    test_torch_sketch_is_deterministic_given_seed()
    test_torch_sketch_different_seeds_give_different_projections()
    test_torch_sketch_preserves_norm_within_jl_epsilon()
    test_torch_and_numpy_projectors_agree_in_expectation()
    test_torch_sketch_timing_at_real_scale()
    print("ALL test_sketch_gpu.py TESTS PASSED (or skipped if torch unavailable)")
