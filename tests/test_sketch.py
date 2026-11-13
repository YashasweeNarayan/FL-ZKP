import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from crypto.sketch import SketchProjector, empirical_jl_distortion


def test_sketch_preserves_norm_within_jl_epsilon_on_synthetic_vector():
    rng = np.random.default_rng(0)
    # CORRECTNESS test only, small d -- see sketch.py's PERFORMANCE NOTE.
    # The real d=23.5M (ResNet-50) case requires a GPU/torch
    # implementation; ROADMAP.md Stage 1 Gate 1 re-runs the correctness
    # check (NOT this CPU reference) at real scale on Colab.
    d = 20_000
    m = 2048
    q = rng.normal(0, 1, size=d).astype(np.float64)

    seed = b"janus-fl-test-seed-0001"
    ratio = empirical_jl_distortion(q, seed, m)
    # JL guarantee for m=8192 gives epsilon ~ 0.05-0.08 for this d with
    # reasonable probability; allow generous slack since this is one draw
    assert 0.7 < ratio < 1.3, f"distortion ratio {ratio} far outside expected JL range"


def test_sketch_is_deterministic_given_seed():
    rng = np.random.default_rng(1)
    d, m = 100_000, 4096
    q = rng.normal(0, 1, size=d)
    seed = b"deterministic-seed"

    p1 = SketchProjector(seed, m, d)
    p2 = SketchProjector(seed, m, d)
    y1 = p1.project(q)
    y2 = p2.project(q)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_give_different_projections():
    rng = np.random.default_rng(2)
    d, m = 100_000, 4096
    q = rng.normal(0, 1, size=d)

    p1 = SketchProjector(b"seed-A", m, d)
    p2 = SketchProjector(b"seed-B", m, d)
    y1, y2 = p1.project(q), p2.project(q)
    assert not np.allclose(y1, y2)


def test_project_batch_matches_project_called_once_per_vector():
    """FIX.md F4: project_batch must give exactly the same output as
    calling project() once per vector under the same seed -- it is
    only a dedup of the sign-chunk generation, not a different
    projection."""
    rng = np.random.default_rng(4)
    d, m = 60_000, 2048
    seed = b"batch-seed"
    qs = [rng.normal(0, 1, size=d) for _ in range(5)]

    individually = [SketchProjector(seed, m, d).project(q) for q in qs]
    batched = SketchProjector(seed, m, d).project_batch(qs)

    assert len(batched) == 5
    for a, b in zip(individually, batched):
        # Same signs, same products, summed in a different order (BLAS
        # reduces (n,width)@(width,m) differently than (m,width)@(width,)) --
        # float32 non-associativity, not a mismatch in which values were
        # multiplied. Loose tolerance reflects that, not slack in the check.
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)


def test_project_batch_empty_and_single():
    d, m = 10_000, 512
    proj = SketchProjector(b"s", m, d)
    assert proj.project_batch([]) == []
    q = np.random.default_rng(5).normal(0, 1, size=d)
    [only] = proj.project_batch([q])
    np.testing.assert_allclose(only, proj.project(q), rtol=1e-6)


def test_project_batch_rejects_wrong_dim():
    proj = SketchProjector(b"s", 512, 10_000)
    bad = np.zeros(999)
    try:
        proj.project_batch([bad])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cost_independent_of_d_in_wallclock_terms():
    """Not a hard assertion (wall-clock is noisy) -- prints timing at
    small, fast sizes just to show cost scales with d on THIS CPU
    reference (it does -- O(m*d) -- see sketch.py's PERFORMANCE NOTE).
    The actual d-independence claim (N4) is about PROVER cost after
    sketching (the range proof is over m=8192 regardless of d); the
    sketch step itself is O(m*d) here and must move to GPU/torch for
    real d. Real timing happens on Colab (ROADMAP.md Stage 6).
    """
    import time
    rng = np.random.default_rng(3)
    m = 1024
    for d in (5_000, 20_000, 80_000):
        q = rng.normal(0, 1, size=d)
        proj = SketchProjector(b"timing-seed", m, d)
        t0 = time.time()
        proj.project(q)
        dt = time.time() - t0
        print(f"  d={d:>9,}  project() wall time = {dt*1000:.1f} ms  (CPU reference, not GPU numbers)")


if __name__ == "__main__":
    test_sketch_preserves_norm_within_jl_epsilon_on_synthetic_vector()
    test_sketch_is_deterministic_given_seed()
    test_different_seeds_give_different_projections()
    test_project_batch_matches_project_called_once_per_vector()
    test_project_batch_empty_and_single()
    test_project_batch_rejects_wrong_dim()
    test_cost_independent_of_d_in_wallclock_terms()
    print("ALL sketch.py TESTS PASSED")
