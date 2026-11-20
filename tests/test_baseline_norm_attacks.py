"""Unit tests for `attacks/baseline_norm_attacks.py`'s four craft
functions (2026-11-20) -- pure NumPy, no EC/crypto backend needed, so
these run unconditionally (no skip gating, unlike `test_round_end_to_end.py`
and friends).

Covers: shape preservation, sign-flip correctness, exact-norm
correctness after scaling (`craft_scale_to_threshold`/`craft_low_and_slow`
via the shared `_scale_to_norm` helper), the zero-real-delta fallback
path (not-VRF-sampled this round), and that the two scale-based attacks
produce different norms for the same inputs at their different default
ratios -- a direct check that `safety_margin` and `target_ratio` aren't
accidentally wired to the same constant.
"""
from __future__ import annotations

import numpy as np
import pytest

from attacks.baseline_norm_attacks import (
    _scale_to_norm,
    craft_gaussian_noise,
    craft_low_and_slow,
    craft_scale_to_threshold,
    craft_sign_flip,
)

MODEL_DIM = 16
B_T = 50.0


def _real_delta(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 10.0, size=MODEL_DIM)


# ---------------------------------------------------------------- sign-flip

def test_sign_flip_negates():
    d = _real_delta()
    flipped = craft_sign_flip(d)
    assert flipped.shape == d.shape
    np.testing.assert_allclose(flipped, -d)


def test_sign_flip_zero_delta_is_noop():
    d = np.zeros(MODEL_DIM)
    flipped = craft_sign_flip(d)
    np.testing.assert_allclose(flipped, d)  # -0 == 0


# ------------------------------------------------------------- gaussian noise

def test_gaussian_noise_shape_and_nonzero_perturbation():
    d = _real_delta()
    rng = np.random.default_rng(42)
    noisy = craft_gaussian_noise(d, sigma=1.0, rng=rng)
    assert noisy.shape == d.shape
    # With sigma=1.0 on a dim-16 vector, the perturbation is essentially
    # certain to be nonzero somewhere.
    assert not np.allclose(noisy, d)


def test_gaussian_noise_sigma_zero_is_noop():
    d = _real_delta()
    rng = np.random.default_rng(1)
    noisy = craft_gaussian_noise(d, sigma=0.0, rng=rng)
    np.testing.assert_allclose(noisy, d)


def test_gaussian_noise_default_rng_runs():
    # No explicit rng -- exercises the `if rng is None` fallback branch.
    d = _real_delta()
    noisy = craft_gaussian_noise(d, sigma=0.5)
    assert noisy.shape == d.shape


# ------------------------------------------------------------- scale_to_norm

def test_scale_to_norm_preserves_direction_and_hits_target():
    d = _real_delta()
    target = 7.5
    scaled = _scale_to_norm(d, target)
    assert scaled.shape == d.shape
    np.testing.assert_allclose(np.linalg.norm(scaled), target, rtol=1e-9)
    # Direction preserved: scaled is a positive multiple of d.
    ratio = scaled / d
    np.testing.assert_allclose(ratio, ratio[0] * np.ones_like(ratio), rtol=1e-9)
    assert ratio[0] > 0


def test_scale_to_norm_zero_delta_fallback_hits_target_norm():
    d = np.zeros(MODEL_DIM)
    target = 12.0
    rng = np.random.default_rng(7)
    scaled = _scale_to_norm(d, target, rng=rng)
    assert scaled.shape == d.shape
    np.testing.assert_allclose(np.linalg.norm(scaled), target, rtol=1e-9)
    # Fallback direction shouldn't just be all-zeros.
    assert not np.allclose(scaled, 0.0)


def test_scale_to_norm_zero_delta_default_rng_runs():
    d = np.zeros(MODEL_DIM)
    scaled = _scale_to_norm(d, 3.0)
    np.testing.assert_allclose(np.linalg.norm(scaled), 3.0, rtol=1e-9)


# ------------------------------------------------------- scale_to_threshold

def test_scale_to_threshold_hits_safety_margin_times_bt():
    d = _real_delta()
    scaled = craft_scale_to_threshold(d, B_T)
    np.testing.assert_allclose(np.linalg.norm(scaled), 0.95 * B_T, rtol=1e-9)


def test_scale_to_threshold_custom_safety_margin():
    d = _real_delta()
    scaled = craft_scale_to_threshold(d, B_T, safety_margin=0.5)
    np.testing.assert_allclose(np.linalg.norm(scaled), 0.5 * B_T, rtol=1e-9)


def test_scale_to_threshold_zero_delta_still_hits_target():
    d = np.zeros(MODEL_DIM)
    rng = np.random.default_rng(3)
    scaled = craft_scale_to_threshold(d, B_T, rng=rng)
    np.testing.assert_allclose(np.linalg.norm(scaled), 0.95 * B_T, rtol=1e-9)


# ----------------------------------------------------------- low_and_slow

def test_low_and_slow_hits_target_ratio_times_bt():
    d = _real_delta()
    scaled = craft_low_and_slow(d, B_T)
    np.testing.assert_allclose(np.linalg.norm(scaled), 0.3 * B_T, rtol=1e-9)


def test_low_and_slow_custom_ratio():
    d = _real_delta()
    scaled = craft_low_and_slow(d, B_T, target_ratio=0.1)
    np.testing.assert_allclose(np.linalg.norm(scaled), 0.1 * B_T, rtol=1e-9)


def test_low_and_slow_is_strictly_smaller_than_scale_to_threshold():
    # Guards against safety_margin/target_ratio accidentally sharing a
    # constant: at their respective defaults (0.95 vs 0.3), low_and_slow
    # must produce a strictly smaller norm for the same inputs.
    d = _real_delta()
    near_boundary = craft_scale_to_threshold(d, B_T)
    patient = craft_low_and_slow(d, B_T)
    assert np.linalg.norm(patient) < np.linalg.norm(near_boundary)


def test_low_and_slow_zero_delta_still_hits_target():
    d = np.zeros(MODEL_DIM)
    rng = np.random.default_rng(9)
    scaled = craft_low_and_slow(d, B_T, rng=rng)
    np.testing.assert_allclose(np.linalg.norm(scaled), 0.3 * B_T, rtol=1e-9)
