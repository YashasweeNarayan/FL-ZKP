"""DP-disabled path (FIX.md F19, 2026-11-17).

The shipped configuration runs with the DP layer OFF (`dp.sigma: 0.0`),
with the privacy claim resting on the cryptographic layer instead. These
tests lock that down at both ends: the mechanism (`dp_noise_share`
returns exact zeros) and the wiring (the shipped config actually has it
off, and re-enabling it still works).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

# Imported from crypto.dp, not fl.client: fl.client pulls in
# crypto.pedersen, which does a hash-to-point at module import time and
# needs a group backend. dp_noise_share is pure numpy, so keeping it in
# crypto.dp lets these tests run in a network-less sandbox without
# switching the whole pytest process onto the slow pure-Python backend.
# fl.client imports this same function; see its call site in
# round_step_phase1.
from crypto.dp import dp_noise_share

DIM = 1024
P = 0.25
N = 50


def test_dp_disabled_gives_exact_zeros_for_sampled_client():
    """sigma_dp = 0 means no noise even for a VRF-SAMPLED client.

    This is the F19 case and is distinct from the F14 dummy-client case
    below -- here b_k is True and the client would normally be the one
    contributing noise.
    """
    nu = dp_noise_share(0.0, P, N, DIM, b_k=True, rng=np.random.default_rng(0))
    assert nu.shape == (DIM,)
    assert np.array_equal(nu, np.zeros(DIM)), "DP disabled must give EXACT zeros, not small noise"


def test_dp_disabled_gives_exact_zeros_for_dummy_client():
    nu = dp_noise_share(0.0, P, N, DIM, b_k=False, rng=np.random.default_rng(0))
    assert np.array_equal(nu, np.zeros(DIM))


def test_negative_sigma_also_treated_as_disabled():
    """Guard is `> 0`, so a nonsense negative sigma disables rather than
    reaching np.random.normal with a negative scale (which raises)."""
    nu = dp_noise_share(-1.0, P, N, DIM, b_k=True, rng=np.random.default_rng(0))
    assert np.array_equal(nu, np.zeros(DIM))


def test_dummy_client_gets_zeros_even_when_dp_enabled():
    """F14 regression: the b_k gate must survive the F19 refactor."""
    nu = dp_noise_share(2.5, P, N, DIM, b_k=False, rng=np.random.default_rng(0))
    assert np.array_equal(nu, np.zeros(DIM))


def test_dp_enabled_path_still_works_and_scales_correctly():
    """Re-enabling DP (the ablation path) must still produce noise at the
    F14-corrected per-client scale sigma_dp / sqrt(p*n)."""
    sigma_dp = 2.5
    expected = sigma_dp / np.sqrt(P * N)
    nu = dp_noise_share(sigma_dp, P, N, 200_000, b_k=True, rng=np.random.default_rng(0))
    assert not np.array_equal(nu, np.zeros(200_000))
    # empirical std within 2% of the target at this dimension
    assert abs(nu.std() - expected) / expected < 0.02, (nu.std(), expected)


def test_shipped_config_has_dp_disabled():
    """The config this project actually runs with. If DP is ever
    re-enabled deliberately, update this test in the same commit so the
    change is explicit rather than silent."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "mvp_3day.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["dp"]["sigma"] == 0.0, (
        "configs/mvp_3day.yaml is expected to ship with the DP layer disabled "
        "(FIX.md F19); got sigma=%r" % cfg["dp"]["sigma"]
    )
