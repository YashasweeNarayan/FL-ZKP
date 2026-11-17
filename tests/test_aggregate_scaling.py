"""FedAvg aggregate scaling under VRF self-sampling (FIX.md F20).

The bug these lock down: `fl/server.py` divided the dequantized
aggregate by `len(accepted)`, but under self-sampling only ~p*n of the
accepted submissions are real contributors -- the rest are dummies
submitting exactly zero. The numerator sums real deltas only, so the
denominator over-counted and every model update was shrunk by ~1/p
(≈3.4x at p=0.25 with ~43 accepted).

Pure arithmetic, no group backend needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from fl.aggregation import real_contributor_count


def test_defaults_to_pre_f20_behaviour_when_not_subsampling():
    """p=1.0 means no dummies, so every accepted client is real and the
    denominator must equal len(accepted) exactly -- the old behaviour,
    which was correct in that regime. Guards the default so existing
    callers that omit sampling_rate are unaffected."""
    for n in (8, 43, 50):
        assert real_contributor_count(n, 1.0) == n


def test_clean_round_at_quarter_sampling():
    # 50 accepted, none rejected, p=0.25 -> the expected sampled cohort.
    assert real_contributor_count(50, 0.25) == 12.5


def test_attack_round_scales_accepted_rather_than_subtracting_rejects():
    """7 attackers rejected out of 50. The naive
    `p*n_clients - n_rejected` gives 12.5-7 = 5.5, which over-corrects
    (attack deltas bypass the sampling gate, so only ~p of those 7
    displaced a genuinely-sampled honest client) and would roughly
    double the applied step. Scaling the accepted count gives 10.75."""
    assert real_contributor_count(43, 0.25) == pytest.approx(10.75)


def test_never_degenerate_under_heavy_rejection():
    """A round with more rejections than the expected cohort must not
    produce a zero or negative denominator -- `p*n_clients-n_rejected`
    goes negative here (observed live with 15 rejections at p=0.25)."""
    for n_accepted in (0, 1, 2, 35):
        d = real_contributor_count(n_accepted, 0.25)
        assert d >= 1.0


def test_denominator_is_smaller_than_accepted_count_when_subsampling():
    """The actual regression: at p<1 the denominator must be strictly
    below len(accepted), otherwise updates are being shrunk again."""
    n_accepted = 43
    assert real_contributor_count(n_accepted, 0.25) < n_accepted


def test_scaling_ratio_against_the_old_buggy_denominator():
    """Quantifies the fix: at the configuration that produced the 42%
    MNIST run, updates were ~3.4x too small."""
    n_accepted = 43
    old = n_accepted
    new = real_contributor_count(n_accepted, 0.25)
    assert old / new == pytest.approx(4.0)


def test_rejects_invalid_sampling_rate():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            real_contributor_count(43, bad)


def test_matches_config_sampling_rate():
    """The shipped config's p, so this test moves if the config does."""
    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "mvp_3day.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    p = cfg["protocol"]["sampling_rate"]
    n = cfg["protocol"]["n_clients"]
    # all 50 accepted, nobody rejected -> expected cohort is p*n
    assert real_contributor_count(n, p) == pytest.approx(p * n)
