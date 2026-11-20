"""Integration tests for the `_delta_transform`/`cheater_delta_transform`
hook (2026-11-20, added alongside `attacks/baseline_norm_attacks.py`) --
`fl/client.py::round_step_phase1`'s new `_delta_transform` parameter and
`fl/round_loop.py::run_training_loop`'s new `cheater_delta_transform`
parameter + its mutual-exclusivity assertion against `cheater_force_delta`.

Gated the same way as `test_round_end_to_end.py`: on whether ANY group
backend is live (real rbcl, or the test-only pure-Python backend via
JANUS_ALLOW_PURE_PYTHON_GROUP=1), not on the env var directly.

Torch is avoided entirely -- not by building submissions by hand like
`test_round_end_to_end.py` does, but by exercising only the two branches
that never reach torch-dependent code:
  (a) `round_step_phase1` directly, with `sampling_rate=0.0` so Step 0's
      VRF self-sampling deterministically yields `b_k=False` and Step 1
      takes the "contribute zero" branch instead of calling
      `local_train_delta` (which needs torch for the model) -- this
      still exercises every real crypto step (2-7) for real, same as
      an honest client's zero-delta round.
  (b) `run_training_loop`'s mutual-exclusivity assertion, which fires
      (per `fl/round_loop.py` line ~384) immediately after normalizing
      `cheater_client_idx`, before `suspect_tracker` is initialized and
      long before any per-client/torch-dependent code runs -- reached
      with minimal, type-tolerant placeholder arguments.
"""
from __future__ import annotations

import numpy as np
import pytest

from crypto.ristretto import backend_name as _backend_name

if _backend_name() is None:
    pytest.skip(
        "no ristretto255 backend available: install rbcl, or set "
        "JANUS_ALLOW_PURE_PYTHON_GROUP=1 to use the test-only pure-Python "
        "backend (crypto/_ristretto_py.py)",
        allow_module_level=True,
    )

from crypto.ristretto import Point, Scalar, scalar_mul_base
from crypto.secagg import new_self_mask_secret
from fl.client import ClientKeys, ClientRoundContext, round_step_phase1
from fl.round_loop import run_training_loop

MODEL_DIM = 8
PK_TRACE = Point.hash_to_point(b"test-pk-trace-cheater-delta-transform")


def _make_ctx(index: int = 0) -> ClientRoundContext:
    sk = Scalar.random()
    pk = scalar_mul_base(sk)
    return ClientRoundContext(
        keys=ClientKeys(sk=sk, pk=pk, index=index),
        ring=[pk],
        pairwise_secrets={},
        self_mask_secret_int=new_self_mask_secret(),
        blinding_self_mask_secret_int=new_self_mask_secret(),
        model_dim=MODEL_DIM,
        quant_bits=16,
    )


def _phase1(ctx, **kwargs):
    defaults = dict(
        w_t=None,
        rho_t=b"rho_t_test",
        rho_e=b"rho_e_test",
        vrf_alpha=b"vrf_alpha_test",
        sampling_rate=0.0,  # forces b_k=False deterministically (VRF p=0 -> never sampled)
        sigma_dp=0.0,       # DP disabled (F19) -- nu_k is the exact zero vector regardless
        n_clients=1,
        local_data=None,
        model=None,
        lr=0.0,
        epochs=0,
        pk_trace=PK_TRACE,
        batch_size=1,
        momentum=0.0,
    )
    defaults.update(kwargs)
    return round_step_phase1(ctx, **defaults)


# ------------------------------------------------- round_step_phase1 wiring

def test_delta_transform_called_with_real_zero_delta_on_unsampled_client():
    ctx = _make_ctx()
    calls = []

    def tf(d):
        calls.append(d.copy())
        return d + 5.0

    sub = _phase1(ctx, _delta_transform=tf)

    assert len(calls) == 1
    # b_k=False (sampling_rate=0.0) -> Step 1's real delta is exact zero.
    np.testing.assert_allclose(calls[0], np.zeros(MODEL_DIM))
    # Submission still comes out the other end of the real crypto pipeline.
    assert sub.was_sampled is False
    assert sub.commitment_point is not None


def test_no_transform_is_honest_default():
    ctx = _make_ctx()
    # No _delta_transform, no _force_delta -- must not raise, must behave
    # exactly like an honest unsampled client.
    sub = _phase1(ctx)
    assert sub.was_sampled is False
    assert sub.commitment_point is not None


def test_force_delta_wins_when_both_given():
    # fl/client.py's own docstring: "If BOTH _force_delta and
    # _delta_transform are given, _force_delta wins (checked first,
    # below)" -- unreachable via run_training_loop (which asserts
    # against this combination), but round_step_phase1 itself still
    # defines this precedence, so it's worth pinning directly.
    ctx = _make_ctx()
    transform_calls = []

    def tf(d):
        transform_calls.append(d)
        return d + 999.0  # would be wildly distinguishable if it ran

    forced = np.full(MODEL_DIM, 3.0)
    sub = _phase1(ctx, _force_delta=forced, _delta_transform=tf)

    assert transform_calls == []  # _delta_transform never invoked
    assert sub.commitment_point is not None


def test_transform_does_not_crash_real_pipeline_for_nonzero_output():
    # A transform that produces a genuinely non-zero, non-trivial vector
    # (standing in for e.g. craft_sign_flip/craft_gaussian_noise on a
    # zero real delta) still has to survive quantize -> double-mask ->
    # Pedersen commit -> ElGamal escrow -> CLSAG sign without error.
    ctx = _make_ctx()
    rng = np.random.default_rng(0)

    def tf(d):
        return d + rng.normal(0.0, 10.0, size=d.shape)

    sub = _phase1(ctx, _delta_transform=tf)
    assert sub.commitment_point is not None
    assert sub.ring_signature is not None


# ---------------------------------------------- run_training_loop assertion

def test_mutual_exclusivity_assertion_fires():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        run_training_loop(
            model=None,
            model_dim=MODEL_DIM,
            client_datasets=[None, None],
            ring=[],
            client_sks=[],
            pairwise_secrets={},
            pk_trace=PK_TRACE,
            server_root_norms=np.array([1.0]),
            CFG={},
            rounds=0,
            cheater_client_idx=0,
            cheater_force_delta=np.zeros(MODEL_DIM),
            cheater_delta_transform=lambda d, bt: d,
        )


def _minimal_cfg():
    # Just enough of CFG's structure for run_training_loop to get past
    # its own pre-loop setup (sampling_rate/sigma_dp/sketch_m/epoch_rounds/
    # k/ema_alpha lookups) -- values are arbitrary, since with rounds=0
    # (or a model=None that trips _require_torch first) the loop body
    # itself never executes.
    return {
        "protocol": {"sampling_rate": 0.0, "epoch_rounds": 1},
        "dp": {"sigma": 0.0},
        "sketch": {"m": 8},
        "norm_rule": {"k": 3.0, "ema_alpha": 0.1},
    }


def test_assertion_does_not_fire_with_only_force_delta():
    # Sanity check that the assertion is specifically about BOTH being
    # set, not a blanket rejection of cheater_force_delta. `model=None`
    # means `flatten_state_dict` (called unconditionally, even before
    # round 0, regardless of `rounds`) hits `fl/models.py`'s own
    # `_require_torch()` and raises RuntimeError -- torch genuinely
    # isn't installed in this sandbox. That RuntimeError, rather than
    # the mutual-exclusivity AssertionError, is itself the proof the
    # assertion did not fire for this single-param case: if it had,
    # we'd see AssertionError here instead, since the assert runs
    # strictly before flatten_state_dict.
    with pytest.raises(RuntimeError, match="torch"):
        run_training_loop(
            model=None,
            model_dim=MODEL_DIM,
            client_datasets=[None, None],
            ring=[],
            client_sks=[],
            pairwise_secrets={},
            pk_trace=PK_TRACE,
            server_root_norms=np.array([1.0]),
            CFG=_minimal_cfg(),
            rounds=0,
            cheater_client_idx=0,
            cheater_force_delta=np.zeros(MODEL_DIM),
        )


def test_assertion_does_not_fire_with_only_delta_transform():
    with pytest.raises(RuntimeError, match="torch"):
        run_training_loop(
            model=None,
            model_dim=MODEL_DIM,
            client_datasets=[None, None],
            ring=[],
            client_sks=[],
            pairwise_secrets={},
            pk_trace=PK_TRACE,
            server_root_norms=np.array([1.0]),
            CFG=_minimal_cfg(),
            rounds=0,
            cheater_client_idx=0,
            cheater_delta_transform=lambda d, bt: d,
        )
