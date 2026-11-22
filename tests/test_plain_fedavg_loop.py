"""Regression test for the 2026-11-22 Kaggle bug in
`baselines/plain_fedavg_loop.py::run_plain_fedavg_loop`: it kept `w_t`
as a raw `numpy.ndarray` across rounds (since the aggregation rules in
`baselines/aggregation_rules.py` operate on plain numpy deltas), but
passed that SAME raw numpy array straight into
`fl/client.py::local_train_delta`, which calls
`model.load_state_dict(unflatten_to_state_dict(w_t, shapes))` and
requires `w_t` to be an actual `torch.Tensor`. This broke on every
single client, for every rule x attack grid cell, on the user's real
Kaggle run of `RUN_MVP_BASELINE_DEFENSES_COMPARISON.ipynb`:

    RuntimeError: Error(s) in loading state_dict for Sequential:
        expected torch.Tensor or Tensor-like object from checkpoint
        but received <class 'numpy.ndarray'>

Fixed by re-converting `w_t` to a `torch.Tensor` at the top of every
round, right before the per-client training loop (see
`run_plain_fedavg_loop`'s own comment at the fix site).

**Why this test fakes torch/fl.client/fl.models instead of importing
them for real**: this sandbox has neither `torch` nor `rbcl` installed
and has no network access to install either (same limitation
documented in `tests/test_f6_batching_logic.py`'s own docstring --
`fl.client` cannot even be imported here, since importing it pulls in
`crypto/pedersen.py`, which needs `rbcl` at module-import time). So
this test cannot run the real local-training step; instead it installs
a minimal fake `torch` module (just enough surface:
`torch.tensor(data, dtype=...)` producing a tagged wrapper type) and a
fake `fl.client.local_train_delta`/`fl.models.flatten_state_dict`/
`unflatten_to_state_dict` via `sys.modules` injection, then exercises
`run_plain_fedavg_loop`'s REAL, unmodified control flow end-to-end and
asserts, directly, that whatever reaches the faked `local_train_delta`
as `w_t` is the fake tensor-wrapper type -- never a raw
`numpy.ndarray` -- which is exactly the type distinction the real bug
violated. This cannot prove torch's real `load_state_dict` accepts the
real tensor (that still needs a real Colab/Kaggle run, same as every
other torch-dependent code path in this repo), but it does pin down,
permanently, in a test that runs in THIS sandbox too, the exact
numpy-vs-tensor type confusion that caused the real failure -- so a
future refactor that reintroduces "pass w_t straight through as
whatever type it already is" breaks this test immediately instead of
waiting for the next live run to find it.
"""
from __future__ import annotations

import sys
import types

import numpy as np


class _FakeTensor:
    """Minimal stand-in for torch.Tensor -- tags itself so the test can
    tell a real conversion happened, distinct from a bare numpy array.
    Also tracks a fake `device` string and mirrors real torch's refusal
    to combine tensors on different devices, so a regression that drops
    `device=` from a `torch.tensor(...)` call site is caught here
    instead of only on a live GPU Kaggle/Colab run."""

    def __init__(self, data, dtype=None, device="cpu"):
        self._data = np.asarray(data, dtype=np.float64)
        self.dtype = dtype
        self.device = device

    def detach(self):
        return self

    def cpu(self):
        return _FakeTensor(self._data, dtype=self.dtype, device="cpu")

    def numpy(self):
        return self._data.copy()

    def __sub__(self, other):
        if isinstance(other, _FakeTensor) and other.device != self.device:
            # The exact 2026-11-23 Kaggle bug: `new_flat` (from
            # flatten_state_dict(model), on model_device) minus `w_t`
            # (built with no device= kwarg, defaulting to CPU) raised
            # "Expected all tensors to be on the same device, but found
            # at least two devices, cuda:0 and cpu!" on every client.
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found "
                f"at least two devices, {self.device} and {other.device}!"
            )
        other_arr = other._data if isinstance(other, _FakeTensor) else np.asarray(other)
        return _FakeTensor(self._data - other_arr, dtype=self.dtype, device=self.device)


class _FakeParam:
    dtype = "float64"
    # The real model lives on cuda:0 on a GPU runtime (`grid_model.to(device)`
    # in the notebook) -- fake that here so a w_t tensor built without
    # `device=` (defaulting to "cpu") triggers the same device-mismatch
    # this test exists to catch.
    device = "cuda:0"


class _FakeModel:
    def __init__(self):
        self._param = _FakeParam()
        self.last_state_dict = None

    def parameters(self):
        return iter([self._param])

    def load_state_dict(self, state_dict):
        # The real bug: the real model.load_state_dict rejects a
        # numpy.ndarray outright. Mirror that contract here so this
        # fake fails the same way the real one did if the type
        # regresses.
        for v in state_dict.values():
            if isinstance(v, np.ndarray):
                raise RuntimeError(
                    "load_state_dict expected torch.Tensor or Tensor-like "
                    f"object but received {type(v)}"
                )
        self.last_state_dict = state_dict


def _install_fakes(monkeypatch, dim, recorded_w_t_types, recorded_w_t_devices=None):
    fake_torch = types.ModuleType("torch")
    fake_torch.float64 = "float64"
    fake_torch.tensor = lambda data, dtype=None, device="cpu": _FakeTensor(data, dtype=dtype, device=device)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_fl_client = types.ModuleType("fl.client")

    def _fake_local_train_delta(model, w_t, local_data, lr, epochs, batch_size, seed=None):
        recorded_w_t_types.append(type(w_t))
        if recorded_w_t_devices is not None:
            recorded_w_t_devices.append(getattr(w_t, "device", None))
        # Exercise the same contract the real model.load_state_dict
        # enforces, via the fake model -- raises if w_t ever regresses
        # back to a raw numpy array.
        model.load_state_dict({"weight": w_t})
        # Mirror the real local_train_delta's final line:
        # `(new_flat - w_t)` where new_flat comes from
        # flatten_state_dict(model) -- i.e. lives on model_device.
        # This is what actually raised the device-mismatch error on
        # Kaggle; reproduce that same subtraction here.
        model_device = next(model.parameters()).device
        new_flat = _FakeTensor(np.zeros(dim, dtype=np.float64), device=model_device)
        delta_tensor = new_flat - w_t
        return delta_tensor.numpy()

    fake_fl_client.local_train_delta = _fake_local_train_delta
    monkeypatch.setitem(sys.modules, "fl.client", fake_fl_client)

    fake_fl_models = types.ModuleType("fl.models")

    def _fake_flatten_state_dict(model):
        model_device = next(model.parameters()).device
        return _FakeTensor(np.zeros(dim, dtype=np.float64), device=model_device), {"shape": (dim,)}

    def _fake_unflatten_to_state_dict(w_t, shapes):
        return {"weight": w_t}

    fake_fl_models.flatten_state_dict = _fake_flatten_state_dict
    fake_fl_models.unflatten_to_state_dict = _fake_unflatten_to_state_dict
    monkeypatch.setitem(sys.modules, "fl.models", fake_fl_models)


def test_run_plain_fedavg_loop_passes_tensor_not_ndarray_to_local_train_delta(monkeypatch):
    dim = 6
    recorded = []
    _install_fakes(monkeypatch, dim, recorded)

    # Force a fresh import so this module picks up the just-installed
    # fakes rather than any previously cached import of the real (or
    # absent) fl.client/fl.models/torch.
    sys.modules.pop("baselines.plain_fedavg_loop", None)
    from baselines.plain_fedavg_loop import run_plain_fedavg_loop
    from baselines.aggregation_rules import AGGREGATION_RULES

    model = _FakeModel()
    client_datasets = [object(), object(), object()]  # local_train_delta is faked; shape unused
    result = run_plain_fedavg_loop(
        model=model,
        model_dim=dim,
        client_datasets=client_datasets,
        rounds=2,
        lr=0.01,
        epochs=1,
        batch_size=8,
        aggregator=AGGREGATION_RULES["mean_no_defense"],
    )

    assert len(result.history) == 2
    # 2 rounds x 3 clients = 6 calls into the (faked) local_train_delta
    assert len(recorded) == 6, f"expected 6 local_train_delta calls, got {len(recorded)}"
    for t in recorded:
        assert t is _FakeTensor, (
            f"run_plain_fedavg_loop passed w_t to local_train_delta as {t}, "
            "not the tensor-wrapper type -- this is the exact 2026-11-22 "
            "Kaggle bug (RuntimeError: load_state_dict expected "
            "torch.Tensor ... but received numpy.ndarray) regressing."
        )


def test_run_plain_fedavg_loop_eval_fn_also_receives_a_tensor(monkeypatch):
    """The eval_fn branch already converted w_t correctly before this
    fix (it builds its own torch.tensor(...) call directly) -- this
    pins that down too, so a future edit can't silently break it while
    leaving the per-client branch fixed, or vice versa."""
    dim = 4
    recorded = []
    _install_fakes(monkeypatch, dim, recorded)

    sys.modules.pop("baselines.plain_fedavg_loop", None)
    from baselines.plain_fedavg_loop import run_plain_fedavg_loop
    from baselines.aggregation_rules import AGGREGATION_RULES

    model = _FakeModel()
    eval_calls = []

    def _eval_fn(m):
        eval_calls.append(m.last_state_dict["weight"])
        return 0.5, 0.4

    result = run_plain_fedavg_loop(
        model=model,
        model_dim=dim,
        client_datasets=[object(), object()],
        rounds=1,
        lr=0.01,
        epochs=1,
        batch_size=8,
        aggregator=AGGREGATION_RULES["mean_no_defense"],
        eval_fn=_eval_fn,
    )

    assert len(eval_calls) == 1
    assert isinstance(eval_calls[0], _FakeTensor), (
        f"eval_fn's model.load_state_dict received {type(eval_calls[0])}, "
        "not the tensor-wrapper type"
    )
    assert result.history[0].test_acc == 0.5
    assert result.history[0].val_acc == 0.4


def test_run_plain_fedavg_loop_builds_w_t_tensor_on_model_device(monkeypatch):
    """Regression test for a 2026-11-23 Kaggle bug: `w_t_tensor =
    torch.tensor(w_t, dtype=model_dtype)` built a CPU tensor (no
    `device=` kwarg) every round, but the model lives on cuda:0 on a
    GPU runtime (`grid_model.to(device)` in the baseline-defense-grid
    notebook). local_train_delta's final line, `new_flat - w_t`, then
    subtracted a cuda:0 tensor (`new_flat`, from
    flatten_state_dict(model)) and a CPU tensor (`w_t`), raising:

        RuntimeError: Expected all tensors to be on the same device,
        but found at least two devices, cuda:0 and cpu!

    on every single client, in every rule x attack grid cell. Fixed by
    building `model_device = next(model.parameters()).device` once and
    passing `device=model_device` to every `torch.tensor(w_t, ...)`
    call site (the per-round w_t_tensor, and the eval_fn branch).

    This test's fakes model device explicitly (_FakeParam.device =
    "cuda:0", _FakeTensor.__sub__ raises on a device mismatch exactly
    like real torch) so this reproduces the real failure mode, not
    just a type check.
    """
    dim = 5
    recorded_types = []
    recorded_devices = []
    _install_fakes(monkeypatch, dim, recorded_types, recorded_devices)

    sys.modules.pop("baselines.plain_fedavg_loop", None)
    from baselines.plain_fedavg_loop import run_plain_fedavg_loop
    from baselines.aggregation_rules import AGGREGATION_RULES

    model = _FakeModel()
    result = run_plain_fedavg_loop(
        model=model,
        model_dim=dim,
        client_datasets=[object(), object()],
        rounds=2,
        lr=0.01,
        epochs=1,
        batch_size=8,
        aggregator=AGGREGATION_RULES["mean_no_defense"],
    )

    assert len(result.history) == 2
    assert len(recorded_devices) == 4  # 2 rounds x 2 clients
    for d in recorded_devices:
        assert d == "cuda:0", (
            f"w_t reached local_train_delta on device {d!r}, not the "
            "model's own device 'cuda:0' -- this is the exact "
            "2026-11-23 device-mismatch bug (RuntimeError: Expected "
            "all tensors to be on the same device, but found at least "
            "two devices, cuda:0 and cpu!) regressing."
        )


def test_run_plain_fedavg_loop_rejects_raw_aggregation_rule_arity(monkeypatch):
    """Regression test for a 2026-11-23 bug caught by re-reading
    notebooks/_build_overhead_comparison_notebook.py's PLAIN_LOOP_CELL
    against run_plain_fedavg_loop's own documented contract (never run
    live -- caught before the cell ever executed, by a line-by-line
    cross-check, the same method the prior two fixes this day used).

    run_plain_fedavg_loop always calls `aggregator(deltas, n_byzantine)`
    -- a 2-ARGUMENT contract matching every
    `baselines/aggregation_rules.py::AGGREGATION_RULES` dict entry (each
    of which is a `lambda deltas, n_byz: ...` wrapper). The raw rule
    functions underneath (`mean_aggregate`, `coordinate_median`,
    `trimmed_mean`) are NOT all 2-argument -- `mean_aggregate(deltas)`
    takes exactly one. The overhead-comparison notebook's PLAIN_LOOP_CELL
    imported and passed `mean_aggregate` directly instead of going
    through `AGGREGATION_RULES["mean_no_defense"]`, which would have
    raised `TypeError: mean_aggregate() takes 1 positional argument but
    2 were given` on round 0's very first aggregation call. Fixed by
    switching that cell to `AGGREGATION_RULES["mean_no_defense"]`.

    This test pins the contract down two ways: the raw function raises
    when called the way run_plain_fedavg_loop calls its aggregator, and
    the correct AGGREGATION_RULES wrapper does not.
    """
    import pytest as _pytest
    from baselines.aggregation_rules import mean_aggregate, AGGREGATION_RULES

    dim = 5
    deltas = [np.zeros(dim) for _ in range(3)]

    with _pytest.raises(TypeError):
        mean_aggregate(deltas, 0)  # the exact call run_plain_fedavg_loop makes

    # The correct, 2-argument wrapper every grid cell in this repo
    # should use instead -- must NOT raise.
    agg = AGGREGATION_RULES["mean_no_defense"](deltas, 0)
    assert agg.shape == (dim,)

    dim = 6
    recorded = []
    _install_fakes(monkeypatch, dim, recorded)
    sys.modules.pop("baselines.plain_fedavg_loop", None)
    from baselines.plain_fedavg_loop import run_plain_fedavg_loop

    model = _FakeModel()
    with _pytest.raises(TypeError):
        run_plain_fedavg_loop(
            model=model,
            model_dim=dim,
            client_datasets=[object(), object()],
            rounds=1,
            lr=0.01,
            epochs=1,
            batch_size=8,
            aggregator=mean_aggregate,  # the exact bug: raw, 1-arg function
        )
