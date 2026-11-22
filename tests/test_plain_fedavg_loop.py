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
    tell a real conversion happened, distinct from a bare numpy array."""

    def __init__(self, data, dtype=None):
        self._data = np.asarray(data, dtype=np.float64)
        self.dtype = dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data.copy()

    def __sub__(self, other):
        other_arr = other._data if isinstance(other, _FakeTensor) else np.asarray(other)
        return _FakeTensor(self._data - other_arr)


class _FakeParam:
    dtype = "float64"


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


def _install_fakes(monkeypatch, dim, recorded_w_t_types):
    fake_torch = types.ModuleType("torch")
    fake_torch.float64 = "float64"
    fake_torch.tensor = lambda data, dtype=None: _FakeTensor(data, dtype=dtype)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_fl_client = types.ModuleType("fl.client")

    def _fake_local_train_delta(model, w_t, local_data, lr, epochs, batch_size, seed=None):
        recorded_w_t_types.append(type(w_t))
        # Exercise the same contract the real model.load_state_dict
        # enforces, via the fake model -- raises if w_t ever regresses
        # back to a raw numpy array.
        model.load_state_dict({"weight": w_t})
        return np.full(dim, 0.01, dtype=np.float64)

    fake_fl_client.local_train_delta = _fake_local_train_delta
    monkeypatch.setitem(sys.modules, "fl.client", fake_fl_client)

    fake_fl_models = types.ModuleType("fl.models")

    def _fake_flatten_state_dict(model):
        return _FakeTensor(np.zeros(dim, dtype=np.float64)), {"shape": (dim,)}

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
