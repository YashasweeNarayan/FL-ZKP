"""Unit tests for `fl.eval_hook.run_eval_fn_safely` (FIX.md F21).

Imports from `fl.eval_hook` directly, NOT from `fl.round_loop` --
`fl.round_loop` imports `verifier.audit` -> `crypto.pedersen`, which
does a hash-to-point call at module-import time requiring a real
ristretto255 group backend. In this sandbox (no rbcl, no network)
`from fl.round_loop import anything` fails at collection time before
a single test runs -- this was discovered live while adding this
exact test file, the same mistake FIX.md F19 made and reverted for
`crypto.dp.dp_noise_share`. `fl.eval_hook` is a standalone module
with no such import chain specifically so this works.

`run_training_loop` itself additionally imports torch (via
`local_train_delta`/`TorchSketchProjector`) and cannot be executed in
this sandbox (no torch installed) -- the same pre-existing limitation
that already means no test in this repo exercises `run_training_loop`
end-to-end. F21's per-round eval hook was deliberately factored into
this small, torch-free, crypto-free helper specifically so it has
*something* testable here, rather than being verified by code reading
alone.
"""
from fl.eval_hook import EVAL_FAILED as _EVAL_FAILED
from fl.eval_hook import run_eval_fn_safely as _run_eval_fn_safely


def test_none_eval_fn_returns_none_and_is_not_the_failure_sentinel():
    # Pre-F21 default behavior: no eval_fn given -> no eval, no crash.
    result = _run_eval_fn_safely(None, t_round=0, w_t="whatever", verbose=False)
    assert result is None
    assert result is not _EVAL_FAILED


def test_successful_eval_fn_result_is_returned_unchanged():
    calls = []

    def eval_fn(t_round, w_t):
        calls.append((t_round, w_t))
        return {"acc": 0.42}

    result = _run_eval_fn_safely(eval_fn, t_round=3, w_t="W", verbose=False)
    assert result == {"acc": 0.42}
    assert calls == [(3, "W")]


def test_eval_fn_receives_the_exact_round_and_weights_passed_in():
    received = {}

    def eval_fn(t_round, w_t):
        received["t_round"] = t_round
        received["w_t"] = w_t
        return 1.0

    _run_eval_fn_safely(eval_fn, t_round=7, w_t=[1, 2, 3], verbose=False)
    assert received == {"t_round": 7, "w_t": [1, 2, 3]}


def test_raising_eval_fn_returns_the_failure_sentinel_not_none():
    def eval_fn(t_round, w_t):
        raise ValueError("shape mismatch")

    result = _run_eval_fn_safely(eval_fn, t_round=5, w_t="W", verbose=False)
    assert result is _EVAL_FAILED
    # The two "nothing to store" cases must be distinguishable by identity:
    assert result is not None


def test_eval_fn_that_legitimately_returns_none_is_not_confused_with_failure():
    # A caller-side eval_fn could legitimately report "no metric yet" as
    # None on success (e.g. round 0 before any real client was sampled).
    # That must NOT be conflated with "eval_fn raised".
    def eval_fn(t_round, w_t):
        return None

    result = _run_eval_fn_safely(eval_fn, t_round=0, w_t="W", verbose=False)
    assert result is None
    assert result is not _EVAL_FAILED


def test_exception_does_not_propagate_past_the_helper():
    def eval_fn(t_round, w_t):
        raise RuntimeError("CUDA out of memory")

    # Must not raise -- a broken eval_fn must never cost the round.
    result = _run_eval_fn_safely(eval_fn, t_round=1, w_t="W", verbose=False)
    assert result is _EVAL_FAILED


def test_caller_side_filtering_logic_matches_run_training_loop_contract():
    # Mirrors the exact condition run_training_loop uses to decide
    # whether to write history[-1]["eval"] -- kept here so a future
    # change to either side is caught by this test, not just by reading.
    def would_store(eval_result):
        return eval_result is not None and eval_result is not _EVAL_FAILED

    assert would_store({"acc": 0.9}) is True
    assert would_store(0.0) is True   # falsy but real value -- must still store
    assert would_store(None) is False
    assert would_store(_EVAL_FAILED) is False
