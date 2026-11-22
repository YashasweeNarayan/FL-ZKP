"""Regression test for a 2026-11-23 bug caught on a live Kaggle run of
`RUN_MVP_OVERHEAD_COMPARISON.ipynb`'s `COMM_SIZE_CELL`:

    RuntimeError: Error(s) in loading state_dict for Sequential:
        expected torch.Tensor or Tensor-like object from checkpoint
        but received <class 'numpy.ndarray'>

This is a DIFFERENT call site from the `plain_fedavg_loop.py` bug fixed
the day before (`tests/test_plain_fedavg_loop.py`) -- same root
confusion (numpy vs. torch.Tensor), but here it was `COMM_SIZE_CELL`
converting `comm_w_t` to a numpy array with
`comm_w_t.detach().cpu().numpy().astype(np.float64)` before passing it
to `fl.client.round_step_phase1`, which calls
`local_train_delta(model, w_t, ...)` internally and requires `w_t` to
be an actual `torch.Tensor` (mirrors `fl/round_loop.py::run_training_loop`'s
own `w_t, model_shapes = flatten_state_dict(model)`, which is NEVER
converted to numpy before being passed to the exact same
`round_step_phase1` call). Fixed by removing the numpy conversion --
`comm_w_t` now stays whatever type `flatten_state_dict` returns, same
as every other real call site in this repo.

**Why this is a static source check, not a live call**: this sandbox
has neither `torch` nor `rbcl` installed and no network access to
install either (same limitation documented in
`tests/test_f6_batching_logic.py` and `tests/test_plain_fedavg_loop.py`),
so there is no way to actually import `fl.client.round_step_phase1`
and exercise it here. What CAN be checked here, permanently, is that
the notebook generator's own source text for this cell never
reintroduces the exact pattern that broke: converting the tensor
`flatten_state_dict` returns into a numpy array before using it as
`w_t`. This is deliberately narrow (it cannot catch a new bug of a
different shape), but it directly pins down this specific regression.
"""
import importlib
import re


def test_comm_size_cell_never_converts_comm_w_t_to_numpy():
    mod = importlib.import_module("notebooks._build_overhead_comparison_notebook")
    cell = mod.COMM_SIZE_CELL

    assert "comm_w_t, _ = flatten_state_dict(model)" in cell, (
        "COMM_SIZE_CELL no longer builds comm_w_t the expected way -- "
        "update this test to match the new pattern if the cell's "
        "structure genuinely changed, but re-verify the fix still holds."
    )

    # The exact line that broke: reassigning comm_w_t to a numpy array
    # derived from itself. Catches both the original
    # `comm_w_t = comm_w_t.detach().cpu().numpy()...` and any
    # equivalent reassignment that routes through .numpy().
    assert not re.search(r"comm_w_t\s*=\s*comm_w_t\.", cell), (
        "COMM_SIZE_CELL reassigns comm_w_t from itself -- this is "
        "exactly the 2026-11-23 bug pattern (converting the tensor "
        "flatten_state_dict() returns into something else, e.g. numpy, "
        "before passing it to round_step_phase1, which requires a real "
        "torch.Tensor)."
    )

    # Whatever reaches round_step_phase1 as the second positional
    # argument must be the SAME name bound directly by flatten_state_dict
    # -- i.e. comm_w_t itself, untouched, consistent with
    # fl/round_loop.py::run_training_loop's own w_t, never converted.
    call_match = re.search(r"round_step_phase1\(\s*comm_ctx,\s*(\w+),", cell)
    assert call_match is not None, "could not find the round_step_phase1(...) call in COMM_SIZE_CELL"
    assert call_match.group(1) == "comm_w_t", (
        f"round_step_phase1 is called with {call_match.group(1)!r}, not comm_w_t directly"
    )


def test_plain_loop_cell_uses_aggregation_rules_wrapper_not_raw_function():
    """Regression test for the 2026-11-23 aggregator-arity bug fixed
    just before this one in the same notebook: PLAIN_LOOP_CELL must use
    a 2-argument AGGREGATION_RULES[...] wrapper, never a raw 1-argument
    rule function directly."""
    mod = importlib.import_module("notebooks._build_overhead_comparison_notebook")
    cell = mod.PLAIN_LOOP_CELL

    assert "from baselines.aggregation_rules import mean_aggregate" not in mod.PLAIN_LOOP_CELL
    assert "aggregator=mean_aggregate" not in cell
    assert 'aggregator=AGGREGATION_RULES["mean_no_defense"]' in cell
