"""Per-round evaluation hook support for `fl.round_loop.run_training_loop`
(FIX.md F21, 2026-11-17).

Deliberately its OWN module, not defined inside `fl/round_loop.py`
itself, for the same reason `crypto/dp.py::dp_noise_share` was pulled
out of `fl/client.py` (FIX.md F19's test-suite-slowdown fix):
`fl.round_loop` imports `verifier.audit` -> `crypto.pedersen`, which
does a hash-to-point call at MODULE IMPORT TIME that requires a real
ristretto255 group backend (rbcl, or the slow pure-Python fallback
behind `JANUS_ALLOW_PURE_PYTHON_GROUP=1`). A plain `from fl.round_loop
import _run_eval_fn_safely` in a test file -- with no group backend
installed, as in this sandbox -- fails the whole test session at
collection time before a single test runs; setting the env var to
work around it switches the ENTIRE pytest process onto the slow
backend, which is exactly the mistake F19 made and then reverted.

This module has no such import chain -- plain Python, no torch, no
crypto -- so it can be imported and unit-tested (see
`tests/test_eval_fn_hook.py`) without either problem.
"""


class _EvalFailed:
    """Sentinel: `eval_fn` was given but raised. Distinct from `None` (no
    `eval_fn` was given at all) so a round where eval was requested but
    genuinely failed is distinguishable from a round where no eval was
    ever requested -- otherwise `eval_fn` returning a legitimate `None`
    on success and `eval_fn` not being passed at all would be
    indistinguishable, and a round that quietly swallowed an exception
    could still write `history[-1]["eval"] = None` as if that were a
    real (if empty) result.
    """


EVAL_FAILED = _EvalFailed()


def run_eval_fn_safely(eval_fn, t_round, w_t, verbose=True):
    """Calls `eval_fn(t_round, w_t)` and returns its result. Never lets an
    exception from `eval_fn` propagate -- an eval bug (e.g. a shape
    mismatch in the caller's own evaluate_accuracy) must not cost a
    round of already-paid-for GPU + crypto work.

    Returns `None` if `eval_fn` is `None` (no eval requested -- the
    pre-F21 default, unchanged behavior). Returns the sentinel
    `EVAL_FAILED` if `eval_fn` raised. Otherwise returns whatever
    `eval_fn` returned (including a legitimate `None`, which the caller
    distinguishes from `EVAL_FAILED` by identity, not equality).
    """
    if eval_fn is None:
        return None
    try:
        result = eval_fn(t_round, w_t)
        if verbose:
            print(f"  round {t_round} eval: {result}")
        return result
    except Exception as e:
        if verbose:
            print(f"  round {t_round}: eval_fn raised {e!r} -- skipping this round's eval, training continues")
        return EVAL_FAILED
