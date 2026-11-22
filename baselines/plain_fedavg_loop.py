"""Plaintext (no crypto at all) FL training loop -- shared by both new
comparison notebooks:

1. Overhead comparison (`RUN_MVP_OVERHEAD_COMPARISON.ipynb`): timed
   against JANUS-FL's real `fl/round_loop.py::run_training_loop` at the
   same model/data/client scale, to answer "what does the crypto layer
   cost" with a real measurement instead of an estimate.
2. Baseline-defense comparison (`RUN_MVP_BASELINE_DEFENSES_COMPARISON.ipynb`):
   run under attack with `baselines/aggregation_rules.py`'s defenses
   swapped in, to answer "how does JANUS-FL's own verified-sum defense
   compare to Krum/trimmed-mean/coordinate-median under the same attack."

**Apples-to-apples by construction, not by claim**: every client's
LOCAL training step calls `fl/client.py::local_train_delta` directly --
the exact same function JANUS-FL's own `round_step_phase1` calls. This
file adds NO new local-training code and does not modify that function;
the only thing that differs between this loop and the real protocol is
what happens to the deltas AFTER local training (plain averaging / a
robust rule here, vs. masking+ZK-verification+secure-sum there). Any
accuracy or timing difference this produces is attributable to the
aggregation/security layer, not to a different training procedure.

Deliberately NOT touching fl/round_loop.py, fl/server.py, fl/client.py,
or anything in crypto/ -- this is a new, standalone, read-only-importing
module so it carries zero risk to the existing, already-tested pipeline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlainRoundRecord:
    round_idx: int
    wall_clock_seconds: float
    test_acc: float | None = None
    val_acc: float | None = None
    aggregator_used: str = ""
    selected_client_idx: int | None = None  # only meaningful for krum


@dataclass
class PlainLoopResult:
    history: list[PlainRoundRecord] = field(default_factory=list)


def run_plain_fedavg_loop(
    model, model_dim: int, client_datasets: list, rounds: int, lr: float, epochs: int,
    batch_size: int, aggregator, n_byzantine: int = 0,
    cheater_indices: "set[int] | None" = None, craft_fn=None,
    eval_fn=None, seed: int = 0,
) -> PlainLoopResult:
    """Runs `rounds` rounds of plaintext FedAvg (or a robust-aggregation
    variant, via `aggregator`) over `client_datasets`, timing each
    round's full wall-clock (local training for every client + the
    aggregation step -- no masking, no sketching, no range proofs, no
    signatures, no verification of any kind).

    `aggregator(deltas: list[np.ndarray], n_byzantine: int) -> np.ndarray`
    -- one of `baselines/aggregation_rules.py`'s `AGGREGATION_RULES`
    entries, or any callable with that signature.

    `cheater_indices` + `craft_fn(real_delta: np.ndarray) -> np.ndarray`:
    if given, every client index in `cheater_indices` has its REAL,
    honestly-trained delta (from `local_train_delta`, same as everyone
    else) passed through `craft_fn` before aggregation -- reusing this
    repo's own already-tested attack-crafting functions
    (`attacks/baseline_norm_attacks.py`, `attacks/a1_scaling.py`)
    unmodified, exactly as JANUS-FL's own `cheater_delta_transform` hook
    does in `fl/round_loop.py` (same idea, applied here to a plaintext
    loop instead).

    `eval_fn(model) -> (test_acc, val_acc)`, called after every round if
    given.
    """
    import torch

    from fl.client import local_train_delta
    from fl.models import flatten_state_dict, unflatten_to_state_dict

    cheater_indices = cheater_indices or set()
    result = PlainLoopResult()
    w_t, shapes = flatten_state_dict(model)
    w_t = w_t.detach().cpu().numpy().astype(np.float64)
    model_dtype = next(model.parameters()).dtype
    # Caught live on Kaggle, 2026-11-23: the model lives on whatever
    # device the notebook moved it to (`grid_model.to(device)` --
    # cuda:0 on a GPU runtime), but a tensor built from a plain numpy
    # array with no `device=` kwarg always lands on CPU. local_train_delta
    # (fl/client.py) accepts that fine for loading INTO the model
    # (load_state_dict copies data across devices), but at the end of
    # training it computes `new_flat - w_t` where `new_flat` comes from
    # flatten_state_dict(model) -- i.e. lives on model_device. Subtracting
    # a CPU tensor from that raised "Expected all tensors to be on the
    # same device, but found at least two devices, cuda:0 and cpu!" on
    # every client in every grid cell. Must build w_t_tensor on
    # model_device explicitly, every round.
    model_device = next(model.parameters()).device

    for r in range(rounds):
        t0 = time.perf_counter()
        # local_train_delta (fl/client.py) loads w_t straight into the
        # model via unflatten_to_state_dict + load_state_dict, which
        # requires an actual torch.Tensor -- w_t itself stays a numpy
        # array across rounds (the aggregators in
        # baselines/aggregation_rules.py operate on plain numpy deltas),
        # so it must be re-converted to a tensor on every round before
        # being handed to local_train_delta. Caught live on Kaggle,
        # 2026-11-22: without this conversion, load_state_dict raises
        # "expected torch.Tensor or Tensor-like object from checkpoint
        # but received <class 'numpy.ndarray'>" on every single client,
        # for every rule x attack grid cell -- this ran all the way
        # through the overhead-comparison and baseline-defenses fixes
        # committed just before this one without being caught, because
        # neither of those earlier fixes (missing numpy import, the
        # local_epochs CFG-key typo) touched this code path.
        w_t_tensor = torch.tensor(w_t, dtype=model_dtype, device=model_device)
        deltas = []
        for k, ds in enumerate(client_datasets):
            delta = local_train_delta(
                model, w_t_tensor, ds, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed * 1000 + r * 50 + k,
            )
            if k in cheater_indices and craft_fn is not None:
                delta = craft_fn(delta)
            deltas.append(delta)

        agg_delta = aggregator(deltas, n_byzantine)
        w_t = w_t + agg_delta

        elapsed = time.perf_counter() - t0
        rec = PlainRoundRecord(round_idx=r, wall_clock_seconds=elapsed)

        if eval_fn is not None:
            state_dict = unflatten_to_state_dict(
                torch.tensor(w_t, dtype=model_dtype, device=model_device), shapes,
            )
            model.load_state_dict(state_dict)
            rec.test_acc, rec.val_acc = eval_fn(model)

        result.history.append(rec)

    return result
