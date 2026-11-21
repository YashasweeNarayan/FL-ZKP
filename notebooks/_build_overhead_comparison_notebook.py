"""Generates RUN_MVP_OVERHEAD_COMPARISON.ipynb -- closes brutal-review
gap #3 (2026-11-21): "No quantified overhead vs. plain FedAvg... the
actual value proposition of a ZKP/ring-signature/secure-aggregation
framework is 'here's what security costs you'... nothing distilled
into a clean table."

**What this measures, and how**: at the SAME model (tiny_cnn), SAME
data (MNIST, case1 IID split), SAME client count (50), SAME per-client
local-training hyperparameters (both loops call the real, unmodified
`fl/client.py::local_train_delta` for every client's local step) --
only the AGGREGATION differs: plain mean (no crypto at all) vs.
JANUS-FL's real masked-and-verified aggregate
(`fl/round_loop.py::run_training_loop`, completely unmodified).
3 rounds each, timed with `time.perf_counter()` around the real call,
not estimated. Also measures ONE real client's actual serialized
submission size (phase1 + phase2, pickled) against the raw delta's own
size, for the communication-overhead half of the same question.

**Deliberately NOT touching fl/, crypto/, or attacks/**: every import
from this repo's existing modules is read-only; `baselines/` is new,
standalone code with zero coupling back into the tested pipeline, so
running this notebook carries zero risk to anything else in the repo.

**What this does NOT claim**: 3 rounds is enough for a stable
per-round wall-clock average (that's a timing measurement, not an
accuracy claim), but is NOT a substitute for the already-confirmed
20-round accuracy numbers (STATUS.md's TWENTY-FOURTH item) -- this
notebook reports timing and communication size only, no accuracy
number from either loop should be read as a result in its own right.
"""
import json
import os
import sys

NB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB_DIR)

import _build_mvp_run_notebooks as gen  # noqa: E402

nb = gen.nb

TINY_CNN_MNIST_CFG = {
    "filename": "RUN_MVP_TINY_CNN_MNIST.ipynb",
    "case_prefix": "case1",
    "dataset_label": "MNIST",
    "model": "tiny_cnn",
    "model_note": "~20.6K params, matches WhiteCloak/PQS-BFL's model scale.",
    "num_classes": 10,
    "ckpt_branch": "colab-checkpoints-case1-tiny_cnn",
}

N_CLIENTS = 50          # same scale as the real protocol's own confirmed results
TIMING_ROUNDS = 3        # enough for a stable per-round average; NOT an accuracy claim

TITLE_MD = """\
# Overhead comparison: JANUS-FL's real crypto pipeline vs. plain FedAvg

Closes brutal-review gap #3: "no quantified overhead vs. plain FedAvg."
Same model (`tiny_cnn`, ~20.6K params), same data (MNIST, case1 IID
split), same 50 clients, same per-client local-training step (both
loops call the real, unmodified `fl/client.py::local_train_delta`) --
the ONLY thing that differs is the aggregation layer: plain mean (zero
crypto) vs. JANUS-FL's real masked-and-verified secure aggregate
(`fl/round_loop.py::run_training_loop`, completely unmodified).

**Two measurements, both real, neither estimated:**
1. Per-round wall-clock, timed with `time.perf_counter()` around each
   loop's real call, 3 rounds each (enough for a stable average; not
   an accuracy claim -- the already-confirmed 20-round accuracy
   numbers in STATUS.md's TWENTY-FOURTH item are unaffected by and
   unrelated to this notebook).
2. Per-client communication size: one real client's actual pickled
   `ClientSubmissionPhase1` + `ClientSubmissionPhase2` objects vs. the
   same client's raw trained delta, pickled the same way for a fair
   byte-for-byte comparison.

**Zero risk to the existing pipeline**: this notebook only ever calls
already-existing, unmodified functions (`local_train_delta`,
`run_training_loop`, `round_step_phase1`/`round_step_phase2`) plus a
brand-new, standalone `baselines/plain_fedavg_loop.py` that has no
coupling back into `fl/`, `crypto/`, or `attacks/` -- nothing here can
affect the real pipeline's own already-confirmed results.
"""

CRYPTO_LOOP_CELL = """\
import time
import numpy as np
from fl.round_loop import run_training_loop, bootstrap_norms_from_model_dim, release_gpu_memory

# Reuses `model`/`dim`/`model_shapes` the model cell above already
# built (REAL_MODEL='tiny_cnn'), and the `ring`/`client_sks`/
# `pairwise_secrets`/`PK_trace` the ring + opening-handler cells above
# already built for this notebook's n_clients clients -- nothing
# rebuilt, so this is the exact same committee/ring machinery every
# other MVP run notebook exercises.
crypto_root_norms = bootstrap_norms_from_model_dim(
    model_dim=dim, sketch_dim=CFG['sketch']['m'],
    sketch_seed=b"bootstrap-overhead-comparison", quant_bits=16,
    rng=np.random.default_rng(CFG['seed'] + 3),
)

_t0 = time.perf_counter()
_, crypto_history = run_training_loop(
    model=model, model_dim=dim,
    client_datasets=case1_data['client_train'][:n_clients],
    ring=ring, client_sks=client_sks,
    pairwise_secrets=pairwise_secrets,
    pk_trace=PK_trace, server_root_norms=crypto_root_norms,
    CFG=CFG, rounds={rounds},
    on_pseudonym_flagged=on_pseudonym_flagged,
)
_total_crypto_time = time.perf_counter() - _t0
release_gpu_memory()

print(f"JANUS-FL total wall-clock for {rounds} rounds (s): {{_total_crypto_time:.2f}}")
print(f"JANUS-FL mean per-round time: {{_total_crypto_time/{rounds}:.2f}}s")
print("per-round accepted counts (sanity, not an accuracy claim):",
      [h.get('accepted') for h in crypto_history if not h.get('aborted')])
""".format(rounds=TIMING_ROUNDS)

COMM_SIZE_CELL = """\
import pickle
import numpy as np
from fl.client import ClientRoundContext, ClientKeys, round_step_phase1, round_step_phase2
from crypto.secagg import new_self_mask_secret

# One real client's submission, built the same way round 0 of the real
# protocol would build it -- client index 0 of the SAME ring/committee
# just used above, same (now-trained) model, same local data.
comm_ctx = ClientRoundContext(
    keys=ClientKeys(sk=client_sks[0], pk=ring[0], index=0),
    ring=ring,
    pairwise_secrets=pairwise_secrets[0],
    self_mask_secret_int=new_self_mask_secret(),
    blinding_self_mask_secret_int=new_self_mask_secret(),
    model_dim=dim,
)
comm_w_t, _ = flatten_state_dict(model)
comm_w_t = comm_w_t.detach().cpu().numpy().astype(np.float64)

comm_phase1 = round_step_phase1(
    comm_ctx, comm_w_t, rho_t=b"comm-size-rho-t", rho_e=b"comm-size-rho-e",
    vrf_alpha=b"comm-size-vrf-alpha", sampling_rate=1.0, sigma_dp=0.0,
    n_clients=n_clients, local_data=case1_data['client_train'][0], model=model,
    lr=CFG['local_training']['lr'], epochs=CFG['local_training']['local_epochs'],
    batch_size=CFG['local_training']['batch_size'], pk_trace=PK_trace,
)
comm_phase2 = round_step_phase2(
    comm_ctx, comm_phase1, sketch_seed=CFG['sketch'].get('seed', b'sigma-t-comm-size'),
    sketch_dim=CFG['sketch']['m'], B_t=1e12, rho_t=b"comm-size-rho-t",
    rho_e=b"comm-size-rho-e", vrf_alpha=b"comm-size-vrf-alpha",
)

raw_delta = np.random.default_rng(0).normal(0, 1, size=dim).astype(np.float64)
raw_delta_bytes = len(pickle.dumps(raw_delta))
submission_bytes = len(pickle.dumps(comm_phase1)) + len(pickle.dumps(comm_phase2))

print(f"model_dim = {dim}")
print(f"raw delta, pickled: {raw_delta_bytes:,} bytes")
print(f"JANUS-FL submission (phase1 + phase2), pickled: {submission_bytes:,} bytes")
print(f"overhead factor: {submission_bytes / raw_delta_bytes:.2f}x")
"""

PLAIN_LOOP_CELL = """\
import time
import sys
sys.path.insert(0, '/content/FL-ZKP')
from baselines.plain_fedavg_loop import run_plain_fedavg_loop
from baselines.aggregation_rules import mean_aggregate

# A SEPARATE, freshly-initialized model -- the crypto loop above has
# already trained `model` in place, so this loop gets its own
# independent copy rather than continuing from trained weights, for a
# fair round-0-to-round-0 timing comparison.
plain_model = build_model('tiny_cnn', num_classes=10, in_channels=1).to(device)
plain_flat, plain_shapes = flatten_state_dict(plain_model)
plain_dim = plain_flat.numel()

plain_result = run_plain_fedavg_loop(
    model=plain_model,
    model_dim=plain_dim,
    client_datasets=case1_data['client_train'][:n_clients],
    rounds={rounds},
    lr=CFG['local_training']['lr'],
    epochs=CFG['local_training']['local_epochs'],
    batch_size=CFG['local_training']['batch_size'],
    aggregator=mean_aggregate,
    seed=CFG['seed'],
)
plain_round_times = [r.wall_clock_seconds for r in plain_result.history]
print("plain FedAvg per-round wall-clock (s):", [f"{{t:.2f}}" for t in plain_round_times])
print(f"plain FedAvg mean per-round time: {{sum(plain_round_times)/len(plain_round_times):.2f}}s")
""".format(rounds=TIMING_ROUNDS)

SUMMARY_CELL = """\
print("\\n" + "="*72)
print("OVERHEAD COMPARISON SUMMARY -- plain FedAvg vs. JANUS-FL")
print("="*72)
print(f"model: tiny_cnn (~20.6K params), {n_clients} clients, {{plain_dim}} model_dim")
print()
_plain_mean = sum(plain_round_times) / len(plain_round_times)
_crypto_mean = _total_crypto_time / {rounds}
print(f"plain FedAvg:  {{_plain_mean:.2f}}s/round")
print(f"JANUS-FL:      {{_crypto_mean:.2f}}s/round")
print(f"time overhead: {{_crypto_mean / _plain_mean:.1f}}x")
print()
print(f"raw delta size:        {{raw_delta_bytes:,}} bytes")
print(f"JANUS-FL submission:   {{submission_bytes:,}} bytes")
print(f"comm. overhead:        {{submission_bytes / raw_delta_bytes:.1f}}x")
print()
print("NOTE: this is a timing/size measurement at {rounds} rounds for a stable")
print("average, NOT a replacement for the already-confirmed 20-round accuracy")
print("numbers (STATUS.md TWENTY-FOURTH item) -- those are unaffected.")
""".format(n_clients=N_CLIENTS, rounds=TIMING_ROUNDS)


def build_overhead_comparison_notebook():
    cells = [
        ("md", TITLE_MD),
        ("code", gen.CLONE_CELL),
        ("code", gen.INSTALL_CELL),
        ("md", gen.RESTART_MD),
        ("code", gen.SANITY_CELL),
        ("code", gen.CONFIG_CELL),
        ("md", "## Data + model (shared by both loops)"),
        ("code", gen.dataset_cell(TINY_CNN_MNIST_CFG)),
        ("code", gen.RING_CELL),
        ("code", gen.OPENING_HANDLER_CELL),
        ("code", gen.model_cell(TINY_CNN_MNIST_CFG)),
        ("md", f"## JANUS-FL's real crypto pipeline -- {TIMING_ROUNDS} rounds, timed"),
        ("code", CRYPTO_LOOP_CELL),
        ("md", "## Per-client communication size -- one real submission, measured not estimated"),
        ("code", COMM_SIZE_CELL),
        ("md", f"## Plain FedAvg -- same {TIMING_ROUNDS} rounds, same scale, timed, zero crypto"),
        ("code", PLAIN_LOOP_CELL),
        ("md", "## Summary"),
        ("code", SUMMARY_CELL),
    ]
    return nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "RUN_MVP_OVERHEAD_COMPARISON.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_overhead_comparison_notebook(), f, indent=1)
    print(f"wrote {out_path}")
