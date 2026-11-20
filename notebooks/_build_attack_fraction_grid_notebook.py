"""Generates RUN_MVP_A1_FRACTION_GRID.ipynb: LEFT.md \u00a71.6's A1-only
fraction-sweep grid (2026-11-20).

LEFT.md \u00a71.6, decided 2026-11-16, originally scoped a grid of
{A1, A3, A4-replay, A5} x {10%, 20%, 30%} malicious clients, 5 rounds
each -- so the attack's defense-trigger-rate is reported as a function
of malicious fraction, comparable to how Krum/Trimmed-mean/FLTrust/RoFL
papers report a fraction sweep rather than one fixed point. That
engineering was never built before this. Investigating what the real
per-round pipeline actually supports (2026-11-20, before writing this)
found that only A1 (model-replacement scaling) has real per-round
integration -- `fl/client.py`'s `_force_delta` hook, exercised by every
attack7 notebook's own run cell (`notebooks/_build_mvp_run_notebooks.py`'s
`attack_run_cell`). A3 (sybil), A4-replay, and A5 (splice) attack the
ring-signature/VRF/key-image layer directly, at the submission-
construction step, not the trained delta -- they were deliberately
tested in isolation instead (`RUN_ATTACKS.ipynb`, 8/8 caught, FIX.md
F10/F11's own writeup explains why that's the right scope, not a gap).
Wiring them into real per-round submissions the way A1 is would mean
new engineering in `fl/round_loop.py`/`fl/client.py` to forge a sybil
resubmission, a cross-round replay, or a spliced key-image INSIDE a
live round -- a materially bigger, riskier task under deadline
pressure, not a notebook addition. Put to the user directly; decided:
**A1-only** for this grid. A3/A4/A5's isolated-test results (already
final) are what the paper cites for those three attack types.

Grid, as actually built here: A1 (model-replacement scaling) x
{10%, 20%, 30%} malicious of the same 50 clients, SAME model/dataset as
LEFT.md item #1 (tiny_cnn + MNIST, case1), 5 rounds each (not 20 --
the security claim, "does V3 reject the attack every round it's
eligible," doesn't need a long run; only accuracy-curve claims do, and
this grid is about the defense-trigger-rate table, not accuracy). No
\u00a76.2 traceability wiring in this notebook -- that's the combo
notebook's job (`RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb`), this grid
is purely the per-round V3 security claim at different fractions.

10%/20%/30% of 50 = 5/10/15 malicious clients -- `fl/round_loop.py`'s
`cheater_client_idx` already accepts a list/set of indices (generalized
2026-11-16, see its own docstring), so no library change was needed
for the fraction axis itself, only for threading three different
fractions through one notebook cleanly.

Reuses every shared cell-generator function from
`notebooks/_build_mvp_run_notebooks.py` (clone/install/sanity/config/
dataset/ring/model/checkpoint/eval) exactly like the combo notebook
does -- nothing here reimplements a cell that generator already owns.
Each of the 3 fraction cells gets its OWN checkpoint branch AND clone
directory (same mechanical reason as the combo notebook: reusing one
clone dir across cells skips `git_checkpoint_cell`'s branch-checkout
logic for the 2nd/3rd cell).

Computed abort-probability context (LEFT.md \u00a71.6's own table,
reproduced here so a reader of this file doesn't have to cross-
reference): at `p=0.25, n=50, k_min=8`, 10%/20%/30% malicious
(f=5/10/15, excluded from k_min-eligibility) predicts abort
probabilities of ~9.4%/~18.2%/~32.2% respectively -- more aborted
rounds at higher fractions is the SAMPLING model working as designed,
not a defense failure; the run cell below reports aborts separately
from V3-rejection verdicts so the two are never conflated.
"""
import json
import os
import sys

NB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB_DIR)

import _build_mvp_run_notebooks as gen  # noqa: E402

nb = gen.nb

CASE1_CFG = {
    "case_prefix": "case1",
    "model": "tiny_cnn",
    "dataset_label": "MNIST",
    "num_classes": 10,
    "model_note": "~20.6K params, same model/dataset as LEFT.md item #1 and every attack7 notebook's base case.",
    "ckpt_branch": "colab-checkpoints-case1-tiny_cnn-fracgrid",
}

# (label, n_cheaters, pct) -- 10%/20%/30% of 50 clients.
GRID_CELLS = [("10pct", 5, 10), ("20pct", 10, 20), ("30pct", 15, 30)]
REAL_ROUNDS = 5

TITLE_MD = f"""\
# A1 malicious-fraction grid: {", ".join(f"{p}%" for _, _, p in GRID_CELLS)}, {REAL_ROUNDS} rounds each

LEFT.md \u00a71.6's fraction-sweep grid, A1 (model-replacement scaling)
only -- see this notebook's generator script
(`notebooks/_build_attack_fraction_grid_notebook.py`) for why A3/A4-
replay/A5 are NOT in this grid (they attack the ring-signature/VRF/
key-image layer directly, not the trained delta, and have no real
per-round integration to sweep a fraction over; their isolated-test
results, already final, are what the paper cites for those three).

**Cells in this notebook, in run order:** {", ".join(f"{p}% malicious ({n} of 50 clients)" for _, n, p in GRID_CELLS)}.
Same model/dataset as LEFT.md item #1 (`tiny_cnn` + MNIST, case1) --
this grid is about the per-round V3-rejection rate as a function of
malicious fraction, not a new accuracy result.

**Deliberately {REAL_ROUNDS} rounds, not 20**: the claim this grid
supports -- "V3 rejects every eligible attacker, every round, at this
fraction" -- is decided on round 0 just as validly as round 19; a short
run gets the same answer at a fraction of the compute.

**Expect more ABORTED rounds at higher fractions** -- at `p=0.25,
n=50, k_min=8`, malicious clients are excluded from `k_min`-eligibility,
so predicted abort probability rises with fraction: ~9.4% at 10%,
~18.2% at 20%, ~32.2% at 30% (LEFT.md \u00a71.6's own table). This is
the sampling model working as designed; the run cell below reports
aborted rounds separately from V3-rejection verdicts so the two are
never conflated into one number.

**Checkpoints**: each of the 3 cells gets its own dedicated git branch
(`colab-checkpoints-case1-tiny_cnn-fracgrid-<10pct|20pct|30pct>`) and its own
clone directory, so none can collide with each other or with any
other notebook in this repo.

**Runtime**: {REAL_ROUNDS} rounds is a small fraction of a full attack7
run's ~5 hours -- expect well under an hour per cell, under 2 hours
total, including DKG/ring setup for each.
"""


def fraction_grid_run_cell(label, n_cheaters, pct):
    return f"""\
# Fraction-grid cell: {pct}% malicious ({n_cheaters} of 50 clients),
# {REAL_ROUNDS} rounds, A1 (model-replacement scaling). Same single-
# mechanism construction as every attack7 notebook's own run cell
# (notebooks/_build_mvp_run_notebooks.py's attack_run_cell) -- all
# {n_cheaters} attackers submit the SAME crafted delta (a simplification
# noted there too, not independently-drawn attacks per client).
import numpy as np
import torch
from fl.round_loop import run_training_loop, bootstrap_norms_from_model_dim, release_gpu_memory, load_checkpoint
from attacks.a1_scaling import craft_scaled_update

REAL_ROUNDS = {REAL_ROUNDS}
CHEATER_CLIENT_IDX = list(range({n_cheaters}))

CKPT_PATH = f'{{CKPT_DIR}}/case1_tiny_cnn_fracgrid_{label}_r{{REAL_ROUNDS}}_nodp_f20.pt'

root_norms = bootstrap_norms_from_model_dim(
    model_dim=dim,
    sketch_dim=CFG['sketch']['m'],
    sketch_seed="bootstrap-tiny_cnn-case1-fracgrid-{label}".encode(),
    quant_bits=16,
    rng=np.random.default_rng(CFG['seed']),
)

w_target = np.random.default_rng(CFG['seed'] + 99).normal(0, 1.0, size=dim)
malicious_delta = craft_scaled_update(flat_w.cpu().numpy().astype(np.float64), w_target)
print(f"[{label}] crafted malicious delta (clients {{CHEATER_CLIENT_IDX}}), raw norm:", np.linalg.norm(malicious_delta))

print("="*70)
print(f"[{label}] MALICIOUS CLIENTS: {{CHEATER_CLIENT_IDX}} ({{len(CHEATER_CLIENT_IDX)}} of {{n_clients}} = {{100*len(CHEATER_CLIENT_IDX)/n_clients:.0f}}%)")
print("="*70)

resume_from = load_checkpoint(CKPT_PATH) if os.path.exists(CKPT_PATH) else None
if resume_from is not None:
    print(f"[{label}] found checkpoint at round {{resume_from['round']}} -- resuming, not restarting from round 0")

_eval_images, _eval_labels = case1_data['test']


def _eval_fn(round_idx, w_t_now):
    from fl.models import unflatten_to_state_dict
    device = next(model.parameters()).device
    model.load_state_dict(unflatten_to_state_dict(w_t_now.to(device), model_shapes))
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(_eval_images), 512):
            bi = _eval_images[i:i + 512].to(device)
            bl = _eval_labels[i:i + 512].to(device)
            correct += (model(bi).argmax(dim=1) == bl).sum().item()
    return {{"acc": correct / len(_eval_images)}}


final_w_t, real_history = run_training_loop(
    model=model,
    model_dim=dim,
    client_datasets=case1_data['client_train'],
    ring=ring,
    client_sks=client_sks,
    pairwise_secrets=pairwise_secrets,
    pk_trace=PK_trace,
    server_root_norms=root_norms,
    CFG=CFG,
    rounds=REAL_ROUNDS,
    checkpoint_path=CKPT_PATH,
    resume_from=resume_from,
    cheater_client_idx=CHEATER_CLIENT_IDX,
    cheater_force_delta=malicious_delta,
    on_checkpoint_saved=push_checkpoint_to_git,
    eval_fn=_eval_fn,
)
release_gpu_memory()

n_aborted = sum(1 for h in real_history if h.get('aborted'))
n_completed = len(real_history) - n_aborted
attacker_ever_accepted = {{}}
attacker_ever_rejected = {{}}
for h in real_history:
    if h.get("aborted"):
        print(f"[{label}]   round {{h.get('round')}}: ABORTED -- {{h.get('reason')}}")
        continue
    reasons = h.get("rejected_reasons", {{}})
    accepted_idxs = set(h.get("accepted_client_indices", []))
    rejected_this_round = sorted(i for i in CHEATER_CLIENT_IDX if i in reasons)
    accepted_this_round = sorted(i for i in CHEATER_CLIENT_IDX if i in accepted_idxs)
    for i in rejected_this_round:
        attacker_ever_rejected[i] = True
    for i in accepted_this_round:
        attacker_ever_accepted[i] = True
    print(f"[{label}]   round {{h.get('round')}}: accepted={{h.get('accepted')}}, "
          f"attackers rejected={{rejected_this_round}}, attackers accepted={{accepted_this_round}}, "
          f"aggregate_exact={{h.get('aggregate_exact')}}")

print(f"\\n[{label}] {{n_completed}} of {{len(real_history)}} rounds completed, {{n_aborted}} aborted "
      f"(predicted abort rate at this fraction: see this notebook's title markdown).")
if attacker_ever_accepted:
    print(f"[{label}] FAIL: attacker(s) {{sorted(attacker_ever_accepted)}} had their scaled update accepted at least once.")
elif attacker_ever_rejected:
    print(f"[{label}] PASS: every eligible attacker ({{sorted(attacker_ever_rejected)}}) was rejected by V3 every time it submitted, across all {{REAL_ROUNDS}} rounds.")
else:
    print(f"[{label}] INCONCLUSIVE: no attacker was ever sampled/eligible in any round -- check sampling_rate/eligibility at this fraction.")

# Preserve this cell's results under a fraction-specific name before
# the next grid cell's RING_CELL/model_cell rebuild ring/model/flat_w
# (fresh per cell, same as the combo notebook's own convention) and
# before `real_history`/`n_aborted`/etc. get overwritten.
real_history_{label} = list(real_history)
n_aborted_{label} = n_aborted
n_completed_{label} = n_completed
attacker_ever_accepted_{label} = dict(attacker_ever_accepted)
attacker_ever_rejected_{label} = dict(attacker_ever_rejected)
CHEATER_CLIENT_IDX_{label} = list(CHEATER_CLIENT_IDX)
"""


def build_grid_notebook():
    cells = [("md", TITLE_MD), ("code", gen.CLONE_CELL), ("code", gen.INSTALL_CELL),
             ("md", gen.RESTART_MD), ("code", gen.SANITY_CELL), ("code", gen.CONFIG_CELL),
             ("code", gen.dataset_cell(CASE1_CFG))]

    for i, (label, n_cheaters, pct) in enumerate(GRID_CELLS):
        new_branch = f"{CASE1_CFG['ckpt_branch']}-{label}"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-fracgrid-{i}"
        cells.append(("md", f"## Grid cell {i + 1} of {len(GRID_CELLS)}: {pct}% malicious ({n_cheaters} of 50 clients)"))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.model_cell(CASE1_CFG)))
        cells.append(("md", gen.CHECKPOINT_MD if i == 0 else
                      "## Checkpoint durability for this cell\n\nSame git-backed checkpointing as the first cell -- this cell gets its own clone directory and its own dedicated branch, so it can never collide with another cell's checkpoint in this same session."))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", fraction_grid_run_cell(label, n_cheaters, pct)))

    # Final combined defense-trigger-rate table across all 3 fractions.
    summary_lines = [
        "print(\"\\n\" + \"=\"*70)",
        "print(\"A1 FRACTION-SWEEP SUMMARY -- defense-trigger-rate vs. malicious fraction\")",
        "print(\"=\"*70)",
        "print(f\"{'fraction':>10} | {'rounds':>7} | {'completed':>10} | {'aborted':>8} | {'attackers rejected':>20} | {'attackers accepted':>20}\")",
    ]
    for label, n_cheaters, pct in GRID_CELLS:
        summary_lines.append(
            f"print(f\"{{'{pct}%':>10}} | {{len(real_history_{label}):>7}} | "
            f"{{n_completed_{label}:>10}} | {{n_aborted_{label}:>8}} | "
            f"{{len(attacker_ever_rejected_{label}):>20}} | {{len(attacker_ever_accepted_{label}):>20}}\")"
        )
    cells.append(("code", "\n".join(summary_lines) + "\n"))

    return nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "RUN_MVP_A1_FRACTION_GRID.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_grid_notebook(), f, indent=1)
    print(f"wrote {out_path}")
