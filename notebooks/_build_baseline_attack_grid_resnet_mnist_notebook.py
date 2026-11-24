"""Generates RUN_MVP_BASELINE_ATTACK_GRID_RESNET_MNIST.ipynb: the
`small_resnet`+MNIST combo of brutal-review gap #4, requested
2026-11-25 ("now give me resnet mnist in 5 splits with the checkpoints
with TOKEN ID attacked to it") as a follow-on to the already-committed
tiny_cnn+FEMNIST combo (2026-11-22).

**Scope note, same honesty as the FEMNIST module's own**: the gap-#4
brainstorm originally flagged THREE remaining combos beyond the
already-committed tiny_cnn+MNIST grid -- tiny_cnn+FEMNIST (closed
2026-11-22), small_resnet+MNIST (this file), and small_resnet+FEMNIST
(still open, not addressed here). This file closes the second of the
three. small_resnet+FEMNIST remains unaddressed -- stated plainly, not
an oversight.

**Everything else is identical in kind to the already-committed
tiny_cnn+MNIST and tiny_cnn+FEMNIST grids** -- same 4 attacks
(sign-flip, Gaussian noise, scale-to-threshold, low-and-slow), same 3
fractions (10/20/30%), same 20 rounds/cell, same \u00a76.2 traceability
wiring, same `cheater_delta_transform` hook. This file imports the
shared ATTACKS/FRACTIONS/GRID_CELLS/_TRANSFORM_EXPR definitions from
the MNIST tiny_cnn grid module rather than redefining them, exactly
like the FEMNIST module already does, so all three grids can never
drift out of sync on what attack or fraction means.

**What's different from the MNIST tiny_cnn grid**: `model="small_resnet"`
(~263K params, WhiteCloak's own second reported model scale -- see
`fl/models.py::build_small_resnet`'s docstring) in place of `tiny_cnn`
(~20.6K); same `case1_data` (MNIST, IID, 50 clients) -- unlike the
FEMNIST module, this one does NOT need a different dataset or a
different eval-unpack pattern, since it stays on MNIST and MNIST's
`case1_data['test']` is already a plain `(images, labels)` tuple, same
as the tiny_cnn MNIST grid uses. A dedicated
`colab-checkpoints-case1-small_resnet-baselinegrid-*` branch/clone-dir
prefix per cell, so none of these 12 cells can ever collide with
either already-committed grid's own 12 branches.

**Split into 5 parts**, same mechanical reasoning and same 3/3/2/2/2
split by original grid index as both prior grids' split modules use --
see `_build_baseline_attack_grid_split_notebooks.py`'s own module
docstring for why. The personalized (real-PAT, `/kaggle/working`)
copies are built by a separate step, never committed to this repo --
same live-credential-handling rule as every prior personalized copy
this project has used.
"""
import json
import os
import sys

NB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB_DIR)

import _build_mvp_run_notebooks as gen  # noqa: E402
import _build_baseline_attack_grid_notebook as mnist_grid  # noqa: E402

nb = gen.nb

# Reuse the MNIST tiny_cnn grid's own attack/fraction/transform
# definitions -- never redefine these independently, so all three
# grids can't drift.
ATTACKS = mnist_grid.ATTACKS
FRACTIONS = mnist_grid.FRACTIONS
REAL_ROUNDS = mnist_grid.REAL_ROUNDS
GRID_CELLS = mnist_grid.GRID_CELLS
_TRANSFORM_EXPR = mnist_grid._TRANSFORM_EXPR
_FRACTION_PCTS = mnist_grid._FRACTION_PCTS
_FRACTION_COUNTS = mnist_grid._FRACTION_COUNTS
_ATTACK_LIST_MD = mnist_grid._ATTACK_LIST_MD

CASE1_RESNET_CFG = {
    "case_prefix": "case1",
    "model": "small_resnet",
    "dataset_label": "MNIST",
    "num_classes": 10,
    "model_note": "~263K params (WhiteCloak's own CIFAR-10-L ResNet scale, see build_small_resnet's docstring) -- same MNIST dataset as the already-committed tiny_cnn grid; only the model differs, per brutal-review gap #4's own scoping.",
    "ckpt_branch": "colab-checkpoints-case1-small_resnet-baselinegrid",
}

TITLE_MD = f"""\
# Baseline-literature attack grid -- small_resnet+MNIST: 4 attacks x {_FRACTION_PCTS}, {REAL_ROUNDS} rounds each, with \u00a76.2 traceability

Closes the `small_resnet+MNIST` combo of brutal-review gap #4 (the
grid originally covered ONE combo, tiny_cnn+MNIST; tiny_cnn+FEMNIST
closed 2026-11-22; this is the second of the three remaining combos).
This notebook is the SAME 12-cell grid as both already-committed
grids, run again on `small_resnet`+MNIST instead -- same attacks, same
fractions, same round count, same \u00a76.2 traceability wiring.
Nothing about what's being tested changes; only the model does.

**Deliberately scoped to this ONE additional combo, not
small_resnet+FEMNIST** (the one remaining combo): per the same "least
time" scoping already used for the FEMNIST grid, one more combo at a
time is enough to show the grid generalizes across model architecture
too, not just dataset -- it is NOT a claim that small_resnet+FEMNIST
was checked too. Stated here plainly, not glossed over.

**4 attacks x 3 fractions = 12 cells, in run order:**
{_ATTACK_LIST_MD}

Each attack cell runs at {_FRACTION_PCTS} malicious of 50 clients
({_FRACTION_COUNTS}), `small_resnet`+MNIST (IID split, same
`case1_data` as the tiny_cnn MNIST grid).

**{REAL_ROUNDS} rounds per cell -- same low-and-slow caveat as both
prior grids.** Sign-flip/noise/scale-to-threshold's claim is decided
within a handful of rounds; low-and-slow's claim is an EPOCH-length
question (`configs/mvp_3day.yaml`'s `protocol.epoch_rounds: 25`) and
{REAL_ROUNDS} rounds is not long enough to claim it survives a full
epoch -- flagged again in this notebook's own summary cell.

**A bigger model than either prior grid**: `small_resnet` is ~263K
params vs. `tiny_cnn`'s ~20.6K -- expect noticeably longer per-round
wall-clock than either tiny_cnn grid, on top of the usual 12-cells x
20-rounds x fresh-DKG-per-cell cost.

**Checkpoints**: each of the 12 cells gets its own dedicated git
branch (`{CASE1_RESNET_CFG['ckpt_branch']}-<attack>-<fraction>`) and
its own clone directory -- distinct from both prior grids' own 12
branches each, so none of the three grids can ever collide.

**Runtime**: longer than either tiny_cnn grid per cell (bigger model);
split into 5 parts below for parallel sessions, same split points as
both prior grids use.
"""


def baseline_grid_run_cell_resnet_mnist(attack_key, attack_label, frac_label, n_cheaters, pct, seed_offset):
    slug = f"{attack_key}_{frac_label}"
    transform_expr = _TRANSFORM_EXPR[attack_key]
    return f"""\
# Baseline-attack grid cell (small_resnet+MNIST): {attack_label} @ {pct}%
# malicious ({n_cheaters} of 50 clients), {REAL_ROUNDS} rounds, \u00a76.2
# traceability wired (opening handler cell above). Same
# cheater_delta_transform hook as both prior grids -- only the model
# differs here.
import numpy as np
import torch
from fl.round_loop import run_training_loop, bootstrap_norms_from_model_dim, release_gpu_memory, load_checkpoint
from fl.models import unflatten_to_state_dict
from attacks.baseline_norm_attacks import (
    craft_sign_flip, craft_gaussian_noise, craft_scale_to_threshold, craft_low_and_slow,
)

REAL_ROUNDS = {REAL_ROUNDS}
CHEATER_CLIENT_IDX = list(range({n_cheaters}))

CKPT_PATH = f'{{CKPT_DIR}}/case1_small_resnet_baselinegrid_{slug}_r{{REAL_ROUNDS}}_nodp_f20.pt'

root_norms = bootstrap_norms_from_model_dim(
    model_dim=dim,
    sketch_dim=CFG['sketch']['m'],
    sketch_seed="bootstrap-small_resnet-case1-baselinegrid-{slug}".encode(),
    quant_bits=16,
    rng=np.random.default_rng(CFG['seed']),
)

# Deterministic per-cell RNG for the attack's own randomness -- same
# offset scheme as both prior grids, shifted by +1000 (MNIST tiny_cnn
# grid uses +1000..+1011, FEMNIST uses +1500..+1511) so no (attack,
# fraction) cell's seed ever collides across any of the three grids
# even though all three notebooks use the same CFG['seed'] base.
_attack_rng = np.random.default_rng(CFG['seed'] + {seed_offset})
cheater_delta_transform = {transform_expr}

print("="*70)
print(f"[resnet-mnist-{slug}] MALICIOUS CLIENTS: {{CHEATER_CLIENT_IDX}} ({{len(CHEATER_CLIENT_IDX)}} of {{n_clients}} = {{100*len(CHEATER_CLIENT_IDX)/n_clients:.0f}}%), attack={attack_label!r}")
print("="*70)

resume_from = load_checkpoint(CKPT_PATH) if os.path.exists(CKPT_PATH) else None
if resume_from is not None:
    print(f"[resnet-mnist-{slug}] found checkpoint at round {{resume_from['round']}} -- resuming, not restarting from round 0")

# case1_data['test'] is a plain (images, labels) tuple -- same as the
# tiny_cnn MNIST grid; MNIST's case1 pathway never needed the FEMNIST
# dataclass unpack fix in the first place.
_eval_images, _eval_labels = case1_data['test']


def _eval_fn(round_idx, w_t_now):
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
    cheater_delta_transform=cheater_delta_transform,
    on_checkpoint_saved=push_checkpoint_to_git,
    eval_fn=_eval_fn,
    on_pseudonym_flagged=on_pseudonym_flagged,
)
release_gpu_memory()

n_aborted = sum(1 for h in real_history if h.get('aborted'))
n_completed = len(real_history) - n_aborted
attacker_ever_accepted = {{}}
attacker_ever_rejected = {{}}
for h in real_history:
    if h.get("aborted"):
        print(f"[resnet-mnist-{slug}]   round {{h.get('round')}}: ABORTED -- {{h.get('reason')}}")
        continue
    reasons = h.get("rejected_reasons", {{}})
    accepted_idxs = set(h.get("accepted_client_indices", []))
    rejected_this_round = sorted(i for i in CHEATER_CLIENT_IDX if i in reasons)
    accepted_this_round = sorted(i for i in CHEATER_CLIENT_IDX if i in accepted_idxs)
    for i in rejected_this_round:
        attacker_ever_rejected[i] = True
    for i in accepted_this_round:
        attacker_ever_accepted[i] = True
    print(f"[resnet-mnist-{slug}]   round {{h.get('round')}}: accepted={{h.get('accepted')}}, "
          f"attackers rejected={{rejected_this_round}}, attackers accepted={{accepted_this_round}}, "
          f"aggregate_exact={{h.get('aggregate_exact')}}")

print(f"\\n[resnet-mnist-{slug}] {{n_completed}} of {{len(real_history)}} rounds completed, {{n_aborted}} aborted.")
if attacker_ever_accepted:
    print(f"[resnet-mnist-{slug}] V3 FAIL: attacker(s) {{sorted(attacker_ever_accepted)}} had a crafted update accepted at least once.")
elif attacker_ever_rejected:
    print(f"[resnet-mnist-{slug}] V3 PASS: every eligible attacker ({{sorted(attacker_ever_rejected)}}) was rejected by V3 every time it submitted.")
else:
    print(f"[resnet-mnist-{slug}] INCONCLUSIVE: no attacker was ever sampled/eligible in any round.")

print(f"\\n[resnet-mnist-{slug}] \u00a76 traceability summary:")
if traceability_log:
    for e in traceability_log:
        if e["opened"]:
            print(f"[resnet-mnist-{slug}]   round {{e['round']}}: pseudonym {{e['pseudonym'][:12]}}... OPENED "
                  f"(evidence={{e['evidence_total']:.3f}} > theta={{e['theta']:.3f}}), "
                  f"H(T)={{e['residual_anonymity_bits']:.3f}} bits")
        else:
            print(f"[resnet-mnist-{slug}]   round {{e['round']}}: pseudonym {{e['pseudonym'][:12]}}... flagged, NOT opened "
                  f"({{e['reason']}})")
    print(f"[resnet-mnist-{slug}]   {{_n_opened_so_far}} identity(ies) opened across {{len(traceability_log)}} flagged pseudonym(s) this cell.")
else:
    print(f"[resnet-mnist-{slug}]   no pseudonym ever crossed theta in these {{REAL_ROUNDS}} rounds.")

real_history_{slug} = list(real_history)
n_aborted_{slug} = n_aborted
n_completed_{slug} = n_completed
attacker_ever_accepted_{slug} = dict(attacker_ever_accepted)
attacker_ever_rejected_{slug} = dict(attacker_ever_rejected)
traceability_log_{slug} = list(traceability_log)
n_opened_{slug} = _n_opened_so_far
"""


def build_grid_notebook():
    cells = [("md", TITLE_MD), ("code", gen.CLONE_CELL), ("code", gen.INSTALL_CELL),
             ("md", gen.RESTART_MD), ("code", gen.SANITY_CELL), ("code", gen.CONFIG_CELL),
             ("code", gen.dataset_cell(CASE1_RESNET_CFG))]

    for i, (attack_key, attack_label, frac_label, n_cheaters, pct) in enumerate(GRID_CELLS):
        slug = f"{attack_key}_{frac_label}"
        new_branch = f"{CASE1_RESNET_CFG['ckpt_branch']}-{attack_key}-{frac_label}"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-baselinegrid-resnet-mnist-{i}"
        cells.append(("md", f"## Grid cell {i + 1} of {len(GRID_CELLS)} (small_resnet+MNIST): {attack_label} @ {pct}% malicious ({n_cheaters} of 50 clients)"))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.OPENING_HANDLER_CELL))
        cells.append(("code", gen.model_cell(CASE1_RESNET_CFG)))
        cells.append(("md", gen.CHECKPOINT_MD if i == 0 else
                      "## Checkpoint durability for this cell\n\nSame git-backed checkpointing as the first cell -- this cell gets its own clone directory and its own dedicated branch, so it can never collide with another cell's checkpoint in this same session, with either prior grid's branches, or with any other notebook in this repo."))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", baseline_grid_run_cell_resnet_mnist(attack_key, attack_label, frac_label, n_cheaters, pct, seed_offset=2000 + i)))

    summary_lines = [
        "print(\"\\n\" + \"=\"*70)",
        "print(\"BASELINE-ATTACK GRID SUMMARY (small_resnet+MNIST) -- V3 + \\u00a76 traceability, by attack and fraction\")",
        "print(\"=\"*70)",
        "print(f\"{'attack':>20} | {'fraction':>8} | {'rounds':>6} | {'aborted':>7} | {'V3 rejected':>11} | {'V3 accepted':>11} | {'traced':>6} | {'opened':>6}\")",
    ]
    for attack_key, attack_label, frac_label, n_cheaters, pct in GRID_CELLS:
        slug = f"{attack_key}_{frac_label}"
        summary_lines.append(
            f"print(f\"{{'{attack_label}':>20}} | {{'{pct}%':>8}} | {{len(real_history_{slug}):>6}} | "
            f"{{n_aborted_{slug}:>7}} | {{len(attacker_ever_rejected_{slug}):>11}} | "
            f"{{len(attacker_ever_accepted_{slug}):>11}} | {{len(traceability_log_{slug}):>6}} | "
            f"{{n_opened_{slug}:>6}}\")"
        )
    summary_lines.append(
        f"print(f\"\\nNOTE: low-and-slow's numbers above are from a "
        f"{REAL_ROUNDS}-round run, not a full epoch_rounds={{CFG['protocol']['epoch_rounds']}}-round "
        f"one -- same caveat as both prior grids. This notebook deliberately "
        f"covers only small_resnet+MNIST, not small_resnet+FEMNIST -- a "
        f"scoping choice (see this notebook's own title markdown), not a "
        f"claim that combo was checked too.\")"
    )
    cells.append(("code", "\n".join(summary_lines) + "\n"))

    return nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "RUN_MVP_BASELINE_ATTACK_GRID_RESNET_MNIST.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_grid_notebook(), f, indent=1)
    print(f"wrote {out_path}")
