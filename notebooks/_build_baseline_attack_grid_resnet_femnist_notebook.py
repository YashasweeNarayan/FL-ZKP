"""Generates RUN_MVP_BASELINE_ATTACK_GRID_RESNET_FEMNIST.ipynb: the
`small_resnet`+FEMNIST combo of brutal-review gap #4, requested
2026-11-27 ("now give me the 5 notebooks of resnet + FMNIST ... i
guess this is the final run") -- this closes the LAST of the three
combos the gap-#4 brainstorm originally flagged beyond the
already-committed tiny_cnn+MNIST grid (tiny_cnn+FEMNIST closed
2026-11-22, small_resnet+MNIST closed 2026-11-25, this file closes
small_resnet+FEMNIST). After this, all 2 models x 2 datasets = 4
combos have a built grid notebook -- no scoping caveat left to state
for gap #4's own combo coverage.

**Mechanically this is the union of the other two grids' own
differences from the base MNIST+tiny_cnn grid**: `model="small_resnet"`
(~263K params, from the resnet-mnist grid) AND `case2_data`/FEMNIST's
`unseen_test` `FemnistClientDataset` dataclass unpack (`.images`/
`.labels`, NOT a plain tuple -- from the FEMNIST grid), together in one
notebook. Same 4 attacks (sign-flip, Gaussian noise, scale-to-threshold,
low-and-slow), same 3 fractions (10/20/30%), same 20 rounds/cell, same
\u00a76.2 traceability wiring, same `cheater_delta_transform` hook,
imported unchanged from the MNIST tiny_cnn grid module so all four
grids can never drift on what attack or fraction means.

A dedicated `colab-checkpoints-case2-small_resnet-baselinegrid-*`
branch/clone-dir prefix per cell and a seed offset of `+2500 + i`
(MNIST tiny_cnn grid: +1000..+1011, FEMNIST tiny_cnn grid:
+1500..+1511, resnet+MNIST grid: +2000..+2011, this grid: +2500..+2511)
so none of these 12 cells can ever collide with any of the other three
grids' own 12 branches or seeds, even though all four notebooks share
the same `CFG['seed']` base.

**Split into 5 parts**, same mechanical reasoning and same 3/3/2/2/2
split by original grid index as all three prior grids' split modules
use -- see `_build_baseline_attack_grid_split_notebooks.py`'s own
module docstring for why. The personalized (real-PAT,
`/kaggle/working`) copies are built by a separate step, never
committed to this repo -- same live-credential-handling rule as every
prior personalized copy this project has used.

**Cost note, stated plainly**: this is the most expensive of the four
grids to actually run (biggest model x non-IID FEMNIST x 12 cells x 20
rounds x fresh DKG per cell) -- building and committing the notebooks
here does not mean this grid has been executed on Kaggle yet. No real
run results exist for this combo as of this commit; FIX.md/STATUS.md
say so.
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
# definitions -- never redefine these independently, so none of the
# four grids can drift.
ATTACKS = mnist_grid.ATTACKS
FRACTIONS = mnist_grid.FRACTIONS
REAL_ROUNDS = mnist_grid.REAL_ROUNDS
GRID_CELLS = mnist_grid.GRID_CELLS
_TRANSFORM_EXPR = mnist_grid._TRANSFORM_EXPR
_FRACTION_PCTS = mnist_grid._FRACTION_PCTS
_FRACTION_COUNTS = mnist_grid._FRACTION_COUNTS
_ATTACK_LIST_MD = mnist_grid._ATTACK_LIST_MD

CASE2_RESNET_CFG = {
    "case_prefix": "case2",
    "model": "small_resnet",
    "dataset_label": "FEMNIST",
    "num_classes": 62,
    "model_note": "~263K params (WhiteCloak's own CIFAR-10-L ResNet scale, see build_small_resnet's docstring) on writer-disjoint non-IID FEMNIST -- the last of the four model x dataset combos for brutal-review gap #4.",
    "ckpt_branch": "colab-checkpoints-case2-small_resnet-baselinegrid",
}

TITLE_MD = f"""\
# Baseline-literature attack grid -- small_resnet+FEMNIST: 4 attacks x {_FRACTION_PCTS}, {REAL_ROUNDS} rounds each, with \u00a76.2 traceability

Closes the LAST remaining combo of brutal-review gap #4 (the grid
originally covered ONE combo, tiny_cnn+MNIST; tiny_cnn+FEMNIST closed
2026-11-22; small_resnet+MNIST closed 2026-11-25; this is the third
and final one). This notebook is the SAME 12-cell grid as all three
already-committed grids, run again on `small_resnet`+FEMNIST (writer-
disjoint non-IID split) instead -- same attacks, same fractions, same
round count, same \u00a76.2 traceability wiring. Nothing about what's
being tested changes; only the model AND the dataset both change at
once, since this is the one combo neither prior grid touched.

**All four model x dataset combos are now covered by a built grid
notebook** (tiny_cnn+MNIST, tiny_cnn+FEMNIST, small_resnet+MNIST,
small_resnet+FEMNIST). This notebook being built does NOT mean it has
been run yet -- FIX.md/STATUS.md say plainly whether a real Kaggle
result exists for this combo at any given time; check there, not here,
before citing a number from this combo in the paper.

**4 attacks x 3 fractions = 12 cells, in run order:**
{_ATTACK_LIST_MD}

Each attack cell runs at {_FRACTION_PCTS} malicious of 50 clients
({_FRACTION_COUNTS}), `small_resnet`+FEMNIST (writer-disjoint non-IID
split, same `case2_data` as the tiny_cnn FEMNIST grid).

**{REAL_ROUNDS} rounds per cell -- same low-and-slow caveat as every
prior grid.** Sign-flip/noise/scale-to-threshold's claim is decided
within a handful of rounds; low-and-slow's claim is an EPOCH-length
question (`configs/mvp_3day.yaml`'s `protocol.epoch_rounds: 25`) and
{REAL_ROUNDS} rounds is not long enough to claim it survives a full
epoch -- flagged again in this notebook's own summary cell.

**The most expensive of the four grids to run**: `small_resnet`
(~263K params) on non-IID FEMNIST -- expect the longest per-round
wall-clock of any of the four, on top of the usual 12-cells x
20-rounds x fresh-DKG-per-cell cost. Split into 5 parts below for
parallel sessions, same split points as every prior grid uses.

**Checkpoints**: each of the 12 cells gets its own dedicated git
branch (`{CASE2_RESNET_CFG['ckpt_branch']}-<attack>-<fraction>`) and
its own clone directory -- distinct from all three prior grids' own 12
branches each, so none of the four grids can ever collide.
"""


def baseline_grid_run_cell_resnet_femnist(attack_key, attack_label, frac_label, n_cheaters, pct, seed_offset):
    slug = f"{attack_key}_{frac_label}"
    transform_expr = _TRANSFORM_EXPR[attack_key]
    return f"""\
# Baseline-attack grid cell (small_resnet+FEMNIST): {attack_label} @ {pct}%
# malicious ({n_cheaters} of 50 clients), {REAL_ROUNDS} rounds, \u00a76.2
# traceability wired (opening handler cell above). Same
# cheater_delta_transform hook as every prior grid -- the model AND
# the dataset both differ here (the one combo neither prior grid used).
import numpy as np
import torch
from fl.round_loop import run_training_loop, bootstrap_norms_from_model_dim, release_gpu_memory, load_checkpoint
from fl.models import unflatten_to_state_dict
from attacks.baseline_norm_attacks import (
    craft_sign_flip, craft_gaussian_noise, craft_scale_to_threshold, craft_low_and_slow,
)

REAL_ROUNDS = {REAL_ROUNDS}
CHEATER_CLIENT_IDX = list(range({n_cheaters}))

CKPT_PATH = f'{{CKPT_DIR}}/case2_small_resnet_baselinegrid_{slug}_r{{REAL_ROUNDS}}_nodp_f20.pt'

root_norms = bootstrap_norms_from_model_dim(
    model_dim=dim,
    sketch_dim=CFG['sketch']['m'],
    sketch_seed="bootstrap-small_resnet-case2-baselinegrid-{slug}".encode(),
    quant_bits=16,
    rng=np.random.default_rng(CFG['seed']),
)

# Deterministic per-cell RNG for the attack's own randomness -- same
# offset scheme as every prior grid, shifted by +2500 (MNIST tiny_cnn
# grid: +1000..+1011, FEMNIST tiny_cnn grid: +1500..+1511, resnet+MNIST
# grid: +2000..+2011) so no (attack, fraction) cell's seed ever
# collides across any of the four grids even though all four notebooks
# use the same CFG['seed'] base.
_attack_rng = np.random.default_rng(CFG['seed'] + {seed_offset})
cheater_delta_transform = {transform_expr}

print("="*70)
print(f"[resnet-femnist-{slug}] MALICIOUS CLIENTS: {{CHEATER_CLIENT_IDX}} ({{len(CHEATER_CLIENT_IDX)}} of {{n_clients}} = {{100*len(CHEATER_CLIENT_IDX)/n_clients:.0f}}%), attack={attack_label!r}")
print("="*70)

resume_from = load_checkpoint(CKPT_PATH) if os.path.exists(CKPT_PATH) else None
if resume_from is not None:
    print(f"[resnet-femnist-{slug}] found checkpoint at round {{resume_from['round']}} -- resuming, not restarting from round 0")

# FEMNIST's headline split (unseen_test) is a FemnistClientDataset
# dataclass (.images/.labels), NOT a plain tuple -- same unpack pattern
# as the tiny_cnn FEMNIST grid, _build_mvp_run_notebooks.py's own
# case2 run cells, and every other FEMNIST notebook in this repo.
_eval_ds = case2_data['unseen_test']
_eval_images, _eval_labels = _eval_ds.images, _eval_ds.labels


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
    client_datasets=case2_data['client_train'],
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
        print(f"[resnet-femnist-{slug}]   round {{h.get('round')}}: ABORTED -- {{h.get('reason')}}")
        continue
    reasons = h.get("rejected_reasons", {{}})
    accepted_idxs = set(h.get("accepted_client_indices", []))
    rejected_this_round = sorted(i for i in CHEATER_CLIENT_IDX if i in reasons)
    accepted_this_round = sorted(i for i in CHEATER_CLIENT_IDX if i in accepted_idxs)
    for i in rejected_this_round:
        attacker_ever_rejected[i] = True
    for i in accepted_this_round:
        attacker_ever_accepted[i] = True
    print(f"[resnet-femnist-{slug}]   round {{h.get('round')}}: accepted={{h.get('accepted')}}, "
          f"attackers rejected={{rejected_this_round}}, attackers accepted={{accepted_this_round}}, "
          f"aggregate_exact={{h.get('aggregate_exact')}}")

print(f"\\n[resnet-femnist-{slug}] {{n_completed}} of {{len(real_history)}} rounds completed, {{n_aborted}} aborted.")
if attacker_ever_accepted:
    print(f"[resnet-femnist-{slug}] V3 FAIL: attacker(s) {{sorted(attacker_ever_accepted)}} had a crafted update accepted at least once.")
elif attacker_ever_rejected:
    print(f"[resnet-femnist-{slug}] V3 PASS: every eligible attacker ({{sorted(attacker_ever_rejected)}}) was rejected by V3 every time it submitted.")
else:
    print(f"[resnet-femnist-{slug}] INCONCLUSIVE: no attacker was ever sampled/eligible in any round.")

print(f"\\n[resnet-femnist-{slug}] \u00a76 traceability summary:")
if traceability_log:
    for e in traceability_log:
        if e["opened"]:
            print(f"[resnet-femnist-{slug}]   round {{e['round']}}: pseudonym {{e['pseudonym'][:12]}}... OPENED "
                  f"(evidence={{e['evidence_total']:.3f}} > theta={{e['theta']:.3f}}), "
                  f"H(T)={{e['residual_anonymity_bits']:.3f}} bits")
        else:
            print(f"[resnet-femnist-{slug}]   round {{e['round']}}: pseudonym {{e['pseudonym'][:12]}}... flagged, NOT opened "
                  f"({{e['reason']}})")
    print(f"[resnet-femnist-{slug}]   {{_n_opened_so_far}} identity(ies) opened across {{len(traceability_log)}} flagged pseudonym(s) this cell.")
else:
    print(f"[resnet-femnist-{slug}]   no pseudonym ever crossed theta in these {{REAL_ROUNDS}} rounds.")

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
             ("code", gen.dataset_cell(CASE2_RESNET_CFG))]

    for i, (attack_key, attack_label, frac_label, n_cheaters, pct) in enumerate(GRID_CELLS):
        slug = f"{attack_key}_{frac_label}"
        new_branch = f"{CASE2_RESNET_CFG['ckpt_branch']}-{attack_key}-{frac_label}"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-baselinegrid-resnet-femnist-{i}"
        cells.append(("md", f"## Grid cell {i + 1} of {len(GRID_CELLS)} (small_resnet+FEMNIST): {attack_label} @ {pct}% malicious ({n_cheaters} of 50 clients)"))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.OPENING_HANDLER_CELL))
        cells.append(("code", gen.model_cell(CASE2_RESNET_CFG)))
        cells.append(("md", gen.CHECKPOINT_MD if i == 0 else
                      "## Checkpoint durability for this cell\n\nSame git-backed checkpointing as the first cell -- this cell gets its own clone directory and its own dedicated branch, so it can never collide with another cell's checkpoint in this same session, with any of the other three grids' branches, or with any other notebook in this repo."))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", baseline_grid_run_cell_resnet_femnist(attack_key, attack_label, frac_label, n_cheaters, pct, seed_offset=2500 + i)))

    summary_lines = [
        "print(\"\\n\" + \"=\"*70)",
        "print(\"BASELINE-ATTACK GRID SUMMARY (small_resnet+FEMNIST) -- V3 + \\u00a76 traceability, by attack and fraction\")",
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
        f"one -- same caveat as every prior grid. This is the LAST of the "
        f"four model x dataset combos for brutal-review gap #4 -- after "
        f"this one, tiny_cnn+MNIST, tiny_cnn+FEMNIST, small_resnet+MNIST "
        f"and small_resnet+FEMNIST all have a built grid notebook.\")"
    )
    cells.append(("code", "\n".join(summary_lines) + "\n"))

    return nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "RUN_MVP_BASELINE_ATTACK_GRID_RESNET_FEMNIST.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_grid_notebook(), f, indent=1)
    print(f"wrote {out_path}")
