"""Generates RUN_MVP_BASELINE_ATTACK_GRID.ipynb: the combined grid the
user asked for (2026-11-20) after the benchmarked-literature search --
sign-flip, Gaussian noise, scale-to-threshold, and low-and-slow
(`attacks/baseline_norm_attacks.py`) x {10%, 20%, 30%} malicious of 50
clients, tiny_cnn+MNIST (same base case as the A1 fraction grid and
every attack7 notebook), WITH \u00a76.2 traceability wired in -- the
one piece the already-committed `RUN_MVP_A1_FRACTION_GRID.ipynb`
deliberately left out (that grid is A1-only, no traceability; see its
own generator's docstring for why).

**Does NOT replace `RUN_MVP_A1_FRACTION_GRID.ipynb`** -- that notebook
is already a finished, confirmed deliverable for A1 specifically (its
own checkpoint branches are real, committed results). This is an
ADDITIONAL notebook, covering the 4 new baseline attacks the A1 grid
was never meant to cover, consistent with this project's "never
delete, always add a new branch/file" convention and the user's own
"keep all existing results" decision earlier in this session.

**Why these 4, and not A1 again, and not LIE/Min-Max/Min-Sum**: see
`attacks/baseline_norm_attacks.py`'s own module docstring for the full
citations and reasoning (Shi et al. 2021 for sign-flip/noise; Lycklama
et al. 2025 "Hammer and Anvil" for why scale-to-threshold -- scaling
to just UNDER a norm bound, not wildly past it like A1's gamma=100 --
is the realistic version of a scaling attack; LIE/Min-Max/Min-Sum
excluded because they need visibility into other clients' real raw
gradients, which this project's own secure-aggregation threat model
specifically denies).

**12 cells (4 attacks x 3 fractions)**, each on its own dedicated
checkpoint branch (`colab-checkpoints-case1-tiny_cnn-baselinegrid-
<attack>-<fraction>`) and clone directory -- same mechanical reason as
every other multi-cell notebook in this repo (`git_checkpoint_cell`'s
branch-checkout logic only has a "clone+checkout" or "reuse exactly
what's checked out" branch, never "same clone, different branch").

**20 rounds per cell** (2026-11-20, raised from an initial 10 --
user asked to match the 20-round length every attack7 notebook
already uses, rather than the A1 fraction grid's shorter 5). Sign-
flip/noise/scale-to-threshold's claim ("does V3 catch this, and/or
does it get traced") is decided within a handful of rounds regardless,
so this mainly brings the new grid's numbers onto the same footing as
the existing attack7 results. Low-and-slow's claim is different -- it
is SPECIFICALLY a patience attack staying under both V3's per-round
bound and the accumulating \u00a76.1 evidence threshold theta, and
whether it ever crosses theta is an EPOCH-length question
(`configs/mvp_3day.yaml`'s `protocol.epoch_rounds: 25`). 20 rounds is
closer to that epoch length than the original 10 was, but still one
short of it -- **this is flagged explicitly in this notebook's own
summary cell and in FIX.md, not glossed over**: a reviewer-safe
low-and-slow result needs a run of at least `epoch_rounds=25` for
that one cell specifically to see a complete epoch's worth of
accumulation, not just "close to it."

Reuses every shared cell-generator function from
`notebooks/_build_mvp_run_notebooks.py` exactly like both the A1 grid
and the attack7-traceability combo notebook do (clone/install/sanity/
config/dataset/model/git-checkpoint cells, plus RING_CELL +
OPENING_HANDLER_CELL for real \u00a76.2 opening against a real DKG
committee) -- nothing here reimplements a cell either of those
notebooks already owns.
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
    "model_note": "~20.6K params, same model/dataset as LEFT.md item #1, the A1 fraction grid, and every attack7 notebook's base case.",
    "ckpt_branch": "colab-checkpoints-case1-tiny_cnn-baselinegrid",
}

# (attack_key, display label, one-line note for the title markdown)
ATTACKS = [
    ("signflip", "Sign-flip", "g_m = -g_b -- Shi et al. 2021. Expected to be caught easily by V3 (a negated real delta is usually still a large-norm outlier); included for completeness."),
    ("noise", "Gaussian noise", "g_m = g_b + N(0, (0.5*B_t)^2 I) -- Shi et al. 2021's Noise Attack. sigma scaled to half this round's active norm bound so the perturbation is comparably sized across rounds/cells rather than a fixed constant that means something different depending on the real delta's own magnitude."),
    ("scale_to_threshold", "Scale-to-threshold", "rescale the real delta to 0.95*B_t, just under this round's R_norm bound -- Lycklama et al. 2025 'Hammer and Anvil': the realistic version of a scaling attack, unlike A1's deliberately-naive gamma=100 which overshoots the bound so far V3 catches it trivially."),
    ("low_and_slow", "Low-and-slow", "rescale the real delta to 0.3*B_t, well under the bound -- this project's own extension, testing whether staying small evades BOTH V3 and the accumulating \u00a76.1 evidence threshold through patience rather than size. See this file's own module docstring for why 20 rounds is not a definitive answer for this one attack specifically."),
]

# (label, n_cheaters, pct) -- 10%/20%/30% of 50 clients, same as the A1 grid.
FRACTIONS = [("10pct", 5, 10), ("20pct", 10, 20), ("30pct", 15, 30)]
REAL_ROUNDS = 20

GRID_CELLS = [
    (attack_key, attack_label, frac_label, n_cheaters, pct)
    for attack_key, attack_label, _ in ATTACKS
    for frac_label, n_cheaters, pct in FRACTIONS
]

_TRANSFORM_EXPR = {
    "signflip": "lambda d, bt: craft_sign_flip(d)",
    "noise": "lambda d, bt: craft_gaussian_noise(d, sigma=0.5 * bt, rng=_attack_rng)",
    "scale_to_threshold": "lambda d, bt: craft_scale_to_threshold(d, bt, rng=_attack_rng)",
    "low_and_slow": "lambda d, bt: craft_low_and_slow(d, bt, rng=_attack_rng)",
}

_FRACTION_PCTS = ", ".join(f"{pct}%" for _, _, pct in FRACTIONS)
_FRACTION_COUNTS = ", ".join(f"{n} of 50" for _, n, _ in FRACTIONS)
_ATTACK_LIST_MD = chr(10).join(f"- **{label}**: {note}" for _, label, note in ATTACKS)

TITLE_MD = f"""\
# Baseline-literature attack grid: 4 attacks x {_FRACTION_PCTS}, {REAL_ROUNDS} rounds each, with \u00a76.2 traceability

New benchmarked baseline attacks (`attacks/baseline_norm_attacks.py`,
2026-11-20), added after a rigorous web search for recent Byzantine-FL
benchmark attacks a reviewer familiar with that literature would
expect to see -- see this notebook's generator script
(`notebooks/_build_baseline_attack_grid_notebook.py`) for the full
citations, exclusion reasoning (true LIE/Min-Max/Min-Sum are
deliberately NOT here -- they'd require breaking this project's own
secure-aggregation privacy guarantee), and why these use a NEW
`cheater_delta_transform` hook rather than A1's existing
`cheater_force_delta` (A1 replaces the real delta outright; these
transform it).

**4 attacks x 3 fractions = 12 cells, in run order:**
{_ATTACK_LIST_MD}

Each attack cell runs at {_FRACTION_PCTS} malicious of 50 clients
({_FRACTION_COUNTS}), same `tiny_cnn`+MNIST base case as the A1
fraction grid and every attack7 notebook.

**This notebook does NOT replace `RUN_MVP_A1_FRACTION_GRID.ipynb`**
-- that one is a separate, already-finished, already-committed A1-only
result; this is an additional notebook covering the 4 new attacks.

**\u00a76.2 traceability wired in** (unlike the A1 fraction grid, which
deliberately left it out as out of scope for a pure V3 defense-
trigger-rate table): each cell builds its own real DKG committee
(`RING_CELL`) and real opening-protocol handler (`OPENING_HANDLER_CELL`),
exactly like the attack7-traceability combo notebook -- a crossing of
the \u00a76.1 evidence threshold theta triggers a real identity opening
against real escrowed ciphertexts, not a simulated one.

**{REAL_ROUNDS} rounds per cell -- read this before trusting the
low-and-slow numbers specifically.** Sign-flip/noise/scale-to-threshold's
claim (does V3 reject it, does it get traced) is decided within a
handful of rounds. Low-and-slow is a DIFFERENT kind of claim -- it is
designed to stay under both V3's bound and the accumulating \u00a76.1
evidence threshold theta through patience, and whether it ever crosses
theta is inherently an EPOCH-length question
(`configs/mvp_3day.yaml`'s `protocol.epoch_rounds: 25`). {REAL_ROUNDS}
rounds shows the evidence accumulating, but is NOT long enough to
claim low-and-slow survives a full epoch -- that needs a dedicated,
longer (>=25-round) run of just that cell, not done here under
deadline pressure. This is flagged again in the final summary cell
below, not just here.

**Expect more ABORTED rounds at higher fractions**, same sampling-model
reason as the A1 grid (malicious clients excluded from `k_min`-eligibility):
~9.4%/~18.2%/~32.2% predicted abort probability at 10%/20%/30%.

**Checkpoints**: each of the 12 cells gets its own dedicated git branch
(`{CASE1_CFG['ckpt_branch']}-<attack>-<fraction>`) and its own clone
directory -- none can collide with each other, with the A1 fraction
grid's branches, or with any other notebook in this repo.

**Runtime**: {REAL_ROUNDS} rounds x 12 cells, each including fresh
DKG/ring setup -- expect noticeably longer than the A1 grid's 3 cells,
plan for several hours total.
"""


def baseline_grid_run_cell(attack_key, attack_label, frac_label, n_cheaters, pct, seed_offset):
    slug = f"{attack_key}_{frac_label}"
    transform_expr = _TRANSFORM_EXPR[attack_key]
    return f"""\
# Baseline-attack grid cell: {attack_label} @ {pct}% malicious
# ({n_cheaters} of 50 clients), {REAL_ROUNDS} rounds, \u00a76.2
# traceability wired (opening handler cell above). Unlike A1's
# cheater_force_delta (replaces the real delta outright), this attack
# TRANSFORMS each attacker's own real Step-1 output via the new
# cheater_delta_transform hook (fl/round_loop.py, 2026-11-20) -- see
# attacks/baseline_norm_attacks.py for the craft function itself.
import numpy as np
import torch
from fl.round_loop import run_training_loop, bootstrap_norms_from_model_dim, release_gpu_memory, load_checkpoint
from attacks.baseline_norm_attacks import (
    craft_sign_flip, craft_gaussian_noise, craft_scale_to_threshold, craft_low_and_slow,
)

REAL_ROUNDS = {REAL_ROUNDS}
CHEATER_CLIENT_IDX = list(range({n_cheaters}))

CKPT_PATH = f'{{CKPT_DIR}}/case1_tiny_cnn_baselinegrid_{slug}_r{{REAL_ROUNDS}}_nodp_f20.pt'

root_norms = bootstrap_norms_from_model_dim(
    model_dim=dim,
    sketch_dim=CFG['sketch']['m'],
    sketch_seed="bootstrap-tiny_cnn-case1-baselinegrid-{slug}".encode(),
    quant_bits=16,
    rng=np.random.default_rng(CFG['seed']),
)

# Deterministic per-cell RNG for the attack's own randomness (Gaussian
# noise draws; the zero-real-delta fallback direction in
# craft_scale_to_threshold/craft_low_and_slow) -- NOT the same stream
# as root_norms' bootstrap RNG above, so one doesn't perturb the other.
_attack_rng = np.random.default_rng(CFG['seed'] + {seed_offset})
cheater_delta_transform = {transform_expr}

print("="*70)
print(f"[{slug}] MALICIOUS CLIENTS: {{CHEATER_CLIENT_IDX}} ({{len(CHEATER_CLIENT_IDX)}} of {{n_clients}} = {{100*len(CHEATER_CLIENT_IDX)/n_clients:.0f}}%), attack={attack_label!r}")
print("="*70)

resume_from = load_checkpoint(CKPT_PATH) if os.path.exists(CKPT_PATH) else None
if resume_from is not None:
    print(f"[{slug}] found checkpoint at round {{resume_from['round']}} -- resuming, not restarting from round 0")

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
        print(f"[{slug}]   round {{h.get('round')}}: ABORTED -- {{h.get('reason')}}")
        continue
    reasons = h.get("rejected_reasons", {{}})
    accepted_idxs = set(h.get("accepted_client_indices", []))
    rejected_this_round = sorted(i for i in CHEATER_CLIENT_IDX if i in reasons)
    accepted_this_round = sorted(i for i in CHEATER_CLIENT_IDX if i in accepted_idxs)
    for i in rejected_this_round:
        attacker_ever_rejected[i] = True
    for i in accepted_this_round:
        attacker_ever_accepted[i] = True
    print(f"[{slug}]   round {{h.get('round')}}: accepted={{h.get('accepted')}}, "
          f"attackers rejected={{rejected_this_round}}, attackers accepted={{accepted_this_round}}, "
          f"aggregate_exact={{h.get('aggregate_exact')}}")

print(f"\\n[{slug}] {{n_completed}} of {{len(real_history)}} rounds completed, {{n_aborted}} aborted.")
if attacker_ever_accepted:
    print(f"[{slug}] V3 FAIL: attacker(s) {{sorted(attacker_ever_accepted)}} had a crafted update accepted at least once.")
elif attacker_ever_rejected:
    print(f"[{slug}] V3 PASS: every eligible attacker ({{sorted(attacker_ever_rejected)}}) was rejected by V3 every time it submitted.")
else:
    print(f"[{slug}] INCONCLUSIVE: no attacker was ever sampled/eligible in any round.")

# \u00a76 traceability summary for this cell, built from traceability_log
# (opening-handler cell above) -- preserved under a slug-specific name
# right after, before the next cell's RING_CELL/OPENING_HANDLER_CELL
# rebuild resets traceability_log to a fresh, empty list.
print(f"\\n[{slug}] \u00a76 traceability summary:")
if traceability_log:
    for e in traceability_log:
        if e["opened"]:
            print(f"[{slug}]   round {{e['round']}}: pseudonym {{e['pseudonym'][:12]}}... OPENED "
                  f"(evidence={{e['evidence_total']:.3f}} > theta={{e['theta']:.3f}}), "
                  f"H(T)={{e['residual_anonymity_bits']:.3f}} bits")
        else:
            print(f"[{slug}]   round {{e['round']}}: pseudonym {{e['pseudonym'][:12]}}... flagged, NOT opened "
                  f"({{e['reason']}})")
    print(f"[{slug}]   {{_n_opened_so_far}} identity(ies) opened across {{len(traceability_log)}} flagged pseudonym(s) this cell.")
else:
    print(f"[{slug}]   no pseudonym ever crossed theta in these {{REAL_ROUNDS}} rounds.")

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
             ("code", gen.dataset_cell(CASE1_CFG))]

    for i, (attack_key, attack_label, frac_label, n_cheaters, pct) in enumerate(GRID_CELLS):
        slug = f"{attack_key}_{frac_label}"
        new_branch = f"{CASE1_CFG['ckpt_branch']}-{attack_key}-{frac_label}"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-baselinegrid-{i}"
        cells.append(("md", f"## Grid cell {i + 1} of {len(GRID_CELLS)}: {attack_label} @ {pct}% malicious ({n_cheaters} of 50 clients)"))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.OPENING_HANDLER_CELL))
        cells.append(("code", gen.model_cell(CASE1_CFG)))
        cells.append(("md", gen.CHECKPOINT_MD if i == 0 else
                      "## Checkpoint durability for this cell\n\nSame git-backed checkpointing as the first cell -- this cell gets its own clone directory and its own dedicated branch, so it can never collide with another cell's checkpoint in this same session."))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", baseline_grid_run_cell(attack_key, attack_label, frac_label, n_cheaters, pct, seed_offset=1000 + i)))

    # Final combined summary across all 12 cells: V3-rejection rate AND
    # traceability outcome, side by side, per (attack, fraction).
    summary_lines = [
        "print(\"\\n\" + \"=\"*70)",
        "print(\"BASELINE-ATTACK GRID SUMMARY -- V3 + \\u00a76 traceability, by attack and fraction\")",
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
        f"one -- see this notebook's own title markdown for why that attack "
        f"specifically needs a longer, dedicated run before its result is "
        f"reviewer-safe; the other 3 attacks' claims do not depend on run "
        f"length the same way.\")"
    )
    cells.append(("code", "\n".join(summary_lines) + "\n"))

    return nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "RUN_MVP_BASELINE_ATTACK_GRID.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_grid_notebook(), f, indent=1)
    print(f"wrote {out_path}")
