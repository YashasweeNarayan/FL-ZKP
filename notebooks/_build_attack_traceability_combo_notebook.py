"""Generates RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb: ONE notebook that
runs all 4 attack7 combos sequentially, in a single Colab session,
each with §6.2 opening wired for real
(notebooks/_build_mvp_run_notebooks.py's OPENING_HANDLER_CELL, added
2026-11-19).

Why this exists (2026-11-19, user's explicit ask): 3 of the 4 attack7
combos (small_resnet+MNIST, tiny_cnn+FEMNIST, small_resnet+FEMNIST)
already had FINAL, GOOD accuracy/security checkpoints from before
traceability wiring existed -- resuming any of them would load the
finished checkpoint and exit immediately (no rounds left in
`range(20, 20)`), producing ZERO new traceability output. Getting a
real traceability result for them means running each fresh from round
0. Rather than 3 (now 4) separate Colab tabs (awkward for one person
to babysit, and free-tier Colab often won't run multiple GPU sessions
on one account at once), this notebook runs them all back-to-back in
one session, unattended.

tiny_cnn+MNIST is ALSO included here (2026-11-20, same user ask,
widened): its own standalone notebook
(`RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb`) already has the identical
traceability wiring and may be running separately at the same time --
that's fine, this combo's copy checkpoints to its OWN
`...-traceability` branch, distinct from both that notebook's branch
AND the other 3 combos' branches here, so the two can never collide
even if run concurrently. Running it twice is redundant but harmless;
not including it here just because it might already be running
elsewhere would leave this notebook's "run every combo, in one place"
promise broken if that other run is interrupted.

**Deliberately on NEW, separate checkpoint branches** (user: "has to
be different") -- `<original-attack7-branch>-traceability` for each
combo, NOT the original `colab-checkpoints-<case>-<model>-attack7`
branches, which keep their existing final results completely
untouched. If this notebook's run needs to be redone, re-running it
again only touches these `-traceability` branches.

Each combo gets its OWN checkpoint-repo clone directory too (not just
branch) -- see CHECKPOINT_MD's "separate clone" rationale in
notebooks/_build_mvp_run_notebooks.py; reusing one clone dir across
combos would skip that cell's "already cloned, reuse it" branch for
combo 2/3, which never checks out a DIFFERENT branch on an existing
clone (it only has logic for "clone + checkout" or "reuse exactly
what's already checked out").

Everything else (model, dataset, ring, DKG committee, attack
construction, run_training_loop call, eval) is pulled directly from
notebooks/_build_mvp_run_notebooks.py's own cell-generating functions
and fl/attacks/mvp_matrix.py's AttackCaseConfig list -- nothing here
reimplements a cell that generator already owns, so this notebook can
never silently drift from the other 4 attack7 notebooks' own content.
"""
import json
import os
import sys

NB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB_DIR)

import _build_mvp_run_notebooks as gen  # noqa: E402
from fl.attacks.mvp_matrix import ATTACK_CASES, N_CHEATERS  # noqa: E402

nb = gen.nb

# All 4 attack7 combos, each run fresh under real traceability wiring.
COMBO_CASES = list(ATTACK_CASES)
assert len(COMBO_CASES) == 4, f"expected all 4 combos, got {len(COMBO_CASES)}"


def _cfg_dict(cfg):
    return {
        "case_prefix": cfg.case_prefix,
        "model": cfg.model,
        "dataset_label": cfg.dataset_label,
        "num_classes": cfg.num_classes,
        "model_note": cfg.model_note,
        "ckpt_branch": cfg.ckpt_branch,
    }


TITLE_MD = f"""\
# Attack7 traceability combo run: all 4 combos, back-to-back

Runs all {len(COMBO_CASES)} attack7 combos in ONE Colab session, each
from round 0, each with \u00a76.2's opening protocol wired for real
(`on_pseudonym_flagged` -> `fl/trace.py::run_opening_protocol` against
the real DKG committee this notebook builds for each combo -- see
`notebooks/_build_mvp_run_notebooks.py`'s `OPENING_HANDLER_CELL`
docstring for exactly what that does).

**Combos in this notebook, in run order:**
{chr(10).join(f"{i+1}. `{c.model}` + `{c.dataset_label}`" for i, c in enumerate(COMBO_CASES))}

**Why fresh, for every combo:** `small_resnet+MNIST`, `tiny_cnn+FEMNIST`,
and `small_resnet+FEMNIST` already had a FINAL, good accuracy/security
checkpoint from before traceability wiring existed (see FIX.md's
attack-matrix milestone, 2026-11-19) -- resuming one of those would
load the already-complete checkpoint and exit immediately (no rounds
left to run), producing no traceability output at all. `tiny_cnn+MNIST`
is included here too even though its own standalone notebook
(`RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb`, same traceability wiring)
may be running separately at the same time -- this combo's copy uses
its OWN distinct checkpoint branch, so the two can never collide even
run concurrently; running it twice is redundant, not harmful.

**Checkpoints go to NEW, separate branches** --
`<combo>-attack7-traceability`, distinct from each combo's existing
`<combo>-attack7` branch (and, for tiny_cnn+MNIST, distinct from the
standalone notebook's own branch too), so nothing here ever touches
an existing final result. Each combo also gets its own checkpoint-repo
clone directory (`CKPT_REPO_DIR`), not a shared one -- see the
checkpoint markdown cell before each combo for why one clone can't
safely serve two different branches in this notebook's design.

**Expected result** (logged before running, so a surprise is visible
as a surprise): each combo's 7 attackers should get their pseudonym
opened very early -- their forced delta is a model-replacement-scale
outlier every single round, independent of and in addition to V3's
existing per-round rejection. Accuracy and V3-rejection numbers should
land close to each combo's already-recorded final result (same
protocol, same attack, fresh random seeds), since nothing about
training or the attack itself changed -- only the new traceability
layer sitting alongside it.

**Runtime**: ~5 hours per combo (same as any single attack7 notebook),
so plan for roughly {len(COMBO_CASES)*5} hours total, unattended, in
one session.
"""

COMBO_INTRO_MD_REPEAT = """\
## Checkpoint durability for this combo

Same git-backed checkpointing as every other run notebook (see the
full explanation above the first combo) -- this combo gets its own
clone directory and its own dedicated `-traceability` branch, so it
can never collide with another combo's checkpoint in this same
session.
"""


def build_combo_notebook():
    cells = [("md", TITLE_MD), ("code", gen.CLONE_CELL), ("code", gen.INSTALL_CELL),
             ("md", gen.RESTART_MD), ("code", gen.SANITY_CELL), ("code", gen.CONFIG_CELL)]

    for i, cfg in enumerate(COMBO_CASES):
        cfg_dict = _cfg_dict(cfg)
        slug = f"{cfg.case_prefix}_{cfg.model}"
        new_branch = f"{cfg.ckpt_branch}-traceability"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-trace-{i}"

        cells.append(("md", f"## Combo {i + 1} of {len(COMBO_CASES)}: `{cfg.model}` + `{cfg.dataset_label}`"))
        cells.append(("code", gen.dataset_cell(cfg_dict)))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.OPENING_HANDLER_CELL))
        cells.append(("code", gen.model_cell(cfg_dict)))
        cells.append(("md", gen.CHECKPOINT_MD if i == 0 else COMBO_INTRO_MD_REPEAT))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", gen.attack_run_cell(cfg)))
        # Preserve this combo's traceability_log/_n_opened_so_far under a
        # combo-specific name BEFORE the next combo's OPENING_HANDLER_CELL
        # resets both to fresh, empty state for its own run.
        cells.append(("code", f"""\
# Preserve combo {i + 1}'s traceability results before the next combo's
# opening-handler cell resets traceability_log / _n_opened_so_far.
traceability_log_{slug} = list(traceability_log)
n_opened_{slug} = _n_opened_so_far
print(f"combo {i + 1} ({{'{cfg.model}'}}+{{'{cfg.dataset_label}'}}) traceability preserved: "
      f"{{len(traceability_log_{slug})}} flagged, {{n_opened_{slug}}} opened")
"""))
        cells.append(("md", gen.EVAL_MD))
        cells.append(("code", gen.eval_cell(cfg_dict)))

    # Final combined summary across all 3 combos.
    combo_labels = [f"{c.model}+{c.dataset_label}" for c in COMBO_CASES]
    slugs = [f"{c.case_prefix}_{c.model}" for c in COMBO_CASES]
    summary_lines = ["print(\"\\n\" + \"=\"*70)",
                      "print(\"COMBINED \\u00a76 TRACEABILITY SUMMARY -- all combos this notebook ran\")",
                      "print(\"=\"*70)"]
    for label, slug in zip(combo_labels, slugs):
        summary_lines.append(f"print(f\"\\n{label}:\")")
        summary_lines.append(f"if traceability_log_{slug}:")
        summary_lines.append(f"    first = traceability_log_{slug}[0]")
        summary_lines.append(
            f"    print(f'  first flagged at round {{first[\"round\"]}}, "
            f"opened={{first[\"opened\"]}}')"
        )
        summary_lines.append(
            f"    print(f'  {{n_opened_{slug}}} identity(ies) opened across "
            f"{{len(traceability_log_{slug})}} flagged pseudonym(s)')"
        )
        summary_lines.append(
            f"    opened_entries = [e for e in traceability_log_{slug} if e[\"opened\"]]"
        )
        summary_lines.append("    if opened_entries:")
        summary_lines.append(
            "        print(f'  final H(T) after this combo\\'s last opening: "
            "{opened_entries[-1][\"residual_anonymity_bits\"]:.3f} bits')"
        )
        summary_lines.append("else:")
        summary_lines.append("    print('  no pseudonym ever crossed theta -- no traceability events.')")
    cells.append(("code", "\n".join(summary_lines) + "\n"))

    return nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_combo_notebook(), f, indent=1)
    print(f"wrote {out_path}")
