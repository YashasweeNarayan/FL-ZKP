"""Generates RUN_MVP_FINAL_GRID.ipynb: ONE notebook running all 4
clean combos AND all 4 attack7 combos, each fresh from round 0, each
with \u00a76.2 traceability wired for real -- the single "final run"
the user asked for (2026-11-20): "training attack and traceability...
final grid notebook ... all the values and accuracy and evaluation
values of all the 4 combination and attacks, after checking there
must not be anything that should be left."

**What "checking" found (see STATUS.md's THIRTIETH/THIRTY-FIRST+
items and this session's own gap read before writing this file)**:
nothing is missing at the LIBRARY level -- the clean 20-round accuracy
matrix (TWENTY-FOURTH item) and the attack7 20-round accuracy+security
matrix (TWENTY-SEVENTH item) are both already complete and confirmed
on real Colab infrastructure, for all 4 model/dataset combinations
each. What was NOT yet consolidated into one place is \u00a76.2
traceability running ALONGSIDE training+eval for every one of those 8
combinations, in a single notebook, with one final table covering
accuracy, V3/security outcome, and traceability outcome together.
Two notebooks already existed that do half of this each
(`RUN_MVP_CLEAN_TRACEABILITY_CONFIRM.ipynb`,
`RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb`, both built 2026-11-20,
neither yet run on Colab) -- this notebook does NOT duplicate new
engineering, it reuses both of their exact cell-generation call
sequences (same generator functions, same `fl/attacks/mvp_matrix.py`
source of truth) merged into one file, plus a genuinely new final
summary cell that reports all 8 runs' accuracy + V3 outcome +
traceability outcome side by side -- the "all the values ... of all
the 4 combination and attacks" table the user asked for.

**Why this is a FRESH run for every one of the 8 combos, not reusing
any existing checkpoint**: every existing clean checkpoint (TWENTY-
FOURTH item) and every existing attack7 checkpoint (TWENTY-SEVENTH
item) was finished BEFORE \u00a76 traceability wiring existed --
resuming any of them loads an already-complete checkpoint and exits
immediately (`range(20, 20)` is empty), producing zero traceability
output. This is the same reasoning the two source notebooks already
used; this notebook does not change that reasoning or try to shortcut
it by reusing an old checkpoint, which would silently produce an empty
traceability log and defeat the point.

**Checkpoints go to brand-new `-final-clean` / `-final-attack7`
branches** (own clone dir too, per combo) -- distinct from every
existing branch in the repo (the plain clean branches, the plain
attack7 branches, AND the two standalone traceability-combo
notebooks' own `-clean-traceability` / `-traceability` branches), so
running this notebook can never collide with, or silently resume,
anything that already exists. Re-running this notebook again only
ever touches these 8 new branches.

**Runtime, stated plainly before any cell runs**: this is the
expensive option. 4 clean combos at roughly a few hours each + 4
attack7 combos at roughly ~5 hours each (same per-combo costs already
recorded in FIX.md for the two source notebooks) is on the order of
30+ hours sequential, in ONE Colab session. If that is not what's
wanted, the already-built, already-split alternative is to run the
two source notebooks (clean-traceability-confirm, attack7-
traceability-combo) separately -- or this file's own `PARTS` list
below could be used to split this notebook the same way the baseline
attack grid was split into 5 parts (THIRTY-FIFTH item), on request.
"""
import json
import os
import sys

NB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB_DIR)

import _build_mvp_run_notebooks as gen  # noqa: E402
from fl.attacks.mvp_matrix import ATTACK_CASES  # noqa: E402

nb = gen.nb

TINY_CNN_MNIST_CFG = {
    "filename": "RUN_MVP_TINY_CNN_MNIST.ipynb",
    "case_prefix": "case1",
    "dataset_label": "MNIST",
    "model": "tiny_cnn",
    "model_note": "~20.6K params, matches WhiteCloak/PQS-BFL's model scale (see fl/models.py build_tiny_cnn docstring).",
    "num_classes": 10,
    "ckpt_branch": "colab-checkpoints-case1-tiny_cnn",
}

CLEAN_CASES = [TINY_CNN_MNIST_CFG] + list(gen.RUN_CONFIGS)
assert len(CLEAN_CASES) == 4, f"expected all 4 clean combos, got {len(CLEAN_CASES)}"

ATTACK7_CASES = list(ATTACK_CASES)
assert len(ATTACK7_CASES) == 4, f"expected all 4 attack7 combos, got {len(ATTACK7_CASES)}"


def _attack_cfg_dict(cfg):
    return {
        "case_prefix": cfg.case_prefix,
        "model": cfg.model,
        "dataset_label": cfg.dataset_label,
        "num_classes": cfg.num_classes,
        "model_note": cfg.model_note,
        "ckpt_branch": cfg.ckpt_branch,
    }


TITLE_MD = f"""\
# FINAL grid: all 4 combinations, clean AND attack7, \u00a76.2 traceability wired in

One notebook, one final run, covering everything the paper's results
section needs per combination: training, per-round accuracy, V3/
security outcome, and \u00a76.2 traceability -- for all {len(CLEAN_CASES)}
clean (all-honest) combos AND all {len(ATTACK7_CASES)} attack7
(7-of-50-malicious) combos, 8 runs total, back-to-back in one session.

**Run order:**
{chr(10).join(f"{i+1}. CLEAN  -- `{c['model']}` + `{c['dataset_label']}`" for i, c in enumerate(CLEAN_CASES))}
{chr(10).join(f"{i+1+len(CLEAN_CASES)}. ATTACK7 -- `{c.model}` + `{c.dataset_label}` ({len(ATTACK_CASES) and 7} of 50 malicious)" for i, c in enumerate(ATTACK7_CASES))}

**What's genuinely new here vs. what already exists**: the clean
20-round accuracy matrix and the attack7 20-round accuracy+security
matrix are BOTH already complete and Colab-confirmed (see STATUS.md's
TWENTY-FOURTH and TWENTY-SEVENTH items) -- this notebook does not
need to re-prove either number exists. What's new is running \u00a76.2
traceability alongside training+eval for every one of these 8
combinations in ONE place, with one final table reporting accuracy +
V3 outcome + traceability outcome together -- the two standalone
notebooks that each cover half of this
(`RUN_MVP_CLEAN_TRACEABILITY_CONFIRM.ipynb`,
`RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb`) exist but have not been
run yet either; this merges both into the single "final" artifact
asked for.

**Fresh from round 0 for every combo, by necessity**: every existing
checkpoint (clean or attack7) predates \u00a76 traceability wiring --
resuming one exits immediately with zero new rounds and zero
traceability output. There is no cheaper way to get a real
traceability log alongside these numbers than running fresh.

**Checkpoints**: brand-new `<branch>-final-clean` / `<branch>-final-attack7`
branches (one per combo, own clone dir too) -- distinct from every
branch that already exists in this repo, including the two standalone
traceability notebooks' own branches. Nothing here ever touches an
existing checkpoint or result.

**Runtime -- read before running**: ~{4 * 3}-{4 * 5} hours for the 4 clean
combos + ~{4 * 5} hours for the 4 attack7 combos, roughly
**30+ hours sequential, in one Colab session**. If that's too long for
one sitting, the cheaper paths are: (a) run the two already-built
standalone notebooks separately instead of this merged one, or
(b) ask for this notebook to be split the same way the baseline-attack
grid was split into 5 parallel parts.

**Expected results, stated before running so a surprise is visible as
a surprise**: clean combos should show zero pseudonyms ever flagged
(no false positives); attack7 combos should show all 7 attackers
flagged and opened early, same as `RUN_ATTACKS.ipynb`'s F11 gate and
the attack-matrix results already predicted. Accuracy for both should
land close to the already-recorded final numbers (TWENTY-FOURTH /
TWENTY-SEVENTH items) -- same protocol, same model, same data, fresh
random seeds; only the new traceability layer running alongside is
different.
"""

COMBO_MD_REPEAT = """\
## Checkpoint durability for this combo

Same git-backed checkpointing as every other run notebook (see the
full explanation above the first combo) -- this combo gets its own
clone directory and its own dedicated branch, so it can never collide
with another combo's checkpoint in this same session.
"""


def _capture_eval_cell(cfg, slug):
    """Capture eval_cell's locally-named accuracy variables under a
    slug-specific name before the next combo's eval_cell overwrites
    them -- same pattern as the traceability_log preservation below,
    needed because every combo in this one notebook shares one global
    namespace."""
    if cfg["case_prefix"] == "case1":
        headline_desc = "official MNIST test accuracy"
        return f"""\
headline_acc_{slug} = test_acc
val_acc_{slug} = val_acc
print(f"combo accuracy preserved: {slug} {headline_desc} = {{headline_acc_{slug}:.4f}}, "
      f"global_val = {{val_acc_{slug}:.4f}}")
"""
    headline_desc = "unseen-writer FEMNIST test accuracy"
    return f"""\
headline_acc_{slug} = unseen_acc
val_acc_{slug} = val_acc
print(f"combo accuracy preserved: {slug} {headline_desc} = {{headline_acc_{slug}:.4f}}, "
      f"global_val = {{val_acc_{slug}:.4f}}")
"""


def _preserve_traceability_and_history_cell(combo_label, slug):
    return f"""\
# Preserve this combo's traceability + round-history results before the
# next combo's OPENING_HANDLER_CELL / run cell reset traceability_log,
# _n_opened_so_far, and real_history to fresh state for its own run.
traceability_log_{slug} = list(traceability_log)
n_opened_{slug} = _n_opened_so_far
n_rounds_{slug} = len(real_history)
n_accepted_{slug} = sum(1 for h in real_history if not h['aborted'])
print(f"combo preserved: {combo_label} -- "
      f"{{n_accepted_{slug}}}/{{n_rounds_{slug}}} rounds accepted, "
      f"{{len(traceability_log_{slug})}} pseudonym(s) flagged, {{n_opened_{slug}}} opened")
"""


def build_final_grid_notebook():
    cells = [("md", TITLE_MD), ("code", gen.CLONE_CELL), ("code", gen.INSTALL_CELL),
             ("md", gen.RESTART_MD), ("code", gen.SANITY_CELL), ("code", gen.CONFIG_CELL)]

    all_slugs = []  # (slug, label, kind) for the final mega-summary

    # --- Part A: all 4 clean combos, fresh, traceability wired ---
    for i, cfg in enumerate(CLEAN_CASES):
        slug = f"clean_{cfg['case_prefix']}_{cfg['model']}"
        label = f"CLEAN {cfg['model']}+{cfg['dataset_label']}"
        all_slugs.append((slug, label, "clean"))
        new_branch = f"{cfg['ckpt_branch']}-final-clean"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-final-clean-{i}"

        cells.append(("md", f"## Combo {i + 1} of 8: {label} (no attackers)"))
        cells.append(("code", gen.dataset_cell(cfg)))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.OPENING_HANDLER_CELL))
        cells.append(("code", gen.model_cell(cfg)))
        cells.append(("md", gen.CHECKPOINT_MD if i == 0 else COMBO_MD_REPEAT))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", gen.run_cell(cfg)))
        cells.append(("code", _preserve_traceability_and_history_cell(label, slug)))
        cells.append(("md", gen.EVAL_MD))
        cells.append(("code", gen.eval_cell(cfg)))
        cells.append(("code", _capture_eval_cell(cfg, slug)))

    # --- Part B: all 4 attack7 combos, fresh, traceability wired ---
    for i, cfg in enumerate(ATTACK7_CASES):
        cfg_dict = _attack_cfg_dict(cfg)
        slug = f"attack7_{cfg.case_prefix}_{cfg.model}"
        label = f"ATTACK7 {cfg.model}+{cfg.dataset_label}"
        all_slugs.append((slug, label, "attack7"))
        new_branch = f"{cfg.ckpt_branch}-final-attack7"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-final-attack7-{i}"

        cells.append(("md", f"## Combo {i + 5} of 8: {label} (7 of 50 malicious)"))
        cells.append(("code", gen.dataset_cell(cfg_dict)))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.OPENING_HANDLER_CELL))
        cells.append(("code", gen.model_cell(cfg_dict)))
        cells.append(("md", COMBO_MD_REPEAT))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", gen.attack_run_cell(cfg)))
        cells.append(("code", _preserve_traceability_and_history_cell(label, slug)))
        cells.append(("md", gen.EVAL_MD))
        cells.append(("code", gen.eval_cell(cfg_dict)))
        cells.append(("code", _capture_eval_cell(cfg_dict, slug)))

    # --- Final combined summary: accuracy + security + traceability, all 8 combos ---
    lines = [
        "print(\"\\n\" + \"=\"*78)",
        "print(\"FINAL GRID SUMMARY -- accuracy, V3/security, \\u00a76 traceability, all 8 runs\")",
        "print(\"=\"*78)",
    ]
    for slug, label, kind in all_slugs:
        lines.append(f"print(f\"\\n{label}:\")")
        lines.append(
            f"print(f'  accuracy: headline={{headline_acc_{slug}:.4f}}, "
            f"global_val={{val_acc_{slug}:.4f}}')"
        )
        lines.append(
            f"print(f'  rounds accepted: {{n_accepted_{slug}}}/{{n_rounds_{slug}}}')"
        )
        lines.append(f"if traceability_log_{slug}:")
        lines.append(
            f"    print(f'  traceability: {{len(traceability_log_{slug})}} flagged, "
            f"{{n_opened_{slug}}} opened')"
        )
        if kind == "clean":
            lines.append(
                "    print('  NOTE: any flag on a clean run is a false positive -- "
                "report in the paper\\'s limitations/calibration discussion.')"
            )
        else:
            lines.append(
                "    print('  expected for attack7: all 7 injected attackers should "
                "appear flagged+opened here.')"
            )
        lines.append("else:")
        if kind == "clean":
            lines.append(
                "    print('  traceability: 0 flagged -- no false positives, as expected.')"
            )
        else:
            lines.append(
                "    print('  traceability: 0 flagged -- UNEXPECTED for an attack7 run, "
                "investigate before trusting this combo\\'s result.')"
            )
    cells.append(("code", "\n".join(lines) + "\n"))

    return nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "RUN_MVP_FINAL_GRID.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_final_grid_notebook(), f, indent=1)
    print(f"wrote {out_path}")
