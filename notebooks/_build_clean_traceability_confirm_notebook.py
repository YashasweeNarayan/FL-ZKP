"""Generates RUN_MVP_CLEAN_TRACEABILITY_CONFIRM.ipynb: the missing
"zero false positives" confirmation for \u00a76.2 traceability
(2026-11-20, user's explicit ask).

The attack7-traceability combo notebook
(`RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb`) only proves traceability
catches real attackers -- every one of its 4 combos runs with 7 of 50
clients forced malicious the whole time. It says nothing about whether
an all-honest run ever gets a false positive, because it never runs
one. Every CLEAN (all-honest) checkpoint that exists in this repo
predates the \u00a76 traceability wiring (2026-11-19) -- resuming any
of them would load an already-finished checkpoint and exit
immediately (`range(20, 20)` is empty), producing zero new
traceability output, exactly the same "resuming a finished checkpoint
proves nothing" problem the attack-combo notebook was built to avoid.
So this needs the same treatment: a fresh, from-round-0 run of every
clean combo, with the opening-protocol handler wired in, confirming
(not assuming) that nobody ever gets flagged.

Covers all 4 model/dataset combinations (matching how the attack7
combo notebook was widened to all 4, 2026-11-20, for the same
completeness reason), each fresh, each on its own NEW, separate
`-clean-traceability` branch and clone directory -- never touches any
combo's existing final clean checkpoint (STATUS.md's TWENTY-FOURTH
item), exactly the same non-destructive pattern the attack-combo
notebook used for the attack7 checkpoints.

Reuses `notebooks/_build_mvp_run_notebooks.py`'s own `run_cell` (the
CLEAN run cell, not `attack_run_cell`) verbatim -- it already threads
`on_pseudonym_flagged` into `run_training_loop` (2026-11-19 wiring),
so nothing new needed there. Everything else (dataset/ring/model/
checkpoint/eval cells) comes from the same generator, same as every
other notebook in this repo that reuses it.

**Expected result, logged before running so a surprise is visible as
a surprise**: zero pseudonyms ever cross theta, in any of the 4
combos, across all 20 rounds each -- the calibration certificate's own
alpha=0.05 bound predicts at most a ~5% chance of a false positive
PER CLIENT PER EPOCH, so across 50 clients x <1 epoch (20 rounds is
0.8 of one 25-round epoch) a small number of false positives is not
itself alarming, but the honest expectation from the dummy-padding
design (ARCHITECTURE.md line 201) is that it should be rare to
nonexistent. If one or more DOES fire, that is real, useful
information for the paper's limitations section, not a bug to hide --
report it plainly.
"""
import json
import os
import sys

NB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB_DIR)

import _build_mvp_run_notebooks as gen  # noqa: E402

nb = gen.nb

# All 4 clean model/dataset combos. RUN_CONFIGS (imported from the main
# generator) has 3 of them -- the 4th, tiny_cnn+MNIST, is the hand-built
# original notebook that predates the generator and was never folded
# into RUN_CONFIGS; added here explicitly so this notebook covers the
# full matrix, matching the attack-combo notebook's own completeness
# (2026-11-20 widening).
TINY_CNN_MNIST_CFG = {
    "filename": "RUN_MVP_TINY_CNN_MNIST.ipynb",
    "case_prefix": "case1",
    "dataset_label": "MNIST",
    "model": "tiny_cnn",
    "model_note": "~20.6K params, matches WhiteCloak/PQS-BFL's model scale (see fl/models.py build_tiny_cnn docstring).",
    "num_classes": 10,
    "ckpt_branch": "colab-checkpoints-case1-tiny_cnn",
}

COMBO_CASES = [TINY_CNN_MNIST_CFG] + list(gen.RUN_CONFIGS)
assert len(COMBO_CASES) == 4, f"expected all 4 clean combos, got {len(COMBO_CASES)}"

TITLE_MD = f"""\
# Clean-run \u00a76.2 traceability confirmation: all 4 combos, back-to-back

Runs all {len(COMBO_CASES)} CLEAN (all-honest, zero attackers) combos
in ONE Colab session, each from round 0, each with \u00a76.2's opening
protocol wired for real -- the exact same wiring the attack7-combo
notebook uses, just with no `cheater_client_idx` set. This is the
"false positive" half of the traceability claim: the attack7 combo
notebook already proved real attackers get caught; this notebook
checks whether an all-honest run ever flags an innocent client.

**Combos in this notebook, in run order:**
{chr(10).join(f"{i+1}. `{c['model']}` + `{c['dataset_label']}`" for i, c in enumerate(COMBO_CASES))}

**Why fresh, for every combo:** every existing clean checkpoint
(STATUS.md's TWENTY-FOURTH item -- the first trustworthy 20-round
result for all 4 combos) was produced BEFORE \u00a76 traceability
wiring existed. Resuming one of those would load the already-complete
checkpoint and exit immediately (no rounds left to run), producing no
traceability output at all -- same reasoning as why the attack7-combo
notebook had to run fresh too.

**Checkpoints go to NEW, separate branches** --
`<clean-branch>-clean-traceability` for each combo, distinct from each
combo's existing final clean checkpoint, so nothing here ever touches
an existing recorded result. Each combo also gets its own checkpoint-
repo clone directory, same mechanical reason as every other multi-
combo notebook in this repo (`git_checkpoint_cell`'s clone-dir reuse
constraint).

**Expected result**: zero pseudonyms flagged, in every combo, across
all 20 rounds. If any ARE flagged, that's a real result for the
paper's limitations/calibration discussion, not something to suppress
-- the final summary cell reports it plainly either way.

**Runtime**: ~20-round clean run, same order of magnitude as any
single clean MVP notebook (shorter than an attack7 run, since no
crypto-heavy repeated attacker submissions) -- budget a few hours per
combo, so plan for most of a day across all 4 in one session.
"""

COMBO_INTRO_MD_REPEAT = """\
## Checkpoint durability for this combo

Same git-backed checkpointing as every other run notebook (see the
full explanation above the first combo) -- this combo gets its own
clone directory and its own dedicated `-clean-traceability` branch, so
it can never collide with another combo's checkpoint in this same
session.
"""


def build_clean_traceability_notebook():
    cells = [("md", TITLE_MD), ("code", gen.CLONE_CELL), ("code", gen.INSTALL_CELL),
             ("md", gen.RESTART_MD), ("code", gen.SANITY_CELL), ("code", gen.CONFIG_CELL)]

    for i, cfg in enumerate(COMBO_CASES):
        slug = f"{cfg['case_prefix']}_{cfg['model']}"
        new_branch = f"{cfg['ckpt_branch']}-clean-traceability"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-cleantrace-{i}"

        cells.append(("md", f"## Combo {i + 1} of {len(COMBO_CASES)}: `{cfg['model']}` + `{cfg['dataset_label']}` (clean, no attackers)"))
        cells.append(("code", gen.dataset_cell(cfg)))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.OPENING_HANDLER_CELL))
        cells.append(("code", gen.model_cell(cfg)))
        cells.append(("md", gen.CHECKPOINT_MD if i == 0 else COMBO_INTRO_MD_REPEAT))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", gen.run_cell(cfg)))
        # Preserve this combo's traceability_log/_n_opened_so_far before
        # the next combo's OPENING_HANDLER_CELL resets both to fresh,
        # empty state -- same pattern as the attack-combo notebook.
        cells.append(("code", f"""\
# Preserve combo {i + 1}'s traceability results before the next combo's
# opening-handler cell resets traceability_log / _n_opened_so_far.
traceability_log_{slug} = list(traceability_log)
n_opened_{slug} = _n_opened_so_far
print(f"combo {i + 1} ({{'{cfg['model']}'}}+{{'{cfg['dataset_label']}'}}) traceability preserved: "
      f"{{len(traceability_log_{slug})}} flagged (expected: 0), {{n_opened_{slug}}} opened (expected: 0)")
"""))
        cells.append(("md", gen.EVAL_MD))
        cells.append(("code", gen.eval_cell(cfg)))

    # Final combined false-positive summary across all 4 combos.
    combo_labels = [f"{c['model']}+{c['dataset_label']}" for c in COMBO_CASES]
    slugs = [f"{c['case_prefix']}_{c['model']}" for c in COMBO_CASES]
    summary_lines = ["print(\"\\n\" + \"=\"*70)",
                      "print(\"CLEAN-RUN \\u00a76 FALSE-POSITIVE SUMMARY -- all combos this notebook ran\")",
                      "print(\"=\"*70)",
                      "_any_flagged = False"]
    for label, slug in zip(combo_labels, slugs):
        summary_lines.append(f"print(f\"\\n{label}:\")")
        summary_lines.append(f"if traceability_log_{slug}:")
        summary_lines.append("    _any_flagged = True")
        summary_lines.append(
            f"    print(f'  {{len(traceability_log_{slug})}} pseudonym(s) flagged, "
            f"{{n_opened_{slug}}} opened -- UNEXPECTED on an all-honest run, see entries below')"
        )
        summary_lines.append(f"    for e in traceability_log_{slug}:")
        summary_lines.append(
            "        print(f'    round {e[\"round\"]}: pseudonym {e[\"pseudonym\"][:12]}... "
            "evidence={e[\"evidence_total\"]:.3f} vs theta={e[\"theta\"]:.3f}, opened={e[\"opened\"]}')"
        )
        summary_lines.append("else:")
        summary_lines.append("    print('  0 pseudonyms flagged across all rounds -- no false positives, as expected.')")
    summary_lines.append("print(\"\\n\" + (\"RESULT: at least one false positive observed -- report in the paper's limitations/calibration discussion.\" if _any_flagged else \"RESULT: zero false positives across all 4 combos -- the \\u00a76.2 claim's other half is now confirmed empirically, not just by the calibration certificate's theoretical bound.\"))")
    cells.append(("code", "\n".join(summary_lines) + "\n"))

    return nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(NB_DIR, "RUN_MVP_CLEAN_TRACEABILITY_CONFIRM.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_clean_traceability_notebook(), f, indent=1)
    print(f"wrote {out_path}")
