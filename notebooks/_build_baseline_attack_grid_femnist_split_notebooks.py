"""Splits `RUN_MVP_BASELINE_ATTACK_GRID_FEMNIST.ipynb`'s 12 grid cells
into 5 independent notebooks -- same mechanical reasoning and SAME
PART_SPLIT index grouping as `_build_baseline_attack_grid_split_notebooks.py`
already uses for the MNIST grid (2026-11-22, user request: "create #4
for tiny_CNN+FMINST, and split it in 5, while keeping the same token
value as you kept in tinyCNN+MNIST").

Each part is a FULLY STANDALONE notebook: its own clone/install/
sanity/config/dataset cells, its own subset of grid-cell blocks, and
its own summary cell scoped to just the slugs it actually ran.

**Split (12 cells -> 5 parts of 3,3,2,2,2), by original grid index
(0-11, unchanged from `_build_baseline_attack_grid_femnist_notebook.py`'s
own `GRID_CELLS` order -- itself imported unchanged from the MNIST
grid module, so the two grids' cell ordering can never drift apart):**
- Part 1 (indices 0-2): sign-flip, all 3 fractions.
- Part 2 (indices 3-5): Gaussian noise, all 3 fractions.
- Part 3 (indices 6-7): scale-to-threshold @ 10%/20%.
- Part 4 (indices 8-9): scale-to-threshold @ 30% + low-and-slow @ 10%
  (straddles an attack-type boundary -- same unavoidable split point
  as the MNIST version, flagged here for the same reason).
- Part 5 (indices 10-11): low-and-slow @ 20%/30%.

**Git-checkpoint cells are produced via the SAME `gen.git_checkpoint_cell`
placeholder template every other grid notebook in this repo uses** --
these 5 files, as committed here, still need the user's own GitHub PAT
filled in before running. This script does NOT embed any real token:
the personalized (real-PAT, `/kaggle/working`) copies are built as a
separate, never-committed step -- see this session's chat for why.
"""
import json
import os
import sys

NB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB_DIR)

import _build_mvp_run_notebooks as gen  # noqa: E402
import _build_baseline_attack_grid_femnist_notebook as bgrid  # noqa: E402

nb = gen.nb

# Same index grouping as the MNIST split -- these are indices into
# bgrid.GRID_CELLS (0-11), which bgrid itself imports unchanged from
# the MNIST grid module, so the grouping below stays meaningful.
PART_SPLIT = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
]

_PART_FILENAMES = [f"RUN_MVP_BASELINE_ATTACK_GRID_FEMNIST_PART{p + 1}.ipynb" for p in range(5)]


def _part_scope_desc(indices):
    labels = []
    for idx in indices:
        attack_key, attack_label, frac_label, n_cheaters, pct = bgrid.GRID_CELLS[idx]
        labels.append(f"{attack_label} @ {pct}%")
    return ", ".join(labels)


def _title_md(part_num, indices):
    scope = _part_scope_desc(indices)
    other_parts = [
        f"Part {p + 1} ({_part_scope_desc(PART_SPLIT[p])})"
        for p in range(5) if p != part_num - 1
    ]
    straddles_boundary = len({bgrid.GRID_CELLS[i][0] for i in indices}) > 1
    return f"""\
# Baseline-literature attack grid (FEMNIST) -- Part {part_num} of 5: {scope}

**One of 5 independent notebooks** splitting the 12-cell
`RUN_MVP_BASELINE_ATTACK_GRID_FEMNIST.ipynb` (tiny_cnn+FEMNIST, 4
attacks x 10%/20%/30%, {bgrid.REAL_ROUNDS} rounds each -- the gap-#4
spot-check combo, same attacks/fractions/rounds as the already-
committed tiny_cnn+MNIST grid) so the total runtime can run as 5
parallel Colab/Kaggle sessions instead of one long sequential one.
Each part is fully standalone -- its own clone/install/dataset setup,
no dependency on any other part.

**This part's cells**: {scope} ({len(indices)} of the 12 total).
{"**Straddles an attack-type boundary**: unavoidable splitting 12 cells into 5 roughly-even parts; nothing attack-specific changes, this part just covers the tail of one attack type and the head of the next." if straddles_boundary else ""}

**The other 4 parts** (run them in separate tabs/sessions, any order,
none depend on each other): {"; ".join(other_parts)}.

See `attacks/baseline_norm_attacks.py` and
`notebooks/_build_baseline_attack_grid_femnist_notebook.py`'s own
title markdown for the full attack citations, the FEMNIST-specific
scope note (this grid deliberately covers tiny_cnn+FEMNIST only, not
small_resnet+MNIST or small_resnet+FEMNIST), and the low-and-slow
epoch-length caveat ({bgrid.REAL_ROUNDS} rounds is not a definitive
answer for that one attack specifically -- still true here if this
part includes it).

**Checkpoints**: identical branch/clone-dir naming to the single
FEMNIST grid notebook -- each cell below keeps its own original grid
index (`{min(indices)}`-`{max(indices)}` of the 12), distinct from
every MNIST-grid branch too. Nothing collides between parts, between
datasets, or with any other notebook in this repo.
"""


def _summary_cell(indices):
    lines = [
        "print(\"\\n\" + \"=\"*70)",
        f"print(\"BASELINE-ATTACK GRID SUMMARY, FEMNIST (part, {len(indices)} of 12 cells) -- V3 + \\u00a76 traceability\")",
        "print(\"=\"*70)",
        "print(f\"{'attack':>20} | {'fraction':>8} | {'rounds':>6} | {'aborted':>7} | {'V3 rejected':>11} | {'V3 accepted':>11} | {'traced':>6} | {'opened':>6}\")",
    ]
    has_low_and_slow = False
    for idx in indices:
        attack_key, attack_label, frac_label, n_cheaters, pct = bgrid.GRID_CELLS[idx]
        slug = f"{attack_key}_{frac_label}"
        if attack_key == "low_and_slow":
            has_low_and_slow = True
        lines.append(
            f"print(f\"{{'{attack_label}':>20}} | {{'{pct}%':>8}} | {{len(real_history_{slug}):>6}} | "
            f"{{n_aborted_{slug}:>7}} | {{len(attacker_ever_rejected_{slug}):>11}} | "
            f"{{len(attacker_ever_accepted_{slug}):>11}} | {{len(traceability_log_{slug}):>6}} | "
            f"{{n_opened_{slug}:>6}}\")"
        )
    if has_low_and_slow:
        lines.append(
            f"print(f\"\\nNOTE: low-and-slow's numbers above are from a "
            f"{bgrid.REAL_ROUNDS}-round run, not a full epoch_rounds={{CFG['protocol']['epoch_rounds']}}-round "
            f"one -- see this part's own title markdown for why that attack "
            f"specifically needs a longer, dedicated run before its result is "
            f"reviewer-safe.\")"
        )
    lines.append(
        "print(\"\\n(This is one of 5 parts, FEMNIST grid -- combine with the "
        "other parts' printed tables for the full 12-cell picture; nothing "
        "here auto-merges across separate notebook sessions.)\")"
    )
    return "\n".join(lines) + "\n"


def build_part(part_num, indices):
    cells = [
        ("md", _title_md(part_num, indices)),
        ("code", gen.CLONE_CELL),
        ("code", gen.INSTALL_CELL),
        ("md", gen.RESTART_MD),
        ("code", gen.SANITY_CELL),
        ("code", gen.CONFIG_CELL),
        ("code", gen.dataset_cell(bgrid.CASE2_CFG)),
    ]

    for j, idx in enumerate(indices):
        attack_key, attack_label, frac_label, n_cheaters, pct = bgrid.GRID_CELLS[idx]
        new_branch = f"{bgrid.CASE2_CFG['ckpt_branch']}-{attack_key}-{frac_label}"
        new_repo_dir = f"/content/FL-ZKP-checkpoints-baselinegrid-femnist-{idx}"  # ORIGINAL index, not local position
        cells.append(("md", f"## Grid cell {idx + 1} of 12 (FEMNIST, part {part_num} of 5): {attack_label} @ {pct}% malicious ({n_cheaters} of 50 clients)"))
        cells.append(("code", gen.RING_CELL))
        cells.append(("code", gen.OPENING_HANDLER_CELL))
        cells.append(("code", gen.model_cell(bgrid.CASE2_CFG)))
        cells.append(("md", gen.CHECKPOINT_MD if j == 0 else
                      "## Checkpoint durability for this cell\n\nSame git-backed checkpointing as the first cell in this part -- this cell gets its own clone directory and its own dedicated branch, so it can never collide with another cell's checkpoint, whether in this part, any of the other 4, or the MNIST grid."))
        cells.append(("code", gen.git_checkpoint_cell(new_branch, ckpt_repo_dir=new_repo_dir)))
        cells.append(("code", bgrid.baseline_grid_run_cell_femnist(attack_key, attack_label, frac_label, n_cheaters, pct, seed_offset=1500 + idx)))

    cells.append(("code", _summary_cell(indices)))
    return nb(cells)


if __name__ == "__main__":
    for p, (indices, filename) in enumerate(zip(PART_SPLIT, _PART_FILENAMES), start=1):
        out_path = os.path.join(NB_DIR, filename)
        with open(out_path, "w") as f:
            json.dump(build_part(p, indices), f, indent=1)
        print(f"wrote {out_path} ({len(indices)} grid cells: {_part_scope_desc(indices)})")
