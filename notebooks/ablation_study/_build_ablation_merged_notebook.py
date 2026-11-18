"""Merges the 4 standalone ablation notebooks (`ABLATION_TINY_CNN_MNIST`,
`ABLATION_TINY_CNN_FEMNIST`, `ABLATION_SMALL_RESNET_MNIST`,
`ABLATION_SMALL_RESNET_FEMNIST`) into ONE notebook, `Ablation_model.ipynb`,
per explicit user request (2026-11-18): one file instead of four,
pushed to its own `ablation` branch.

**What's deduped vs. what's kept per-combo.** The 4 standalone notebooks
each clone/install/sanity-check/load-config/build-the-ring from scratch
-- all 5 of those are IDENTICAL across all 4 combos (same repo, same
CFG, same n_clients=50 ring/keys; `RING_CELL` only reads
`CFG['protocol']`, never the model or dataset). Running them 4 times in
one notebook would re-clone, re-pip-install, and rebuild an identical
crypto ring three times over for nothing. Deduped here to run ONCE.
Likewise each dataset (MNIST for case1, FEMNIST for case2) is loaded
ONCE and reused by both models that use it, instead of once per combo.

What CANNOT be deduped, and is re-run once per combo: `model_cell`
(different architecture each time) and the checkpoint-setup cell
(different dedicated branch each time -- see `merged_checkpoint_cell`
below for why this one needed a real code change, not just reuse).

**The one real subtlety**: the per-notebook checkpoint-setup cell
(`git_checkpoint_cell` in `notebooks/_build_mvp_run_notebooks.py`) only
clones + picks a branch ONCE per Colab session, under the assumption
baked into every other notebook in this repo -- one notebook, one
branch, one clone -- via `if not os.path.exists(CKPT_REPO_DIR): <clone
+ pick branch> else: print("reusing it")`. Reused verbatim here, combo
2's checkpoint-setup cell would see the directory already exists (from
combo 1) and silently skip picking combo 2's branch -- quietly pushing
combo 2, 3, and 4's checkpoints into combo 1's branch. `merged_checkpoint_cell`
below fixes this: the clone+PAT-prompt still happens only once, but the
fetch-or-create-branch step runs every time the cell is called, and the
"create fresh" path explicitly branches from `origin/main` rather than
"wherever HEAD happens to be" (which, on combo 2+, would otherwise be
combo 1's branch tip, carrying its checkpoint history into a branch
that should start clean).

Imports `ablation_run_cell`/`ABLATION_CONFIGS` from
`_build_ablation_notebooks.py` (not reimplemented) and the shared cells
from the parent `_build_mvp_run_notebooks.py` (also not reimplemented)
-- this script only adds the merge-specific glue (title cells,
dataset-dedup ordering, `merged_checkpoint_cell`).
"""
import json
import os
import sys

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.dirname(ABLATION_DIR)
REPO_ROOT = os.path.dirname(NB_DIR)
sys.path.insert(0, ABLATION_DIR)
sys.path.insert(0, NB_DIR)
sys.path.insert(0, REPO_ROOT)

import _build_mvp_run_notebooks as base             # noqa: E402
import _build_ablation_notebooks as abl              # noqa: E402


def merged_checkpoint_cell(ckpt_branch, ckpt_repo_dir="/content/FL-ZKP-checkpoints"):
    return f"""\
# Switch this session's checkpoint clone to THIS combo's own branch.
#
# Unlike the single-combo ablation notebooks' checkpoint-setup cell,
# this version runs the fetch-or-create-branch step EVERY time it's
# called (once per combo in this merged notebook), not only on first
# clone -- otherwise combo 2, 3, and 4 would silently keep pushing to
# combo 1's branch (see this script's own module docstring).
import subprocess
from getpass import getpass

CKPT_REPO_DIR = "{ckpt_repo_dir}"
CKPT_BRANCH = "{ckpt_branch}"

def _git(args, cwd=CKPT_REPO_DIR, check=True):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {{' '.join(args)}} failed:\\n{{r.stdout}}\\n{{r.stderr}}")
    return r

if not os.path.exists(CKPT_REPO_DIR):
    _token = getpass("Paste a GitHub PAT with push access to YashasweeNarayan/FL-ZKP (hidden input, not stored): ").strip()
    _remote = f"https://{{_token}}@github.com/YashasweeNarayan/FL-ZKP.git"
    _git(["clone", "--depth", "1", _remote, CKPT_REPO_DIR], cwd=None)
    _git(["config", "user.email", "yashasweenarayan.yn@gmail.com"])
    _git(["config", "user.name", "Yashaswee Narayan"])
    del _token
else:
    print("checkpoint clone already present this session (from an earlier combo) -- reusing it, switching branch below.")

# Runs every call (not just on first clone) -- see module docstring.
_fetch = _git(["fetch", "origin", f"{{CKPT_BRANCH}}:refs/remotes/origin/{{CKPT_BRANCH}}"], check=False)
if _fetch.returncode == 0:
    _git(["checkout", "-B", CKPT_BRANCH, f"origin/{{CKPT_BRANCH}}"])
    print(f"found an existing '{{CKPT_BRANCH}}' branch on the remote -- resuming from it.")
else:
    # Explicit start point (origin/main), NOT "current HEAD" -- on combo
    # 2+, HEAD is whatever the previous combo's branch left it at, and
    # branching from there would carry that combo's checkpoint history
    # into a branch that should start clean.
    _git(["fetch", "origin", "main:refs/remotes/origin/main"], check=False)
    _git(["checkout", "-B", CKPT_BRANCH, "origin/main"])
    print(f"no existing '{{CKPT_BRANCH}}' branch -- starting fresh from main.")

CKPT_DIR = f"{{CKPT_REPO_DIR}}/_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def push_checkpoint_to_git(ckpt_path, round_idx):
    \"\"\"Same contract as the per-notebook version (fl/round_loop.py's
    on_checkpoint_saved) -- never raises; a push failure costs a retry
    next round, not the training run.\"\"\"
    try:
        _git(["add", "-A"])
        commit = _git(["commit", "-m", f"checkpoint: round {{round_idx}}"], check=False)
        if commit.returncode != 0 and "nothing to commit" not in commit.stdout:
            print(f"  [checkpoint push WARNING] round {{round_idx}} commit failed: {{commit.stdout.strip()}}")
            return
        push = _git(["push", "-u", "origin", CKPT_BRANCH], check=False)
        if push.returncode != 0:
            _git(["fetch", "origin", CKPT_BRANCH], check=False)
            rebase = _git(["rebase", f"origin/{{CKPT_BRANCH}}"], check=False)
            if rebase.returncode == 0:
                push2 = _git(["push", "-u", "origin", CKPT_BRANCH], check=False)
                if push2.returncode == 0:
                    print(f"  [checkpoint] round {{round_idx}}: lost a race, rebased, retried -- now pushed.")
                    return
            else:
                _git(["rebase", "--abort"], check=False)
            print(f"  [checkpoint push WARNING] round {{round_idx}} saved locally but push FAILED even after rebase retry "
                  f"(will retry next round): {{push.stderr.strip()[-300:]}}")
        else:
            print(f"  [checkpoint] round {{round_idx}} pushed to '{{CKPT_BRANCH}}' -- safe against a disconnect now.")
    except Exception as e:
        print(f"  [checkpoint push WARNING] round {{round_idx}}: {{e!r}} -- local checkpoint is still fine, will retry next round.")
"""


TITLE_MD = """\
# Ablation_model: all 4 full-participation (p=1.0) scalability checks, merged

**This notebook merges the 4 standalone ablation notebooks
(`ABLATION_TINY_CNN_MNIST`, `ABLATION_TINY_CNN_FEMNIST`,
`ABLATION_SMALL_RESNET_MNIST`, `ABLATION_SMALL_RESNET_FEMNIST`) into
one file**, per explicit request (2026-11-18) -- same purpose as each
of those (see `ARCHITECTURE.md` SS5.5.1): proving the protocol is
*capable* of training with all 50 clients real (zero dummies), not
merely configured to avoid it, for the reviewer question "does it only
work at `sampling_rate=0.25`?" 2 rounds per combo, not 20 -- this is a
correctness/scalability check, not a convergence experiment.

**What's deduped vs. kept separate, and why**: clone/install/sanity/
config/ring setup are IDENTICAL across all 4 combos (same repo, same
`CFG`, same 50-client crypto ring) -- run ONCE, not 4 times. Each
dataset (MNIST, FEMNIST) is loaded ONCE and reused by both models that
use it. What differs per combo -- the model itself, and the checkpoint
branch -- is rebuilt fresh for each of the 4 sections below. Every cell
that *can* be shared is imported verbatim from the same
`notebooks/_build_mvp_run_notebooks.py` / `notebooks/ablation_study/
_build_ablation_notebooks.py` the 4 standalone notebooks use -- nothing
here is a reimplementation, except the checkpoint-setup cell, which
needed one real change to work correctly across 4 branch-switches in a
single session (see that cell's own comment for exactly why).

**Run order**: tiny_cnn+MNIST -> tiny_cnn+FEMNIST -> small_resnet+MNIST
-> small_resnet+FEMNIST. Each combo pushes to its own dedicated
checkpoint branch (`colab-checkpoints-ablation-case{{1,2}}-{{model}}`),
exactly as the standalone notebooks did -- merging the notebook file
does NOT merge the checkpoint branches.
"""

CLOSING_MD = """\
## Summary

All 4 combos above ran 2 rounds each at `sampling_rate=1.0` (every one
of the 50 clients real, zero dummy submissions). Each combo printed its
own PASS/FAIL line; scroll up to confirm all 4 said PASS before citing
this as the full-participation proof in the paper (ARCHITECTURE.md
SS5.5.1). Each combo also wrote its own checkpoint to its own dedicated
branch -- resuming any one of them independently is still possible,
exactly as if the 4 standalone notebooks had been run separately.
"""


def build_merged_notebook():
    cells = [
        ("md", TITLE_MD),
        ("code", base.CLONE_CELL),
        ("code", base.INSTALL_CELL),
        ("md", base.RESTART_MD),
        ("code", base.SANITY_CELL),
        ("code", base.CONFIG_CELL),
        ("code", base.RING_CELL),
        ("md", "## Load both datasets once (reused across all 4 combos below)"),
        ("code", base.dataset_cell({"case_prefix": "case1"})),
        ("code", base.dataset_cell({"case_prefix": "case2"})),
    ]
    for i, cfg in enumerate(abl.ABLATION_CONFIGS, start=1):
        cells.append(("md", f"## Combo {i}/4: {cfg['model']} + {cfg['dataset_label']}  (branch: `{cfg['ckpt_branch']}`)"))
        cells.append(("code", base.model_cell(cfg)))
        cells.append(("code", merged_checkpoint_cell(cfg["ckpt_branch"])))
        cells.append(("code", abl.ablation_run_cell(cfg)))
        cells.append(("code", base.eval_cell(cfg)))
    cells.append(("md", CLOSING_MD))
    return base.nb(cells)


if __name__ == "__main__":
    out_path = os.path.join(ABLATION_DIR, "Ablation_model.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_merged_notebook(), f, indent=1)
    print(f"wrote {out_path}")
