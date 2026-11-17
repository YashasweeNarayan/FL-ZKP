"""Generates the full-participation (p=1.0) ablation-study notebooks:
all 4 (model, dataset) combinations in the MVP matrix (tiny_cnn x
MNIST/FEMNIST, small_resnet x MNIST/FEMNIST), 2 rounds each.

Purpose (ARCHITECTURE.md SS5.5.1, FIX.md F21): the shipped config runs
at `sampling_rate: 0.25` -- only ~12 of 50 clients train real deltas
per round, the other ~38 submit dummy zero updates for participation
anonymity (SS5.5). A reviewer may reasonably ask "is the protocol
actually CAPABLE of training with all 50 real clients, or does it
only work at p=0.25?" This matrix answers that directly: it forces
`CFG['protocol']['sampling_rate'] = 1.0` so every one of the 50
clients is real (VRF self-sampling threshold set to pass everyone,
zero dummy submissions), then runs 2 rounds -- enough to prove the
VRF-sampling, masking, Shamir-share, ring-signature, and aggregation
layers all complete correctly at full load, NOT enough to report a
convergence curve (that's what the main `RUN_MVP_*` notebooks are
for, at the production p=0.25 config).

**Deliberately NOT a full 20-round run.** Running all 50 clients'
real local SGD every round, for all 4 combinations, for 20 rounds
each, would cost ~4x the GPU/TPU time of the main matrix for a result
this paper doesn't need (accuracy at p=1.0 isn't a claim being made --
only "the pipeline doesn't structurally require subsampling" is).
2 rounds is enough to see round 0 complete without error and round 1
confirm it wasn't a fluke.

**Pipeline kept IDENTICAL to the main run notebooks** (per explicit
instruction, 2026-11-17): every cell here -- clone/install/sanity/
config/dataset/ring/model/checkpoint-setup/eval -- is imported
VERBATIM from `notebooks/_build_mvp_run_notebooks.py`, not
reimplemented. The ONLY thing that differs is the run cell, which
overrides `CFG['protocol']['sampling_rate']` to 1.0 and `REAL_ROUNDS`
to 2 before calling the exact same `fl.round_loop.run_training_loop`
every other notebook calls -- no library code changes, no new
aggregation logic, nothing "ablation-specific" in `fl/`.

Each of the 4 notebooks gets its own dedicated checkpoint branch
(same per-notebook-branch pattern as FIX.md F17, to avoid any
collision with the main matrix's branches or with each other).

Run this file (not the notebooks it emits) if the shared cells ever
change -- it re-imports them fresh from the parent generator, so a
fix made there (e.g. a future F2x) is picked up automatically next
time this is run.
"""
import json
import os
import sys

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.dirname(ABLATION_DIR)        # notebooks/
REPO_ROOT = os.path.dirname(NB_DIR)            # repo root
sys.path.insert(0, NB_DIR)                     # for `import _build_mvp_run_notebooks`
sys.path.insert(0, REPO_ROOT)                  # for `import fl` (unused directly here, but
                                                # kept for parity with the parent generator)

import _build_mvp_run_notebooks as base        # noqa: E402  (reused verbatim: nb, cell constants,
                                                # dataset_cell, model_cell, eval_cell, git_checkpoint_cell)


ABLATION_CONFIGS = [
    {
        "filename": "ABLATION_TINY_CNN_MNIST.ipynb",
        "case_prefix": "case1",
        "dataset_label": "MNIST",
        "model": "tiny_cnn",
        "model_note": "~20.6K params, matches WhiteCloak/PQS-BFL's model scale (see fl/models.py build_tiny_cnn docstring).",
        "num_classes": 10,
        "ckpt_branch": "colab-checkpoints-ablation-case1-tiny_cnn",
    },
    {
        "filename": "ABLATION_TINY_CNN_FEMNIST.ipynb",
        "case_prefix": "case2",
        "dataset_label": "FEMNIST",
        "model": "tiny_cnn",
        "model_note": "~20.6K params, matches WhiteCloak/PQS-BFL's model scale (see fl/models.py build_tiny_cnn docstring) -- same backbone as the MNIST ablation, different (non-IID, writer-split) dataset.",
        "num_classes": 62,
        "ckpt_branch": "colab-checkpoints-ablation-case2-tiny_cnn",
    },
    {
        "filename": "ABLATION_SMALL_RESNET_MNIST.ipynb",
        "case_prefix": "case1",
        "dataset_label": "MNIST",
        "model": "small_resnet",
        "model_note": "~263K params, anchored to WhiteCloak's own second reported model (CIFAR-10-L ResNet, 273K params -- see fl/models.py build_small_resnet docstring).",
        "num_classes": 10,
        "ckpt_branch": "colab-checkpoints-ablation-case1-small_resnet",
    },
    {
        "filename": "ABLATION_SMALL_RESNET_FEMNIST.ipynb",
        "case_prefix": "case2",
        "dataset_label": "FEMNIST",
        "model": "small_resnet",
        "model_note": "~263K params (see fl/models.py build_small_resnet docstring) on the non-IID, writer-split FEMNIST dataset -- completes the 2-model x 2-dataset ablation matrix.",
        "num_classes": 62,
        "ckpt_branch": "colab-checkpoints-ablation-case2-small_resnet",
    },
]


def ablation_title_cell(cfg):
    return f"""\
# Ablation: {cfg['model']} + {cfg['dataset_label']}, FULL participation (p=1.0), 2 rounds

**What this notebook is, and is not.** This is a scalability/
correctness stress test, not a convergence experiment. It exists to
answer one reviewer question directly: *"is the protocol actually
capable of training with all 50 clients, or does it only work at the
paper's chosen `sampling_rate: 0.25`?"* -- see `ARCHITECTURE.md`
SS5.5.1 for the full framing.

**Industry framing (for the paper's industrial-application section):**
production cross-device FL systems -- Google's and Apple's deployments
over millions to billions of devices -- never require full-population
participation either; they sample a small cohort per round by
construction, for the same device-availability/bandwidth/compute-
budget reasons this paper uses `p=0.25` at MVP scale (`n=50`).
Partial participation is the industry-standard operating mode, not a
workaround unique to this paper. This notebook is the empirical half
of that argument: it proves the protocol COULD run all 50 real every
round, it just chooses not to, for the same reason production systems
choose not to.

**What changes from the main `RUN_MVP_{cfg['model'].upper()}_{cfg['dataset_label']}.ipynb` run**:
only `CFG['protocol']['sampling_rate']` (forced to `1.0` here, so the
VRF self-sampling gate `b_k` passes for every one of the 50 clients --
zero dummy/zero-vector submissions this run) and the round count
(`2`, not `20`). Every other cell -- clone/install/sanity/config/
dataset/ring/model/checkpoint-setup/eval -- is IDENTICAL to the main
run notebook, imported verbatim from
`notebooks/_build_mvp_run_notebooks.py`, not reimplemented. No change
to `fl/round_loop.py`, `fl/client.py`, `fl/server.py`, or
`fl/aggregation.py` -- same pipeline, different config value.

**Why only 2 rounds, not 20.** All 50 clients running real local SGD
every round (instead of ~12 of 50) costs ~4x the GPU time of the main
matrix. This paper isn't claiming an accuracy number at `p=1.0` --
only that the pipeline doesn't structurally require subsampling to
function. Round 0 completing without a crypto/shape/FloatingPointError
and round 1 confirming it wasn't a fluke is sufficient evidence for
that claim; a third of your GPU budget is not worth spending on a
result nobody will cite.

**Checkpoint durability: git, not Drive**, same pattern as the main
matrix (FIX.md F17) -- this notebook's own dedicated
`{cfg['ckpt_branch']}` branch, distinct from every main-matrix and
attack-matrix branch, so it can never collide with or be confused for
a real training checkpoint.
"""


def ablation_run_cell(cfg):
    case_data_var = f"{cfg['case_prefix']}_data"
    return f"""\
# 8. The ablation run: FORCE full participation (p=1.0), 2 rounds only.
#
# This is the ONLY cell that differs from the main
# RUN_MVP_{cfg['model'].upper()}_{cfg['dataset_label']}.ipynb notebook. Everything
# above and below this cell (clone/install/sanity/config/dataset/ring/
# model/checkpoint-setup/eval) is the exact same code, imported
# verbatim from notebooks/_build_mvp_run_notebooks.py -- see this
# notebook's own intro markdown and ARCHITECTURE.md SS5.5.1.
import numpy as np
from fl.round_loop import run_training_loop, bootstrap_norms_from_model_dim, release_gpu_memory, load_checkpoint

# FIX.md F21 (2026-11-17): the ablation's whole point -- override the
# shipped config's sampling_rate (0.25) to 1.0 so the VRF self-sampling
# gate (fl/client.py's vrf_is_sampled, ARCHITECTURE.md SS5.5) passes for
# EVERY one of the 50 clients. Zero dummy (q_k=0) submissions this run;
# every accepted submission is a real, locally-trained delta. This is a
# config mutation in the notebook's own CFG dict, not a library change --
# run_training_loop reads CFG['protocol']['sampling_rate'] fresh every
# call (fl/round_loop.py), so nothing in fl/ needs to know this is an
# ablation.
CFG['protocol']['sampling_rate'] = 1.0
ABLATION_SAMPLING_RATE = CFG['protocol']['sampling_rate']
REAL_ROUNDS = 2  # deliberately NOT 20 -- see intro markdown for why.

# '_ablation_p1_r2' suffix: distinct from every main-matrix and
# attack-matrix checkpoint filename by construction, so this can never
# be mistaken for, or accidentally resumed as, a real training run.
CKPT_PATH = f'{{CKPT_DIR}}/{cfg["case_prefix"]}_{{REAL_MODEL}}_ablation_p1_r{{REAL_ROUNDS}}.pt'

root_norms = bootstrap_norms_from_model_dim(
    model_dim=dim,
    sketch_dim=CFG['sketch']['m'],
    sketch_seed=f"ablation-bootstrap-{{REAL_MODEL}}-{cfg['case_prefix']}".encode(),
    quant_bits=16,
    rng=np.random.default_rng(CFG['seed']),
)

resume_from = load_checkpoint(CKPT_PATH) if os.path.exists(CKPT_PATH) else None
if resume_from is not None:
    print(f"found ablation checkpoint at round {{resume_from['round']}} -- resuming, not restarting from round 0")

final_w_t, real_history = run_training_loop(
    model=model,
    model_dim=dim,
    client_datasets={case_data_var}['client_train'],
    ring=ring,
    client_sks=client_sks,
    pairwise_secrets=pairwise_secrets,
    pk_trace=PK_trace,
    server_root_norms=root_norms,
    CFG=CFG,
    rounds=REAL_ROUNDS,
    checkpoint_path=CKPT_PATH,
    resume_from=resume_from,
    on_checkpoint_saved=push_checkpoint_to_git,
)
release_gpu_memory()

print("="*70)
print(f"ABLATION (p={{ABLATION_SAMPLING_RATE}}) RESULT -- {{REAL_MODEL}} + {cfg['dataset_label']}")
print("="*70)
for h in real_history:
    if h.get('aborted'):
        print(f"  round {{h.get('round')}}: ABORTED -- {{h.get('reason')}} <-- UNEXPECTED at p=1.0, report this")
    else:
        n_acc = h.get('accepted')
        flag = "OK (all 50 real, no dummies)" if n_acc == n_clients else f"UNEXPECTED: only {{n_acc}}/{{n_clients}} accepted at p=1.0"
        print(f"  round {{h.get('round')}}: accepted={{n_acc}}/{{n_clients}}, aggregate_exact={{h.get('aggregate_exact')}} -- {{flag}}")

n_ok_rounds = sum(1 for h in real_history if not h.get('aborted') and h.get('accepted') == n_clients)
if n_ok_rounds == REAL_ROUNDS:
    print(f"\\nPASS: all {{REAL_ROUNDS}} rounds completed with all {{n_clients}} clients accepted as real contributors -- "
          f"the protocol is NOT structurally limited to p<1.0, subsampling is a config choice (ARCHITECTURE.md SS5.5.1).")
else:
    print(f"\\nINCONCLUSIVE/FAIL: only {{n_ok_rounds}}/{{REAL_ROUNDS}} rounds had all {{n_clients}} clients accepted -- investigate before citing this as the full-participation proof.")
"""


def build_ablation_notebook(cfg):
    cells = [
        ("md", ablation_title_cell(cfg)),
        ("code", base.CLONE_CELL),
        ("code", base.INSTALL_CELL),
        ("md", base.RESTART_MD),
        ("code", base.SANITY_CELL),
        ("code", base.CONFIG_CELL),
        ("code", base.dataset_cell(cfg)),
        ("code", base.RING_CELL),
        ("code", base.model_cell(cfg)),
        ("md", base.CHECKPOINT_MD),
        ("code", base.git_checkpoint_cell(cfg["ckpt_branch"])),
        ("code", ablation_run_cell(cfg)),
        ("md", base.EVAL_MD),
        ("code", base.eval_cell(cfg)),
    ]
    return base.nb(cells)


if __name__ == "__main__":
    for cfg in ABLATION_CONFIGS:
        out_path = os.path.join(ABLATION_DIR, cfg["filename"])
        with open(out_path, "w") as f:
            json.dump(build_ablation_notebook(cfg), f, indent=1)
        print(f"wrote {out_path}")
