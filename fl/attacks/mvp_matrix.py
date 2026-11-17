"""Single source of truth for the MVP attack-variant run matrix.

Four WITH_ATTACK notebook variants exist, one per (model, dataset)
combination in the MVP matrix -- the same 2x2 matrix the CLEAN run
notebooks cover (notebooks/_build_mvp_run_notebooks.py):

    tiny_cnn   x MNIST    (RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb,     hand-built, F11)
    tiny_cnn   x FEMNIST  (RUN_MVP_TINY_CNN_FEMNIST_WITH_ATTACK.ipynb,   generator-built)
    small_resnet x MNIST  (RUN_MVP_SMALL_RESNET_MNIST_WITH_ATTACK.ipynb, generator-built)
    small_resnet x FEMNIST(RUN_MVP_SMALL_RESNET_FEMNIST_WITH_ATTACK.ipynb, generator-built)

This module exists so the attack PARAMETERS (which clients are
malicious, which git branch each variant checkpoints to) can't
silently drift apart across the four -- the same concern that made
notebooks/_build_mvp_run_notebooks.py's own module docstring insist on
one generator for the CLEAN run notebooks. Both the generator's
`notebooks/_build_mvp_run_notebooks.py` and the one hand-built notebook
that predates it (`RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb`) should
treat the numbers below as the one place they live; the hand-built
notebook currently has `CHEATER_INDICES = list(range(7))` written out
locally rather than imported from here (it predates this module) --
functionally identical to `list(CHEATER_INDICES)` below, just not
import-linked yet.

The attack CONSTRUCTION itself (how a malicious delta is crafted) is
NOT here -- see `attacks.a1_scaling.craft_scaled_update`, used by every
variant identically regardless of case.
"""

from dataclasses import dataclass

# ARCHITECTURE.md's own contributions table, N7: "n=50 ... 7 adversaries".
# 7 of 50 = 14%, the number every WITH_ATTACK notebook validates
# against -- NOT ARCHITECTURE.md A2's theoretical tolerance ceiling
# (f < n/2 = 25), which is a stress-test bound, not a target to run at.
# See FIX.md's discussion of why 7, not 25, and LEFT.md's F17 recommendation
# document for why the system does not persistently ban these clients
# (per-round V3 rejection every round is the intended, literature-aligned
# behavior -- see LEFT.md section on V3-vs-persistent-ban).
N_CLIENTS = 50
N_CHEATERS = 7
CHEATER_FRACTION = N_CHEATERS / N_CLIENTS  # 0.14

# clients 0..N_CHEATERS-1 -- all 7 share ONE crafted delta (a simplifying
# choice, noted as such in every WITH_ATTACK notebook's own markdown,
# not independently-drawn attacks per client). See fl/client.py's
# `_force_delta` docstring for why this is not the same as a lying/
# forged submission -- everything else about each attacker's
# submission (ring signature, VRF, masking, sketching, range proof)
# runs exactly as it would for an honest client.
CHEATER_INDICES = tuple(range(N_CHEATERS))


@dataclass(frozen=True)
class AttackCaseConfig:
    filename: str
    case_prefix: str  # "case1" (MNIST) or "case2" (FEMNIST)
    model: str  # "tiny_cnn" or "small_resnet"
    dataset_label: str
    num_classes: int
    model_note: str
    ckpt_branch: str  # dedicated git checkpoint branch (FIX.md F17) -- distinct from the clean run's own branch for the same (model, dataset)
    clean_run_filename: str  # the companion clean-run notebook this variant is diffed against in its own markdown


ATTACK_CASES: list[AttackCaseConfig] = [
    AttackCaseConfig(
        filename="RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb",
        case_prefix="case1",
        model="tiny_cnn",
        dataset_label="MNIST",
        num_classes=10,
        model_note=(
            "~20.6K params, matches WhiteCloak/PQS-BFL's model scale "
            "(see fl/models.py build_tiny_cnn docstring)."
        ),
        ckpt_branch="colab-checkpoints-case1-tiny_cnn-attack7",
        clean_run_filename="RUN_MVP_TINY_CNN_MNIST.ipynb",
    ),
    AttackCaseConfig(
        filename="RUN_MVP_TINY_CNN_FEMNIST_WITH_ATTACK.ipynb",
        case_prefix="case2",
        model="tiny_cnn",
        dataset_label="FEMNIST",
        num_classes=62,
        model_note=(
            "~20.6K params, matches WhiteCloak/PQS-BFL's model scale "
            "(see fl/models.py build_tiny_cnn docstring) -- same backbone "
            "as the MNIST attack run, different (non-IID, writer-split) "
            "dataset."
        ),
        ckpt_branch="colab-checkpoints-case2-tiny_cnn-attack7",
        clean_run_filename="RUN_MVP_TINY_CNN_FEMNIST.ipynb",
    ),
    AttackCaseConfig(
        filename="RUN_MVP_SMALL_RESNET_MNIST_WITH_ATTACK.ipynb",
        case_prefix="case1",
        model="small_resnet",
        dataset_label="MNIST",
        num_classes=10,
        model_note=(
            "~263K params, anchored to WhiteCloak's own second reported "
            "model (CIFAR-10-L ResNet, 273K params -- see fl/models.py "
            "build_small_resnet docstring)."
        ),
        ckpt_branch="colab-checkpoints-case1-small_resnet-attack7",
        clean_run_filename="RUN_MVP_SMALL_RESNET_MNIST.ipynb",
    ),
    AttackCaseConfig(
        filename="RUN_MVP_SMALL_RESNET_FEMNIST_WITH_ATTACK.ipynb",
        case_prefix="case2",
        model="small_resnet",
        dataset_label="FEMNIST",
        num_classes=62,
        model_note=(
            "~263K params (see fl/models.py build_small_resnet docstring) "
            "on the non-IID, writer-split FEMNIST dataset -- completes "
            "the 2-model x 2-dataset attack matrix."
        ),
        ckpt_branch="colab-checkpoints-case2-small_resnet-attack7",
        clean_run_filename="RUN_MVP_SMALL_RESNET_FEMNIST.ipynb",
    ),
]
