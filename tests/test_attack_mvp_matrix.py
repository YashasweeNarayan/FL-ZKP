"""Validates fl/attacks/mvp_matrix.py -- the single source of truth for
the MVP attack-variant notebook matrix (FIX.md: new fl/attacks folder).

These are structural/consistency checks on the configuration data
itself (no crypto, no model code) -- the equivalent of what would
otherwise be caught only by a human re-reading 4 near-identical
notebooks and spotting a typo'd branch name or an off-by-one in the
cheater count.
"""

import importlib
import sys
from pathlib import Path

from fl.attacks.mvp_matrix import (
    ATTACK_CASES,
    CHEATER_FRACTION,
    CHEATER_INDICES,
    N_CHEATERS,
    N_CLIENTS,
)


def test_cheater_count_matches_architecture_n7_target():
    # ARCHITECTURE.md's own contributions table, N7: "n=50 ... 7 adversaries".
    assert N_CLIENTS == 50
    assert N_CHEATERS == 7
    assert CHEATER_FRACTION == 7 / 50


def test_cheater_indices_consistent_with_count():
    assert len(CHEATER_INDICES) == N_CHEATERS
    assert len(set(CHEATER_INDICES)) == N_CHEATERS  # all distinct
    assert all(0 <= i < N_CLIENTS for i in CHEATER_INDICES)
    assert tuple(sorted(CHEATER_INDICES)) == tuple(range(N_CHEATERS))


def test_four_cases_one_per_model_dataset_combination():
    assert len(ATTACK_CASES) == 4
    combos = {(c.model, c.dataset_label) for c in ATTACK_CASES}
    assert combos == {
        ("tiny_cnn", "MNIST"),
        ("tiny_cnn", "FEMNIST"),
        ("small_resnet", "MNIST"),
        ("small_resnet", "FEMNIST"),
    }


def test_filenames_and_branches_unique():
    filenames = [c.filename for c in ATTACK_CASES]
    branches = [c.ckpt_branch for c in ATTACK_CASES]
    assert len(filenames) == len(set(filenames)), "duplicate notebook filename"
    assert len(branches) == len(set(branches)), "duplicate checkpoint branch -- would collide (see FIX.md F16/F17)"


def test_num_classes_matches_dataset():
    for c in ATTACK_CASES:
        if c.dataset_label == "MNIST":
            assert c.num_classes == 10
        elif c.dataset_label == "FEMNIST":
            assert c.num_classes == 62
        else:
            raise AssertionError(f"unexpected dataset_label: {c.dataset_label!r}")


def test_case_prefix_matches_dataset():
    # case1 = MNIST, case2 = FEMNIST -- same convention as
    # notebooks/_build_mvp_run_notebooks.py's RUN_CONFIGS.
    for c in ATTACK_CASES:
        if c.dataset_label == "MNIST":
            assert c.case_prefix == "case1"
        elif c.dataset_label == "FEMNIST":
            assert c.case_prefix == "case2"


def test_attack_branches_distinct_from_clean_run_branches():
    """An attack variant's checkpoint branch must never collide with its
    own companion clean-run notebook's branch, or with any other
    attack variant's -- that's exactly the F16/F17 bug class."""
    notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
    sys.path.insert(0, str(notebooks_dir))
    try:
        gen = importlib.import_module("_build_mvp_run_notebooks")
    finally:
        sys.path.remove(str(notebooks_dir))

    clean_branches = {cfg["ckpt_branch"] for cfg in gen.RUN_CONFIGS}
    # The hand-built baseline notebook's own branch isn't in RUN_CONFIGS
    # (it predates the generator) -- add it explicitly.
    clean_branches.add("colab-checkpoints-case1-tiny_cnn")

    attack_branches = {c.ckpt_branch for c in ATTACK_CASES}
    assert clean_branches.isdisjoint(attack_branches), (
        f"attack branch collides with a clean-run branch: "
        f"{clean_branches & attack_branches}"
    )
