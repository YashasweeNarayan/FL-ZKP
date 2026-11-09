"""PURE-MATH tests for fl/splits.py -- disjointness and coverage of the
stratified-IID (MNIST) and writer-disjoint (FEMNIST) splits. Synthetic
data only, no torch, no real dataset download -- see ROADMAP.md Gate 3.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fl.splits import (
    stratified_iid_split, assert_mnist_disjoint,
    writer_disjoint_split, assert_writer_disjoint, local_val_holdout,
    N_CLIENTS, MNIST_CLIENT_TRAIN_PER_CLIENT, MNIST_CLIENT_VAL_PER_CLIENT,
    MNIST_SERVER_ROOT, MNIST_GLOBAL_VAL,
    FEMNIST_CLIENT_WRITERS, FEMNIST_SERVER_ROOT_WRITERS,
    FEMNIST_GLOBAL_VAL_WRITERS, FEMNIST_UNSEEN_TEST_WRITERS,
    FEMNIST_MIN_SAMPLES_PER_WRITER,
)


def make_synthetic_mnist_labels(n=60000, n_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n)


def test_mnist_split_sizes_match_spec():
    labels = make_synthetic_mnist_labels()
    split = stratified_iid_split(labels, seed=0)
    assert len(split.client_train) == N_CLIENTS
    assert len(split.client_val) == N_CLIENTS
    # sizes are APPROXIMATE due to per-class rounding, but should be
    # within a small tolerance of the spec per client.
    for ct in split.client_train:
        assert abs(len(ct) - MNIST_CLIENT_TRAIN_PER_CLIENT) <= 10
    for cv in split.client_val:
        assert abs(len(cv) - MNIST_CLIENT_VAL_PER_CLIENT) <= 5
    assert abs(len(split.server_root) - MNIST_SERVER_ROOT) <= 50
    assert abs(len(split.global_val) - MNIST_GLOBAL_VAL) <= 50


def test_mnist_split_is_disjoint():
    labels = make_synthetic_mnist_labels()
    split = stratified_iid_split(labels, seed=1)
    assert_mnist_disjoint(split)  # must not raise


def test_mnist_split_is_stratified_not_just_random():
    """Each client's train set should have roughly the same class
    proportions as the full set (within sampling noise) -- this is
    what distinguishes a STRATIFIED split from an arbitrary one.
    """
    n_classes = 10
    labels = make_synthetic_mnist_labels(n_classes=n_classes)
    split = stratified_iid_split(labels, seed=2)
    overall_freq = np.bincount(labels, minlength=n_classes) / len(labels)

    client0_labels = labels[split.client_train[0]]
    client0_freq = np.bincount(client0_labels, minlength=n_classes) / len(client0_labels)

    max_dev = np.max(np.abs(client0_freq - overall_freq))
    assert max_dev < 0.03, f"client 0's class distribution deviates too much: {max_dev}"


def test_mnist_split_raises_if_not_enough_data():
    labels = make_synthetic_mnist_labels(n=1000)  # far too few
    try:
        stratified_iid_split(labels, seed=0)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def make_synthetic_femnist_writer_counts(n_writers=800, seed=0):
    rng = np.random.default_rng(seed)
    counts = {}
    for i in range(n_writers):
        # most writers eligible (>=120); a minority intentionally too
        # small, to also exercise the eligibility filter itself.
        if i % 5 == 0:
            n = int(rng.integers(50, 119))
        else:
            n = int(rng.integers(150, 400))
        counts[f"writer_{i:04d}"] = n
    return counts


def test_femnist_split_sizes_and_eligibility():
    counts = make_synthetic_femnist_writer_counts(n_writers=800, seed=0)
    split = writer_disjoint_split(counts, seed=0)
    assert len(split.client_writer_groups) == N_CLIENTS
    total_client_writers = sum(len(g) for g in split.client_writer_groups)
    assert total_client_writers == FEMNIST_CLIENT_WRITERS
    assert len(split.server_root_writers) == FEMNIST_SERVER_ROOT_WRITERS
    assert len(split.global_val_writers) == FEMNIST_GLOBAL_VAL_WRITERS
    assert len(split.unseen_test_writers) == FEMNIST_UNSEEN_TEST_WRITERS

    # every assigned writer must meet the minimum-sample threshold
    all_assigned = set()
    for g in split.client_writer_groups:
        all_assigned |= set(g)
    all_assigned |= set(split.server_root_writers)
    all_assigned |= set(split.global_val_writers)
    all_assigned |= set(split.unseen_test_writers)
    for w in all_assigned:
        assert counts[w] >= FEMNIST_MIN_SAMPLES_PER_WRITER


def test_femnist_split_is_writer_disjoint():
    counts = make_synthetic_femnist_writer_counts(n_writers=800, seed=1)
    split = writer_disjoint_split(counts, seed=1)
    assert_writer_disjoint(split)  # must not raise


def test_femnist_split_raises_if_not_enough_eligible_writers():
    # make almost all writers ineligible
    counts = {f"w{i}": 10 for i in range(700)}
    counts["w0"] = 500  # only one eligible writer
    try:
        writer_disjoint_split(counts, seed=0)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_femnist_split_deterministic_given_seed():
    counts = make_synthetic_femnist_writer_counts(n_writers=800, seed=2)
    split_a = writer_disjoint_split(counts, seed=42)
    split_b = writer_disjoint_split(counts, seed=42)
    assert split_a.client_writer_groups == split_b.client_writer_groups
    assert split_a.unseen_test_writers == split_b.unseen_test_writers


def test_local_val_holdout_is_disjoint_and_covers_all():
    indices = list(range(1000))
    train, val = local_val_holdout(indices, seed=0, fraction=0.2)
    assert len(set(train) & set(val)) == 0
    assert sorted(train + val) == sorted(indices)
    assert abs(len(val) - 200) <= 2


if __name__ == "__main__":
    test_mnist_split_sizes_match_spec()
    print("test_mnist_split_sizes_match_spec PASSED")
    test_mnist_split_is_disjoint()
    print("test_mnist_split_is_disjoint PASSED")
    test_mnist_split_is_stratified_not_just_random()
    print("test_mnist_split_is_stratified_not_just_random PASSED")
    test_mnist_split_raises_if_not_enough_data()
    print("test_mnist_split_raises_if_not_enough_data PASSED")
    test_femnist_split_sizes_and_eligibility()
    print("test_femnist_split_sizes_and_eligibility PASSED")
    test_femnist_split_is_writer_disjoint()
    print("test_femnist_split_is_writer_disjoint PASSED")
    test_femnist_split_raises_if_not_enough_eligible_writers()
    print("test_femnist_split_raises_if_not_enough_eligible_writers PASSED")
    test_femnist_split_deterministic_given_seed()
    print("test_femnist_split_deterministic_given_seed PASSED")
    test_local_val_holdout_is_disjoint_and_covers_all()
    print("test_local_val_holdout_is_disjoint_and_covers_all PASSED")
    print("\nALL splits TESTS PASSED")
