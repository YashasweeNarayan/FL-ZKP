"""Data-splitting logic for both cases (ROADMAP.md §3.4 is the exact
spec this implements). PURE-MATH / TESTABLE HERE: operates on plain
label arrays (MNIST) or writer-id -> sample-count dicts (FEMNIST), not
actual images, so it runs with plain numpy and is fully tested with
synthetic data in tests/test_splits.py -- no torch, no network, no
real dataset download needed to verify the split LOGIC is correct.

Every role is disjoint by construction and the disjointness is
asserted here (not just hoped for), per ROADMAP.md's Gate 3
("splits.py asserts pairwise-disjointness of every role... and the
assertions run in CI"). FEMNIST disjointness is enforced at the
WRITER level (not sample level) specifically because the same writer
appearing in both train and test would silently inflate the
generalization number -- see assert_writer_disjoint's docstring.

Committed MVP parameters (n=50 clients; see configs/mvp_3day.yaml):

MNIST (stratified IID):
    client_train:  50 x 950  = 47,500
    client_val:    50 x 100  =  5,000
    server_root:        5,000
    global_val:          2,500
    (sums to 60,000 = full MNIST train set, zero leftover, zero overlap)

FEMNIST (writer-disjoint non-IID), writers filtered to >=120 samples,
shuffled with a fixed seed, dealt to roles in this exact order so no
writer holds two roles:
    client writers:      200  (4 writers/client x 50 clients)
    server_root writers:  60
    global_val writers:   40
    unseen_test writers: 300
    (600 writers total consumed; remainder unused / reserved for n>50)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_CLIENTS = 50
MNIST_CLIENT_TRAIN_PER_CLIENT = 950
MNIST_CLIENT_VAL_PER_CLIENT = 100
MNIST_SERVER_ROOT = 5000
MNIST_GLOBAL_VAL = 2500

FEMNIST_WRITERS_PER_CLIENT = 4
FEMNIST_CLIENT_WRITERS = FEMNIST_WRITERS_PER_CLIENT * N_CLIENTS  # 200
FEMNIST_SERVER_ROOT_WRITERS = 60
FEMNIST_GLOBAL_VAL_WRITERS = 40
FEMNIST_UNSEEN_TEST_WRITERS = 300
FEMNIST_MIN_SAMPLES_PER_WRITER = 120
FEMNIST_LOCAL_VAL_FRACTION = 0.20


# ---------------------------------------------------------------------------
# MNIST: stratified IID split over sample INDICES.
# ---------------------------------------------------------------------------

@dataclass
class MnistSplit:
    client_train: list[np.ndarray]   # len N_CLIENTS, each an index array
    client_val: list[np.ndarray]
    server_root: np.ndarray
    global_val: np.ndarray


def stratified_iid_split(labels: np.ndarray, seed: int = 0) -> MnistSplit:
    """labels: 1D array of class labels for the full training set
    (len 60,000 for real MNIST). Stratifies by class so every role
    gets (as close as integer division allows) the same class
    proportions as the full set -- this is what makes it IID across
    clients, not just "randomly split."
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    needed_per_role_total = (
        N_CLIENTS * MNIST_CLIENT_TRAIN_PER_CLIENT
        + N_CLIENTS * MNIST_CLIENT_VAL_PER_CLIENT
        + MNIST_SERVER_ROOT
        + MNIST_GLOBAL_VAL
    )
    if needed_per_role_total > len(labels):
        raise ValueError(
            f"requested split needs {needed_per_role_total} samples, "
            f"only {len(labels)} available"
        )

    # per-class index pools, shuffled
    class_pools = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        class_pools[c] = idx

    client_train = [[] for _ in range(N_CLIENTS)]
    client_val = [[] for _ in range(N_CLIENTS)]
    server_root = []
    global_val = []

    # proportional-to-class-frequency allocation, done class-by-class
    # so every role's class distribution matches the overall frequency.
    n_total = len(labels)
    for c in classes:
        pool = class_pools[c]
        frac = len(pool) / n_total

        n_client_train = int(round(frac * MNIST_CLIENT_TRAIN_PER_CLIENT))
        n_client_val = int(round(frac * MNIST_CLIENT_VAL_PER_CLIENT))
        n_server_root = int(round(frac * MNIST_SERVER_ROOT))

        cursor = 0
        for client_id in range(N_CLIENTS):
            take = pool[cursor : cursor + n_client_train]
            client_train[client_id].extend(take.tolist())
            cursor += n_client_train
        for client_id in range(N_CLIENTS):
            take = pool[cursor : cursor + n_client_val]
            client_val[client_id].extend(take.tolist())
            cursor += n_client_val
        server_root.extend(pool[cursor : cursor + n_server_root].tolist())
        cursor += n_server_root
        # global_val gets EVERYTHING remaining in this class's pool,
        # rather than its own independent round(frac * target): per-class
        # rounding in the first three roles doesn't sum exactly back to
        # len(pool), and giving the last-allocated role an independent
        # rounded target too would let that rounding error compound into
        # a real shortfall (observed empirically: ~80/2500 short across
        # 10 classes). Taking the remainder instead guarantees the whole
        # split sums to exactly len(labels) with no silently-dropped
        # samples, at the cost of global_val's size being determined
        # rather than independently rounded -- it still lands within a
        # few samples of MNIST_GLOBAL_VAL in aggregate across classes.
        global_val.extend(pool[cursor:].tolist())
        cursor = len(pool)

    return MnistSplit(
        client_train=[np.array(sorted(x)) for x in client_train],
        client_val=[np.array(sorted(x)) for x in client_val],
        server_root=np.array(sorted(server_root)),
        global_val=np.array(sorted(global_val)),
    )


def assert_mnist_disjoint(split: MnistSplit) -> None:
    all_groups = list(split.client_train) + list(split.client_val) + [split.server_root, split.global_val]
    seen = set()
    for g in all_groups:
        g_set = set(g.tolist())
        overlap = seen & g_set
        if overlap:
            raise AssertionError(f"MNIST split overlap detected: {len(overlap)} shared indices")
        seen |= g_set


# ---------------------------------------------------------------------------
# FEMNIST: writer-disjoint split over writer IDs (not sample indices --
# the dataset loader in data_noniid.py resolves writer -> sample
# indices separately, at actual-data time).
# ---------------------------------------------------------------------------

@dataclass
class FemnistWriterSplit:
    client_writer_groups: list[list[str]]   # len N_CLIENTS, each FEMNIST_WRITERS_PER_CLIENT writer ids
    server_root_writers: list[str]
    global_val_writers: list[str]
    unseen_test_writers: list[str]


def writer_disjoint_split(writer_sample_counts: dict[str, int], seed: int = 0) -> FemnistWriterSplit:
    """writer_sample_counts: {writer_id: num_samples}. Filters to
    writers with >= FEMNIST_MIN_SAMPLES_PER_WRITER, shuffles with a
    fixed seed, and deals writers to roles IN THIS ORDER (client,
    server_root, global_val, unseen_test) so no writer can hold two
    roles -- order matters because it's what "no writer holds two
    roles" actually means operationally (first-come, first-assigned).
    """
    eligible = [w for w, n in writer_sample_counts.items() if n >= FEMNIST_MIN_SAMPLES_PER_WRITER]
    needed = (
        FEMNIST_CLIENT_WRITERS
        + FEMNIST_SERVER_ROOT_WRITERS
        + FEMNIST_GLOBAL_VAL_WRITERS
        + FEMNIST_UNSEEN_TEST_WRITERS
    )
    if len(eligible) < needed:
        raise ValueError(
            f"only {len(eligible)} writers have >= {FEMNIST_MIN_SAMPLES_PER_WRITER} "
            f"samples; need {needed} for the committed split"
        )

    rng = np.random.default_rng(seed)
    eligible_sorted = sorted(eligible)  # deterministic order before shuffling
    order = rng.permutation(len(eligible_sorted))
    shuffled = [eligible_sorted[i] for i in order]

    cursor = 0
    client_writers_flat = shuffled[cursor : cursor + FEMNIST_CLIENT_WRITERS]
    cursor += FEMNIST_CLIENT_WRITERS
    server_root_writers = shuffled[cursor : cursor + FEMNIST_SERVER_ROOT_WRITERS]
    cursor += FEMNIST_SERVER_ROOT_WRITERS
    global_val_writers = shuffled[cursor : cursor + FEMNIST_GLOBAL_VAL_WRITERS]
    cursor += FEMNIST_GLOBAL_VAL_WRITERS
    unseen_test_writers = shuffled[cursor : cursor + FEMNIST_UNSEEN_TEST_WRITERS]
    cursor += FEMNIST_UNSEEN_TEST_WRITERS

    client_writer_groups = [
        client_writers_flat[i * FEMNIST_WRITERS_PER_CLIENT : (i + 1) * FEMNIST_WRITERS_PER_CLIENT]
        for i in range(N_CLIENTS)
    ]

    return FemnistWriterSplit(
        client_writer_groups=client_writer_groups,
        server_root_writers=server_root_writers,
        global_val_writers=global_val_writers,
        unseen_test_writers=unseen_test_writers,
    )


def assert_writer_disjoint(split: FemnistWriterSplit) -> None:
    """Enforces disjointness at the WRITER level. This is the specific
    check ROADMAP.md calls out: if the same writer's samples appeared
    in both a client's training set and the unseen-writer test set,
    the model would have implicitly seen that writer's handwriting
    style during training, and "unseen-writer accuracy" would be
    silently inflated -- a correctness bug that looks like a good
    result, which is the most dangerous kind.
    """
    groups = list(split.client_writer_groups) + [
        split.server_root_writers,
        split.global_val_writers,
        split.unseen_test_writers,
    ]
    seen = set()
    for g in groups:
        g_set = set(g)
        if len(g_set) != len(g):
            raise AssertionError("duplicate writer within a single role group")
        overlap = seen & g_set
        if overlap:
            raise AssertionError(f"writer-level overlap detected across roles: {overlap}")
        seen |= g_set


def local_val_holdout(writer_sample_indices: list[int], seed: int, fraction: float = FEMNIST_LOCAL_VAL_FRACTION) -> tuple[list[int], list[int]]:
    """Splits one client's (or role's) already-resolved sample indices
    into (train, val) -- used for the "same writers, 20% holdout"
    seen-writer local-val construction. Pure index partitioning, no
    data access.
    """
    rng = np.random.default_rng(seed)
    idx = np.array(writer_sample_indices)
    rng.shuffle(idx)
    n_val = int(round(len(idx) * fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())
