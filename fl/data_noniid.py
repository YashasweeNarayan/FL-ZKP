"""FEMNIST loading (HuggingFace, via flwr-datasets' NaturalIdPartitioner
on writer_id) + the writer-disjoint split applied to real tensors
(CASE2). TORCH-DEPENDENT -- not run in this sandbox (no network, no
torch/datasets/flwr_datasets installed). The split LOGIC (which
WRITERS go to which role) is already tested torch-free in
fl/splits.py / tests/test_splits.py; this file resolves writer ids
against actual per-writer sample tensors.

Source: `flwrlabs/femnist` on the HuggingFace Hub -- 814,277 samples,
62 classes (10 digits + 26 lower + 26 upper), 28x28 grayscale, with a
`writer_id` field (ROADMAP.md §3.3). First load caches ~200MB to
`MyDrive/FL-ZKP/data/femnist_cache/` -- mount Drive first, same as
data_iid.py's CSV path, then pass that path as `cache_dir` below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .splits import (
    writer_disjoint_split, assert_writer_disjoint, local_val_holdout,
    FemnistWriterSplit, FEMNIST_LOCAL_VAL_FRACTION,
)


def _require_deps():
    try:
        import torch  # noqa: F401
        import datasets  # noqa: F401
        from flwr_datasets import FederatedDataset  # noqa: F401
        from flwr_datasets.partitioner import NaturalIdPartitioner  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "fl/data_noniid.py needs torch + datasets + flwr_datasets "
            "(not installed in this sandbox). Validate on Colab."
        ) from e


@dataclass
class FemnistClientDataset:
    images: "object"   # torch.Tensor, shape (N, 1, 28, 28), float32
    labels: "object"   # torch.Tensor, shape (N,), int64
    writer_ids: list    # which writer each sample came from (for audit)


def load_femnist_writer_index(cache_dir: str = "/content/drive/MyDrive/FL-ZKP/data/femnist_cache/") -> dict:
    """Loads the FULL FEMNIST train split once and returns a
    writer_id -> list[row_index] map plus the underlying dataset
    object, WITHOUT materializing per-writer tensors yet (that happens
    lazily in resolve_writer_group, since 814k samples as dense
    tensors up front would be wasteful when the MVP only touches
    ~145k of them per the committed split sizes).
    """
    _require_deps()
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
        cache_dir=cache_dir,
    )
    full_train = fds.load_split("train")
    writer_ids_column = full_train["writer_id"]
    index_by_writer: dict[str, list[int]] = {}
    for i, w in enumerate(writer_ids_column):
        index_by_writer.setdefault(str(w), []).append(i)
    return {"dataset": full_train, "index_by_writer": index_by_writer}


def writer_sample_counts(index_by_writer: dict) -> dict:
    return {w: len(idxs) for w, idxs in index_by_writer.items()}


def resolve_writer_group(dataset, index_by_writer: dict, writer_ids: list) -> FemnistClientDataset:
    _require_deps()
    import torch
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    all_images = []
    all_labels = []
    all_writer_tags = []
    for w in writer_ids:
        for idx in index_by_writer[w]:
            sample = dataset[idx]
            img = transform(sample["image"])  # PIL -> tensor, already [0,1]
            all_images.append(img)
            all_labels.append(int(sample["character"]))
            all_writer_tags.append(w)
    images = torch.stack(all_images) if all_images else torch.empty(0, 1, 28, 28)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return FemnistClientDataset(images, labels, all_writer_tags)


def prepare_case2(seed: int = 0, cache_dir: str = "/content/drive/MyDrive/FL-ZKP/data/femnist_cache/"):
    """End-to-end CASE2 setup: load FEMNIST's writer index, compute
    the writer-disjoint split, resolve EACH role's writers into real
    tensors, and apply the 80/20 seen-writer local-val holdout to the
    client-training writers (ROADMAP.md §3.4: "Client local val: same
    writers, 20% holdout").
    """
    loaded = load_femnist_writer_index(cache_dir=cache_dir)
    dataset = loaded["dataset"]
    index_by_writer = loaded["index_by_writer"]
    counts = writer_sample_counts(index_by_writer)

    split = writer_disjoint_split(counts, seed=seed)
    assert_writer_disjoint(split)

    client_train_datasets = []
    client_val_datasets = []
    for client_id, writer_group in enumerate(split.client_writer_groups):
        resolved = resolve_writer_group(dataset, index_by_writer, writer_group)
        n = len(resolved.labels)
        train_idx, val_idx = local_val_holdout(list(range(n)), seed=seed * 1000 + client_id, fraction=FEMNIST_LOCAL_VAL_FRACTION)
        client_train_datasets.append(FemnistClientDataset(resolved.images[train_idx], resolved.labels[train_idx], [resolved.writer_ids[i] for i in train_idx]))
        client_val_datasets.append(FemnistClientDataset(resolved.images[val_idx], resolved.labels[val_idx], [resolved.writer_ids[i] for i in val_idx]))

    server_root = resolve_writer_group(dataset, index_by_writer, split.server_root_writers)
    global_val = resolve_writer_group(dataset, index_by_writer, split.global_val_writers)
    unseen_test = resolve_writer_group(dataset, index_by_writer, split.unseen_test_writers)

    return {
        "client_train": client_train_datasets,
        "client_val": client_val_datasets,       # seen-writer accuracy source
        "server_root": server_root,
        "global_val": global_val,
        "unseen_test": unseen_test,               # unseen-writer accuracy source -- the headline metric
        "split": split,
    }
