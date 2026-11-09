"""MNIST loading + the stratified-IID split applied to real tensors
(CASE1). TORCH-DEPENDENT -- not run in this sandbox. The split LOGIC
itself (which indices go to which role) is already tested torch-free
in fl/splits.py / tests/test_splits.py; this file only resolves those
index arrays against actual image tensors.

Two supported sources, both documented so the data can be staged from
Google Drive into Colab before this module is imported there:

1. torchvision's built-in MNIST download (simplest; needs network,
   which Colab has but this sandbox doesn't):
       torchvision.datasets.MNIST(root=..., train=True, download=True)

2. A Kaggle-style flat CSV (`label, pixel0, pixel1, ..., pixel783`),
   for the person's own "upload from Drive" workflow mentioned when
   this project was scoped. Expected Drive path:

       MyDrive/FL-ZKP/data/mnist/train.csv

   Mount Drive first (`from google.colab import drive;
   drive.mount('/content/drive')`), then pass
   csv_path="/content/drive/MyDrive/FL-ZKP/data/mnist/train.csv" to
   load_mnist_from_csv below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .splits import stratified_iid_split, assert_mnist_disjoint, MnistSplit


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "fl/data_iid.py needs torch + torchvision (not installed in "
            "this sandbox). Validate on Colab."
        ) from e


@dataclass
class MnistClientDataset:
    images: "object"   # torch.Tensor, shape (N, 1, 28, 28), float32, normalized
    labels: "object"   # torch.Tensor, shape (N,), int64


def load_mnist_from_torchvision(root: str = "/content/mnist_data", download: bool = True):
    _require_torch()
    import torch
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)

    train_images = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))], dtype=torch.long)
    test_images = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_labels = torch.tensor([test_set[i][1] for i in range(len(test_set))], dtype=torch.long)
    return (train_images, train_labels), (test_images, test_labels)


def load_mnist_from_csv(csv_path: str, test_csv_path: str | None = None):
    """Kaggle-format CSV: first column 'label', remaining 784 columns
    pixel values 0-255. Normalized the same way as the torchvision
    path (mean 0.1307, std 0.3081) so model code doesn't need to know
    which loader was used.
    """
    _require_torch()
    import torch

    def _load_one(path):
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        labels = arr[:, 0].astype(np.int64)
        pixels = arr[:, 1:].astype(np.float32) / 255.0
        pixels = (pixels - 0.1307) / 0.3081
        images = pixels.reshape(-1, 1, 28, 28)
        return torch.from_numpy(images), torch.from_numpy(labels)

    train = _load_one(csv_path)
    test = _load_one(test_csv_path) if test_csv_path else None
    return train, test


def build_client_datasets(train_images, train_labels, split: MnistSplit) -> dict:
    """Resolves fl.splits.MnistSplit's index arrays against real
    tensors. Re-asserts disjointness here too (cheap, and this is the
    boundary where an indexing bug would actually corrupt real data,
    not just synthetic test indices).
    """
    assert_mnist_disjoint(split)

    client_datasets = []
    for idx in split.client_train:
        client_datasets.append(MnistClientDataset(train_images[idx], train_labels[idx]))
    client_val_datasets = []
    for idx in split.client_val:
        client_val_datasets.append(MnistClientDataset(train_images[idx], train_labels[idx]))
    server_root = MnistClientDataset(train_images[split.server_root], train_labels[split.server_root])
    global_val = MnistClientDataset(train_images[split.global_val], train_labels[split.global_val])

    return {
        "client_train": client_datasets,
        "client_val": client_val_datasets,
        "server_root": server_root,
        "global_val": global_val,
    }


def prepare_case1(seed: int = 0, source: str = "torchvision", csv_path: str | None = None, root: str = "/content/mnist_data"):
    """End-to-end CASE1 setup: load MNIST, compute the stratified-IID
    split, resolve it against real tensors, return everything the
    client/server loop needs plus the held-out test set for the
    headline accuracy number.
    """
    if source == "torchvision":
        (train_images, train_labels), (test_images, test_labels) = load_mnist_from_torchvision(root=root)
    elif source == "csv":
        if csv_path is None:
            raise ValueError("csv_path required when source='csv'")
        (train_images, train_labels), test = load_mnist_from_csv(csv_path)
        test_images, test_labels = test if test is not None else (None, None)
    else:
        raise ValueError(f"unknown source '{source}'")

    split = stratified_iid_split(train_labels.numpy(), seed=seed)
    datasets = build_client_datasets(train_images, train_labels, split)
    datasets["test"] = (test_images, test_labels)
    datasets["split"] = split
    return datasets
