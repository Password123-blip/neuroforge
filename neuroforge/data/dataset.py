"""Datasets and batching.

``Dataset`` is the abstract interface; ``DataLoader`` turns a dataset into an
iterable of shuffled, batched NumPy arrays.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Dataset", "TensorDataset", "LanguageModelDataset", "DataLoader",
    "train_val_split", "random_split",
]


class Dataset:
    """Abstract dataset: implement ``__len__`` and ``__getitem__``."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Wrap one or more equal-length arrays; indexing returns a tuple."""

    def __init__(self, *arrays):
        if not arrays:
            raise ValueError("TensorDataset needs at least one array")
        self.arrays = [np.asarray(a) for a in arrays]
        n = len(self.arrays[0])
        for a in self.arrays:
            if len(a) != n:
                raise ValueError("all arrays must have the same length")
        self._len = n

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        return tuple(a[index] for a in self.arrays)


class LanguageModelDataset(Dataset):
    """Sliding windows over a token stream for next-token prediction.

    Item ``i`` is ``(tokens[i : i+block_size], tokens[i+1 : i+block_size+1])``.
    """

    def __init__(self, tokens, block_size: int, stride: int = 1):
        self.tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        self.block_size = int(block_size)
        self.stride = int(stride)
        if len(self.tokens) <= self.block_size:
            raise ValueError("token stream must be longer than block_size")
        self._len = (len(self.tokens) - self.block_size - 1) // self.stride + 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        start = index * self.stride
        chunk = self.tokens[start:start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def _collate(batch):
    """Turn a list of per-item tuples into a tuple of stacked arrays."""
    if isinstance(batch[0], tuple):
        return tuple(np.stack([item[i] for item in batch]) for i in range(len(batch[0])))
    return np.stack(list(batch))


class DataLoader:
    """Iterate over a dataset in shuffled mini-batches.

    Args:
        dataset: anything with ``__len__`` and ``__getitem__``.
        batch_size: number of examples per batch.
        shuffle: reshuffle the order at the start of every epoch.
        drop_last: discard a final partial batch.
        seed: seed for the shuffling RNG (for reproducible runs).
    """

    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True,
                 drop_last: bool = False, seed: int = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        if not self.shuffle:
            order = np.arange(n)
        elif self.seed is None:
            order = self._rng.permutation(n)
        else:
            # Deterministic per-epoch ordering: two loaders built with the same
            # seed produce the same sequence of shuffles, which makes a whole
            # training run reproducible.
            order = np.random.default_rng(self.seed + self._epoch).permutation(n)
            self._epoch += 1
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            if len(idx) < self.batch_size and self.drop_last:
                continue
            yield _collate([self.dataset[int(i)] for i in idx])


def train_val_split(*arrays, val_fraction: float = 0.2, shuffle: bool = True,
                    seed: int = None):
    """Split arrays into train/validation halves along the first axis."""
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    n_val = int(round(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = tuple(a[train_idx] for a in arrays)
    val = tuple(a[val_idx] for a in arrays)
    return train, val


def random_split(dataset: Dataset, lengths, seed: int = None):
    """Split a dataset into disjoint subsets of the given sizes."""
    lengths = list(lengths)
    if sum(lengths) != len(dataset):
        raise ValueError("split lengths must sum to the dataset length")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset))
    subsets, start = [], 0
    for length in lengths:
        subsets.append(_Subset(dataset, idx[start:start + length]))
        start += length
    return subsets


class _Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[int(self.indices[i])]
