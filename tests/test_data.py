"""Tests for datasets, data loaders, tokenisers and corpus generators."""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge import data


# ----------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------
def test_tensor_dataset_indexing():
    ds = data.TensorDataset(np.arange(10), np.arange(10) * 2)
    assert len(ds) == 10
    x, y = ds[3]
    assert x == 3 and y == 6


def test_tensor_dataset_length_mismatch():
    with pytest.raises(ValueError):
        data.TensorDataset(np.zeros(5), np.zeros(4))


def test_tensor_dataset_requires_arrays():
    with pytest.raises(ValueError):
        data.TensorDataset()


def test_language_model_dataset_windows():
    tokens = np.arange(10)
    ds = data.LanguageModelDataset(tokens, block_size=3)
    x, y = ds[0]
    assert np.array_equal(x, [0, 1, 2])
    assert np.array_equal(y, [1, 2, 3])   # y is x shifted by one


def test_language_model_dataset_rejects_short_stream():
    with pytest.raises(ValueError):
        data.LanguageModelDataset(np.arange(3), block_size=5)


# ----------------------------------------------------------------------
# data loader
# ----------------------------------------------------------------------
def test_loader_batches_cover_dataset():
    ds = data.TensorDataset(np.arange(20), np.arange(20))
    loader = data.DataLoader(ds, batch_size=6, shuffle=False)
    seen = np.concatenate([y for _, y in loader])
    assert len(seen) == 20
    assert set(seen) == set(range(20))


def test_loader_batch_count():
    ds = data.TensorDataset(np.arange(20))
    assert len(data.DataLoader(ds, batch_size=6)) == 4          # ceil(20/6)
    assert len(data.DataLoader(ds, batch_size=6, drop_last=True)) == 3


def test_loader_shuffles_but_is_seedable():
    ds = data.TensorDataset(np.arange(50))
    a = np.concatenate([b for (b,) in data.DataLoader(ds, 10, shuffle=True, seed=1)])
    b = np.concatenate([b for (b,) in data.DataLoader(ds, 10, shuffle=True, seed=1)])
    c = np.concatenate([b for (b,) in data.DataLoader(ds, 10, shuffle=True, seed=2)])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_loader_epochs_differ_within_a_run():
    """Within one run each epoch should reshuffle (but reproducibly)."""
    ds = data.TensorDataset(np.arange(50))
    loader = data.DataLoader(ds, 10, shuffle=True, seed=1)
    first = np.concatenate([b for (b,) in loader])
    second = np.concatenate([b for (b,) in loader])
    assert not np.array_equal(first, second)
    # ...and a second loader with the same seed replays the same sequence
    other = data.DataLoader(ds, 10, shuffle=True, seed=1)
    assert np.array_equal(first, np.concatenate([b for (b,) in other]))


def test_loader_single_array_collates():
    ds = data.TensorDataset(np.arange(12))
    batch = next(iter(data.DataLoader(ds, batch_size=4, shuffle=False)))
    # TensorDataset always yields tuples, so a one-array dataset gives 1-tuples
    assert isinstance(batch, tuple) and len(batch) == 1
    assert batch[0].shape == (4,)


# ----------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------
def test_train_val_split_sizes_and_disjointness():
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    (Xtr, ytr), (Xva, yva) = data.train_val_split(X, y, val_fraction=0.25, seed=0)
    assert len(Xtr) == 75 and len(Xva) == 25
    assert set(Xtr[:, 0]).isdisjoint(set(Xva[:, 0]))


def test_random_split():
    ds = data.TensorDataset(np.arange(10))
    a, b = data.random_split(ds, [6, 4], seed=0)
    assert len(a) == 6 and len(b) == 4


def test_random_split_length_mismatch():
    ds = data.TensorDataset(np.arange(10))
    with pytest.raises(ValueError):
        data.random_split(ds, [6, 5])


# ----------------------------------------------------------------------
# tokenisers
# ----------------------------------------------------------------------
def test_char_tokenizer_round_trip():
    text = "hello world"
    tok = data.CharTokenizer(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))


def test_char_tokenizer_rejects_unknown_characters():
    tok = data.CharTokenizer("abc")
    with pytest.raises(ValueError):
        tok.encode("abcd")


def test_char_tokenizer_save_load(tmp_path):
    tok = data.CharTokenizer("the quick brown fox")
    path = tmp_path / "tok.json"
    tok.save(str(path))
    other = data.CharTokenizer.load(str(path))
    assert other.vocab_size == tok.vocab_size
    assert np.array_equal(other.encode("fox"), tok.encode("fox"))


def test_bpe_round_trip():
    text = "the cat sat on the mat, the cat sat"
    bpe = data.BPETokenizer(vocab_size=40).fit(text)
    ids = bpe.encode(text)
    assert bpe.decode(ids) == text


def test_bpe_shortens_sequences():
    text = data.BUILTIN_TEXT
    char_ids = data.CharTokenizer(text).encode(text)
    bpe = data.BPETokenizer(vocab_size=200).fit(text)
    bpe_ids = bpe.encode(text)
    assert len(bpe_ids) < len(char_ids)


def test_bpe_learns_frequent_pairs_first():
    text = "ababababababab cdcdcdcdcdcd"
    bpe = data.BPETokenizer(vocab_size=10).fit(text)
    assert bpe.merges[0] == (bpe.stoi["a"], bpe.stoi["b"])


def test_bpe_save_load(tmp_path):
    bpe = data.BPETokenizer(vocab_size=50).fit(data.BUILTIN_TEXT)
    path = tmp_path / "bpe.json"
    bpe.save(str(path))
    other = data.BPETokenizer.load(str(path))
    assert np.array_equal(other.encode("machine"), bpe.encode("machine"))


# ----------------------------------------------------------------------
# corpora
# ----------------------------------------------------------------------
def test_builtin_text_exists():
    assert len(data.BUILTIN_TEXT) > 500


def test_load_text_builtin():
    assert data.load_text("builtin") == data.BUILTIN_TEXT


def test_load_text_unknown_name():
    with pytest.raises(ValueError):
        data.load_text("nope")


def test_load_text_offline_falls_back():
    """With download disabled, a remote corpus falls back rather than raising."""
    assert data.load_text("shakespeare", download=False) == data.BUILTIN_TEXT


def test_make_spiral():
    X, y = data.make_spiral(n_points=50, n_classes=3, seed=0)
    assert X.shape == (150, 2)
    assert set(np.unique(y)) == {0, 1, 2}


def test_make_moons():
    X, y = data.make_moons(n_points=100, seed=0)
    assert X.shape == (100, 2)
    assert set(np.unique(y)) == {0, 1}


def test_make_blobs():
    X, y = data.make_blobs(n_points=90, n_classes=3, seed=0)
    assert X.shape == (90, 2)


def test_make_shape_images():
    X, y = data.make_shape_images(n_samples=30, size=12, seed=0)
    assert X.shape == (30, 1, 12, 12)
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_make_sequence_task():
    X, y = data.make_sequence_task(n_samples=100, seq_len=10, seed=0)
    assert X.shape == (100, 10, 1)
    assert set(np.unique(y)).issubset({0, 1})
    # the label must actually agree with the sign of the sum
    assert np.array_equal(y, (X.sum(axis=(1, 2)) > 0).astype(int))


def test_one_hot():
    out = data.one_hot([0, 2, 1], num_classes=3)
    assert out.shape == (3, 3)
    assert np.array_equal(out.sum(axis=1), np.ones(3))
