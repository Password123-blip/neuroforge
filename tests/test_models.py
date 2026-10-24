"""Tests for the model architectures."""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge import nn
from neuroforge.models import (
    MLP, CNN, RNN, LSTM, GRU, SequenceClassifier, GPT, TransformerClassifier,
)
from neuroforge.tensor import Tensor


# ----------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------
def test_mlp_shapes():
    model = MLP(8, (16, 12), 3)
    assert model(Tensor(np.zeros((5, 8)))).shape == (5, 3)


def test_mlp_gradients_flow():
    model = MLP(4, (8,), 2)
    out = model(Tensor(np.random.default_rng(0).normal(size=(6, 4))))
    nn.cross_entropy(out, np.zeros(6, dtype=int)).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_mlp_output_activation():
    model = MLP(4, (8,), 1, out_activation=nn.Sigmoid)
    out = model(Tensor(np.zeros((3, 4)))).data
    assert np.all((out >= 0) & (out <= 1))


# ----------------------------------------------------------------------
# CNN
# ----------------------------------------------------------------------
def test_cnn_shapes_and_lazy_head():
    model = CNN(in_channels=1, num_classes=4, channels=(4, 8), hidden=16)
    assert model(Tensor(np.zeros((3, 1, 16, 16)))).shape == (3, 4)
    assert model.classifier is not None


def test_cnn_handles_different_input_sizes():
    """The lazy head means the same model can adapt to the feature size."""
    model = CNN(in_channels=3, num_classes=2, channels=(4,), hidden=8)
    assert model(Tensor(np.zeros((1, 3, 8, 8)))).shape == (1, 2)


# ----------------------------------------------------------------------
# recurrent
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cell", ["rnn", "lstm", "gru"])
def test_rnn_sequence_shapes(cell):
    model = RNN(input_size=5, hidden_size=7, num_layers=2, cell_type=cell,
                return_sequences=True)
    assert model(Tensor(np.zeros((4, 6, 5)))).shape == (4, 6, 7)


@pytest.mark.parametrize("cell", ["rnn", "lstm", "gru"])
def test_rnn_last_hidden_shape(cell):
    model = RNN(input_size=5, hidden_size=7, cell_type=cell, return_sequences=False)
    assert model(Tensor(np.zeros((4, 6, 5)))).shape == (4, 7)


@pytest.mark.parametrize("cell", ["rnn", "lstm", "gru"])
def test_rnn_backprop_through_time(cell):
    model = RNN(input_size=3, hidden_size=5, cell_type=cell, return_sequences=True)
    x = Tensor(np.random.default_rng(0).normal(size=(2, 7, 3)), requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None and np.isfinite(x.grad).all()


def test_rnn_bidirectional():
    model = RNN(5, 7, return_sequences=True, bidirectional=True)
    assert model(Tensor(np.zeros((2, 6, 5)))).shape == (2, 6, 14)
    assert model.output_size == 14


def test_rnn_unknown_cell_type():
    with pytest.raises(ValueError):
        RNN(3, 3, cell_type="magic")


def test_sequence_classifier():
    model = SequenceClassifier(input_size=5, num_classes=3, hidden_size=8)
    assert model(Tensor(np.zeros((4, 6, 5)))).shape == (4, 3)


def test_lstm_can_learn_memory_task():
    """A tiny LSTM should beat chance on the running-sum task."""
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 1.0], size=(400, 12, 1))
    y = (X.sum(axis=(1, 2)) > 0).astype(int)

    model = SequenceClassifier(1, 2, hidden_size=16)
    opt = nn.Adam(model.parameters(), lr=1e-2)
    Xt = Tensor(X)
    for _ in range(120):
        opt.zero_grad()
        nn.cross_entropy(model(Xt), y).backward()
        opt.step()

    model.eval()
    acc = (model(Xt).data.argmax(-1) == y).mean()
    assert acc > 0.8, f"accuracy only {acc:.3f}"


# ----------------------------------------------------------------------
# transformer
# ----------------------------------------------------------------------
def test_gpt_shapes():
    model = GPT(vocab_size=11, block_size=12, embed_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    assert model(np.zeros((3, 12), dtype=int)).shape == (3, 12, 11)


def test_gpt_backward():
    model = GPT(vocab_size=11, block_size=8, embed_dim=16, num_heads=2, num_layers=1, dropout=0.0)
    idx = np.random.default_rng(0).integers(0, 11, (2, 8))
    logits = model(idx)
    nn.cross_entropy(logits.reshape(-1, 11), idx.reshape(-1)).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_gpt_is_causal():
    """A change at position t must not affect logits before t."""
    model = GPT(vocab_size=10, block_size=8, embed_dim=16, num_heads=2, num_layers=1, dropout=0.0)
    model.eval()
    a = np.array([[1, 2, 3, 4, 5]])
    b = np.array([[1, 2, 3, 4, 9]])
    la, lb = model(a).data, model(b).data
    assert np.allclose(la[0, :-1], lb[0, :-1], atol=1e-12)
    assert not np.allclose(la[0, -1], lb[0, -1])


def test_gpt_block_size_enforced():
    model = GPT(vocab_size=5, block_size=4, embed_dim=8, num_heads=2, num_layers=1)
    with pytest.raises(ValueError):
        model(np.zeros((1, 5), dtype=int))


def test_gpt_rejects_1d_input():
    model = GPT(vocab_size=5, block_size=4, embed_dim=8, num_heads=2, num_layers=1)
    with pytest.raises(ValueError):
        model(np.zeros(4, dtype=int))


def test_gpt_generate_shape_and_range():
    model = GPT(vocab_size=7, block_size=8, embed_dim=16, num_heads=2, num_layers=1, dropout=0.0)
    out = model.generate(np.array([[1, 2]]), max_new_tokens=6, seed=0)
    assert out.shape == (1, 8)
    assert out.min() >= 0 and out.max() < 7


def test_gpt_generate_is_seeded():
    model = GPT(vocab_size=7, block_size=8, embed_dim=16, num_heads=2, num_layers=1, dropout=0.0)
    a = model.generate(np.array([[1, 2]]), max_new_tokens=5, seed=42)
    b = model.generate(np.array([[1, 2]]), max_new_tokens=5, seed=42)
    assert np.array_equal(a, b)


def test_gpt_generate_top_k():
    model = GPT(vocab_size=20, block_size=8, embed_dim=16, num_heads=2, num_layers=1, dropout=0.0)
    out = model.generate(np.array([[1]]), max_new_tokens=5, top_k=3, seed=0)
    assert out.shape == (1, 6)


def test_gpt_can_overfit_a_sequence():
    """A small GPT must be able to memorise one short sequence exactly."""
    data = np.array([[1, 2, 3, 4, 5, 6]])
    model = GPT(vocab_size=8, block_size=6, embed_dim=32, num_heads=2, num_layers=1, dropout=0.0)
    opt = nn.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    for _ in range(250):
        opt.zero_grad()
        logits = model(data[:, :-1])
        nn.cross_entropy(logits.reshape(-1, 8), data[:, 1:].reshape(-1)).backward()
        opt.step()
    pred = model(data[:, :-1]).data.argmax(-1)
    assert np.array_equal(pred, data[:, 1:])


def test_multihead_attention_requires_divisible_heads():
    from neuroforge.models import MultiHeadAttention

    with pytest.raises(ValueError):
        MultiHeadAttention(embed_dim=10, num_heads=3)


def test_transformer_classifier():
    model = TransformerClassifier(vocab_size=20, num_classes=2, block_size=16,
                                  embed_dim=16, num_heads=2, num_layers=1)
    assert model(np.zeros((4, 16), dtype=int)).shape == (4, 2)


def test_transformer_classifier_is_not_causal():
    """Unlike GPT, the classifier may look both ways, so a later change matters."""
    model = TransformerClassifier(vocab_size=10, num_classes=2, block_size=8,
                                  embed_dim=16, num_heads=2, num_layers=1)
    model.eval()
    a = np.array([[1, 2, 3, 4]])
    b = np.array([[1, 2, 3, 9]])
    assert not np.allclose(model(a).data, model(b).data)
