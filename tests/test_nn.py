"""Tests for layers, losses, normalisation and optimisers."""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge import nn
from neuroforge.tensor import Tensor


def numeric_grad(f, arr, eps=1e-6):
    arr = np.array(arr, dtype=np.float64, copy=True)
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps
        fp = f(arr)
        arr[i] = old - eps
        fm = f(arr)
        arr[i] = old
        grad[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


# ----------------------------------------------------------------------
# Linear
# ----------------------------------------------------------------------
def test_linear_shapes():
    layer = nn.Linear(6, 4)
    assert layer(Tensor(np.zeros((5, 6)))).shape == (5, 4)
    assert layer(Tensor(np.zeros((2, 3, 6)))).shape == (2, 3, 4)
    assert layer(Tensor(np.zeros(6))).shape == (4,)


def test_linear_gradients():
    layer = nn.Linear(6, 4)
    x = Tensor(np.random.default_rng(0).normal(size=(5, 6)), requires_grad=True)
    layer(x).sum().backward()
    assert x.grad.shape == (5, 6)
    assert layer.weight.grad.shape == (4, 6)
    assert layer.bias.grad.shape == (4,)


def test_linear_no_bias():
    layer = nn.Linear(3, 2, bias=False)
    assert layer.bias is None
    assert len(layer.parameters()) == 1


# ----------------------------------------------------------------------
# Conv2d
# ----------------------------------------------------------------------
def test_conv2d_shapes():
    conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
    assert conv(Tensor(np.zeros((1, 2, 8, 8)))).shape == (1, 3, 8, 8)
    conv2 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=0)
    assert conv2(Tensor(np.zeros((1, 1, 9, 9)))).shape == (1, 4, 4, 4)


def test_conv2d_numeric_gradient():
    conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
    img = np.random.default_rng(3).normal(size=(1, 2, 5, 5))

    x = Tensor(img, requires_grad=True)
    (conv(x) ** 2).sum().backward()

    expected = numeric_grad(lambda a: (conv(Tensor(a)) ** 2).sum().item(), img)
    assert np.allclose(x.grad, expected, atol=1e-5)


def test_conv2d_weight_gradient_is_finite():
    conv = nn.Conv2d(1, 2, 3, padding=1)
    conv(Tensor(np.random.default_rng(0).normal(size=(2, 1, 6, 6)))).sum().backward()
    assert np.isfinite(conv.weight.grad).all()
    assert np.isfinite(conv.bias.grad).all()


# ----------------------------------------------------------------------
# pooling
# ----------------------------------------------------------------------
def test_maxpool():
    pool = nn.MaxPool2d(2, stride=2)
    x = Tensor(np.random.default_rng(0).normal(size=(1, 1, 4, 4)), requires_grad=True)
    out = pool(x)
    assert out.shape == (1, 1, 2, 2)
    out.sum().backward()
    assert np.isfinite(x.grad).all()
    # gradient routes only to the maxima
    assert (x.grad > 0).sum() == 4


def test_maxpool_picks_maximum():
    x = Tensor(np.arange(16, dtype=float).reshape(1, 1, 4, 4))
    out = nn.MaxPool2d(2, stride=2)(x)
    assert np.allclose(out.data[0, 0], [[5, 7], [13, 15]])


def test_avgpool():
    x = Tensor(np.ones((1, 1, 4, 4)))
    out = nn.AvgPool2d(2, stride=2)(x)
    assert np.allclose(out.data, 1.0)
    assert out.shape == (1, 1, 2, 2)


# ----------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------
def test_layernorm_normalises():
    ln = nn.LayerNorm(8)
    out = ln(Tensor(np.random.default_rng(0).normal(size=(4, 8))))
    assert np.allclose(out.data.mean(axis=-1), 0.0, atol=1e-9)
    assert np.allclose(out.data.var(axis=-1), 1.0, atol=1e-3)


def test_layernorm_gradient():
    ln = nn.LayerNorm(6)
    x = Tensor(np.random.default_rng(1).normal(size=(3, 6)), requires_grad=True)
    (ln(x) ** 2).sum().backward()
    assert x.grad is not None and np.isfinite(x.grad).all()


def test_rmsnorm():
    out = nn.RMSNorm(8)(Tensor(np.random.default_rng(0).normal(size=(4, 8))))
    assert out.shape == (4, 8)


def test_batchnorm_running_stats():
    bn = nn.BatchNorm1d(4)
    x = Tensor(np.random.default_rng(0).normal(size=(32, 4)))
    bn.train()
    out = bn(x)
    assert out.shape == (32, 4)
    bn.eval()
    assert bn(Tensor(np.zeros((2, 4)))).shape == (2, 4)


# ----------------------------------------------------------------------
# dropout
# ----------------------------------------------------------------------
def test_dropout_is_identity_in_eval():
    d = nn.Dropout(0.5).eval()
    x = Tensor(np.ones((5, 5)))
    assert np.allclose(d(x).data, 1.0)


def test_dropout_masks_and_scales():
    d = nn.Dropout(0.5).train()
    out = d(Tensor(np.ones((500, 500)))).data
    assert 0.45 < (out == 0).mean() < 0.55
    assert np.allclose(out[out != 0], 2.0)


def test_dropout_probability_validation():
    with pytest.raises(ValueError):
        nn.Dropout(1.0)


# ----------------------------------------------------------------------
# Embedding / Flatten
# ----------------------------------------------------------------------
def test_embedding_lookup():
    emb = nn.Embedding(10, 4)
    out = emb(np.array([[1, 2], [3, 4]]))
    assert out.shape == (2, 2, 4)
    assert np.allclose(out.data[0, 0], emb.weight.data[1])


def test_embedding_gradient_scatter():
    emb = nn.Embedding(5, 3)
    emb(np.array([1, 1, 2])).sum().backward()
    # token 1 appears twice and each occurrence contributes 1.0 to each of the
    # 3 embedding dims, so the gradient row is all 2.0
    assert np.allclose(emb.weight.grad[1], 2.0)
    assert np.allclose(emb.weight.grad[2], 1.0)
    assert np.allclose(emb.weight.grad[0], 0.0)   # never looked up


def test_flatten():
    assert nn.Flatten()(Tensor(np.zeros((2, 3, 4, 4)))).shape == (2, 48)


# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------
def test_mse_loss_value():
    pred = Tensor([[1.0, 2.0]])
    target = Tensor([[1.0, 4.0]])
    assert nn.mse_loss(pred, target).item() == pytest.approx(2.0)


def test_cross_entropy_matches_reference():
    rng = np.random.default_rng(0)
    logits = Tensor(rng.normal(size=(6, 5)), requires_grad=True)
    targets = np.array([0, 1, 2, 3, 4, 0])
    loss = nn.cross_entropy(logits, targets)

    z = logits.data
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    expected = -logp[np.arange(6), targets].mean()
    assert loss.item() == pytest.approx(expected, abs=1e-9)


def test_cross_entropy_gradient_shape():
    logits = Tensor(np.random.default_rng(0).normal(size=(4, 3)), requires_grad=True)
    nn.cross_entropy(logits, np.array([0, 1, 2, 0])).backward()
    assert logits.grad.shape == (4, 3)


def test_cross_entropy_target_mismatch():
    logits = Tensor(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        nn.cross_entropy(logits, np.array([0, 1]))


def test_cross_entropy_stable_with_large_logits():
    logits = Tensor(np.array([[1000.0, -1000.0]]), requires_grad=True)
    loss = nn.cross_entropy(logits, np.array([0]))
    assert np.isfinite(loss.item())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_bce_with_logits():
    logits = Tensor(np.random.default_rng(0).normal(size=(5, 1)), requires_grad=True)
    loss = nn.binary_cross_entropy_with_logits(logits, np.random.default_rng(1).integers(0, 2, (5, 1)))
    loss.backward()
    assert loss.item() >= 0
    assert np.isfinite(logits.grad).all()


# ----------------------------------------------------------------------
# optimisers
# ----------------------------------------------------------------------
@pytest.mark.parametrize("Opt", [nn.SGD, nn.RMSprop, nn.Adam, nn.AdamW])
def test_optimiser_reduces_loss(Opt):
    rng = np.random.default_rng(0)
    model = nn.Linear(4, 2)
    X = Tensor(rng.normal(size=(16, 4)))
    y = rng.integers(0, 2, 16)
    opt = Opt(model.parameters(), lr=1e-2)

    before = nn.cross_entropy(model(X), y).item()
    for _ in range(80):
        opt.zero_grad()
        loss = nn.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
    after = nn.cross_entropy(model(X), y).item()
    assert after < before


def test_sgd_momentum_and_nesterov():
    for kwargs in ({"momentum": 0.9}, {"momentum": 0.9, "nesterov": True}):
        model = nn.Linear(4, 2)
        opt = nn.SGD(model.parameters(), lr=1e-2, **kwargs)
        X = Tensor(np.random.default_rng(0).normal(size=(16, 4)))
        y = np.random.default_rng(1).integers(0, 2, 16)
        before = nn.cross_entropy(model(X), y).item()
        for _ in range(60):
            opt.zero_grad()
            nn.cross_entropy(model(X), y).backward()
            opt.step()
        assert nn.cross_entropy(model(X), y).item() < before


def test_nesterov_requires_momentum():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError):
        nn.SGD(model.parameters(), nesterov=True)


def test_optimiser_rejects_empty_params():
    with pytest.raises(ValueError):
        nn.Adam([])


def test_adamw_decays_weights():
    """AdamW applies decay directly to the weights, not through the gradient."""
    model = nn.Linear(3, 3, bias=False)
    model.weight.data[:] = 10.0
    opt = nn.AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)
    for _ in range(5):
        opt.zero_grad()
        # an explicit zero gradient: the update should be decay-only
        model.weight.grad = np.zeros_like(model.weight.data)
        opt.step()
    assert np.all(model.weight.data < 10.0)


def test_adam_without_decay_leaves_weights_when_grad_zero():
    model = nn.Linear(3, 3, bias=False)
    model.weight.data[:] = 10.0
    opt = nn.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(5):
        opt.zero_grad()
        model.weight.grad = np.zeros_like(model.weight.data)
        opt.step()
    assert np.allclose(model.weight.data, 10.0)


def test_zero_grad_clears():
    model = nn.Linear(3, 2)
    opt = nn.SGD(model.parameters())
    nn.mse_loss(model(Tensor(np.ones((2, 3)))), Tensor(np.zeros((2, 2)))).backward()
    assert model.weight.grad is not None
    opt.zero_grad()
    assert model.weight.grad is None


# ----------------------------------------------------------------------
# module plumbing
# ----------------------------------------------------------------------
def test_sequential_registers_parameters():
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
    assert model.num_parameters() == 10 * 20 + 20 + 20 * 2 + 2


def test_state_dict_round_trip():
    a = nn.Linear(5, 3)
    b = nn.Linear(5, 3)
    b.load_state_dict(a.state_dict())
    assert np.allclose(b.weight.data, a.weight.data)


def test_load_state_dict_rejects_shape_mismatch():
    a = nn.Linear(5, 3)
    b = nn.Linear(5, 4)
    with pytest.raises(ValueError):
        b.load_state_dict(a.state_dict())


def test_load_state_dict_rejects_unknown_keys():
    a = nn.Linear(5, 3)
    with pytest.raises(KeyError):
        a.load_state_dict({"nope": np.zeros(3)})


def test_module_list_registers_children():
    model = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
    assert len(model) == 3
    assert len(model.parameters()) == 6


def test_nested_module_parameter_names():
    model = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 2)))
    names = [n for n, _ in model.named_parameters()]
    assert any("0.weight" in n for n in names)
    assert any("1.0.weight" in n for n in names)


def test_train_eval_propagates():
    inner = nn.Dropout(0.5)
    model = nn.Sequential(nn.Linear(2, 2), inner)
    model.eval()
    assert inner.training is False
    model.train()
    assert inner.training is True


def test_initialisation_schemes():
    p = nn.Parameter(np.zeros((100, 50)))
    nn.init.xavier_uniform_(p)
    assert np.isfinite(p.data).all() and p.data.std() > 0
    nn.init.kaiming_normal_(p)
    assert np.isfinite(p.data).all() and p.data.std() > 0
    nn.init.zeros_(p)
    assert np.all(p.data == 0)
    nn.init.ones_(p)
    assert np.all(p.data == 1)
