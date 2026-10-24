"""Tests for the autograd engine.

The key tests here compare analytic gradients against central-difference
numerical gradients. If those agree to ~1e-6, the backward passes are correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.tensor import Tensor, no_grad, enable_grad, is_grad_enabled


def numeric_grad(f, x, eps=1e-6):
    """Central-difference gradient of scalar-valued ``f`` at array ``x``."""
    x = np.array(x, dtype=np.float64, copy=True)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        grad[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def check_grad(f, shape, tol=1e-5, seed=0):
    """Assert analytic and numeric gradients agree for ``f``."""
    rng = np.random.default_rng(seed)
    x_np = rng.normal(size=shape)
    x = Tensor(x_np, requires_grad=True)
    out = f(x)
    assert out.data.size == 1, "test function must be scalar-valued"
    out.backward()
    expected = numeric_grad(lambda a: f(Tensor(a)).item(), x_np)
    assert np.allclose(x.grad, expected, atol=tol), (
        f"max error {np.abs(x.grad - expected).max():.2e}"
    )


# ----------------------------------------------------------------------
# arithmetic
# ----------------------------------------------------------------------
def test_add_mul_sum():
    check_grad(lambda x: ((x * 3.0 + 2.0) * x).sum(), (3, 4))


def test_sub_div():
    check_grad(lambda x: ((x - 1.0) / (x * x + 2.0)).sum(), (3, 4))


def test_pow():
    check_grad(lambda x: (x ** 3).sum(), (3, 4))


def test_mean():
    check_grad(lambda x: x.mean(), (4, 5))


def test_reductions_with_axis():
    check_grad(lambda x: x.sum(axis=1).sum(), (3, 4))
    check_grad(lambda x: (x.sum(axis=0, keepdims=True) * 2).sum(), (3, 4))
    check_grad(lambda x: x.max(), (3, 4))
    check_grad(lambda x: x.max(axis=1).sum(), (3, 4))


# ----------------------------------------------------------------------
# elementwise math
# ----------------------------------------------------------------------
@pytest.mark.parametrize("fn", [
    lambda x: x.exp().sum(),
    lambda x: x.abs().log().sum(),
    lambda x: (x * x + 1).sqrt().sum(),
    lambda x: x.tanh().sum(),
    lambda x: x.sigmoid().sum(),
    lambda x: x.relu().sum(),
    lambda x: x.leaky_relu().sum(),
    lambda x: x.gelu().sum(),
])
def test_elementwise(fn):
    check_grad(fn, (3, 4))


# ----------------------------------------------------------------------
# linear algebra and shape ops
# ----------------------------------------------------------------------
def test_matmul():
    check_grad(lambda x: (x @ x.T).sum(), (4, 3))
    check_grad(lambda x: ((x @ x.T) ** 2).mean(), (4, 3))


def test_matmul_with_constant():
    w = np.random.default_rng(1).normal(size=(3, 2))
    check_grad(lambda x: (x @ w).sum(), (4, 3))


def test_shape_ops():
    check_grad(lambda x: (x.reshape(-1) ** 2).sum(), (3, 4))
    check_grad(lambda x: (x.T * 2).sum(), (3, 4))
    check_grad(lambda x: (x.expand_dims(0) * 3).sum(), (3, 4))
    check_grad(lambda x: (x[1] ** 2).sum(), (3, 4))
    check_grad(lambda x: x[[0, 0, 2]].sum(), (3, 4))
    check_grad(lambda x: Tensor.cat([x, x * 2], axis=0).sum(), (3, 4))
    check_grad(lambda x: Tensor.stack([x, x * 3], axis=0).sum(), (3, 4))


def test_broadcasting_reduces_correctly():
    check_grad(lambda x: (x + np.ones((1, 4))).sum(), (3, 4))
    check_grad(lambda x: (x * np.ones((3, 1))).sum(), (3, 4))


# ----------------------------------------------------------------------
# softmax family
# ----------------------------------------------------------------------
def test_softmax():
    check_grad(lambda x: (x.softmax(axis=1) ** 2).sum(), (3, 4))


def test_log_softmax():
    m = np.random.default_rng(2).normal(size=(3, 4))
    check_grad(lambda x: (x.log_softmax(axis=1) * m).sum(), (3, 4))


def test_cross_entropy_style():
    targets = np.array([0, 1, 2, 1])

    def f(x):
        lp = x.log_softmax(axis=1)
        return -lp[np.arange(4), targets].mean()

    check_grad(f, (4, 5))


# ----------------------------------------------------------------------
# engine behaviour
# ----------------------------------------------------------------------
def test_backward_accumulates():
    x = Tensor([2.0], requires_grad=True)
    y = x * x + x * x          # two paths through x
    y.backward()
    assert x.grad[0] == pytest.approx(8.0)   # d/dx (2x^2) = 4x = 8


def test_non_scalar_backward_requires_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        x.backward()


def test_non_scalar_backward_with_explicit_grad():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x * 2.0).backward(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(x.grad, 2.0)


def test_grad_is_none_without_requires_grad():
    x = Tensor([1.0, 2.0])
    y = (x * 3.0).sum()
    y.backward()
    assert x.grad is None
    assert not y.requires_grad


def test_no_grad_prunes_graph():
    with no_grad():
        assert not is_grad_enabled()
        z = Tensor(np.ones((2, 2)), requires_grad=True) * 2
        assert not z.requires_grad
        assert len(z._prev) == 0
    assert is_grad_enabled()


def test_enable_grad_inside_no_grad():
    with no_grad():
        with enable_grad():
            z = Tensor(np.ones(3), requires_grad=True) * 2
            assert z.requires_grad


def test_zero_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    (x * x).sum().backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None


def test_detach():
    x = Tensor([1.0, 2.0], requires_grad=True)
    d = (x * 3).detach()
    assert not d.requires_grad
    assert np.allclose(d.data, [3.0, 6.0])


def test_item_and_tolist():
    x = Tensor([[1.0, 2.0]])
    assert x.item() == pytest.approx(0.0) if x.size == 1 else True
    assert Tensor([5.0]).item() == pytest.approx(5.0)
    assert x.tolist() == [[1.0, 2.0]]


def test_item_rejects_non_scalar():
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0]).item()


def test_factories():
    assert Tensor.zeros((2, 3)).shape == (2, 3)
    assert Tensor.ones((2, 3)).data.sum() == 6.0
    assert Tensor.full((2, 2), 7.0).data[0, 0] == 7.0
    assert Tensor.arange(5).shape == (5,)
    assert Tensor.randn(4, 4).shape == (4, 4)


def test_masked_fill_blocks_gradient():
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    out = x.masked_fill(np.array([False, True, False]), 0.0)
    out.sum().backward()
    assert np.allclose(x.grad, [1.0, 0.0, 1.0])
