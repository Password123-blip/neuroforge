"""A small reverse-mode automatic differentiation engine built on NumPy.

This is the foundation of NeuroForge. Every operation you can perform on a
:class:`Tensor` records how to propagate gradients backwards through itself, so
that calling :meth:`Tensor.backward` walks the whole computation graph and fills
in ``.grad`` for every tensor that participated.

The design is deliberately close to the "define-by-run" style used by real
frameworks: you write ordinary Python expressions, and the graph is built as a
side effect.

    >>> from neuroforge.tensor import Tensor
    >>> x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> y = (x * x).sum()
    >>> y.backward()
    >>> x.grad
    array([[2., 4.],
           [6., 8.]])

Nothing here depends on PyTorch, JAX or TensorFlow — only NumPy.
"""

from __future__ import annotations

import contextlib
import numpy as np

__all__ = ["Tensor", "no_grad", "enable_grad", "is_grad_enabled"]

# Global switch flipped by the `no_grad` context manager. When False, newly
# created tensors never require gradients, which makes inference cheap.
_GRAD_ENABLED = [True]


def is_grad_enabled() -> bool:
    """Return True if gradient tracking is currently switched on."""
    return _GRAD_ENABLED[0]


@contextlib.contextmanager
def no_grad():
    """Context manager that disables graph construction inside the block."""
    prev = _GRAD_ENABLED[0]
    _GRAD_ENABLED[0] = False
    try:
        yield
    finally:
        _GRAD_ENABLED[0] = prev


@contextlib.contextmanager
def enable_grad():
    """Context manager that forces graph construction on inside the block."""
    prev = _GRAD_ENABLED[0]
    _GRAD_ENABLED[0] = True
    try:
        yield
    finally:
        _GRAD_ENABLED[0] = prev


def _accumulate(tensor: "Tensor", grad) -> None:
    """Add ``grad`` into ``tensor.grad``, allocating on first write."""
    if grad is None:
        return
    grad = np.asarray(grad, dtype=np.float64)
    if tensor.grad is None:
        tensor.grad = np.array(grad, dtype=np.float64, copy=True)
    else:
        tensor.grad = tensor.grad + grad


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` by summing over broadcast dimensions.

    NumPy silently broadcasts shapes during arithmetic; the gradient of a
    broadcast is a sum over the axes that were stretched.
    """
    grad = np.asarray(grad)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _ensure(value) -> "Tensor":
    """Wrap plain numbers / arrays in a Tensor (no-op for Tensors)."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


class Tensor:
    """An N-dimensional array that remembers how to differentiate itself.

    Attributes:
        data: the underlying :class:`numpy.ndarray` (always float64).
        grad: gradient of some scalar with respect to this tensor, or None.
        requires_grad: whether gradients should flow through this tensor.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad: bool = False, _children=(), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad) and _GRAD_ENABLED[0]
        self.grad = None
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def detach(self) -> "Tensor":
        """Return a new tensor sharing the data but cut off from the graph."""
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        """Clear the accumulated gradient."""
        self.grad = None

    def item(self) -> float:
        """Return the single Python float held by a scalar tensor."""
        if self.data.size != 1:
            raise ValueError("item() is only valid for single-element tensors")
        return float(self.data.reshape(-1)[0])

    def numpy(self) -> np.ndarray:
        return self.data

    def tolist(self):
        return self.data.tolist()

    def astype(self, dtype) -> "Tensor":
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad)

    def _new(self, data, children, op: str) -> "Tensor":
        """Create a child tensor, inheriting the grad requirement of inputs.

        When no input requires a gradient the graph is pruned immediately: the
        result is returned with no parents at all. This keeps inference (and
        any ``no_grad`` block) from building a graph it will never use.
        """
        if not any(c.requires_grad for c in children):
            return Tensor(data, requires_grad=False, _op=op)
        return Tensor(data, requires_grad=True, _children=children, _op=op)

    # ------------------------------------------------------------------
    # factories
    # ------------------------------------------------------------------
    @staticmethod
    def zeros(shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def full(shape, value, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.full(shape, value), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, std: float = 1.0, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.random.randn(*shape) * std, requires_grad=requires_grad)

    @staticmethod
    def uniform(shape, low: float = 0.0, high: float = 1.0,
                requires_grad: bool = False) -> "Tensor":
        return Tensor(np.random.uniform(low, high, size=shape),
                      requires_grad=requires_grad)

    @staticmethod
    def arange(*args, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.arange(*args), requires_grad=requires_grad)

    @staticmethod
    def from_numpy(array, requires_grad: bool = False) -> "Tensor":
        return Tensor(array, requires_grad=requires_grad)

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other) -> "Tensor":
        other = _ensure(other)
        out = self._new(self.data + other.data, (self, other), "+")

        def _backward():
            _accumulate(self, _unbroadcast(out.grad, self.shape))
            _accumulate(other, _unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __sub__(self, other) -> "Tensor":
        other = _ensure(other)
        out = self._new(self.data - other.data, (self, other), "-")

        def _backward():
            _accumulate(self, _unbroadcast(out.grad, self.shape))
            _accumulate(other, _unbroadcast(-out.grad, other.shape))

        out._backward = _backward
        return out

    def __rsub__(self, other) -> "Tensor":
        return _ensure(other) - self

    def __mul__(self, other) -> "Tensor":
        other = _ensure(other)
        out = self._new(self.data * other.data, (self, other), "*")

        def _backward():
            _accumulate(self, _unbroadcast(out.grad * other.data, self.shape))
            _accumulate(other, _unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __truediv__(self, other) -> "Tensor":
        other = _ensure(other)
        out = self._new(self.data / other.data, (self, other), "/")

        def _backward():
            _accumulate(self, _unbroadcast(out.grad / other.data, self.shape))
            _accumulate(
                other,
                _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape),
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other) -> "Tensor":
        return _ensure(other) / self

    def __pow__(self, exponent) -> "Tensor":
        if not isinstance(exponent, (int, float)):
            raise TypeError("only scalar exponents are supported")
        out = self._new(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            _accumulate(self, out.grad * exponent * (self.data ** (exponent - 1)))

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __matmul__(self, other) -> "Tensor":
        other = _ensure(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError("matmul expects tensors with at least 2 dimensions")
        out = self._new(self.data @ other.data, (self, other), "@")

        def _backward():
            g = out.grad
            _accumulate(self, _unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            _accumulate(other, _unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))

        out._backward = _backward
        return out

    def __rmatmul__(self, other) -> "Tensor":
        return _ensure(other) @ self

    # ------------------------------------------------------------------
    # reductions
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = self._new(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is None:
                _accumulate(self, np.broadcast_to(g, self.shape))
                return
            axes = axis if isinstance(axis, tuple) else (axis,)
            if not keepdims:
                for ax in sorted(a % self.ndim for a in axes):
                    g = np.expand_dims(g, ax)
            _accumulate(self, np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = 1
            for a in axes:
                n *= self.shape[a % self.ndim]
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        """Maximum along an axis. Ties split the gradient evenly."""
        out = self._new(self.data.max(axis=axis, keepdims=keepdims), (self,), "max")
        if axis is None:
            mask = self.data == out.data
            count = mask.sum()

            def _backward():
                _accumulate(self, np.broadcast_to(out.grad, self.shape) * mask / count)

            out._backward = _backward
            return out

        keep = self.data.max(axis=axis, keepdims=True)
        mask = self.data == keep
        count = mask.sum(axis=axis, keepdims=True)
        axes = axis if isinstance(axis, tuple) else (axis,)

        def _backward():
            g = out.grad
            if not keepdims:
                for ax in sorted(a % self.ndim for a in axes):
                    g = np.expand_dims(g, ax)
            _accumulate(self, np.broadcast_to(g, self.shape) * mask / count)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # elementwise math
    # ------------------------------------------------------------------
    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = self._new(e, (self,), "exp")
        out._backward = lambda: _accumulate(self, out.grad * e)
        return out

    def log(self) -> "Tensor":
        out = self._new(np.log(self.data), (self,), "log")
        out._backward = lambda: _accumulate(self, out.grad / self.data)
        return out

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def abs(self) -> "Tensor":
        out = self._new(np.abs(self.data), (self,), "abs")
        out._backward = lambda: _accumulate(self, out.grad * np.sign(self.data))
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = self._new(t, (self,), "tanh")
        out._backward = lambda: _accumulate(self, out.grad * (1.0 - t * t))
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._new(s, (self,), "sigmoid")
        out._backward = lambda: _accumulate(self, out.grad * s * (1.0 - s))
        return out

    def relu(self) -> "Tensor":
        mask = self.data > 0
        out = self._new(self.data * mask, (self,), "relu")
        out._backward = lambda: _accumulate(self, out.grad * mask)
        return out

    def leaky_relu(self, slope: float = 0.01) -> "Tensor":
        mask = self.data > 0
        out = self._new(np.where(mask, self.data, self.data * slope), (self,), "leaky_relu")
        out._backward = lambda: _accumulate(
            self, out.grad * np.where(mask, 1.0, slope)
        )
        return out

    def gelu(self) -> "Tensor":
        """Tanh approximation of GELU (as used in GPT-2)."""
        c = np.sqrt(2.0 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = self._new(0.5 * x * (1.0 + t), (self,), "gelu")
        d_inner = c * (1.0 + 3 * 0.044715 * x ** 2)
        grad_local = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
        out._backward = lambda: _accumulate(self, out.grad * grad_local)
        return out

    # ------------------------------------------------------------------
    # shape ops
    # ------------------------------------------------------------------
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original = self.shape
        out = self._new(self.data.reshape(shape), (self,), "reshape")
        out._backward = lambda: _accumulate(self, out.grad.reshape(original))
        return out

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        out = self._new(np.transpose(self.data, axes), (self,), "transpose")
        inverse = np.argsort(axes)
        out._backward = lambda: _accumulate(self, np.transpose(out.grad, inverse))
        return out

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def expand_dims(self, axis: int) -> "Tensor":
        out = self._new(np.expand_dims(self.data, axis), (self,), "expand_dims")
        out._backward = lambda: _accumulate(self, np.squeeze(out.grad, axis))
        return out

    def squeeze(self, axis=None) -> "Tensor":
        out = self._new(np.squeeze(self.data, axis=axis), (self,), "squeeze")
        out._backward = lambda: _accumulate(self, out.grad.reshape(self.shape))
        return out

    def __getitem__(self, index) -> "Tensor":
        out = self._new(self.data[index], (self,), "getitem")

        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, index, out.grad)
            _accumulate(self, g)

        out._backward = _backward
        return out

    def masked_fill(self, mask: np.ndarray, value: float) -> "Tensor":
        """Return a copy with positions where ``mask`` is True set to ``value``.

        Gradient does not flow through the filled positions.
        """
        mask = np.asarray(mask, dtype=bool)
        data = np.where(mask, value, self.data)
        out = self._new(data, (self,), "masked_fill")
        out._backward = lambda: _accumulate(self, np.where(mask, 0.0, out.grad))
        return out

    # ------------------------------------------------------------------
    # multi-tensor ops
    # ------------------------------------------------------------------
    @staticmethod
    def cat(tensors, axis: int = 0) -> "Tensor":
        tensors = [_ensure(t) for t in tensors]
        rg = any(t.requires_grad for t in tensors)
        out = Tensor(
            np.concatenate([t.data for t in tensors], axis=axis),
            requires_grad=rg,
            _children=tuple(tensors) if rg else (),
            _op="cat",
        )
        if not rg:
            return out
        sizes = [t.shape[axis] for t in tensors]
        cuts = np.cumsum(sizes)[:-1]

        def _backward():
            for t, g in zip(tensors, np.split(out.grad, cuts, axis=axis)):
                _accumulate(t, g)

        out._backward = _backward
        return out

    @staticmethod
    def stack(tensors, axis: int = 0) -> "Tensor":
        tensors = [_ensure(t) for t in tensors]
        rg = any(t.requires_grad for t in tensors)
        out = Tensor(
            np.stack([t.data for t in tensors], axis=axis),
            requires_grad=rg,
            _children=tuple(tensors) if rg else (),
            _op="stack",
        )
        if not rg:
            return out

        def _backward():
            for i, t in enumerate(tensors):
                _accumulate(t, np.take(out.grad, i, axis=axis))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # differentiable primitives used by the nn package
    # ------------------------------------------------------------------
    def log_softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable log-softmax along ``axis``."""
        x = self.data
        shifted = x - x.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        logp = shifted - lse
        out = self._new(logp, (self,), "log_softmax")
        soft = np.exp(logp)

        def _backward():
            g = out.grad
            _accumulate(self, g - soft * g.sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable softmax along ``axis``."""
        x = self.data
        shifted = x - x.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = self._new(s, (self,), "softmax")

        def _backward():
            g = out.grad
            _accumulate(self, s * (g - (g * s).sum(axis=axis, keepdims=True)))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # backprop
    # ------------------------------------------------------------------
    def backward(self, grad=None) -> None:
        """Backpropagate from this (usually scalar) tensor through the graph."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError(
                    "grad must be provided when calling backward() on a "
                    f"non-scalar tensor of shape {self.shape}"
                )
            grad = np.ones_like(self.data)
        self.grad = np.array(grad, dtype=np.float64, copy=True)

        topo, visited = [], set()

        def build(t: "Tensor"):
            if id(t) in visited:
                return
            visited.add(id(t))
            for parent in t._prev:
                build(parent)
            topo.append(t)

        build(self)
        for node in reversed(topo):
            node._backward()
