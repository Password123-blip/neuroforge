"""Neural-network layers: Linear, Conv2d, Embedding, Dropout, pooling and more.

Every layer is a :class:`~neuroforge.nn.module.Module` whose ``forward`` is
written entirely in terms of :class:`~neuroforge.tensor.Tensor` operations, so
gradients come for free from the autograd engine.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor, _accumulate
from . import init
from .module import Module, Parameter

__all__ = [
    "Linear",
    "Embedding",
    "Dropout",
    "Flatten",
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
]


# ----------------------------------------------------------------------
# helpers implemented as custom differentiable operations
# ----------------------------------------------------------------------
def _im2col(x: Tensor, kh: int, kw: int, stride: int, pad: int) -> Tensor:
    """Rearrange image patches into columns so convolution becomes a matmul."""
    xd = x.data
    n, c, h, w = xd.shape
    hp, wp = h + 2 * pad, w + 2 * pad
    out_h = (hp - kh) // stride + 1
    out_w = (wp - kw) // stride + 1
    xp = np.pad(xd, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

    cols = np.empty((n, c, kh, kw, out_h, out_w), dtype=xd.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = xp[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.reshape(n, c * kh * kw, out_h * out_w)

    out = Tensor(
        cols,
        requires_grad=x.requires_grad,
        _children=(x,) if x.requires_grad else (),
        _op="im2col",
    )
    if not x.requires_grad:
        return out

    def _backward():
        dxp = np.zeros((n, c, hp, wp), dtype=np.float64)
        g = out.grad.reshape(n, c, kh, kw, out_h, out_w)
        for i in range(kh):
            i_max = i + stride * out_h
            for j in range(kw):
                j_max = j + stride * out_w
                dxp[:, :, i:i_max:stride, j:j_max:stride] += g[:, :, i, j, :, :]
        dx = dxp[:, :, pad:pad + h, pad:pad + w] if pad else dxp
        _accumulate(x, dx)

    out._backward = _backward
    return out


def _max_pool2d(x: Tensor, kh: int, kw: int, stride: int) -> Tensor:
    xd = x.data
    n, c, h, w = xd.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    out = np.empty((n, c, out_h, out_w), dtype=xd.dtype)
    argmax = np.empty((n, c, out_h, out_w), dtype=np.int64)
    for i in range(out_h):
        for j in range(out_w):
            window = xd[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            flat = window.reshape(n, c, -1)
            idx = flat.argmax(axis=-1)
            argmax[:, :, i, j] = idx
            out[:, :, i, j] = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]

    result = Tensor(
        out,
        requires_grad=x.requires_grad,
        _children=(x,) if x.requires_grad else (),
        _op="max_pool2d",
    )
    if not x.requires_grad:
        return result

    def _backward():
        dx = np.zeros_like(xd)
        g = result.grad
        for i in range(out_h):
            for j in range(out_w):
                idx = argmax[:, :, i, j]
                for k in range(kh):
                    for m in range(kw):
                        sel = idx == (k * kw + m)
                        if sel.any():
                            dx[:, :, i * stride + k, j * stride + m] += np.where(
                                sel, g[:, :, i, j], 0.0
                            )
        _accumulate(x, dx)

    result._backward = _backward
    return result


def _avg_pool2d(x: Tensor, kh: int, kw: int, stride: int) -> Tensor:
    xd = x.data
    n, c, h, w = xd.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    out = np.zeros((n, c, out_h, out_w), dtype=xd.dtype)
    for i in range(out_h):
        for j in range(out_w):
            window = xd[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, :, i, j] = window.mean(axis=(2, 3))

    result = Tensor(
        out,
        requires_grad=x.requires_grad,
        _children=(x,) if x.requires_grad else (),
        _op="avg_pool2d",
    )
    if not x.requires_grad:
        return result

    scale = 1.0 / (kh * kw)

    def _backward():
        dx = np.zeros_like(xd)
        g = result.grad * scale
        for i in range(out_h):
            for j in range(out_w):
                dx[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw] += g[:, :, i, j][:, :, None, None]
        _accumulate(x, dx)

    result._backward = _backward
    return result


# ----------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------
class Linear(Module):
    """A fully-connected layer: ``y = x @ W.T + b``.

    Accepts inputs of shape ``(..., in_features)`` and produces
    ``(..., out_features)``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.empty((out_features, in_features)))
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if bias:
            bound = 1.0 / np.sqrt(in_features)
            self.bias = Parameter(np.random.uniform(-bound, bound, size=out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            out = (self.weight @ x.reshape(-1, 1)).reshape(-1)
        else:
            out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """A lookup table mapping integer token ids to dense vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(np.empty((num_embeddings, embedding_dim)))
        init.normal_(self.weight, 0.0, 1.0)

    def forward(self, indices) -> Tensor:
        idx = np.asarray(indices, dtype=np.int64)
        return self.weight[idx]


class Dropout(Module):
    """Inverted dropout: active only while ``training`` is True."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = np.random.rand(*x.shape) >= self.p
        scale = 1.0 / (1.0 - self.p)
        return x.masked_fill(~keep, 0.0) * scale


class Flatten(Module):
    """Collapse every dimension except the batch dimension."""

    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = (x.shape[0], -1) if self.start_dim == 1 else x.shape
        return x.reshape(shape)


class Conv2d(Module):
    """2-D convolution over ``(N, C_in, H, W)`` inputs.

    Implemented with the im2col trick, so the heavy lifting is a matrix
    multiply that the autograd engine already knows how to differentiate.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            kh, kw = kernel_size
        else:
            kh = kw = int(kernel_size)
        if isinstance(stride, (tuple, list)):
            sh, sw = stride
        else:
            sh = sw = int(stride)
        if isinstance(padding, (tuple, list)):
            ph, pw = padding
        else:
            ph = pw = int(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.padding = (ph, pw)

        self.weight = Parameter(np.empty((out_channels, in_channels, kh, kw)))
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if bias:
            bound = 1.0 / np.sqrt(in_channels * kh * kw)
            self.bias = Parameter(np.random.uniform(-bound, bound, size=out_channels))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        if sh != sw or ph != pw:
            raise NotImplementedError("asymmetric stride/padding is not supported yet")

        n = x.shape[0]
        cols = _im2col(x, kh, kw, sh, ph)          # (N, C*kh*kw, L)
        cols = cols.transpose(0, 2, 1)             # (N, L, C*kh*kw)
        length = cols.shape[1]

        w = self.weight.reshape(self.out_channels, -1)   # (C_out, K)
        out = cols @ w.T                                 # (N, L, C_out)
        if self.bias is not None:
            out = out + self.bias

        h_out = (x.shape[2] + 2 * ph - kh) // sh + 1
        w_out = (x.shape[3] + 2 * pw - kw) // sw + 1
        out = out.reshape(n, h_out, w_out, self.out_channels)
        return out.transpose(0, 3, 1, 2)                 # (N, C_out, H_out, W_out)


class MaxPool2d(Module):
    """Max pooling over ``(N, C, H, W)`` inputs."""

    def __init__(self, kernel_size, stride=None, padding: int = 0):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        else:
            self.kernel_size = (int(kernel_size), int(kernel_size))
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, (tuple, list)):
            self.stride = (int(stride[0]), int(stride[1]))
        else:
            self.stride = (int(stride), int(stride))
        self.padding = int(padding)
        if self.padding:
            raise NotImplementedError("padding in MaxPool2d is not supported yet")

    def forward(self, x: Tensor) -> Tensor:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        if sh != sw:
            raise NotImplementedError("asymmetric stride is not supported yet")
        return _max_pool2d(x, kh, kw, sh)


class AvgPool2d(Module):
    """Average pooling over ``(N, C, H, W)`` inputs."""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        else:
            self.kernel_size = (int(kernel_size), int(kernel_size))
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, (tuple, list)):
            self.stride = (int(stride[0]), int(stride[1]))
        else:
            self.stride = (int(stride), int(stride))

    def forward(self, x: Tensor) -> Tensor:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        if sh != sw:
            raise NotImplementedError("asymmetric stride is not supported yet")
        return _avg_pool2d(x, kh, kw, sh)
