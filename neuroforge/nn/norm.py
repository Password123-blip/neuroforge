"""Normalisation layers: LayerNorm, RMSNorm and BatchNorm1d.

These are written as compositions of ordinary tensor operations rather than
custom backward functions — a good demonstration of how much the autograd
engine buys you.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from .module import Module, Parameter

__all__ = ["LayerNorm", "RMSNorm", "BatchNorm1d"]


class LayerNorm(Module):
    """Normalise over the last dimension, then scale and shift.

    Used in essentially every modern transformer.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape))
            self.bias = Parameter(np.zeros(self.normalized_shape))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered * centered).mean(axis=-1, keepdims=True)
        out = centered / (var + self.eps).sqrt()
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out


class RMSNorm(Module):
    """Root-mean-square normalisation (as used by LLaMA and others).

    Cheaper than LayerNorm: it skips mean-centering entirely.
    """

    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(self.normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        ms = (x * x).mean(axis=-1, keepdims=True)
        out = x / (ms + self.eps).sqrt()
        return out * self.weight


class BatchNorm1d(Module):
    """Normalise each feature across the batch, keeping running statistics."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("BatchNorm1d expects a 2-D (N, features) input")
        if self.training:
            mu = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mu = self.running_mean
            var = self.running_var

        centered = x - mu
        out = centered / np.sqrt(var + self.eps)
        return out * self.weight + self.bias
