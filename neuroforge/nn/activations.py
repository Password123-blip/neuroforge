"""Activation functions, both as free functions and as ``Module`` wrappers."""

from __future__ import annotations

from ..tensor import Tensor
from .module import Module

__all__ = [
    "relu", "leaky_relu", "gelu", "tanh", "sigmoid", "softmax", "log_softmax",
    "ReLU", "LeakyReLU", "GELU", "Tanh", "Sigmoid", "Softmax", "Identity",
]


# ----------------------------------------------------------------------
# functional forms
# ----------------------------------------------------------------------
def relu(x: Tensor) -> Tensor:
    return x.relu()


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return x.leaky_relu(negative_slope)


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax(axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis=axis)


# ----------------------------------------------------------------------
# module forms
# ----------------------------------------------------------------------
class Identity(Module):
    """Passes its input straight through."""

    def forward(self, x):
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(self.negative_slope)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)
