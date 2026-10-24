"""Loss functions.

Each is a plain function over :class:`~neuroforge.tensor.Tensor` objects, so the
result is a differentiable scalar you can call ``.backward()`` on. ``Module``
wrappers are provided for the common ones so they can be dropped into
``Sequential``-style code if you like.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from .module import Module

__all__ = [
    "mse_loss", "l1_loss", "huber_loss", "cross_entropy", "nll_loss",
    "binary_cross_entropy_with_logits", "MSELoss", "CrossEntropyLoss",
    "BCEWithLogitsLoss",
]


def _softplus(x: Tensor) -> Tensor:
    """log(1 + exp(x)), computed in a numerically stable way."""
    return (1.0 + (-x.abs()).exp()).log() + x.relu()


def mse_loss(prediction: Tensor, target) -> Tensor:
    """Mean squared error, averaged over every element."""
    target = target if isinstance(target, Tensor) else Tensor(target)
    diff = prediction - target
    return (diff * diff).mean()


def l1_loss(prediction: Tensor, target) -> Tensor:
    """Mean absolute error."""
    target = target if isinstance(target, Tensor) else Tensor(target)
    return (prediction - target).abs().mean()


def huber_loss(prediction: Tensor, target, delta: float = 1.0) -> Tensor:
    """Huber loss: quadratic for small errors, linear for large ones."""
    target = target if isinstance(target, Tensor) else Tensor(target)
    diff = (prediction - target).abs()
    quadratic = (diff * diff).masked_fill(diff.data > delta, 0.0)
    linear = (diff * 2.0 * delta - delta * delta).masked_fill(diff.data <= delta, 0.0)
    return (0.5 * quadratic + 0.5 * linear).mean()


def nll_loss(log_probs: Tensor, targets) -> Tensor:
    """Negative log-likelihood given *log* probabilities of shape (N, C)."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    n = log_probs.shape[0]
    picked = log_probs[np.arange(n), targets]
    return -picked.mean()


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Softmax cross-entropy for ``(N, C)`` logits and integer targets.

    The log-softmax is fused in, which is both faster and far more numerically
    stable than computing softmax and then taking a log.
    """
    if logits.ndim != 2:
        raise ValueError("cross_entropy expects logits of shape (N, num_classes)")
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if targets.shape[0] != logits.shape[0]:
        raise ValueError(
            f"got {logits.shape[0]} logits but {targets.shape[0]} targets"
        )
    return nll_loss(logits.log_softmax(axis=-1), targets)


def binary_cross_entropy_with_logits(logits: Tensor, targets) -> Tensor:
    """Binary cross-entropy that takes raw logits (stable softplus form)."""
    targets = targets if isinstance(targets, Tensor) else Tensor(targets)
    loss = targets * _softplus(-logits) + (1.0 - targets) * _softplus(logits)
    return loss.mean()


class MSELoss(Module):
    def forward(self, prediction, target):
        return mse_loss(prediction, target)


class CrossEntropyLoss(Module):
    def forward(self, logits, targets):
        return cross_entropy(logits, targets)


class BCEWithLogitsLoss(Module):
    def forward(self, logits, targets):
        return binary_cross_entropy_with_logits(logits, targets)
