"""Optimisers: SGD, SGD with momentum, RMSProp, Adam and AdamW.

Every optimiser follows the same contract: hold a reference to a list of
parameters, and on each :meth:`step` read ``param.grad`` and update
``param.data`` in place.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Optimizer", "SGD", "RMSprop", "Adam", "AdamW"]


class Optimizer:
    """Base class holding the parameter list and shared bookkeeping."""

    def __init__(self, params, lr: float = 1e-3):
        self.params = [p for p in params]
        if not self.params:
            raise ValueError("optimizer received an empty parameter list")
        self.lr = lr
        self._step_count = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"lr": self.lr, "step": self._step_count}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(lr={self.lr}, params={len(self.params)})"


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum and L2 decay.

    ``momentum`` smooths updates by accumulating an exponential moving average
    of past gradients. ``nesterov`` makes a small look-ahead correction.
    """

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        super().__init__(params, lr)
        if nesterov and momentum <= 0:
            raise ValueError("nesterov requires momentum > 0")
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self._step_count += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                v = self.velocity[i]
                v *= self.momentum
                v += g
                if self.nesterov:
                    g = g + self.momentum * v
                else:
                    g = v
            p.data = p.data - self.lr * g


class RMSprop(Optimizer):
    """Divide the learning rate by a running RMS of recent gradients."""

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.square_avg = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self._step_count += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            s = self.square_avg[i]
            s *= self.alpha
            s += (1 - self.alpha) * g * g
            p.data = p.data - self.lr * g / (np.sqrt(s) + self.eps)


class Adam(Optimizer):
    """Adaptive Moment Estimation — the default choice for most work.

    Keeps a running mean (``m``) and uncentered variance (``v``) of the
    gradient, bias-corrects both, and steps along ``m / sqrt(v)``.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self._step_count += 1
        t = self._step_count
        b1, b2 = self.beta1, self.beta2
        bias1 = 1.0 - b1 ** t
        bias2 = 1.0 - b2 ** t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            m, v = self.m[i], self.v[i]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / bias1
            v_hat = v / bias2
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """Adam with *decoupled* weight decay.

    In plain Adam, L2 regularisation is folded into the gradient and therefore
    scaled by the adaptive denominator. AdamW applies the decay directly to the
    weights instead, which is why it is the standard optimiser for transformers.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self._step_count += 1
        t = self._step_count
        b1, b2 = self.beta1, self.beta2
        bias1 = 1.0 - b1 ** t
        bias2 = 1.0 - b2 ** t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.weight_decay:
                p.data = p.data - self.lr * self.weight_decay * p.data
            g = p.grad
            m, v = self.m[i], self.v[i]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / bias1
            v_hat = v / bias2
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
