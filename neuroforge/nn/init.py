"""Weight initialisation schemes.

Getting initialisation right matters more than people expect: too small and the
signal vanishes through the layers, too large and it explodes. These helpers
implement the standard fixes (Xavier/Glorot for tanh-like nets, Kaiming/He for
ReLU nets).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "zeros_",
    "ones_",
    "normal_",
    "uniform_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
]


def zeros_(tensor) -> None:
    tensor.data[...] = 0.0


def ones_(tensor) -> None:
    tensor.data[...] = 1.0


def normal_(tensor, mean: float = 0.0, std: float = 0.01) -> None:
    tensor.data[...] = np.random.normal(mean, std, size=tensor.shape)


def uniform_(tensor, low: float = -0.05, high: float = 0.05) -> None:
    tensor.data[...] = np.random.uniform(low, high, size=tensor.shape)


def _fan_in_out(tensor):
    """Return (fan_in, fan_out) treating every dim after the first as input."""
    if tensor.ndim < 2:
        fan_in = fan_out = tensor.shape[0] if tensor.ndim else 1
    else:
        num_input = int(np.prod(tensor.shape[1:]))
        num_output = tensor.shape[0]
        fan_in, fan_out = num_input, num_output
    return fan_in, fan_out


def xavier_uniform_(tensor, gain: float = 1.0) -> None:
    fan_in, fan_out = _fan_in_out(tensor)
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    tensor.data[...] = np.random.uniform(-limit, limit, size=tensor.shape)


def xavier_normal_(tensor, gain: float = 1.0) -> None:
    fan_in, fan_out = _fan_in_out(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data[...] = np.random.normal(0.0, std, size=tensor.shape)


def kaiming_uniform_(tensor, a: float = 0.0, mode: str = "fan_in") -> None:
    fan_in, fan_out = _fan_in_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = np.sqrt(2.0 / (1.0 + a * a))
    bound = gain * np.sqrt(3.0 / fan)
    tensor.data[...] = np.random.uniform(-bound, bound, size=tensor.shape)


def kaiming_normal_(tensor, a: float = 0.0, mode: str = "fan_in") -> None:
    fan_in, fan_out = _fan_in_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = np.sqrt(2.0 / (1.0 + a * a))
    std = gain / np.sqrt(fan)
    tensor.data[...] = np.random.normal(0.0, std, size=tensor.shape)
