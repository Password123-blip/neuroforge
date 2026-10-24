"""Multi-layer perceptron (fully-connected feed-forward network)."""

from __future__ import annotations

from ..tensor import Tensor
from ..nn.activations import ReLU
from ..nn.layers import Dropout, Linear
from ..nn.module import Module, Sequential

__all__ = ["MLP"]


class MLP(Module):
    """A stack of fully-connected layers with activations between them.

    Args:
        in_features: size of each input vector.
        hidden_sizes: widths of the hidden layers, e.g. ``(256, 128)``.
        out_features: size of the output vector.
        activation: a ``Module`` subclass (not an instance), e.g. ``nn.GELU``.
        dropout: dropout probability applied after each hidden activation.
        out_activation: optional ``Module`` subclass for the final layer
            (e.g. ``nn.Sigmoid`` for binary classification).
    """

    def __init__(self, in_features: int, hidden_sizes=(128,), out_features: int = 10,
                 activation=ReLU, dropout: float = 0.0, out_activation=None):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        sizes = [in_features, *hidden_sizes, out_features]

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(Linear(sizes[i], sizes[i + 1]))
            last = i == len(sizes) - 2
            if last:
                if out_activation is not None:
                    layers.append(out_activation())
            else:
                layers.append(activation())
                if dropout:
                    layers.append(Dropout(dropout))

        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
