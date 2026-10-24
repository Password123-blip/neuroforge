"""Recurrent networks: vanilla RNN, LSTM and GRU, built from scratch.

Sequences are handled by looping over time steps in Python. That is slow
compared to a fused kernel, but it is completely transparent: every gate is an
ordinary differentiable expression, so the autograd engine handles
backpropagation-through-time for free.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from ..nn.layers import Dropout, Linear
from ..nn.module import Module, ModuleList, Parameter
from ..nn import init

__all__ = ["RNNCell", "LSTMCell", "GRUCell", "RNN", "LSTM", "GRU", "SequenceClassifier"]


class RNNCell(Module):
    """Elman cell: ``h' = activation(W_ih x + W_hh h + b)``."""

    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = "tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(np.empty((hidden_size, input_size)))
        self.weight_hh = Parameter(np.empty((hidden_size, hidden_size)))
        self.bias = Parameter(np.zeros(hidden_size))
        bound = 1.0 / np.sqrt(hidden_size)
        init.uniform_(self.weight_ih, -bound, bound)
        init.uniform_(self.weight_hh, -bound, bound)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        pre = x @ self.weight_ih.T + h @ self.weight_hh.T + self.bias
        return pre.relu() if self.nonlinearity == "relu" else pre.tanh()


class LSTMCell(Module):
    """Long Short-Term Memory cell with input/forget/output gates."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(np.empty((4 * hidden_size, input_size)))
        self.weight_hh = Parameter(np.empty((4 * hidden_size, hidden_size)))
        self.bias = Parameter(np.zeros(4 * hidden_size))
        bound = 1.0 / np.sqrt(hidden_size)
        init.uniform_(self.weight_ih, -bound, bound)
        init.uniform_(self.weight_hh, -bound, bound)

    def forward(self, x: Tensor, h: Tensor, c: Tensor):
        gates = x @ self.weight_ih.T + h @ self.weight_hh.T + self.bias
        hsz = self.hidden_size
        i = gates[:, :hsz].sigmoid()
        f = gates[:, hsz:2 * hsz].sigmoid()
        g = gates[:, 2 * hsz:3 * hsz].tanh()
        o = gates[:, 3 * hsz:].sigmoid()
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        return h_new, c_new


class GRUCell(Module):
    """Gated Recurrent Unit — a lighter alternative to LSTM."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(np.empty((3 * hidden_size, input_size)))
        self.weight_hh = Parameter(np.empty((3 * hidden_size, hidden_size)))
        self.bias_ih = Parameter(np.zeros(3 * hidden_size))
        self.bias_hh = Parameter(np.zeros(3 * hidden_size))
        bound = 1.0 / np.sqrt(hidden_size)
        init.uniform_(self.weight_ih, -bound, bound)
        init.uniform_(self.weight_hh, -bound, bound)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        hsz = self.hidden_size
        gi = x @ self.weight_ih.T + self.bias_ih
        gh = h @ self.weight_hh.T + self.bias_hh
        r = (gi[:, :hsz] + gh[:, :hsz]).sigmoid()
        z = (gi[:, hsz:2 * hsz] + gh[:, hsz:2 * hsz]).sigmoid()
        n = (gi[:, 2 * hsz:] + r * gh[:, 2 * hsz:]).tanh()
        return (1.0 - z) * n + z * h


class RNN(Module):
    """A multi-layer recurrent network.

    Args:
        input_size: feature size of each time step.
        hidden_size: size of the hidden state.
        num_layers: number of stacked recurrent layers.
        cell_type: ``"rnn"``, ``"lstm"`` or ``"gru"``.
        dropout: dropout applied to the output of each layer (except the last).
        return_sequences: if True, return ``(B, T, H)``; otherwise the final
            hidden state ``(B, H)``.
        bidirectional: run a second pass over the reversed sequence and
            concatenate the two hidden states.

    Inputs are ``(batch, time, features)``.
    """

    _CELLS = {"rnn": RNNCell, "lstm": LSTMCell, "gru": GRUCell}

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 cell_type: str = "lstm", dropout: float = 0.0,
                 return_sequences: bool = False, bidirectional: bool = False):
        super().__init__()
        if cell_type not in self._CELLS:
            raise ValueError(f"unknown cell_type '{cell_type}'")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional

        self.forward_cells = ModuleList()
        self.backward_cells = ModuleList()
        for layer in range(num_layers):
            layer_in = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            self.forward_cells.append(self._CELLS[cell_type](layer_in, hidden_size))
            if bidirectional:
                self.backward_cells.append(self._CELLS[cell_type](layer_in, hidden_size))

        self.dropouts = ModuleList()
        for _ in range(max(0, num_layers - 1)):
            self.dropouts.append(Dropout(dropout) if dropout else _Identity())

    def _run_direction(self, x: Tensor, cells, reverse: bool):
        batch, steps, _ = x.shape
        hsz = self.hidden_size
        is_lstm = self.cell_type == "lstm"

        h = Tensor.zeros((batch, hsz))
        c = Tensor.zeros((batch, hsz)) if is_lstm else None
        outputs = []

        order = range(steps - 1, -1, -1) if reverse else range(steps)
        for t in order:
            xt = x[:, t, :]
            if is_lstm:
                h, c = cells(xt, h, c)
            else:
                h = cells(xt, h)
            outputs.append(h)

        if reverse:
            outputs = outputs[::-1]
        if self.return_sequences:
            return Tensor.stack(outputs, axis=1)
        return h

    def _run_layer(self, x: Tensor, layer: int) -> Tensor:
        forward_out = self._run_direction(x, self.forward_cells[layer], reverse=False)
        if not self.bidirectional:
            return forward_out
        backward_out = self._run_direction(x, self.backward_cells[layer], reverse=True)
        if self.return_sequences:
            return Tensor.cat([forward_out, backward_out], axis=-1)
        return Tensor.cat([forward_out, backward_out], axis=-1)

    def forward(self, x: Tensor, h0=None) -> Tensor:
        out = x
        for layer in range(self.num_layers):
            out = self._run_layer(out, layer)
            if layer < self.num_layers - 1 and len(self.dropouts) > layer:
                out = self.dropouts[layer](out)
        return out

    @property
    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)


class _Identity(Module):
    def forward(self, x):
        return x


class LSTM(RNN):
    def __init__(self, *args, **kwargs):
        kwargs["cell_type"] = "lstm"
        super().__init__(*args, **kwargs)


class GRU(RNN):
    def __init__(self, *args, **kwargs):
        kwargs["cell_type"] = "gru"
        super().__init__(*args, **kwargs)


class SequenceClassifier(Module):
    """A recurrent encoder followed by a linear classification head."""

    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 64,
                 num_layers: int = 1, cell_type: str = "lstm", dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.encoder = RNN(
            input_size, hidden_size, num_layers=num_layers, cell_type=cell_type,
            dropout=dropout, return_sequences=False, bidirectional=bidirectional,
        )
        self.dropout = Dropout(dropout) if dropout else None
        self.head = Linear(self.encoder.output_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return self.head(h)
