"""A small convolutional network for image classification.

The classifier head is built lazily on the first forward pass, so you never
have to hand-compute the flattened feature size.
"""

from __future__ import annotations

from ..tensor import Tensor
from ..nn.activations import ReLU
from ..nn.layers import Conv2d, Dropout, Flatten, Linear, MaxPool2d
from ..nn.module import Module, ModuleList, Sequential

__all__ = ["CNN"]


class CNN(Module):
    """Configurable convolutional network.

    Args:
        in_channels: number of input channels (1 for greyscale, 3 for RGB).
        num_classes: number of output classes.
        channels: width of each conv stage, e.g. ``(16, 32)``.
        kernel_size: conv kernel size (square).
        pool: whether to max-pool after each conv stage.
        hidden: width of the fully-connected head.
        dropout: dropout applied before the head and inside it.
    """

    def __init__(self, in_channels: int, num_classes: int, channels=(16, 32),
                 kernel_size: int = 3, pool: bool = True, hidden: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden = hidden
        self.dropout_p = dropout

        blocks = []
        prev = in_channels
        for width in channels:
            blocks.append(Conv2d(prev, width, kernel_size=kernel_size, padding=kernel_size // 2))
            blocks.append(ReLU())
            if pool:
                blocks.append(MaxPool2d(2, stride=2))
            prev = width
        self.features = Sequential(*blocks)
        self.classifier = None  # created lazily on first forward

    def forward(self, x: Tensor) -> Tensor:
        feats = self.features(x)
        flat = feats.reshape((feats.shape[0], -1))
        if self.classifier is None:
            layers = []
            if self.dropout_p:
                layers.append(Dropout(self.dropout_p))
            layers.append(Linear(flat.shape[1], self.hidden))
            layers.append(ReLU())
            if self.dropout_p:
                layers.append(Dropout(self.dropout_p))
            layers.append(Linear(self.hidden, self.num_classes))
            self.classifier = Sequential(*layers)
        return self.classifier(flat)

    def feature_maps(self, x: Tensor) -> Tensor:
        """Return the convolutional features (useful for visualisation)."""
        return self.features(x)
