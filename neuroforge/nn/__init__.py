"""The ``neuroforge.nn`` package: layers, losses, optimisers and activations."""

from .module import Module, ModuleList, Parameter, Sequential
from .layers import (
    Linear, Embedding, Dropout, Flatten, Conv2d, MaxPool2d, AvgPool2d,
)
from .norm import LayerNorm, RMSNorm, BatchNorm1d
from .activations import (
    ReLU, LeakyReLU, GELU, Tanh, Sigmoid, Softmax, Identity,
    relu, leaky_relu, gelu, tanh, sigmoid, softmax, log_softmax,
)
from .losses import (
    mse_loss, l1_loss, huber_loss, cross_entropy, nll_loss,
    binary_cross_entropy_with_logits, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss,
)
from .optim import Optimizer, SGD, RMSprop, Adam, AdamW
from . import init

__all__ = [
    # containers
    "Module", "ModuleList", "Parameter", "Sequential",
    # layers
    "Linear", "Embedding", "Dropout", "Flatten",
    "Conv2d", "MaxPool2d", "AvgPool2d",
    # normalisation
    "LayerNorm", "RMSNorm", "BatchNorm1d",
    # activations
    "ReLU", "LeakyReLU", "GELU", "Tanh", "Sigmoid", "Softmax", "Identity",
    "relu", "leaky_relu", "gelu", "tanh", "sigmoid", "softmax", "log_softmax",
    # losses
    "mse_loss", "l1_loss", "huber_loss", "cross_entropy", "nll_loss",
    "binary_cross_entropy_with_logits", "MSELoss", "CrossEntropyLoss",
    "BCEWithLogitsLoss",
    # optimisers
    "Optimizer", "SGD", "RMSprop", "Adam", "AdamW",
    # misc
    "init",
]
