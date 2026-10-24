"""NeuroForge — a neural-network toolkit written from scratch on NumPy.

Everything in this package is implemented by hand: the autodiff engine, the
layers, the optimisers and the models. There is no dependency on PyTorch,
TensorFlow or JAX.

Quick start::

    import neuroforge as nf
    from neuroforge import nn, models, data, train

    model = models.MLP(2, (32, 32), 3, activation=nn.ReLU)
    opt = nn.Adam(model.parameters(), lr=1e-2)
    trainer = train.Trainer(model, opt, nn.cross_entropy)

    X, y = data.make_spiral(200, 3, seed=0)
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True)
    trainer.fit(loader, epochs=50)
"""

from .tensor import Tensor, no_grad, enable_grad, is_grad_enabled
from . import nn
from . import models
from . import data
from . import train
from . import utils

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "nn",
    "models",
    "data",
    "train",
    "utils",
    "__version__",
]
