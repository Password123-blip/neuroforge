"""The training loop.

:class:`Trainer` glues a model, an optimiser and a loss together and runs the
standard loop: forward, loss, backward, step — with optional validation,
metric tracking and callbacks.
"""

from __future__ import annotations

import time

import numpy as np

from ..tensor import Tensor, no_grad
from .callbacks import Callback, History
from .metrics import accuracy

__all__ = ["Trainer"]


class Trainer:
    """Train a :class:`~neuroforge.nn.module.Module`.

    Args:
        model: the module to train.
        optimizer: any optimiser from ``neuroforge.nn``.
        loss_fn: callable ``(prediction, target) -> Tensor``.
        callbacks: list of :class:`Callback` instances.
        verbose: print a line per epoch.
        seed: seed the global NumPy RNG for reproducible runs.
    """

    def __init__(self, model, optimizer, loss_fn, callbacks=None,
                 verbose: bool = True, seed: int = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = list(callbacks or [])
        self.verbose = verbose
        self.stop_training = False
        self.history = History()
        self.step_count = 0
        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    def _unpack(self, batch):
        """Split a batch into (inputs, targets); targets is None if absent."""
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                return batch[0], None
            return batch[0], batch[1]
        return batch, None

    def _to_tensor(self, array):
        """Prepare model inputs.

        Floating-point data becomes a Tensor; anything else (notably integer
        token ids) is passed through so token-based models receive the dtype
        they expect. ``Module.__call__`` handles plain float arrays as well.
        """
        if isinstance(array, Tensor):
            return array
        array = np.asarray(array)
        if np.issubdtype(array.dtype, np.floating):
            return Tensor(array.astype(np.float64))
        return array

    # ------------------------------------------------------------------
    def train_step(self, inputs, targets) -> float:
        """One optimisation step; returns the scalar loss."""
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self._to_tensor(inputs))
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        return float(loss.item())

    def evaluate(self, loader) -> dict:
        """Run the model over a loader without updating weights."""
        self.model.eval()
        total_loss, n_batches, n_examples = 0.0, 0, 0
        all_logits, all_targets = [], []

        with no_grad():
            for batch in loader:
                inputs, targets = self._unpack(batch)
                output = self.model(self._to_tensor(inputs))
                if targets is not None:
                    loss = self.loss_fn(output, targets)
                    total_loss += float(loss.item())
                    all_targets.append(np.asarray(targets))
                all_logits.append(output.data)
                n_batches += 1
                n_examples += len(inputs)

        metrics = {"loss": total_loss / max(n_batches, 1), "n": n_examples}
        if all_targets:
            logits = np.concatenate(all_logits, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            if logits.ndim == 2 and targets.ndim == 1:
                metrics["accuracy"] = accuracy(logits, targets)
        return metrics

    # ------------------------------------------------------------------
    def fit(self, train_loader, epochs: int = 10, val_loader=None) -> History:
        """Run the full training loop and return the :class:`History`."""
        self.stop_training = False
        for cb in self.callbacks:
            cb.on_train_begin(self, self.history)

        for epoch in range(epochs):
            start = time.time()
            self.model.train()
            running, batches = 0.0, 0
            for batch in train_loader:
                inputs, targets = self._unpack(batch)
                if targets is None:
                    raise ValueError("training batches must contain targets")
                running += self.train_step(inputs, targets)
                batches += 1

            record = {"loss": running / max(batches, 1), "epoch_time": time.time() - start}
            if val_loader is not None:
                val = self.evaluate(val_loader)
                record["val_loss"] = val["loss"]
                if "accuracy" in val:
                    record["val_accuracy"] = val["accuracy"]

            self.history.append(**record)

            if self.verbose:
                msg = f"epoch {epoch + 1:3d}/{epochs}  loss={record['loss']:.4f}"
                if "val_loss" in record:
                    msg += f"  val_loss={record['val_loss']:.4f}"
                if "val_accuracy" in record:
                    msg += f"  val_acc={record['val_accuracy']:.4f}"
                msg += f"  ({record['epoch_time']:.2f}s)"
                print(msg)

            for cb in self.callbacks:
                cb.on_epoch_end(self, self.history, epoch)
            if self.stop_training:
                break

        for cb in self.callbacks:
            cb.on_train_end(self, self.history)
        return self.history

    # ------------------------------------------------------------------
    def predict(self, inputs) -> np.ndarray:
        """Return raw model outputs for an array or a loader."""
        self.model.eval()
        with no_grad():
            if hasattr(inputs, "__iter__") and not isinstance(inputs, np.ndarray):
                outputs = [self.model(self._to_tensor(x)).data for x, _ in
                           (self._unpack(b) for b in inputs)]
                return np.concatenate(outputs, axis=0)
            return self.model(self._to_tensor(inputs)).data

    def predict_classes(self, inputs) -> np.ndarray:
        """Argmax of the model output (for classification)."""
        return self.predict(inputs).argmax(axis=-1)
