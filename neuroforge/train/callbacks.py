"""Callbacks: hooks that run at the end of each epoch.

A callback receives the trainer and the current :class:`History`, and may set
``trainer.stop_training = True`` to end the run early.
"""

from __future__ import annotations

import csv
import os

import numpy as np

__all__ = ["Callback", "History", "EarlyStopping", "ModelCheckpoint", "CSVLogger",
           "LRSchedulerCallback"]


class History(dict):
    """A dict of metric-name -> list of per-epoch values, with attribute access."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def append(self, **values) -> None:
        for key, value in values.items():
            self.setdefault(key, []).append(value)

    def latest(self) -> dict:
        return {k: (v[-1] if v else None) for k, v in self.items()}

    def best(self, key: str, mode: str = "min"):
        values = self.get(key, [])
        if not values:
            return None
        return min(values) if mode == "min" else max(values)


class Callback:
    """Base class. Override any hook you need."""

    def on_train_begin(self, trainer, history) -> None: ...

    def on_epoch_end(self, trainer, history, epoch: int) -> None: ...

    def on_train_end(self, trainer, history) -> None: ...


class EarlyStopping(Callback):
    """Stop when a monitored quantity stops improving.

    Args:
        monitor: key in the history, e.g. ``"val_loss"``.
        patience: how many non-improving epochs to tolerate.
        min_delta: smallest change counted as an improvement.
        mode: ``"min"`` or ``"max"``.
        restore_best: put the best-seen weights back at the end.
    """

    def __init__(self, monitor: str = "val_loss", patience: int = 5,
                 min_delta: float = 0.0, mode: str = "min",
                 restore_best: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best = None
        self.best_epoch = None
        self.best_weights = None
        self.wait = 0

    def _improved(self, current) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def on_epoch_end(self, trainer, history, epoch: int) -> None:
        values = history.get(self.monitor)
        if not values:
            return
        current = values[-1]
        if self._improved(current):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best:
                self.best_weights = {k: v.copy() for k, v in trainer.model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.stop_training = True
                if trainer.verbose:
                    print(f"  early stopping at epoch {epoch + 1} "
                          f"(best {self.monitor}={self.best:.4f} @ epoch {self.best_epoch + 1})")

    def on_train_end(self, trainer, history) -> None:
        if self.restore_best and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)


class ModelCheckpoint(Callback):
    """Save the model whenever a monitored metric improves."""

    def __init__(self, path: str = "best_model.npz", monitor: str = "val_loss",
                 mode: str = "min", save_best_only: bool = True):
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None

    def on_epoch_end(self, trainer, history, epoch: int) -> None:
        from ..utils.serialization import save_checkpoint

        values = history.get(self.monitor)
        if not values:
            return
        current = values[-1]
        improved = self.best is None or (
            current < self.best if self.mode == "min" else current > self.best
        )
        if improved:
            self.best = current
        if improved or not self.save_best_only:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            save_checkpoint(self.path, trainer.model, optimizer=trainer.optimizer,
                            extra={"epoch": epoch, self.monitor: current})


class CSVLogger(Callback):
    """Append one row per epoch to a CSV file."""

    def __init__(self, path: str = "history.csv"):
        self.path = path
        self._header_written = False

    def on_epoch_end(self, trainer, history, epoch: int) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        row = {"epoch": epoch + 1, **history.latest()}
        write_header = not self._header_written and not os.path.exists(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._header_written = True


class LRSchedulerCallback(Callback):
    """Multiply the learning rate by ``factor`` every ``every`` epochs."""

    def __init__(self, every: int = 5, factor: float = 0.5, min_lr: float = 1e-6):
        self.every = every
        self.factor = factor
        self.min_lr = min_lr

    def on_epoch_end(self, trainer, history, epoch: int) -> None:
        if (epoch + 1) % self.every == 0:
            new_lr = max(trainer.optimizer.lr * self.factor, self.min_lr)
            trainer.optimizer.lr = new_lr
            if trainer.verbose:
                print(f"  lr -> {new_lr:.2e}")
