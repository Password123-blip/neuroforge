"""The ``neuroforge.train`` package: trainer, callbacks and metrics."""

from .trainer import Trainer
from .callbacks import (
    Callback, History, EarlyStopping, ModelCheckpoint, CSVLogger, LRSchedulerCallback,
)
from .metrics import (
    accuracy, top_k_accuracy, perplexity, confusion_matrix, precision_recall_f1,
)

__all__ = [
    "Trainer",
    "Callback", "History", "EarlyStopping", "ModelCheckpoint", "CSVLogger",
    "LRSchedulerCallback",
    "accuracy", "top_k_accuracy", "perplexity", "confusion_matrix",
    "precision_recall_f1",
]
