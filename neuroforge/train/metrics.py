"""Metrics for evaluating models."""

from __future__ import annotations

import numpy as np

__all__ = ["accuracy", "top_k_accuracy", "perplexity", "confusion_matrix",
           "precision_recall_f1"]


def _as_numpy(x):
    # NB: NumPy arrays also expose a `.data` attribute (a memoryview), so we must
    # check for a Tensor specifically rather than just for `.data`.
    if hasattr(x, "requires_grad") and hasattr(x, "data"):
        return np.asarray(x.data)
    return np.asarray(x)


def accuracy(prediction, target) -> float:
    """Fraction of correct predictions.

    ``prediction`` may be class indices or a ``(N, C)`` score matrix.
    """
    pred = _as_numpy(prediction)
    target = _as_numpy(target).reshape(-1)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    return float((pred.reshape(-1) == target).mean())


def top_k_accuracy(scores, target, k: int = 5) -> float:
    """Fraction of examples whose true class is in the top-k scores."""
    scores = _as_numpy(scores)
    target = _as_numpy(target).reshape(-1)
    if scores.ndim == 1:
        scores = scores[:, None]
    k = min(k, scores.shape[-1])
    topk = np.argsort(-scores, axis=-1)[:, :k]
    return float((topk == target[:, None]).any(axis=1).mean())


def perplexity(loss) -> float:
    """Exponentiate a cross-entropy loss (the standard LM metric)."""
    loss = float(loss.item() if hasattr(loss, "item") else loss)
    return float(np.exp(loss))


def confusion_matrix(prediction, target, num_classes: int = None) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    pred = _as_numpy(prediction)
    target = _as_numpy(target).reshape(-1).astype(int)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    pred = pred.reshape(-1).astype(int)
    if num_classes is None:
        num_classes = int(max(pred.max(), target.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (target, pred), 1)
    return cm


def precision_recall_f1(prediction, target, num_classes: int = None):
    """Per-class precision, recall and F1 as three arrays."""
    cm = confusion_matrix(prediction, target, num_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1
