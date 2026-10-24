"""Plotting helpers. Uses the non-interactive Agg backend so they work headless."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..tensor import Tensor, no_grad  # noqa: E402

__all__ = ["plot_history", "plot_confusion_matrix", "plot_decision_boundary",
           "plot_image_grid"]


def plot_history(history, path: str = None, keys=None, title: str = "Training history"):
    """Plot one or more metrics against epoch number.

    Args:
        history: a ``History`` (or plain dict of lists).
        path: if given, the figure is saved there.
        keys: metric names to plot; defaults to loss + val_loss.
    """
    keys = keys or [k for k in ("loss", "val_loss") if k in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in keys:
        values = history.get(key) if hasattr(history, "get") else history[key]
        if values:
            ax.plot(range(1, len(values) + 1), values, label=key, linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig


def plot_confusion_matrix(cm, path: str = None, class_names=None, normalize: bool = False):
    """Render a confusion matrix as a heatmap with counts annotated."""
    cm = np.asarray(cm)
    shown = cm.astype(float)
    if normalize:
        row_sums = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, row_sums, out=np.zeros_like(shown), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(shown, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = cm.shape[0]
    labels = class_names or [str(i) for i in range(n)]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")

    threshold = shown.max() / 2 if shown.size else 0.5
    for i in range(n):
        for j in range(n):
            text = f"{shown[i, j]:.2f}" if normalize else str(cm[i, j])
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if shown[i, j] > threshold else "black", fontsize=9)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig


def plot_decision_boundary(model, X, y, path: str = None, resolution: int = 120):
    """Show how a 2-D classifier divides the plane, over the training points."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]

    model.eval()
    with no_grad():
        logits = model(Tensor(grid))
    scores = logits.data.argmax(axis=-1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, scores, alpha=0.25, cmap="coolwarm", levels=np.arange(scores.max() + 2) - 0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=18, edgecolors="k", linewidths=0.3)
    ax.set_title("Decision boundary")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig


def plot_image_grid(images, path: str = None, ncols: int = 8, titles=None,
                    cmap: str = "gray"):
    """Show a grid of images (shape ``(N, 1, H, W)`` or ``(N, H, W)``)."""
    images = np.asarray(images)
    if images.ndim == 4:
        images = images[:, 0]
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        axes[i].imshow(images[i], cmap=cmap)
        if titles is not None:
            axes[i].set_title(str(titles[i]), fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig
