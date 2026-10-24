"""Dataset generators and text corpora.

Everything here runs offline. The synthetic generators are designed to be
*learnable* — each one has real structure that a model can pick up — so they
make honest test beds rather than noise.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

__all__ = [
    "make_spiral", "make_moons", "make_blobs", "make_shape_images",
    "make_sequence_task", "one_hot", "BUILTIN_TEXT", "load_text",
]


# ----------------------------------------------------------------------
# 2-D classification
# ----------------------------------------------------------------------
def make_spiral(n_points: int = 200, n_classes: int = 3, noise: float = 0.2,
                seed: int = None):
    """The classic interleaved-spirals problem.

    Returns ``(X, y)`` with ``X`` of shape ``(n_points * n_classes, 2)``.
    """
    rng = np.random.default_rng(seed)
    per_class = n_points
    X = np.zeros((per_class * n_classes, 2))
    y = np.zeros(per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        ix = slice(c * per_class, (c + 1) * per_class)
        r = np.linspace(0.0, 1.0, per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, per_class) + rng.normal(0, noise, per_class)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


def make_moons(n_points: int = 300, noise: float = 0.1, seed: int = None):
    """Two interleaving half-circles."""
    rng = np.random.default_rng(seed)
    n_out = n_points // 2
    n_in = n_points - n_out
    theta_out = np.linspace(0, np.pi, n_out)
    theta_in = np.linspace(0, np.pi, n_in)
    outer = np.c_[np.cos(theta_out), np.sin(theta_out)]
    inner = np.c_[1 - np.cos(theta_in), 0.5 - np.sin(theta_in)]
    X = np.vstack([outer, inner]) + rng.normal(0, noise, (n_points, 2))
    y = np.r_[np.zeros(n_out, dtype=np.int64), np.ones(n_in, dtype=np.int64)]
    perm = rng.permutation(n_points)
    return X[perm], y[perm]


def make_blobs(n_points: int = 300, n_classes: int = 3, spread: float = 0.6,
               seed: int = None):
    """Isotropic Gaussian clusters arranged on a circle."""
    rng = np.random.default_rng(seed)
    centers = np.c_[np.cos(np.linspace(0, 2 * np.pi, n_classes, endpoint=False)),
                    np.sin(np.linspace(0, 2 * np.pi, n_classes, endpoint=False))] * 2.0
    per = n_points // n_classes
    X, y = [], []
    for c in range(n_classes):
        pts = rng.normal(centers[c], spread, (per, 2))
        X.append(pts)
        y.append(np.full(per, c, dtype=np.int64))
    return np.vstack(X), np.concatenate(y)


# ----------------------------------------------------------------------
# synthetic images
# ----------------------------------------------------------------------
def make_shape_images(n_samples: int = 600, size: int = 16, noise: float = 0.4,
                      seed: int = None):
    """Tiny greyscale images containing one of three line orientations.

    Class 0: vertical line, class 1: horizontal line, class 2: diagonal line.
    A convnet has to learn oriented edge detectors to solve this — which is
    exactly what convolutions are for.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 1, size, size), dtype=np.float64)
    y = rng.integers(0, 3, n_samples)

    for i in range(n_samples):
        img = np.zeros((size, size))
        thickness = rng.integers(1, 3)
        if y[i] == 0:                      # vertical
            col = rng.integers(2, size - 2)
            img[:, col:col + thickness] = 1.0
        elif y[i] == 1:                    # horizontal
            row = rng.integers(2, size - 2)
            img[row:row + thickness, :] = 1.0
        else:                              # diagonal
            for k in range(size):
                j = k
                if 0 <= j < size:
                    img[k, j] = 1.0
        img += rng.normal(0, noise, (size, size))
        X[i, 0] = img
    return X, y


# ----------------------------------------------------------------------
# sequence tasks
# ----------------------------------------------------------------------
def make_sequence_task(n_samples: int = 800, seq_len: int = 20, seed: int = None):
    """Binary classification of random walks.

    Each sequence is a series of +1/-1 steps; the label is 1 when the total
    sum is positive. A model must integrate evidence across the whole sequence,
    so a recurrent network (or a transformer) is the natural fit.
    """
    rng = np.random.default_rng(seed)
    X = rng.choice([-1.0, 1.0], size=(n_samples, seq_len, 1))
    y = (X.sum(axis=(1, 2)) > 0).astype(np.int64)
    return X, y


def one_hot(labels, num_classes: int = None) -> np.ndarray:
    """Convert integer labels to a one-hot matrix."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    out = np.zeros((labels.shape[0], num_classes))
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


# ----------------------------------------------------------------------
# text
# ----------------------------------------------------------------------
BUILTIN_TEXT = """\
A machine that learns is not a machine that knows. It is a machine that
changes. At first it guesses, and its guesses are wrong in uninteresting ways.
But each wrong guess carries a small piece of information, and if you can turn
that information into a nudge, the guessing slowly stops being random.

The nudge is the whole trick. You measure how far the output is from what you
wanted, and you push every number in the direction that would have made the
error smaller. Do this once and nothing happens. Do it a thousand times and
something begins to look like understanding, though no one can quite say where
it lives.

There is a strange comfort in this. The rules are simple: multiply, add, take
a slope, step downhill. There is no magic in any single line. The magic, if you
want to call it that, is only in the number of times you are willing to repeat
the same small honest step.

And so we build. We stack layers because depth lets simple features compose into
complicated ones. We add attention because not everything in a sentence matters
equally, and a model should be free to decide what to look at. We normalise
because without it the numbers drift and the learning stalls. None of this is
mysterious once you have written it down.

What remains mysterious is the scale. A few thousand numbers can memorise a
poem. A few billion can write one. The difference is not a new idea, it is the
same idea repeated with more patience than any person has. That is the honest
lesson of the field: the cleverness is small and the patience is enormous.

So start small. Train a network to separate two spirals. Watch the loss fall and
believe it. Then make it bigger, and keep the parts you understand.
"""


def load_text(name: str = "builtin", cache_dir: str = None,
              download: bool = False) -> str:
    """Return a text corpus.

    Args:
        name: ``"builtin"`` for the bundled corpus, otherwise a key in the
            download table (e.g. ``"shakespeare"``).
        cache_dir: where to cache downloaded text.
        download: if True and ``name`` is a known remote corpus, fetch it;
            on any failure this falls back to the built-in text rather than
            raising, so examples always run.

    Returns:
        The corpus as a single string.
    """
    if name == "builtin":
        return BUILTIN_TEXT

    remote = {
        "shakespeare": (
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt"
        ),
    }
    if name not in remote:
        raise ValueError(f"unknown corpus '{name}'; choose from {['builtin', *remote]}")

    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".neuroforge", "data")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.txt")

    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    if not download:
        return BUILTIN_TEXT

    try:
        with urllib.request.urlopen(remote[name], timeout=20) as resp:
            text = resp.read().decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception:
        # Offline, blocked, or the URL moved — the built-in corpus still works.
        return BUILTIN_TEXT
