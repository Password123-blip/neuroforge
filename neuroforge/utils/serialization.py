"""Saving and loading models, optimiser state and run metadata."""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["save_checkpoint", "load_checkpoint", "save_json", "load_json"]


def save_checkpoint(path: str, model, optimizer=None, extra: dict = None) -> str:
    """Write model weights (and optionally optimiser state) to a ``.npz`` file.

    Args:
        path: destination path. ``.npz`` is appended if missing.
        model: any ``Module``.
        optimizer: optional optimiser whose ``state_dict`` should be stored.
        extra: any JSON-serialisable metadata (epoch, metric values, config).

    Returns:
        The path actually written.
    """
    if not path.endswith(".npz"):
        path += ".npz"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    payload = {f"param::{name}": value for name, value in model.state_dict().items()}
    meta = {"extra": extra or {}, "n_params": model.num_parameters()}
    if optimizer is not None:
        meta["optimizer"] = optimizer.state_dict()
    payload["__meta__"] = np.array(json.dumps(meta))

    np.savez(path, **payload)
    return path


def load_checkpoint(path: str, model=None) -> dict:
    """Read a checkpoint. If ``model`` is given, load its weights in place.

    Returns a dict with keys ``"params"``, ``"meta"`` and ``"path"``.
    """
    if not path.endswith(".npz"):
        path += ".npz"
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        params = {k.split("::", 1)[1]: data[k] for k in data.files if k.startswith("param::")}
        meta = json.loads(str(data["__meta__"])) if "__meta__" in data.files else {}

    if model is not None:
        model.load_state_dict(params)
    return {"params": params, "meta": meta, "path": path}


def save_json(path: str, obj) -> str:
    """Write a JSON-serialisable object, coercing NumPy types along the way."""
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)
    return path


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
