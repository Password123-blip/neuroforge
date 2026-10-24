"""Base classes for neural-network modules and parameters."""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor

__all__ = ["Parameter", "Module", "ModuleList", "Sequential"]



class Parameter(Tensor):
    """A :class:`Tensor` that is, by definition, trainable.

    Parameters are what optimisers see: anything stored on a ``Module`` as a
    ``Parameter`` is picked up automatically by ``module.parameters()``.
    """

    def __init__(self, data, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Parameter(shape={self.shape})"


class Module:
    """Base class for every layer and model.

    Subclasses implement :meth:`forward`. Assigning a ``Parameter`` or another
    ``Module`` to an attribute registers it automatically, so ``parameters()``
    and ``state_dict()`` see it without any extra bookkeeping.
    """

    def __init__(self) -> None:
        # object.__setattr__ bypasses our own __setattr__ during setup.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    # ------------------------------------------------------------------
    # attribute registration
    # ------------------------------------------------------------------
    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif (
            not name.startswith("_")
            and isinstance(value, (list, tuple))
            and value
            and all(isinstance(v, Module) for v in value)
        ):
            self._modules[name] = ModuleList(list(value))
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # parameter traversal
    # ------------------------------------------------------------------
    def named_parameters(self, prefix: str = ""):
        """Yield ``(qualified_name, Parameter)`` for this module and children."""
        for name, param in self._parameters.items():
            yield (f"{prefix}{name}" if not prefix else f"{prefix}.{name}"), param
        for mod_name, module in self._modules.items():
            child_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(child_prefix)

    def parameters(self):
        """Return a flat list of every trainable parameter."""
        return [p for _, p in self.named_parameters()]

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name, module in self._modules.items():
            child = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(child)

    def num_parameters(self, trainable_only: bool = True) -> int:
        total = 0
        for _, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            total += p.size
        return total

    # ------------------------------------------------------------------
    # training / evaluation mode
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict) -> None:
        own = dict(self.named_parameters())
        missing = [k for k in state if k not in own]
        if missing:
            raise KeyError(f"unexpected keys in state_dict: {missing}")
        for name, param in own.items():
            if name in state:
                arr = np.asarray(state[name])
                if arr.shape != param.shape:
                    raise ValueError(
                        f"shape mismatch for '{name}': expected {param.shape}, got {arr.shape}"
                    )
                param.data = arr.astype(np.float64)

    # ------------------------------------------------------------------
    # calling
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        # Accept plain NumPy arrays as a convenience: floating-point arrays are
        # wrapped in a Tensor, while integer arrays are passed through untouched
        # so that models taking token ids (GPT, TransformerClassifier) work too.
        args = tuple(
            Tensor(a)
            if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating)
            else a
            for a in args
        )
        return self.forward(*args, **kwargs)

    def summary(self, depth: int = 0) -> str:
        """A readable tree of the module and its parameter counts."""
        lines = []
        indent = "  " * depth
        own = sum(p.size for p in self._parameters.values())
        lines.append(f"{indent}{type(self).__name__}  (own params: {own})")
        for name, module in self._modules.items():
            lines.append(f"{indent}  └─ {name}:")
            lines.append(module.summary(depth + 2))
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(params={self.num_parameters()})"


class ModuleList(Module):
    """A list of modules that is itself a module (so children are registered)."""

    def __init__(self, modules=None):
        super().__init__()
        object.__setattr__(self, "_list", list(modules or []))
        for i, module in enumerate(self._list):
            object.__setattr__(self, str(i), module)
            if isinstance(module, Module):
                self._modules[str(i)] = module

    def append(self, module: Module) -> "ModuleList":
        self._list.append(module)
        idx = str(len(self._list) - 1)
        object.__setattr__(self, idx, module)
        self._modules[idx] = module
        return self

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, idx):
        return self._list[idx]

    def __iter__(self):
        return iter(self._list)

    def forward(self, x):
        for module in self._list:
            x = module(x)
        return x


class Sequential(Module):
    """Run a stack of modules in order."""

    def __init__(self, *modules):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = tuple(modules[0])
        self._layers = list(modules)
        for i, module in enumerate(self._layers):
            name = str(i)
            object.__setattr__(self, name, module)
            if isinstance(module, Module):
                self._modules[name] = module

    def forward(self, x):
        for module in self._layers:
            x = module(x)
        return x

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, idx):
        return self._layers[idx]

    def __iter__(self):
        return iter(self._layers)
