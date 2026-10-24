# Architecture

This document explains how NeuroForge works internally — the parts that are easy to get subtly wrong.

---

## 1. The autograd engine

Everything is built on `neuroforge/tensor.py`. A `Tensor` wraps a NumPy array and remembers how to compute its own gradient.

```python
class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")
```

The three fields that matter:

- `data` — the NumPy array (always `float64`).
- `_prev` — the tensors that produced this one.
- `_backward` — a closure that pushes this tensor's gradient into its parents' `.grad`.

### Define-by-run

There is no static graph. When you write `y = (x * x).sum()`, `__mul__` creates a tensor whose `_backward` knows the local derivative of multiplication, and `sum` creates one whose `_backward` knows how to distribute a gradient back over the summed elements. The graph is built as a side effect of ordinary Python arithmetic.

### The backward pass

`Tensor.backward()` topologically sorts the graph (parents before children), then walks it in reverse, calling each node's `_backward`:

```python
topo, visited = [], set()
def build(t):
    if id(t) in visited: return
    visited.add(id(t))
    for parent in t._prev:
        build(parent)
    topo.append(t)
build(self)
for node in reversed(topo):
    node._backward()
```

Reversing a topological order guarantees a node's `.grad` is complete before it is used to compute its parents' gradients.

### Why gradients accumulate

`_accumulate` **adds** into `.grad` rather than assigning. A tensor used twice in an expression receives a contribution from each path:

```python
x = Tensor([2.0], requires_grad=True)
y = x * x + x * x       # x is used twice
y.backward()
x.grad                  # 8.0 = 4x, not 4.0
```

This is why `zero_grad()` exists and why the training loop calls it every step. Forgetting it is the single most common bug in hand-written training loops.

### Broadcasting, and the trickiest bug

NumPy broadcasts silently. `(3,1) + (3,4)` produces a `(3,4)` result, and the `(3,1)` operand was *conceptually* stretched. Its gradient must therefore be **summed back down** to `(3,1)`:

```python
def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad
```

Without this, biases and layer norms silently accumulate wrong gradients that still *look* plausible — the model trains, just badly. `tests/test_tensor.py::test_broadcasting_reduces_correctly` guards it.

### Numerical stability

Two operations need care:

**`log_softmax`** subtracts the row maximum before exponentiating:

```
shifted = x - max(x)
logp    = shifted - log(sum(exp(shifted)))
```

Its gradient is `g - softmax(x) * sum(g)`, which is why softmax and log-softmax are implemented as single fused ops rather than composed from `exp`/`log`.

**Cross-entropy** is computed on log-probabilities, never on probabilities:

```python
return nll_loss(logits.log_softmax(axis=-1), targets)
```

Computing `-log(softmax(x))` directly overflows for large logits. `test_cross_entropy_stable_with_large_logits` feeds it ±1000 and checks the result stays finite.

---

## 2. Modules

`Module` handles parameter bookkeeping so you never write a registration call. `__setattr__` inspects what is being assigned:

- a `Parameter` → record it in `self._parameters`
- a `Module` → record it in `self._modules`
- a list of `Module`s (under a public name) → wrap it in a `ModuleList`

`parameters()` then walks the tree recursively, producing qualified names like `blocks.0.attn.q_proj.weight`. `state_dict()` uses those same names, which is what makes checkpointing and `load_state_dict` line up.

The underscore guard in `__setattr__` matters: without it, `self._layers = [...]` inside `Sequential` would try to wrap itself in a `ModuleList`, recursing until the stack overflows.

---

## 3. Convolution without a custom layer kernel

`Conv2d` uses **im2col**: gather every kernel-sized patch of the input into a row, so convolution becomes one matrix multiply.

```
input  (N, C, H, W)
  ↓ im2col
cols   (N, C*kh*kw, L)
  ↓ transpose + matmul with weight (C_out, C*kh*kw)
out    (N, C_out, H_out, W_out)
```

The scatter in `im2col`'s backward (`col2im`) accumulates with `+=` because neighbouring output positions can read the same input pixel. `test_conv2d_numeric_gradient` checks the whole thing against finite differences.

---

## 4. The transformer

`MultiHeadAttention` projects the input into queries, keys and values, splits them into `num_heads` subspaces, and computes scaled dot-product attention per head:

```
scores = (Q @ Kᵀ) / sqrt(head_dim)
scores = scores.masked_fill(causal_mask, -1e9)
attn   = softmax(scores)
out    = attn @ V
```

The causal mask is an upper-triangular boolean array, `True` where attention is forbidden. Masked positions are pushed to `-1e9` *before* the softmax, so they receive effectively zero weight. Because `masked_fill` also blocks gradient flow through filled positions, the mask is enforced in both directions.

`test_gpt_is_causal` is the decisive check: it changes only the last token of a sequence and asserts that all earlier logits are unchanged to within `1e-12` — and that the last one *did* change. Without a correct mask, the first assertion fails immediately.

`TransformerBlock` is pre-norm:

```python
x = x + attn(ln1(x), mask)
x = x + ff(ln2(x))
```

Pre-norm (normalising the *input* to each sublayer) is more stable to train than the original post-norm arrangement, and is what modern models use.

---

## 5. Recurrent networks

Sequences are processed by looping over time steps in Python. The hidden state is a `Tensor`, so each step extends the graph by one node and backpropagation-through-time is just ordinary reverse-mode autodiff over that chain. This is slow but completely transparent — there is no separate BPTT implementation to get wrong.

The LSTM gates are ordinary differentiable expressions:

```python
i, f, g, o = split(gates, 4)
c = f * c + i * g
h = o * tanh(c)
```

Note that `c` (the cell state) is carried separately from `h`, which is what lets gradients flow across long sequences without vanishing.

---

## 6. The training loop

`Trainer.fit` is deliberately plain:

```
for epoch:
    for batch:
        zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
    evaluate()          # optional
    callbacks()
```

Two details are worth noting. Inputs are prepared by `_to_tensor`, which wraps floating-point data in a `Tensor` but passes integer arrays through untouched — token-based models like `GPT` need raw indices, not floats. And `evaluate` runs inside a `no_grad()` block, which prunes the graph at construction so validation costs only forward passes.

---

## 7. Why numerical gradient checking is the right test

Every operation has a hand-derived backward pass, and every one of those is a place to make a sign error. The test that catches them compares the analytic gradient against a central-difference approximation:

```
f'(x) ≈ (f(x + ε) - f(x - ε)) / 2ε
```

If they agree to ~1e-6, the backward pass is correct. `tests/test_tensor.py` applies this to every operation; `tests/test_nn.py` applies it to `Conv2d`. This is not decoration — several genuine bugs in this codebase were found by these checks rather than by any training run.
