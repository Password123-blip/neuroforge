# NeuroForge

A neural-network toolkit written **from scratch on NumPy** — autograd engine, layers, optimisers, and working MLP / CNN / LSTM / GPT models, plus datasets, a training loop and tests.

No PyTorch. No TensorFlow. No JAX. Just NumPy and the maths, written out by hand.

---

## Read this first: what this is, and what it is not

This is a **complete, working, educational deep-learning framework**. Every gradient in it is computed by code in this repository. You can train real models with it and watch them learn.

It is **not** a frontier-scale generalist model like GPT-4 or Gemini, and no project of this kind can be. Those models cost tens of millions of dollars, run on tens of thousands of GPUs, and are trained on trillions of tokens. What makes them powerful is not a secret algorithm — it is scale and data. The algorithms are the ones implemented here.

So this repository gives you the honest foundation: the actual mechanisms, small enough to read end to end, correct enough to train on, and tested against numerical gradients so you can trust them.

---

## Install

Requires Python 3.10+ and NumPy. Nothing else is needed to train.

```bash
git clone <your-repo-url>
cd neuroforge
pip install -r requirements.txt
```

For development (tests + plots):

```bash
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
import neuroforge as nf
from neuroforge import nn, models, data, train

# 1. data
X, y = data.make_spiral(n_points=200, n_classes=3, seed=0)
loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=0)

# 2. model
model = models.MLP(2, (64, 64), 3, activation=nn.GELU)

# 3. optimiser + training loop
trainer = train.Trainer(model, nn.Adam(model.parameters(), lr=5e-3), nn.cross_entropy)
trainer.fit(loader, epochs=100)

# 4. use it
print(trainer.predict_classes(X[:5]))
```

Or train a language model:

```python
from neuroforge import data, models, nn, train

text = data.load_text("builtin")
tok = data.CharTokenizer(text)
tokens = tok.encode(text)

model = models.GPT(vocab_size=tok.vocab_size, block_size=32,
                   embed_dim=64, num_heads=4, num_layers=3)

loader = data.DataLoader(data.LanguageModelDataset(tokens, 32), batch_size=16, shuffle=True)

def lm_loss(logits, targets):
    b, t, v = logits.shape
    return nn.cross_entropy(logits.reshape(b * t, v), targets.reshape(-1))

train.Trainer(model, nn.AdamW(model.parameters(), lr=3e-3), lm_loss).fit(loader, epochs=20)

print(tok.decode(model.generate(np.array([tok.encode("A machine")]), 200, temperature=0.8)[0]))
```

---

## Project layout

```
neuroforge/
├── neuroforge/
│   ├── tensor.py          # reverse-mode autograd engine (the foundation)
│   ├── nn/
│   │   ├── module.py      # Module, Parameter, Sequential, ModuleList
│   │   ├── layers.py      # Linear, Conv2d, Embedding, Dropout, pooling
│   │   ├── norm.py        # LayerNorm, RMSNorm, BatchNorm1d
│   │   ├── activations.py # ReLU, GELU, Tanh, Sigmoid, Softmax
│   │   ├── losses.py      # cross-entropy, MSE, BCE-with-logits, Huber
│   │   ├── optim.py       # SGD, RMSprop, Adam, AdamW
│   │   └── init.py        # Xavier, Kaiming, normal, uniform
│   ├── models/
│   │   ├── mlp.py         # fully-connected network
│   │   ├── cnn.py         # convolutional network
│   │   ├── rnn.py         # RNN, LSTM, GRU (+ bidirectional, sequence classifier)
│   │   └── transformer.py # multi-head attention, transformer block, GPT
│   ├── data/
│   │   ├── dataset.py     # Dataset, TensorDataset, DataLoader, splits
│   │   ├── tokenizer.py   # CharTokenizer, BPETokenizer (BPE from scratch)
│   │   └── corpora.py     # synthetic datasets + text corpora
│   ├── train/
│   │   ├── trainer.py     # the training loop
│   │   ├── callbacks.py   # EarlyStopping, ModelCheckpoint, CSVLogger, LR schedule
│   │   └── metrics.py     # accuracy, top-k, perplexity, confusion matrix, P/R/F1
│   └── utils/
│       ├── serialization.py  # save/load checkpoints and JSON
│       └── plotting.py       # loss curves, confusion matrices, boundaries
├── examples/              # five runnable end-to-end scripts
├── tests/                 # 160 tests
├── docs/                  # architecture notes and a tutorial
└── conftest.py
```

---

## What is implemented

| Area | Contents |
|---|---|
| **Autograd** | Reverse-mode autodiff over NumPy arrays: broadcasting-aware gradient reduction, topological backprop, `no_grad`, graph pruning |
| **Ops** | add, sub, mul, div, pow, matmul, sum, mean, max, exp, log, sqrt, abs, tanh, sigmoid, relu, leaky_relu, gelu, reshape, transpose, getitem, cat, stack, masked_fill, softmax, log_softmax |
| **Layers** | `Linear`, `Embedding`, `Conv2d` (im2col), `MaxPool2d`, `AvgPool2d`, `Dropout`, `Flatten`, `LayerNorm`, `RMSNorm`, `BatchNorm1d` |
| **Models** | `MLP`, `CNN`, `RNN`/`LSTM`/`GRU`, `SequenceClassifier`, `MultiHeadAttention`, `TransformerBlock`, `GPT`, `TransformerClassifier` |
| **Optimisers** | `SGD` (momentum, Nesterov, weight decay), `RMSprop`, `Adam`, `AdamW` |
| **Data** | `TensorDataset`, `LanguageModelDataset`, `DataLoader` (reproducible shuffling), train/val split, `CharTokenizer`, `BPETokenizer` |
| **Training** | `Trainer` with validation, metrics, callbacks, checkpointing, LR scheduling |

---

## The examples, and what they actually achieve

Each script trains a real model and writes plots, metrics and checkpoints to `runs/`.

```bash
python examples/01_train_mlp_spiral.py --epochs 200
python examples/02_train_cnn_shapes.py --epochs 25
python examples/03_train_lstm_sequence.py --epochs 12
python examples/04_train_char_gpt.py --epochs 30
python examples/05_tokenizer_demo.py
```

Measured results from the runs used to develop this repository:

| Example | Task | Result |
|---|---|---|
| `01` MLP | 3-class interleaved spirals | **99.2%** validation accuracy |
| `02` CNN | line-orientation images (16×16) | **97.5%** validation accuracy (chance = 33%) |
| `03` LSTM | classify random walks by final sign | **100%** validation accuracy (chance = 50%) |
| `04` GPT | character-level language modelling | train loss 2.82 → **0.84**, samples become readable English |

The GPT, after training on the bundled corpus, produces text like:

> *"A machine that le nows. The no magic in any single stalls. At few billion be cate that knows. The magic, if you are can turn that information into a nudge, the guessing slopses"*

That is a 28k-parameter model on a 1,700-character corpus. It is memorising, not generalising — which is exactly the point, and worth seeing for yourself. Point it at the tiny-shakespeare corpus (`--download`) for a more convincing result.

---

## Tests

```bash
pytest              # or: python -m pytest tests/
```

**160 tests**, covering:

- **Numerical gradient checks** — every operation's analytic gradient is compared against a central-difference numerical gradient. This is the test that matters most: it proves the backward passes are correct.
- Layer and model shape/gradient behaviour, including Conv2d's numeric gradient
- **Causal-mask verification** — changing a token at position `t` provably produces *exactly zero* change in all logits before `t`
- Overfit sanity checks — a small GPT must memorise a sequence exactly
- Dataset, loader, tokeniser and corpus behaviour
- Trainer, callbacks, metrics and serialisation

---

## Design notes

**Why autograd rather than hand-written backward passes?** Because it removes an entire class of bugs. Each layer is written once, forward-only, and its gradient falls out of the engine. The trade-off is speed: this is roughly 10–100× slower than a fused framework, which is the correct trade for something meant to be read.

**Numerical stability is handled explicitly.** `log_softmax` subtracts the row maximum before exponentiating, cross-entropy is computed in log-space, and BCE-with-logits uses the softplus form — so large logits don't overflow.

**Gradient reduction over broadcast dimensions** is the subtlest part of the engine. When NumPy broadcasts a `(3,1)` tensor against a `(3,4)` one, the gradient must be summed back down to `(3,1)`. `_unbroadcast` does this and is covered by dedicated tests.

**Reproducibility** requires seeding NumPy *before* constructing the model, because weight initialisation draws from the global RNG. The `DataLoader` reseeds deterministically per epoch when given a `seed`.

## Limitations (honestly)

- Pure NumPy, single-threaded, no GPU. Expect seconds per epoch, not milliseconds.
- No fused kernels — attention and convolution are readable, not fast.
- No automatic mixed precision, no distributed training, no CUDA.
- The `Conv2d` and pooling implementations use Python loops over kernel positions; fine for small images, slow for large ones.
- Asymmetric stride/padding is not implemented (raises `NotImplementedError` rather than failing silently).

## License

MIT. See `LICENSE`.
