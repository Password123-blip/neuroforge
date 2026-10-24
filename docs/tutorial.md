# Tutorial: build and train a model, step by step

This walks through the whole pipeline the way the examples do, but slowly, explaining what each piece is for. If you read one document in this repository, read this one.

---

## Step 1 — Understand what a gradient is doing here

Before any neural network, make sure the engine behaves. Create a tensor that requires gradients, do some arithmetic, and call `.backward()`:

```python
from neuroforge.tensor import Tensor

x = Tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()      # 4 + 9 = 13
y.backward()
print(x.grad)          # [4.0, 6.0]  = 2x
```

The derivative of $$x^2$$ is $$2x$$, so the gradient at `[2, 3]` is `[4, 6]`. That is the entire idea. A neural network is a very large expression like this one, and `backward()` computes its derivative with respect to every parameter at once.

Two things to internalise now, because they cause most beginner bugs:

1. **Gradients accumulate.** If you use a tensor twice, contributions add. Call `zero_grad()` before each step.
2. **`.backward()` needs a scalar** (or an explicit gradient). We always reduce a loss to a single number first.

---

## Step 2 — Your first layer

```python
import numpy as np
from neuroforge import nn
from neuroforge.tensor import Tensor

layer = nn.Linear(3, 2)                 # 3 inputs -> 2 outputs
x = Tensor(np.random.randn(5, 3), requires_grad=True)
y = layer(x)
print(y.shape)                          # (5, 2)
```

A `Linear` layer computes $$y = xW^T + b$$. You never wrote that backward pass — the engine derived it from `@` and `+`.

Check the shapes of the gradients:

```python
y.sum().backward()
print(layer.weight.grad.shape)          # (2, 3)  — same as the weight
print(layer.bias.grad.shape)            # (2,)
print(x.grad.shape)                     # (5, 3)  — same as the input
```

A gradient always has the same shape as the thing it belongs to. If it doesn't, something is wrong.

---

## Step 3 — A loss, and one manual training step

We need something to minimise. Cross-entropy is the standard loss for classification:

```python
logits = layer(x)                       # (5, 2) raw scores
targets = np.array([0, 1, 0, 1, 1])
loss = nn.cross_entropy(logits, targets)
print(loss.item())                      # a positive number
```

Now do one full optimisation step by hand, so the loop is not a black box:

```python
optimizer = nn.Adam(layer.parameters(), lr=1e-2)

optimizer.zero_grad()        # 1. clear old gradients
loss = nn.cross_entropy(layer(x), targets)
loss.backward()              # 2. compute new gradients
optimizer.step()             # 3. nudge the parameters downhill
```

Those three lines, repeated, are all of deep learning. Everything else is bookkeeping.

- `zero_grad` — because gradients accumulate, start clean each step.
- `backward` — fill in `.grad` for every parameter.
- `step` — move each parameter a little in the direction that reduces the loss.

---

## Step 4 — Use the Trainer

The `Trainer` wraps that loop, adds batching, validation, metrics and callbacks.

```python
import numpy as np
from neuroforge import data, models, nn, train

X, y = data.make_spiral(n_points=200, n_classes=3, noise=0.15, seed=0)
loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=0)

model = models.MLP(2, (64, 64), 3, activation=nn.GELU)
optimizer = nn.Adam(model.parameters(), lr=5e-3)

trainer = train.Trainer(model, optimizer, nn.cross_entropy, verbose=True)
history = trainer.fit(loader, epochs=100)
```

You should see the loss fall from around 1.1 to below 0.1. On the spiral problem this reaches about **99% accuracy**.

### Why the spiral is a good first task

Three interleaved spirals cannot be separated by any straight line. A network with no hidden layer will plateau around 33% — pure chance. Add hidden layers and it solves it. That difference *is* the value of depth, and you can see it in your own terminal.

Try it:

```python
flat = models.MLP(2, (), 3)          # no hidden layer
```

---

## Step 5 — Train a language model

Now the same loop, on text. The only new ideas are tokenisation and next-token prediction.

```python
from neuroforge import data, models, nn, train

text = data.load_text("builtin")
tok = data.CharTokenizer(text)
tokens = tok.encode(text)

split = int(0.9 * len(tokens))
train_loader = data.DataLoader(
    data.LanguageModelDataset(tokens[:split], block_size=32),
    batch_size=16, shuffle=True, seed=0,
)

model = models.GPT(vocab_size=tok.vocab_size, block_size=32,
                   embed_dim=64, num_heads=4, num_layers=3, dropout=0.1)
```

The loss needs flattening, because the model returns logits for every position at once:

```python
def lm_loss(logits, targets):
    b, t, vocab = logits.shape
    return nn.cross_entropy(logits.reshape(b * t, vocab), targets.reshape(-1))

train.Trainer(model, nn.AdamW(model.parameters(), lr=3e-3), lm_loss).fit(
    train_loader, epochs=30
)
```

Then sample:

```python
model.eval()
prompt = np.array([tok.encode("A machine")])
out = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=20, seed=0)
print(tok.decode(out[0]))
```

### What to expect, honestly

On the bundled 1,700-character corpus, a model this size will start producing English-looking text within ~20 epochs — real words, plausible grammar fragments. It will also **memorise** the corpus, so validation loss rises while training loss falls. That is not a bug in the framework; it is what happens when the dataset is tiny.

To see genuine generalisation, use more text:

```bash
python examples/04_train_char_gpt.py --download --epochs 30
```

### Temperature and top-k

- `temperature` scales the logits before sampling. Below 1.0 the distribution sharpens (safer, more repetitive); above 1.0 it flattens (more surprising, more mistakes).
- `top_k` discards everything but the k most likely tokens before sampling, which removes the long tail of nonsense.

---

## Step 6 — Read the loss curve

```python
from neuroforge import utils
utils.plot_history(history, "history.png")
```

What the shapes mean:

| Pattern | Meaning | Fix |
|---|---|---|
| Both losses falling steadily | Healthy | — |
| Train falls, validation rises | Overfitting | More data, more dropout, smaller model |
| Both flat and high | Underfitting / bad LR | Bigger model, higher learning rate |
| Loss jumps to `nan` | Exploding gradients | Lower the learning rate |
| Loss barely moves | Learning rate too small, or gradients not flowing | Check `zero_grad` is called, raise LR |

The most common cause of "it isn't learning" is a learning rate that is wrong by an order of magnitude. Adam tolerates `1e-3` to `1e-2` well; SGD usually wants more.

---

## Step 7 — Add early stopping and checkpointing

```python
trainer = train.Trainer(
    model, optimizer, nn.cross_entropy,
    callbacks=[
        train.EarlyStopping("val_loss", patience=10, restore_best=True),
        train.ModelCheckpoint("runs/best.npz", monitor="val_loss"),
        train.CSVLogger("runs/history.csv"),
    ],
)
```

`EarlyStopping` stops when validation stops improving and (with `restore_best=True`) puts the best weights back — so you keep the model from the best epoch, not the last one.

Reload a checkpoint later:

```python
from neuroforge import utils
utils.load_checkpoint("runs/best.npz", model)
```

---

## Step 8 — Where to go next

- **Read `docs/architecture.md`** for why the engine is built the way it is, and the specific bugs the tests are designed to catch.
- **Break things on purpose.** Delete the `optimizer.zero_grad()` call and watch training degrade. Remove the causal mask in `GPT` and watch the model cheat. Change `_unbroadcast` to a plain assignment and watch it fail to learn. Breaking working code is the fastest way to understand it.
- **Try the other examples.** The CNN learns oriented edge detectors; the LSTM integrates evidence across time.
- **Scale it up.** More layers, wider embeddings, more text. You will hit the speed wall quickly — that wall is precisely why real frameworks exist, and you will understand it much better for having hit it.

---

## Common errors and what they mean

| Error | Cause |
|---|---|
| `grad must be provided when calling backward() on a non-scalar tensor` | You called `.backward()` on a non-scalar. Reduce with `.sum()` or `.mean()` first. |
| Loss is `nan` | Learning rate too high, or `log(0)` somewhere. Use `log_softmax`-based losses. |
| `shape mismatch` in `load_state_dict` | The checkpoint was saved from a different architecture. |
| Model output has wrong shape | Usually a missing `Flatten` before a `Linear` layer in a CNN. |
| Gradients are all `None` | You forgot `.backward()`, or called it inside a `no_grad()` block. |
| Training loss won't fall | Check `zero_grad()` is called, check the LR, check the labels are aligned with the inputs. |
