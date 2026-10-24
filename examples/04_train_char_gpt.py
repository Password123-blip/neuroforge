#!/usr/bin/env python3
"""Example 4 — train a character-level GPT from scratch and sample from it.

Run:
    python examples/04_train_char_gpt.py --epochs 30 --download
    python examples/04_train_char_gpt.py --epochs 30          # offline, built-in text

This is the full language-modelling pipeline: character tokeniser, sliding-window
dataset, a decoder-only transformer, cross-entropy training, and autoregressive
sampling at the end.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import neuroforge as nf  # noqa: E402
from neuroforge import data, models, nn, train, utils  # noqa: E402


def lm_loss(logits, targets):
    """Cross-entropy over every position in the sequence at once."""
    b, t, vocab = logits.shape
    return nn.cross_entropy(logits.reshape(b * t, vocab), np.asarray(targets).reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--download", action="store_true",
                        help="try to fetch the tiny-shakespeare corpus")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/04_char_gpt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- text + tokeniser -------------------------------------------
    text = data.load_text("shakespeare" if args.download else "builtin",
                          download=args.download)
    tokenizer = data.CharTokenizer(text)
    tokens = tokenizer.encode(text)
    print(f"corpus: {len(text)} characters, vocab={tokenizer.vocab_size}")

    split = int(0.9 * len(tokens))
    train_tokens, val_tokens = tokens[:split], tokens[split:]

    train_loader = data.DataLoader(
        data.LanguageModelDataset(train_tokens, args.block_size),
        batch_size=args.batch_size, shuffle=True, seed=args.seed,
    )
    val_loader = data.DataLoader(
        data.LanguageModelDataset(val_tokens, args.block_size),
        batch_size=args.batch_size, shuffle=False,
    )
    print(f"batches per epoch: {len(train_loader)}")

    # ---- model -------------------------------------------------------
    model = models.GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    optimizer = nn.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    print(f"parameters: {model.num_parameters()}")

    # ---- sample before training (should be gibberish) ----------------
    prompt = np.array([tokenizer.encode(text[:16])])
    print("\n--- before training ---")
    print(tokenizer.decode(model.generate(prompt, max_new_tokens=120, temperature=0.8, seed=args.seed)[0]))

    trainer = train.Trainer(
        model, optimizer, lm_loss,
        callbacks=[
            train.CSVLogger(os.path.join(args.out, "history.csv")),
            train.LRSchedulerCallback(every=max(1, args.epochs // 3), factor=0.5),
        ],
        verbose=True, seed=args.seed,
    )

    # ---- train -------------------------------------------------------
    history = trainer.fit(train_loader, epochs=args.epochs, val_loader=val_loader)

    val = trainer.evaluate(val_loader)
    print(f"\nfinal: val_loss={val['loss']:.4f}  "
          f"perplexity={train.perplexity(val['loss']):.2f}")

    # ---- sample after training ---------------------------------------
    model.eval()
    print("\n--- after training ---")
    for temperature in (0.5, 0.8):
        sample = model.generate(prompt, max_new_tokens=240, temperature=temperature,
                                top_k=20, seed=args.seed)
        print(f"\n[temperature={temperature}]")
        print(tokenizer.decode(sample[0]))

    utils.plot_history(history, os.path.join(args.out, "history.png"))
    utils.save_checkpoint(os.path.join(args.out, "model.npz"), model, optimizer)
    tokenizer.save(os.path.join(args.out, "tokenizer.json"))
    print(f"\nartefacts written to {args.out}/")


if __name__ == "__main__":
    main()
