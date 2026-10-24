#!/usr/bin/env python3
"""Example 3 — train an LSTM to classify random walks by their final sign.

Run:
    python examples/03_train_lstm_sequence.py --epochs 25

A recurrent network must carry evidence across the whole sequence to decide
whether the running total ends up positive. It is a clean test of memory.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import neuroforge as nf  # noqa: E402
from neuroforge import data, models, nn, train, utils  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--cell", type=str, default="lstm", choices=["rnn", "lstm", "gru"])
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/03_lstm_sequence")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- data --------------------------------------------------------
    X, y = data.make_sequence_task(n_samples=args.samples, seq_len=args.seq_len, seed=args.seed)
    (X_train, y_train), (X_val, y_val) = data.train_val_split(
        X, y, val_fraction=0.2, seed=args.seed
    )
    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = data.DataLoader(
        data.TensorDataset(X_val, y_val), batch_size=256, shuffle=False
    )
    print(f"train: {X_train.shape}  val: {X_val.shape}  cell={args.cell}")

    # ---- model -------------------------------------------------------
    model = models.SequenceClassifier(
        input_size=1, num_classes=2, hidden_size=32, num_layers=1,
        cell_type=args.cell, dropout=0.0,
    )
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    trainer = train.Trainer(
        model, optimizer, nn.cross_entropy,
        callbacks=[
            train.EarlyStopping("val_loss", patience=10, restore_best=True),
            train.CSVLogger(os.path.join(args.out, "history.csv")),
        ],
        verbose=True, seed=args.seed,
    )

    history = trainer.fit(train_loader, epochs=args.epochs, val_loader=val_loader)
    print(f"\nparameters: {model.num_parameters()}")

    metrics = trainer.evaluate(val_loader)
    print(f"final: val_loss={metrics['loss']:.4f}  val_accuracy={metrics['accuracy']:.4f}")
    print("(a coin flip scores 0.50, so anything well above that is real learning)")

    cm = train.confusion_matrix(trainer.predict_classes(X_val), y_val, num_classes=2)
    utils.plot_history(history, os.path.join(args.out, "history.png"))
    utils.plot_confusion_matrix(cm, os.path.join(args.out, "confusion.png"),
                                class_names=["negative", "positive"])
    print(f"\nartefacts written to {args.out}/")


if __name__ == "__main__":
    main()
