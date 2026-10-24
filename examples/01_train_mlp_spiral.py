#!/usr/bin/env python3
"""Example 1 — train an MLP to separate three interleaved spirals.

Run:
    python examples/01_train_mlp_spiral.py --epochs 200

This is the "hello world" of NeuroForge: a small fully-connected network on a
problem that a linear model cannot solve.
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/01_mlp_spiral")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- data --------------------------------------------------------
    X, y = data.make_spiral(n_points=200, n_classes=3, noise=0.15, seed=args.seed)
    (X_train, y_train), (X_val, y_val) = data.train_val_split(
        X, y, val_fraction=0.2, seed=args.seed
    )
    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = data.DataLoader(
        data.TensorDataset(X_val, y_val), batch_size=256, shuffle=False
    )
    print(f"train: {X_train.shape}  val: {X_val.shape}")

    # ---- model -------------------------------------------------------
    model = models.MLP(2, (args.hidden, args.hidden), 3, activation=nn.GELU, dropout=0.0)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    print(f"parameters: {model.num_parameters()}")

    trainer = train.Trainer(
        model, optimizer, nn.cross_entropy,
        callbacks=[
            train.EarlyStopping("val_loss", patience=40, restore_best=True),
            train.ModelCheckpoint(os.path.join(args.out, "best.npz"), "val_loss"),
            train.CSVLogger(os.path.join(args.out, "history.csv")),
        ],
        verbose=True, seed=args.seed,
    )

    # ---- train -------------------------------------------------------
    history = trainer.fit(train_loader, epochs=args.epochs, val_loader=val_loader)

    # ---- evaluate ----------------------------------------------------
    metrics = trainer.evaluate(val_loader)
    print(f"\nfinal: val_loss={metrics['loss']:.4f}  val_accuracy={metrics['accuracy']:.4f}")

    preds = trainer.predict_classes(X_val)
    cm = train.confusion_matrix(preds, y_val, num_classes=3)
    print("confusion matrix:\n", cm)

    utils.plot_history(history, os.path.join(args.out, "history.png"))
    utils.plot_confusion_matrix(cm, os.path.join(args.out, "confusion.png"))
    utils.plot_decision_boundary(model, X, y, os.path.join(args.out, "boundary.png"))
    utils.save_json(os.path.join(args.out, "metrics.json"),
                    {"val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]})
    print(f"\nartefacts written to {args.out}/")


if __name__ == "__main__":
    main()
