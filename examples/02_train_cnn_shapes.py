#!/usr/bin/env python3
"""Example 2 — train a CNN to recognise line orientations in tiny images.

Run:
    python examples/02_train_cnn_shapes.py --epochs 12

The task is synthetic but genuinely visual: the network must learn oriented
edge detectors, which is exactly what convolution is for.
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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--samples", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/02_cnn_shapes")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- data --------------------------------------------------------
    X, y = data.make_shape_images(n_samples=args.samples, size=16, noise=0.35, seed=args.seed)
    (X_train, y_train), (X_val, y_val) = data.train_val_split(
        X, y, val_fraction=0.2, seed=args.seed
    )
    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = data.DataLoader(
        data.TensorDataset(X_val, y_val), batch_size=128, shuffle=False
    )
    print(f"train: {X_train.shape}  val: {X_val.shape}")

    # ---- model -------------------------------------------------------
    model = models.CNN(in_channels=1, num_classes=3, channels=(8, 16),
                       kernel_size=3, hidden=64, dropout=0.1)
    optimizer = nn.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = train.Trainer(
        model, optimizer, nn.cross_entropy,
        callbacks=[
            train.EarlyStopping("val_loss", patience=8, restore_best=True),
            train.CSVLogger(os.path.join(args.out, "history.csv")),
        ],
        verbose=True, seed=args.seed,
    )

    history = trainer.fit(train_loader, epochs=args.epochs, val_loader=val_loader)
    print(f"\nparameters: {model.num_parameters()}")

    metrics = trainer.evaluate(val_loader)
    print(f"final: val_loss={metrics['loss']:.4f}  val_accuracy={metrics['accuracy']:.4f}")

    preds = trainer.predict_classes(X_val)
    cm = train.confusion_matrix(preds, y_val, num_classes=3)
    print("confusion matrix:\n", cm)

    utils.plot_history(history, os.path.join(args.out, "history.png"))
    utils.plot_confusion_matrix(cm, os.path.join(args.out, "confusion.png"),
                                class_names=["vertical", "horizontal", "diagonal"])
    utils.plot_image_grid(X_val[:16], os.path.join(args.out, "samples.png"),
                          titles=[f"y={t}" for t in y_val[:16]])
    utils.save_checkpoint(os.path.join(args.out, "model.npz"), model, optimizer)
    print(f"\nartefacts written to {args.out}/")


if __name__ == "__main__":
    main()
