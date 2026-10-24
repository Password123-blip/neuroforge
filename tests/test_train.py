"""Tests for the training loop, callbacks, metrics and serialisation."""

from __future__ import annotations

import os

import numpy as np
import pytest

from neuroforge import data, nn, train, utils
from neuroforge.models import MLP
from neuroforge.tensor import Tensor


def make_problem(n=300, seed=0):
    X, y = data.make_spiral(n_points=n // 3, n_classes=3, noise=0.2, seed=seed)
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=seed)
    return X, y, loader


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def test_accuracy_from_indices():
    assert train.accuracy([0, 1, 2, 1], [0, 1, 2, 2]) == pytest.approx(0.75)


def test_accuracy_from_scores():
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert train.accuracy(scores, [0, 1]) == pytest.approx(1.0)


def test_top_k_accuracy():
    scores = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    # row 0 ranks: 2, 1, 0    row 1 ranks: 0, 1, 2
    assert train.top_k_accuracy(scores, [2, 0], k=1) == pytest.approx(1.0)
    assert train.top_k_accuracy(scores, [1, 1], k=1) == pytest.approx(0.0)
    assert train.top_k_accuracy(scores, [1, 1], k=2) == pytest.approx(1.0)  # 1 is 2nd in both
    assert train.top_k_accuracy(scores, [2, 2], k=2) == pytest.approx(0.5)  # only row 0


def test_perplexity():
    assert train.perplexity(0.0) == pytest.approx(1.0)
    assert train.perplexity(np.log(10.0)) == pytest.approx(10.0, rel=1e-6)


def test_confusion_matrix_counts():
    cm = train.confusion_matrix([0, 1, 1, 0], [0, 0, 1, 1], num_classes=2)
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 0] == 1 and cm[1, 1] == 1


def test_precision_recall_f1():
    p, r, f = train.precision_recall_f1([0, 1, 1, 1], [0, 1, 1, 0], num_classes=2)
    assert p.shape == (2,) and r.shape == (2,) and f.shape == (2,)
    assert np.all((p >= 0) & (p <= 1))


# ----------------------------------------------------------------------
# trainer
# ----------------------------------------------------------------------
def test_trainer_reduces_loss():
    X, y, loader = make_problem()
    model = MLP(2, (32, 32), 3)
    optimizer = nn.Adam(model.parameters(), lr=5e-3)
    trainer = train.Trainer(model, optimizer, nn.cross_entropy, verbose=False, seed=0)

    history = trainer.fit(loader, epochs=30)
    assert history["loss"][-1] < history["loss"][0]


def test_trainer_records_history_length():
    X, y, loader = make_problem()
    model = MLP(2, (16,), 3)
    trainer = train.Trainer(model, nn.Adam(model.parameters(), lr=1e-2),
                            nn.cross_entropy, verbose=False)
    history = trainer.fit(loader, epochs=4)
    assert len(history["loss"]) == 4
    assert len(history["epoch_time"]) == 4


def test_trainer_validation_metrics():
    X, y, loader = make_problem()
    val_loader = data.DataLoader(data.TensorDataset(X, y), batch_size=64, shuffle=False)
    model = MLP(2, (16,), 3)
    trainer = train.Trainer(model, nn.Adam(model.parameters(), lr=1e-2),
                            nn.cross_entropy, verbose=False)
    history = trainer.fit(loader, epochs=3, val_loader=val_loader)
    assert "val_loss" in history
    assert "val_accuracy" in history
    assert 0.0 <= history["val_accuracy"][-1] <= 1.0


def test_trainer_evaluate_and_predict():
    X, y, loader = make_problem()
    model = MLP(2, (16,), 3)
    trainer = train.Trainer(model, nn.Adam(model.parameters()), nn.cross_entropy, verbose=False)
    metrics = trainer.evaluate(loader)
    assert "loss" in metrics and "accuracy" in metrics
    assert trainer.predict(X).shape == (len(X), 3)
    assert trainer.predict_classes(X).shape == (len(X),)


def test_trainer_requires_targets():
    ds = data.TensorDataset(np.random.default_rng(0).normal(size=(10, 2)))
    loader = data.DataLoader(ds, batch_size=5)
    model = MLP(2, (4,), 2)
    trainer = train.Trainer(model, nn.Adam(model.parameters()), nn.cross_entropy, verbose=False)
    with pytest.raises(ValueError):
        trainer.fit(loader, epochs=1)


def test_trainer_is_reproducible():
    X, y, _ = make_problem(seed=1)

    def run():
        # Seed NumPy *before* building the model: weight init draws from the
        # global RNG, so the seed has to be in place first.
        np.random.seed(7)
        model = MLP(2, (16,), 3)
        loader = data.DataLoader(
            data.TensorDataset(X, y), batch_size=32, shuffle=True, seed=7
        )
        trainer = train.Trainer(model, nn.Adam(model.parameters(), lr=1e-2),
                                nn.cross_entropy, verbose=False)
        return trainer.fit(loader, epochs=3)["loss"]

    assert run() == pytest.approx(run())


# ----------------------------------------------------------------------
# callbacks
# ----------------------------------------------------------------------
def test_early_stopping_triggers():
    X, y, loader = make_problem()
    model = MLP(2, (8,), 3)
    stopper = train.EarlyStopping("loss", patience=2, min_delta=1e9, restore_best=True)
    trainer = train.Trainer(model, nn.Adam(model.parameters(), lr=1e-4),
                            nn.cross_entropy, callbacks=[stopper], verbose=False)
    history = trainer.fit(loader, epochs=50)
    assert len(history["loss"]) < 50      # stopped early


def test_model_checkpoint_writes_file(tmp_path):
    X, y, loader = make_problem()
    model = MLP(2, (8,), 3)
    path = tmp_path / "ckpt.npz"
    trainer = train.Trainer(
        model, nn.Adam(model.parameters(), lr=1e-2), nn.cross_entropy,
        callbacks=[train.ModelCheckpoint(str(path), monitor="loss")], verbose=False,
    )
    trainer.fit(loader, epochs=3)
    assert os.path.exists(path)


def test_csv_logger_writes_rows(tmp_path):
    X, y, loader = make_problem()
    model = MLP(2, (8,), 3)
    path = tmp_path / "history.csv"
    trainer = train.Trainer(
        model, nn.Adam(model.parameters(), lr=1e-2), nn.cross_entropy,
        callbacks=[train.CSVLogger(str(path))], verbose=False,
    )
    trainer.fit(loader, epochs=3)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 4          # header + 3 epochs


def test_lr_scheduler_reduces_lr():
    X, y, loader = make_problem()
    model = MLP(2, (8,), 3)
    optimizer = nn.Adam(model.parameters(), lr=1e-2)
    trainer = train.Trainer(
        model, optimizer, nn.cross_entropy,
        callbacks=[train.LRSchedulerCallback(every=2, factor=0.5)], verbose=False,
    )
    trainer.fit(loader, epochs=4)
    assert optimizer.lr == pytest.approx(1e-2 * 0.5 ** 2)


def test_history_attribute_access():
    h = train.History()
    h.append(loss=1.0)
    h.append(loss=0.5)
    assert h.loss == [1.0, 0.5]
    assert h.best("loss") == 0.5
    assert h.latest()["loss"] == 0.5
    with pytest.raises(AttributeError):
        _ = h.nonexistent


# ----------------------------------------------------------------------
# serialisation
# ----------------------------------------------------------------------
def test_checkpoint_round_trip(tmp_path):
    model = MLP(4, (8,), 2)
    path = utils.save_checkpoint(str(tmp_path / "m.npz"), model, extra={"epoch": 3})
    other = MLP(4, (8,), 2)
    loaded = utils.load_checkpoint(path, other)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), other.named_parameters()):
        assert np.allclose(p1.data, p2.data)
    assert loaded["meta"]["extra"]["epoch"] == 3


def test_checkpoint_appends_npz_suffix(tmp_path):
    model = MLP(3, (4,), 2)
    path = utils.save_checkpoint(str(tmp_path / "noext"), model)
    assert path.endswith(".npz")


def test_load_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.load_checkpoint(str(tmp_path / "missing.npz"))


def test_json_round_trip(tmp_path):
    path = utils.save_json(str(tmp_path / "x.json"),
                           {"a": 1, "b": np.float64(2.5), "c": np.arange(3)})
    loaded = utils.load_json(path)
    assert loaded["a"] == 1
    assert loaded["b"] == 2.5
    assert loaded["c"] == [0, 1, 2]


# ----------------------------------------------------------------------
# plotting (smoke tests — must not raise, must write files)
# ----------------------------------------------------------------------
def test_plot_history(tmp_path):
    h = train.History()
    for i in range(5):
        h.append(loss=1.0 / (i + 1), val_loss=1.2 / (i + 1))
    out = utils.plot_history(h, str(tmp_path / "h.png"))
    assert os.path.exists(out)


def test_plot_confusion_matrix(tmp_path):
    cm = np.array([[5, 1], [2, 7]])
    out = utils.plot_confusion_matrix(cm, str(tmp_path / "cm.png"), normalize=True)
    assert os.path.exists(out)


def test_plot_decision_boundary(tmp_path):
    X, y = data.make_spiral(30, 3, seed=0)
    model = MLP(2, (8,), 3)
    out = utils.plot_decision_boundary(model, X, y, str(tmp_path / "db.png"), resolution=20)
    assert os.path.exists(out)


def test_plot_image_grid(tmp_path):
    X, _ = data.make_shape_images(8, size=8, seed=0)
    out = utils.plot_image_grid(X, str(tmp_path / "grid.png"))
    assert os.path.exists(out)
