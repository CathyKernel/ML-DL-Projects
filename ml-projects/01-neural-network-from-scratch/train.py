"""Train an MLP built entirely from NumPy primitives.

Examples
--------
Spiral classification with Adam::

    python train.py --dataset spirals --hidden 128 128 --optimizer adam --epochs 300

MNIST (requires torchvision)::

    python train.py --dataset mnist --hidden 256 128 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import get_dataset, one_hot
from nn import (Adam, AdamW, BatchNorm, CosineAnnealingLR, Dropout, Linear,
                MSELoss, ReLU, SGD, SGDMomentum, Sequential, Softmax,
                SoftmaxCrossEntropy, StepLR, Tanh)
from plots import plot_decision_boundary, plot_training_curves


def build_model(args, in_dim: int, out_dim: int, rng: np.random.Generator) -> Sequential:
    """MLP: [Linear -> BatchNorm -> Tanh] * L -> Linear head (or with Dropout)."""
    dims = [in_dim, *args.hidden, out_dim]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(Linear(dims[i], dims[i + 1], rng=rng))
        if args.batchnorm:
            layers.append(BatchNorm(dims[i + 1]))
        layers.append(Tanh() if args.activation == "tanh" else ReLU())
        if args.dropout > 0:
            layers.append(Dropout(args.dropout, rng=rng))
    layers.append(Linear(dims[-2], dims[-1], weight_init="xavier", rng=rng))
    return Sequential(*layers)


def build_optimizer(args, model: Sequential):
    params = [(layer, name) for layer in model.trainable_layers() for name in layer.params]
    if args.optimizer == "sgd":
        return SGD(params, lr=args.lr)
    if args.optimizer == "momentum":
        return SGDMomentum(params, lr=args.lr, momentum=0.9)
    if args.optimizer == "adam":
        return Adam(params, lr=args.lr)
    if args.optimizer == "adamw":
        return AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(args.optimizer)


def accuracy(model: Sequential, X: np.ndarray, y: np.ndarray) -> float:
    logits = model.predict(X)
    return float((logits.argmax(axis=1) == y).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="spirals",
                   choices=["spirals", "moons", "circles", "mnist"])
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    p.add_argument("--optimizer", default="adam",
                   choices=["sgd", "momentum", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batchnorm", action="store_true")
    p.add_argument("--scheduler", default="none", choices=["none", "step", "cosine"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------
    X, y, X_val, y_val = get_dataset(args.dataset, args.n_samples, args.seed)
    in_dim, n_classes = X.shape[1], int(y.max()) + 1
    print(f"dataset={args.dataset}  train={len(X)}  val={len(X_val)}  "
          f"in_dim={in_dim}  classes={n_classes}")

    # ---- model / loss / optimizer -------------------------------------------
    model = build_model(args, in_dim, n_classes, rng)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = build_optimizer(args, model)
    scheduler = None
    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ---- training loop -------------------------------------------------------
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    n = len(X)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            logits = model(X[idx], train=True)
            loss = loss_fn(logits, y[idx])
            model.zero_grad()
            model.backward(loss_fn.backward())
            optimizer.step()
            epoch_loss += loss * len(idx)

        if scheduler is not None:
            scheduler.step()

        train_logits = model.predict(X)
        val_logits = model.predict(X_val)
        history["train_loss"].append(epoch_loss / n)
        history["train_acc"].append(float((train_logits.argmax(1) == y).mean()))
        history["val_loss"].append(loss_fn(val_logits, y_val))
        history["val_acc"].append(float((val_logits.argmax(1) == y_val).mean()))
        history["lr"].append(optimizer.lr)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"epoch {epoch:4d}/{args.epochs}  loss={history['train_loss'][-1]:.4f}  "
                  f"train_acc={history['train_acc'][-1]:.4f}  "
                  f"val_acc={history['val_acc'][-1]:.4f}  lr={optimizer.lr:.2e}")

    elapsed = time.time() - t0
    print(f"\nfinished in {elapsed:.1f}s  best val_acc="
          f"{max(history['val_acc']):.4f}")

    # ---- persist artifacts ----------------------------------------------------
    model.save(out_dir / "model.npz")
    with open(out_dir / "history.json", "w") as f:
        json.dump({**history, "args": vars(args)}, f, indent=2)

    plot_training_curves(history, out_dir / "curves.png")
    if in_dim == 2:
        plot_decision_boundary(model, X, y, out_dir / "decision_boundary.png",
                               title=f"{args.dataset} — MLP decision boundary")
    print(f"artifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
