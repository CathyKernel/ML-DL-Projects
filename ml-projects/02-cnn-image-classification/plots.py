"""Plot training curves (loss / accuracy / learning rate) from history.csv.

Reads the CSV written by ``train.py`` and renders a three-panel figure to
``curves.png``.

Example::

    python plots.py --history runs/history.csv --out runs/curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


def read_history(path: str | Path) -> dict[str, list[float]]:
    """Parse ``history.csv`` into a dict of float columns."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} contains no epochs")
    return {key: [float(r[key]) for r in rows] for key in rows[0]}


def plot_curves(history: dict[str, list[float]], path: str | Path) -> None:
    """Render loss + accuracy + LR panels (one row, three columns)."""
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("epoch")

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Top-1 accuracy (%)")
    axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, history["lr"])
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("epoch")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Training curves")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--history", default="runs/history.csv",
                   help="CSV written by train.py")
    p.add_argument("--out", default="runs/curves.png", help="output PNG path")
    args = p.parse_args()

    history = read_history(args.history)
    plot_curves(history, args.out)
    print(f"saved {args.out} ({len(history['epoch'])} epochs)")


if __name__ == "__main__":
    main()
