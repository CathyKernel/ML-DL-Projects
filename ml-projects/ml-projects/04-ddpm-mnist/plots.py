"""Plot the DDPM training loss curve from runs/history.json.

Examples::

    python plots.py
    python plots.py --history runs/history.json --out loss_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


def plot_loss_curve(history: dict, path: str) -> None:
    """Render epoch-mean training loss as a PNG line plot."""
    epochs, losses = history["epoch"], history["loss"]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(epochs, losses, marker="o", markersize=3, color="#d62728")
    ax.set_title("DDPM training loss (noise-prediction MSE)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--history", default="runs/history.json",
                   help="history.json written by train.py")
    p.add_argument("--out", default="loss_curve.png")
    args = p.parse_args()

    path = Path(args.history)
    if not path.exists():
        raise SystemExit(f"history file not found: {path} — run train.py first")
    history = json.loads(path.read_text())
    if not history.get("loss"):
        raise SystemExit(f"{path} contains no loss values — was training interrupted?")

    plot_loss_curve(history, args.out)
    print(f"saved loss curve ({len(history['loss'])} epochs) -> {args.out}")


if __name__ == "__main__":
    main()
