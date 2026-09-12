"""Visualization helpers: training curves and 2-D decision boundaries."""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


def plot_training_curves(history: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
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


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, path: str,
                           title: str = "Decision boundary", step: int = 300) -> None:
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, step),
                         np.linspace(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    probs = model.predict(grid).argmax(axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.contourf(xx, yy, probs, alpha=0.35, levels=np.arange(-0.5, probs.max() + 1.5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=12, edgecolors="k", linewidths=0.3)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    fig.savefig(path, dpi=150)
    plt.close(fig)
