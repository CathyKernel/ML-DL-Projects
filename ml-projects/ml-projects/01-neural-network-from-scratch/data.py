"""Synthetic 2-D classification datasets + optional MNIST loader.

The synthetic datasets (two-spirals, moons, circles) are classic benchmarks for
visualizing decision boundaries of small MLPs. Implementations use only NumPy.
"""

from __future__ import annotations

import numpy as np


def make_spirals(n_samples: int = 600, n_classes: int = 3, noise: float = 0.2,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The classic two/three-spiral dataset — non-linear, great for MLP demos."""
    rng = rng or np.random.default_rng(0)
    X, y = [], []
    points_per_class = n_samples // n_classes
    for c in range(n_classes):
        r = np.linspace(0.05, 1.0, points_per_class)                    # radius
        t = np.linspace(c * 4 * np.pi / n_classes,
                        (c + 1) * 4 * np.pi / n_classes, points_per_class) \
            + rng.normal(0, noise, points_per_class)                    # angle + noise
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(points_per_class, c))
    return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64)


def make_moons(n_samples: int = 600, noise: float = 0.15,
               rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles (same idea as ``sklearn.datasets.make_moons``)."""
    rng = rng or np.random.default_rng(0)
    n = n_samples // 2
    t = rng.uniform(0, np.pi, n)
    upper = np.c_[np.cos(t), np.sin(t)]
    t = rng.uniform(0, np.pi, n)
    lower = np.c_[1.0 - np.cos(t), 0.5 - np.sin(t)]
    X = np.concatenate([upper, lower]) + rng.normal(0, noise, (2 * n, 2))
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X.astype(np.float32), y.astype(np.int64)


def make_circles(n_samples: int = 600, noise: float = 0.08, factor: float = 0.4,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Concentric circles — linearly inseparable in 2-D."""
    rng = rng or np.random.default_rng(0)
    n = n_samples // 2
    outer_angle = rng.uniform(0, 2 * np.pi, n)
    inner_angle = rng.uniform(0, 2 * np.pi, n)
    outer = np.c_[np.cos(outer_angle), np.sin(outer_angle)]
    inner = factor * np.c_[np.cos(inner_angle), np.sin(inner_angle)]
    X = np.concatenate([outer, inner]) + rng.normal(0, noise, (2 * n, 2))
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X.astype(np.float32), y.astype(np.int64)


def make_regression(n_samples: int = 400, n_features: int = 1, noise: float = 0.1,
                    rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linear regression toy data: ``y = Xw + b + noise``."""
    rng = rng or np.random.default_rng(0)
    X = rng.normal(0, 1.0, (n_samples, n_features))
    w = rng.normal(0, 2.0, n_features)
    y = X @ w + rng.normal(0, noise, n_samples) + 0.5
    return X.astype(np.float32), y.astype(np.float32)


def load_mnist(val_split: float = 0.1,
               rng: np.random.Generator | None = None) -> tuple[np.ndarray, ...]:
    """Load MNIST via torchvision (if available) as flat float arrays in [0, 1].

    Returns (X_train, y_train, X_val, y_val). Requires ``torchvision`` — the
    synthetic datasets above exist precisely so this optional dependency can be
    avoided for the core demos.
    """
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_mnist requires torchvision: pip install torchvision") from exc

    train = MNIST("data", train=True, download=True)
    test = MNIST("data", train=False, download=True)
    X = np.concatenate([train.data.numpy(), test.data.numpy()]) \
        .reshape(-1, 784).astype(np.float32) / 255.0
    y = np.concatenate([train.targets.numpy(), test.targets.numpy()]).astype(np.int64)

    rng = rng or np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_split)
    return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]


def get_dataset(name: str, n_samples: int = 600, seed: int = 0,
                val_split: float = 0.2):
    """Convenience dispatcher used by ``train.py``.

    Always returns a 4-tuple ``(X_train, y_train, X_val, y_val)`` so callers
    don't need dataset-specific logic.
    """
    rng = np.random.default_rng(seed)
    if name == "spirals":
        X, y = make_spirals(n_samples, rng=rng)
    elif name == "moons":
        X, y = make_moons(n_samples, rng=rng)
    elif name == "circles":
        X, y = make_circles(n_samples, rng=rng)
    elif name == "regression":
        X, y = make_regression(n_samples, rng=rng)
        return X, y, X, y  # regression callers handle evaluation themselves
    elif name == "mnist":
        return load_mnist(val_split=val_split, rng=rng)
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from "
                         "spirals|moons|circles|regression|mnist")

    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_split))
    return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out
