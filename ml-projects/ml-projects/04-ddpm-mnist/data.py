"""Datasets for diffusion training: real MNIST or an offline synthetic set.

The synthetic option generates 28×28 grayscale images of filled shapes
(circle / rectangle / triangle on a dark background) with pure NumPy, so the
whole training + sampling pipeline runs with zero network access.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

IMG_SIZE = 28

# torchvision is imported lazily so that `--dataset synthetic` works even in
# environments where torchvision (or its image backend) is unavailable.
try:
    from torchvision import datasets
    from torchvision.transforms import Compose, Normalize, ToTensor
except Exception as exc:  # ImportError or a broken image backend
    datasets = None
    Compose = Normalize = ToTensor = None
    _TORCHVISION_ERROR = exc


# --------------------------------------------------------------------- MNIST
def get_dataloader(dataset: str = "mnist", batch_size: int = 128, num_workers: int = 2,
                   root: str = "data", n_synthetic: int = 8192,
                   seed: int = 42, shuffle: bool = True) -> DataLoader:
    """Return a training DataLoader of images in [-1, 1].

    Args:
        dataset: "mnist" (60k real digits, downloaded on first use) or
            "synthetic" (offline random shapes, no network needed).
        batch_size: batch size.
        num_workers: DataLoader workers (forced to 0 for the synthetic set,
            which is a small in-memory TensorDataset).
        root: directory where MNIST is downloaded/cached.
        n_synthetic: number of synthetic images to generate.
        seed: RNG seed for the synthetic generator.
    """
    if dataset == "mnist":
        return _mnist_loader(batch_size, num_workers, root, shuffle)
    if dataset == "synthetic":
        images, labels = make_synthetic_shapes(n_synthetic, seed=seed)
        ds = TensorDataset(images, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    raise ValueError(f"unknown dataset {dataset!r} (expected 'mnist' or 'synthetic')")


def _mnist_loader(batch_size: int, num_workers: int, root: str, shuffle: bool) -> DataLoader:
    if datasets is None:
        raise ImportError(
            "torchvision is required for --dataset mnist but could not be imported. "
            f"Underlying error: {_TORCHVISION_ERROR}. "
            "Use --dataset synthetic for a fully offline run."
        )
    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])   # [0,1] -> [-1,1]
    try:
        ds = datasets.MNIST(root, train=True, download=True, transform=transform)
    except Exception as exc:
        raise RuntimeError(
            "\nCould not download the MNIST dataset (no network, or the hosting "
            "server is unreachable).\n"
            "You can still train and sample fully offline with the built-in "
            "synthetic shape dataset:\n"
            "    python train.py --dataset synthetic --epochs 1 --limit-batches 20\n"
            f"(underlying error: {exc})"
        ) from exc
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


# ----------------------------------------------------------------- synthetic
def make_synthetic_shapes(n: int, size: int = IMG_SIZE,
                          seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``n`` images of filled shapes on a dark background.

    Each image contains one filled shape — a circle, a rectangle, or a
    triangle — with random position, size and brightness. Returns
    ``(images, labels)`` with images shaped (n, 1, 28, 28) in [-1, 1]
    (same normalization as MNIST) and labels = shape class (0/1/2).
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 1, size, size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        kind = int(rng.integers(0, 3))
        labels[i] = kind
        images[i, 0] = _draw_shape(rng, size, kind)
    images = torch.from_numpy(images) * 2.0 - 1.0                  # [0,1] -> [-1,1]
    return images, torch.from_numpy(labels)


def _draw_shape(rng: np.random.Generator, size: int, kind: int) -> np.ndarray:
    """Rasterize one random shape into a (size, size) float image in [0, 1]."""
    img = np.zeros((size, size), dtype=np.float32)
    intensity = rng.uniform(0.6, 1.0)
    margin = 5
    if kind == 0:                                                  # filled circle
        r = rng.uniform(4.0, size * 0.4)
        cy = rng.uniform(r + 1, size - r - 1)
        cx = rng.uniform(r + 1, size - r - 1)
        yy, xx = np.mgrid[0:size, 0:size]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        img[mask] = intensity
    elif kind == 1:                                                # filled rectangle
        r0, r1 = sorted(rng.uniform(2, size - 2, size=2))
        c0, c1 = sorted(rng.uniform(2, size - 2, size=2))
        r1, c1 = max(r1, r0 + 3), max(c1, c0 + 3)                  # avoid slivers
        img[int(r0):int(r1) + 1, int(c0):int(c1) + 1] = intensity
    else:                                                          # filled triangle
        pts = rng.uniform(margin, size - margin, size=(3, 2))
        yy, xx = np.mgrid[0:size, 0:size]
        mask = _points_in_triangle(xx, yy, pts)
        img[mask] = intensity
    return img


def _points_in_triangle(px: np.ndarray, py: np.ndarray,
                        pts: np.ndarray) -> np.ndarray:
    """Boolean mask of grid points inside the triangle with vertices ``pts``.

    Uses the sign of the three edge cross-products (works for either winding
    order: a point is inside iff it is not on both sides of any edge).
    """
    (x0, y0), (x1, y1), (x2, y2) = pts
    d1 = (px - x1) * (y0 - y1) - (x0 - x1) * (py - y1)
    d2 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
    d3 = (px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


if __name__ == "__main__":
    loader = get_dataloader("synthetic", batch_size=8)
    x, y = next(iter(loader))
    print(f"synthetic batch: {tuple(x.shape)}  range [{x.min():.2f}, {x.max():.2f}]  "
          f"labels {y.tolist()}")
    loader = get_dataloader("mnist", batch_size=8)
    x, y = next(iter(loader))
    print(f"mnist batch:     {tuple(x.shape)}  range [{x.min():.2f}, {x.max():.2f}]  "
          f"labels {y.tolist()}")
