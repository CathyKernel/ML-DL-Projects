"""Training utilities: seeding, average meters, top-k accuracy, checkpoints,
and a tiny CSV logger.

Everything here is deliberately dependency-light (torch + stdlib) so the
training loop stays readable.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int) -> None:
    """Seed ``random``, NumPy and torch (incl. all CUDA devices) for
    reproducibility.  Call once at program start."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # no-op on CPU-only builds
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Running mean over mini-batches: ``update(val, n)`` weights ``val`` by
    the batch size ``n`` (the standard ImageNet-training idiom)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)) -> list[float]:
    """Top-k accuracies in percent for a classification batch.

    ``output`` holds raw logits of shape ``(N, C)``, ``target`` the integer
    labels ``(N,)``.  Returns one percentage per k in ``topk``, e.g.
    ``accuracy(logits, y, topk=(1, 5)) -> [top1, top5]``.
    """
    maxk = max(topk)
    if maxk > output.shape[1]:
        raise ValueError(f"topk={topk} larger than {output.shape[1]} classes")
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()                                            # (maxk, N)
    correct = pred.eq(target.view(1, -1).expand_as(pred))      # (maxk, N)
    return [correct[:k].reshape(-1).float().sum(0).item() * 100.0 / target.size(0)
            for k in topk]


def save_checkpoint(state: dict, path: str | Path) -> None:
    """Persist ``state`` (a dict of tensors/primitives) to ``path``,
    creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


class CSVLogger:
    """Append-only CSV writer with a fixed header row.

    >>> log = CSVLogger("runs/history.csv", ["epoch", "loss"])
    >>> log.log(epoch=1, loss=0.5)
    """

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, **row) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
