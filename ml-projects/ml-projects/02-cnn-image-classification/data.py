"""Datasets, normalization and augmentation for CIFAR-style classification.

Supported datasets:

* ``cifar10`` / ``cifar100`` -- downloaded once into ``data_dir`` by
  torchvision (internet required the first time only).
* ``fake`` -- small synthetic 32x32x3 tensors generated locally from a seeded
  :class:`torch.Generator`; works fully offline and is used by the tests and
  smoke runs.

``get_dataloaders`` always returns ``(train_loader, val_loader)``: CIFAR's
50k training images are split into 45k train / 5k val with a fixed permutation
(the official 10k test set is left untouched for final evaluation).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# Per-channel statistics of the training split (hardcoded, standard values).
CIFAR_STATS: dict[str, dict[str, tuple[float, ...]]] = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
    },
}

N_CLASSES: dict[str, int] = {"cifar10": 10, "cifar100": 100, "fake": 10}

# Split of the CIFAR train set used for model selection (the 10k test set is
# never touched during training).
VAL_SIZE = 5_000


class Cutout:
    """RandomCutout augmentation (DeVries & Taylor, 2017).

    Zeros a single ``size x size`` square patch per image.  The patch centre is
    sampled uniformly and the patch is *clipped* at the borders (as in the
    original paper); the same patch is removed from all channels.  Applied to
    a float CHW tensor (i.e. after ``ToTensor``).
    """

    def __init__(self, size: int = 16) -> None:
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.shape[-2], img.shape[-1]
        cy = int(torch.randint(0, h, (1,)).item())
        cx = int(torch.randint(0, w, (1,)).item())
        return self.apply(img, (cy, cx))

    def apply(self, img: Tensor, center: tuple[int, int]) -> Tensor:
        """Zero the patch centred at ``center=(y, x)`` (deterministic)."""
        c, h, w = img.shape
        half = self.size // 2
        y1, y2 = max(0, center[0] - half), min(h, center[0] - half + self.size)
        x1, x2 = max(0, center[1] - half), min(w, center[1] - half + self.size)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(size={self.size})"


def build_transforms(name: str, augment: bool, cutout: bool) -> tuple:
    """Return ``(train_transform, val_transform)`` for a dataset."""
    stats = CIFAR_STATS[name]
    normalize = transforms.Normalize(stats["mean"], stats["std"])

    if augment:
        ops: list[nn.Module] = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if cutout:
            ops.append(Cutout(size=16))
        train_t = transforms.Compose([*ops, transforms.ToTensor(), normalize])
    else:
        train_t = transforms.Compose([transforms.ToTensor(), normalize])
    val_t = transforms.Compose([transforms.ToTensor(), normalize])
    return train_t, val_t


def _fake_dataset(n: int, seed: int) -> TensorDataset:
    """Synthetic (image, label) pairs -- deterministic, fully offline."""
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 32, 32, generator=g)
    labels = torch.randint(0, N_CLASSES["fake"], (n,), generator=g)
    return TensorDataset(images, labels)


def get_dataloaders(name: str, data_dir: str = "data", batch_size: int = 128,
                    augment: bool = True, cutout: bool = False,
                    num_workers: int = 2, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """Build ``(train_loader, val_loader)`` for ``cifar10|cifar100|fake``."""
    g = torch.Generator().manual_seed(seed)

    if name == "fake":
        train_ds = _fake_dataset(512, seed)
        val_ds = _fake_dataset(128, seed + 1)
    elif name in ("cifar10", "cifar100"):
        cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        train_t, val_t = build_transforms(name, augment, cutout)
        base_train = cls(data_dir, train=True, download=True, transform=train_t)
        base_val = cls(data_dir, train=True, download=True, transform=val_t)

        perm = torch.randperm(50_000, generator=g)
        val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
        train_ds = Subset(base_train, train_idx.tolist())
        val_ds = Subset(base_val, val_idx.tolist())
    else:
        raise ValueError(f"unknown dataset {name!r}; "
                         f"expected one of {sorted(N_CLASSES)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader
