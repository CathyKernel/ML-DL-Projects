"""Evaluate a trained checkpoint: top-1/top-5 accuracy and confusion matrix.

Loads ``best.pt``/``last.pt`` written by ``train.py``, rebuilds the model and
the validation split from the arguments stored inside the checkpoint (override
with ``--dataset``/``--data-dir``), then prints top-1/top-5 and writes
``confusion_matrix.csv`` (rows = true class, cols = predicted class).
``--heatmap`` additionally renders a row-normalized PNG heatmap.

Example::

    python evaluate.py --checkpoint runs/best.pt --dataset cifar10 --heatmap
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data import N_CLASSES, get_dataloaders
from models import resnet18, resnet34, resnet50, simplecnn
from utils import AverageMeter, accuracy, seed_everything

MODEL_FACTORIES = {
    "simplecnn": simplecnn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


@torch.no_grad()
def confusion_matrix(model, loader, device, n_classes: int) -> np.ndarray:
    """Count predictions over the loader: rows = true, cols = predicted."""
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        idx = y * n_classes + pred
        cm += np.bincount(idx.numpy(), minlength=n_classes * n_classes)\
            .reshape(n_classes, n_classes)
    return cm


def save_confusion_csv(cm: np.ndarray, path: Path) -> None:
    """Write the matrix with a leading row/column of class indices."""
    n = cm.shape[0]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *range(n)])
        writer.writerows([[i, *row] for i, row in enumerate(cm)])


def save_heatmap(cm: np.ndarray, path: Path) -> None:
    """Row-normalized heatmap (each row = recall breakdown of one class)."""
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="recall (row-normalized)")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Confusion matrix")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="best.pt / last.pt from train.py")
    p.add_argument("--dataset", default=None, choices=sorted(N_CLASSES),
                   help="override the dataset recorded in the checkpoint")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: alongside the checkpoint)")
    p.add_argument("--heatmap", action="store_true",
                   help="also save a normalized confusion-matrix PNG")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_args = ckpt["args"]
    dataset = args.dataset or train_args["dataset"]
    n_classes = N_CLASSES[dataset]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MODEL_FACTORIES[train_args["model"]](num_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model"])

    _, val_loader = get_dataloaders(
        dataset, data_dir=args.data_dir, batch_size=args.batch_size,
        augment=False, num_workers=args.workers, seed=args.seed)

    # ---- metrics ------------------------------------------------------------
    model.eval()
    top1, top5, loss_m = AverageMeter(), AverageMeter(), AverageMeter()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            t1, t5 = accuracy(logits, y, topk=(1, 5))
            top1.update(t1, y.size(0))
            top5.update(t5, y.size(0))
            loss_m.update(criterion(logits, y).item(), y.size(0))

    cm = confusion_matrix(model, val_loader, device, n_classes)
    save_confusion_csv(cm, out_dir / "confusion_matrix.csv")
    print(f"checkpoint={args.checkpoint}  epoch={ckpt['epoch']}  dataset={dataset}")
    print(f"val top-1 {top1.avg:.2f}%  top-5 {top5.avg:.2f}%  loss {loss_m.avg:.4f}")
    print(f"confusion matrix -> {out_dir / 'confusion_matrix.csv'}")
    if args.heatmap:
        save_heatmap(cm, out_dir / "confusion_matrix.png")
        print(f"heatmap -> {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
