"""Train an image classifier on CIFAR with a modern recipe.

Models: ``simplecnn`` (VGG-style, CPU-friendly) or ResNet-18/34/50 trained
from scratch, with augmentation (+ optional Cutout), SGD+Nesterov or AdamW,
cosine/step LR schedules, label smoothing and optional CPU AMP (bfloat16).

Quick CPU smoke run on synthetic data (no download, ~seconds)::

    python train.py --dataset fake --model simplecnn --epochs 1 \\
        --steps-per-epoch 20 --workers 0

Full CIFAR-10 recipe (GPU recommended; resnet18 -> ~93-94% top-1)::

    python train.py --model resnet18 --dataset cifar10 --epochs 200 \\
        --scheduler cosine --label-smoothing 0.1 --cutout
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from data import N_CLASSES, get_dataloaders
from models import resnet18, resnet34, resnet50, simplecnn
from utils import AverageMeter, CSVLogger, accuracy, save_checkpoint, seed_everything

MODEL_FACTORIES = {
    "simplecnn": simplecnn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def build_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    """SGD+Nesterov or AdamW; weight decay skips BatchNorm gains and biases
    (decay on those only hurts by shrinking activations/offsetting means)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=args.lr, momentum=0.9, nesterov=True)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr)
    raise ValueError(f"unknown optimizer {args.optimizer!r}")


def build_scheduler(args, optimizer) -> torch.optim.LRScheduler | None:
    """Per-epoch LR schedule: cosine (SGDR-style), step decay, or none."""
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    if args.scheduler == "none":
        return None
    raise ValueError(f"unknown scheduler {args.scheduler!r}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    steps: int = 0, amp_dtype: torch.dtype = torch.bfloat16) -> tuple[float, float]:
    """One pass over ``loader`` (or ``steps`` batches if ``steps > 0``).

    Returns ``(mean_loss, top1_accuracy_percent)``.
    """
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for i, (x, y) in enumerate(loader):
        if steps and i >= steps:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device, dtype=amp_dtype, enabled=scaler.is_enabled()):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(accuracy(logits.detach(), y, topk=(1,))[0], bs)
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Validation pass. Returns ``(mean_loss, top1_accuracy_percent)``."""
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_m.update(criterion(logits, y).item(), y.size(0))
        acc_m.update(accuracy(logits, y, topk=(1,))[0], y.size(0))
    return loss_m.avg, acc_m.avg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="resnet18", choices=sorted(MODEL_FACTORIES))
    p.add_argument("--dataset", default="cifar10", choices=sorted(N_CLASSES))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--cutout", action="store_true",
                   help="apply Cutout(16x16) on top of the standard augmentation")
    p.add_argument("--amp", action="store_true",
                   help="bfloat16 autocast + GradScaler (CPU-safe)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="cap optimizer steps per epoch (0 = full epoch); "
                        "use a small value for smoke tests")
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----------------------------------------------------------------
    train_loader, val_loader = get_dataloaders(
        args.dataset, data_dir=args.data_dir, batch_size=args.batch_size,
        augment=True, cutout=args.cutout, num_workers=args.workers, seed=args.seed)
    n_classes = N_CLASSES[args.dataset]
    print(f"dataset={args.dataset}  model={args.model}  device={device}  "
          f"steps/epoch={'all' if not args.steps_per_epoch else args.steps_per_epoch}")

    # ---- model / loss / optimizer ---------------------------------------------
    model = MODEL_FACTORIES[args.model](num_classes=n_classes).to(device)
    print(f"parameters: {model.num_params():,}")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler(device, enabled=args.amp)

    # ---- training loop -----------------------------------------------------------
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    logger = CSVLogger(out_dir / "history.csv", fields)
    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            steps=args.steps_per_epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.log(epoch=epoch, train_loss=f"{train_loss:.4f}",
                   train_acc=f"{train_acc:.2f}", val_loss=f"{val_loss:.4f}",
                   val_acc=f"{val_acc:.2f}", lr=f"{lr_now:.6f}")
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train loss {train_loss:.4f} acc {train_acc:5.2f}%  "
              f"val loss {val_loss:.4f} acc {val_acc:5.2f}%  lr {lr_now:.2e}")

        state = {"model": model.state_dict(), "args": vars(args),
                 "epoch": epoch, "best_acc": best_acc}
        save_checkpoint(state, out_dir / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            state["best_acc"] = best_acc
            save_checkpoint(state, out_dir / "best.pt")

        if scheduler is not None:
            scheduler.step()

    print(f"\nfinished in {time.time() - t0:.1f}s  best val top-1 {best_acc:.2f}%")
    print(f"artifacts saved to {out_dir}/  (best.pt, last.pt, history.csv)")


if __name__ == "__main__":
    main()
