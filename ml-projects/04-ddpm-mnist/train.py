"""Train a DDPM on MNIST (or offline synthetic shapes).

Quick CPU smoke run (~2 min, no network)::

    python train.py --dataset synthetic --epochs 1 --limit-batches 20

Full MNIST run (GPU strongly recommended)::

    python train.py --dataset mnist --epochs 30

After every epoch the script saves an EMA sample grid (`samples_epoch{N}.png`)
so you can watch digits emerge from noise; at the end it also saves a forward
diffusion strip and the training history.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from data import get_dataloader
from diffusion import DDPM
from model import UNet


class EMA:
    """Exponential moving average of model parameters.

    Keeps a shadow copy  shadow <- decay * shadow + (1 - decay) * param  after
    each optimizer step. EMA weights average out SGD noise and give visibly
    smoother samples. Use :meth:`store` / :meth:`copy_to` / :meth:`restore` to
    temporarily evaluate with the EMA weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: torch.nn.Module) -> None:
        """(Re)initialize the shadow parameters from ``model``."""
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Copy the EMA (shadow) weights into ``model``."""
        for name, p in model.named_parameters():
            p.copy_(self.shadow[name])

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        """Stash the current training weights (call before ``copy_to``)."""
        self.backup = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Bring back the training weights stashed by :meth:`store`."""
        for name, p in model.named_parameters():
            p.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: v.clone() for name, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if set(state_dict) != set(self.shadow):
            raise ValueError("EMA state_dict keys do not match the model parameters")
        for name, v in self.shadow.items():
            v.copy_(state_dict[name])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def save_sample_grid(ddpm: DDPM, model: UNet, path: Path, n: int = 32,
                     ddim_steps: int = 50, generator: torch.Generator | None = None) -> None:
    """Sample an n-image grid (nrow=8) with DDIM and save it as a PNG."""
    imgs = ddpm.sample(model, n=n, method="ddim", ddim_steps=ddim_steps,
                       generator=generator)
    grid = (imgs.clamp(-1, 1) + 1.0) / 2.0                       # [-1,1] -> [0,1]
    torchvision.utils.save_image(grid, path, nrow=8)


@torch.no_grad()
def save_forward_process_strip(ddpm: DDPM, x0: torch.Tensor, path: Path,
                               timesteps: tuple[int, ...] = (0, 250, 500, 750, 999)) -> None:
    """Save one image noised at increasing timesteps (q(x_t | x_0))."""
    noise = torch.randn_like(x0)
    ts = tuple(sorted({min(t, ddpm.T - 1) for t in timesteps}))
    fig, axes = plt.subplots(1, len(ts), figsize=(2.0 * len(ts), 2.4),
                             constrained_layout=True)
    for ax, t in zip(np.atleast_1d(axes), ts):
        x_t = ddpm.q_sample(x0, torch.tensor([t], device=x0.device), noise)
        ax.imshow(((x_t[0, 0] + 1.0) / 2.0).cpu().numpy(), cmap="gray",
                  vmin=0.0, vmax=1.0)
        ax.set_title(f"t = {t}")
        ax.axis("off")
    fig.suptitle("Forward diffusion process  q(x_t | x_0)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["mnist", "synthetic"], default="mnist",
                   help="'synthetic' needs no network — use it if MNIST download fails")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000, help="number of diffusion timesteps")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--limit-batches", type=int, default=0,
                   help="0 = all batches; >0 caps batches per epoch (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data / model / diffusion -------------------------------------------
    loader = get_dataloader(args.dataset, args.batch_size,
                            num_workers=args.num_workers, seed=args.seed)
    model = UNet().to(device)
    ema = EMA(model, decay=args.ema_decay)
    ddpm = DDPM(T=args.T, schedule=args.schedule, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"dataset={args.dataset}  device={device}  T={args.T}  schedule={args.schedule}")
    print(f"UNet parameters: {model.num_params():,}")

    # ---- training loop --------------------------------------------------------
    history: dict[str, list] = {"epoch": [], "loss": []}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch_idx, (x0, _) in enumerate(loader):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            x0 = x0.to(device)
            t = torch.randint(0, args.T, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)

            loss = ddpm.p_losses(model, x0, t, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        history["epoch"].append(epoch)
        history["loss"].append(mean_loss)
        print(f"epoch {epoch:3d}/{args.epochs}  loss {mean_loss:.4f}  "
              f"({n_batches} batches, {time.time() - start:.0f}s elapsed)")

        # progress grid with EMA weights (leaves training weights untouched)
        ema.store(model)
        ema.copy_to(model)
        gen = torch.Generator(device=device).manual_seed(args.seed + epoch)
        save_sample_grid(ddpm, model, out_dir / f"samples_epoch{epoch}.png", generator=gen)
        ema.restore(model)

    # ---- save checkpoint + history -------------------------------------------
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "args": vars(args),
        "config": {"in_channels": 1, "img_size": 28, "T": args.T,
                   "schedule": args.schedule, "ema_decay": args.ema_decay},
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump({**history, "args": vars(args)}, f, indent=2)

    # ---- forward diffusion strip (one sample noised at increasing t) ----------
    x0 = next(iter(loader))[0][:1].to(device)                    # a single image
    save_forward_process_strip(ddpm, x0, out_dir / "forward_process.png")

    print(f"done in {time.time() - start:.0f}s  ->  {out_dir}/"
          f"{{checkpoint.pt, history.json, samples_epoch*.png, forward_process.png}}")


if __name__ == "__main__":
    main()
