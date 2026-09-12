"""Generate images from a trained DDPM checkpoint.

Examples::

    python sample.py --num-images 64 --sampler ddim --ddim-steps 50
    python sample.py --sampler ddpm                 # full ancestral loop
    python sample.py --sampler ddim --eta 1.0       # stochastic DDIM
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision

from diffusion import DDPM
from model import UNet
from train import EMA


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", default="runs/checkpoint.pt")
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM stochasticity: 0 = deterministic, 1 ~ DDPM variance")
    p.add_argument("--out", default="samples.png")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt.get("config", {})
    in_channels = int(config.get("in_channels", 1))
    img_size = int(config.get("img_size", 28))
    T = int(config.get("T", 1000))
    schedule = str(config.get("schedule", "linear"))

    model = UNet(in_channels=in_channels).to(args.device)
    model.load_state_dict(ckpt["model"])
    if "ema" in ckpt and ckpt["ema"]:
        ema = EMA(model)                    # EMA weights sample best
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
        print("loaded EMA weights from checkpoint")
    model.eval()

    ddpm = DDPM(T=T, schedule=schedule, device=args.device)
    generator = (torch.Generator(device=args.device).manual_seed(args.seed)
                 if args.seed is not None else None)
    imgs = ddpm.sample(model, n=args.num_images, img_size=img_size, channels=in_channels,
                       method=args.sampler, ddim_steps=args.ddim_steps, eta=args.eta,
                       generator=generator)

    grid = (imgs.clamp(-1, 1) + 1.0) / 2.0                       # [-1,1] -> [0,1]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, out_path, nrow=8)
    print(f"saved {args.num_images} images ({args.sampler}, ddim_steps={args.ddim_steps}, "
          f"eta={args.eta}) -> {out_path}")


if __name__ == "__main__":
    main()
