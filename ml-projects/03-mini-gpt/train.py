"""Train a character-level GPT.

Quick CPU run (a few minutes)::

    python train.py --corpus builtin --max-iters 500

Full tiny-Shakespeare (GPU recommended)::

    python train.py --corpus shakespeare --n-layer 6 --n-head 6 --n-embd 384 \\
        --block-size 256 --batch-size 64 --max-iters 5000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data import CharTokenizer, load_text
from model import GPT, GPTConfig


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str,
              generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random (context, next-token) pairs — the standard LM batch."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device, generator) -> dict[str, float]:
    model.eval()
    out = {}
    for name, split in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split, args.block_size, args.batch_size, device, generator)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--corpus", default="builtin",
                   help="'builtin', 'shakespeare', or a path to a .txt file")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)

    # ---- data -----------------------------------------------------------------
    text = load_text(args.corpus)
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = max(1, int(0.1 * len(ids)))
    train_data, val_data = ids[:-n_val], ids[-n_val:]
    print(f"corpus chars={len(text):,}  vocab={tokenizer.vocab_size}  "
          f"train tokens={len(train_data):,}  device={device}")

    # ---- model / optimizer ------------------------------------------------------
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"parameters: {model.num_params():,} (non-embedding)")

    # GPT recipe: AdamW, weight decay on 2-D matrices only, LR warmup + cosine.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def lr_at(it: int) -> float:
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
        return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ---- training loop -----------------------------------------------------------
    history = {"iter": [], "train_loss": [], "val_loss": [], "lr": []}
    t0 = time.time()
    model.train()
    for it in range(1, args.max_iters + 1):
        lr = lr_at(it)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.block_size, args.batch_size, device, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.eval_interval == 0 or it == 1:
            metrics = estimate_loss(model, train_data, val_data, args, device, generator)
            history["iter"].append(it)
            history["train_loss"].append(metrics["train"])
            history["val_loss"].append(metrics["val"])
            history["lr"].append(lr)
            print(f"iter {it:5d}/{args.max_iters}  train {metrics['train']:.4f}  "
                  f"val {metrics['val']:.4f}  lr {lr:.2e}")

    elapsed = time.time() - t0
    print(f"training finished in {elapsed:.1f}s")

    # ---- save + sample --------------------------------------------------------------
    torch.save({"model": model.state_dict(), "config": config.__dict__},
               out_dir / "checkpoint.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump({**history, "args": vars(args)}, f, indent=2)

    model.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # newline token
    sample = model.generate(prompt, max_new_tokens=300, temperature=0.8, top_k=None)
    (out_dir / "sample.txt").write_text(tokenizer.decode(sample[0].tolist()))
    print(f"checkpoint + sample saved to {out_dir}/")


if __name__ == "__main__":
    main()
