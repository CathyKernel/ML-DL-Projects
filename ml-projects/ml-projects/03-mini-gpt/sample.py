"""Sample text from a trained mini-GPT checkpoint.

Examples::

    python sample.py --prompt "First Citizen:" --max-new-tokens 400 --top-k 20
    python sample.py --temperature 1.2           # more random
    python sample.py --temperature 0.5           # more deterministic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import CharTokenizer, load_text
from model import GPT, GPTConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", default="runs/checkpoint.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None, help="e.g. 20; omit for none")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = CharTokenizer(load_text("builtin"))  # vocab is char-level & stable
    start_ids = tokenizer.encode(args.prompt) or [0]
    idx = torch.tensor([start_ids], dtype=torch.long, device=args.device)

    out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
