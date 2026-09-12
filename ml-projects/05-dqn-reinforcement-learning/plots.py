"""Plot DQN training rewards from a history CSV written by train.py.

Produces ``rewards.png``: raw per-episode rewards (low alpha), the 100-episode
moving average, and an optional solved-threshold line.

Example::

    python plots.py --history runs/history.csv --out rewards.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


def moving_average(rewards: np.ndarray, window: int = 100) -> np.ndarray:
    """Trailing mean; entries before the window fills use the available prefix."""
    rewards = np.asarray(rewards, dtype=float)
    cumsum = np.cumsum(np.insert(rewards, 0, 0.0))
    lo = np.maximum(0, np.arange(len(rewards)) - window + 1)
    counts = np.arange(len(rewards)) - lo + 1
    return (cumsum[1:] - cumsum[lo]) / counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--history", default="runs/history.csv")
    p.add_argument("--out", default="rewards.png")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--threshold", type=float, default=475.0,
                   help="solved-threshold line; use <= 0 to disable")
    args = p.parse_args()

    data = np.genfromtxt(args.history, delimiter=",", names=True)
    if data.size == 0:
        raise SystemExit(f"no episodes found in {args.history}")
    episodes = np.atleast_1d(data["episode"]).astype(int)
    rewards = np.atleast_1d(data["reward"]).astype(float)
    ma = moving_average(rewards, args.window)

    title = "DQN training rewards"
    args_json = Path(args.history).parent / "args.json"
    if args_json.exists():
        env_name = json.loads(args_json.read_text()).get("env")
        if env_name:
            title += f" — {env_name}"

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(episodes, rewards, alpha=0.3, color="tab:blue", label="episode reward")
    ax.plot(episodes, ma, color="tab:blue", linewidth=2,
            label=f"{args.window}-episode moving average")
    if args.threshold > 0:
        ax.axhline(args.threshold, color="tab:green", linestyle="--", linewidth=1,
                   label=f"solved threshold ({args.threshold:g})")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"saved {args.out}  ({len(rewards)} episodes)")


if __name__ == "__main__":
    main()
