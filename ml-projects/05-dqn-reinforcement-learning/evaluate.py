"""Evaluate a trained DQN checkpoint with fully greedy actions.

Loads the checkpoint (architecture flags are stored inside it), rolls out
greedy episodes (epsilon = 0), and prints reward statistics.

Examples
--------
Evaluate ``runs/best.pt`` on CartPole-v1::

    python evaluate.py --checkpoint runs/best.pt --env CartPole-v1 --episodes 20

Watch it play (opens a window; needs a display)::

    python evaluate.py --checkpoint runs/best.pt --render
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from agent import DQNAgent, load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", default="runs/best.pt")
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    p.add_argument("--render", action="store_true",
                   help="open a render window (render_mode='human')")
    args = p.parse_args()

    # The checkpoint stores the architecture flags, so evaluation always
    # reconstructs exactly the network that was trained.
    ckpt = load_checkpoint(args.checkpoint, args.device)
    agent = DQNAgent(ckpt["state_dim"], ckpt["n_actions"], device=args.device,
                     use_double=ckpt.get("use_double", True),
                     use_dueling=ckpt.get("use_dueling", False))
    agent.load(args.checkpoint)
    print(f"loaded {args.checkpoint}  (use_double={agent.use_double}  "
          f"use_dueling={agent.use_dueling}  env_steps={agent.step})")

    env = gym.make(args.env, render_mode="human" if args.render else None)
    rewards = np.empty(args.episodes, dtype=float)
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + i)
        total, done = 0.0, False
        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards[i] = total
        print(f"episode {i + 1:2d}/{args.episodes}  reward={total:.1f}")
    env.close()

    print(f"\ngreedy evaluation over {args.episodes} episodes of {args.env}")
    print(f"mean={rewards.mean():.2f}  std={rewards.std():.2f}  "
          f"min={rewards.min():.1f}  max={rewards.max():.1f}")


if __name__ == "__main__":
    main()
