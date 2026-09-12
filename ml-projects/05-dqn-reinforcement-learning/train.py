"""Train a Deep Q-Network (DQN) agent on a Gymnasium classic-control task.

Quickstart (solves CartPole-v1 in ~5-10 min on CPU)::

    python train.py --env CartPole-v1 --episodes 600

Ablations::

    python train.py --no-double      # vanilla DQN target (overestimation bias)
    python train.py --dueling        # dueling architecture (Wang et al. 2016)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent


def play_episode(env: gym.Env, agent: DQNAgent, learning_starts: int,
                 reset_seed: int | None = None) -> tuple[float, int, float | None]:
    """Run one episode; return ``(total_reward, n_steps, mean_loss_or_None)``.

    Each transition goes through the agent loop: ``act`` (epsilon-greedy) ->
    ``env.step`` -> ``buffer.add`` -> ``train_step`` once ``learning_starts``
    environment steps have elapsed.
    """
    obs, _ = env.reset(seed=reset_seed)
    total, steps, losses = 0.0, 0, []
    done = False
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        total += reward
        steps += 1
        if agent.step >= learning_starts:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
    return total, steps, float(np.mean(losses)) if losses else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-starts", type=int, default=1000,
                   help="environment steps of pure exploration before training")
    p.add_argument("--target-update", type=int, default=1000,
                   help="gradient steps between hard target-network syncs")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-steps", type=int, default=50_000,
                   help="linear epsilon decay window, in environment steps")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--no-double", action="store_true",
                   help="disable Double DQN (use the vanilla max target)")
    p.add_argument("--dueling", action="store_true",
                   help="use the dueling architecture (value + advantage streams)")
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env)
    state_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)  # cast: space.n is a numpy int
    threshold = getattr(env.spec, "reward_threshold", None)

    agent = DQNAgent(state_dim, n_actions, device="cpu", lr=args.lr,
                     gamma=args.gamma, buffer_size=args.buffer_size,
                     batch_size=args.batch_size,
                     target_update_freq=args.target_update,
                     use_double=not args.no_double, use_dueling=args.dueling,
                     eps_start=args.eps_start, eps_end=args.eps_end,
                     eps_decay_steps=args.eps_decay_steps,
                     grad_clip=args.grad_clip, seed=args.seed)

    print(f"env={args.env}  state_dim={state_dim}  n_actions={n_actions}")
    print(f"double_dqn={not args.no_double}  dueling={args.dueling}  "
          f"learning_starts={args.learning_starts}  target_update={args.target_update}")
    if threshold is not None:
        print(f"solved when 100-episode moving average >= {threshold:g}")

    rewards_history: list[float] = []
    best_avg = -float("inf")
    t0 = time.time()
    csv_path = out_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "epsilon", "avg100"])
        for episode in range(1, args.episodes + 1):
            reset_seed = args.seed if episode == 1 else None
            ep_reward, ep_steps, ep_loss = play_episode(
                env, agent, args.learning_starts, reset_seed)
            rewards_history.append(ep_reward)
            avg100 = float(np.mean(rewards_history[-100:]))
            eps = agent.epsilon
            writer.writerow([episode, f"{ep_reward:.1f}", ep_steps,
                             f"{eps:.4f}", f"{avg100:.2f}"])

            if avg100 > best_avg:
                best_avg = avg100
                agent.save(out_dir / "best.pt")

            if episode == 1 or episode % 25 == 0 or episode == args.episodes:
                loss_str = f"{ep_loss:.4f}" if ep_loss is not None else "   n/a"
                print(f"episode {episode:4d}/{args.episodes}  reward={ep_reward:7.1f}  "
                      f"steps={ep_steps:4d}  eps={eps:.3f}  avg100={avg100:7.2f}  "
                      f"loss={loss_str}  best_avg={best_avg:7.2f}")

            if threshold is not None and avg100 >= threshold:
                print(f"\nsolved: 100-episode average {avg100:.2f} >= {threshold:g} "
                      f"at episode {episode}")
                break

    env.close()
    agent.save(out_dir / "last.pt")
    with open(out_dir / "args.json", "w") as f:
        json.dump({**vars(args), "episodes_trained": len(rewards_history),
                   "best_avg100": best_avg}, f, indent=2)

    print(f"\ndone in {time.time() - t0:.0f}s  best 100-episode average={best_avg:.2f}")
    print(f"artifacts saved to {out_dir}/  (history.csv, best.pt, last.pt, args.json)")


if __name__ == "__main__":
    main()
