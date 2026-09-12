"""DQN agent: replay buffer, epsilon-greedy policy, and TD gradient updates.

Implements the core of Mnih et al. (2015) on top of plain PyTorch:

* **Replay buffer** — a preallocated NumPy ring buffer; random minibatches
  break the correlation between consecutive transitions and let each
  transition contribute to many gradient steps.
* **Target network** — a frozen copy of the online network, hard-synced every
  ``target_update_freq`` gradient steps, so the TD target is held fixed while
  the online network chases it (the "moving target" problem).
* **Double DQN** (on by default) — the online network *selects* the next
  action, the target network *evaluates* it, decoupling selection from
  evaluation to reduce the overestimation bias of the ``max`` operator
  (van Hasselt, 2010; van Hasselt et al., 2016).
* **Dueling architecture** — optional, see ``network.py``.

Example
-------
>>> agent = DQNAgent(state_dim=4, n_actions=2)
>>> action = agent.act(obs)                        # epsilon-greedy
>>> agent.buffer.add(obs, action, reward, next_obs, done)
>>> loss = agent.train_step()                      # one gradient step (or None)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from network import make_q_network


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    """Load a checkpoint written by :meth:`DQNAgent.save` (version-tolerant)."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch builds without the weights_only kwarg
        return torch.load(path, map_location=device)


class ReplayBuffer:
    """Fixed-capacity FIFO ring buffer of transitions.

    Backed by preallocated NumPy arrays (no per-step allocation): states and
    next states are ``float32``, actions ``int64``, rewards and dones
    ``float32`` (a done flag is stored as 1.0). Sampling is uniform with
    replacement, which is the standard DQN behavior.
    """

    def __init__(self, state_dim: int, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0     # next write position (wraps around when full)
        self.size = 0    # number of valid entries, saturates at capacity
        self._rng = np.random.default_rng(seed)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Insert one transition, overwriting the oldest one when full."""
        p = self.pos
        self.states[p] = state
        self.actions[p] = int(action)
        self.rewards[p] = float(reward)
        self.next_states[p] = next_state
        self.dones[p] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor,
                                               torch.Tensor]:
        """Uniform random minibatch as ``(states, actions, rewards,
        next_states, dones)`` torch tensors (CPU; the caller moves them)."""
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx]),        # (B, state_dim) float32
            torch.as_tensor(self.actions[idx]),       # (B,)            int64
            torch.as_tensor(self.rewards[idx]),       # (B,)            float32
            torch.as_tensor(self.next_states[idx]),   # (B, state_dim) float32
            torch.as_tensor(self.dones[idx]),         # (B,)            float32
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """Deep Q-learning agent with replay buffer and target network.

    Epsilon decays linearly from ``eps_start`` to ``eps_end`` over
    ``eps_decay_steps`` global environment steps (``self.step``, incremented
    once per :meth:`act` call). ``train_step`` performs one gradient update
    (Huber loss, gradient clipping) and hard-syncs the target network every
    ``target_update_freq`` gradient steps.

    Example:
        >>> agent = DQNAgent(4, 2, device="cpu", lr=1e-4)
        >>> a = agent.act(np.zeros(4), greedy=True)
        >>> isinstance(a, int)
        True
    """

    def __init__(self, state_dim: int, n_actions: int, device: str = "cpu",
                 lr: float = 1e-4, gamma: float = 0.99,
                 buffer_size: int = 100_000, batch_size: int = 64,
                 target_update_freq: int = 1000, use_double: bool = True,
                 use_dueling: bool = False, eps_start: float = 1.0,
                 eps_end: float = 0.02, eps_decay_steps: int = 50_000,
                 grad_clip: float = 10.0, seed: int = 0) -> None:
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.grad_clip = grad_clip

        self.rng = np.random.default_rng(seed)
        self.buffer = ReplayBuffer(state_dim, buffer_size, seed=seed)

        self.online = make_q_network(state_dim, n_actions,
                                     dueling=use_dueling).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

        self.step = 0          # global environment steps (drives epsilon)
        self._train_steps = 0  # gradient steps (drive target sync)

    # ------------------------------------------------------------------ policy
    @property
    def epsilon(self) -> float:
        """Current exploration rate (linear decay, clamped at ``eps_end``)."""
        frac = min(1.0, self.step / max(1, self.eps_decay_steps))
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Pick an action for one observation (epsilon-greedy unless greedy).

        Also increments the global step counter, which drives the epsilon
        schedule. ``greedy=True`` always exploits the online network — used by
        ``evaluate.py`` and fully deterministic.
        """
        if not greedy and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.n_actions))
        else:
            obs = torch.as_tensor(np.asarray(state, dtype=np.float32),
                                  device=self.device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            with torch.no_grad():
                action = int(self.online(obs).argmax(dim=1).item())
        self.step += 1
        return action

    # ----------------------------------------------------------------- learning
    def sync_target_network(self) -> None:
        """Hard update of the target network: θ⁻ ← θ (full state-dict copy)."""
        self.target.load_state_dict(self.online.state_dict())

    def train_step(self) -> float | None:
        """One gradient step on a random minibatch from the replay buffer.

        The TD target is ``y = r + γ·(1 − done)·Q_target(s', a*)`` where
        ``a* = argmax_a Q_online(s', a)`` for Double DQN (selection online,
        evaluation target) and ``a* = argmax_a Q_target(s', a)`` for vanilla
        DQN. Returns the Huber (SmoothL1) loss, or ``None`` while the buffer
        holds fewer than ``batch_size`` transitions.
        """
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = (
            t.to(self.device) for t in self.buffer.sample(self.batch_size))

        q_sa = self.online(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            if self.use_double:
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions)
            else:
                next_q = self.target(next_states).max(dim=1, keepdim=True).values
            target_q = rewards.unsqueeze(1) \
                + self.gamma * (1.0 - dones.unsqueeze(1)) * next_q

        loss = F.smooth_l1_loss(q_sa, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.sync_target_network()
        return float(loss.item())

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Save online/target weights, optimizer state, step counters, and the
        architecture flags needed to reconstruct the agent."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "train_steps": self._train_steps,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "use_double": self.use_double,
            "use_dueling": self.use_dueling,
        }, path)

    def load(self, path: str | Path) -> None:
        """Restore everything saved by :meth:`save` into this agent."""
        ckpt = load_checkpoint(path, self.device)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step = int(ckpt.get("step", 0))
        self._train_steps = int(ckpt.get("train_steps", 0))


if __name__ == "__main__":
    torch.manual_seed(0)
    agent = DQNAgent(4, 2, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(64):
        agent.buffer.add(rng.normal(size=4), 0, 0.5, rng.normal(size=4), False)
    print("loss:", agent.train_step())
    print("epsilon at step 0:", agent.epsilon)
