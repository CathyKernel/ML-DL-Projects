"""Tests for the DQN agent, Q-networks, and replay buffer.

Run with::

    python -m pytest tests/ -v

Everything except the final end-to-end test runs on synthetic data only
(CPU, no gymnasium needed); the whole suite finishes well under a minute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import DQNAgent, ReplayBuffer          # noqa: E402
from network import (DuelingQNetwork, QNetwork,   # noqa: E402
                     make_q_network)

RNG = np.random.default_rng(42)


def fill_buffer(agent: DQNAgent, n: int, seed: int = 0) -> None:
    """Fill the agent's buffer with deterministic synthetic transitions."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        agent.buffer.add(rng.normal(size=agent.state_dim), i % agent.n_actions,
                         float(i % 5), rng.normal(size=agent.state_dim),
                         done=(i == n - 1))


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------
def test_replay_buffer_shapes_and_dtypes():
    buf = ReplayBuffer(state_dim=4, capacity=100, seed=0)
    for _ in range(10):
        buf.add(np.zeros(4, dtype=np.float32), action=1, reward=0.5,
                next_state=np.ones(4, dtype=np.float32), done=False)
    s, a, r, s2, d = buf.sample(8)
    assert s.shape == (8, 4) and s2.shape == (8, 4)
    assert a.shape == (8,) and r.shape == (8,) and d.shape == (8,)
    assert s.dtype == torch.float32 and s2.dtype == torch.float32
    assert a.dtype == torch.int64
    assert r.dtype == torch.float32 and d.dtype == torch.float32
    assert len(buf) == 10
    # values round-trip unchanged
    assert torch.all(a == 1) and torch.all(r == 0.5)
    assert torch.all(s == 0) and torch.all(s2 == 1) and torch.all(d == 0)


def test_replay_buffer_fifo_overwrite_and_dones():
    buf = ReplayBuffer(state_dim=1, capacity=5, seed=0)
    for i in range(8):  # state value == i; done only on transition i == 6
        buf.add([float(i)], action=i % 2, reward=float(i),
                next_state=[float(i) + 0.5], done=(i == 6))
    assert len(buf) == 5                      # capacity respected

    seen: set[int] = set()
    for _ in range(20):
        s, a, r, s2, d = buf.sample(5)
        seen.update(int(v) for v in s[:, 0])
    assert seen == {3, 4, 5, 6, 7}            # oldest 3 transitions overwritten

    s, a, r, s2, d = buf.sample(2000)
    assert torch.all(s[:, 0] == r)            # rewards stored correctly
    assert torch.all(s2[:, 0] == s[:, 0] + 0.5)
    assert d.mean().item() == pytest.approx(0.2, abs=0.05)  # 1 of 5 slots is done
    assert torch.all(s[d == 1][:, 0] == 6.0)  # the done flag lands on i == 6


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def test_qnetwork_shapes_and_factory():
    net = QNetwork(4, 2)
    out = net(torch.randn(7, 4))
    assert out.shape == (7, 2) and torch.isfinite(out).all()

    dnet = DuelingQNetwork(4, 3)
    out = dnet(torch.randn(9, 4))
    assert out.shape == (9, 3) and torch.isfinite(out).all()

    assert isinstance(make_q_network(4, 2, dueling=False), QNetwork)
    assert isinstance(make_q_network(4, 2, dueling=True), DuelingQNetwork)
    # default hidden sizes are (256, 256)
    assert net.net[0].out_features == 256 and net.net[2].out_features == 256


def test_dueling_mean_advantage_invariance():
    """Shifting every advantage output by the same constant must leave Q
    unchanged (that is exactly what ``A - mean(A)`` guarantees), while the
    same shift on the value stream must move Q by that constant."""
    torch.manual_seed(0)
    net = DuelingQNetwork(3, 4)
    x = torch.randn(5, 3)
    q0 = net(x)
    with torch.no_grad():
        net.advantage[-1].bias += 2.5
    assert torch.allclose(net(x), q0, atol=1e-6)

    with torch.no_grad():
        net.value[-1].bias += 1.0
    assert torch.allclose(net(x), q0 + 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Policy (act) and epsilon schedule
# ---------------------------------------------------------------------------
def test_act_valid_and_greedy_deterministic():
    agent = DQNAgent(4, 2, seed=0)
    state = np.zeros(4, dtype=np.float32)
    for _ in range(20):
        assert agent.act(state) in (0, 1)     # valid action index

    greedy = {agent.act(state, greedy=True) for _ in range(10)}
    assert len(greedy) == 1                   # greedy is deterministic

    # an agent with eps_start == eps_end == 0 never explores either
    det = DQNAgent(4, 2, seed=1, eps_start=0.0, eps_end=0.0)
    assert len({det.act(state) for _ in range(10)}) == 1


def test_epsilon_linear_schedule():
    agent = DQNAgent(4, 2, seed=0, eps_start=1.0, eps_end=0.1, eps_decay_steps=100)
    assert agent.epsilon == pytest.approx(1.0)          # eps_start at step 0

    state = np.zeros(4, dtype=np.float32)
    for _ in range(50):                                 # halfway through decay
        agent.act(state)
    assert agent.epsilon == pytest.approx(0.55)

    for _ in range(50):                                 # end of decay window
        agent.act(state)
    assert agent.epsilon == pytest.approx(0.1)

    for _ in range(10):                                 # clamped after the window
        agent.act(state)
    assert agent.epsilon == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Learning: train_step and target-network sync
# ---------------------------------------------------------------------------
def test_train_step_finite_and_online_moves_off_target():
    agent = DQNAgent(4, 2, batch_size=16, buffer_size=64,
                     target_update_freq=1000, seed=0)
    fill_buffer(agent, 32)                    # target starts as a copy of online
    target_before = [p.detach().clone() for p in agent.target.parameters()]

    loss = agent.train_step()
    assert loss is not None and np.isfinite(loss)
    assert any(not torch.allclose(po, pt) for po, pt
               in zip(agent.online.parameters(), target_before))
    for pb, pt in zip(target_before, agent.target.parameters()):
        assert torch.equal(pt, pb)            # target untouched by the update


def test_target_network_explicit_sync():
    agent = DQNAgent(4, 2, batch_size=8, buffer_size=32,
                     target_update_freq=1000, seed=0)
    fill_buffer(agent, 16)
    agent.train_step()
    assert any(not torch.allclose(po, pt) for po, pt
               in zip(agent.online.parameters(), agent.target.parameters()))
    agent.sync_target_network()               # hard sync: θ⁻ ← θ
    for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(po, pt)


def test_target_network_auto_sync_frequency():
    agent = DQNAgent(4, 2, batch_size=8, buffer_size=32,
                     target_update_freq=1, seed=0)   # sync after every step
    fill_buffer(agent, 16)
    agent.train_step()
    for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(po, pt)            # auto-synced without calling sync


def test_save_load_roundtrip(tmp_path):
    agent = DQNAgent(4, 3, seed=0)
    fill_buffer(agent, 16, seed=1)
    agent.train_step()
    path = tmp_path / "agent.pt"
    agent.save(path)

    # checkpoint carries the architecture flags evaluate.py needs
    from agent import load_checkpoint
    ckpt = load_checkpoint(path)
    assert ckpt["state_dim"] == 4 and ckpt["n_actions"] == 3
    assert ckpt["use_double"] is True and ckpt["use_dueling"] is False

    clone = DQNAgent(4, 3, seed=99)           # different init on purpose
    clone.load(path)
    assert clone.step == agent.step
    for p1, p2 in zip(agent.online.parameters(), clone.online.parameters()):
        assert torch.equal(p1, p2)
    state = RNG.normal(size=4)
    assert clone.act(state, greedy=True) == agent.act(state, greedy=True)


# ---------------------------------------------------------------------------
# End-to-end (needs gymnasium, still CPU-only and fast)
# ---------------------------------------------------------------------------
def test_end_to_end_cartpole_agent_loop():
    gym = pytest.importorskip("gymnasium")
    agent = DQNAgent(4, 2, device="cpu", lr=1e-3, buffer_size=5000,
                     batch_size=16, target_update_freq=100,
                     eps_start=1.0, eps_end=0.02, eps_decay_steps=200, seed=0)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    steps, first_loss_checked = 0, False
    while steps < 300:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        steps += 1
        if steps >= 100:                      # learning_starts=100
            loss = agent.train_step()
            if not first_loss_checked:
                assert loss is not None       # buffer is way past batch_size=16
                first_loss_checked = True
        if done:
            obs, _ = env.reset()
    env.close()

    assert len(agent.buffer) == 300           # buffer grew with experience
    assert agent.step == 300                  # global step counter advanced


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
