"""Q-network architectures for Deep Q-Learning.

Two MLP-based function approximators mapping states to per-action Q values:

* ``QNetwork`` — the standard DQN network (Mnih et al., 2015): a plain MLP
  ``state_dim -> hidden -> hidden -> n_actions``.
* ``DuelingQNetwork`` — the dueling architecture (Wang et al., 2016), which
  splits the estimate into a state-value stream ``V(s)`` and an action
  advantage stream ``A(s, a)``, combined as::

      Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))

  The subtraction of the mean makes the decomposition identifiable: ``V``
  only has to track how good the state is, while ``A`` captures *relative*
  action quality, so ``V`` can be learned without visiting every action.

Use :func:`make_q_network` to build whichever architecture is requested.
"""

from __future__ import annotations

import torch
from torch import nn

HiddenSizes = tuple[int, ...]


def _mlp(in_dim: int, out_dim: int, hidden: HiddenSizes) -> nn.Sequential:
    """ReLU MLP: ``in_dim -> *hidden -> out_dim`` (linear output head)."""
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Standard DQN network: one MLP emitting a Q value per action."""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden: HiddenSizes = (256, 256)) -> None:
        super().__init__()
        self.net = _mlp(state_dim, n_actions, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, state_dim) -> (batch, n_actions)`` raw Q values."""
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling architecture: separate value and advantage streams.

    ``Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')`` — the advantage-centered
    form from Wang et al. (2016), which keeps ``V`` and ``A`` identifiable.
    """

    def __init__(self, state_dim: int, n_actions: int,
                 hidden: HiddenSizes = (256, 256)) -> None:
        super().__init__()
        self.value = _mlp(state_dim, 1, hidden)               # V(s)
        self.advantage = _mlp(state_dim, n_actions, hidden)   # A(s, a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, state_dim) -> (batch, n_actions)`` raw Q values."""
        v = self.value(x)                                     # (B, 1)
        a = self.advantage(x)                                 # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))


def make_q_network(state_dim: int, n_actions: int, dueling: bool = False,
                   hidden: HiddenSizes = (256, 256)) -> nn.Module:
    """Factory returning the requested Q-network architecture.

    Args:
        state_dim: Dimensionality of the observation space.
        n_actions: Number of discrete actions.
        dueling: If ``True``, build a :class:`DuelingQNetwork`, else a plain
            :class:`QNetwork`.
        hidden: Widths of the hidden ReLU layers.

    Example:
        >>> net = make_q_network(4, 2, dueling=True)
        >>> net(torch.zeros(8, 4)).shape
        torch.Size([8, 2])
    """
    if dueling:
        return DuelingQNetwork(state_dim, n_actions, hidden)
    return QNetwork(state_dim, n_actions, hidden)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = make_q_network(4, 2, dueling=True)
    x = torch.randn(8, 4)
    print(net(x).shape)
