"""Optimizers and learning-rate schedulers.

Optimizers follow the PyTorch-style contract::

    opt = Adam(model.params, lr=1e-3)
    opt.step()          # apply one update using model-layer .grads
    opt.zero_grad()     # reset gradients (delegates to layers)

State is kept *per (layer, param_name)* so multiple layers never collide.
"""

from __future__ import annotations

import numpy as np

from .layers import Layer


class Optimizer:
    def __init__(self, params: list[tuple[Layer, str]], lr: float) -> None:
        self.params = params          # list of (layer, param_name) pairs
        self.lr = lr

    @classmethod
    def from_model(cls, layers: list[Layer], lr: float) -> "Optimizer":
        params = [(layer, name) for layer in layers for name in layer.params]
        return cls(params, lr=lr)

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for layer, _ in self.params:
            layer.zero_grad()


class SGD(Optimizer):
    """Vanilla stochastic gradient descent: ``p <- p - lr * g``."""

    def step(self) -> None:
        for layer, name in self.params:
            layer.params[name] -= self.lr * layer.grads[name]


class SGDMomentum(Optimizer):
    """SGD with classical momentum: ``v <- mu * v - lr * g; p <- p + v``."""

    def __init__(self, params: list[tuple[Layer, str]], lr: float, momentum: float = 0.9,
                 nesterov: bool = False) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self._velocity: dict[tuple[int, str], np.ndarray] = {}

    def step(self) -> None:
        for i, (layer, name) in enumerate(self.params):
            key = (i, name)
            g = layer.grads[name]
            v = self._velocity.get(key, np.zeros_like(g))
            v = self.momentum * v - self.lr * g
            self._velocity[key] = v
            if self.nesterov:
                layer.params[name] += self.momentum * v - self.lr * g
            else:
                layer.params[name] += v


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015): adaptive moments with bias correction."""

    def __init__(self, params: list[tuple[Layer, str]], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._m: dict[tuple[int, str], np.ndarray] = {}
        self._v: dict[tuple[int, str], np.ndarray] = {}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias1 = 1.0 - b1 ** self.t
        bias2 = 1.0 - b2 ** self.t
        for i, (layer, name) in enumerate(self.params):
            key = (i, name)
            g = layer.grads[name]
            m_prev = self._m.get(key, np.zeros_like(g))
            v_prev = self._v.get(key, np.zeros_like(g))
            m = b1 * m_prev + (1 - b1) * g
            v = b2 * v_prev + (1 - b2) * (g ** 2)
            self._m[key], self._v[key] = m, v
            m_hat = m / bias1
            v_hat = v / bias2
            layer.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Weight decay is applied directly to the weights instead of being folded
    into the adaptive gradient, which fixes Adam's poor generalization on
    L2-regularized problems.
    """

    def __init__(self, params: list[tuple[Layer, str]], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: dict[tuple[int, str], np.ndarray] = {}
        self._v: dict[tuple[int, str], np.ndarray] = {}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias1 = 1.0 - b1 ** self.t
        bias2 = 1.0 - b2 ** self.t
        for i, (layer, name) in enumerate(self.params):
            key = (i, name)
            g = layer.grads[name]
            # Decoupled weight decay (skip biases).
            if self.weight_decay > 0 and name == "W":
                layer.params[name] -= self.lr * self.weight_decay * layer.params[name]
            m_prev = self._m.get(key, np.zeros_like(g))
            v_prev = self._v.get(key, np.zeros_like(g))
            m = b1 * m_prev + (1 - b1) * g
            v = b2 * v_prev + (1 - b2) * (g ** 2)
            self._m[key], self._v[key] = m, v
            m_hat = m / bias1
            v_hat = v / bias2
            layer.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Learning-rate schedulers
# ---------------------------------------------------------------------------
class StepLR:
    """Multiply ``lr`` by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        self.opt = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        if self._epoch % self.step_size == 0:
            self.opt.lr *= self.gamma


class CosineAnnealingLR:
    """Cosine annealing from ``base_lr`` down to ``eta_min`` over ``T_max`` epochs."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        self.opt = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        import math
        self.opt.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self._epoch / self.T_max)
        )
