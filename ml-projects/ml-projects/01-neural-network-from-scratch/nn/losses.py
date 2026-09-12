"""Loss functions.

All losses expose ``forward(pred, target) -> float`` and
``backward() -> dL/dpred``. ``backward`` takes no arguments because it reuses
caches stored during ``forward``.
"""

from __future__ import annotations

import numpy as np


class Loss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        return self.forward(pred, target)


class MSELoss(Loss):
    """Mean squared error, averaged over batch *and* feature dimensions."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * self._diff / self._diff.size


class BCELoss(Loss):
    """Binary cross-entropy on probabilities (expects ``pred`` in (0, 1)).

    Clipped internally for numerical safety.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        p = np.clip(pred, self.eps, 1.0 - self.eps)
        self._p, self._t = p, target
        loss = -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
        return float(loss.mean())

    def backward(self) -> np.ndarray:
        n = self._p.shape[0]
        return (self._p - self._t) / (self._p * (1.0 - self._p)) / n


class SoftmaxCrossEntropy(Loss):
    """Fused ``Softmax + CrossEntropy`` — the standard classification loss.

    Expects raw logits from the final ``Linear`` layer and integer class labels.
    Combining the two operations into one gradient (``softmax - onehot``) is both
    faster and numerically stable, which is why frameworks expose it as a single
    op (e.g. ``torch.nn.CrossEntropyLoss``).
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True) + self.eps)
        self._log_probs = shifted - log_sum_exp          # log-softmax, stable
        self._target = target

        n = logits.shape[0]
        if target.ndim == 1:                              # integer class ids
            nll = -self._log_probs[np.arange(n), target.astype(int)]
        else:                                             # one-hot / soft labels
            nll = -(target * self._log_probs).sum(axis=-1)
        return float(nll.mean())

    def backward(self) -> np.ndarray:
        n = self._log_probs.shape[0]
        if self._target.ndim == 1:
            grad = np.exp(self._log_probs)
            grad[np.arange(n), self._target.astype(int)] -= 1.0
        else:
            grad = np.exp(self._log_probs) - self._target
        return grad / n
