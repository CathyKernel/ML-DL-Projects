"""Layer primitives implemented with NumPy.

Every layer follows the same contract::

    out = layer.forward(x)            # during training or inference
    dx  = layer.backward(grad_out)    # gradient of loss w.r.t. layer input

Trainable layers (currently ``Linear`` and ``BatchNorm``) store their parameters
and parameter gradients in ``params`` / ``grads`` dictionaries so that the
``Sequential`` container and optimizers can treat all layers uniformly.
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class for all layers."""

    #: Trainable parameters, e.g. {"W": ..., "b": ...}
    params: dict[str, np.ndarray] = {}
    #: Gradients with the same keys/shapes as ``params``
    grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for key in self.grads:
            self.grads[key] = np.zeros_like(self.grads[key])

    def __call__(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        return self.forward(x, train=train)


# ---------------------------------------------------------------------------
# Affine layer
# ---------------------------------------------------------------------------
class Linear(Layer):
    """Fully-connected affine layer: ``y = x @ W + b``.

    Parameters
    ----------
    in_features, out_features : int
    weight_init : str
        ``"he"``  -> He/Kaiming normal (good default for ReLU networks).
        ``"xavier"`` -> Xavier/Glorot uniform (good for tanh/sigmoid networks).
    """

    def __init__(self, in_features: int, out_features: int, weight_init: str = "he",
                 rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        self.in_features = in_features
        self.out_features = out_features

        if weight_init == "he":
            std = np.sqrt(2.0 / in_features)
            W = self.rng.normal(0.0, std, size=(in_features, out_features))
        elif weight_init == "xavier":
            limit = np.sqrt(6.0 / (in_features + out_features))
            W = self.rng.uniform(-limit, limit, size=(in_features, out_features))
        else:
            raise ValueError(f"unknown weight_init: {weight_init!r}")

        self.params = {"W": W, "b": np.zeros(out_features)}
        self.grads = {
            "W": np.zeros_like(self.params["W"]),
            "b": np.zeros_like(self.params["b"]),
        }
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._x is not None, "backward() called before forward()"
        self.grads["W"] += self._x.T @ grad_out
        self.grads["b"] += grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
class ReLU(Layer):
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class LeakyReLU(Layer):
    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._mask = x > 0
        return np.where(self._mask, x, self.negative_slope * x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return np.where(self._mask, grad_out, self.negative_slope * grad_out)


class Tanh(Layer):
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1.0 - self._out ** 2)


class Sigmoid(Layer):
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        # Numerically stable sigmoid: avoid overflow for large |x|.
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        self._out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._out * (1.0 - self._out)


class Softmax(Layer):
    """Row-wise softmax. Use with :class:`nn.losses.MSELoss` if labels are one-hot.

    For classification prefer ``Linear -> SoftmaxCrossEntropy`` (fused, stable).
    """

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        self._out = e / e.sum(axis=-1, keepdims=True)
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Jacobian-vector product for each row:
        # dL/dx_i = p_i * (g_i - sum_j g_j p_j)
        s = (grad_out * self._out).sum(axis=-1, keepdims=True)
        return self._out * (grad_out - s)


# ---------------------------------------------------------------------------
# Regularization layers
# ---------------------------------------------------------------------------
class Dropout(Layer):
    """Inverted dropout: activations are scaled at train time, identity at eval."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = rng or np.random.default_rng()

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if not train or self.p == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        self._mask = (self.rng.random(x.shape) < keep) / keep
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad_out
        return grad_out * self._mask


class BatchNorm(Layer):
    """Batch normalization over the batch dimension for 2-D inputs (N, D).

    Uses running statistics at inference; batch statistics during training.
    """

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5) -> None:
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.params = {
            "gamma": np.ones(num_features),
            "beta": np.zeros(num_features),
        }
        self.grads = {
            "gamma": np.zeros(num_features),
            "beta": np.zeros(num_features),
        }
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean, var = self.running_mean, self.running_var

        self._inv_std = 1.0 / np.sqrt(var + self.eps)
        self._x_hat = (x - mean) * self._inv_std
        return self.params["gamma"] * self._x_hat + self.params["beta"]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        N = grad_out.shape[0]
        self.grads["gamma"] += (grad_out * self._x_hat).sum(axis=0)
        self.grads["beta"] += grad_out.sum(axis=0)

        dxhat = grad_out * self.params["gamma"]
        # Full batch-gradient of the standardization step (see CS231n notes).
        dx = (dxhat - dxhat.mean(axis=0) - self._x_hat * (dxhat * self._x_hat).mean(axis=0))
        return dx * self._inv_std
