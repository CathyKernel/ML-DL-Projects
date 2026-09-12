"""Unit + gradient tests for the NumPy neural network library.

Run with::

    python -m pytest tests/ -v

The gradient checks compare ``backward()`` against central-difference numerical
gradients; anything loosely wrong fails loudly here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradcheck import numerical_gradient, rel_error            # noqa: E402
from nn import (Adam, BatchNorm, BCELoss, Dropout, Linear, MSELoss,  # noqa: E402
                ReLU, Sigmoid, SGD, SGDMomentum, Softmax,
                SoftmaxCrossEntropy, Tanh)

RNG = np.random.default_rng(42)


def assert_grads_close(analytic: np.ndarray, numeric: np.ndarray,
                       rtol: float = 1e-4, atol: float = 1e-6) -> None:
    """Mixed absolute/relative comparison: pure relative error blows up when
    both gradients are near zero, so standard allclose semantics are used."""
    assert analytic.shape == numeric.shape
    if not np.allclose(analytic, numeric, rtol=rtol, atol=atol):
        err = rel_error(analytic, numeric).max()
        raise AssertionError(f"gradient mismatch (max rel err {err:.2e})\n"
                             f"analytic: {analytic.ravel()[:8]}\n"
                             f"numeric:  {numeric.ravel()[:8]}")


def check_layer_grads(layer, x: np.ndarray, train: bool = False) -> None:
    """Verify that dL/dx (and every param grad) matches numerical gradients."""

    def loss_fn() -> float:
        return float((layer.forward(x, train=train) ** 2).sum())

    # Analytic grads: d(sum(out^2))/d* = 2 * out * d(out)/d*
    out = layer.forward(x, train=train)
    analytic_dx = layer.backward(2.0 * out)

    numeric_dx = numerical_gradient(loss_fn, x)
    assert_grads_close(analytic_dx, numeric_dx)

    for name in layer.params:
        analytic = layer.grads[name].copy()
        numeric = numerical_gradient(loss_fn, layer.params[name])
        assert_grads_close(analytic, numeric)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def test_linear_forward_shape():
    layer = Linear(4, 3, rng=RNG)
    out = layer(RNG.normal(size=(7, 4)), train=False)
    assert out.shape == (7, 3)


def test_linear_grads():
    layer = Linear(5, 3, rng=RNG)
    check_layer_grads(layer, RNG.normal(size=(6, 5)))


def test_relu_grads():
    check_layer_grads(ReLU(), RNG.normal(size=(6, 5)) + 0.1)


def test_tanh_grads():
    check_layer_grads(Tanh(), RNG.normal(size=(6, 5)))


def test_sigmoid_grads():
    check_layer_grads(Sigmoid(), RNG.normal(size=(6, 5)) * 2)


def test_softmax_grads():
    check_layer_grads(Softmax(), RNG.normal(size=(6, 5)))


def test_batchnorm_grads():
    # Batch statistics are a function of the input, so the check must run in
    # train mode; in eval mode the layer uses frozen running stats instead.
    layer = BatchNorm(5)
    check_layer_grads(layer, RNG.normal(size=(8, 5)) * 2 + 1, train=True)


def test_batchnorm_eval_uses_running_stats():
    layer = BatchNorm(4)
    x = RNG.normal(size=(64, 4)) * 3 + 5
    for _ in range(200):                       # running stats converge to batch stats
        layer.forward(x, train=True)
    assert np.allclose(layer.running_mean, x.mean(0), atol=1e-2)
    out = layer.forward(x[:10], train=False)   # eval mode: no batch coupling
    assert out.shape == (10, 4)


def test_dropout_train_eval():
    layer = Dropout(p=0.5, rng=RNG)
    x = np.ones((1000, 10))
    out = layer.forward(x, train=True)
    assert 0.9 < out.mean() < 1.1              # inverted dropout preserves scale
    grad = np.ones_like(x)
    assert np.allclose(layer.backward(grad), layer._mask)  # backward uses same mask
    layer.forward(x, train=False)              # eval mode is the identity
    assert np.allclose(layer.forward(x, train=False), x)
    assert np.allclose(layer.backward(grad), grad)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_softmax_xentropy_grad():
    logits = RNG.normal(size=(6, 4))
    target = RNG.integers(0, 4, 6)
    loss = SoftmaxCrossEntropy()
    loss.forward(logits, target)
    analytic = loss.backward()
    numeric = numerical_gradient(lambda: loss.forward(logits, target), logits)
    assert_grads_close(analytic, numeric)


def test_softmax_xentropy_matches_reference():
    """Fused loss must equal log-softmax NLL computed the naive way."""
    logits = RNG.normal(size=(8, 5)) * 3
    target = RNG.integers(0, 5, 8)
    ref = -np.log(np.exp(logits - logits.max(1, keepdims=True)).T
                  / np.exp(logits - logits.max(1, keepdims=True)).sum(1)
                  )[target, np.arange(8)].mean()
    assert abs(SoftmaxCrossEntropy().forward(logits, target) - ref) < 1e-6


def test_mse_grad():
    pred = RNG.normal(size=(6, 3))
    target = RNG.normal(size=(6, 3))
    loss = MSELoss()
    loss.forward(pred, target)
    numeric = numerical_gradient(lambda: loss.forward(pred, target), pred)
    assert_grads_close(loss.backward(), numeric)


def test_bce_grad():
    pred = RNG.uniform(0.05, 0.95, size=(6, 1))
    target = RNG.integers(0, 2, (6, 1)).astype(float)
    loss = BCELoss()
    loss.forward(pred, target)
    numeric = numerical_gradient(lambda: loss.forward(pred, target), pred)
    assert_grads_close(loss.backward(), numeric)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
def _quadratic_minimizer(opt_cls, **kwargs) -> float:
    """Optimizer should minimize ||w||^2 + c from a random start."""
    layer = Linear(1, 1, rng=RNG)
    layer.params["W"][...] = 3.0
    layer.grads["W"] = np.array([6.0])       # d/dW (W^2) = 2W = 6
    opt = opt_cls([(layer, "W")], lr=0.1, **kwargs)
    for _ in range(200):
        layer.grads["W"] = 2.0 * layer.params["W"]   # analytic grad of W^2
        opt.step()
    return float(np.abs(layer.params["W"]).max())


def test_sgd_minimizes_quadratic():
    assert _quadratic_minimizer(SGD) < 1e-2


def test_sgd_momentum_minimizes_quadratic():
    assert _quadratic_minimizer(SGDMomentum, momentum=0.9) < 1e-2


def test_adam_minimizes_quadratic():
    assert _quadratic_minimizer(Adam) < 1e-2


# ---------------------------------------------------------------------------
# End-to-end convergence (fast)
# ---------------------------------------------------------------------------
def test_mlp_learns_moons():
    from data import make_moons
    from nn import Linear, ReLU, Sequential, SoftmaxCrossEntropy

    X, y = make_moons(300, rng=np.random.default_rng(0))
    model = Sequential(Linear(2, 32), ReLU(), Linear(32, 2))
    loss_fn = SoftmaxCrossEntropy()
    params = [(l, n) for l in model.trainable_layers() for n in l.params]
    opt = Adam(params, lr=0.05)

    first_loss = None
    for _ in range(300):
        logits = model(X, train=True)
        loss = loss_fn(logits, y)
        if first_loss is None:
            first_loss = loss
        model.zero_grad()
        model.backward(loss_fn.backward())
        opt.step()

    acc = (model.predict(X).argmax(1) == y).mean()
    assert acc > 0.95, f"expected >95% train accuracy, got {acc:.3f}"
    assert loss < first_loss * 0.2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
