"""Numerical gradient checking utilities.

The single most useful debugging tool when implementing backpropagation by
hand: compare analytic gradients (from ``backward()``) against numerically
estimated gradients via the central-difference formula::

    f'(x) ~= (f(x + h) - f(x - h)) / (2h)

If your analytic gradient is wrong, this catches it immediately. All layers in
this repo are verified against these checks in ``tests/test_nn.py``.
"""

from __future__ import annotations

import numpy as np


def numerical_gradient(fn, param: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar ``fn`` w.r.t. ``param`` (in place)."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + h
        f_plus = fn()
        param[idx] = original - h
        f_minus = fn()
        param[idx] = original

        grad[idx] = (f_plus - f_minus) / (2.0 * h)
        it.iternext()
    return grad


def rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Element-wise relative error, the standard metric in CS231n-style checks."""
    return np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b))
