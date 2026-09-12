"""A minimal NumPy-only neural network library with manual backpropagation.

Modules
-------
- ``nn.layers``    : Layer primitives (Linear, activations, Dropout, BatchNorm)
- ``nn.losses``    : Loss functions (MSE, Binary/Softmax cross-entropy)
- ``nn.optim``     : Optimizers (SGD, Momentum, Adam, AdamW) + LR schedulers
- ``nn.sequential``: The ``Sequential`` model container

Everything is implemented with plain NumPy: forward passes, backward passes
(manual calculus), and parameter updates. No autograd anywhere.
"""

from .layers import Linear, ReLU, LeakyReLU, Tanh, Sigmoid, Softmax, Dropout, BatchNorm
from .losses import MSELoss, BCELoss, SoftmaxCrossEntropy
from .optim import SGD, SGDMomentum, Adam, AdamW, StepLR, CosineAnnealingLR
from .sequential import Sequential

__all__ = [
    "Linear", "ReLU", "LeakyReLU", "Tanh", "Sigmoid", "Softmax", "Dropout", "BatchNorm",
    "MSELoss", "BCELoss", "SoftmaxCrossEntropy",
    "SGD", "SGDMomentum", "Adam", "AdamW", "StepLR", "CosineAnnealingLR",
    "Sequential",
]
