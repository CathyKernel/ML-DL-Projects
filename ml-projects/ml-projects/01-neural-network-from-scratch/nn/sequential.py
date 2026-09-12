"""The ``Sequential`` model container: stacks layers, runs the full forward and
backward passes, and exposes an iterable of trainable layers for optimizers."""

from __future__ import annotations

import numpy as np

from .layers import Layer


class Sequential:
    """A linear chain of layers, analogous to ``torch.nn.Sequential``."""

    def __init__(self, *layers: Layer) -> None:
        self.layers: list[Layer] = list(layers)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def __call__(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        return self.forward(x, train=train)

    # -- optimizer plumbing -------------------------------------------------
    def trainable_layers(self) -> list[Layer]:
        return [l for l in self.layers if l.params]

    def zero_grad(self) -> None:
        for layer in self.trainable_layers():
            layer.zero_grad()

    def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Inference-mode forward pass in mini-batches (keeps memory bounded)."""
        outputs = [
            self.forward(x[i:i + batch_size], train=False)
            for i in range(0, len(x), batch_size)
        ]
        return np.concatenate(outputs, axis=0)

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for name, value in layer.params.items():
                out[f"layer{i}.{name}"] = value
        return out

    def save(self, path: str) -> None:
        np.savez(path, **self.state_dict())

    def load(self, path: str) -> None:
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            for name in layer.params:
                key = f"layer{i}.{name}"
                layer.params[name] = data[key]
