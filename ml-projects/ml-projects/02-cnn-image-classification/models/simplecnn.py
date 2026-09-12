"""A compact VGG-style CNN baseline for CIFAR-sized inputs.

Three conv blocks, each ``[3x3 Conv -> BatchNorm -> ReLU] x 2`` followed by
2x2 max-pooling (32 -> 16 -> 8 -> 4), then a two-layer FC head.  This is the
"classic 2015 recipe" the ResNet paper was measured against, and at ~0.8M
parameters it trains on a laptop CPU in minutes -- a useful sanity baseline
before scaling up to ResNet.

Design notes:
* BatchNorm after every conv stabilizes training so plain SGD works.
* Doubling channels when halving spatial resolution keeps per-layer FLOPs
  roughly constant (the VGG/ResNet convention).
* Dropout before the final layer is the cheapest regularizer for the FC head.
"""

from __future__ import annotations

from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """VGG-style baseline: 3 conv blocks + BN + FC head."""

    def __init__(self, num_classes: int = 10, width: int = 32,
                 dropout: float = 0.3) -> None:
        super().__init__()

        def conv_block(c_in: int, c_out: int) -> list[nn.Module]:
            return [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]

        c1, c2, c3 = width, width * 2, width * 4  # 32, 64, 128 by default
        self.features = nn.Sequential(
            *conv_block(3, c1),    # 32x32 -> 16x16
            *conv_block(c1, c2),   # 16x16 -> 8x8
            *conv_block(c2, c3),   # 8x8   -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))

    def num_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def simplecnn(num_classes: int = 10) -> SimpleCNN:
    """Default SimpleCNN (~0.8M params) -- the CPU-friendly baseline."""
    return SimpleCNN(num_classes)
