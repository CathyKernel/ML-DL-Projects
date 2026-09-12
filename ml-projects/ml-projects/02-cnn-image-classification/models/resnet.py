"""ResNet re-implemented from scratch, adapted for small 32x32 images.

Residual learning (He et al., 2015)
-----------------------------------
Instead of asking a stack of layers to fit a desired mapping ``H(x)``
directly, residual learning reformulates the problem: the stack learns the
*residual* ``F(x) = H(x) - x`` and the block output is recomputed as
``H(x) = F(x) + x`` through a shortcut connection.  Two reasons why this makes
very deep networks trainable:

* If identity is the optimal transformation, it is far easier to push
  ``F(x) -> 0`` (weights toward zero) than to fit ``H(x) = x`` through a stack
  of nonlinear layers.
* The identity shortcut gives gradients an unattenuated path backward through
  the whole network, mitigating vanishing gradients.

Differences from the ImageNet ResNet (important for CIFAR)
----------------------------------------------------------
* **Stem**: a single 3x3 conv with stride 1 and *no* max-pool, instead of the
  7x7/stride-2 conv + max-pool.  On 32x32 inputs early aggressive
  down-sampling would destroy most of the spatial information before the
  residual stages ever see it, so feature maps stay 32x32 through stage 1.
* **Head**: global average pooling -> a single ``Linear(width, num_classes)``
  sized for 10/100 classes instead of the 1000-way ImageNet head.

Reference: He, Zhang, Ren & Sun, *Deep Residual Learning for Image
Recognition*, CVPR 2016 (arXiv:1512.03385).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BasicBlock(nn.Module):
    """Two 3x3 convs with an identity shortcut; used by ResNet-18/34."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut: identity when shape matches, 1x1 conv otherwise.
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)  # H(x) = F(x) + x


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 block (width reduced 4x, then expanded back).

    The 3x3 conv -- the only spatially expensive op -- runs on ``planes``
    channels instead of ``planes * expansion``, which is what makes
    ResNet-50+ cheaper and deeper than widening BasicBlocks.
    """

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return torch.relu(out + identity)


class ResNet(nn.Module):
    """ResNet for 32x32 inputs (CIFAR-style stem, stages at 32/16/8/4 px)."""

    def __init__(self, block: type[nn.Module], layers: tuple[int, ...],
                 num_classes: int = 10, width: int = 64) -> None:
        super().__init__()
        self.in_planes = width

        # CIFAR stem: 3x3, stride 1, no max-pool (keeps 32x32).
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.layer1 = self._make_layer(block, width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, width * 8, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 8 * block.expansion, num_classes)

        # Kaiming init for convs, standard init for BN (as in the paper).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block: type[nn.Module], planes: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        """First block may downsample; the rest keep the resolution."""
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        layers += [block(self.in_planes, planes) for _ in range(n_blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))   # 32x32
        x = self.layer1(x)                        # 32x32
        x = self.layer2(x)                        # 16x16
        x = self.layer3(x)                        # 8x8
        x = self.layer4(x)                        # 4x4
        x = self.pool(x).flatten(1)               # (N, width*8*expansion)
        return self.fc(x)

    def num_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def resnet18(num_classes: int = 10) -> ResNet:
    """ResNet-18 (BasicBlocks, [2, 2, 2, 2]) -- ~11.2M params on CIFAR-10."""
    return ResNet(BasicBlock, (2, 2, 2, 2), num_classes)


def resnet34(num_classes: int = 10) -> ResNet:
    """ResNet-34 (BasicBlocks, [3, 4, 6, 3]) -- ~21.3M params on CIFAR-10."""
    return ResNet(BasicBlock, (3, 4, 6, 3), num_classes)


def resnet50(num_classes: int = 10) -> ResNet:
    """ResNet-50 (Bottlenecks, [3, 4, 6, 3]) -- ~23.5M params on CIFAR-10."""
    return ResNet(Bottleneck, (3, 4, 6, 3), num_classes)
