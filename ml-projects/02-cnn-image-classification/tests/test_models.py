"""Offline CPU tests for project 02: model shapes/params, one training step,
Cutout augmentation, top-k accuracy, and the fake dataset.

Run with::

    python -m pytest tests/ -v

Everything runs on synthetic data only -- no CIFAR download, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import Cutout, get_dataloaders                       # noqa: E402
from models import resnet18, resnet34, resnet50, simplecnn     # noqa: E402
from utils import AverageMeter, accuracy                       # noqa: E402

BATCH = (2, 3, 32, 32)
CLASS_FACTORIES = [resnet18, resnet34, resnet50, simplecnn]


# ---------------------------------------------------------------------------
# Models: forward shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", CLASS_FACTORIES, ids=lambda f: f.__name__)
def test_forward_shape_cifar10(factory):
    model = factory(num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH))
    assert out.shape == (2, 10)


def test_forward_shape_cifar100():
    model = resnet34(num_classes=100)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH))
    assert out.shape == (2, 100)


def test_resnet18_param_count():
    # The classic CIFAR ResNet-18 keeps the 64..512 channel plan but swaps the
    # 1000-way ImageNet head for a 10-way one: ~11.17M parameters.
    assert resnet18(num_classes=10).num_params() > 11_000_000


def test_resnet_stays_spatial_through_stage1():
    # CIFAR stem: 3x3 conv, stride 1, NO maxpool -> 32x32 feature maps enter
    # stage 1 (an ImageNet stem would already be down to 8x8 here).
    model = resnet18()
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert not any(isinstance(m, torch.nn.MaxPool2d) for m in model.modules())


def test_bottleneck_widens_by_4():
    model = resnet50()
    out = torch.randn(BATCH)
    assert model.layer1[0].bn3.num_features == 256   # 64 * expansion
    assert model.layer4[-1].bn3.num_features == 2048
    assert model(out).shape == (2, 10)


def test_shortcut_identity_when_shape_matches():
    from models.resnet import BasicBlock
    block = BasicBlock(64, 64, stride=1)
    assert isinstance(block.downsample, torch.nn.Identity)
    x = torch.randn(2, 64, 32, 32)
    assert block(x).shape == x.shape


def test_shortcut_conv_when_shape_changes():
    from models.resnet import BasicBlock
    block = BasicBlock(64, 128, stride=2)
    assert isinstance(block.downsample, torch.nn.Sequential)
    x = torch.randn(2, 64, 32, 32)
    assert block(x).shape == (2, 128, 16, 16)


# ---------------------------------------------------------------------------
# Training step on fake data
# ---------------------------------------------------------------------------
def test_training_step_reduces_loss():
    """A few optimizer steps on a fixed batch must drive the loss down and
    keep every intermediate finite (catches broken gradients/BN in eval)."""
    torch.manual_seed(0)
    train_loader, _ = get_dataloaders("fake", batch_size=64, num_workers=0)
    x, y = next(iter(train_loader))

    model = simplecnn(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    first = last = None
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
        optimizer.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_checkpoint_roundtrip():
    """save_checkpoint(...) -> torch.load(...) restores weights exactly."""
    from utils import save_checkpoint
    model = simplecnn(num_classes=10)
    path = Path(__file__).parent / "_tmp_ckpt.pt"
    save_checkpoint({"model": model.state_dict(), "epoch": 3,
                     "best_acc": 12.5, "args": {"model": "simplecnn"}}, path)
    ckpt = torch.load(path, weights_only=True)
    model2 = simplecnn(num_classes=10)
    model2.load_state_dict(ckpt["model"])
    assert ckpt["epoch"] == 3 and ckpt["best_acc"] == 12.5
    x = torch.randn(BATCH)
    model.eval(), model2.eval()
    assert torch.allclose(model(x), model2(x))
    path.unlink()  # cleanup


# ---------------------------------------------------------------------------
# Cutout augmentation
# ---------------------------------------------------------------------------
def test_cutout_zeros_centered_region():
    img = torch.ones(3, 32, 32)
    cut = Cutout(size=16)
    out = cut.apply(img, center=(16, 16))  # fully interior patch
    assert out[:, 8:24, 8:24].eq(0).all(), "16x16 patch must be zeroed"
    assert out.sum().item() == 3 * 32 * 32 - 3 * 16 * 16
    # Borders untouched.
    assert out[:, :4, :].eq(1).all() and out[:, -4:, :].eq(1).all()


def test_cutout_clips_at_border():
    img = torch.ones(3, 8, 8)
    out = Cutout(size=6).apply(img, center=(0, 0))
    assert out[:, 0:3, 0:3].eq(0).all()          # clipped half-patch
    assert out.sum().item() == 3 * 8 * 8 - 3 * 9


def test_cutout_random_call_returns_clone():
    img = torch.rand(3, 32, 32)
    out = Cutout(size=16)(img)
    assert out is not img, "must not mutate the input in place"
    assert out.shape == img.shape and out.min().item() == 0.0


# ---------------------------------------------------------------------------
# Top-k accuracy: hand-checked example
# ---------------------------------------------------------------------------
def test_accuracy_manual():
    # argmax preds: [2, 0, 1, 0]; targets: [2, 1, 1, 0] -> 3/4 correct.
    output = torch.tensor([[0.1, 0.2, 0.7],
                           [0.5, 0.4, 0.1],
                           [0.3, 0.6, 0.1],
                           [0.9, 0.05, 0.05]])
    target = torch.tensor([2, 1, 1, 0])
    top1, top2 = accuracy(output, target, topk=(1, 2))
    assert top1 == pytest.approx(75.0)
    assert top2 == pytest.approx(100.0)  # every target inside top-2
    # With only 3 classes, top-3 saturates to 100%.
    assert accuracy(output, target, topk=(3,))[0] == pytest.approx(100.0)


def test_accuracy_topk_validation():
    with pytest.raises(ValueError):
        accuracy(torch.randn(2, 3), torch.tensor([0, 1]), topk=(10,))


# ---------------------------------------------------------------------------
# Utils + fake dataset plumbing
# ---------------------------------------------------------------------------
def test_average_meter_batch_weighting():
    meter = AverageMeter()
    meter.update(0.5, n=1)
    meter.update(0.25, n=3)      # weighted by batch size
    assert meter.avg == pytest.approx((0.5 + 0.75) / 4)
    meter.reset()
    assert meter.avg == 0.0


def test_fake_dataloaders_shapes_and_determinism():
    train_loader, val_loader = get_dataloaders("fake", batch_size=64,
                                               num_workers=0, seed=42)
    x, y = next(iter(train_loader))
    assert x.shape == (64, 3, 32, 32) and y.shape == (64,)
    assert y.min() >= 0 and y.max() <= 9

    # Same seed -> identical split and batches (offline reproducibility).
    train2, _ = get_dataloaders("fake", batch_size=64, num_workers=0, seed=42)
    x2, y2 = next(iter(train2))
    assert torch.equal(x, x2) and torch.equal(y, y2)

    # Different seed -> different split.
    train3, _ = get_dataloaders("fake", batch_size=64, num_workers=0, seed=7)
    x3, _ = next(iter(train3))
    assert not torch.equal(x, x3)


def test_get_dataloaders_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_dataloaders("mnist", num_workers=0)
