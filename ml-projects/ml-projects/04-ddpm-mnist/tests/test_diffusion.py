"""Unit tests for the DDPM diffusion machinery (offline, CPU-only).

Covers the math that actually has to be right: the closed-form forward
process, schedule invariants, the U-Net contract (output shape == input
shape), gradient flow, both samplers, and the x0-from-eps inversion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import make_synthetic_shapes                      # noqa: E402
from diffusion import DDPM, cosine_beta_schedule            # noqa: E402
from model import UNet                                      # noqa: E402
from train import EMA                                       # noqa: E402


def tiny_unet() -> UNet:
    """Small UNet (16/32/32 channels) that keeps tests fast on CPU."""
    return UNet(in_channels=1, base_channels=16, num_res_blocks=1, time_emb_dim=64)


def tiny_setup(T: int = 20, schedule: str = "linear") -> tuple[DDPM, UNet]:
    torch.manual_seed(0)
    return DDPM(T=T, schedule=schedule), tiny_unet()


# --------------------------------------------------------------- schedules
def test_alphas_cumprod_monotone_and_bounded():
    for schedule in ("linear", "cosine"):
        ddpm = DDPM(T=500, schedule=schedule)
        ab = ddpm.alphas_cumprod
        assert ((ab >= 0.0) & (ab <= 1.0)).all(), f"{schedule}: alphabar out of [0, 1]"
        assert (torch.diff(ab) <= 0).all(), f"{schedule}: alphabar not decreasing"
        assert ab[0] < 1.0 and ab[-1] > 0.0


def test_cosine_schedule_clips_betas():
    betas = cosine_beta_schedule(T=1000)
    assert betas.max() <= 0.999, "cosine betas must be clipped at 0.999"
    assert betas.min() >= 0.0
    ddpm = DDPM(T=1000, schedule="cosine")
    assert torch.allclose(ddpm.betas, betas.float())


def test_linear_schedule_matches_paper():
    ddpm = DDPM(T=1000, schedule="linear")
    expected = torch.linspace(1e-4, 0.02, 1000)
    assert torch.allclose(ddpm.betas, expected, atol=1e-6)


def test_invalid_schedule_and_sampler_raise():
    with pytest.raises(ValueError):
        DDPM(T=10, schedule="quadratic")
    ddpm, model = tiny_setup()
    with pytest.raises(ValueError):
        ddpm.sample(model, n=1, method="eo")


# --------------------------------------------------------- forward process
def test_q_sample_shape_range_and_finite():
    ddpm, _ = tiny_setup(T=500)
    torch.manual_seed(1)
    x0 = torch.rand(4, 1, 28, 28) * 2 - 1                  # images in [-1, 1]
    t = torch.tensor([0, 100, 250, 499])
    noise = torch.randn_like(x0)

    x_t = ddpm.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    assert torch.isfinite(x_t).all()

    # deterministic part only: x_t = sqrt(abar_t) * x0 stays within [-1, 1]
    x_t_clean = ddpm.q_sample(x0, t, torch.zeros_like(x0))
    assert x_t_clean.abs().max() <= 1.0 + 1e-5

    # at the last step the image is (almost) pure noise for a linear schedule
    x_t_last = ddpm.q_sample(x0, torch.full((4,), 499), noise)
    assert torch.allclose(x_t_last, noise, atol=0.2), \
        "x_T should be ~ N(0, I): alphabar_T must be tiny"


def test_predict_x0_from_eps_matches_direct_formula():
    ddpm, _ = tiny_setup(T=100)
    torch.manual_seed(2)
    x_t = torch.randn(8, 1, 28, 28)
    eps = torch.randn(8, 1, 28, 28)
    t = torch.randint(0, 100, (8,))

    x0 = ddpm._predict_x0_from_eps(x_t, t, eps)
    ab = ddpm.alphas_cumprod[t][:, None, None, None]
    expected = (x_t - (1.0 - ab).sqrt() * eps) / ab.sqrt()
    assert torch.allclose(x0, expected, atol=1e-5)


def test_p_losses_scalar_and_positive():
    ddpm, model = tiny_setup()
    torch.manual_seed(3)
    x0 = torch.rand(4, 1, 28, 28) * 2 - 1
    t = torch.randint(0, 20, (4,))
    noise = torch.randn_like(x0)
    loss = ddpm.p_losses(model, x0, t, noise)
    assert loss.ndim == 0 and loss.item() > 0.0 and torch.isfinite(loss)


# ------------------------------------------------------------------- U-Net
def test_unet_output_same_shape_as_input():
    model = tiny_unet()
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0, 19])
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_default_unet_param_count_in_range():
    """The shipped model should sit in the promised 2-8 M parameter budget."""
    n = UNet().num_params()
    assert 2_000_000 < n < 8_000_000, f"unexpected param count: {n}"


def test_backward_pass_produces_finite_grads():
    ddpm, model = tiny_setup()
    torch.manual_seed(4)
    x0 = torch.rand(4, 1, 28, 28) * 2 - 1
    t = torch.randint(0, 20, (4,))
    loss = ddpm.p_losses(model, x0, t, torch.randn_like(x0))
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads), "some parameter received no gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient found"


# ---------------------------------------------------------------- samplers
def test_ddim_sampler_bounded_and_deterministic():
    ddpm, model = tiny_setup(T=20)
    gen = torch.Generator().manual_seed(0)
    imgs = ddpm.sample(model, n=4, method="ddim", ddim_steps=5,
                       eta=0.0, generator=gen)
    assert imgs.shape == (4, 1, 28, 28)
    assert torch.isfinite(imgs).all()
    assert imgs.min() >= -1.0 and imgs.max() <= 1.0

    # eta = 0 is deterministic: same seed -> identical images
    gen = torch.Generator().manual_seed(0)
    again = ddpm.sample(model, n=4, method="ddim", ddim_steps=5,
                        eta=0.0, generator=gen)
    assert torch.allclose(imgs, again)

    # a different initial noise must give a different sample
    gen = torch.Generator().manual_seed(1)
    other = ddpm.sample(model, n=4, method="ddim", ddim_steps=5,
                        eta=0.0, generator=gen)
    assert not torch.allclose(imgs, other)


def test_ddpm_sampler_produces_finite_images():
    ddpm, model = tiny_setup(T=20)
    gen = torch.Generator().manual_seed(0)
    imgs = ddpm.sample(model, n=2, method="ddpm", generator=gen)
    assert imgs.shape == (2, 1, 28, 28)
    assert torch.isfinite(imgs).all()
    assert imgs.abs().max() < 100.0        # sane magnitudes, not exploded


def test_ddim_steps_larger_than_T_is_handled():
    """ddim_steps > T must be handled (steps are clamped and deduped)."""
    ddpm, model = tiny_setup(T=10)
    imgs = ddpm.sample(model, n=1, method="ddim", ddim_steps=50)
    assert imgs.shape == (1, 1, 28, 28) and torch.isfinite(imgs).all()


# -------------------------------------------------------------------- EMA
def test_ema_update_and_weight_swap():
    torch.manual_seed(5)
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    ema.update(model)
    expected = 0.9 * initial["weight"] + 0.1 * torch.ones_like(initial["weight"])
    assert torch.allclose(ema.shadow["weight"], expected, atol=1e-6)

    # canonical eval pattern: store training weights -> copy EMA in -> restore
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)                       # pretend these are "training" weights
    ema.update(model)
    training_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema.store(model)                           # stash training weights
    ema.copy_to(model)                         # model now carries EMA weights
    assert torch.allclose(list(model.parameters())[0], ema.shadow["weight"])
    ema.restore(model)                         # back to the training weights
    for n, p in model.named_parameters():
        assert torch.allclose(p, training_weights[n]), \
            "restore() did not recover training weights"

    # state_dict round-trip into a fresh EMA
    ema2 = EMA(model, decay=0.9)
    ema2.load_state_dict(ema.state_dict())
    for name in ema2.shadow:
        assert torch.allclose(ema2.shadow[name], ema.shadow[name])


# ------------------------------------------------------------------- data
def test_synthetic_dataset_shapes_and_range():
    images, labels = make_synthetic_shapes(64, seed=0)
    assert images.shape == (64, 1, 28, 28)
    assert images.dtype == torch.float32
    assert images.min() >= -1.0 and images.max() <= 1.0
    assert set(labels.unique().tolist()) <= {0, 1, 2}
    assert not torch.allclose(images[0], images[1])     # images are not identical


def test_synthetic_end_to_end_training_step():
    """One real training step on offline synthetic data (the offline path)."""
    images, _ = make_synthetic_shapes(8, seed=1)
    ddpm, model = tiny_setup()
    t = torch.randint(0, 20, (8,))
    loss = ddpm.p_losses(model, images, t, torch.randn_like(images))
    loss.backward()
    assert torch.isfinite(loss)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
