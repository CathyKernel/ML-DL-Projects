"""DDPM forward/reverse process and samplers (DDPM ancestral + DDIM).

Implements the diffusion math from Ho et al. 2020 and the DDIM sampler from
Song et al. 2020, plus the cosine noise schedule from Nichol & Dhariwal 2021.
All images live in [-1, 1].

Forward process (fixed Markov chain that gradually adds Gaussian noise):

    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I)
    q(x_t | x_0)     = N(x_t; sqrt(alphabar_t) x_0, (1 - alphabar_t) I)

where alpha_t = 1 - beta_t and alphabar_t = prod_{s<=t} alpha_s. The second
closed form means a noisy sample at *any* timestep can be drawn in one line:

    x_t = sqrt(alphabar_t) x_0 + sqrt(1 - alphabar_t) eps,   eps ~ N(0, I)

Reverse process: a network eps_theta(x_t, t) predicts the noise, which gives

    p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 I)

with the simplified training objective  || eps - eps_theta(x_t, t) ||^2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule from Ho et al. 2020 (T=1000: beta from 1e-4 to 0.02)."""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def cosine_beta_schedule(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal 2021, betas clipped at 0.999.

    Defines alphabar_t directly as a smooth cosine, then derives
    beta_t = 1 - alphabar_t / alphabar_{t-1}. The clip avoids numerical blow-up
    near t = T where the raw cosine would approach beta = 1.
    """
    t = torch.arange(T + 1, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(min=0.0, max=max_beta)


class DDPM(nn.Module):
    """Gaussian diffusion with T timesteps and precomputed schedule buffers.

    This module holds no trainable parameters — only the constant schedule
    tensors (registered as non-persistent buffers so checkpoints stay small
    and schedules are always rebuilt from (T, schedule)).

    Args:
        T: total number of diffusion timesteps.
        schedule: "linear" or "cosine" noise schedule.
        device: where the schedule buffers live.
    """

    def __init__(self, T: int = 1000, schedule: str = "linear", device: str | torch.device = "cpu") -> None:
        super().__init__()
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"unknown schedule {schedule!r} (expected 'linear' or 'cosine')")
        self.T = T
        self.schedule = schedule

        betas = linear_beta_schedule(T) if schedule == "linear" else cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)                    # ᾱ_t
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64),
                                         alphas_cumprod[:-1]])           # ᾱ_{t-1}, ᾱ_{-1} := 1

        # Schedules are computed in float64 for numerical accuracy, stored as
        # float32 buffers. Non-persistent: rebuilt from (T, schedule) on load.
        def buf(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor.to(torch.float32), persistent=False)

        buf("betas", betas)
        buf("alphas", alphas)
        buf("alphas_cumprod", alphas_cumprod)
        buf("alphas_cumprod_prev", alphas_cumprod_prev)
        buf("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        buf("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        # posterior variance beta-tilde_t = beta_t (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        buf("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        self.to(device)

    # ------------------------------------------------------------------ forward
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Draw x_t ~ q(x_t | x_0) in closed form.

        Args:
            x0: (B, C, H, W) clean images in [-1, 1].
            t:  (B,) integer timesteps.
            noise: (B, C, H, W) standard Gaussian noise (use the same noise in
                the training loss so the target is known).
        """
        sqrt_ab = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1m_ab = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1m_ab * noise

    def p_losses(self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Simplified DDPM objective: MSE between true and predicted noise.

        The variational lower bound reduces to a weighted sum of per-timestep
        MSEs with weight lambda(t) = beta_t^2 / (2 sigma_t^2 alpha_t (1-alphabar_t));
        Ho et al. 2020 (Thm 4.1 + §3.4) instead set the weight to 1, i.e. this
        plain noise-prediction MSE, and found it trains better in practice.
        """
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model(x_t, t)
        return F.mse_loss(eps_pred, noise)

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                             eps: torch.Tensor) -> torch.Tensor:
        """Invert the closed-form forward process: x0 = (x_t - sqrt(1-ᾱ) eps) / sqrt(ᾱ)."""
        sqrt_ab = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1m_ab = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return (x_t - sqrt_1m_ab * eps) / sqrt_ab

    # ------------------------------------------------------------------ reverse
    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, img_size: int = 28, channels: int = 1,
               method: str = "ddpm", ddim_steps: int = 50, eta: float = 0.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Generate ``n`` images by iteratively denoising pure Gaussian noise.

        Args:
            model: the trained noise-prediction network eps_theta(x_t, t).
            n: number of images to generate.
            img_size / channels: shape of each image.
            method: "ddpm" (ancestral, T network calls) or "ddim"
                (Song et al. 2020; ``ddim_steps`` network calls).
            ddim_steps: number of evenly spaced timesteps for DDIM.
            eta: DDIM stochasticity; 0.0 = deterministic, 1.0 ≈ DDPM variance.
            generator: optional torch.Generator (must live on the same device)
                for reproducible sampling.

        Returns:
            (n, channels, img_size, img_size) tensor in [-1, 1].
        """
        if method not in ("ddpm", "ddim"):
            raise ValueError(f"unknown sampler {method!r} (expected 'ddpm' or 'ddim')")
        device = self.betas.device
        was_training = model.training
        model.eval()
        shape = (n, channels, img_size, img_size)
        x = self._randn(shape, device, generator)
        out = self._sample_ddim(model, x, ddim_steps, eta, generator) if method == "ddim" \
            else self._sample_ddpm(model, x, generator)
        if was_training:
            model.train()
        return out

    def _sample_ddpm(self, model: nn.Module, x: torch.Tensor,
                     generator: torch.Generator | None) -> torch.Tensor:
        """Ancestral sampling (Algorithm 2 of Ho et al. 2020), T model calls.

        Uses the true posterior variance beta-tilde_t (equivalent up to a
        rescaling of sigma; Algorithm 2's beta_t also works — see paper §3.2).
        """
        for t in reversed(range(self.T)):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps = model(x, t_batch)
            mu = (1.0 / self.alphas[t].sqrt()) * (x - self.betas[t] /
                                                  self.sqrt_one_minus_alphas_cumprod[t] * eps)
            if t > 0:
                sigma = self.posterior_variance[t].sqrt()
                x = mu + sigma * self._randn(x.shape, x.device, generator)
            else:
                x = mu                                            # final step: no noise
        return x

    def _sample_ddim(self, model: nn.Module, x: torch.Tensor, ddim_steps: int,
                     eta: float, generator: torch.Generator | None) -> torch.Tensor:
        """DDIM sampling (Song et al. 2020) on ``ddim_steps`` evenly spaced
        timesteps, skipping intermediate ones.

        Update rule for consecutive schedule points (t, t_prev), where the
        point before the first selected timestep is t_prev = -1 (ᾱ = 1):

            x0_pred = (x_t - sqrt(1 - ᾱ_t) eps) / sqrt(ᾱ_t)
            x_{t_prev} = sqrt(ᾱ_{t_prev}) x0_pred
                         + sqrt(1 - ᾱ_{t_prev} - sigma^2) eps + sigma z
            sigma = eta sqrt((1-ᾱ_{t_prev})/(1-ᾱ_t)) sqrt(1 - ᾱ_t/ᾱ_{t_prev})

        eta = 0 makes the process deterministic: the sample is a function of
        the initial noise only, and fewer steps merely coarsen the trajectory.
        """
        times = torch.linspace(0, self.T - 1, min(ddim_steps, self.T)).round().long()
        times = torch.unique(times).flip(0)                       # descending, deduped
        for i, t in enumerate(times):
            t_prev = int(times[i + 1]) if i + 1 < len(times) else -1
            t_batch = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
            eps = model(x, t_batch)
            x0 = self._predict_x0_from_eps(x, t_batch, eps).clamp(-1.0, 1.0)
            ab_t = float(self.alphas_cumprod[t])
            ab_prev = 1.0 if t_prev < 0 else float(self.alphas_cumprod[t_prev])
            sigma = eta * math.sqrt((1.0 - ab_prev) / (1.0 - ab_t)
                                    * (1.0 - ab_t / ab_prev))
            direction = math.sqrt(max(0.0, 1.0 - ab_prev - sigma ** 2))
            x = math.sqrt(ab_prev) * x0 + direction * eps
            if eta > 0.0 and t_prev >= 0:
                x = x + sigma * self._randn(x.shape, x.device, generator)
        return x

    @staticmethod
    def _randn(shape: tuple, device: torch.device,
               generator: torch.Generator | None) -> torch.Tensor:
        """randn that honours a generator when it lives on the right device."""
        if generator is not None and torch.device(generator.device) == torch.device(device):
            return torch.randn(*shape, device=device, generator=generator)
        return torch.randn(*shape, device=device)


if __name__ == "__main__":
    ddpm = DDPM(T=1000, schedule="cosine")
    print(f"T={ddpm.T} schedule={ddpm.schedule}")
    print(f"betas  : [{ddpm.betas[0]:.5f} ... {ddpm.betas[-1]:.5f}] max={ddpm.betas.max():.4f}")
    print(f"ᾱ_0={ddpm.alphas_cumprod[0]:.6f}  ᾱ_T={ddpm.alphas_cumprod[-1]:.2e}")
    x0 = torch.rand(2, 1, 28, 28) * 2 - 1
    t = torch.tensor([0, 999])
    x_t = ddpm.q_sample(x0, t, torch.randn_like(x0))
    print(f"q_sample: {tuple(x_t.shape)}, finite={torch.isfinite(x_t).all().item()}")
