"""A compact U-Net noise predictor for diffusion models, implemented from scratch.

The network maps a noisy image ``x_t`` and its diffusion timestep ``t`` to the
noise ``epsilon`` that was added to ``x_0``, i.e. it learns
``epsilon_theta(x_t, t)`` — the denoiser at the heart of DDPM
(Ho et al. 2020).

Architecture (sized for 28×28 images, ~3.1 M parameters):

    time t ──► sinusoidal embedding ──► Linear+SiLU+Linear  (time_emb, 256-d)
    x_t   ──► Conv3x3
       ──► [down level 0 @ 28×28, 64 ch:  ResBlock] ──► skip
       ──► downsample 28→14
       ──► [down level 1 @ 14×14, 128 ch: ResBlock] ──► skip
       ──► downsample 14→7
       ──► [down level 2 @ 7×7,  128 ch:  ResBlock] ──► skip
       ──► mid: ResBlock → SelfAttention(7×7) → ResBlock
       ──► [up level 2: ResBlock (skip cat)] ──► upsample 7→14
       ──► [up level 1: ResBlock (skip cat)] ──► upsample 14→28
       ──► [up level 0: ResBlock (skip cat)]
       ──► GroupNorm → SiLU → Conv3x3  (same shape as input)

Each ResBlock is conditioned on the timestep by adding a linear projection of
the time embedding after its first convolution.

Reference: Ho et al. 2020 "Denoising Diffusion Probabilistic Models";
Nichol & Dhariwal 2021 "Improved Denoising Diffusion Probabilistic Models".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding of integer timesteps (Transformer-style).

    Maps each timestep to a ``dim``-dimensional vector using sin/cos of
    geometrically spaced frequencies, so nearby timesteps get nearby vectors
    without any learned parameters.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "time embedding dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t``: (B,) integer timesteps -> (B, dim) float embedding."""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None, :]          # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Time-conditioned residual block.

    GroupNorm(8) -> SiLU -> Conv3x3 -> (+ time-embedding projection)
    -> GroupNorm -> SiLU -> Conv3x3, with a 1×1 conv skip connection when the
    channel count changes.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        assert in_channels % num_groups == 0 and out_channels % num_groups == 0, \
            "channel counts must be divisible by the GroupNorm group count"
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb)[:, :, None, None]   # inject timestep
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over the spatial positions of a feature map.

    Used at the 7×7 bottleneck, where every position can attend to every
    other position (49 tokens of 128 channels — cheap and effective).
    Pre-normalized with GroupNorm and wrapped in a residual connection.
    """

    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)   # (B, HW, C)
        q, k, v = self.qkv(h).chunk(3, dim=-1)               # single head
        scale = 1.0 / math.sqrt(C)
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        h = self.proj(attn @ v).transpose(1, 2).reshape(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    """Strided 3×3 conv that halves the spatial resolution (28→14→7)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor 2× upsampling followed by a 3×3 conv (7→14→28)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class UNet(nn.Module):
    """Compact U-Net predicting the noise ``eps`` in a noisy image.

    Args:
        in_channels: number of input image channels (1 for grayscale MNIST).
        base_channels: channels at the highest resolution (levels multiply by 2).
        num_res_blocks: residual blocks per resolution level (down and up).
        time_emb_dim: width of the sinusoidal timestep embedding.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 num_res_blocks: int = 1, time_emb_dim: int = 256) -> None:
        super().__init__()
        c0, c1, c2 = base_channels, base_channels * 2, base_channels * 2

        # ---- timestep conditioning ------------------------------------------
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        def res(cin: int, cout: int) -> ResidualBlock:
            return ResidualBlock(cin, cout, time_emb_dim)

        # ---- encoder ----------------------------------------------------------
        self.conv_in = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.down0 = nn.ModuleList([res(c0, c0) for _ in range(num_res_blocks)])
        self.downsample0 = Downsample(c0)
        self.down1 = nn.ModuleList([res(c0, c1), *[res(c1, c1) for _ in range(num_res_blocks - 1)]])
        self.downsample1 = Downsample(c1)
        self.down2 = nn.ModuleList([res(c1, c2) for _ in range(num_res_blocks)])

        # ---- bottleneck -------------------------------------------------------
        self.mid1 = res(c2, c2)
        self.mid_attn = SelfAttention(c2)
        self.mid2 = res(c2, c2)

        # ---- decoder (skip channels are concatenated onto the upsampled map) ---
        self.up2 = nn.ModuleList([res(c2 + c2, c2), *[res(c2, c2) for _ in range(num_res_blocks - 1)]])
        self.upsample1 = Upsample(c2)
        self.up1 = nn.ModuleList([res(c2 + c1, c2), *[res(c2, c2) for _ in range(num_res_blocks - 1)]])
        self.upsample0 = Upsample(c2)
        self.up0 = nn.ModuleList([res(c2 + c0, c0), *[res(c0, c0) for _ in range(num_res_blocks - 1)]])

        # ---- output head --------------------------------------------------------
        self.final_norm = nn.GroupNorm(8, c0)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(c0, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise added to ``x``.

        Args:
            x: (B, in_channels, H, W) noisy images in [-1, 1].
            t: (B,) integer diffusion timesteps in [0, T-1].
        Returns:
            (B, in_channels, H, W) noise prediction — same shape as ``x``.
        """
        time_emb = self.time_embed(t)

        h = self.conv_in(x)
        skips = []
        for block in self.down0:                     # 28×28, 64 ch
            h = block(h, time_emb)
        skips.append(h)
        h = self.downsample0(h)                      # 14×14
        for block in self.down1:
            h = block(h, time_emb)
        skips.append(h)
        h = self.downsample1(h)                      # 7×7
        for block in self.down2:
            h = block(h, time_emb)
        skips.append(h)

        h = self.mid1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, time_emb)

        # Only the first block of each level consumes the level's skip; the
        # remaining blocks refine without additional concatenation.
        for i, block in enumerate(self.up2):         # 7×7, cat skip2
            h = block(torch.cat([h, skips.pop()], dim=1), time_emb) if i == 0 else block(h, time_emb)
        h = self.upsample1(h)                        # 14×14
        for i, block in enumerate(self.up1):         # 14×14, cat skip1
            h = block(torch.cat([h, skips.pop()], dim=1), time_emb) if i == 0 else block(h, time_emb)
        h = self.upsample0(h)                        # 28×28
        for i, block in enumerate(self.up0):         # 28×28, cat skip0
            h = block(torch.cat([h, skips.pop()], dim=1), time_emb) if i == 0 else block(h, time_emb)

        return self.final_conv(self.final_act(self.final_norm(h)))

    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = UNet()
    print(f"UNet parameters: {model.num_params():,}")
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0, 999])
    with torch.no_grad():
        out = model(x, t)
    print(f"input  {tuple(x.shape)} -> output {tuple(out.shape)}")
