# DDPM — Denoising Diffusion Probabilistic Models on MNIST

An educational, from-scratch implementation of **Denoising Diffusion
Probabilistic Models** (Ho et al., 2020): a U-Net learns to reverse a fixed
noising process one Gaussian step at a time, turning pure noise into
recognizable MNIST digits. Includes the **cosine schedule** (Nichol & Dhariwal,
2021), **EMA weights**, and both the full **ancestral DDPM sampler** and the
fast deterministic **DDIM sampler** (Song et al., 2020).

This is the research-track project of this repository: diffusion models power
modern image generators (Stable Diffusion, Imagen, DALL-E 2), and this code
implements the exact objective they all build on — "predict the noise" — with
every equation written out in the module docstrings.

## What's Implemented

| File | Contents |
|------|----------|
| `model.py` | Compact U-Net (~3.2 M params): sinusoidal time embeddings, GroupNorm+SiLU residual blocks with time-embedding injection, self-attention at 7×7, encoder-decoder with skip connections |
| `diffusion.py` | `DDPM` module: linear/cosine β schedules (float64-accurate buffers), closed-form `q_sample`, simplified MSE loss, `_predict_x0_from_eps`, ancestral DDPM sampler + DDIM sampler (with η stochasticity) |
| `data.py` | MNIST loader normalized to [-1, 1] **plus an offline `synthetic` shapes dataset** so the full pipeline runs with zero downloads |
| `train.py` | Training loop with EMA, per-epoch sample grids, forward-process visualization, checkpointing |
| `sample.py` | Generate grids from a checkpoint (`--sampler ddpm|ddim`, `--ddim-steps`, `--eta`) |
| `plots.py` | Loss curve from `history.json` |
| `tests/` | 16 pytest cases: schedule monotonicity, q_sample shape/range, U-Net shape preservation, gradient flow, DDIM finite output, x0-from-ε identity |

## The Math (the 5 equations that matter)

**Forward process** — a fixed chain that adds noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1−β_t)·x_{t-1}, β_t I)
q(x_t | x_0)     = N(x_t; √ᾱ_t·x_0, (1−ᾱ_t) I)      ᾱ_t = ∏_{s≤t} (1−β_s)
```

The closed form means you can jump to any timestep `t` in **one** line — this
is what makes training tractable (`diffusion.py:q_sample`).

**Training objective** — the variational lower bound (ELBO) decomposes into
per-timestep KL terms; Ho et al. showed that re-weighting and re-parameterizing
with the noise prediction `ε_θ(x_t, t)` reduces the whole thing to:

```
L_simple = E_{x0, ε, t} || ε − ε_θ(√ᾱ_t·x_0 + √(1−ᾱ_t)·ε, t) ||²
```

i.e. *plain MSE between the known input noise and the network's prediction*
(`diffusion.py:p_losses`). No adversary, no instability — just regression.

**Reverse process** — sample iteratively from `x_T ~ N(0, I)` down to `x_0`:

```
μ_θ(x_t, t) = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t))
x_{t−1} = μ_θ(x_t, t) + √β̃_t · z,     β̃_t = β_t (1−ᾱ_{t−1}) / (1−ᾱ_t)
```

**DDIM** skips steps along the trajectory — with as few as 20–50 steps it
produces near-DDPM quality at a fraction of the cost, and `η = 0` makes
generation fully deterministic (same noise → same image).

## Quickstart

```bash
pip install -r requirements.txt          # torch + torchvision

# CPU smoke test (no downloads): synthetic shapes, 1 epoch
python train.py --dataset synthetic --epochs 1 --limit-batches 20

# Full MNIST run (GPU strongly recommended; ~30-60 min on a laptop GPU)
python train.py --dataset mnist --epochs 30 --schedule cosine

# Generate
python sample.py --checkpoint runs/checkpoint.pt --num-images 64 --sampler ddim --ddim-steps 50
python sample.py --sampler ddpm                      # full 1000-step ancestral sampling
python plots.py                                      # loss curve
```

`train.py` saves a sample grid every epoch (`samples_epoch*.png`) so you can
watch structure emerge from noise: blobs → coarse shapes → digits.

## Results

On MNIST (linear schedule, T=1000, EMA on), the classic behavior emerges:

| Epochs | What the samples look like |
|--------|---------------------------|
| 1      | Blurry blobs, faint strokes |
| 5      | Recognizable digit-like shapes |
| 15+    | Clean, diverse handwritten digits |

The training loss converges to roughly **0.03–0.05** (noise-prediction MSE);
note this loss does *not* monotonically improve sample quality — early
timesteps (small t) dominate it, which is a well-known quirk worth reading
about in the Improved DDPM paper.

## Tests

```bash
python -m pytest tests/ -v
```

Highlights: `alphas_cumprod` must be monotonically decreasing in [0, 1]; the
U-Net must preserve input shape; one backward pass must produce finite grads;
`_predict_x0_from_eps` must invert `q_sample` exactly; DDIM with 5 steps must
return finite images in [-1, 1].

## Topics Covered By Courses (for further study)

- Stanford **CS236** (Deep Generative Models) — the ELBO, VAE-vs-diffusion
- MIT **6.S898** / Berkeley **CS294-158** — likelihood-based generative models
- The modern follow-ups: classifier-free guidance, latent diffusion

## References

1. Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
2. Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021
3. Song, Meng & Ermon, *Denoising Diffusion Implicit Models* (DDIM), ICLR 2021
4. Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium
   Thermodynamics*, ICML 2015
5. Luo, *Understanding Diffusion Models: A Unified Perspective* (excellent
   derivation-by-hand tutorial), 2022
