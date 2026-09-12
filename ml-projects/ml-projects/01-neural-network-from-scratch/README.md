# Neural Network from Scratch (NumPy only)

A complete feed-forward neural network library — forward passes, **manual
backpropagation**, optimizers, regularization — implemented in pure NumPy with
**zero deep-learning frameworks**. This is the classic rite-of-passage project
in undergraduate ML courses: if you can debug your own gradients, you
understand backpropagation.

## What's Implemented

| Module | Contents |
|--------|----------|
| `nn/layers.py` | `Linear` (He/Xavier init), `ReLU`, `LeakyReLU`, `Tanh`, `Sigmoid`, `Softmax`, `Dropout` (inverted), `BatchNorm` |
| `nn/losses.py` | `MSELoss`, `BCELoss`, `SoftmaxCrossEntropy` (fused, numerically stable log-softmax) |
| `nn/optim.py` | `SGD`, `SGDMomentum` (+ Nesterov), `Adam`, `AdamW`, `StepLR`, `CosineAnnealingLR` |
| `nn/sequential.py` | `Sequential` container with train/eval modes, batched inference, save/load |
| `data.py` | Two-spirals, moons, circles, linear regression (NumPy-only) + optional MNIST |
| `gradcheck.py` | Numerical gradient checking (central differences) |
| `tests/` | 17 pytest cases: per-layer gradient checks, loss checks, optimizer sanity, end-to-end convergence |

## The Math Behind `backward()`

Every layer implements `forward(x)` and `backward(grad_out)`, where
`backward` returns `∂L/∂x` given `∂L/∂out`, accumulating `∂L/∂params` along
the way. For the `Linear` layer `y = xW + b`:

```
∂L/∂W = xᵀ · ∂L/∂y        ∂L/∂b = Σ_batch ∂L/∂y        ∂L/∂x = ∂L/∂y · Wᵀ
```

For the fused `Softmax + CrossEntropy` loss with logits `z` and one-hot target
`t`, the gradient collapses to the beautifully simple `(softmax(z) - t) / N` —
implementing the two separately (and naively) is slower *and* numerically
fragile. See `nn/losses.py` for the stable log-softmax trick
(`log p = z - logsumexp(z)`).

`BatchNorm`'s input gradient is the trickiest — it must account for the fact
that the batch mean/variance themselves depend on each input:

```
dx = γ · inv_std / N · (N·dy - Σdy - x̂ · Σ(dy·x̂))
```

## Gradient Checking

The core discipline of this project: **every layer is verified against
numerical gradients** via central differences. Run the checks yourself:

```bash
python -m pytest tests/ -v
```

The helper in `gradcheck.py` perturbs each parameter element and compares:

```
relative error = |analytic − numeric| / (|analytic| + |numeric|)
```

This is exactly the methodology used in Stanford CS231n's assignments — and
the single best habit to build when implementing (or *reviewing*) automatic
differentiation code.

## Quickstart

```bash
pip install -r requirements.txt

# Spiral dataset — the "hello world" of non-linear classification
python train.py --dataset spirals --epochs 150
# → val accuracy ~1.00 in about a second on CPU

# Harder: 3-arm spiral with more noise, dropout + batchnorm
python train.py --dataset spirals --noise-defaults ... # (see train.py --help)

# Compare optimizers on moons
python train.py --dataset moons --optimizer sgd       --lr 0.5  --epochs 400
python train.py --dataset moons --optimizer momentum  --lr 0.1  --epochs 400
python train.py --dataset moons --optimizer adam      --lr 0.01 --epochs 400

# MNIST (needs torchvision; ~98% with a plain MLP)
python train.py --dataset mnist --hidden 256 128 --epochs 10 --batchnorm
```

Outputs land in `runs/`: `model.npz` (weights), `history.json` (metrics),
`curves.png` (loss/accuracy/LR), and `decision_boundary.png` for 2-D datasets.

## Results

| Dataset | Model | Val accuracy | Notes |
|---------|-------|--------------|-------|
| spirals (3-class) | MLP 128×128, Adam | **1.000** | 150 epochs, ~1 s CPU |
| moons | MLP 128×128, Adam | **1.000** | 300 epochs |
| circles | MLP 128×128, Adam | **1.000** | 300 epochs |
| MNIST | MLP 256-128 + BatchNorm | **~0.98** | 10 epochs, CPU |

## Repository-Specific Notes

- Optimizers keep state per `(layer, param)` pair — inspect `nn/optim.py` to
  see how Adam's bias-corrected moments work without any framework support.
- `Dropout` uses *inverted* dropout: activations are scaled at **train** time
  so inference is a pure identity (the modern convention).
- `BatchNorm` maintains running statistics for eval mode; the tests verify
  both its batch-mode gradient and the train/eval statistical behavior.

## Topics Covered By Courses (for further study)

- Stanford **CS229** / MIT **6.036** — intro ML: the MLP, loss functions, SGD
- Stanford **CS231n** — backprop derivation, gradient checking, BatchNorm,
  Dropout, He/Xavier initialization
- CMU **10-601/10-701** — optimization variants (momentum, Adam, weight decay)

## References

1. Rumelhart, Hinton & Williams, *Learning representations by back-propagating
   errors*, Nature 323 (1986)
2. Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015
3. Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), ICLR 2019
4. Ioffe & Szegedy, *Batch Normalization*, ICML 2015
5. Srivastava et al., *Dropout*, JMLR 15 (2014)
