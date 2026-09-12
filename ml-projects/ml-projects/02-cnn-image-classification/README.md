# CIFAR-10 Image Classification: ResNet from Scratch (PyTorch)

A complete, dependency-light training stack for small-image classification:
a from-scratch ResNet (adapted to 32x32 inputs), a compact VGG-style baseline,
CIFAR augmentation with optional Cutout, a modern optimization recipe
(SGD+Nesterov / AdamW, cosine LR, label smoothing, CPU-safe AMP), and
confusion-matrix evaluation. Runs on a laptop CPU in minutes via the
``simplecnn`` model or the offline ``fake`` dataset.

## What's Implemented

| Module | Contents |
|--------|----------|
| `models/resnet.py` | `BasicBlock`, `Bottleneck`, `ResNet` with CIFAR stem + Kaiming init; factories `resnet18/34/50(num_classes)` |
| `models/simplecnn.py` | `SimpleCNN` — 3 VGG-style conv blocks (Conv-BN-ReLU ×2 + maxpool) + FC head (~0.8M params) |
| `data.py` | `cifar10` / `cifar100` / offline `fake` loaders, hardcoded normalization stats, RandomCrop + Flip + `Cutout` |
| `train.py` | CLI training loop: schedulers, label smoothing, AMP, `--steps-per-epoch`, best/last checkpoints, `history.csv` |
| `evaluate.py` | Top-1/top-5 accuracy, `confusion_matrix.csv` (rows=true, cols=pred), optional PNG heatmap |
| `plots.py` | `curves.png` — loss / accuracy / LR panels from `history.csv` |
| `utils.py` | `seed_everything`, `AverageMeter`, top-k `accuracy`, `save_checkpoint`, `CSVLogger` |
| `tests/` | 20 offline pytest cases (shapes, param counts, training step, Cutout, hand-checked accuracy) |

## Residual Learning in One Paragraph

A plain deep stack must fit a target mapping `H(x)` through layer after layer
of nonlinearity — and as depth grows, both optimization (vanishing gradients)
and the *difficulty of even representing identity* get in the way. He et al.
(2015) noticed it is much easier to let each stack of 2–3 conv layers learn
the **residual** `F(x) = H(x) − x` and re-add the input through a shortcut:
`H(x) = F(x) + x`. If a block is useless the network can simply drive
`F(x) → 0` instead of warping conv weights to reproduce identity, and the
shortcut doubles as an unattenuated gradient highway. Every block in
`models/resnet.py` implements exactly this: `relu(F(x) + shortcut(x))`.

## CIFAR Stem vs. ImageNet Stem

The original ImageNet ResNet opens with a **7×7 conv, stride 2 + max-pool**,
reaching 56×56 immediately — sensible for 224×224 images, disastrous for
32×32 CIFAR images (you would discard 7/8 of the spatial information before
the first residual stage). This implementation follows the CIFAR convention:

| | ImageNet ResNet | This repo (CIFAR) |
|---|---|---|
| Stem | 7×7 conv, stride 2, + 3×3 max-pool | 3×3 conv, stride 1, **no max-pool** |
| Feature map entering stage 1 | 56×56 | 32×32 |
| Stage strides | 4, 8, 16, 32 | 1, 2, 2, 2 (32 → 32 → 16 → 8 → 4) |
| Head | Linear(512, 1000) | global avg-pool → Linear(512·expansion, 10/100) |

Consequence: CIFAR ResNet-18 has ~11.17M parameters (vs 11.69M ImageNet) and
spends its capacity on fine 32×32 spatial detail.

## The Training Recipe — and Why Each Piece Helps

| Ingredient | What it does | Why it matters |
|------------|--------------|----------------|
| **RandomCrop(32, padding=4) + HorizontalFlip** | Free extra training data via label-preserving perturbations | The single biggest accuracy lever on CIFAR (~+4–5%); acts as a prior that classification should be invariant to shifts/mirrors |
| **Cutout (16×16 patch zeroed)** | Randomly occludes one region per image (DeVries & Taylor 2017) | Forces the net to use context instead of one distinctive feature; pairs especially well with ResNets (~+1%) |
| **Cosine LR (SGDR)** | Smoothly anneals LR from `--lr` to ~0 (Loshchilov & Hutter 2017) | Large steps early explore; tiny steps late let the net settle into sharp minima — no manual step schedule tuning |
| **Label smoothing (0.1)** | Trains toward a mix of one-hot and uniform (Müller et al. 2019) | Prevents overconfident logits, improves calibration and val accuracy |
| **SGD + Nesterov, wd on matrices only** | Momentum + lookahead; decay skipped for BN gains/biases | The classic CV recipe; decaying BN scale/bias parameters only hurts |
| **AMP (bfloat16)** | `torch.amp.autocast` + GradScaler | ~1.5–2× faster on GPU; on CPU uses bf16 which needs no loss scaling but exercises the same code path |

## Quickstart

```bash
pip install -r requirements.txt

# CPU smoke run — synthetic data, no download, seconds:
python train.py --dataset fake --model simplecnn --epochs 1 --steps-per-epoch 20 --workers 0

# Real training on CPU in minutes (simplecnn, 100 epochs ≈ 20–30 min):
python train.py --model simplecnn --dataset cifar10 --epochs 100 --optimizer sgd --lr 0.05

# Full recipe on GPU (resnet18 → ~93-94% top-1 in ~1–2 h):
python train.py --model resnet18 --epochs 200 --label-smoothing 0.1 --cutout --amp

# CIFAR-100: bump the LR a touch and add cutout
python train.py --model resnet18 --dataset cifar100 --epochs 200 --cutout

# Inspect the result
python plots.py                                            # runs/curves.png
python evaluate.py --checkpoint runs/best.pt --heatmap     # top-1/top-5 + confusion matrix
```

Outputs land in `runs/`: `best.pt` + `last.pt` (weights + args + epoch +
best accuracy), `history.csv`, `curves.png`, and from `evaluate.py`
`confusion_matrix.csv` / `confusion_matrix.png`.

## Results (expected when you run)

Numbers are the standard results this recipe reproduces — **expect them when
you run the commands above** (exact values vary a few tenths with seed/hardware):

| Dataset | Model | Val top-1 | Recipe | Notes |
|---------|-------|-----------|--------|-------|
| cifar10 | simplecnn | **~70%** | 100 epochs, SGD lr 0.05 | minutes on CPU |
| cifar10 | resnet18 | **~93–94%** | 200 epochs, cosine, label smoothing, cutout | GPU |
| cifar10 | resnet18 | ~91–93% | same, no cutout / smoothing | ablation |
| cifar10 | resnet50 | ~95% | 200 epochs, cosine, cutout | GPU |
| cifar100 | resnet18 | ~75–77% | 200 epochs, cosine, cutout | 100 classes |
| fake | any | random-ish | smoke test only | fully offline |

## Tests

```bash
python -m pytest tests/ -v
```

20 offline CPU tests cover: forward shapes for every model, the >11M
parameter count of CIFAR ResNet-18, CIFAR-vs-ImageNet stem structure, a real
training step on `fake` data driving the loss down, checkpoint round-trip,
Cutout zeroing/clipping semantics, `accuracy()` against a hand-checked
top-1/top-2 example, and dataloader determinism. No internet required.

## Repository-Specific Notes

- **Weight decay grouping**: `train.py` splits parameters into 2-D (decayed)
  and 1-D (not decayed) groups — a small but easy-to-miss detail of every
  production CV recipe.
- **`--steps-per-epoch N`** caps optimizer steps per epoch (0 = full epoch).
  It exists so a full run of the pipeline (train → plot → evaluate) fits in
  seconds on any machine.
- **Checkpoints are self-describing**: `evaluate.py` rebuilds the model and
  dataset from the `args` stored inside the checkpoint, so you never have to
  re-state flags at eval time (override with `--dataset` if needed).
- The `fake` dataset uses a seeded `torch.Generator`, so train/val batches are
  bit-for-bit reproducible offline.

## Topics Covered By Courses (for further study)

- Stanford **CS231n** — conv layers, BatchNorm, data augmentation, training
  dynamics, reading loss/accuracy curves
- Stanford **CS230 / CS229n** — transfer-learning-style recipes and
  hyperparameter search
- fast.ai *Practical Deep Learning* — the one-cycle/cosine family of LR
  schedules, label smoothing as regularization
- NYU **DL (LeCun)** — optimization (momentum, Nesterov, AdamW), monitoring
  generalization gaps

## References

1. He, Zhang, Ren & Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016 (arXiv:1512.03385)
2. DeVries & Taylor, *Improved Regularization of CNNs with Cutout*, arXiv 2017 (arXiv:1708.04552)
3. Müller, Kornblith & Hinton, *When Does Label Smoothing Help?*, NeurIPS 2019
4. Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017
5. Ioffe & Szegedy, *Batch Normalization*, ICML 2015
