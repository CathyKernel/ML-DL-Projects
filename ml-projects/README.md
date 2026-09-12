# Machine Learning / Deep Learning Projects

A collection of **five ML/DL projects** covering the core
topics taught in CS programs — from undergraduate machine learning
fundamentals up to PhD-level generative modeling.

> **Note**: All code is written from scratch as original educational implementations of
> classic algorithms and papers. Projects are *inspired by* the topics covered in courses
> such as Stanford CS229/CS231n/CS224n/CS236, Berkeley CS188/CS285, and MIT 6.036/6.S898 —
> they are **not** official course solutions and contain no course starter code.

## Projects

| # | Project | Level | Framework | Key Topics |
|---|---------|-------|-----------|------------|
| 1 | [neural-network-from-scratch](./01-neural-network-from-scratch) | Undergraduate | NumPy only | Backpropagation, autodiff-free layers, gradient checking, optimizers (SGD/Momentum/Adam/AdamW), BatchNorm, Dropout |
| 2 | [cnn-image-classification](./02-cnn-image-classification) | Undergraduate / Master | PyTorch | ResNet, data augmentation (Cutout), cosine LR schedule, label smoothing, mixed precision |
| 3 | [mini-gpt](./03-mini-gpt) | Master | PyTorch | Transformer decoder, causal self-attention, pre-LN blocks, AdamW + warmup/cosine, top-k sampling |
| 4 | [ddpm-mnist](./04-ddpm-mnist) | PhD | PyTorch | Denoising Diffusion Probabilistic Models, U-Net with time conditioning, DDPM & DDIM samplers, EMA |
| 5 | [dqn-reinforcement-learning](./05-dqn-reinforcement-learning) | Master / PhD | PyTorch + Gymnasium | Deep Q-Networks, replay buffer, target networks, Double DQN, dueling architecture, epsilon-greedy |

Difficulty increases roughly from project 1 → 4, mirroring the progression
undergrad intro ML → graduate deep learning → research-level generative models.

## Repository Layout

```
ml-projects/
├── 01-neural-network-from-scratch/   # Pure-NumPy MLP with manual backprop
├── 02-cnn-image-classification/      # CIFAR-10 image classification (ResNet)
├── 03-mini-gpt/                      # Character-level GPT language model
├── 04-ddpm-mnist/                    # Diffusion model generating MNIST digits
├── 05-dqn-reinforcement-learning/    # DQN agent playing CartPole / LunarLander
├── requirements.txt                  # Global dependencies (all projects)
└── README.md
```

## Quickstart

Each project is fully independent with its own `README.md`, `requirements.txt`,
tests, and CLI training scripts.

```bash
# (optional) create one environment per project, or one shared env:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Project 1 — train an MLP on the spiral dataset (CPU, ~10 seconds)
cd 01-neural-network-from-scratch
python train.py --dataset spirals --epochs 300
python -m pytest tests/ -v          # gradient checks + unit tests

# Project 2 — ResNet-18 on CIFAR-10 (GPU recommended; CPU smoke test below)
cd ../02-cnn-image-classification
python train.py --model resnet18 --epochs 100

# Project 3 — train a small GPT on built-in text (CPU-friendly)
cd ../03-mini-gpt
python train.py --max-iters 2000
python sample.py --prompt "First Citizen:" --max-new-tokens 300

# Project 4 — diffusion model on MNIST (GPU recommended)
cd ../04-ddpm-mnist
python train.py --epochs 30
python sample.py --num-images 64 --sampler ddim

# Project 5 — DQN on CartPole-v1 (CPU-friendly)
cd ../05-dqn-reinforcement-learning
python train.py --env CartPole-v1 --episodes 600
python evaluate.py --checkpoint runs/latest/best.pt
```

Every `train.py` exposes `--help` with all available hyperparameters, and every
project includes lightweight tests runnable with `pytest` that complete in
seconds-to-minutes on CPU.

## Learning Path

If you are using this repository to study, a suggested order:

1. **01** — implement backprop by hand and pass numerical gradient checks. This
   builds the foundation everything else relies on.
2. **02** — move to PyTorch, learn modern training recipes (augmentation,
   schedules, regularization) and how to reach publication-quality accuracy.
3. **03** — understand the Transformer architecture that powers modern LLMs.
4. **05** — switch from supervised learning to the RL loop (agent, environment,
   reward) with DQN as the canonical entry point.
5. **04** — finish with a research-level generative model and see how
   "learning to denoise" relates to the evidence lower bound (ELBO).

## References

Each project's README lists the original papers and free online course materials
that cover the same material in depth.
