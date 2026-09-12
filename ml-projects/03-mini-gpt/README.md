# mini-GPT — A Char-Level Transformer Language Model

A compact but complete **decoder-only Transformer** (GPT architecture)
implemented from scratch in PyTorch: causal multi-head self-attention,
pre-LayerNorm blocks, GELU MLPs, weight tying, AdamW with LR warmup +
cosine decay, and temperature/top-k sampling. Roughly 200 lines of model code,
no `transformers` library anywhere.

This is the "look inside an LLM" project: after finishing it you can explain
every matrix multiplication between your prompt and the model's next-token
prediction — the same architecture taught in Stanford CS224n / Berkeley
CS182-style deep learning courses.

## Architecture

```
tokens ──► tok_emb + pos_emb ──► dropout
   ──► [ Block × n_layer ]          each Block:
   │        x = x + Attn(LN(x))       ├─ causal self-attention (n_head heads)
   │        x = x + MLP(LN(x))        └─ Linear(4d) → GELU → Linear(d)
   ──► LN_f ──► LM head (weight-tied to tok_emb)
   ──► softmax over vocab = next-token distribution
```

Key implementation details worth reading the code for:

- **Causal masking** — `CausalSelfAttention` builds a lower-triangular boolean
  mask so position `i` attends only to positions `≤ i`; `tests/test_model.py`
  verifies it by perturbing a future token and asserting past logits are
  unchanged (the single most important correctness test for any LM).
- **`F.scaled_dot_product_attention`** (PyTorch 2+) transparently dispatches to
  the fused FlashAttention kernel when available — one line, big speedup.
- **Pre-LN** residuals (LayerNorm *inside* the residual branch) — the modern
  stable choice vs. the original post-LN Transformer.
- **Weight tying** — the LM head shares weights with the token embedding
  (Press & Wolf, 2017): fewer parameters, better rare-token learning.
- **GPT-2 init recipe** — N(0, 0.02) everywhere, residual projections scaled by
  `1/√(2·n_layer)`.

## Training Recipe (`train.py`)

AdamW (β₂ = 0.95) with **decoupled weight decay applied only to matrices
(≥2-D params)**, linear warmup → cosine decay, gradient clipping at 1.0,
periodic held-out loss estimation, and automatic checkpointing + sampling.

```bash
pip install -r requirements.txt          # just torch

# CPU quickstart (built-in text, ~15 s): watch loss fall 3.9 -> ~1.8
python train.py --corpus builtin --max-iters 200 --n-layer 2 --n-embd 64

# Real run: tiny-Shakespeare (auto-downloads ~1 MB), GPU strongly advised
python train.py --corpus shakespeare \
    --n-layer 6 --n-head 6 --n-embd 384 --block-size 256 \
    --batch-size 64 --max-iters 5000 --lr 1e-3
```

## Sampling (`sample.py`)

```bash
python sample.py --prompt "First Citizen:" --max-new-tokens 400 --top-k 20
python sample.py --temperature 0.5    # conservative
python sample.py --temperature 1.3    # unhinged
```

`generate()` supports temperature scaling and top-k truncation, and crops the
context window to `block_size` once the sequence grows beyond it.

## Results

Built-in corpus, 2-layer/64-dim, 200 iterations (~15 s on laptop CPU):

| iter | train loss | val loss |
|------|-----------|----------|
| 1    | 3.95      | 3.95     |
| 100  | 2.32      | 2.30     |
| 200  | 1.82      | 1.81     |

Full tiny-Shakespeare (6-layer/384-dim, 5000 iters, single GPU) typically
reaches **val loss ≈ 1.5** and produces clearly Shakespeare-flavored text —
for reference, the settings and loss curve mirror the well-known nanoGPT
"shakespeare-char" baseline.

## Tests

```bash
python -m pytest tests/ -v
```

- `test_causality` — the flagship check described above
- `test_overfit_single_batch` — model can memorize 8 tokens (optimization sanity)
- shapes, block-size assertion, generation bounds, tokenizer round-trip

## Topics Covered By Courses (for further study)

- Stanford **CS224n** (NLP with Deep Learning) — Transformers, LM training
- Stanford **CS336** (Language Modeling from Scratch) — this project's spirit
- Berkeley **CS182 / Princeton COS 485** — attention mechanics, residual nets

## References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
2. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019
3. Brown et al., *Language Models are Few-Shot Learners* (GPT-3), NeurIPS 2020
4. Press & Wolf, *Using the Output Embedding to Improve Language Models*, EACL 2017
5. Su et al., *RoFormer* (positional encoding variants worth reading next), 2021
