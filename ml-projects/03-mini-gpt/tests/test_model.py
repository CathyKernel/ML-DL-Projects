"""Unit tests for the mini-GPT model.

The flagship test verifies the *causal* property: changing a future token must
not change the logits of past positions. If the mask is broken, language
modeling becomes trivially leaky — the model could just read the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import CharTokenizer, BUILTIN_TEXT          # noqa: E402
from model import GPT, GPTConfig                      # noqa: E402


def tiny_config() -> GPTConfig:
    return GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2,
                     n_embd=16, dropout=0.0)


def test_param_count_scales():
    small = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=1, n_embd=16))
    big = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=2, n_head=1, n_embd=16))
    assert big.num_params() > small.num_params() > 0


def test_output_shapes_and_loss():
    model = GPT(tiny_config())
    x = torch.randint(0, 32, (4, 10))
    y = torch.randint(0, 32, (4, 10))
    logits, loss = model(x, y)
    assert logits.shape == (4, 10, 32)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_causality():
    """Perturbing token t must leave logits at positions < t unchanged."""
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    x = torch.randint(0, 32, (1, 12))
    with torch.no_grad():
        base, _ = model(x, x)
        x_pert = x.clone()
        x_pert[0, 7] = (x_pert[0, 7] + 1) % 32       # change position 7
        pert, _ = model(x_pert, x_pert)
    assert torch.allclose(base[0, :7], pert[0, :7], atol=1e-5), \
        "future token leaked into past logits!"
    assert not torch.allclose(base[0, 7], pert[0, 7])


def test_block_size_assertion():
    model = GPT(tiny_config())
    with pytest.raises(AssertionError):
        model(torch.randint(0, 32, (1, tiny_config().block_size + 1)))


def test_overfit_single_batch():
    """A 2-layer GPT should memorize 8 tokens in a few hundred steps (CPU-fast)."""
    torch.manual_seed(0)
    model = GPT(tiny_config())
    x = torch.randint(0, 32, (8, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2, f"failed to overfit: loss={loss.item():.3f}"


def test_generate_shapes_and_topk():
    model = GPT(tiny_config()).eval()
    idx = torch.zeros((2, 4), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=6, temperature=0.8, top_k=5)
    assert out.shape == (2, 10)
    assert (out >= 0).all() and (out < 32).all()


def test_tokenizer_roundtrip():
    tok = CharTokenizer(BUILTIN_TEXT[:500])
    text = "First Citizen:"
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == len(set(BUILTIN_TEXT[:500]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
