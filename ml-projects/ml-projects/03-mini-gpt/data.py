"""Character-level tokenizer + text corpora.

The default corpus is a small excerpt of public-domain Shakespeare (the same
setting nanoGPT popularized). To keep this repository fully offline-friendly,
a short built-in sample is included; the full tiny-Shakespeare file is
downloaded automatically when training with ``--data shakespeare`` and a
network connection is available.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# A tiny public-domain sample so that the repo works with zero network access.
# Real training uses the ~1 MB tiny-Shakespeare file (see `download_corpus`).
BUILTIN_TEXT = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

Second Citizen:
One word, good citizens.

First Citizen:
Peace, I say. What me, what would you?

All:
Would you proceed especially against Caius Marcius?

First Citizen:
No, I speak not for him. He is a proud man, and one
that would have all men die for his pleasure. Hear me.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?

First Citizen:
We cannot, sir, we are undone already.

MENENIUS:
I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them
Against the Roman state.

Second Citizen:
You are a learned judge; hear me.

MENENIUS:
I would the state were what you would it be!
What then? Then would you have the state
Dispense with taxes, and become your debtor?
""" * 12  # repeat to give the model a few thousand tokens to chew on

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_corpus(data_dir: Path) -> Path:
    """Download tiny-Shakespeare (~1.1 MB) into ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "input.txt"
    if not path.exists():
        print(f"downloading tiny-Shakespeare to {path} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path


def load_text(corpus: str, data_dir: Path = Path("data")) -> str:
    if corpus == "builtin":
        return BUILTIN_TEXT
    if corpus == "shakespeare":
        return download_corpus(data_dir).read_text(encoding="utf-8")
    path = Path(corpus)
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"corpus {corpus!r} not found and is not builtin/shakespeare")


class CharTokenizer:
    """Deterministic character-level vocabulary (sort-ordered)."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)
