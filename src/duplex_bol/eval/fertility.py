"""Tokenizer fertility — the quantitative case for Track A's tokenizer swap.

"Fertility" is the average number of tokens a tokenizer spends per word. It is the
cleanest single number for *why the Urdu tokenizer swap matters*:

* A tokenizer with no Urdu sub-words (Moshi's English vocab) has no choice but to
  fall back to bytes on Nastaliq. Urdu code points are 2 bytes in UTF-8, so a word
  shatters into ~2 tokens **per character** — fertility goes through the roof and
  the model never sees a coherent Urdu unit.
* An Urdu SentencePiece model learns real sub-words and pulls fertility back toward
  one-token-per-word, so every training step covers far more text.

Lower is better. This is measured directly (see ``scripts/run_benchmarks.py``), not
asserted — the byte-fallback baseline is exactly how a vocab-less tokenizer degrades,
so the comparison is honest rather than a strawman.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

Encoder = Callable[[str], list[int]]


@dataclass(frozen=True)
class FertilityResult:
    """Aggregate token/word/char counts for one tokenizer over a text set."""

    name: str
    n_words: int
    n_tokens: int
    n_chars: int

    @property
    def tokens_per_word(self) -> float:
        return self.n_tokens / self.n_words if self.n_words else 0.0

    @property
    def chars_per_token(self) -> float:
        return self.n_chars / self.n_tokens if self.n_tokens else 0.0

    def speedup_over(self, baseline: FertilityResult) -> float:
        """How many times fewer tokens per word than ``baseline`` (>1 means better)."""
        mine = self.tokens_per_word
        return baseline.tokens_per_word / mine if mine else 0.0


def measure_fertility(name: str, texts: Iterable[str], encode: Encoder) -> FertilityResult:
    """Run ``encode`` over ``texts`` and tally words / tokens / characters."""
    n_words = n_tokens = n_chars = 0
    for text in texts:
        n_words += len(text.split())
        n_chars += len(text)
        n_tokens += len(encode(text))
    return FertilityResult(name=name, n_words=n_words, n_tokens=n_tokens, n_chars=n_chars)


def byte_fallback_encode(text: str) -> list[int]:
    """The worst case: a tokenizer with no Urdu vocab emits one token per UTF-8 byte.

    This is literally what SentencePiece/BPE byte-fallback does for unseen scripts,
    so using the UTF-8 byte length as the token count is a faithful — not unfair —
    model of an Urdu-blind tokenizer.
    """
    return list(text.encode("utf-8"))
