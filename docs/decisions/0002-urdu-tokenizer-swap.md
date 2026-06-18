# ADR-0002: Swap Moshi's tokenizer for an Urdu one before fine-tuning

- **Status:** Accepted
- **Date:** 2026-06-17

## Context

Moshi reads text through a tokenizer trained on English. Feed Nastaliq through it and
every word shatters into byte-fallback fragments — the model never sees coherent Urdu
sub-words, so a LoRA fine-tune spends its budget fighting the vocabulary instead of
learning prosody and turn-taking.

The J-Moshi team hit the same wall for Japanese. Their expensive step was from-scratch
pre-training (128 GPUs); the step that actually delivered the language was **swapping
the tokenizer** and fine-tuning — which ran on 16 GPUs in ~2 hours. That's the
reachable part, and it's the one we copy.

## Decision

Train an Urdu SentencePiece tokenizer on normalized Urdu transcripts and swap it in
before the LoRA run. Two details that matter:

1. **Normalize first.** The corpus goes through `UrduNormalizer` so the tokenizer
   learns *one* canonical spelling, not three keyboard variants of every yeh. The
   tokenizer training and the eval share the same normalizer — consistency by
   construction. (`moshi/tokenizer.py` → `prepare_corpus`.)
2. **Byte fallback on, high character coverage.** Nastaliq has a long tail of rare
   ligatures; we keep them representable instead of collapsing to `<unk>`.

The wrapper exposes `train_urdu_tokenizer` / `UrduTokenizer` and is import-safe without
the `[moshi]` extra (it raises a clear install hint only when actually called).

## Consequences

- **Good:** the highest-leverage Urdu change is isolated, testable on CPU (we train a
  tiny tokenizer in the test suite and assert an encode/decode round-trip), and done
  before any GPU time is spent.
- **Trade-off:** swapping the tokenizer means the text-embedding rows for the new vocab
  start cold; the LoRA run has to move them. Acceptable for a proof; a fuller effort
  would warm-start from a multilingual checkpoint.
- **Tiny-corpus ergonomics:** real runs use the defaults (vocab in the thousands, byte
  fallback on); `byte_fallback`/`hard_vocab_limit` are exposed so a small corpus can
  still train without SentencePiece erroring on size.
