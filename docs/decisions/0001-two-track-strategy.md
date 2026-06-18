# ADR-0001: Run two tracks in parallel instead of betting on one

- **Status:** Accepted
- **Date:** 2026-06-17

## Context

The ask is a full-duplex Pakistani-Urdu calling agent, proven in one week. There are
two honestly different ways to get there:

- **Cascade** — chain streaming ASR → LLM → TTS and fake full-duplex with fast
  barge-in. Mature, runs on a free GPU, every part swappable.
- **Moshi** — a genuinely full-duplex model that listens and speaks at once. The
  "real" answer, but English-first, ~24 GB to run, and there is no two-party Urdu
  audio to fine-tune it on.

Betting the week on Moshi alone risks ending Friday with nothing demonstrable. Betting
only on the cascade leaves the actual research question (can we get *true* full-duplex
in Urdu?) unanswered.

## Decision

Run both, and judge each on its own merits.

- **Track B (cascade)** is the primary deliverable — a working Urdu call by Friday.
- **Track A (Moshi)** is the research bet — prove the adaptation *path* (tokenizer
  swap + LoRA on synthesized two-party audio), not a polished product.

The codebase reflects this split at the top level (`cascade/` vs `moshi/`) with shared
foundations (`text/`, `audio/`, `data/`, `eval/`).

## Consequences

- **Good:** there is always a demo (Track B), and a credible answer on the hard
  question (Track A). The shared foundation means neither track duplicates data or
  text handling.
- **Cost:** two integration surfaces to maintain. Mitigated by keeping the model-heavy
  parts in notebooks and the testable logic in the package.
- The two tracks have different "full-duplex": Track B is fast barge-in (feels like
  it); Track A is true simultaneity (is it). The README states this plainly so nobody
  oversells the cascade.
