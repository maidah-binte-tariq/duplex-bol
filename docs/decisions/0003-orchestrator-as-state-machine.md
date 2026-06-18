# ADR-0003: Model the barge-in policy as a synchronous state machine

- **Status:** Accepted
- **Date:** 2026-06-17

## Context

The behavior that makes or breaks this product — keep listening while speaking, stop
fast when interrupted — is real-time and asynchronous. The obvious implementation
(asyncio tasks, audio callbacks, a cancellation token on the TTS coroutine) tangles
the *policy* (when to stop) with *I/O timing* (threads, device buffers, the event
loop). That tangle is almost impossible to unit-test: barge-in timing tests become
flaky sleeps, and CI can't run them without audio hardware.

We need the barge-in policy to be a first-class, testable artifact, because H4 (the
bot stops within ~0.3–0.5 s) is a make-or-break acceptance criterion.

## Decision

Factor the policy out as a **pure, synchronous, frame-driven state machine**
(`DuplexOrchestrator`). It consumes an iterable of `AudioFrame` and yields `Event`s.
It talks to the ASR/LLM/TTS/VAD only through Protocols, and it owns a `LatencyTracker`
with an injectable clock.

Key simplification: **one input frame = one tick**, and while speaking, one TTS chunk
is consumed per tick. Input and output share a fixed frame duration, which collapses
the real-time concurrency into a deterministic loop.

VAD debounce (`BargeInDetector`) is separated from the policy so a single noisy frame
can't cancel the bot.

## Consequences

- **Good:** the barge-in latency and the full event sequence are asserted in plain
  unit tests with scripted frame patterns (`"SSSS.....SSSSS....."`), no GPU, no audio,
  no sleeps. H4/H5 are checked in CI. The same policy object drives the live system
  when wrapped in async transport.
- **Trade-off:** the 1:1 frame/chunk cadence is an idealization. Real ASR partials and
  TTS chunks don't arrive on a perfectly fixed clock. The orchestrator encodes the
  *decision logic*; the async wrapper (Pipecat/LiveKit) owns the messy timing. We
  accept a small fidelity gap in exchange for testability.
- **Boundary:** anything genuinely concurrent (network, audio devices) lives outside
  the orchestrator, in the adapters. The orchestrator never imports an I/O library.
