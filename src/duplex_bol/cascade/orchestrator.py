"""The barge-in orchestrator: a synchronous, frame-driven full-duplex policy.

The whole point of the project is that the agent must not freeze mid-turn — when
the caller starts talking, the bot has to *stop talking* fast. A cascade can't be
truly simultaneous like Moshi, but it can fake it convincingly: keep the VAD live
while the bot speaks, and the instant the caller barges in, kill the bot's audio
and start listening again. Done within ~half a second it feels full-duplex.

Time model
----------
One :class:`AudioFrame` in == one tick. While the bot is speaking, each tick also
plays one :class:`TtsChunk`, so input frames and output chunks share a fixed
duration (``frame_duration_ms``). That 1:1 cadence is what makes an inherently
async problem deterministic enough to unit-test.

Latency accounting
------------------
* **barge-in stop** — first speech frame of the interrupting utterance → the tick
  the bot goes quiet. Equals the onset debounce window, so it's bounded by design.
* **response start** — caller's last speech frame → the bot's first audio chunk.
  Covers the offset debounce plus the (here instant) brain + TTS spin-up.

Both are pushed into a :class:`~duplex_bol.eval.LatencyTracker` so the demo can be
graded against the H4/H5 budget.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto

from duplex_bol.cascade.interfaces import (
    AgentBrain,
    AgentReply,
    AudioFrame,
    BargeIn,
    CaptureStarted,
    Event,
    PartialTranscript,
    SpeechEnded,
    SpeechStarted,
    StreamingASR,
    StreamingTTS,
    TtsChunk,
    UserUtterance,
    VoiceActivityDetector,
)
from duplex_bol.eval.latency import LatencyTracker


class VadEdge(Enum):
    NONE = auto()
    ONSET = auto()  # confirmed: caller started speaking
    OFFSET = auto()  # confirmed: caller stopped speaking


@dataclass
class _BargeInUpdate:
    edge: VadEdge
    run_start_index: int | None  # frame index where the current speech run began


class BargeInDetector:
    """Debounces a raw VAD into confirmed speech onset/offset edges.

    A single noisy frame must not cancel the bot (a cough, a door). Onset requires
    ``onset_frames`` consecutive speech frames; offset requires ``hangover_frames``
    consecutive silent ones. The hangover also stops the agent from finalizing the
    instant someone pauses for breath mid-sentence.
    """

    def __init__(self, onset_frames: int = 3, hangover_frames: int = 5) -> None:
        if onset_frames < 1 or hangover_frames < 1:
            raise ValueError("debounce windows must be >= 1 frame")
        self.onset_frames = onset_frames
        self.hangover_frames = hangover_frames
        self.reset()

    def reset(self) -> None:
        self._in_speech = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._run_start_index: int | None = None

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def update(self, is_speech: bool, index: int) -> _BargeInUpdate:
        if is_speech:
            self._silence_streak = 0
            if self._speech_streak == 0:
                self._run_start_index = index
            self._speech_streak += 1
            if not self._in_speech and self._speech_streak >= self.onset_frames:
                self._in_speech = True
                return _BargeInUpdate(VadEdge.ONSET, self._run_start_index)
        else:
            self._speech_streak = 0
            if self._in_speech:
                self._silence_streak += 1
                if self._silence_streak >= self.hangover_frames:
                    self._in_speech = False
                    return _BargeInUpdate(VadEdge.OFFSET, None)
        return _BargeInUpdate(VadEdge.NONE, None)


class _State(Enum):
    LISTENING = auto()
    SPEAKING = auto()


class DuplexOrchestrator:
    """Drives one phone call from a stream of input frames to a stream of events.

    Construct it with the four components (any objects satisfying the cascade
    Protocols), then feed :meth:`run` the caller's audio frames. It yields
    :class:`Event` objects describing everything that happened, and records the
    barge-in / response latencies into :attr:`tracker`.
    """

    def __init__(
        self,
        *,
        vad: VoiceActivityDetector,
        asr: StreamingASR,
        agent: AgentBrain,
        tts: StreamingTTS,
        bargein: BargeInDetector | None = None,
        frame_duration_ms: float = 20.0,
        tracker: LatencyTracker | None = None,
    ) -> None:
        self.vad = vad
        self.asr = asr
        self.agent = agent
        self.tts = tts
        self.bargein = bargein or BargeInDetector()
        self.frame_ms = frame_duration_ms
        self.tracker = tracker or LatencyTracker()

    def run(self, frames: Iterable[AudioFrame]) -> Iterator[Event]:
        state = _State.LISTENING
        capturing = False
        last_speech_frame: int | None = None
        speech_iter: Iterator[TtsChunk] | None = None
        speech_started = False
        pending_reply = ""

        self.bargein.reset()
        self.asr.reset()

        idx = -1
        for idx, frame in enumerate(frames):
            is_speech = self.vad.is_speech(frame)
            edge = self.bargein.update(is_speech, idx)

            if state is _State.LISTENING:
                if is_speech:
                    if not capturing:
                        capturing = True
                        self.asr.reset()
                        yield CaptureStarted(idx)
                    partial = self.asr.accept(frame)
                    if partial is not None and partial.text:
                        yield PartialTranscript(idx, partial.text)
                    last_speech_frame = idx

                if edge.edge is VadEdge.OFFSET and capturing:
                    final = self.asr.finalize()
                    yield UserUtterance(idx, final.text)
                    pending_reply = self.agent.respond(final.text)
                    yield AgentReply(idx, pending_reply)
                    speech_iter = iter(self.tts.synthesize(pending_reply))
                    speech_started = False
                    capturing = False
                    state = _State.SPEAKING

            elif state is _State.SPEAKING:
                # Caller barges in: stop the bot now, start listening immediately.
                if edge.edge is VadEdge.ONSET:
                    run_start = edge.run_start_index if edge.run_start_index is not None else idx
                    stop_latency = (idx - run_start + 1) * self.frame_ms
                    self.tracker.record("barge_in_stop", stop_latency)
                    yield BargeIn(idx, stop_latency)
                    speech_iter = None
                    state = _State.LISTENING
                    capturing = True
                    self.asr.reset()
                    yield CaptureStarted(idx)
                    partial = self.asr.accept(frame)
                    if partial is not None and partial.text:
                        yield PartialTranscript(idx, partial.text)
                    last_speech_frame = idx
                    continue

                assert speech_iter is not None
                chunk = next(speech_iter, None)
                if not speech_started:
                    speech_started = True
                    if last_speech_frame is not None:
                        self.tracker.record(
                            "response_start", (idx - last_speech_frame) * self.frame_ms
                        )
                    yield SpeechStarted(idx, pending_reply)
                if chunk is None or chunk.is_last:
                    yield SpeechEnded(idx)
                    speech_iter = None
                    state = _State.LISTENING

        # If the stream ends while the caller was still mid-utterance, commit it so
        # nothing is silently dropped.
        if capturing and idx >= 0:
            final = self.asr.finalize()
            if final.text:
                yield UserUtterance(idx, final.text)  # idx is the last seen frame
