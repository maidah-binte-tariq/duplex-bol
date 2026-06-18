"""The contracts every cascade component speaks, plus the event types it emits.

Keeping these as ``typing.Protocol`` (structural typing) means a real adapter
doesn't have to import or subclass anything from us — if it has the right methods,
it fits. That's what lets the same orchestrator drive both the in-memory fakes and
a faster-whisper / Orpheus stack.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Audio = npt.NDArray[np.float32]


@dataclass(frozen=True)
class AudioFrame:
    """A fixed-duration slice of mono audio. The unit the call clock ticks on."""

    samples: Audio


@dataclass(frozen=True)
class Transcript:
    """ASR output. ``is_final`` distinguishes a stable result from a live partial."""

    text: str
    is_final: bool


@dataclass(frozen=True)
class TtsChunk:
    """A slice of synthesized speech. ``is_last`` lets the loop know the turn is over."""

    samples: Audio
    is_last: bool = False


@runtime_checkable
class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: AudioFrame) -> bool: ...


@runtime_checkable
class StreamingASR(Protocol):
    """Incremental speech-to-text. ``accept`` streams partials; ``finalize`` commits."""

    def reset(self) -> None: ...
    def accept(self, frame: AudioFrame) -> Transcript | None: ...
    def finalize(self) -> Transcript: ...


@runtime_checkable
class AgentBrain(Protocol):
    """The LLM turn: caller's finalized text in, reply text out."""

    def respond(self, user_text: str) -> str: ...


@runtime_checkable
class StreamingTTS(Protocol):
    def synthesize(self, text: str) -> Iterator[TtsChunk]: ...


# --- events the orchestrator emits (observability + eval + test assertions) ----
@dataclass(frozen=True)
class Event:
    frame_index: int


@dataclass(frozen=True)
class CaptureStarted(Event):
    """The agent began capturing a caller utterance."""


@dataclass(frozen=True)
class PartialTranscript(Event):
    text: str


@dataclass(frozen=True)
class UserUtterance(Event):
    """A finalized caller turn."""

    text: str


@dataclass(frozen=True)
class AgentReply(Event):
    text: str


@dataclass(frozen=True)
class SpeechStarted(Event):
    """The bot began speaking ``text``."""

    text: str


@dataclass(frozen=True)
class SpeechEnded(Event):
    """The bot finished its turn naturally (was not interrupted)."""


@dataclass(frozen=True)
class BargeIn(Event):
    """The caller talked over the bot; the bot stopped ``stop_latency_ms`` ms later."""

    stop_latency_ms: float
