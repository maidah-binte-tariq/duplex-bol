"""Track B: the streaming cascade and its barge-in orchestrator.

The real production stack (Whisper → LLM → Urdu TTS, wrapped in Pipecat/LiveKit)
is async and GPU-bound, which makes the *policy* — when to start listening, when
to stop talking — almost impossible to unit-test. So we factor that policy out:
the orchestrator is a pure, synchronous, frame-driven state machine that talks to
components only through the Protocols in :mod:`duplex_bol.cascade.interfaces`. Swap
the fakes for real ASR/TTS adapters and the same policy drives a live call.
"""

from __future__ import annotations

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
    Transcript,
    TtsChunk,
    UserUtterance,
    VoiceActivityDetector,
)
from duplex_bol.cascade.orchestrator import (
    BargeInDetector,
    DuplexOrchestrator,
    VadEdge,
)
from duplex_bol.cascade.vad import EnergyVAD

__all__ = [
    "AgentBrain",
    "AgentReply",
    "AudioFrame",
    "BargeIn",
    "BargeInDetector",
    "CaptureStarted",
    "DuplexOrchestrator",
    "EnergyVAD",
    "Event",
    "PartialTranscript",
    "SpeechEnded",
    "SpeechStarted",
    "StreamingASR",
    "StreamingTTS",
    "Transcript",
    "TtsChunk",
    "UserUtterance",
    "VadEdge",
    "VoiceActivityDetector",
]
