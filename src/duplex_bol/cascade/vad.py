"""A dead-simple energy VAD.

This is a baseline, not Silero. An RMS threshold is enough to exercise the
orchestrator and to demo on clean audio; for a real call you'd drop in a learned
VAD (the ``cascade`` extra pulls one) behind the same
:class:`~duplex_bol.cascade.interfaces.VoiceActivityDetector` Protocol. Honest
about what it is: energy VADs trip on background noise and music, which is exactly
why barge-in needs the debounce layer on top (see ``BargeInDetector``).
"""

from __future__ import annotations

from duplex_bol.audio.transforms import rms_dbfs
from duplex_bol.cascade.interfaces import AudioFrame


class EnergyVAD:
    """Flags a frame as speech when its RMS exceeds ``threshold_dbfs``."""

    def __init__(self, threshold_dbfs: float = -40.0) -> None:
        self.threshold_dbfs = threshold_dbfs

    def is_speech(self, frame: AudioFrame) -> bool:
        return rms_dbfs(frame.samples) > self.threshold_dbfs
