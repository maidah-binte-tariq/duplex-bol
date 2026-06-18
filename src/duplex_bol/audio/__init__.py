"""Audio I/O and the sample-rate / channel plumbing the trainers care about."""

from __future__ import annotations

from duplex_bol.audio.io import read_wav, write_wav
from duplex_bol.audio.transforms import (
    duration_s,
    mix_to_stereo,
    peak_normalize,
    resample,
    rms_dbfs,
    to_mono,
)

__all__ = [
    "duration_s",
    "mix_to_stereo",
    "peak_normalize",
    "read_wav",
    "resample",
    "rms_dbfs",
    "to_mono",
    "write_wav",
]
