"""Shared fixtures.

Two things almost every test wants: a deterministic clock (so latency assertions
are exact, not flaky) and a scratch WAV on disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest


class FakeClock:
    """A clock that returns pre-scripted readings, one per call.

    Hand it the exact sequence of ``perf_counter`` values you want the code under
    test to observe. Raises if it runs dry, which catches "called the clock more
    times than I expected" bugs.
    """

    def __init__(self, readings: Sequence[float]) -> None:
        self._readings = list(readings)
        self._i = 0

    def __call__(self) -> float:
        if self._i >= len(self._readings):
            raise AssertionError("FakeClock exhausted — code read the clock more than scripted")
        value = self._readings[self._i]
        self._i += 1
        return value


@pytest.fixture
def fake_clock() -> type[FakeClock]:
    return FakeClock


@pytest.fixture
def sine() -> np.ndarray:
    """A 1 kHz tone at 16 kHz, half a second, amplitude 0.5."""
    sr = 16_000
    t = np.arange(int(0.5 * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    return tmp_path / "clip.wav"
