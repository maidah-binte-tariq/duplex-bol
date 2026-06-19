"""Evaluation: word/character error rate and the latency budget."""

from __future__ import annotations

from duplex_bol.eval.fertility import (
    FertilityResult,
    byte_fallback_encode,
    measure_fertility,
)
from duplex_bol.eval.latency import (
    LatencyBudget,
    LatencyReport,
    LatencyTracker,
    Stopwatch,
)
from duplex_bol.eval.wer import (
    ErrorCounts,
    aggregate_wer,
    character_error_rate,
    word_error_rate,
)

__all__ = [
    "ErrorCounts",
    "FertilityResult",
    "LatencyBudget",
    "LatencyReport",
    "LatencyTracker",
    "Stopwatch",
    "aggregate_wer",
    "byte_fallback_encode",
    "character_error_rate",
    "measure_fertility",
    "word_error_rate",
]
