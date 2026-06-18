"""Latency measurement and the budget the demo is graded against.

A voice agent lives or dies on two numbers (the report calls them H4 and H5):

* **barge-in stop latency** — how long after the caller starts talking before the
  bot actually goes quiet. Target ≤ 500 ms.
* **response latency** — how long after the caller stops before audio comes back.
  Target ≤ 1000 ms.

The clock is injectable so tests can drive time deterministically instead of
sleeping. In production you'd hand it ``time.perf_counter``; in a test you hand
it a list-backed fake and assert exact millisecond values.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType

Clock = Callable[[], float]


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (the numpy default), pure-Python.

    ``q`` is in [0, 100]. Kept dependency-free so the latency tooling can be
    lifted into a notebook without pulling numpy.
    """
    if not values:
        raise ValueError("cannot take a percentile of zero samples")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (q / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


class Stopwatch:
    """Context manager that records one elapsed-milliseconds reading.

    >>> sw = Stopwatch()
    >>> with sw:
    ...     do_work()
    >>> sw.elapsed_ms  # doctest: +SKIP
    """

    def __init__(self, clock: Clock = time.perf_counter) -> None:
        self._clock = clock
        self._start: float | None = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = self._clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._start is not None  # set in __enter__
        self.elapsed_ms = (self._clock() - self._start) * 1000.0


@dataclass
class LatencyTracker:
    """Accumulates named latency observations and summarizes them.

    One tracker per run; call :meth:`record` (or :meth:`time`) as the agent emits
    events, then :meth:`summary` for the p50/p95/max table.
    """

    clock: Clock = time.perf_counter
    _samples: dict[str, list[float]] = field(default_factory=dict)

    def record(self, metric: str, milliseconds: float) -> None:
        self._samples.setdefault(metric, []).append(milliseconds)

    def time(self, metric: str) -> _ScopedTimer:
        """``with tracker.time("response"): ...`` — records on exit."""
        return _ScopedTimer(self, metric, self.clock)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for metric, samples in self._samples.items():
            out[metric] = {
                "count": float(len(samples)),
                "mean": sum(samples) / len(samples),
                "p50": _percentile(samples, 50),
                "p95": _percentile(samples, 95),
                "max": max(samples),
            }
        return out


class _ScopedTimer:
    def __init__(self, tracker: LatencyTracker, metric: str, clock: Clock) -> None:
        self._tracker = tracker
        self._metric = metric
        self._sw = Stopwatch(clock)

    def __enter__(self) -> _ScopedTimer:
        self._sw.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._sw.__exit__(exc_type, exc, tb)
        self._tracker.record(self._metric, self._sw.elapsed_ms)


@dataclass(frozen=True)
class LatencyReport:
    """Pass/fail of a tracker's p95 against a budget."""

    results: dict[str, _BudgetLine]

    @property
    def ok(self) -> bool:
        return all(line.ok for line in self.results.values())

    def __str__(self) -> str:
        rows = ["metric                 p95(ms)   budget(ms)   status"]
        for metric, line in self.results.items():
            status = "PASS" if line.ok else "FAIL"
            rows.append(f"{metric:<22} {line.p95_ms:>7.1f}   {line.budget_ms:>9.1f}   {status}")
        return "\n".join(rows)


@dataclass(frozen=True)
class _BudgetLine:
    p95_ms: float
    budget_ms: float

    @property
    def ok(self) -> bool:
        return self.p95_ms <= self.budget_ms


@dataclass(frozen=True)
class LatencyBudget:
    """Map of metric → max acceptable p95 in milliseconds.

    The default mirrors the report's acceptance criteria: barge-in stop ≤ 500 ms,
    response start ≤ 1000 ms. Metrics present in the tracker but absent from the
    budget are ignored (and vice-versa — a budgeted metric with no samples is a
    silent skip, not a crash, so partial runs still report).
    """

    thresholds_ms: dict[str, float]

    @classmethod
    def voice_agent_default(cls) -> LatencyBudget:
        return cls(
            thresholds_ms={
                "barge_in_stop": 500.0,
                "response_start": 1000.0,
            }
        )

    def evaluate(self, tracker: LatencyTracker) -> LatencyReport:
        summary = tracker.summary()
        results: dict[str, _BudgetLine] = {}
        for metric, budget in self.thresholds_ms.items():
            if metric in summary:
                results[metric] = _BudgetLine(p95_ms=summary[metric]["p95"], budget_ms=budget)
        return LatencyReport(results=results)
