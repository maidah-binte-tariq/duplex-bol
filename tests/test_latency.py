"""Latency tooling — driven by a scripted clock so every number is exact."""

from __future__ import annotations

from duplex_bol.eval import LatencyBudget, LatencyTracker, Stopwatch


def test_stopwatch_uses_injected_clock(fake_clock):
    # start at t=0.0s, stop at t=0.5s -> 500 ms
    sw = Stopwatch(clock=fake_clock([0.0, 0.5]))
    with sw:
        pass
    assert sw.elapsed_ms == 500.0


def test_tracker_scoped_timer_records(fake_clock):
    tracker = LatencyTracker(clock=fake_clock([1.0, 1.5]))  # 500 ms (0.5 is exact in float)
    with tracker.time("response_start"):
        pass
    summary = tracker.summary()
    assert summary["response_start"]["count"] == 1.0
    assert summary["response_start"]["p50"] == 500.0


def test_percentiles_match_linear_interpolation():
    tracker = LatencyTracker()
    for v in (10, 20, 30, 40, 50):
        tracker.record("m", v)
    stats = tracker.summary()["m"]
    assert stats["mean"] == 30.0
    assert stats["p50"] == 30.0  # rank 2.0 -> exact middle
    assert stats["p95"] == 48.0  # rank 3.8 -> 40 + 0.8*(50-40)
    assert stats["max"] == 50.0


def test_budget_passes_when_under_threshold():
    tracker = LatencyTracker()
    for v in (300, 350, 400, 450):
        tracker.record("barge_in_stop", v)
    report = LatencyBudget.voice_agent_default().evaluate(tracker)
    assert report.ok
    assert report.results["barge_in_stop"].ok


def test_budget_fails_when_p95_exceeds_threshold():
    tracker = LatencyTracker()
    for v in (800, 900, 1000, 1500):
        tracker.record("response_start", v)
    report = LatencyBudget.voice_agent_default().evaluate(tracker)
    assert not report.ok
    assert "FAIL" in str(report)


def test_budget_ignores_unsampled_metrics():
    # Only barge_in_stop has samples; response_start should be silently skipped,
    # not crash, so a partial run still reports.
    tracker = LatencyTracker()
    tracker.record("barge_in_stop", 100)
    report = LatencyBudget.voice_agent_default().evaluate(tracker)
    assert set(report.results) == {"barge_in_stop"}
    assert report.ok
