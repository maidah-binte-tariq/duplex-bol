"""The barge-in policy: clean turn-taking, interruption latency, and debounce.

Frames are written as a pattern string — ``S`` is a speech frame (loud), ``.`` is
silence — so each scenario's timeline is readable at a glance.
"""

from __future__ import annotations

import numpy as np

from duplex_bol.cascade import (
    AudioFrame,
    BargeIn,
    BargeInDetector,
    CaptureStarted,
    DuplexOrchestrator,
    SpeechEnded,
    SpeechStarted,
    UserUtterance,
    VadEdge,
)
from duplex_bol.cascade.fakes import ChunkedTTS, RuleBasedAgent, ScriptedASR
from duplex_bol.cascade.vad import EnergyVAD
from duplex_bol.eval import LatencyBudget

FRAME_SAMPLES = 320  # 20 ms @ 16 kHz


def _frames(pattern: str) -> list[AudioFrame]:
    out = []
    for ch in pattern:
        samples = (
            np.full(FRAME_SAMPLES, 0.3, np.float32)
            if ch == "S"
            else np.zeros(FRAME_SAMPLES, np.float32)
        )
        out.append(AudioFrame(samples))
    return out


def _orchestrator(utterances: list[str]) -> DuplexOrchestrator:
    return DuplexOrchestrator(
        vad=EnergyVAD(threshold_dbfs=-40.0),
        asr=ScriptedASR(utterances),
        agent=RuleBasedAgent(default="وعلیکم السلام"),  # 2 words -> 6 TTS chunks
        tts=ChunkedTTS(frames_per_word=3, frame_samples=FRAME_SAMPLES),
        frame_duration_ms=20.0,
    )


def _types(events):
    return [type(e).__name__ for e in events]


# --- BargeInDetector ----------------------------------------------------------
def test_detector_confirms_onset_after_threshold_frames():
    det = BargeInDetector(onset_frames=3, hangover_frames=2)
    assert det.update(True, 0).edge is VadEdge.NONE
    assert det.update(True, 1).edge is VadEdge.NONE
    update = det.update(True, 2)
    assert update.edge is VadEdge.ONSET
    assert update.run_start_index == 0  # points back to the first speech frame


def test_detector_ignores_a_single_noisy_frame():
    det = BargeInDetector(onset_frames=3, hangover_frames=2)
    det.update(True, 0)  # a lone blip...
    det.update(False, 1)  # ...then silence resets the streak
    assert det.update(True, 2).edge is VadEdge.NONE
    assert not det.in_speech


def test_detector_confirms_offset_after_hangover():
    det = BargeInDetector(onset_frames=1, hangover_frames=2)
    det.update(True, 0)  # onset
    assert det.update(False, 1).edge is VadEdge.NONE  # within hangover
    assert det.update(False, 2).edge is VadEdge.OFFSET


# --- orchestrator: a clean uninterrupted turn ---------------------------------
def test_clean_turn_produces_full_event_sequence():
    orch = _orchestrator(["السلام علیکم"])
    events = list(orch.run(_frames("SSSS............")))
    names = _types(events)

    # The key events appear in order and the bot finishes on its own (no barge-in).
    assert "CaptureStarted" in names
    assert names.index("UserUtterance") < names.index("AgentReply") < names.index("SpeechStarted")
    assert "SpeechEnded" in names
    assert "BargeIn" not in names

    user = next(e for e in events if isinstance(e, UserUtterance))
    assert user.text == "السلام علیکم"


def test_clean_turn_meets_response_latency_budget():
    orch = _orchestrator(["السلام علیکم"])
    list(orch.run(_frames("SSSS............")))
    report = LatencyBudget.voice_agent_default().evaluate(orch.tracker)
    assert "response_start" in report.results
    assert report.results["response_start"].ok  # well under 1000 ms


# --- orchestrator: barge-in ---------------------------------------------------
def test_barge_in_stops_the_bot_quickly():
    orch = _orchestrator(["السلام علیکم", "میں ٹھیک ہوں"])
    # caller, silence (offset), bot starts speaking AND caller talks over it, silence.
    events = list(orch.run(_frames("SSSS.....SSSSS.....")))

    barge = next((e for e in events if isinstance(e, BargeIn)), None)
    assert barge is not None, "expected the caller to interrupt the bot"
    # onset debounce is 3 frames @ 20 ms -> the bot goes quiet 60 ms after the
    # interrupting speech began.
    assert barge.stop_latency_ms == 60.0

    # A fresh capture must open immediately after the interruption.
    after = events[events.index(barge) :]
    assert any(isinstance(e, CaptureStarted) for e in after)


def test_barge_in_latency_meets_budget():
    orch = _orchestrator(["السلام علیکم", "میں ٹھیک ہوں"])
    list(orch.run(_frames("SSSS.....SSSSS.....")))
    report = LatencyBudget.voice_agent_default().evaluate(orch.tracker)
    assert report.results["barge_in_stop"].ok  # 60 ms <= 500 ms


def test_speech_started_carries_reply_text():
    orch = _orchestrator(["السلام علیکم"])
    events = list(orch.run(_frames("SSSS............")))
    started = next(e for e in events if isinstance(e, SpeechStarted))
    assert started.text == "وعلیکم السلام"


def test_no_speech_yields_no_turns():
    orch = _orchestrator(["unused"])
    events = list(orch.run(_frames("................")))
    assert events == []
    assert not any(isinstance(e, SpeechEnded) for e in events)
