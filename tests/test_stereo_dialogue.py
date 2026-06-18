"""Two-party stereo synthesis — the channel routing and the overlap behavior."""

from __future__ import annotations

import numpy as np
import pytest

from duplex_bol.data import DialogueConfig, SpeakerClip, build_dialogue
from duplex_bol.data.stereo_dialogue import LEFT, RIGHT


def _clip(speaker: str, seconds: float, sr: int = 24_000, amp: float = 0.5) -> SpeakerClip:
    return SpeakerClip(speaker, np.full(int(seconds * sr), amp, dtype=np.float32), "متن", sr)


def test_two_speakers_route_to_left_and_right():
    a = _clip("agent", 0.1)
    b = _clip("user", 0.1)
    stereo, _dialogue = build_dialogue([a, b])  # default 0.3s gap, no overlap

    assert stereo.ndim == 2 and stereo.shape[1] == 2
    # agent on the left for its turn, silent on the right there
    assert np.allclose(stereo[0:2400, LEFT], 0.5)
    assert np.allclose(stereo[0:2400, RIGHT], 0.0)
    # user on the right for its turn (starts at 0.1 + 0.3 = 0.4s -> sample 9600)
    assert np.allclose(stereo[9600:12000, RIGHT], 0.5)
    assert np.allclose(stereo[9600:12000, LEFT], 0.0)


def test_turn_metadata_matches_timeline():
    _stereo, dialogue = build_dialogue([_clip("agent", 0.1), _clip("user", 0.1)])
    assert dialogue.sample_rate == 24_000
    assert dialogue.duration_s == pytest.approx(0.5)  # 0.1 + 0.3 + 0.1
    t0, t1 = dialogue.turns
    assert (t0.channel, t0.speaker_id) == (LEFT, "agent")
    assert t0.start_s == pytest.approx(0.0) and t0.end_s == pytest.approx(0.1)
    assert (t1.channel, t1.speaker_id) == (RIGHT, "user")
    assert t1.start_s == pytest.approx(0.4) and t1.end_s == pytest.approx(0.5)


def test_overlap_produces_simultaneous_speech():
    # Zero gap, 50 ms overlap -> the second turn starts before the first ends, so
    # both channels are live at the same time. This is the barge-in signal Moshi needs.
    cfg = DialogueConfig(inter_turn_gap_s=0.0, overlap_s=0.05)
    stereo, _ = build_dialogue([_clip("agent", 0.1), _clip("user", 0.1)], cfg)
    both_live = (np.abs(stereo[:, LEFT]) > 0) & (np.abs(stereo[:, RIGHT]) > 0)
    assert both_live.any(), "expected an overlap region with both speakers active"
    assert both_live.sum() == pytest.approx(0.05 * 24_000, abs=2)  # ~50 ms of overlap


def test_resamples_clip_to_dialogue_rate():
    # A 16 kHz clip placed into a 24 kHz dialogue must be resampled, not truncated.
    _stereo, dialogue = build_dialogue([_clip("agent", 0.1, sr=16_000)])
    assert dialogue.sample_rate == 24_000
    assert dialogue.turns[0].end_s == pytest.approx(0.1, abs=1e-3)


def test_more_than_two_speakers_is_rejected():
    clips = [_clip("a", 0.1), _clip("b", 0.1), _clip("c", 0.1)]
    with pytest.raises(ValueError, match="two-party"):
        build_dialogue(clips)


def test_empty_clip_list_is_rejected():
    with pytest.raises(ValueError, match="at least one clip"):
        build_dialogue([])
