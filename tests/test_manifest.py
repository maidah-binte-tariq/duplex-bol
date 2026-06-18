"""Manifest schemas, JSONL round-trips (Urdu-safe), and pre-flight validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from duplex_bol.data import (
    StereoDialogue,
    Turn,
    Utterance,
    read_jsonl,
    validate_manifest,
    write_jsonl,
)


def test_utterance_rejects_nonpositive_duration():
    with pytest.raises(ValidationError):
        Utterance(audio_path="a.wav", text="x", speaker_id="s", duration_s=0, sample_rate=16000)


def test_turn_rejects_end_before_start():
    with pytest.raises(ValidationError):
        Turn(speaker_id="s", text="x", start_s=1.0, end_s=0.5, channel=0)


def test_turn_rejects_out_of_range_channel():
    with pytest.raises(ValidationError):
        Turn(speaker_id="s", text="x", start_s=0.0, end_s=1.0, channel=2)


def test_jsonl_roundtrip_preserves_urdu(tmp_path):
    items = [
        Utterance(
            audio_path="a.wav",
            text="السلام علیکم",
            speaker_id="s1",
            duration_s=1.2,
            sample_rate=16000,
        ),
        Utterance(
            audio_path="b.wav",
            text="کیا حال ہے",
            speaker_id="s2",
            duration_s=0.9,
            sample_rate=16000,
        ),
    ]
    path = tmp_path / "manifest.jsonl"
    assert write_jsonl(path, items) == 2

    # The Urdu must survive as real UTF-8, not as escaped \uXXXX.
    raw = path.read_text(encoding="utf-8")
    assert "السلام علیکم" in raw

    loaded = read_jsonl(path, Utterance)
    assert loaded == items


def test_read_jsonl_reports_line_number_on_bad_row(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"audio_path": "a.wav"}\n', encoding="utf-8")  # missing fields
    with pytest.raises(ValueError, match=r"bad\.jsonl:1:"):
        read_jsonl(path, Utterance)


def _dialogue() -> StereoDialogue:
    return StereoDialogue(
        audio_path="call.wav",
        duration_s=2.0,
        sample_rate=24000,
        turns=[
            Turn(speaker_id="agent", text="سلام", start_s=0.0, end_s=0.8, channel=0),
            Turn(speaker_id="user", text="جی", start_s=0.9, end_s=1.6, channel=1),
        ],
    )


def test_validate_clean_dialogue_has_no_problems():
    assert validate_manifest([_dialogue()]) == []


def test_validate_flags_missing_audio(tmp_path):
    problems = validate_manifest([_dialogue()], audio_root=tmp_path)
    assert any("not found" in p for p in problems)


def test_validate_flags_turn_past_clip_end():
    d = _dialogue()
    d.turns[1].end_s = 5.0  # past the 2.0s clip
    problems = validate_manifest([d])
    assert any("past clip end" in p for p in problems)


def test_validate_flags_same_channel_overlap():
    d = _dialogue()
    # Two turns on the same channel that overlap in time -> a speaker over themselves.
    d.turns = [
        Turn(speaker_id="agent", text="a", start_s=0.0, end_s=1.0, channel=0),
        Turn(speaker_id="agent", text="b", start_s=0.5, end_s=1.5, channel=0),
    ]
    problems = validate_manifest([d])
    assert any("overlaps" in p for p in problems)
