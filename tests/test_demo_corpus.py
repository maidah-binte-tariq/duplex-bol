"""End-to-end: the synthetic-corpus script must emit valid, loadable manifests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from duplex_bol.data import StereoDialogue, Utterance, read_jsonl, validate_manifest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_corpus.py"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return out


@pytest.mark.slow
def test_track_b_manifest_is_valid(corpus: Path):
    utterances = read_jsonl(corpus / "trackB" / "utterances.jsonl", Utterance)
    assert len(utterances) == 6  # 2 speakers x 3 lines
    # Every referenced WAV must actually exist on disk.
    assert validate_manifest(utterances, audio_root=corpus) == []


@pytest.mark.slow
def test_track_a_stereo_call_is_valid(corpus: Path):
    dialogues = read_jsonl(corpus / "trackA" / "dialogues.jsonl", StereoDialogue)
    assert len(dialogues) == 1
    call = dialogues[0]
    assert call.sample_rate == 24_000
    assert len(call.turns) == 6
    assert {t.channel for t in call.turns} == {0, 1}  # genuinely two-channel
    assert validate_manifest(dialogues, audio_root=corpus) == []


@pytest.mark.slow
def test_moshi_index_written(corpus: Path):
    lines = (corpus / "trackA" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"path"' in lines[0] and '"duration"' in lines[0]
