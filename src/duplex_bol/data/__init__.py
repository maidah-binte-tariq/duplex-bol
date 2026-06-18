"""Dataset engineering: manifests, speaker selection, two-party stereo synthesis."""

from __future__ import annotations

from duplex_bol.data.common_voice import (
    CVClip,
    count_by_speaker,
    read_cv_tsv,
    select_speakers,
)
from duplex_bol.data.manifest import (
    StereoDialogue,
    Turn,
    Utterance,
    read_jsonl,
    validate_manifest,
    write_jsonl,
)
from duplex_bol.data.stereo_dialogue import DialogueConfig, SpeakerClip, build_dialogue

__all__ = [
    "CVClip",
    "DialogueConfig",
    "SpeakerClip",
    "StereoDialogue",
    "Turn",
    "Utterance",
    "build_dialogue",
    "count_by_speaker",
    "read_cv_tsv",
    "read_jsonl",
    "select_speakers",
    "validate_manifest",
    "write_jsonl",
]
