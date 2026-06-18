"""Manufacture two-party stereo "calls" from single-speaker clips.

This is the load-bearing trick of Track A. Moshi learns from stereo audio where
one speaker is on the left channel and the other is on the right, overlaps and
all. No such two-party *Urdu* corpus exists, so we build one: take single-speaker
clips, assign two speakers to the two channels, and lay them out on a shared
timeline with realistic gaps (and, optionally, deliberate overlap so the model
sees simultaneous speech — the thing that makes full-duplex full-duplex).

The output is a stereo waveform plus a :class:`~duplex_bol.data.manifest.StereoDialogue`
with per-turn timestamps, which is exactly what the fine-tuning toolkit reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from duplex_bol.audio.transforms import resample, to_mono
from duplex_bol.data.manifest import StereoDialogue, Turn

Audio = npt.NDArray[np.float32]

LEFT, RIGHT = 0, 1


@dataclass(frozen=True)
class DialogueConfig:
    """Timeline shape for a synthesized call.

    ``overlap_s`` is the interesting knob: a positive value starts each turn
    *before* the previous one finishes, so the two channels overlap in time. A
    little overlap mimics natural barge-in; too much turns it into a shouting
    match, so keep it small (≤ ~0.4 s).
    """

    sample_rate: int = 24_000  # Moshi's rate
    inter_turn_gap_s: float = 0.3
    overlap_s: float = 0.0
    lead_in_s: float = 0.0


@dataclass
class SpeakerClip:
    """One single-speaker clip headed into a dialogue."""

    speaker_id: str
    audio: Audio
    text: str
    sample_rate: int

    def __post_init__(self) -> None:
        self.audio = to_mono(np.asarray(self.audio, dtype=np.float32))


@dataclass
class _Placement:
    channel: int
    start_sample: int
    audio: Audio
    text: str
    speaker_id: str


@dataclass
class _Timeline:
    config: DialogueConfig
    placements: list[_Placement] = field(default_factory=list)


def _assign_channels(clips: list[SpeakerClip]) -> dict[str, int]:
    """First distinct speaker → left, second → right. Reject anything beyond two.

    Moshi's stereo format is strictly two-party; if you hand this three speakers
    it's a data-prep bug, so fail loudly rather than silently dropping a voice.
    """
    speakers: list[str] = []
    for clip in clips:
        if clip.speaker_id not in speakers:
            speakers.append(clip.speaker_id)
    if len(speakers) > 2:
        raise ValueError(
            f"a stereo dialogue is two-party; got {len(speakers)} speakers: {speakers}"
        )
    return {sid: idx for idx, sid in enumerate(speakers)}


def build_dialogue(
    clips: list[SpeakerClip], config: DialogueConfig | None = None
) -> tuple[Audio, StereoDialogue]:
    """Lay ``clips`` out as one two-party stereo call.

    ``clips`` is the turn order (clip 0 is spoken first). Returns the stereo
    waveform (shape ``(n, 2)``, float32) and the matching manifest entry. The
    ``audio_path`` on the returned manifest is left blank — the caller writes the
    WAV and fills it in, so this function stays pure and trivially testable.
    """
    if not clips:
        raise ValueError("need at least one clip to build a dialogue")
    cfg = config or DialogueConfig()
    channel_of = _assign_channels(clips)
    sr = cfg.sample_rate

    gap = round(cfg.inter_turn_gap_s * sr)
    overlap = round(cfg.overlap_s * sr)
    cursor = round(cfg.lead_in_s * sr)

    placements: list[_Placement] = []
    for clip in clips:
        audio = clip.audio
        if clip.sample_rate != sr:
            audio = resample(audio, clip.sample_rate, sr)
        start = max(0, cursor)
        placements.append(
            _Placement(
                channel=channel_of[clip.speaker_id],
                start_sample=start,
                audio=audio,
                text=clip.text,
                speaker_id=clip.speaker_id,
            )
        )
        # Next turn starts after this clip + gap, pulled back by the overlap.
        cursor = start + len(audio) + gap - overlap

    total = max((p.start_sample + len(p.audio) for p in placements), default=0)
    stereo = np.zeros((total, 2), dtype=np.float32)
    turns: list[Turn] = []
    for p in placements:
        end = p.start_sample + len(p.audio)
        # Additive mix: lets cross-channel overlap coexist without clobbering.
        stereo[p.start_sample : end, p.channel] += p.audio
        turns.append(
            Turn(
                speaker_id=p.speaker_id,
                text=p.text,
                start_s=p.start_sample / sr,
                end_s=end / sr,
                channel=p.channel,
            )
        )

    dialogue = StereoDialogue(
        audio_path="",  # caller fills this after writing the WAV
        duration_s=total / sr,
        sample_rate=sr,
        turns=turns,
    )
    return stereo, dialogue
