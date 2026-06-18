"""Manifest schemas — the contract between data prep and the trainers.

Both tracks are picky about input shape, and a malformed manifest is the single
most common way to waste a GPU-hour: the run starts, then dies three minutes in
on row 4,000. So the manifests are validated *before* anything touches a GPU.

* :class:`Utterance`      — one single-speaker clip. Track B (cascade) ingests these.
* :class:`StereoDialogue` — one synthetic two-party call. Track A (Moshi) ingests these.

On disk both are JSON Lines (one JSON object per line): streamable, greppable,
and diff-friendly, which a single giant JSON array is not.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field, model_validator

ModelT = TypeVar("ModelT", bound=BaseModel)


class Utterance(BaseModel):
    """A single-speaker clip and its transcript."""

    audio_path: str
    text: str
    speaker_id: str
    duration_s: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    language: str = "ur"


class Turn(BaseModel):
    """One speaker turn inside a two-party dialogue, pinned to a stereo channel."""

    speaker_id: str
    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    channel: int = Field(ge=0, le=1)  # 0 = left, 1 = right

    @model_validator(mode="after")
    def _end_after_start(self) -> Turn:
        if self.end_s <= self.start_s:
            raise ValueError(f"turn end ({self.end_s}) must be after start ({self.start_s})")
        return self


class StereoDialogue(BaseModel):
    """A synthesized two-party call: two voices, two channels, timestamped turns."""

    audio_path: str
    duration_s: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    turns: list[Turn]


def write_jsonl(path: str | Path, items: Iterable[BaseModel]) -> int:
    """Write models as JSON Lines. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            # ensure_ascii would mangle Urdu into \uXXXX soup; we want real UTF-8.
            fh.write(item.model_dump_json())
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path, model: type[ModelT]) -> list[ModelT]:
    """Parse a JSON Lines manifest into ``model`` instances (blank lines skipped)."""
    out: list[ModelT] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return out


def validate_manifest(
    items: Iterable[Utterance | StereoDialogue], *, audio_root: str | Path | None = None
) -> list[str]:
    """Return a list of human-readable problems (empty list == clean manifest).

    Checks the things pydantic can't: that audio files actually exist on disk, and
    that a dialogue's turns stay inside the clip and don't overlap *on the same
    channel* (cross-channel overlap is allowed — that's the simultaneous speech
    Moshi learns from).
    """
    problems: list[str] = []
    root = Path(audio_root) if audio_root is not None else None

    for i, item in enumerate(items):
        tag = f"item[{i}] ({item.audio_path})"
        if root is not None and not (root / item.audio_path).exists():
            problems.append(f"{tag}: audio file not found under {root}")

        if isinstance(item, StereoDialogue):
            per_channel: dict[int, float] = {}
            for j, turn in enumerate(item.turns):
                if turn.end_s > item.duration_s + 1e-6:
                    problems.append(
                        f"{tag}: turn[{j}] ends at {turn.end_s:.3f}s, past clip end "
                        f"{item.duration_s:.3f}s"
                    )
                last_end = per_channel.get(turn.channel)
                if last_end is not None and turn.start_s + 1e-6 < last_end:
                    problems.append(
                        f"{tag}: turn[{j}] overlaps a previous turn on channel "
                        f"{turn.channel} (same speaker can't talk over themselves)"
                    )
                per_channel[turn.channel] = turn.end_s
    return problems
