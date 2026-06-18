"""Pick a clean handful of speakers out of a Common Voice TSV.

Common Voice ships ``validated.tsv`` — a few hundred Urdu speakers jumbled
together, keyed by an opaque ``client_id``. For the POC we want a *small, known*
set (the report's "3 speakers, mixed gender"), so this module groups by speaker,
counts clips, and selects a gender-balanced top-N deterministically.

Stdlib ``csv`` only — pandas would be a heavy dependency for what is, in the end,
a group-by and a sort.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Common Voice column names have drifted across releases; ``gender`` became
# ``gender`` → values like "male_masculine". We read defensively.
_GENDER_KEYS = ("gender", "sex")


@dataclass(frozen=True)
class CVClip:
    path: str
    sentence: str
    client_id: str
    gender: str | None = None
    age: str | None = None

    @property
    def gender_bucket(self) -> str:
        """Collapse the various release-specific gender strings to male/female/other."""
        g = (self.gender or "").lower()
        if g.startswith("male") or g == "m":
            return "male"
        if g.startswith("female") or g == "f":
            return "female"
        return "other"


def read_cv_tsv(path: str | Path) -> list[CVClip]:
    """Read a Common Voice ``*.tsv`` into :class:`CVClip` rows."""
    clips: list[CVClip] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gender = next((row[k] for k in _GENDER_KEYS if row.get(k)), None)
            clips.append(
                CVClip(
                    path=row.get("path", ""),
                    sentence=row.get("sentence", ""),
                    client_id=row.get("client_id", ""),
                    gender=gender,
                    age=row.get("age") or None,
                )
            )
    return clips


def count_by_speaker(clips: list[CVClip]) -> dict[str, int]:
    """Map ``client_id`` → clip count."""
    counts: dict[str, int] = defaultdict(int)
    for clip in clips:
        counts[clip.client_id] += 1
    return dict(counts)


def select_speakers(
    clips: list[CVClip],
    *,
    n: int = 3,
    balance_gender: bool = True,
    min_clips: int = 1,
) -> dict[str, list[CVClip]]:
    """Select ``n`` speakers and return their clips, keyed by ``client_id``.

    Speakers with fewer than ``min_clips`` clips are dropped first (you cannot
    fine-tune a voice on three utterances). With ``balance_gender`` we interleave
    the male and female pools so the result isn't three speakers of the same
    gender — useful when the demo needs distinct agent/user voices.

    Deterministic: ties break on ``client_id`` so the same TSV always yields the
    same pick (reproducible corpora matter).
    """
    by_speaker: dict[str, list[CVClip]] = defaultdict(list)
    for clip in clips:
        by_speaker[clip.client_id].append(clip)

    eligible = {sid: cl for sid, cl in by_speaker.items() if len(cl) >= min_clips}

    def rank(sid: str) -> tuple[int, str]:
        # Most clips first; client_id as a stable tiebreaker (negate count for desc).
        return (-len(eligible[sid]), sid)

    if not balance_gender:
        top = sorted(eligible, key=rank)[:n]
        return {sid: eligible[sid] for sid in top}

    # Split into gender pools, then round-robin so the pick stays mixed.
    pools: dict[str, list[str]] = defaultdict(list)
    for sid in sorted(eligible, key=rank):
        bucket = eligible[sid][0].gender_bucket
        pools[bucket].append(sid)

    order = ["female", "male", "other"]  # start with the usually-scarcer pool
    chosen: list[str] = []
    cursors = dict.fromkeys(pools, 0)
    while len(chosen) < n and any(cursors[k] < len(pools[k]) for k in pools):
        for bucket in order:
            if len(chosen) >= n:
                break
            if cursors.get(bucket, 0) < len(pools.get(bucket, [])):
                chosen.append(pools[bucket][cursors[bucket]])
                cursors[bucket] += 1
    return {sid: eligible[sid] for sid in chosen}
