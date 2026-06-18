#!/usr/bin/env python3
"""Generate a tiny synthetic corpus end-to-end — no downloads, no GPU.

This is the "does the data pipeline actually produce the right files" smoke test
you run on day 0. It fabricates a few single-speaker clips (distinct tones stand
in for distinct voices), writes the Track B utterance manifest, then stitches two
of them into a two-party stereo "call" and writes the Track A stereo manifest +
Moshi index. Open the resulting WAVs and JSONL and you can see exactly the shape
each trainer expects, with real bytes on disk.

    python scripts/make_demo_corpus.py --out data/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from duplex_bol.audio import write_wav
from duplex_bol.data import DialogueConfig, SpeakerClip, Utterance, build_dialogue, write_jsonl
from duplex_bol.moshi import build_index

# Speaker -> (tone frequency Hz, a line of Urdu). The tone is just so the channels
# are audibly distinct when you play the stereo file back.
_SPEAKERS = {
    "spk_agent": (180.0, ["السلام علیکم", "میں آپ کی کیا مدد کر سکتا ہوں", "بہت شکریہ"]),
    "spk_user": (240.0, ["وعلیکم السلام", "مجھے اپنا بیلنس جاننا ہے", "ٹھیک ہے"]),
}
_SR = 24_000


def _tone(freq: float, seconds: float = 1.2, sr: int = _SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    # gentle fade so the synthetic clips don't click at the edges
    env = np.minimum(1.0, np.minimum(t * 10, (seconds - t) * 10))
    return (0.4 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/demo"))
    args = ap.parse_args()
    out: Path = args.out

    # --- Track B: single-speaker utterances ----------------------------------
    utterances: list[Utterance] = []
    clips_by_speaker: dict[str, list[SpeakerClip]] = {}
    for sid, (freq, lines) in _SPEAKERS.items():
        clips_by_speaker[sid] = []
        for i, line in enumerate(lines):
            audio = _tone(freq)
            rel = f"trackB/{sid}/{i:03d}.wav"
            write_wav(out / rel, audio, _SR)
            utterances.append(
                Utterance(
                    audio_path=rel,
                    text=line,
                    speaker_id=sid,
                    duration_s=len(audio) / _SR,
                    sample_rate=_SR,
                )
            )
            clips_by_speaker[sid].append(SpeakerClip(sid, audio, line, _SR))
    write_jsonl(out / "trackB/utterances.jsonl", utterances)

    # --- Track A: one synthetic two-party stereo call ------------------------
    agent, user = clips_by_speaker["spk_agent"], clips_by_speaker["spk_user"]
    turn_order = [agent[0], user[0], agent[1], user[1], agent[2], user[2]]
    stereo, dialogue = build_dialogue(turn_order, DialogueConfig(overlap_s=0.15))
    stereo_rel = "trackA/call_000.wav"
    write_wav(out / stereo_rel, stereo, dialogue.sample_rate)
    dialogue.audio_path = stereo_rel
    write_jsonl(out / "trackA/dialogues.jsonl", [dialogue])

    # The Moshi index is plain {path, duration} rows, not pydantic models.
    index_path = out / "trackA/index.jsonl"
    with index_path.open("w", encoding="utf-8") as fh:
        for row in build_index([dialogue]):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote demo corpus to {out}/")
    print(f"  Track B: {len(utterances)} utterances across {len(_SPEAKERS)} speakers")
    print(
        f"  Track A: 1 stereo call, {len(dialogue.turns)} turns, "
        f"{dialogue.duration_s:.1f}s, agent=left / user=right"
    )


if __name__ == "__main__":
    main()
