# Data engineering

How a pile of downloaded Urdu audio becomes something a trainer will actually accept.
This is most of the real work in a one-week POC, and it's where the
[`duplex_bol.data`](../src/duplex_bol/data) and [`duplex_bol.audio`](../src/duplex_bol/audio)
modules earn their keep.

## The five things every speech trainer is fussy about

Every step below is just setting one of these correctly:

1. **File format** — downloads arrive as WEBM/MP3; trainers want WAV. Convert first.
2. **Sample rate** — each model demands one exact rate (Moshi 24 kHz; many TTS 16 or
   22.05 kHz). The wrong rate fails *silently* — training runs, output is garbage.
   `duplex_bol.audio.resample` handles it; match the model card exactly.
3. **Channels** — mono `(n,)` vs stereo `(n, 2)`. Track B wants mono; Track A wants a
   very specific stereo layout (below).
4. **The transcript** — the words, normalized so the same word is always spelled the
   same way. That's [`duplex_bol.text`](../src/duplex_bol/text); see why it matters
   below.
5. **The pairing** — the manifest linking each clip to its text. That's
   [`duplex_bol.data.manifest`](../src/duplex_bol/data/manifest.py).

## Where the data comes from

| Dataset | What it is | License | Use it for |
|---|---|---|---|
| [Mozilla 3-speaker Urdu](https://mozilladatacollective.com/datasets/cmmvykcrs0050ny07vkwww5gi) | ~10 h, 3 speakers (2 M / 1 F), read | **CC-BY-NC** | the core training voices |
| [Common Voice Urdu](https://github.com/common-voice/cv-dataset/tree/main/datasets/scripted-speech) | hundreds of speakers, read | **CC0** | speaker variety, ASR eval |
| [Urdu-NSW](https://huggingface.co/datasets/humairawan/Urdu-NSW) | ~234 h, one synthetic voice | Apache-2.0 | casual rhythm, not variety |

> The exact Common Voice Urdu hour count drifts per release — read the `ur` entry in
> the official `cv-dataset` JSON rather than trusting a survey figure. License and
> hours caveats are tracked in the [feasibility report](feasibility-report.md).

## Why normalization comes first

Urdu is written in the Arabic script and the *same word* arrives spelled differently
depending on the keyboard: ARABIC YEH (`U+064A`) vs FARSI YEH (`U+06CC`), ARABIC KAF
vs KEHEH, and so on. To a tokenizer those are different characters, so one word
fragments into several vocab entries and both ASR scoring and TTS training degrade.

[`UrduNormalizer`](../src/duplex_bol/text/urdu.py) folds these to one canonical
spelling. Two things it gets right that naive cleaners don't:

- **It keeps ZWNJ (`U+200C`).** The zero-width non-joiner is *content* in Urdu — it
  controls letter joining. Stripping it is the classic bug.
- **It keeps the hamza letters (ؤ ئ ء).** Those are real Urdu letters
  (آؤ، کوئی، جزء), not Arabic artifacts to fold away.

```python
from duplex_bol.text import normalize_urdu
normalize_urdu("کيا حال هے")   # "کیا حال ہے"  (Arabic yeh + heh folded)
```

## Track B — the cascade (do this first; it's the Friday demo)

Two small datasets, because the ear and the mouth want different things.

**The ear (ASR), from Common Voice.** You're mostly *checking* it, not retraining.

```python
from duplex_bol.data import read_cv_tsv, select_speakers

clips = read_cv_tsv("validated.tsv")          # validated = human-confirmed transcripts
chosen = select_speakers(clips, n=3)          # gender-mixed, most-clips-first, deterministic
```

Then convert each chosen clip to **16 kHz mono WAV**, normalize the transcripts, and
run your ASR over them. The mismatch is the word error rate — hypothesis **H6** wants
it ≤ ~30 %. `duplex_bol.eval.aggregate_wer(..., normalizer=normalize_urdu)` computes
it correctly (pooled over the corpus, scored on normalized text).

**The mouth (TTS), from the 3-speaker set.** This is what you actually fine-tune.
Unzip, convert WEBM → WAV at the rate your TTS wants, **split by speaker** (a TTS
learns one voice at a time), normalize transcripts, write a pairing manifest, and hold
back a few clips to judge the result (hypothesis **H7**).

```python
from duplex_bol.data import Utterance, write_jsonl, validate_manifest

manifest = [Utterance(audio_path="spk1/000.wav", text="...", speaker_id="spk1",
                      duration_s=3.1, sample_rate=16000)]
write_jsonl("trackB/utterances.jsonl", manifest)
problems = validate_manifest(manifest, audio_root="data/")   # [] means clean
```

## Track A — Moshi (the fiddly one)

Moshi learns from **two people talking at once, on separate stereo channels**: the
agent on the left, the user on the right, overlaps and all. That overlap is exactly
what teaches it to listen while it speaks.

The problem: **no two-party Urdu audio exists to download.** So we manufacture it —
the same move the J-Moshi team used for Japanese. "Single-voice clip" means one person
per file; you still need *two different speakers* across your clips, one per channel.

```python
from duplex_bol.data import SpeakerClip, DialogueConfig, build_dialogue

# one mono clip per turn; speaker 1 -> left channel, speaker 2 -> right
turns = [SpeakerClip("agent", a0, "السلام علیکم", 24000),
         SpeakerClip("user",  u0, "وعلیکم السلام", 24000), ...]

stereo, dialogue = build_dialogue(turns, DialogueConfig(overlap_s=0.15))
# stereo: float32 (n, 2);  dialogue: a StereoDialogue manifest with timestamped turns
```

A little `overlap_s` makes the channels overlap in time — deliberate, so the model
sees simultaneous speech. `build_dialogue` rejects more than two speakers (Moshi's
format is strictly two-party), resamples clips to the dialogue rate, and emits
per-turn timestamps. Then build the toolkit's index and the Urdu tokenizer:

```python
from duplex_bol.moshi import build_index, prepare_corpus, train_urdu_tokenizer

prepare_corpus(all_transcripts, "corpus.txt")          # normalized, one line each
train_urdu_tokenizer("corpus.txt", "trackA/urdu")      # the English->Urdu vocab swap
index = build_index([dialogue])                        # [{path, duration}, ...]
```

The tokenizer swap is the single highest-leverage step — see
[ADR-0002](decisions/0002-urdu-tokenizer-swap.md).

## A folder layout that keeps it straight

```
data/
  raw/                     # untouched downloads — never edit, always re-derive from here
    common_voice_ur/
    mozilla_3speaker/
  trackB/
    spk1_wav_16k/          # one voice, converted
    utterances.jsonl       # pairing manifest
    holdout/               # a few clips you did NOT train on (for H7)
  trackA/
    stereo_wav_24k/        # two-party calls, agent=left / user=right
    dialogues.jsonl        # StereoDialogue manifests (timestamped turns)
    index.jsonl            # {path, duration} the fine-tune toolkit reads
    urdu_tokenizer/        # the swapped-in SentencePiece model
```

Everything under `data/` is git-ignored — see the
[contributing guide](../CONTRIBUTING.md). Re-derive, don't commit.

## The golden rule: tiny slice first

Prepare 2–3 clips, push them through the trainer, confirm it *accepts the format*,
then scale. A wrong sample rate or a malformed manifest then costs you ten minutes
instead of a day. That's also why `validate_manifest` exists and why the manifests are
JSON Lines (greppable, line-numbered errors). `scripts/make_demo_corpus.py` is exactly
this tiny slice, fully synthetic, so you can dry-run the whole pipeline before any
download:

```bash
make demo-corpus     # writes a real but synthetic corpus to data/demo/
```
