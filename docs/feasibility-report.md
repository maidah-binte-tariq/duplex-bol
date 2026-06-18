# Feasibility report: a one-week full-duplex Urdu voice agent

This is the *why* behind the code: which models, which datasets, what hardware, and
the licensing traps. It's written to be skimmed by a reviewer deciding whether the POC
is worth a week. Figures confirmed against primary sources are marked; the rest are
labeled estimates on purpose.

## The goal

A caller picks up the phone, hears an AI that sounds like a real Pakistani person
speaking Urdu, and can interrupt it mid-sentence. The hard part is the interruption:
the agent must keep listening while it talks — **full-duplex** — instead of taking
strict turns.

## Why two tracks

Two true things shape everything:

- **A fully fluent end-to-end Urdu full-duplex model in one week is not realistic.**
  The best open model of this kind (Moshi) was trained from scratch on ~1,000 GPUs,
  and there is no two-party Urdu phone corpus to adapt it on.
- **A working agent that *feels* full-duplex in one week is realistic** — via a
  streaming cascade with fast barge-in.

So: **Track B** (cascade) is the safe bet that demos by Friday; **Track A** (Moshi)
is the ambitious bet that proves the genuine full-duplex path. Judge each on its own.

## Model survey

| Model | Full-duplex? | Urdu? | License (weights) | Verdict for this POC |
|---|---|---|---|---|
| **Moshi** (Kyutai) | **Yes** (native) | No — adapt it | **CC-BY-4.0** (code MIT + Apache-2.0) | **Track A base.** Commercial-friendly weights; ~24 GB to run. |
| J-Moshi (nu-dialogue) | Yes | Japanese | **CC-BY-NC-4.0** ⚠ | The *recipe* to copy (tokenizer swap + LoRA), not the weights, if commercial. |
| Qwen3-Omni | No | **Input only** ⚠ | Apache-2.0 | Can be the **ear**; it understands Urdu but **cannot speak it**. |
| LFM2 / LFM2.5-Audio | Streaming | Limited | open | Watch; 1.5B is lighter than Moshi. (LFM2 vs **LFM2.5** are distinct releases.) |
| CSM-1B (Sesame) | No (TTS) | No native Urdu | Apache-2.0 | A "mouth" only — needs a brain in front. |

Key correction worth flagging because it's easy to get wrong: **Moshi's weights are
CC-BY-4.0 and its code is MIT/Apache-2.0** — two licenses on two things, both
commercial-friendly. It is J-Moshi's *released weights* that are CC-BY-NC.

- Moshi: <https://github.com/kyutai-labs/moshi> · weights
  [moshiko](https://huggingface.co/kyutai/moshiko-pytorch-bf16) (M) /
  [moshika](https://huggingface.co/kyutai/moshika-pytorch-bf16) (F)
- Fine-tune toolkit (LoRA): <https://github.com/kyutai-labs/moshi-finetune> ·
  multilingual fork: <https://github.com/nu-dialogue/moshi-finetune>

## Datasets

| Dataset | Size | Speakers | License | Role |
|---|---|---|---|---|
| Mozilla 3-speaker Urdu TTS | ~10 h | 3 (2 M / 1 F) | CC-BY-NC-4.0 ⚠ | core training voices |
| Common Voice Urdu | *[est. ~45 h validated]* | hundreds | **CC0** | speaker variety, ASR eval |
| Urdu-NSW | ~234 h | 1 (synthetic) | Apache-2.0 | casual rhythm only |
| UrduSER (optional) | 3,500 clips | 10 actors | research | emotion/dialect color |
| CLE / ELRA-S0403 | *[est., not on page]* | — | **paid (€12k+)** ⚠ | skip for the sprint |

Urdu TTS voices to reuse as the "mouth":
[Orpheus-Urdu](https://huggingface.co/mahwizzzz/orpheus-urdu-tts),
[SpeechT5-Urdu](https://huggingface.co/TheUpperCaseGuy/Guy-Urdu-TTS).

> **Estimates, on purpose.** The "~45 h" Common Voice figure comes from a survey, not
> Mozilla's own numbers; the CLE corpus hours aren't on its official page. Read the
> primary source before quoting either. Confirmed items (Moshi's license and VRAM, the
> 3-speaker set's count/format, Qwen3-Omni's Urdu-input-only limit) are stated plainly
> above.

## Hardware

| | Track B (cascade) | Track A (Moshi) |
|---|---|---|
| Min VRAM | ~16 GB (free Kaggle T4 / P100) | **~24 GB just to run** → rent an A100 |
| Where | free Kaggle, or one modest GPU | RunPod / Lightning / vast.ai (~$1.5–3/hr) |
| The real bottleneck | none — runs today | **data**, not compute: you must synthesize the two-party Urdu audio first |

`duplex-bol moshi vram --config configs/moshi_lora.yaml` gives a back-of-envelope
estimate (~16 GB with gradient checkpointing on) so you can sanity-check *before*
renting.

## Acceptance criteria (test-if-true hypotheses)

Each is a bet. Pass = proven; fail = a real finding, not a broken plan.

| # | Hypothesis | Pass condition |
|---|---|---|
| **H1** | A full-duplex model runs on rentable hardware | Moshi holds an interruptible English exchange on a 24 GB GPU |
| **H2** | The Urdu adaptation path is real | After tokenizer swap + LoRA, Moshi emits Urdu-sounding speech by day 4 |
| **H3** | The cascade talks Urdu | Caller speaks Urdu, bot replies in Urdu, on a recording, by Friday |
| **H4** | It feels full-duplex | Bot goes quiet within ~0.3–0.5 s of the caller barging in |
| **H5** | It's fast enough | Cascade reply starts < ~1 s after the caller stops |
| **H6** | The ear is good enough | WER on Common Voice Urdu ≤ ~30 % |
| **H7** | The 3-speaker data is usable | Fine-tuned voice clearly reflects the dataset speaker |

**Go / no-go:** if H1, H3, H4, H5 pass, the POC is a success; H3 and H4 are
make-or-break. H2 passing means the genuine full-duplex path is worth funding. H6/H7
failing is fixable with more data, not a reason to stop.

H4 and H5 are checked in code today against the same thresholds — see the latency
budget in [`eval/latency.py`](../src/duplex_bol/eval/latency.py) and run `make demo`.

## Licensing summary (read before shipping)

- **Clean for commercial use:** Moshi weights (CC-BY-4.0) + code (MIT/Apache-2.0),
  Common Voice (CC0), CSM-1B / Urdu-NSW (Apache-2.0).
- **Non-commercial — POC only:** the Mozilla 3-speaker set (CC-BY-NC) and J-Moshi's
  released weights (CC-BY-NC). To go commercial, re-run the recipe from Moshi's own
  weights and rebuild the corpus from Common Voice.
- **Paid / skip:** CLE (ELRA), ARL (LDC).
