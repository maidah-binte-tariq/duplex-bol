#!/usr/bin/env python3
"""Run the duplex-bol benchmarks and emit real, reproducible numbers.

Two things are measured here, both honestly:

1. **Tokenizer fertility** — train the Urdu SentencePiece tokenizer on a held-in
   split of ``benchmarks/urdu_sentences.txt`` and measure tokens-per-word on the
   held-out split, against the byte-fallback baseline (how an Urdu-blind tokenizer
   actually degrades). This is the quantitative case for Track A's tokenizer swap.

2. **Barge-in / response latency** — sweep the debounce config and measure the
   stop and response latencies from real :class:`DuplexOrchestrator` runs. These
   are deterministic guarantees, not noisy samples: barge-in stop is bounded by the
   onset window by construction.

What is NOT here: ASR WER / TTS MOS from a trained model. Those need a GPU run
(see ``notebooks/``); fabricating them would be dishonest. The harness to compute
WER ships in ``duplex_bol.eval`` and is wired into the notebooks.

    make bench        # or: python scripts/run_benchmarks.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from duplex_bol.cascade import AudioFrame, BargeIn, DuplexOrchestrator
from duplex_bol.cascade.fakes import ChunkedTTS, RuleBasedAgent, ScriptedASR
from duplex_bol.cascade.orchestrator import BargeInDetector
from duplex_bol.cascade.vad import EnergyVAD
from duplex_bol.eval import byte_fallback_encode, measure_fertility
from duplex_bol.eval.latency import LatencyBudget
from duplex_bol.moshi import UrduTokenizer, prepare_corpus, train_urdu_tokenizer
from duplex_bol.text import normalize_urdu

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "urdu_sentences.txt"
FRAME_MS = 20.0


def _frames(pattern: str) -> list[AudioFrame]:
    s = np.full(320, 0.3, np.float32)
    q = np.zeros(320, np.float32)
    return [AudioFrame(s if c == "S" else q) for c in pattern]


def fertility_benchmark(tmp: Path) -> dict:
    lines = [
        normalize_urdu(x) for x in CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    split = int(len(lines) * 0.8)
    train, held_out = lines[:split], lines[split:]

    corpus_txt = tmp / "train.txt"
    prepare_corpus(train, corpus_txt)
    model = train_urdu_tokenizer(
        corpus_txt,
        tmp / "urdu",
        vocab_size=512,
        character_coverage=1.0,
        byte_fallback=False,
        hard_vocab_limit=False,
    )
    tok = UrduTokenizer(model)

    results = {
        "byte_fallback": measure_fertility(
            "byte-fallback (no Urdu vocab)", held_out, byte_fallback_encode
        ),
        "char": measure_fertility("character-level", held_out, lambda t: list(t)),
        "urdu_spm": measure_fertility(
            f"Urdu SentencePiece (vocab {tok.vocab_size})", held_out, tok.encode
        ),
    }
    return {
        "held_out_sentences": len(held_out),
        "held_out_words": results["byte_fallback"].n_words,
        "rows": {
            k: {
                "name": v.name,
                "tokens_per_word": round(v.tokens_per_word, 3),
                "chars_per_token": round(v.chars_per_token, 3),
            }
            for k, v in results.items()
        },
        "spm_vs_bytes_speedup": round(
            results["urdu_spm"].speedup_over(results["byte_fallback"]), 2
        ),
    }


def _barge_in_stop(onset_frames: int) -> float:
    pattern = "SSSSSS" + "." * 11 + "S" * 7 + "." * 5
    orch = DuplexOrchestrator(
        vad=EnergyVAD(),
        asr=ScriptedASR(["السلام علیکم", "نہیں رکو"]),
        agent=RuleBasedAgent(default="وعلیکم السلام جی فرمائیے میں سن رہا ہوں"),
        tts=ChunkedTTS(frames_per_word=3),
        bargein=BargeInDetector(onset_frames=onset_frames, hangover_frames=5),
        frame_duration_ms=FRAME_MS,
    )
    barge = next(e for e in orch.run(_frames(pattern)) if isinstance(e, BargeIn))
    return barge.stop_latency_ms


def _response_start(hangover_frames: int) -> float:
    pattern = "SSSSSS" + "." * (hangover_frames + 28)
    orch = DuplexOrchestrator(
        vad=EnergyVAD(),
        asr=ScriptedASR(["السلام علیکم"]),
        agent=RuleBasedAgent(default="وعلیکم السلام"),
        tts=ChunkedTTS(frames_per_word=3),
        bargein=BargeInDetector(onset_frames=3, hangover_frames=hangover_frames),
        frame_duration_ms=FRAME_MS,
    )
    list(orch.run(_frames(pattern)))
    return orch.tracker.summary()["response_start"]["p50"]


def latency_benchmark() -> dict:
    onset_sweep = {o: _barge_in_stop(o) for o in (2, 3, 4, 5)}
    hangover_sweep = {h: _response_start(h) for h in (3, 5, 7)}
    # default config vs budget
    tracker_default = DuplexOrchestrator(
        vad=EnergyVAD(),
        asr=ScriptedASR(["السلام علیکم", "نہیں رکو"]),
        agent=RuleBasedAgent(default="وعلیکم السلام جی فرمائیے میں سن رہا ہوں"),
        tts=ChunkedTTS(frames_per_word=3),
        frame_duration_ms=FRAME_MS,
    )
    list(tracker_default.run(_frames("SSSSSS" + "." * 11 + "S" * 7 + "." * 5)))
    report = LatencyBudget.voice_agent_default().evaluate(tracker_default.tracker)
    return {
        "barge_in_stop_ms_by_onset_frames": onset_sweep,
        "response_start_ms_by_hangover_frames": hangover_sweep,
        "default_within_budget": report.ok,
    }


def _figure(fert: dict, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping figure; `pip install duplex-bol[figures]`)")
        return
    rows = fert["rows"]
    names = [
        "byte-fallback\n(Moshi's English vocab)",
        "character\nlevel",
        "Urdu SentencePiece\n(this repo)",
    ]
    vals = [
        rows["byte_fallback"]["tokens_per_word"],
        rows["char"]["tokens_per_word"],
        rows["urdu_spm"]["tokens_per_word"],
    ]
    colors = ["#c0392b", "#6b7a89", "#2e6f9e"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(names, vals, color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.15,
            f"{v:.1f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#1b2a3a",
        )
    ax.set_ylabel("tokens per word  (lower is better)", fontsize=11)
    ax.set_title(
        f"Tokenizer fertility on held-out Urdu — the Urdu vocab is "
        f"{fert['spm_vs_bytes_speedup']}× leaner than byte fallback",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.grid(axis="y", color="#e6ebef", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def _print_table(fert: dict, lat: dict) -> None:
    print("\n## Tokenizer fertility  (held-out Urdu, lower is better)\n")
    print("| tokenizer | tokens/word | chars/token |")
    print("|---|---:|---:|")
    for r in fert["rows"].values():
        print(f"| {r['name']} | {r['tokens_per_word']:.2f} | {r['chars_per_token']:.2f} |")
    print(
        f"\n→ the Urdu tokenizer is **{fert['spm_vs_bytes_speedup']}× leaner** than byte fallback "
        f"({fert['held_out_words']} held-out words).\n"
    )
    print("## Barge-in stop latency by onset debounce\n")
    print("| onset_frames | barge-in stop (ms) |")
    print("|---:|---:|")
    for o, ms in lat["barge_in_stop_ms_by_onset_frames"].items():
        print(f"| {o} | {ms:.0f} |")
    print(f"\ndefault config within H4/H5 budget: **{lat['default_within_budget']}**")


def main() -> None:
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results.json")
    ap.add_argument(
        "--figure", type=Path, default=ROOT / "docs" / "assets" / "tokenizer_fertility.png"
    )
    args = ap.parse_args()

    print("running benchmarks (real measurements):")
    with tempfile.TemporaryDirectory() as td:
        fert = fertility_benchmark(Path(td))
    lat = latency_benchmark()
    _print_table(fert, lat)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"fertility": fert, "latency": lat}, ensure_ascii=False, indent=2)
    )
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    _figure(fert, args.figure)


if __name__ == "__main__":
    main()
