"""Command-line entry point: ``duplex-bol <group> <command>``.

Groups mirror the package: ``text``, ``data``, ``eval``, ``moshi``, plus a
top-level ``demo`` that runs the fake cascade end-to-end so you can watch the
barge-in policy work without a GPU or a microphone. The demo is the fastest way to
convince yourself (or a reviewer) the orchestration is real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from duplex_bol._logging import setup_logging
from duplex_bol.cascade import AudioFrame, BargeIn, DuplexOrchestrator
from duplex_bol.cascade.fakes import ChunkedTTS, RuleBasedAgent, ScriptedASR
from duplex_bol.cascade.vad import EnergyVAD
from duplex_bol.data import count_by_speaker, read_cv_tsv, select_speakers
from duplex_bol.eval import LatencyBudget, word_error_rate
from duplex_bol.moshi import MoshiFinetuneConfig, estimate_vram_gb
from duplex_bol.text import normalize_urdu

app = typer.Typer(help="Tools for the full-duplex Urdu speech-to-speech POC.", no_args_is_help=True)
text_app = typer.Typer(help="Urdu text normalization.")
data_app = typer.Typer(help="Corpus engineering helpers.")
eval_app = typer.Typer(help="Scoring: WER/CER.")
moshi_app = typer.Typer(help="Track A (Moshi) config helpers.")
app.add_typer(text_app, name="text")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(moshi_app, name="moshi")


@app.callback()
def _main() -> None:
    setup_logging()


@text_app.command("normalize")
def text_normalize(text: str = typer.Argument(..., help="Urdu text to normalize")) -> None:
    """Print the normalized form of an Urdu string."""
    typer.echo(normalize_urdu(text))


@data_app.command("select-speakers")
def data_select_speakers(
    tsv: Path = typer.Option(..., exists=True, help="Common Voice validated.tsv"),
    n: int = typer.Option(3, help="How many speakers to keep"),
    balance_gender: bool = typer.Option(True, help="Mix genders in the selection"),
) -> None:
    """Pick N speakers out of a Common Voice TSV and report their clip counts."""
    clips = read_cv_tsv(tsv)
    chosen = select_speakers(clips, n=n, balance_gender=balance_gender)
    counts = count_by_speaker(clips)
    typer.echo(f"selected {len(chosen)} of {len(counts)} speakers:")
    for sid, sel in chosen.items():
        bucket = sel[0].gender_bucket
        typer.echo(f"  {sid}  clips={len(sel)}  gender={bucket}")


@eval_app.command("wer")
def eval_wer(
    ref: str = typer.Option(..., help="Reference (ground-truth) text"),
    hyp: str = typer.Option(..., help="Hypothesis (system output) text"),
    normalize: bool = typer.Option(True, help="Normalize Urdu before scoring"),
) -> None:
    """Word error rate between a reference and a hypothesis."""
    normalizer = normalize_urdu if normalize else None
    counts = word_error_rate(ref, hyp, normalizer=normalizer)
    typer.echo(
        f"WER {counts.error_rate:.3f}  (S={counts.substitutions} "
        f"D={counts.deletions} I={counts.insertions} N={counts.ref_length})"
    )


@moshi_app.command("init-config")
def moshi_init_config(
    out: Path = typer.Option(Path("configs/moshi_lora.yaml"), help="Where to write the config"),
    tokenizer: str = typer.Option("data/trackA/urdu_tokenizer/urdu.model"),
) -> None:
    """Write a default Moshi LoRA fine-tune config to YAML."""
    cfg = MoshiFinetuneConfig(tokenizer_path=tokenizer)
    cfg.to_yaml(out)
    typer.echo(f"wrote {out}  (estimated training VRAM ~{estimate_vram_gb(cfg)} GB)")


@moshi_app.command("vram")
def moshi_vram(config: Path = typer.Option(..., exists=True, help="moshi_lora.yaml")) -> None:
    """Estimate training VRAM for a saved config and flag config problems."""
    cfg = MoshiFinetuneConfig.from_yaml(config)
    typer.echo(f"estimated training VRAM ~{estimate_vram_gb(cfg)} GB")
    for problem in cfg.validate():
        typer.echo(f"  warning: {problem}")


def _pattern_frames(pattern: str, frame_samples: int = 320) -> list[AudioFrame]:
    speech = np.full(frame_samples, 0.3, np.float32)
    silence = np.zeros(frame_samples, np.float32)
    return [AudioFrame(speech if c == "S" else silence) for c in pattern]


@app.command("demo")
def demo(barge_in: bool = typer.Option(True, help="Include a caller interruption")) -> None:
    """Run the fake cascade end-to-end and print the event trace + latency budget."""
    orch = DuplexOrchestrator(
        vad=EnergyVAD(),
        asr=ScriptedASR(["السلام علیکم", "میں ٹھیک ہوں"]),
        agent=RuleBasedAgent(default="وعلیکم السلام، میں آپ کی کیا مدد کر سکتا ہوں"),
        tts=ChunkedTTS(frames_per_word=3),
    )
    # Caller speaks, pauses; bot replies and (optionally) gets interrupted mid-reply.
    # The no-interrupt case needs enough trailing silence for the full reply to play.
    pattern = "SSSS.....SSSSS........." if barge_in else "SSSS" + "." * 42
    for event in orch.run(_pattern_frames(pattern)):
        marker = "  <-- interrupted" if isinstance(event, BargeIn) else ""
        typer.echo(f"  [{event.frame_index:>3}] {type(event).__name__}{marker}")

    typer.echo("\nlatency budget (H4 barge-in <= 500ms, H5 response <= 1000ms):")
    report = LatencyBudget.voice_agent_default().evaluate(orch.tracker)
    typer.echo(str(report))
    typer.echo(f"\noverall: {'PASS' if report.ok else 'FAIL'}")


if __name__ == "__main__":  # pragma: no cover
    app()
