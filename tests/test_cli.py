"""CLI smoke tests via Typer's runner. Cheap, but they catch wiring breakage."""

from __future__ import annotations

from typer.testing import CliRunner

from duplex_bol.cli import app

runner = CliRunner()

_TSV = (
    "client_id\tpath\tsentence\tgender\n"
    "s1\ta.mp3\tجملہ\tmale_masculine\n"
    "s1\tb.mp3\tجملہ\tmale_masculine\n"
    "s2\tc.mp3\tجملہ\tfemale_feminine\n"
)


def test_text_normalize_folds_yeh():
    result = runner.invoke(app, ["text", "normalize", "کيا"])  # ARABIC YEH
    assert result.exit_code == 0
    assert "کیا" in result.output  # FARSI YEH


def test_eval_wer_reports_rate():
    result = runner.invoke(app, ["eval", "wer", "--ref", "a b c", "--hyp", "a b"])
    assert result.exit_code == 0
    assert "WER" in result.output


def test_demo_runs_and_passes_budget():
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "BargeIn" in result.output
    assert "PASS" in result.output


def test_demo_without_bargein_finishes_speech():
    result = runner.invoke(app, ["demo", "--no-barge-in"])
    assert result.exit_code == 0
    assert "SpeechEnded" in result.output
    assert "BargeIn" not in result.output


def test_moshi_init_config_writes_file(tmp_path):
    out = tmp_path / "moshi_lora.yaml"
    result = runner.invoke(app, ["moshi", "init-config", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "VRAM" in result.output


def test_data_select_speakers(tmp_path):
    tsv = tmp_path / "validated.tsv"
    tsv.write_text(_TSV, encoding="utf-8")
    result = runner.invoke(app, ["data", "select-speakers", "--tsv", str(tsv), "--n", "2"])
    assert result.exit_code == 0
    assert "s1" in result.output and "s2" in result.output
