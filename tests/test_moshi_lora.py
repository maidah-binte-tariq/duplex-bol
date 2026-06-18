"""Moshi fine-tune config: serialization, validation, VRAM guardrail, indexing."""

from __future__ import annotations

import pytest

from duplex_bol.data import StereoDialogue, Turn
from duplex_bol.moshi import (
    LoraConfig,
    MoshiFinetuneConfig,
    build_index,
    estimate_vram_gb,
)


def test_defaults_match_finetune_example():
    cfg = MoshiFinetuneConfig()
    assert cfg.lora.rank == 128
    assert cfg.base_model.startswith("kyutai/")


def test_yaml_roundtrip(tmp_path):
    cfg = MoshiFinetuneConfig(tokenizer_path="urdu.model", max_steps=1500, lora=LoraConfig(rank=64))
    path = tmp_path / "moshi_lora.yaml"
    cfg.to_yaml(path)
    assert MoshiFinetuneConfig.from_yaml(path) == cfg


def test_yaml_is_unicode_not_escaped(tmp_path):
    # allow_unicode so the config stays human-readable; no \uXXXX soup.
    text = MoshiFinetuneConfig(data_index="ڈیٹا/index.jsonl").to_yaml()
    assert "ڈیٹا" in text


def test_validate_flags_missing_tokenizer():
    problems = MoshiFinetuneConfig().validate()
    assert any("tokenizer_path" in p for p in problems)


def test_validate_clean_config_passes():
    cfg = MoshiFinetuneConfig(tokenizer_path="urdu.model")
    assert cfg.validate() == []


def test_validate_catches_bad_hyperparams():
    cfg = MoshiFinetuneConfig(tokenizer_path="t.model", batch_size=0, learning_rate=5.0)
    problems = cfg.validate()
    assert any("batch_size" in p for p in problems)
    assert any("learning_rate" in p for p in problems)


def test_vram_estimate_fits_24gb_with_checkpointing():
    cfg = MoshiFinetuneConfig(tokenizer_path="t.model", gradient_checkpointing=True)
    assert estimate_vram_gb(cfg) <= 24.0  # the whole point: fit a single A100/4090


def test_vram_estimate_grows_without_checkpointing():
    base = MoshiFinetuneConfig(tokenizer_path="t.model", gradient_checkpointing=True)
    heavy = MoshiFinetuneConfig(tokenizer_path="t.model", gradient_checkpointing=False)
    assert estimate_vram_gb(heavy) > estimate_vram_gb(base)


def _dialogue(path: str) -> StereoDialogue:
    return StereoDialogue(
        audio_path=path,
        duration_s=3.0,
        sample_rate=24000,
        turns=[Turn(speaker_id="a", text="x", start_s=0.0, end_s=1.0, channel=0)],
    )


def test_build_index_rows():
    rows = build_index([_dialogue("a.wav"), _dialogue("b.wav")])
    assert rows == [
        {"path": "a.wav", "duration": 3.0},
        {"path": "b.wav", "duration": 3.0},
    ]


def test_build_index_rejects_unwritten_clip():
    with pytest.raises(ValueError, match="audio_path"):
        build_index([_dialogue("")])
