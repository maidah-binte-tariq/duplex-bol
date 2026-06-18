"""LoRA + fine-tune configuration for the Moshi run, plus a VRAM sanity check.

LoRA (Low-Rank Adaptation) trains a few small add-on matrices instead of the whole
7B-parameter model, which is the only reason this fits on a single rented A100
rather than a cluster. The defaults mirror the moshi-finetune example config
(rank 128, the LoRA scaling, gradient checkpointing on).

The config knows how to (de)serialize to YAML so it round-trips with the
``configs/moshi_lora.yaml`` the toolkit reads, and :func:`estimate_vram_gb` gives a
back-of-envelope go/no-go before you spend money on a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from duplex_bol.data.manifest import StereoDialogue


@dataclass
class LoraConfig:
    rank: int = 128
    scaling: float = 2.0
    ff: bool = True  # also adapt the feed-forward blocks, not just attention


@dataclass
class MoshiFinetuneConfig:
    """Everything the fine-tune needs that isn't the data itself."""

    base_model: str = "kyutai/moshiko-pytorch-bf16"  # CC-BY-4.0 weights
    tokenizer_path: str | None = None  # the swapped-in Urdu SentencePiece model
    data_index: str = "data/trackA/index.jsonl"
    duration_sec: float = 100.0  # length of each training window
    batch_size: int = 16
    max_steps: int = 2000
    learning_rate: float = 2e-5
    gradient_checkpointing: bool = True
    seed: int = 0
    lora: LoraConfig = field(default_factory=LoraConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> MoshiFinetuneConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        lora = LoraConfig(**data.pop("lora", {}))
        return cls(lora=lora, **data)

    def validate(self) -> list[str]:
        """Return human-readable config problems (empty == sane)."""
        problems: list[str] = []
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.max_steps < 1:
            problems.append("max_steps must be >= 1")
        if not (0 < self.learning_rate < 1):
            problems.append(f"learning_rate {self.learning_rate} looks off (expected 0 < lr < 1)")
        if self.lora.rank not in (8, 16, 32, 64, 128, 256):
            problems.append(f"unusual LoRA rank {self.lora.rank} (typical: 16-256, power of two)")
        if self.tokenizer_path is None:
            problems.append("tokenizer_path is unset — the Urdu vocab swap is the whole point")
        return problems


def estimate_vram_gb(config: MoshiFinetuneConfig, base_params_b: float = 7.0) -> float:
    """Rough training VRAM estimate in GB. Deliberately conservative.

    Moshi is ~7B params. In bf16 the frozen weights are ~2 bytes/param ≈ 14 GB.
    LoRA adds a small optimizer/gradient footprint that scales with rank and batch
    size. This is a guardrail to catch "this won't fit on a 24 GB card" *before*
    renting one, not a precise profiler — real usage depends on activation memory
    and sequence length.
    """
    weights_gb = base_params_b * 2.0  # bf16 frozen base
    # LoRA adapter + Adam states: ~tens of MB per rank unit, scaled by batch.
    adapter_gb = (config.lora.rank / 128.0) * (config.batch_size / 16.0) * 2.0
    activation_gb = 0.0 if config.gradient_checkpointing else 6.0
    return round(weights_gb + adapter_gb + activation_gb, 1)


def build_index(dialogues: list[StereoDialogue]) -> list[dict[str, Any]]:
    """Build the toolkit's index rows: one ``{path, duration}`` per stereo clip.

    Skips clips with no ``audio_path`` (the synthesizer leaves it blank until the
    WAV is written), so a half-prepared corpus fails loudly here rather than three
    minutes into training.
    """
    rows: list[dict[str, Any]] = []
    for d in dialogues:
        if not d.audio_path:
            raise ValueError("dialogue has no audio_path — write the WAV before indexing")
        rows.append({"path": d.audio_path, "duration": round(d.duration_s, 4)})
    return rows
