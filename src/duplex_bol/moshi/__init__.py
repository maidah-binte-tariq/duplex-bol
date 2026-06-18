"""Track A: the Moshi fine-tuning path (tokenizer swap + LoRA config).

Nothing here imports torch or moshi. It prepares the two things the fine-tune
needs before any GPU is involved: an Urdu SentencePiece tokenizer to replace the
English one (the single highest-leverage change, straight from the J-Moshi recipe)
and a validated training config. ``sentencepiece`` lives behind the ``[moshi]``
extra.
"""

from __future__ import annotations

from duplex_bol.moshi.lora import (
    LoraConfig,
    MoshiFinetuneConfig,
    build_index,
    estimate_vram_gb,
)
from duplex_bol.moshi.tokenizer import (
    UrduTokenizer,
    prepare_corpus,
    train_urdu_tokenizer,
)

__all__ = [
    "LoraConfig",
    "MoshiFinetuneConfig",
    "UrduTokenizer",
    "build_index",
    "estimate_vram_gb",
    "prepare_corpus",
    "train_urdu_tokenizer",
]
