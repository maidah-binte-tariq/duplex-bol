"""Build the Urdu SentencePiece tokenizer that replaces Moshi's English one.

Why this is the highest-leverage step in Track A: Moshi reads text through a
tokenizer trained on English. Feed it Nastaliq and it shatters every word into
byte-fallback fragments, so the model never sees coherent Urdu sub-words and the
fine-tune fights the vocabulary the whole way. J-Moshi's win for Japanese came
mostly from swapping this piece. We do the same for Urdu — and we normalize the
corpus first (same normalizer the eval uses), so the tokenizer learns one
canonical spelling instead of three.

``sentencepiece`` is optional; importing this module is fine without it, calling
the training function without it raises a clear install hint.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

from duplex_bol.text import UrduNormalizer


def _require_spm() -> ModuleType:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "sentencepiece is needed to train/load a tokenizer. "
            "Install the extra:  pip install 'duplex-bol[moshi]'"
        ) from exc
    module: ModuleType = spm  # sentencepiece ships no stubs; pin the type here
    return module


def prepare_corpus(texts: Iterable[str], out_path: str | Path, *, normalize: bool = True) -> int:
    """Write one normalized transcript per line — the input SentencePiece trains on.

    Returns the number of non-empty lines written.
    """
    normalizer = UrduNormalizer() if normalize else None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for text in texts:
            line = normalizer(text) if normalizer else text.strip()
            if line:
                fh.write(line + "\n")
                written += 1
    return written


def train_urdu_tokenizer(
    corpus_path: str | Path,
    model_prefix: str | Path,
    *,
    vocab_size: int = 8000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    byte_fallback: bool = True,
    hard_vocab_limit: bool = True,
) -> Path:
    """Train a SentencePiece model on a one-line-per-sentence corpus.

    Defaults match the J-Moshi-style setup: a unigram model, high character
    coverage (Nastaliq has a long tail of rare ligatures you don't want dropped),
    and byte fallback so unseen glyphs stay representable instead of collapsing to
    ``<unk>``. Note byte fallback adds 256 byte pieces, so a real ``vocab_size`` is
    in the thousands; the test path turns it off to train a tiny model. Returns the
    path to the ``.model`` file.
    """
    spm = _require_spm()
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        # Soft cap lets a small corpus train to *whatever* vocab it can support
        # instead of erroring out; real runs leave it hard.
        hard_vocab_limit=hard_vocab_limit,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )
    return Path(f"{model_prefix}.model")


class UrduTokenizer:
    """Thin wrapper over a trained SentencePiece model: encode / decode / size."""

    def __init__(self, model_path: str | Path) -> None:
        spm = _require_spm()
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str) -> list[int]:
        return list(self._sp.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return str(self._sp.decode(ids))

    @property
    def vocab_size(self) -> int:
        return int(self._sp.get_piece_size())
