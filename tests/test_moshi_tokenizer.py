"""Urdu SentencePiece tokenizer training + the normalize-first corpus step.

Skips cleanly if the ``[moshi]`` extra (sentencepiece) isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentencepiece")

from duplex_bol.moshi import UrduTokenizer, prepare_corpus, train_urdu_tokenizer

_LINES = [
    "السلام علیکم آپ کیسے ہیں",
    "میں بالکل ٹھیک ہوں شکریہ",
    "آج موسم بہت اچھا ہے",
    "کیا آپ اردو بول سکتے ہیں",
    "جی ہاں میں اردو بولتا ہوں",
    "یہ ایک ٹیسٹ جملہ ہے",
] * 20  # repeated so SentencePiece has enough to seed pieces


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    prepare_corpus(_LINES, path)
    return path


def test_prepare_corpus_normalizes_text(tmp_path):
    # Arabic yeh in the input must be folded to Farsi yeh on disk.
    path = tmp_path / "c.txt"
    n = prepare_corpus(["کيا حال"], path)  # ARABIC YEH
    assert n == 1
    written = path.read_text(encoding="utf-8")
    assert "کیا" in written  # FARSI YEH
    assert "کي" not in written


def test_prepare_corpus_skips_blank_lines(tmp_path):
    path = tmp_path / "c.txt"
    assert prepare_corpus(["متن", "   ", ""], path) == 1


def _train(corpus, tmp_path):
    # Tiny-corpus settings: no byte fallback (its 256 pieces dwarf the data) and a
    # soft vocab cap so training can't error out on size. Real runs use the defaults.
    return train_urdu_tokenizer(
        corpus,
        tmp_path / "urdu",
        vocab_size=200,
        character_coverage=1.0,
        byte_fallback=False,
        hard_vocab_limit=False,
    )


def test_tokenizer_trains_and_is_loadable(corpus, tmp_path):
    model = _train(corpus, tmp_path)
    assert model.exists()
    tok = UrduTokenizer(model)
    assert tok.vocab_size > 4  # at least the special tokens + some pieces


def test_tokenizer_roundtrips_urdu(corpus, tmp_path):
    tok = UrduTokenizer(_train(corpus, tmp_path))
    text = "السلام علیکم"
    ids = tok.encode(text)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert tok.decode(ids).strip() == text
