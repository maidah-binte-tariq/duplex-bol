"""Tokenizer fertility metric — and the real comparison it's built to make."""

from __future__ import annotations

import pytest

from duplex_bol.eval import byte_fallback_encode, measure_fertility


def test_byte_fallback_is_one_token_per_utf8_byte():
    # "اب" is 2 Nastaliq chars, 4 UTF-8 bytes -> a vocab-less tokenizer spends 4 tokens.
    assert byte_fallback_encode("اب") == list("اب".encode())
    r = measure_fertility("bytes", ["اب"], byte_fallback_encode)
    assert (r.n_words, r.n_tokens, r.n_chars) == (1, 4, 2)
    assert r.tokens_per_word == 4.0
    assert r.chars_per_token == 0.5


def test_measure_fertility_with_word_tokenizer():
    # A toy "1 token per word" encoder -> fertility is exactly 1.0.
    enc = lambda t: list(range(len(t.split())))  # noqa: E731
    r = measure_fertility("words", ["a b c", "d e"], enc)
    assert r.n_words == 5 and r.n_tokens == 5
    assert r.tokens_per_word == 1.0


def test_speedup_over_baseline():
    good = measure_fertility("good", ["a b"], lambda t: [0, 1])  # 2 tokens / 2 words = 1.0
    bad = byte_fallback_encode  # bytes
    baseline = measure_fertility("bytes", ["a b"], bad)  # "a b" = 3 bytes / 2 words = 1.5
    assert good.speedup_over(baseline) == pytest.approx(1.5)


def test_empty_inputs_do_not_divide_by_zero():
    r = measure_fertility("empty", [], byte_fallback_encode)
    assert r.tokens_per_word == 0.0
    assert r.chars_per_token == 0.0


@pytest.mark.slow
def test_trained_urdu_tokenizer_beats_byte_fallback(tmp_path):
    # The thesis, as an assertion: a trained Urdu tokenizer must spend fewer tokens
    # per word on held-out Urdu than the byte-fallback baseline.
    pytest.importorskip("sentencepiece")
    from duplex_bol.moshi import UrduTokenizer, prepare_corpus, train_urdu_tokenizer

    train = [
        "السلام علیکم آپ کیسے ہیں",
        "میں اپنا اکاؤنٹ بیلنس جاننا چاہتا ہوں",
        "براہ مہربانی تھوڑا انتظار کریں",
        "شکریہ آپ کا دن اچھا گزرے",
        "مجھے ایک نیا کارڈ چاہیے",
    ] * 8
    held_out = ["آپ کی بات سمجھ گئی میں ابھی دیکھتا ہوں"]

    corpus = tmp_path / "c.txt"
    prepare_corpus(train, corpus)
    model = train_urdu_tokenizer(
        corpus,
        tmp_path / "ur",
        vocab_size=200,
        character_coverage=1.0,
        byte_fallback=False,
        hard_vocab_limit=False,
    )
    tok = UrduTokenizer(model)

    spm = measure_fertility("urdu-spm", held_out, tok.encode)
    bytes_ = measure_fertility("bytes", held_out, byte_fallback_encode)
    assert spm.tokens_per_word < bytes_.tokens_per_word
