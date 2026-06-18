"""WER / CER alignment counts, including the Urdu-normalized scoring path."""

from __future__ import annotations

from duplex_bol.eval import aggregate_wer, character_error_rate, word_error_rate
from duplex_bol.text import normalize_urdu


def test_identical_strings_have_no_errors():
    counts = word_error_rate("aap kaise hain", "aap kaise hain")
    assert counts.errors == 0
    assert counts.error_rate == 0.0
    assert counts.hits == 3


def test_single_substitution():
    counts = word_error_rate("aap kaise hain", "aap acche hain")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 0)
    assert counts.error_rate == 1 / 3


def test_single_deletion():
    counts = word_error_rate("aap kaise hain", "aap hain")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 0)


def test_single_insertion():
    counts = word_error_rate("aap hain", "aap bilkul hain")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)


def test_empty_reference_with_hypothesis_is_full_error():
    counts = word_error_rate("", "kuch to hai")
    assert counts.insertions == 3
    assert counts.error_rate == 1.0


def test_both_empty_is_zero():
    assert word_error_rate("", "").error_rate == 0.0


def test_corpus_aggregation_weights_by_length():
    # A short utterance with 1 error and a long clean one. Corpus WER must be the
    # pooled count (1 error / 6 ref words), not the mean of per-utterance rates.
    pairs = [("a b c d e", "a b c d e"), ("x", "y")]
    counts = aggregate_wer(pairs)
    assert counts.ref_length == 6
    assert counts.errors == 1
    assert counts.error_rate == 1 / 6


def test_urdu_normalizer_collapses_yeh_variants():
    # Same word, Arabic yeh in the hypothesis. Raw scoring sees a substitution;
    # normalized scoring sees a match. This is why eval passes a normalizer.
    ref = "کیا"  # FARSI YEH
    hyp = "کيا"  # ARABIC YEH
    assert word_error_rate(ref, hyp).errors == 1
    assert word_error_rate(ref, hyp, normalizer=normalize_urdu).errors == 0


def test_character_error_rate_runs_over_chars():
    # "kya" vs "kyaa" — one inserted character.
    counts = character_error_rate("kya", "kyaa")
    assert counts.insertions == 1
    assert counts.ref_length == 3
