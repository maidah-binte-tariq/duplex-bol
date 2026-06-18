"""Urdu normalization — the equivalence classes, and the things we must NOT touch.

Characters are written as ``\\uXXXX`` escapes so the test asserts on codepoints,
not on glyphs that may render identically in your editor.
"""

from __future__ import annotations

import pytest

from duplex_bol.text import NormalizationConfig, UrduNormalizer, normalize_urdu

ARABIC_YEH = "ي"
FARSI_YEH = "ی"
ARABIC_KAF = "ك"
KEHEH = "ک"
ARABIC_HEH = "ه"
HEH_GOAL = "ہ"
ZWNJ = "‌"
ZWSP = "​"
TATWEEL = "ـ"
FATHA = "َ"


def test_arabic_yeh_folds_to_farsi_yeh():
    assert normalize_urdu(ARABIC_YEH) == FARSI_YEH


def test_arabic_kaf_folds_to_keheh():
    assert normalize_urdu(ARABIC_KAF) == KEHEH


def test_arabic_heh_folds_to_heh_goal():
    assert normalize_urdu(ARABIC_HEH) == HEH_GOAL


def test_diacritics_are_stripped():
    # کَ  -> ک  (FATHA removed)
    assert normalize_urdu(KEHEH + FATHA) == KEHEH


def test_tatweel_is_removed():
    assert normalize_urdu(f"کام{TATWEEL}یاب") == "کامیاب"


def test_zwnj_is_preserved_by_default():
    # ZWNJ is content in Urdu; it must survive the default preset.
    assert ZWNJ in normalize_urdu(f"کام{ZWNJ}کاج")


def test_other_zero_width_chars_are_stripped():
    assert ZWSP not in normalize_urdu(f"کام{ZWSP}کاج")


@pytest.mark.parametrize(
    "keep",
    ["ؤ", "ئ", "ء", "آ"],  # ؤ  ئ  ء  آ — all genuine Urdu letters
)
def test_genuine_urdu_letters_are_not_folded(keep):
    # The classic normalizer bug is folding these away. We must leave them alone.
    assert keep in normalize_urdu(f"a{keep}b")


def test_digits_kept_by_default_folded_on_request():
    urdu_2026 = "۲۰۲۶"
    assert normalize_urdu(urdu_2026) == urdu_2026
    folded = normalize_urdu(urdu_2026, NormalizationConfig(fold_digits=True))
    assert folded == "2026"


def test_whitespace_is_collapsed_and_trimmed():
    assert normalize_urdu("  کیا   حال  ") == "کیا حال"


def test_punctuation_removal_is_opt_in():
    text = "کیا، حال؟"
    assert normalize_urdu(text) == text  # default keeps it
    stripped = normalize_urdu(text, NormalizationConfig(remove_punctuation=True))
    assert "،" not in stripped and "؟" not in stripped


def test_mixed_real_world_string():
    # Arabic yeh + Arabic heh, the most common cross-keyboard drift.
    assert normalize_urdu("کيا حال هے") == "کیا حال ہے"


def test_normalization_is_idempotent():
    norm = UrduNormalizer()
    once = norm.normalize("کيا   حال هے ")
    assert norm.normalize(once) == once


def test_empty_string():
    assert normalize_urdu("") == ""


def test_callable_interface_matches_method():
    norm = UrduNormalizer()
    assert norm("کيا") == norm.normalize("کيا")
