"""Urdu (Nastaliq) text normalization.

Why this module exists
----------------------
Urdu is written in the Arabic script, and the same word arrives spelled three
different ways depending on whose keyboard typed it: an Arabic keyboard emits
ARABIC YEH (U+064A), an Urdu keyboard emits FARSI YEH (U+06CC), and a web form
might paste either. To a tokenizer those are different characters, so "the same
word" fragments into several vocab entries and both ASR scoring and TTS training
degrade. Normalization collapses these equivalence classes to a single canonical
Urdu spelling.

Design choices worth knowing about (these are the easy ones to get wrong):

* **ZWNJ (U+200C) is preserved by default.** In Urdu the zero-width non-joiner is
  *content*, not whitespace — it controls letter joining and changes how a word
  renders (and sometimes what it means). Stripping it is the classic bug in
  naive "just remove zero-width chars" cleaners. We strip ZWSP/ZWJ/BOM and the
  bidi marks, but leave ZWNJ alone unless asked.
* **We do NOT fold the hamza letters ؤ / ئ / ء.** Those are legitimate Urdu
  letters (آؤ، کوئی، جزء), not Arabic artifacts. Folding them — which several
  off-the-shelf normalizers do — corrupts real words. We only fold characters
  that have a genuine Arabic→Urdu canonical mapping.

The mappings below follow common Urdu NLP practice (the urduhack lineage). Every
fold is written as an explicit ``\\uXXXX`` escape with the Unicode name in the
comment so a reviewer can check it without trusting that two glyphs which look
identical actually are.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --- combining marks we drop entirely (harakat / tashkeel) --------------------
# These are short-vowel and gemination marks. Urdu is almost always written
# without them; when they do appear they fragment the vocabulary.
_DIACRITICS = (
    "ً"  # ARABIC FATHATAN
    "ٌ"  # ARABIC DAMMATAN
    "ٍ"  # ARABIC KASRATAN
    "َ"  # ARABIC FATHA
    "ُ"  # ARABIC DAMMA
    "ِ"  # ARABIC KASRA
    "ّ"  # ARABIC SHADDA
    "ْ"  # ARABIC SUKUN
    "ٓ"  # ARABIC MADDAH ABOVE
    "ٔ"  # ARABIC HAMZA ABOVE (as a combining mark, not the letter)
    "ٕ"  # ARABIC HAMZA BELOW
    "ٖ"  # ARABIC SUBSCRIPT ALEF
    "ٰ"  # ARABIC LETTER SUPERSCRIPT ALEF (khari zabar)
)
_DIACRITIC_RE = re.compile(f"[{_DIACRITICS}]")

# --- character folding: Arabic codepoint -> canonical Urdu codepoint ----------
_CHAR_FOLD: dict[str, str] = {
    "ي": "ی",  # ARABIC YEH        -> FARSI YEH (ی)
    "ى": "ی",  # ALEF MAKSURA      -> FARSI YEH (ی)
    "ك": "ک",  # ARABIC KAF        -> KEHEH (ک)
    "ه": "ہ",  # ARABIC HEH        -> HEH GOAL (ہ)  [gol he]
    "ة": "ہ",  # TEH MARBUTA       -> HEH GOAL (ہ)  [heuristic; loanwords]
    "أ": "ا",  # ALEF W/ HAMZA ABV -> ALEF (ا)
    "إ": "ا",  # ALEF W/ HAMZA BLW -> ALEF (ا)
    "ٱ": "ا",  # ALEF WASLA        -> ALEF (ا)
    # NOTE: ALEF WITH MADDA ABOVE (U+0622, آ) is intentionally NOT folded — it is
    # standard Urdu. Likewise ؤ/ئ/ء are left untouched (see module docstring).
}
_FOLD_TABLE = str.maketrans(_CHAR_FOLD)

# --- digit folding (off by default) -------------------------------------------
# Two digit families show up: ARABIC-INDIC (U+0660‑0669) and EXTENDED ARABIC-INDIC
# (U+06F0‑06F9, the ones Urdu actually prefers). Some downstream models want
# ASCII; that is a lossy choice, hence opt-in.
_DIGIT_FOLD = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}
_DIGIT_TABLE = str.maketrans(_DIGIT_FOLD)

# --- zero-width + formatting characters ---------------------------------------
_ZWNJ = "‌"  # preserved by default — it is meaningful in Urdu
_ZERO_WIDTH_STRIP = (
    "​"  # ZERO WIDTH SPACE
    "‍"  # ZERO WIDTH JOINER
    "‎"  # LEFT-TO-RIGHT MARK
    "‏"  # RIGHT-TO-LEFT MARK
    "‪‫‬‭‮"  # bidi embeddings/overrides
    "⁦⁧⁨⁩"  # bidi isolates
    "﻿"  # BOM / ZERO WIDTH NO-BREAK SPACE
)
_TATWEEL = "ـ"  # ARABIC TATWEEL (kashida) — pure decoration, always dropped

# Urdu sentence punctuation. Kept by default (TTS prosody, ASR scoring), droppable.
_URDU_PUNCT = (
    "۔"  # ARABIC FULL STOP (Urdu khatma ۔)
    "،"  # ARABIC COMMA ،
    "؛"  # ARABIC SEMICOLON ؛
    "؟"  # ARABIC QUESTION MARK ؟
)
_ASCII_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
_PUNCT_RE = re.compile(f"[{re.escape(_URDU_PUNCT)}{re.escape(_ASCII_PUNCT)}]")

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationConfig:
    """Knobs for :class:`UrduNormalizer`. Defaults are the safe ASR/TTS preset."""

    strip_diacritics: bool = True
    fold_characters: bool = True
    strip_tatweel: bool = True
    fold_digits: bool = False  # lossy: ۰۱۲ -> 012
    preserve_zwnj: bool = True  # leave U+200C in place (it is content)
    collapse_whitespace: bool = True
    remove_punctuation: bool = False
    nfc: bool = True  # apply Unicode NFC so composed/decomposed forms match


class UrduNormalizer:
    """Normalize Urdu text to a single canonical spelling.

    Stateless and cheap — construct one and reuse it. The pipeline order matters:
    NFC first (so combining sequences are composed predictably), then strip
    decoration, then fold codepoints, then squeeze whitespace last.

    >>> UrduNormalizer().normalize("کيا حال هے")   # Arabic yeh + Arabic heh
    'کیا حال ہے'
    """

    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        cfg = self.config

        if cfg.nfc:
            text = unicodedata.normalize("NFC", text)

        # Drop zero-width + bidi marks. ZWNJ gets a temporary placeholder so the
        # strip can't touch it, then we restore (or drop) it per config.
        if cfg.preserve_zwnj:
            text = text.replace(_ZWNJ, "\x00")
        text = text.translate({ord(c): None for c in _ZERO_WIDTH_STRIP + _ZWNJ})
        if cfg.preserve_zwnj:
            text = text.replace("\x00", _ZWNJ)

        if cfg.strip_tatweel:
            text = text.replace(_TATWEEL, "")
        if cfg.strip_diacritics:
            text = _DIACRITIC_RE.sub("", text)
        if cfg.fold_characters:
            text = text.translate(_FOLD_TABLE)
        if cfg.fold_digits:
            text = text.translate(_DIGIT_TABLE)
        if cfg.remove_punctuation:
            text = _PUNCT_RE.sub(" ", text)

        # NBSP and friends become normal spaces before the final squeeze.
        text = text.replace(" ", " ")
        if cfg.collapse_whitespace:
            text = _WS_RE.sub(" ", text).strip()
        return text

    def __call__(self, text: str) -> str:
        return self.normalize(text)


# Module-level convenience for one-off calls and the CLI.
_DEFAULT = UrduNormalizer()


def normalize_urdu(text: str, config: NormalizationConfig | None = None) -> str:
    """Normalize ``text`` with the default preset, or a custom ``config``."""
    if config is None:
        return _DEFAULT.normalize(text)
    return UrduNormalizer(config).normalize(text)
