"""Word- and character-error-rate scoring.

WER is just Levenshtein distance over *tokens* divided by the reference length,
but the breakdown into substitutions / deletions / insertions is what actually
tells you where an ASR system is failing, so we keep the full alignment counts
rather than collapsing to a single float.

The one Urdu-specific wrinkle: scoring is only fair if reference and hypothesis
are normalized the same way first (ARABIC YEH vs FARSI YEH would otherwise count
as a substitution on every word that contains a yeh). Pass a ``normalizer`` —
:class:`duplex_bol.text.UrduNormalizer` is the intended one — and both sides go
through it before alignment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

Normalizer = Callable[[str], str]


@dataclass(frozen=True)
class ErrorCounts:
    """Alignment breakdown for one or many reference/hypothesis pairs."""

    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        # Edge case: an empty reference. If the hypothesis is also empty the rate
        # is 0.0; if it invented words, every one is an insertion → rate 1.0.
        if self.ref_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.ref_length

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            hits=self.hits + other.hits,
            ref_length=self.ref_length + other.ref_length,
        )


def _align(ref: Sequence[str], hyp: Sequence[str]) -> ErrorCounts:
    """Standard edit-distance DP with a backtrace to recover S/D/I counts.

    O(len(ref) * len(hyp)) time and memory. Fine for utterance-level scoring;
    a corpus is just the sum over utterances (see :func:`aggregate_wer`).
    """
    n, m = len(ref), len(hyp)
    # cost[i][j] = edit distance between ref[:i] and hyp[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i  # delete all of ref
    for j in range(1, m + 1):
        cost[0][j] = j  # insert all of hyp
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # substitution
                    cost[i - 1][j],  # deletion
                    cost[i][j - 1],  # insertion
                )

    # Walk back along the cheapest path, tallying operation types.
    i, j = n, m
    subs = dels = ins = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return ErrorCounts(subs, dels, ins, hits, ref_length=n)


def _prep(text: str, normalizer: Normalizer | None) -> str:
    return normalizer(text) if normalizer is not None else text


def word_error_rate(ref: str, hyp: str, *, normalizer: Normalizer | None = None) -> ErrorCounts:
    """WER between two strings, tokenized on whitespace.

    Returns the full :class:`ErrorCounts`; ``.error_rate`` is the scalar WER.
    """
    ref_tokens = _prep(ref, normalizer).split()
    hyp_tokens = _prep(hyp, normalizer).split()
    return _align(ref_tokens, hyp_tokens)


def character_error_rate(
    ref: str, hyp: str, *, normalizer: Normalizer | None = None
) -> ErrorCounts:
    """CER — same alignment, but over characters with whitespace removed.

    Useful for Urdu because word-boundary placement is itself noisy; CER sidesteps
    the question of where one word ends and the next begins.
    """
    ref_chars = list(_prep(ref, normalizer).replace(" ", ""))
    hyp_chars = list(_prep(hyp, normalizer).replace(" ", ""))
    return _align(ref_chars, hyp_chars)


def aggregate_wer(
    pairs: Iterable[tuple[str, str]], *, normalizer: Normalizer | None = None
) -> ErrorCounts:
    """Corpus-level WER: sum the per-utterance counts, then divide once.

    This is the correct way to aggregate — averaging per-utterance WERs lets a
    short noisy utterance dominate a long clean one.
    """
    total = ErrorCounts(0, 0, 0, 0, 0)
    for ref, hyp in pairs:
        total = total + word_error_rate(ref, hyp, normalizer=normalizer)
    return total
