"""Deterministic stand-ins for the three model components.

These exist to test *orchestration*, not transcription quality. The ASR doesn't
look at the audio at all — it replays a script — because what we're verifying is
the policy: does capture start on speech, does the bot stop on barge-in, are the
latencies within budget. Swapping in a real model can't change those answers.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from duplex_bol.cascade.interfaces import AudioFrame, Transcript, TtsChunk


class ScriptedASR:
    """Replays a list of utterances, one per capture phase.

    Each ``accept`` of a speech frame reveals one more word as a partial; each
    ``finalize`` commits the current utterance and advances to the next. Mirrors
    how a streaming recognizer firms up text as more audio arrives.
    """

    def __init__(self, utterances: list[str]) -> None:
        self._utterances = utterances
        self._index = 0
        self._words_revealed = 0

    def reset(self) -> None:
        self._words_revealed = 0

    def _current_words(self) -> list[str]:
        if self._index >= len(self._utterances):
            return []
        return self._utterances[self._index].split()

    def accept(self, frame: AudioFrame) -> Transcript | None:
        self._words_revealed += 1
        words = self._current_words()
        k = min(len(words), self._words_revealed)
        if k == 0:
            return None
        return Transcript(" ".join(words[:k]), is_final=False)

    def finalize(self) -> Transcript:
        words = self._current_words()
        self._index += 1
        self._words_revealed = 0
        return Transcript(" ".join(words), is_final=True)


class RuleBasedAgent:
    """A lookup-table brain with an echo fallback. No LLM, fully deterministic."""

    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None) -> None:
        self._responses = responses or {}
        self._default = default

    def respond(self, user_text: str) -> str:
        if user_text in self._responses:
            return self._responses[user_text]
        if self._default is not None:
            return self._default
        return f"آپ نے کہا: {user_text}"  # "you said: ..."


class ChunkedTTS:
    """Emits a fixed number of audio chunks proportional to the reply length.

    The samples are filler — what matters for the orchestrator is *how long* the
    bot talks (more words → more chunks → a longer window in which it can be
    interrupted).
    """

    def __init__(
        self, frames_per_word: int = 3, frame_samples: int = 320, amplitude: float = 0.3
    ) -> None:
        self.frames_per_word = frames_per_word
        self.frame_samples = frame_samples
        self.amplitude = amplitude

    def synthesize(self, text: str) -> Iterator[TtsChunk]:
        n_words = max(1, len(text.split()))
        n_chunks = n_words * self.frames_per_word
        block = np.full(self.frame_samples, self.amplitude, dtype=np.float32)
        for i in range(n_chunks):
            yield TtsChunk(block, is_last=(i == n_chunks - 1))
