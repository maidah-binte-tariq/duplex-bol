# Contributing

Thanks for looking. This is a research scaffold, so the bar is "clear, typed, and
tested" rather than "feature complete".

## Getting set up

```bash
make setup        # uv venv + editable install (dev + moshi extras) + pre-commit
make check        # what CI runs: ruff, mypy, pytest
```

If you don't have `uv`, grab it from <https://docs.astral.sh/uv/> — it's the only
prerequisite besides Python 3.11+.

## The rules of the road

- **Everything stays typed.** `mypy --strict` runs in CI and there are no
  `# type: ignore`s in the core. If numpy fights you, annotate the local; don't
  loosen the config.
- **New behavior comes with a test.** The whole point of the orchestrator being a
  pure state machine, and the audio/data code taking injected inputs, is that you
  can test it without a GPU. Keep it that way.
- **No data or weights in git.** `data/`, `*.wav`, `*.safetensors` etc. are
  git-ignored. The pipeline re-derives everything from `data/raw/`.
- **Urdu strings are written as real UTF-8**, not `\uXXXX` escapes — except inside
  `text/urdu.py` and its tests, where explicit codepoints are the point.

## Commit style

Conventional-ish: `area: short imperative summary` (e.g.
`data: add gender-balanced speaker selection`). Keep the subject under ~72 chars.

## Before you open a PR

Run `make check` locally. Pre-commit will also catch formatting and the mypy
basics on the way in.
