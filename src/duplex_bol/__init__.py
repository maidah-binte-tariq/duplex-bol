"""duplex-bol — a full-duplex Pakistani-Urdu speech-to-speech research scaffold.

The package is split along the two tracks described in the feasibility report:

* ``duplex_bol.text``    — Urdu (Nastaliq) text normalization. Both tracks need it.
* ``duplex_bol.audio``   — sample-rate / channel plumbing the trainers are fussy about.
* ``duplex_bol.data``    — manifest schemas, speaker selection, two-party stereo synthesis.
* ``duplex_bol.cascade`` — Track B: a deterministic barge-in orchestrator + component Protocols.
* ``duplex_bol.moshi``   — Track A: tokenizer swap + LoRA config for the Moshi fine-tune.
* ``duplex_bol.eval``    — WER/CER and the latency budget the demo is judged against.

Nothing here loads a model at import time; the heavy stacks live behind the
``[moshi]`` / ``[cascade]`` extras so the core stays cheap to import on Kaggle.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
