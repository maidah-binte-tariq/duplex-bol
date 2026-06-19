#!/usr/bin/env python3
"""Generate a REAL Urdu speech sample from an open TTS model.

This is the honest way to get an Urdu voice clip: run an actual open model. It does
**not** run in this repo's build sandbox (no model/network access there), so the
.wav is produced wherever you have Hugging Face access — a Kaggle/Colab cell or your
own machine:

    pip install "duplex-bol[tts]"            # transformers + torch
    python samples/generate_tts_sample.py    # -> samples/urdu_tts_mms.wav

Default model: ``facebook/mms-tts-urd`` (Meta MMS-TTS Urdu) — small (~150 MB),
CPU-runnable, one forward pass. It is the **base open model** (a candidate Track B
"mouth"), *not* a model we fine-tuned, and the output should be labeled that way.
For the fine-tuned-voice path (H7), see ``notebooks/01_cascade_track_b_kaggle.ipynb``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from duplex_bol.audio import write_wav
from duplex_bol.text import normalize_urdu

DEFAULT_TEXT = "السلام علیکم، میں آپ کی کیا مدد کر سکتا ہوں؟"


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize a real Urdu clip with an open TTS model.")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Urdu text to speak")
    ap.add_argument("--model", default="facebook/mms-tts-urd", help="HF model id")
    ap.add_argument("--out", type=Path, default=Path("samples/urdu_tts_mms.wav"))
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, VitsModel
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise SystemExit("this needs transformers + torch:  pip install 'duplex-bol[tts]'") from exc

    text = normalize_urdu(args.text)  # same normalizer the rest of the pipeline uses
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform[0].cpu().numpy().astype(np.float32)

    sample_rate = int(model.config.sampling_rate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(args.out, waveform, sample_rate)
    print(f"wrote {args.out}  ({len(waveform) / sample_rate:.2f}s @ {sample_rate} Hz)")
    print(f"model: {args.model}  (BASE open model, not a fine-tune)")
    print(f"text:  {text}")


if __name__ == "__main__":
    main()
