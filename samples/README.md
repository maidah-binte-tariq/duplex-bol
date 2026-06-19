# Sample pack

What's real and what isn't — stated up front, because that distinction matters:

| file | what it is | real audio? |
|---|---|---|
| `two_party_demo.wav` | a two-party stereo "call" built by `duplex_bol.data.build_dialogue` — **agent on the LEFT channel, user on the RIGHT**, with overlap | **format demo only** — the voices are placeholder tones, not speech |
| `two_party_demo.jsonl` | the `StereoDialogue` manifest for that WAV (timestamped turns, one per channel) | — |
| `example_call.md` | a reference Urdu conversation the agent is built to handle (greeting → code-switching → a mid-sentence barge-in) | illustrative target dialogue, **not** model output |

`two_party_demo.wav` is a genuine artifact of the data pipeline: pan it left/right
and you can hear the exact stereo layout Moshi trains on (and the deliberate
overlap that teaches simultaneous speech). It is **not** a fine-tuned voice, and it
isn't presented as one.

## Where the real fine-tuned voice samples come from

A genuine fine-tuned **Urdu voice** sample requires a GPU run — we don't ship
fabricated clips. The notebooks produce them:

- `notebooks/01_cascade_track_b_kaggle.ipynb` — fine-tunes an Urdu TTS voice on the
  3-speaker set; the held-out clips it generates are the H7 listening test.
- `notebooks/02_moshi_lora_track_a.ipynb` — the Moshi LoRA path.

## Regenerate the demo artifact

```python
import numpy as np
from duplex_bol.audio import write_wav
from duplex_bol.data import SpeakerClip, DialogueConfig, build_dialogue, write_jsonl

sr = 24_000
def tone(f, sec):           # placeholder "voice"
    t = np.arange(int(sec*sr))/sr
    env = np.minimum(1.0, np.minimum(t*12, (sec-t)*12))
    return (0.5*env*np.sin(2*np.pi*f*t)).astype("float32")

turns = [
    SpeakerClip("agent", tone(180, 0.9), "السلام علیکم، میں آپ کی کیا مدد کر سکتا ہوں", sr),
    SpeakerClip("user",  tone(250, 0.8), "میرا card block ہو گیا ہے", sr),
    SpeakerClip("agent", tone(180, 0.9), "جی میں ابھی دیکھتا ہوں، تھوڑا انتظار کریں", sr),
    SpeakerClip("user",  tone(250, 0.6), "نہیں رکیں، پہلے balance بتائیں", sr),  # barge-in
]
stereo, dlg = build_dialogue(turns, DialogueConfig(overlap_s=0.15))
write_wav("samples/two_party_demo.wav", stereo, dlg.sample_rate)
dlg.audio_path = "two_party_demo.wav"
write_jsonl("samples/two_party_demo.jsonl", [dlg])
```
