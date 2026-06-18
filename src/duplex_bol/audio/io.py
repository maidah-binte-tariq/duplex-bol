"""WAV read/write on the standard-library ``wave`` module.

Deliberately no ``soundfile``/``libsndfile`` dependency: the core pipeline only
needs PCM WAV, and keeping it stdlib means CI and a fresh Kaggle worker don't
have to apt-install anything. The cascade/Moshi extras can pull soundfile if you
want FLAC or float WAV; for everything here, PCM 8/16/24/32-bit is enough.

Convention used everywhere in the package:
    * audio is float32 in [-1.0, 1.0]
    * mono is shape (n_samples,), multi-channel is (n_samples, n_channels)
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

Audio = npt.NDArray[np.float32]


def read_wav(path: str | Path) -> tuple[Audio, int]:
    """Read a PCM WAV file. Returns (float32 audio, sample_rate)."""
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    samples = _decode_pcm(raw, sample_width)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    return samples, sample_rate


def write_wav(path: str | Path, audio: Audio, sample_rate: int) -> None:
    """Write float32 audio to 16-bit PCM WAV, clipping out-of-range samples.

    16-bit is the right default here — it's what every downstream ASR/TTS trainer
    in the pipeline ingests, and it halves disk vs 32-bit for no audible loss on
    speech.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim not in (1, 2):
        raise ValueError(f"expected mono (n,) or multichannel (n, c), got shape {arr.shape}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_channels = 1 if arr.ndim == 1 else arr.shape[1]
    clipped = np.clip(arr, -1.0, 1.0)
    # Asymmetric scale: int16 spans [-32768, 32767]. Using 32767 keeps a full-
    # scale +1.0 from wrapping to -32768.
    pcm = (clipped * 32767.0).round().astype("<i2")

    with wave.open(str(path), "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _decode_pcm(raw: bytes, sample_width: int) -> Audio:
    """Decode interleaved PCM bytes to float32 in [-1, 1]."""
    if sample_width == 1:
        # 8-bit WAV is unsigned, centered on 128.
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return data / 32768.0
    if sample_width == 3:
        return _decode_24bit(raw)
    if sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        return data / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


def _decode_24bit(raw: bytes) -> Audio:
    """24-bit has no native numpy dtype, so rebuild it from bytes by hand.

    Pack the 3 little-endian bytes into the low 24 bits of an int32 and sign-
    extend via the arithmetic shift trick (<< 8 then >> 8).
    """
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    packed = (packed << 8) >> 8  # sign-extend bit 23 into bits 24-31
    return (packed.astype(np.float32)) / 8388608.0
