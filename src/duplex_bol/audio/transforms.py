"""Channel and sample-rate transforms.

The unglamorous half of speech work: every model wants a specific sample rate and
channel layout, and feeding it the wrong one fails *silently* — training runs, the
loss looks fine, the output is garbage. These helpers make the conversions
explicit and testable.

A note on :func:`resample`: it's linear interpolation, which is honest but not
anti-aliased. For final corpus prep prefer ``soxr``/``libsoxr`` (the cascade extra
pulls it). Linear is fine for the synthetic fixtures and for downsampling speech
where the content sits well below Nyquist; it is *not* fine for high-fidelity
upsampling. This tradeoff is called out in docs/data-engineering.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Audio = npt.NDArray[np.float32]

_EPS = 1e-12


def to_mono(audio: Audio) -> Audio:
    """Collapse a multi-channel signal to mono by averaging channels."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    mono: Audio = arr.mean(axis=1).astype(np.float32)
    return mono


def duration_s(audio: Audio, sample_rate: int) -> float:
    """Length of ``audio`` in seconds. Works for mono or multi-channel."""
    return float(np.asarray(audio).shape[0]) / float(sample_rate)


def rms_dbfs(audio: Audio) -> float:
    """RMS level in dBFS. Silence returns -inf rather than blowing up on log(0)."""
    arr = to_mono(np.asarray(audio, dtype=np.float32))
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    if rms <= _EPS:
        return float("-inf")
    return 20.0 * float(np.log10(rms))


def peak_normalize(audio: Audio, peak_dbfs: float = -1.0) -> Audio:
    """Scale so the loudest sample sits at ``peak_dbfs``.

    Leaves a hair of headroom by default (-1 dBFS) so later mixing / int16
    rounding can't clip. A fully-silent clip is returned unchanged.
    """
    arr = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= _EPS:
        return arr
    target_linear = 10.0 ** (peak_dbfs / 20.0)
    scaled: Audio = (arr * (target_linear / peak)).astype(np.float32)
    return scaled


def resample(audio: Audio, sr_in: int, sr_out: int) -> Audio:
    """Resample mono or multi-channel audio from ``sr_in`` to ``sr_out``.

    Linear interpolation (see module docstring for the caveat). Channel layout is
    preserved.
    """
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("sample rates must be positive")
    arr = np.asarray(audio, dtype=np.float32)
    if sr_in == sr_out or arr.shape[0] == 0:
        return arr

    n_in = arr.shape[0]
    n_out = round(n_in * sr_out / sr_in)
    if n_out <= 0:
        return arr[:0]
    # Sample positions of the output grid expressed in input-sample coordinates.
    src_idx = np.linspace(0.0, n_in - 1, num=n_out, dtype=np.float64)
    grid = np.arange(n_in, dtype=np.float64)

    if arr.ndim == 1:
        return np.interp(src_idx, grid, arr).astype(np.float32)
    channels = [np.interp(src_idx, grid, arr[:, c]) for c in range(arr.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def mix_to_stereo(left: Audio, right: Audio) -> Audio:
    """Place two mono signals on the L and R channels of one stereo array.

    The shorter clip is zero-padded to match the longer. This is the literal
    operation behind Track A's "agent on the left, user on the right" two-party
    synthesis — see :mod:`duplex_bol.data.stereo_dialogue`.
    """
    left_m = to_mono(np.asarray(left, dtype=np.float32))
    right_m = to_mono(np.asarray(right, dtype=np.float32))
    n = max(left_m.shape[0], right_m.shape[0])
    out = np.zeros((n, 2), dtype=np.float32)
    out[: left_m.shape[0], 0] = left_m
    out[: right_m.shape[0], 1] = right_m
    return out
