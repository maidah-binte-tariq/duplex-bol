"""Audio transforms + WAV round-trips."""

from __future__ import annotations

import numpy as np

from duplex_bol.audio import (
    duration_s,
    mix_to_stereo,
    peak_normalize,
    read_wav,
    resample,
    rms_dbfs,
    to_mono,
    write_wav,
)


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [0.0, 0.5])


def test_to_mono_passes_through_mono():
    mono = np.array([0.1, 0.2], dtype=np.float32)
    assert np.allclose(to_mono(mono), mono)


def test_duration_seconds():
    assert duration_s(np.zeros(16_000, dtype=np.float32), 16_000) == 1.0


def test_resample_changes_length_proportionally():
    # One second of audio at each rate; resampling must preserve the 1-second span.
    assert resample(np.zeros(16_000, dtype=np.float32), 16_000, 24_000).shape[0] == 24_000
    assert resample(np.zeros(24_000, dtype=np.float32), 24_000, 16_000).shape[0] == 16_000


def test_resample_is_noop_when_rates_match(sine):
    assert np.array_equal(resample(sine, 16_000, 16_000), sine)


def test_resample_preserves_endpoints(sine):
    out = resample(sine, 16_000, 8_000)
    # Linear interp pins the first and last samples exactly.
    assert np.isclose(out[0], sine[0])
    assert np.isclose(out[-1], sine[-1])


def test_resample_preserves_stereo_layout():
    stereo = np.zeros((100, 2), dtype=np.float32)
    out = resample(stereo, 16_000, 24_000)
    assert out.ndim == 2 and out.shape[1] == 2


def test_peak_normalize_hits_target():
    x = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = peak_normalize(x, peak_dbfs=0.0)
    assert np.isclose(np.max(np.abs(out)), 1.0, atol=1e-6)


def test_peak_normalize_leaves_silence_alone():
    x = np.zeros(10, dtype=np.float32)
    assert np.array_equal(peak_normalize(x), x)


def test_rms_dbfs_of_silence_is_neg_inf():
    assert rms_dbfs(np.zeros(100, dtype=np.float32)) == float("-inf")


def test_rms_dbfs_of_full_scale_dc_is_zero():
    assert np.isclose(rms_dbfs(np.ones(100, dtype=np.float32)), 0.0)


def test_mix_to_stereo_routes_and_pads():
    left = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    right = np.array([0.5], dtype=np.float32)
    out = mix_to_stereo(left, right)
    assert out.shape == (3, 2)
    assert np.allclose(out[:, 0], [1.0, 1.0, 1.0])  # left intact
    assert np.allclose(out[:, 1], [0.5, 0.0, 0.0])  # right zero-padded


def test_wav_roundtrip_mono_within_quantization(sine, wav_path):
    write_wav(wav_path, sine, 16_000)
    loaded, sr = read_wav(wav_path)
    assert sr == 16_000
    assert loaded.shape == sine.shape
    # 16-bit quantization step is ~3e-5; allow a hair more.
    assert np.max(np.abs(loaded - sine)) < 1e-4


def test_wav_roundtrip_stereo(wav_path):
    stereo = np.stack([np.linspace(-0.5, 0.5, 1000), np.linspace(0.5, -0.5, 1000)], axis=1).astype(
        np.float32
    )
    write_wav(wav_path, stereo, 24_000)
    loaded, sr = read_wav(wav_path)
    assert sr == 24_000
    assert loaded.shape == (1000, 2)


def test_wav_write_clips_out_of_range(wav_path):
    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    write_wav(wav_path, loud, 16_000)
    loaded, _ = read_wav(wav_path)
    assert np.max(np.abs(loaded)) <= 1.0
