from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded audio and the source metadata needed for an audit ledger.

    A mono waveform has shape ``[samples]``. When ``mono=False``, the shape is
    ``[channels, samples]``.
    """

    waveform: np.ndarray
    sample_rate: int
    source_sample_rate: int
    source_channels: int


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)

    divisor = gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    resampled = resample_poly(waveform, up=up, down=down, axis=-1)

    # Pin the duration-to-sample conversion so SciPy version changes cannot add
    # or remove a trailing sample from an experiment.
    expected_samples = round(waveform.shape[-1] * target_rate / source_rate)
    if resampled.shape[-1] > expected_samples:
        resampled = resampled[..., :expected_samples]
    elif resampled.shape[-1] < expected_samples:
        padding = [(0, 0)] * resampled.ndim
        padding[-1] = (0, expected_samples - resampled.shape[-1])
        resampled = np.pad(resampled, padding)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def load_audio(
    path: str | Path,
    target_sample_rate: int = 16_000,
    mono: bool = True,
) -> LoadedAudio:
    """Load finite PCM audio, optionally average channels, and resample.

    Channel averaging happens before resampling. Audio is returned as float32,
    using SoundFile's standard normalization for integer PCM sources.
    """

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    resolved = Path(path)
    waveform, source_rate = sf.read(
        resolved,
        dtype="float32",
        always_2d=True,
    )
    if waveform.shape[0] == 0 or waveform.shape[1] == 0:
        raise ValueError(f"Audio file is empty: {resolved}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio contains non-finite samples: {resolved}")

    source_channels = int(waveform.shape[1])
    if mono:
        prepared = waveform.mean(axis=1, dtype=np.float32)
    else:
        prepared = waveform.T
    prepared = np.ascontiguousarray(prepared, dtype=np.float32)
    prepared = _resample(prepared, int(source_rate), target_sample_rate)
    if not np.isfinite(prepared).all():
        raise ValueError(f"Resampling produced non-finite samples: {resolved}")

    return LoadedAudio(
        waveform=prepared,
        sample_rate=target_sample_rate,
        source_sample_rate=int(source_rate),
        source_channels=source_channels,
    )
