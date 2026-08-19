from __future__ import annotations

import numpy as np


def log_magnitude_stft(
    signal: np.ndarray,
    sample_rate: int = 16_000,
    n_fft: int = 512,
    window_ms: int = 25,
    hop_ms: int = 10,
    normalization_floor: float = 1.0e-6,
) -> np.ndarray:
    """Compute an uncentered log-magnitude STFT with per-sample standardization."""

    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    window_samples = round(window_ms * sample_rate / 1000.0)
    hop_samples = round(hop_ms * sample_rate / 1000.0)
    if n_fft < window_samples:
        raise ValueError("n_fft must be at least as large as the analysis window")
    if len(values) < window_samples:
        values = np.pad(values, (0, window_samples - len(values)))

    starts = np.arange(0, len(values) - window_samples + 1, hop_samples)
    # NumPy's hanning(N) is symmetric. Dropping the final sample of hanning(N + 1)
    # produces the periodic Hann convention used by torch.hann_window(periodic=True).
    window = np.hanning(window_samples + 1)[:-1].astype(np.float32)
    frames = np.stack([values[start : start + window_samples] * window for start in starts])
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    feature = np.log1p(np.abs(spectrum)).T.astype(np.float32)
    mean = float(feature.mean())
    std = max(float(feature.std()), normalization_floor)
    return ((feature - mean) / std).astype(np.float32)
