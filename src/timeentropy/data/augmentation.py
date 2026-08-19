from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    """Frozen waveform and spectrogram augmentation distribution."""

    gain_db: tuple[float, float] = (-3.0, 3.0)
    additive_noise_probability: float = 0.5
    additive_noise_snr_db: tuple[float, float] = (20.0, 35.0)
    spectrogram_mask_probability: float = 0.5
    frequency_mask_max_bins: int = 16
    time_mask_max_frames: int = 4

    def __post_init__(self) -> None:
        gain_min, gain_max = self.gain_db
        snr_min, snr_max = self.additive_noise_snr_db
        if not np.isfinite([gain_min, gain_max, snr_min, snr_max]).all():
            raise ValueError("Augmentation ranges must be finite")
        if gain_min > gain_max or snr_min > snr_max:
            raise ValueError("Augmentation range minima cannot exceed maxima")
        for name, probability in (
            ("additive_noise_probability", self.additive_noise_probability),
            ("spectrogram_mask_probability", self.spectrogram_mask_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.frequency_mask_max_bins < 0 or self.time_mask_max_frames < 0:
            raise ValueError("Mask maxima must be non-negative")


def augmentation_rng(
    *,
    seed: int,
    epoch: int,
    sample_id: str,
    stage: str,
) -> np.random.Generator:
    """Create a row-order-independent random stream for one augmentation stage."""

    if seed < 0 or epoch < 0:
        raise ValueError("seed and epoch must be non-negative")
    material = f"{seed}\0{epoch}\0{sample_id}\0{stage}".encode()
    value = int.from_bytes(hashlib.sha256(material).digest()[:16], "big", signed=False)
    return np.random.default_rng(value)


def augment_waveform(
    waveform: np.ndarray,
    *,
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply gain and optional white noise at a sampled signal-to-noise ratio."""

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("waveform must be non-empty and finite")
    gain_db = float(rng.uniform(*config.gain_db))
    augmented = values.astype(np.float64) * 10.0 ** (gain_db / 20.0)
    if rng.random() < config.additive_noise_probability:
        signal_rms = float(np.sqrt(np.mean(np.square(augmented))))
        if signal_rms > 0.0:
            requested_snr = float(rng.uniform(*config.additive_noise_snr_db))
            noise = rng.standard_normal(len(augmented))
            noise_rms = float(np.sqrt(np.mean(np.square(noise))))
            if noise_rms > 0.0:
                target_noise_rms = signal_rms / 10.0 ** (requested_snr / 20.0)
                augmented = augmented + noise * (target_noise_rms / noise_rms)
    return augmented.astype(np.float32)


def mask_spectrogram(
    feature: np.ndarray,
    *,
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one zero frequency mask and one zero time mask under one Bernoulli draw."""

    values = np.asarray(feature, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("feature must be a non-empty finite frequency-by-time matrix")
    if rng.random() >= config.spectrogram_mask_probability:
        return values.copy()
    masked = values.copy()
    maximum_frequency = min(config.frequency_mask_max_bins, masked.shape[0])
    maximum_time = min(config.time_mask_max_frames, masked.shape[1])
    frequency_width = int(rng.integers(0, maximum_frequency + 1))
    time_width = int(rng.integers(0, maximum_time + 1))
    if frequency_width:
        start = int(rng.integers(0, masked.shape[0] - frequency_width + 1))
        masked[start : start + frequency_width, :] = 0.0
    if time_width:
        start = int(rng.integers(0, masked.shape[1] - time_width + 1))
        masked[:, start : start + time_width] = 0.0
    return masked
