from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseResult:
    waveform: np.ndarray
    seed_sha256: str
    signal_rms: float
    noise_rms: float
    realized_snr_db: float | None
    applied: bool


def white_noise_seed_sha256(*, primary_seed: int, sample_id: str, snr_db: float) -> str:
    """Return the frozen row-order-independent seed digest for one sample/SNR."""

    if primary_seed < 0:
        raise ValueError("primary_seed must be non-negative")
    if not sample_id:
        raise ValueError("sample_id must not be empty")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    material = f"{primary_seed}\0{sample_id}\0{float(snr_db):.17g}".encode()
    return hashlib.sha256(material).hexdigest()


def add_white_gaussian_noise(
    waveform: np.ndarray,
    *,
    primary_seed: int,
    sample_id: str,
    snr_db: float,
) -> NoiseResult:
    """Add zero-mean Gaussian noise after cropping at an exact RMS SNR.

    The random stream is a pure function of ``(primary_seed, sample_id, snr_db)``.
    A silent crop remains silent because a finite relative SNR is undefined; this is
    explicitly reported through ``applied=False`` and ``realized_snr_db=None``.
    """

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("waveform must be non-empty and finite")
    digest = white_noise_seed_sha256(
        primary_seed=primary_seed,
        sample_id=sample_id,
        snr_db=snr_db,
    )
    signal = values.astype(np.float64)
    signal_rms = float(np.sqrt(np.mean(np.square(signal))))
    if signal_rms == 0.0:
        return NoiseResult(
            waveform=values.copy(),
            seed_sha256=digest,
            signal_rms=0.0,
            noise_rms=0.0,
            realized_snr_db=None,
            applied=False,
        )

    generator = np.random.default_rng(int.from_bytes(bytes.fromhex(digest)[:16], "big"))
    noise = generator.standard_normal(values.size)
    noise -= float(noise.mean())
    raw_noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    if raw_noise_rms == 0.0:
        raise RuntimeError("Gaussian generator unexpectedly produced zero-variance noise")
    target_noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    scaled_noise = noise * (target_noise_rms / raw_noise_rms)
    perturbed = signal + scaled_noise
    actual_noise_rms = float(np.sqrt(np.mean(np.square(scaled_noise))))
    realized_snr = 20.0 * math.log10(signal_rms / actual_noise_rms)
    return NoiseResult(
        waveform=np.ascontiguousarray(perturbed, dtype=np.float32),
        seed_sha256=digest,
        signal_rms=signal_rms,
        noise_rms=actual_noise_rms,
        realized_snr_db=realized_snr,
        applied=True,
    )


def shifted_crop_anchor(
    *,
    reference_anchor_sample: int,
    waveform_samples: int,
    offset_ms: int,
    sample_rate: int,
) -> tuple[int, int]:
    """Return requested and clipped onset anchors under the frozen sign convention."""

    if reference_anchor_sample < 0:
        raise ValueError("reference_anchor_sample must be non-negative")
    if waveform_samples < 1 or sample_rate <= 0:
        raise ValueError("waveform_samples and sample_rate must be positive")
    offset_samples = round(int(offset_ms) * sample_rate / 1000.0)
    requested = reference_anchor_sample + offset_samples
    clipped = min(max(requested, 0), waveform_samples)
    return requested, clipped
