from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OnsetMode = Literal["causal_fixed", "causal_noise_floor", "recovered_offline_peak"]


@dataclass(frozen=True)
class OnsetConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    hop_ms: int = 10
    absolute_rms_floor: float = 0.003
    recovered_relative_db_below_peak: float = 35.0
    causal_noise_reference_ms: int = 100
    causal_snr_db: float = 15.0
    consecutive_active_frames: int = 2


@dataclass(frozen=True)
class OnsetResult:
    onset_sample: int | None
    threshold: float
    frame_index: int | None
    confirmation_sample: int | None
    detected: bool
    mode: OnsetMode


def _milliseconds_to_samples(milliseconds: float, sample_rate: int) -> int:
    return round(milliseconds * sample_rate / 1000.0)


def frame_rms(signal: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
    """Compute left-aligned frame RMS without centered padding or future reflection."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if frame_samples <= 0 or hop_samples <= 0:
        raise ValueError("Frame and hop sizes must be positive")
    if len(values) < frame_samples:
        padded = np.pad(values, (0, frame_samples - len(values)))
        return np.asarray([np.sqrt(np.mean(np.square(padded)))])
    starts = np.arange(0, len(values) - frame_samples + 1, hop_samples)
    return np.asarray(
        [np.sqrt(np.mean(np.square(values[start : start + frame_samples]))) for start in starts]
    )


def _first_consecutive_active(active: np.ndarray, required: int) -> int | None:
    if required < 1:
        raise ValueError("consecutive_active_frames must be at least one")
    run = 0
    for index, value in enumerate(active):
        run = run + 1 if bool(value) else 0
        if run >= required:
            return index - required + 1
    return None


def detect_onset(
    signal: np.ndarray,
    config: OnsetConfig,
    mode: OnsetMode = "causal_fixed",
) -> OnsetResult:
    frame_samples = _milliseconds_to_samples(config.frame_ms, config.sample_rate)
    hop_samples = _milliseconds_to_samples(config.hop_ms, config.sample_rate)
    rms = frame_rms(signal, frame_samples, hop_samples)

    if mode == "recovered_offline_peak":
        relative = float(rms.max(initial=0.0)) * 10.0 ** (
            -config.recovered_relative_db_below_peak / 20.0
        )
        threshold = max(config.absolute_rms_floor, relative)
    elif mode == "causal_fixed":
        threshold = config.absolute_rms_floor
    elif mode == "causal_noise_floor":
        reference_frames = max(
            1,
            int(np.ceil(config.causal_noise_reference_ms / config.hop_ms)),
        )
        noise_rms = float(np.median(rms[:reference_frames]))
        threshold = max(
            config.absolute_rms_floor,
            noise_rms * 10.0 ** (config.causal_snr_db / 20.0),
        )
    else:
        raise ValueError(f"Unsupported onset mode: {mode}")

    first = _first_consecutive_active(rms >= threshold, config.consecutive_active_frames)
    if first is None:
        return OnsetResult(None, threshold, None, None, False, mode)

    onset_sample = first * hop_samples
    confirmation_frame = first + config.consecutive_active_frames - 1
    confirmation_sample = confirmation_frame * hop_samples + frame_samples
    return OnsetResult(
        onset_sample=onset_sample,
        threshold=threshold,
        frame_index=first,
        confirmation_sample=confirmation_sample,
        detected=True,
        mode=mode,
    )


def crop_post_onset_with_preroll(
    signal: np.ndarray,
    onset_sample: int,
    post_onset_ms: int,
    pre_roll_ms: int,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Return exactly pre-roll + post-onset samples, padding only at signal boundaries."""

    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    pre_samples = _milliseconds_to_samples(pre_roll_ms, sample_rate)
    post_samples = _milliseconds_to_samples(post_onset_ms, sample_rate)
    target = pre_samples + post_samples

    requested_start = onset_sample - pre_samples
    requested_end = onset_sample + post_samples
    source_start = max(0, requested_start)
    source_end = min(len(values), requested_end)
    left_pad = max(0, -requested_start)
    right_pad = max(0, requested_end - len(values))
    crop = values[source_start:source_end]
    if left_pad or right_pad:
        crop = np.pad(crop, (left_pad, right_pad))
    if len(crop) != target:
        raise RuntimeError(f"Crop length {len(crop)} differs from target {target}")
    return crop.astype(np.float32, copy=False)
