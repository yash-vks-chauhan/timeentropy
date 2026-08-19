import numpy as np

from timeentropy.signal.onset import (
    OnsetConfig,
    crop_post_onset_with_preroll,
    detect_onset,
)


def test_causal_detector_finds_sustained_signal() -> None:
    config = OnsetConfig(
        sample_rate=1_000,
        frame_ms=20,
        hop_ms=10,
        causal_noise_reference_ms=50,
        consecutive_active_frames=2,
    )
    signal = np.concatenate([np.zeros(100), np.ones(200, dtype=np.float32) * 0.1])
    result = detect_onset(signal, config, mode="causal_fixed")
    assert result.detected
    assert result.onset_sample == 90
    assert result.confirmation_sample == 120


def test_crop_contains_full_post_onset_context_plus_preroll() -> None:
    signal = np.arange(1_000, dtype=np.float32)
    crop = crop_post_onset_with_preroll(
        signal,
        onset_sample=200,
        post_onset_ms=300,
        pre_roll_ms=30,
        sample_rate=1_000,
    )
    assert len(crop) == 330
    np.testing.assert_array_equal(crop, signal[170:500])


def test_crop_left_pads_missing_preroll() -> None:
    signal = np.arange(100, dtype=np.float32)
    crop = crop_post_onset_with_preroll(
        signal,
        onset_sample=10,
        post_onset_ms=20,
        pre_roll_ms=30,
        sample_rate=1_000,
    )
    assert len(crop) == 50
    np.testing.assert_array_equal(crop[:20], np.zeros(20, dtype=np.float32))
    np.testing.assert_array_equal(crop[20:], signal[:30])
