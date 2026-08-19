from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf

from timeentropy.data import (
    AugmentationConfig,
    PrefixFeatureDataset,
    augment_waveform,
    augmentation_rng,
    mask_spectrogram,
)
from timeentropy.signal.onset import OnsetConfig


def test_augmentation_stream_is_stable_and_stage_specific() -> None:
    first = augmentation_rng(seed=7, epoch=2, sample_id="x", stage="waveform").normal(10)
    repeated = augmentation_rng(seed=7, epoch=2, sample_id="x", stage="waveform").normal(10)
    other = augmentation_rng(seed=7, epoch=2, sample_id="x", stage="spectrogram").normal(10)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)


def test_noise_reaches_requested_snr_when_gain_is_fixed() -> None:
    waveform = np.sin(np.linspace(0.0, 20.0, 16_000, dtype=np.float32))
    config = AugmentationConfig(
        gain_db=(0.0, 0.0),
        additive_noise_probability=1.0,
        additive_noise_snr_db=(20.0, 20.0),
        spectrogram_mask_probability=0.0,
    )
    augmented = augment_waveform(
        waveform,
        config=config,
        rng=np.random.default_rng(11),
    )
    noise = augmented.astype(np.float64) - waveform.astype(np.float64)
    snr = 20.0 * np.log10(np.sqrt(np.mean(waveform**2)) / np.sqrt(np.mean(noise**2)))
    assert abs(float(snr) - 20.0) < 0.01


def test_masking_is_bounded_and_deterministic() -> None:
    feature = np.ones((20, 10), dtype=np.float32)
    config = AugmentationConfig(
        additive_noise_probability=0.0,
        spectrogram_mask_probability=1.0,
        frequency_mask_max_bins=4,
        time_mask_max_frames=3,
    )
    first = mask_spectrogram(feature, config=config, rng=np.random.default_rng(5))
    second = mask_spectrogram(feature, config=config, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first == 0.0) <= 4 * 10 + 3 * 20


def test_training_dataset_augments_by_epoch_but_evaluation_does_not(tmp_path) -> None:
    sample_rate = 16_000
    waveform = np.concatenate(
        [np.zeros(800, dtype=np.float32), np.full(8_000, 0.1, dtype=np.float32)]
    )
    path = tmp_path / "sample.wav"
    sf.write(path, waveform, sample_rate, subtype="FLOAT")
    manifest = pd.DataFrame(
        [{"sample_id": "s", "path": path, "speaker_id": "F01", "label_id": 1}]
    )
    options = {
        "prefixes_ms": (150,),
        "onset_config": OnsetConfig(sample_rate=sample_rate, consecutive_active_frames=1),
        "augmentation": AugmentationConfig(
            gain_db=(-3.0, 3.0),
            additive_noise_probability=1.0,
            spectrogram_mask_probability=1.0,
        ),
        "seed": 19,
    }
    training = PrefixFeatureDataset(manifest, training=True, **options)
    epoch_zero = training[0]["features"].numpy()
    np.testing.assert_array_equal(epoch_zero, training[0]["features"].numpy())
    training.set_epoch(1)
    assert not np.array_equal(epoch_zero, training[0]["features"].numpy())

    evaluation_with_config = PrefixFeatureDataset(manifest, training=False, **options)[0]
    evaluation_plain = PrefixFeatureDataset(
        manifest,
        training=False,
        prefixes_ms=(150,),
        onset_config=options["onset_config"],
        seed=19,
    )[0]
    np.testing.assert_array_equal(
        evaluation_with_config["features"].numpy(),
        evaluation_plain["features"].numpy(),
    )
