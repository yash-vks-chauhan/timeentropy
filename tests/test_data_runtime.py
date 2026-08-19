from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import soundfile as sf

from timeentropy.data.audio import load_audio
from timeentropy.data.dataset import PrefixFeatureDataset, collate_prefix_features
from timeentropy.data.sampling import class_balanced_weights, make_class_balanced_sampler
from timeentropy.data.validation import (
    assign_validation_partitions,
    attach_validation_partitions,
    validation_partition,
)
from timeentropy.signal.onset import OnsetConfig


def test_validation_subpartition_is_sha256_based_and_row_order_independent() -> None:
    sample_ids = [f"sample-{index:04d}" for index in range(1_000)]
    expected_names = ("checkpoint", "checkpoint", "calibration", "policy")
    expected = tuple(
        expected_names[
            int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big") % 4
        ]
        for value in sample_ids
    )
    assert assign_validation_partitions(sample_ids) == expected

    counts = {name: expected.count(name) for name in set(expected)}
    assert 430 <= counts["checkpoint"] <= 570
    assert 190 <= counts["calibration"] <= 310
    assert 190 <= counts["policy"] <= 310

    frame = pd.DataFrame({"sample_id": sample_ids[:10], "value": range(10)})
    attached = attach_validation_partitions(frame)
    reversed_attached = attach_validation_partitions(frame.iloc[::-1])
    by_id = attached.set_index("sample_id")["validation_partition"].to_dict()
    reversed_by_id = reversed_attached.set_index("sample_id")[
        "validation_partition"
    ].to_dict()
    assert by_id == reversed_by_id
    assert validation_partition(sample_ids[0]) == expected[0]


def test_audio_loader_averages_channels_and_resamples(tmp_path) -> None:
    source_rate = 8_000
    first = np.linspace(-0.6, 0.6, 800, dtype=np.float32)
    second = np.linspace(0.2, -0.2, 800, dtype=np.float32)
    stereo = np.stack([first, second], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, source_rate, subtype="FLOAT")

    mono = load_audio(path, target_sample_rate=16_000, mono=True)
    assert mono.source_sample_rate == source_rate
    assert mono.source_channels == 2
    assert mono.sample_rate == 16_000
    assert mono.waveform.dtype == np.float32
    assert mono.waveform.shape == (1_600,)

    channels = load_audio(path, target_sample_rate=source_rate, mono=False)
    assert channels.waveform.shape == (2, 800)
    np.testing.assert_allclose(channels.waveform, stereo.T, rtol=0, atol=1.0e-6)
    np.testing.assert_allclose(
        channels.waveform.mean(axis=0),
        stereo.mean(axis=1),
        rtol=0,
        atol=1.0e-6,
    )


def _write_onset_fixture(path, sample_rate: int = 1_000) -> None:
    silence = np.zeros(100, dtype=np.float32)
    speech = np.sin(np.linspace(0, 30, 500, dtype=np.float32)) * 0.1
    sf.write(path, np.concatenate([silence, speech]), sample_rate, subtype="FLOAT")


def _fixture_manifest(path, rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"fixture-{index}" for index in range(rows)],
            "path": [str(path)] * rows,
            "speaker_id": [f"S{index}" for index in range(rows)],
            "label_id": [index % 2 for index in range(rows)],
        }
    )


def test_shared_training_dataset_selects_one_reproducible_prefix_per_epoch(tmp_path) -> None:
    path = tmp_path / "onset.wav"
    _write_onset_fixture(path)
    manifest = _fixture_manifest(path, rows=20)
    onset = OnsetConfig(
        sample_rate=1_000,
        frame_ms=20,
        hop_ms=10,
        consecutive_active_frames=2,
    )
    dataset = PrefixFeatureDataset(
        manifest,
        training=True,
        prefixes_ms=(150, 200, 250, 300),
        target_sample_rate=1_000,
        onset_config=onset,
        pre_roll_ms=30,
        n_fft=64,
        seed=17,
    )
    assert len(dataset) == len(manifest)

    dataset.set_epoch(3)
    first = tuple(dataset.prefix_for_index(index) for index in range(len(dataset)))
    dataset.set_epoch(3)
    assert tuple(dataset.prefix_for_index(index) for index in range(len(dataset))) == first
    assert len(set(first)) >= 3

    dataset.set_epoch(4)
    second = tuple(dataset.prefix_for_index(index) for index in range(len(dataset)))
    assert first != second

    item = dataset[0]
    assert item["features"].shape[0:2] == (1, 33)
    assert item["onset_detected"]
    assert item["onset_sample"] == 90
    assert item["input_num_samples"] == item["prefix_ms"] + 30


def test_evaluation_dataset_enumerates_prefixes_and_collates_time_padding(tmp_path) -> None:
    path = tmp_path / "onset.wav"
    _write_onset_fixture(path)
    dataset = PrefixFeatureDataset(
        _fixture_manifest(path),
        training=False,
        prefixes_ms=(150, 200),
        target_sample_rate=1_000,
        onset_config=OnsetConfig(sample_rate=1_000),
        pre_roll_ms=30,
        n_fft=64,
    )
    assert len(dataset) == 4
    assert [dataset.prefix_for_index(index) for index in range(4)] == [150, 200, 150, 200]
    assert dataset.labels == (0, 0, 1, 1)

    short = dataset[0]
    long = dataset[1]
    assert short["features"].shape == (1, 33, 16)
    assert long["features"].shape == (1, 33, 21)
    batch = collate_prefix_features([short, long])
    assert batch["features"].shape == (2, 1, 33, 21)
    assert batch["feature_lengths"].tolist() == [16, 21]
    assert batch["padding_mask"][0].tolist() == [False] * 16 + [True] * 5
    assert not batch["padding_mask"][1].any()
    assert batch["label_id"].tolist() == [0, 0]


def test_inverse_frequency_sampler_balances_imbalanced_labels() -> None:
    labels = [0] * 90 + [1] * 10
    weights = class_balanced_weights(labels)
    assert weights[:90].unique().item() == 1.0 / 90.0
    assert weights[90:].unique().item() == 1.0 / 10.0

    first = list(make_class_balanced_sampler(labels, num_samples=20_000, seed=9))
    second = list(make_class_balanced_sampler(labels, num_samples=20_000, seed=9))
    assert first == second
    sampled_labels = np.asarray([labels[index] for index in first])
    assert 0.48 <= float(sampled_labels.mean()) <= 0.52
