from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from timeentropy.data.dataset import PrefixFeatureDataset
from timeentropy.data.raw import RawPrefixDataset, collate_raw_prefix_batch
from timeentropy.signal.onset import OnsetConfig


def _write_onset_audio(path, *, sample_rate: int = 1_000, stereo: bool = False) -> None:
    silence = np.zeros(round(sample_rate * 0.1), dtype=np.float32)
    speech_samples = round(sample_rate * 0.6)
    speech = np.sin(np.linspace(0, 40, speech_samples, dtype=np.float32)) * 0.1
    mono = np.concatenate([silence, speech])
    values = np.stack([mono, mono * 0.5], axis=1) if stereo else mono
    sf.write(path, values, sample_rate, subtype="FLOAT")


def _manifest(path, rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"raw-{index:03d}" for index in range(rows)],
            "path": [str(path)] * rows,
            "speaker_id": [f"S{index:03d}" for index in range(rows)],
            "label_id": [index % 2 for index in range(rows)],
            "validation_partition": [
                "checkpoint" if index % 2 == 0 else "calibration" for index in range(rows)
            ],
        }
    )


def test_raw_prefix_has_exact_post_onset_plus_preroll_length_and_metadata(tmp_path) -> None:
    path = tmp_path / "speech.wav"
    _write_onset_audio(path)
    manifest = _manifest(path, rows=1)
    options = {
        "prefixes_ms": (150, 200),
        "target_sample_rate": 1_000,
        "pre_roll_ms": 30,
        "onset_config": OnsetConfig(sample_rate=1_000),
    }
    raw = RawPrefixDataset(manifest, training=False, **options)
    spectral = PrefixFeatureDataset(manifest, training=False, n_fft=64, **options)

    first = raw[0]
    second = raw[1]
    assert first["input_values"].shape == (180,)
    assert second["input_values"].shape == (230,)
    assert first["input_num_samples"] == 30 + 150
    assert second["input_num_samples"] == 30 + 200
    assert first["onset_sample"] == 90
    assert first["validation_partition"] == "checkpoint"
    assert first["input_values"].dtype == torch.float32

    raw_metadata = set(first) - {"input_values"}
    spectral_metadata = set(spectral[0]) - {"features"}
    assert raw_metadata == spectral_metadata


def test_raw_evaluation_expands_every_sample_over_every_prefix(tmp_path) -> None:
    path = tmp_path / "speech.wav"
    _write_onset_audio(path)
    dataset = RawPrefixDataset(
        _manifest(path),
        training=False,
        prefixes_ms=(150, 200, 250, 300),
        target_sample_rate=1_000,
        onset_config=OnsetConfig(sample_rate=1_000),
    )
    assert len(dataset) == 8
    assert [dataset.prefix_for_index(index) for index in range(8)] == [
        150,
        200,
        250,
        300,
        150,
        200,
        250,
        300,
    ]
    assert dataset.labels == (0, 0, 0, 0, 1, 1, 1, 1)


def test_raw_training_prefix_selection_is_deterministic_by_epoch(tmp_path) -> None:
    path = tmp_path / "speech.wav"
    _write_onset_audio(path)
    dataset = RawPrefixDataset(
        _manifest(path, rows=32),
        training=True,
        target_sample_rate=1_000,
        onset_config=OnsetConfig(sample_rate=1_000),
        seed=82,
    )
    dataset.set_epoch(5)
    first = tuple(dataset.prefix_for_index(index) for index in range(len(dataset)))
    dataset.set_epoch(5)
    assert tuple(dataset.prefix_for_index(index) for index in range(len(dataset))) == first
    dataset.set_epoch(6)
    second = tuple(dataset.prefix_for_index(index) for index in range(len(dataset)))
    assert first != second
    assert set(first) == {150, 200, 250, 300}


def test_raw_prefix_reuses_stereo_averaging_and_resampling(tmp_path) -> None:
    path = tmp_path / "stereo-500hz.wav"
    _write_onset_audio(path, sample_rate=500, stereo=True)
    dataset = RawPrefixDataset(
        _manifest(path, rows=1),
        training=False,
        evaluation_prefix_ms=150,
        target_sample_rate=1_000,
        pre_roll_ms=30,
        onset_config=OnsetConfig(sample_rate=1_000),
    )
    item = dataset[0]
    assert item["source_channels"] == 2
    assert item["source_sample_rate"] == 500
    assert item["sample_rate"] == 1_000
    assert item["input_values"].shape == (180,)
    assert torch.isfinite(item["input_values"]).all()


def test_raw_failed_onset_is_flagged_and_strict_policy_can_raise(tmp_path) -> None:
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(500, dtype=np.float32), 1_000, subtype="FLOAT")
    manifest = _manifest(path, rows=1)
    options = {
        "training": False,
        "evaluation_prefix_ms": 150,
        "target_sample_rate": 1_000,
        "pre_roll_ms": 30,
        "onset_config": OnsetConfig(sample_rate=1_000),
    }
    recorded = RawPrefixDataset(manifest, **options)[0]
    assert not recorded["onset_detected"]
    assert recorded["onset_sample"] == -1
    assert recorded["confirmation_sample"] == -1
    assert recorded["crop_anchor_sample"] == 0
    assert recorded["input_values"].shape == (180,)

    strict = RawPrefixDataset(manifest, no_onset_policy="raise", **options)
    with pytest.raises(RuntimeError, match="No onset detected"):
        strict[0]


def test_raw_collator_uses_lengths_for_mask_with_adversarial_nonzero_padding(tmp_path) -> None:
    path = tmp_path / "speech.wav"
    _write_onset_audio(path)
    dataset = RawPrefixDataset(
        _manifest(path, rows=1),
        training=False,
        prefixes_ms=(150, 200),
        target_sample_rate=1_000,
        pre_roll_ms=30,
        onset_config=OnsetConfig(sample_rate=1_000),
    )
    short, long = dataset[0], dataset[1]
    batch = collate_raw_prefix_batch([short, long], padding_value=7.25)

    assert batch["input_values"].shape == (2, 230)
    assert batch["input_lengths"].tolist() == [180, 230]
    assert batch["attention_mask"].dtype == torch.long
    assert batch["attention_mask"][0].tolist() == [1] * 180 + [0] * 50
    assert batch["attention_mask"][1].tolist() == [1] * 230
    torch.testing.assert_close(
        batch["input_values"][0, 180:],
        torch.full((50,), 7.25, dtype=torch.float32),
    )
    assert batch["validation_partition"] == ["checkpoint", "checkpoint"]
    assert batch["input_num_samples"].tolist() == batch["input_lengths"].tolist()
