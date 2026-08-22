from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from torch import nn

from timeentropy.config import load_yaml
from timeentropy.data.validation import validation_partition
from timeentropy.evaluation.ledger import validate_prefix_ledger
from timeentropy.models import WavLMReference
from timeentropy.training.ssl_fold import create_ssl_reference_model, run_ssl_fold


class _TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int, layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(layers))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return hidden


class _TinyDownsamplingWavLM(nn.Module):
    """Fast injected backbone implementing the subset used by WavLMReference."""

    stride = 160

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=hidden_size,
            conv_kernel=(1,),
            conv_stride=(self.stride,),
            num_hidden_layers=2,
            _commit_hash="1" * 40,
        )
        self.input_projection = nn.Linear(1, hidden_size)
        self.encoder = _TinyEncoder(hidden_size)
        self.forward_training_modes: list[bool] = []

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> types.SimpleNamespace:
        del attention_mask
        self.forward_training_modes.append(self.training)
        subsampled = input_values[:, :: self.stride].unsqueeze(-1)
        hidden = self.encoder(self.input_projection(subsampled))
        return types.SimpleNamespace(last_hidden_state=hidden)

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return attention_mask[:, :: self.stride][:, :feature_vector_length].to(torch.bool)


def _partitioned_validation_ids() -> list[str]:
    selected: dict[str, str] = {}
    index = 0
    while len(selected) < 3:
        sample_id = f"validation-{index:03d}"
        selected.setdefault(validation_partition(sample_id), sample_id)
        index += 1
    return [selected[name] for name in ("checkpoint", "calibration", "policy")]


def _write_audio(path: Path) -> None:
    sample_rate = 16_000
    silence = np.zeros(round(0.1 * sample_rate), dtype=np.float32)
    speech = 0.1 * np.sin(np.linspace(0.0, 90.0, round(0.6 * sample_rate), dtype=np.float32))
    sf.write(path, np.concatenate([silence, speech]), sample_rate, subtype="FLOAT")


def _write_fold_fixture(tmp_path: Path) -> Path:
    audio_path = tmp_path / "speech.wav"
    _write_audio(audio_path)
    train_ids = [f"train-{index}" for index in range(4)]
    validation_ids = _partitioned_validation_ids()
    test_ids = ["test-0", "test-1"]
    sample_ids = train_ids + validation_ids + test_ids
    roles = ["train"] * len(train_ids) + ["validation"] * 3 + ["test"] * 2
    labels = [0, 1, 0, 1, 0, 1, 0, 0, 1]
    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "path": [str(audio_path)] * len(sample_ids),
            "speaker_id": [f"speaker-{index}" for index in range(len(sample_ids))],
            "label_id": labels,
        }
    )
    fold_manifest = pd.DataFrame(
        {"fold": [0] * len(sample_ids), "sample_id": sample_ids, "role": roles}
    )
    manifest_path = tmp_path / "manifest.csv"
    fold_path = tmp_path / "folds.csv"
    manifest.to_csv(manifest_path, index=False)
    fold_manifest.to_csv(fold_path, index=False)

    config = {
        "schema_version": 1,
        "status": "frozen_before_test_evaluation",
        "experiment_name": "synthetic_ssl",
        "data": {
            "manifest": str(manifest_path),
            "fold_manifest": str(fold_path),
        },
        "audio": {"sample_rate": 16_000, "mono": True},
        "onset": {
            "primary_mode": "causal_fixed",
            "frame_ms": 20,
            "hop_ms": 10,
            "absolute_rms_floor": 0.003,
            "recovered_relative_db_below_peak": 35.0,
            "causal_noise_reference_ms": 100,
            "causal_snr_db": 15.0,
            "consecutive_active_frames": 2,
            "pre_roll_ms": 30,
            "no_onset_policy": "record_detector_failure",
        },
        "prefixes": {"training_ms": [150, 200, 250, 300]},
        "models": {
            "modern_reference_config": {
                "model_id": "wavlm_base_plus",
                "checkpoint_id": "microsoft/wavlm-base-plus",
                "revision": "main",
                "input_representation": "raw_mono_waveform",
                "sample_rate": 16_000,
                "pooling": "masked_mean",
                "fine_tuning": "frozen_backbone",
                "unfreeze_top_n": 0,
                "dropout": 0.0,
            }
        },
        "training": {
            "primary_seed": 29,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 2,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "gradient_clip_norm": 1.0,
            "label_smoothing": 0.0,
            "augmentation": {
                "gain_db": [0.0, 0.0],
                "additive_noise_probability": 0.0,
                "additive_noise_snr_db": [20.0, 35.0],
                "spectrogram_mask_probability": 0.0,
                "frequency_mask_max_bins": 0,
                "time_mask_max_frames": 0,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _read_ledger(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def test_frozen_config_has_precise_modern_reference_identity() -> None:
    reference = load_yaml("configs/rebuild_v1.yaml")["models"]["modern_reference_config"]
    assert reference == {
        "model_id": "wavlm_base_plus",
        "checkpoint_id": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "input_representation": "raw_mono_waveform",
        "sample_rate": 16_000,
        "pooling": "masked_mean",
        "fine_tuning": "frozen_backbone",
        "unfreeze_top_n": 0,
        "dropout": 0.30,
    }


def test_model_loader_defaults_to_local_cache_only(tmp_path, monkeypatch) -> None:
    config_path = _write_fold_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(cls, **kwargs):
        calls.append(kwargs)
        return cls(
            _TinyDownsamplingWavLM(),
            checkpoint_id=str(kwargs["checkpoint_id"]),
            checkpoint_revision=kwargs["revision"],
            dropout=float(kwargs["dropout"]),
        )

    monkeypatch.setattr(WavLMReference, "from_pretrained", classmethod(fake_from_pretrained))
    create_ssl_reference_model(config)
    create_ssl_reference_model(config, allow_download=True)

    assert calls[0]["local_files_only"] is True
    assert calls[1]["local_files_only"] is False
    assert calls[0]["checkpoint_id"] == "microsoft/wavlm-base-plus"
    assert calls[0]["revision"] == "main"


def test_offline_injected_ssl_fold_exports_canonical_auditable_ledger(
    tmp_path, monkeypatch
) -> None:
    config_path = _write_fold_fixture(tmp_path)
    backbone = _TinyDownsamplingWavLM()

    def forbidden_pretrained_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("injected fold must not load or download pretrained weights")

    monkeypatch.setattr(WavLMReference, "from_pretrained", forbidden_pretrained_load)
    result = run_ssl_fold(
        config_path,
        fold=0,
        device_preference="cpu",
        output_root=tmp_path / "runs",
        backbone=backbone,
    )

    ledger = _read_ledger(result.ledger_path)
    report = validate_prefix_ledger(ledger)
    assert report.rows == 20
    assert ledger.groupby("sample_id")["prefix_ms"].nunique().eq(4).all()
    assert set(ledger["split_role"]) == {
        "checkpoint",
        "calibration",
        "policy_tuning",
        "test",
    }
    assert ledger["raw_input_samples"].equals(ledger["input_num_samples"])
    assert {
        150: [2_880],
        200: [3_680],
        250: [4_480],
        300: [5_280],
    } == {
        int(prefix): sorted(group["raw_input_samples"].unique().tolist())
        for prefix, group in ledger.groupby("prefix_ms")
    }

    summary = json.loads(result.summary_path.read_text())
    checkpoint = summary["pretrained_checkpoint"]
    assert checkpoint["checkpoint_id"] == "microsoft/wavlm-base-plus"
    assert checkpoint["requested_revision"] == "main"
    assert checkpoint["resolved_revision"] == "1" * 40
    assert checkpoint["local_files_only"] is True
    assert checkpoint["download_authorized"] is False
    assert checkpoint["fine_tuning_status"] == "frozen_backbone"
    assert summary["model"]["backbone_trainable_parameters"] == 0
    assert summary["model"]["trainable_parameters"] == summary["model"]["head_trainable_parameters"]
    assert summary["training"]["class_balance"] == "weighted_random_sampler"
    assert summary["training"]["original_train_class_counts"] == {
        "control": 2,
        "dysarthric": 2,
    }
    assert summary["training"]["sampling"]["draws_per_epoch"] == 4
    assert summary["training"]["optimizer_parameter_scope"] == "requires_grad_only"
    assert summary["training"]["validation_partition_method"] == "sha256_sample_id"
    assert (
        summary["history"][0]["train_control_draws"]
        + summary["history"][0]["train_dysarthric_draws"]
        == 4
    )
    assert all(len(value) == 64 for value in summary["hashes"].values())
    assert not any(backbone.forward_training_modes)

    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    optimizer_parameter_count = sum(
        len(group["params"]) for group in payload["optimizer_state_dict"]["param_groups"]
    )
    assert optimizer_parameter_count == 2
    assert payload["metadata"]["pretrained_checkpoint"] == checkpoint
    assert payload["metadata"]["validation_partition_method"] == "sha256_sample_id"


def test_ssl_training_config_uses_the_configured_selection_metric() -> None:
    """The SSL path must honour the frozen config, not silently fall back to a default.

    It previously never passed selection_metric, so it happened to match the compact models
    only because the dataclass default agreed with the config. Changing the config would have
    silently left the pretrained reference on a different checkpoint-selection rule, which is
    exactly the confound a condition-matched comparison must avoid.
    """

    import inspect

    from timeentropy.training import ssl_fold

    source = inspect.getsource(ssl_fold)
    assert "checkpoint_selection_metric" in source, (
        "ssl_fold must read training.checkpoint_selection_metric from the config"
    )
    assert "selection_metric=" in source
    # The run summary must record what was actually used.
    assert "best_checkpoint_speaker_balanced_macro_f1" in source
