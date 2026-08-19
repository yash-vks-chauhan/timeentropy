from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from timeentropy.data.dataset import collate_prefix_features
from timeentropy.evaluation.ledger import validate_prefix_ledger
from timeentropy.training.engine import (
    TrainingConfig,
    call_model_for_batch,
    seed_everything,
    select_device,
    train_with_early_stopping,
)
from timeentropy.training.inference import evaluate_all_prefixes, export_prefix_ledger


class SyntheticPrefixDataset(Dataset):
    def __init__(self, *, all_prefixes: bool) -> None:
        self.prefixes = (150, 200, 250, 300) if all_prefixes else (150,)
        self.samples = tuple(
            (f"sample-{index}", index % 2, 13 + (index % 4) * 5)
            for index in range(8)
        )
        self.epoch = -1

    def __len__(self) -> int:
        return len(self.samples) * len(self.prefixes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        sample_index, prefix_index = divmod(index, len(self.prefixes))
        sample_id, label, base_frames = self.samples[sample_index]
        prefix = self.prefixes[prefix_index]
        frames = base_frames if len(self.prefixes) == 1 else 13 + prefix_index * 5
        generator = torch.Generator().manual_seed(sample_index * 31 + prefix_index)
        feature = torch.randn((1, 257, frames), generator=generator) * 0.05
        feature[:, :3] += float(label)
        return {
            "features": feature,
            "label_id": label,
            "sample_id": sample_id,
            "speaker_id": f"speaker-{sample_index}",
            "path": f"/{sample_id}.wav",
            "prefix_ms": prefix,
            "sample_rate": 16_000,
            "source_sample_rate": 16_000,
            "source_channels": 1,
            "input_num_samples": prefix * 16,
            "onset_detected": True,
            "onset_sample": 10,
            "crop_anchor_sample": 10,
            "confirmation_sample": 30,
            "onset_threshold": 0.003,
            "onset_mode": "causal_fixed",
        }


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return self.classifier(pooled)


class MaskAwareClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 2)
        self.received_lengths: torch.Tensor | None = None
        self.received_mask: torch.Tensor | None = None

    def forward(
        self,
        features: torch.Tensor,
        *,
        feature_lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.received_lengths = feature_lengths.detach().cpu()
        self.received_mask = padding_mask.detach().cpu()
        pooled = features.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return self.classifier(pooled)


def _loader(dataset: Dataset, batch_size: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_prefix_features)


def test_call_adapter_passes_masks_only_when_supported() -> None:
    batch = next(iter(_loader(SyntheticPrefixDataset(all_prefixes=False))))
    aware = MaskAwareClassifier()
    logits = call_model_for_batch(aware, batch)
    assert logits.shape == (4, 2)
    assert torch.equal(aware.received_lengths, batch["feature_lengths"])
    assert torch.equal(aware.received_mask, batch["padding_mask"])

    plain = TinyClassifier()
    plain_logits = call_model_for_batch(plain, batch)
    individually = torch.cat(
        [plain(batch["features"][row : row + 1, ..., :length]) for row, length in enumerate(batch["feature_lengths"])]
    )
    torch.testing.assert_close(plain_logits, individually)


def test_training_is_reproducible_stops_early_and_hashes_checkpoint(tmp_path) -> None:
    def run(directory: Path):
        seed_everything(71)
        model = TinyClassifier()
        train_dataset = SyntheticPrefixDataset(all_prefixes=False)
        config = TrainingConfig(
            learning_rate=0.01,
            weight_decay=0.0005,
            label_smoothing=0.05,
            gradient_clip_norm=0.25,
            max_epochs=5,
            early_stopping_patience=1,
            minimum_improvement=100.0,
        )
        result = train_with_early_stopping(
            model,
            _loader(train_dataset),
            _loader(SyntheticPrefixDataset(all_prefixes=False)),
            device=torch.device("cpu"),
            config=config,
            checkpoint_path=directory / "best.pt",
            checkpoint_metadata={"run_id": "synthetic", "fold": 0},
            provenance_hashes={"experiment_config_sha256": "a" * 64},
        )
        return model, train_dataset, result

    first_model, first_dataset, first = run(tmp_path / "first")
    second_model, second_dataset, second = run(tmp_path / "second")
    assert first.stopped_early and first.epochs_completed == 2 and first.best_epoch == 0
    assert first_dataset.epoch == second_dataset.epoch == 1
    assert first.history == second.history
    for first_value, second_value in zip(
        first_model.state_dict().values(), second_model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(first_value, second_value)

    payload = torch.load(first.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["run_id"] == "synthetic"
    assert payload["hashes"]["experiment_config_sha256"] == "a" * 64
    assert len(payload["hashes"]["training_config_sha256"]) == 64
    assert len(payload["hashes"]["checkpoint_metadata_sha256"]) == 64


def test_all_prefix_inference_and_csv_fallback(tmp_path, monkeypatch) -> None:
    seed_everything(3)
    model = TinyClassifier()
    frame = evaluate_all_prefixes(
        model,
        _loader(SyntheticPrefixDataset(all_prefixes=True), batch_size=7),
        device=torch.device("cpu"),
        run_id="synthetic-fold-0",
        fold=0,
        seed=3,
        model_id="tiny",
        split_role="test",
        onset_variant="causal_fixed",
        pre_roll_ms=30,
    )
    assert len(frame) == 32
    assert frame.groupby("sample_id")["prefix_ms"].nunique().eq(4).all()
    report = validate_prefix_ledger(frame)
    assert report.rows == 32 and report.samples == 8
    np.testing.assert_allclose(
        frame[["prob_control", "prob_dysarthric"]].sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )

    def unavailable(*args, **kwargs):
        del args, kwargs
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", unavailable)
    output = export_prefix_ledger(frame, tmp_path / "ledger.parquet")
    assert output == tmp_path / "ledger.csv"
    assert output.is_file()
    restored = pd.read_csv(output)
    assert len(restored) == len(frame)


def test_device_selection_has_explicit_cpu_and_rejects_unknown() -> None:
    assert select_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="Unsupported"):
        select_device("gpu")  # type: ignore[arg-type]
