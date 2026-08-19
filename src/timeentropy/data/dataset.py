from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from timeentropy.data.audio import load_audio
from timeentropy.data.augmentation import (
    AugmentationConfig,
    augment_waveform,
    augmentation_rng,
    mask_spectrogram,
)
from timeentropy.signal.features import log_magnitude_stft
from timeentropy.signal.onset import (
    OnsetConfig,
    OnsetMode,
    crop_post_onset_with_preroll,
    detect_onset,
)

NoOnsetPolicy = Literal["record_detector_failure", "raise"]


class PrefixFeatureDataset(Dataset[dict[str, Any]]):
    """Build onset-aligned log-STFT prefixes for one shared classifier.

    Training mode has one item per utterance and selects one prefix uniformly
    and reproducibly from ``(seed, epoch, sample_id)``. Evaluation mode either
    enumerates every prefix for every utterance or uses one explicitly selected
    prefix. This makes fixed-duration and temporal-policy predictions come from
    the same model and the same feature implementation.

    Under ``record_detector_failure``, a failed detector is recorded in the
    returned metadata and sample zero is used only as an explicit crop fallback.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        training: bool = False,
        evaluation_prefix_ms: int | None = None,
        prefixes_ms: Sequence[int] = (150, 200, 250, 300),
        target_sample_rate: int = 16_000,
        pre_roll_ms: int = 30,
        onset_config: OnsetConfig | None = None,
        onset_mode: OnsetMode = "causal_fixed",
        no_onset_policy: NoOnsetPolicy = "record_detector_failure",
        n_fft: int = 512,
        window_ms: int = 25,
        hop_ms: int = 10,
        normalization_floor: float = 1.0e-6,
        augmentation: AugmentationConfig | None = None,
        seed: int = 20260819,
    ) -> None:
        required = {"sample_id", "path", "speaker_id", "label_id"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        if manifest.empty:
            raise ValueError("Manifest must contain at least one row")
        if manifest["sample_id"].astype(str).duplicated().any():
            raise ValueError("PrefixFeatureDataset requires unique sample_id values")

        prefixes = tuple(int(value) for value in prefixes_ms)
        if not prefixes or any(value <= 0 for value in prefixes):
            raise ValueError("prefixes_ms must contain positive durations")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("prefixes_ms must not contain duplicates")
        if tuple(sorted(prefixes)) != prefixes:
            raise ValueError("prefixes_ms must be in increasing order")
        if target_sample_rate <= 0 or pre_roll_ms < 0:
            raise ValueError("Invalid sample rate or pre-roll duration")
        if training and evaluation_prefix_ms is not None:
            raise ValueError("Training mode cannot use a fixed evaluation prefix")
        if evaluation_prefix_ms is not None and int(evaluation_prefix_ms) not in prefixes:
            raise ValueError("evaluation_prefix_ms must be one of prefixes_ms")
        if no_onset_policy not in ("record_detector_failure", "raise"):
            raise ValueError(f"Unsupported no-onset policy: {no_onset_policy}")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        selected_columns = ["sample_id", "path", "speaker_id", "label_id"]
        if "validation_partition" in manifest.columns:
            selected_columns.append("validation_partition")
        self._records = manifest[selected_columns].copy().to_dict(orient="records")
        self.training = bool(training)
        self.evaluation_prefix_ms = (
            None if evaluation_prefix_ms is None else int(evaluation_prefix_ms)
        )
        self.prefixes_ms = prefixes
        self.target_sample_rate = int(target_sample_rate)
        self.pre_roll_ms = int(pre_roll_ms)
        self.onset_config = onset_config or OnsetConfig(sample_rate=target_sample_rate)
        if self.onset_config.sample_rate != self.target_sample_rate:
            raise ValueError("OnsetConfig sample rate must match target_sample_rate")
        self.onset_mode = onset_mode
        self.no_onset_policy = no_onset_policy
        self.n_fft = int(n_fft)
        self.window_ms = int(window_ms)
        self.hop_ms = int(hop_ms)
        self.normalization_floor = float(normalization_floor)
        self.augmentation = augmentation
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        if not self.training and self.evaluation_prefix_ms is None:
            return len(self._records) * len(self.prefixes_ms)
        return len(self._records)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def labels(self) -> tuple[int, ...]:
        base = tuple(int(record["label_id"]) for record in self._records)
        if not self.training and self.evaluation_prefix_ms is None:
            return tuple(label for label in base for _ in self.prefixes_ms)
        return base

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic per-sample training prefixes for an epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)

    def _training_prefix(self, sample_id: str) -> int:
        material = f"{self.seed}\0{self._epoch}\0{sample_id}".encode()
        digest = hashlib.sha256(material).digest()
        selected = int.from_bytes(digest, byteorder="big", signed=False) % len(
            self.prefixes_ms
        )
        return self.prefixes_ms[selected]

    def _resolve_index(self, index: int) -> tuple[int, int]:
        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if resolved < 0 or resolved >= len(self):
            raise IndexError(index)

        if self.training:
            sample_index = resolved
            sample_id = str(self._records[sample_index]["sample_id"])
            return sample_index, self._training_prefix(sample_id)
        if self.evaluation_prefix_ms is not None:
            return resolved, self.evaluation_prefix_ms
        sample_index, prefix_index = divmod(resolved, len(self.prefixes_ms))
        return sample_index, self.prefixes_ms[prefix_index]

    def prefix_for_index(self, index: int) -> int:
        """Expose prefix selection without loading audio."""

        return self._resolve_index(index)[1]

    def _load_prefix(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Load the common onset-aligned crop and its audit metadata."""

        sample_index, prefix_ms = self._resolve_index(index)
        record = self._records[sample_index]
        sample_id = str(record["sample_id"])
        loaded = load_audio(
            record["path"],
            target_sample_rate=self.target_sample_rate,
            mono=True,
        )
        onset = detect_onset(
            loaded.waveform,
            config=self.onset_config,
            mode=self.onset_mode,
        )
        if onset.detected:
            crop_anchor_sample = int(onset.onset_sample)
        elif self.no_onset_policy == "raise":
            raise RuntimeError(f"No onset detected for sample {sample_id}")
        else:
            crop_anchor_sample = 0

        crop = crop_post_onset_with_preroll(
            loaded.waveform,
            onset_sample=crop_anchor_sample,
            post_onset_ms=prefix_ms,
            pre_roll_ms=self.pre_roll_ms,
            sample_rate=self.target_sample_rate,
        )
        if self.training and self.augmentation is not None:
            crop = augment_waveform(
                crop,
                config=self.augmentation,
                rng=augmentation_rng(
                    seed=self.seed,
                    epoch=self.epoch,
                    sample_id=sample_id,
                    stage="waveform",
                ),
            )
        metadata: dict[str, Any] = {
            "label_id": int(record["label_id"]),
            "sample_id": sample_id,
            "speaker_id": str(record["speaker_id"]),
            "path": str(record["path"]),
            "prefix_ms": prefix_ms,
            "sample_rate": self.target_sample_rate,
            "source_sample_rate": loaded.source_sample_rate,
            "source_channels": loaded.source_channels,
            "input_num_samples": int(crop.shape[0]),
            "onset_detected": onset.detected,
            "onset_sample": -1 if onset.onset_sample is None else int(onset.onset_sample),
            "crop_anchor_sample": crop_anchor_sample,
            "confirmation_sample": (
                -1 if onset.confirmation_sample is None else int(onset.confirmation_sample)
            ),
            "onset_threshold": float(onset.threshold),
            "onset_mode": onset.mode,
        }
        if "validation_partition" in record:
            value = record["validation_partition"]
            metadata["validation_partition"] = None if pd.isna(value) else str(value)
        return crop, metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        crop, metadata = self._load_prefix(index)
        feature = log_magnitude_stft(
            crop,
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            window_ms=self.window_ms,
            hop_ms=self.hop_ms,
            normalization_floor=self.normalization_floor,
        )
        if self.training and self.augmentation is not None:
            feature = mask_spectrogram(
                feature,
                config=self.augmentation,
                rng=augmentation_rng(
                    seed=self.seed,
                    epoch=self.epoch,
                    sample_id=str(metadata["sample_id"]),
                    stage="spectrogram",
                ),
            )

        return {
            "features": torch.from_numpy(np.ascontiguousarray(feature)).unsqueeze(0),
            **metadata,
        }


def _collate_prefix_metadata(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate metadata shared by spectral and waveform prefix batches."""

    integer_fields = (
        "label_id",
        "prefix_ms",
        "sample_rate",
        "source_sample_rate",
        "source_channels",
        "input_num_samples",
        "onset_sample",
        "crop_anchor_sample",
        "confirmation_sample",
    )
    result: dict[str, Any] = {
        "onset_detected": torch.tensor(
            [item["onset_detected"] for item in batch], dtype=torch.bool
        ),
        "onset_threshold": torch.tensor(
            [item["onset_threshold"] for item in batch], dtype=torch.float32
        ),
    }
    for field in integer_fields:
        result[field] = torch.tensor([item[field] for item in batch], dtype=torch.long)
    for field in ("sample_id", "speaker_id", "path", "onset_mode"):
        result[field] = [item[field] for item in batch]
    if any("validation_partition" in item for item in batch):
        if not all("validation_partition" in item for item in batch):
            raise ValueError("validation_partition metadata is inconsistent within a batch")
        result["validation_partition"] = [item["validation_partition"] for item in batch]
    return result


def collate_prefix_features(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad variable prefix features and mark Transformer padding frames."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    features = [item["features"] for item in batch]
    if any(tensor.ndim != 3 for tensor in features):
        raise ValueError("Each feature tensor must have shape [channel, frequency, time]")
    leading_shape = features[0].shape[:-1]
    if any(tensor.shape[:-1] != leading_shape for tensor in features):
        raise ValueError("Channel and frequency dimensions must match within a batch")

    lengths = torch.tensor([tensor.shape[-1] for tensor in features], dtype=torch.long)
    max_length = int(lengths.max().item())
    padded = features[0].new_zeros((len(features), *leading_shape, max_length))
    padding_mask = torch.ones((len(features), max_length), dtype=torch.bool)
    for row, tensor in enumerate(features):
        length = tensor.shape[-1]
        padded[row, ..., :length] = tensor
        padding_mask[row, :length] = False

    result = _collate_prefix_metadata(batch)
    result.update(
        {
            "features": padded,
            "feature_lengths": lengths,
            "padding_mask": padding_mask,
        }
    )
    return result
