from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from timeentropy.data.audio import load_audio
from timeentropy.data.dataset import PrefixFeatureDataset, collate_prefix_features
from timeentropy.robustness.config import RobustnessAnalysis
from timeentropy.robustness.perturbations import (
    add_white_gaussian_noise,
    shifted_crop_anchor,
)
from timeentropy.signal.onset import crop_post_onset_with_preroll, detect_onset


class RobustnessPrefixFeatureDataset(PrefixFeatureDataset):
    """Evaluation-only log-STFT prefixes under one frozen test-time condition."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        analysis: RobustnessAnalysis,
        primary_seed: int,
        **dataset_options: Any,
    ) -> None:
        if analysis.family != "test_time":
            raise ValueError("RobustnessPrefixFeatureDataset accepts test-time analyses only")
        options = dict(dataset_options)
        if options.pop("training", False):
            raise ValueError("Test-time perturbations cannot be used for training")
        if options.pop("augmentation", None) is not None:
            raise ValueError("Training augmentation is forbidden in robustness inference")
        options.pop("seed", None)
        if primary_seed < 0:
            raise ValueError("primary_seed must be non-negative")
        super().__init__(
            manifest,
            training=False,
            augmentation=None,
            seed=primary_seed,
            **options,
        )
        if self.onset_mode != analysis.onset_mode:
            raise ValueError("Dataset onset mode must match the frozen analysis")
        if self.pre_roll_ms != analysis.pre_roll_ms:
            raise ValueError("Dataset pre-roll must match the frozen analysis")
        self.analysis = analysis
        self.primary_seed = int(primary_seed)

    def _load_prefix(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
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
            reference_anchor = int(onset.onset_sample)
        elif self.no_onset_policy == "raise":
            raise RuntimeError(f"No onset detected for sample {sample_id}")
        else:
            reference_anchor = 0

        requested_anchor = reference_anchor
        crop_anchor = reference_anchor
        offset_applied = False
        if self.analysis.perturbation_kind == "onset_shift" and onset.detected:
            if self.analysis.onset_offset_ms is None:
                raise RuntimeError("Onset-shift analysis lacks onset_offset_ms")
            requested_anchor, crop_anchor = shifted_crop_anchor(
                reference_anchor_sample=reference_anchor,
                waveform_samples=len(loaded.waveform),
                offset_ms=self.analysis.onset_offset_ms,
                sample_rate=self.target_sample_rate,
            )
            offset_applied = True

        crop = crop_post_onset_with_preroll(
            loaded.waveform,
            onset_sample=crop_anchor,
            post_onset_ms=prefix_ms,
            pre_roll_ms=self.pre_roll_ms,
            sample_rate=self.target_sample_rate,
        )
        noise_seed_sha256 = ""
        noise_applied = False
        realized_snr_db: float | None = None
        signal_rms: float | None = None
        noise_rms: float | None = None
        if self.analysis.perturbation_kind == "white_noise":
            if self.analysis.snr_db is None:
                raise RuntimeError("White-noise analysis lacks snr_db")
            noise = add_white_gaussian_noise(
                crop,
                primary_seed=self.primary_seed,
                sample_id=sample_id,
                snr_db=self.analysis.snr_db,
            )
            crop = noise.waveform
            noise_seed_sha256 = noise.seed_sha256
            noise_applied = noise.applied
            realized_snr_db = noise.realized_snr_db
            signal_rms = noise.signal_rms
            noise_rms = noise.noise_rms

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
            "onset_sample": crop_anchor if onset.detected else -1,
            "crop_anchor_sample": crop_anchor,
            "confirmation_sample": (
                -1 if onset.confirmation_sample is None else int(onset.confirmation_sample)
            ),
            "onset_threshold": float(onset.threshold),
            "onset_mode": onset.mode,
            "robustness_analysis_id": self.analysis.analysis_id,
            "perturbation_kind": self.analysis.perturbation_kind,
            "snr_db": self.analysis.snr_db,
            "onset_offset_ms": self.analysis.onset_offset_ms,
            "reference_onset_sample": (
                -1 if onset.onset_sample is None else int(onset.onset_sample)
            ),
            "requested_crop_anchor_sample": requested_anchor,
            "onset_offset_applied": offset_applied,
            "noise_seed_sha256": noise_seed_sha256,
            "noise_applied": noise_applied,
            "realized_snr_db": realized_snr_db,
            "signal_rms": signal_rms,
            "noise_rms": noise_rms,
        }
        if "validation_partition" in record:
            value = record["validation_partition"]
            metadata["validation_partition"] = None if pd.isna(value) else str(value)
        return crop, metadata


def collate_robustness_prefix_features(
    batch: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Collate spectral inputs while preserving per-crop perturbation provenance."""

    result = collate_prefix_features(batch)
    for field in (
        "robustness_analysis_id",
        "perturbation_kind",
        "noise_seed_sha256",
    ):
        result[field] = [str(item[field]) for item in batch]
    for field in (
        "reference_onset_sample",
        "requested_crop_anchor_sample",
    ):
        result[field] = torch.tensor([int(item[field]) for item in batch], dtype=torch.long)
    for field in ("onset_offset_applied", "noise_applied"):
        result[field] = torch.tensor([bool(item[field]) for item in batch], dtype=torch.bool)
    for field in ("snr_db", "onset_offset_ms", "realized_snr_db", "signal_rms", "noise_rms"):
        result[field] = [item[field] for item in batch]
    return result
