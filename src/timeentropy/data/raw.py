from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from timeentropy.data.augmentation import AugmentationConfig
from timeentropy.data.dataset import (
    NoOnsetPolicy,
    PrefixFeatureDataset,
    _collate_prefix_metadata,
)
from timeentropy.signal.onset import OnsetConfig, OnsetMode


class RawPrefixDataset(PrefixFeatureDataset):
    """Return the same onset-aligned prefixes as unnormalized mono waveforms.

    Prefix selection, crop semantics, onset failure handling, labels, and audit
    metadata are inherited from :class:`PrefixFeatureDataset`. The only changed
    representation is ``input_values``: a one-dimensional float32 waveform for
    WavLM-style encoders instead of a log-STFT image.
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
        augmentation: AugmentationConfig | None = None,
        seed: int = 20260819,
    ) -> None:
        super().__init__(
            manifest,
            training=training,
            evaluation_prefix_ms=evaluation_prefix_ms,
            prefixes_ms=prefixes_ms,
            target_sample_rate=target_sample_rate,
            pre_roll_ms=pre_roll_ms,
            onset_config=onset_config,
            onset_mode=onset_mode,
            no_onset_policy=no_onset_policy,
            augmentation=augmentation,
            seed=seed,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        crop, metadata = self._load_prefix(index)
        return {
            "input_values": torch.from_numpy(np.ascontiguousarray(crop)),
            **metadata,
        }


def collate_raw_prefix_batch(
    batch: Sequence[dict[str, Any]],
    *,
    padding_value: float = 0.0,
) -> dict[str, Any]:
    """Right-pad raw waveforms and create a Hugging Face-style attention mask.

    ``attention_mask`` is an integer tensor with one for a real crop sample and
    zero only for batch-level right padding. Its values are derived from exact
    lengths rather than waveform amplitude, including when a nonzero padding
    value is deliberately supplied for an adversarial check.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if not math.isfinite(padding_value):
        raise ValueError("padding_value must be finite")
    waveforms = [item["input_values"] for item in batch]
    if any(not isinstance(waveform, torch.Tensor) for waveform in waveforms):
        raise TypeError("Every input_values item must be a torch.Tensor")
    if any(waveform.ndim != 1 for waveform in waveforms):
        raise ValueError("Each input_values tensor must have shape [samples]")
    first = waveforms[0]
    if any(waveform.dtype != first.dtype or waveform.device != first.device for waveform in waveforms):
        raise ValueError("Waveform dtype and device must match within a batch")

    input_lengths = torch.tensor(
        [waveform.numel() for waveform in waveforms],
        dtype=torch.long,
        device=first.device,
    )
    maximum_length = int(input_lengths.max().item())
    input_values = first.new_full(
        (len(waveforms), maximum_length),
        fill_value=float(padding_value),
    )
    attention_mask = torch.zeros(
        (len(waveforms), maximum_length),
        dtype=torch.long,
        device=first.device,
    )
    for row, waveform in enumerate(waveforms):
        length = waveform.numel()
        input_values[row, :length] = waveform
        attention_mask[row, :length] = 1

    result = _collate_prefix_metadata(batch)
    result.update(
        {
            "input_values": input_values,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask,
        }
    )
    return result
