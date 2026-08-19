from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from timeentropy.evaluation.calibration import probabilities_from_logits
from timeentropy.evaluation.entropy import normalized_binary_entropy
from timeentropy.training.engine import call_model_for_batch, move_batch_to_device

CANONICAL_PREFIX_COLUMNS = (
    "run_id",
    "sample_id",
    "speaker_id",
    "fold",
    "seed",
    "model_id",
    "split_role",
    "onset_variant",
    "pre_roll_ms",
    "prefix_ms",
    "label_id",
    "logit_control",
    "logit_dysarthric",
    "prob_control",
    "prob_dysarthric",
    "predicted_label",
    "normalized_entropy",
)


def evaluate_all_prefixes(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    run_id: str,
    fold: int,
    seed: int,
    model_id: str,
    split_role: str,
    onset_variant: str,
    pre_roll_ms: int,
    expected_prefixes_ms: Sequence[int] = (150, 200, 250, 300),
) -> pd.DataFrame:
    """Export raw logits and reproducible derived values for every prefix."""

    expected_prefixes = tuple(int(value) for value in expected_prefixes_ms)
    if not expected_prefixes:
        raise ValueError("expected_prefixes_ms must not be empty")
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            logits_tensor = call_model_for_batch(model, batch)
            logits = logits_tensor.detach().cpu().numpy().astype(np.float64)
            if logits.shape[1] != 2:
                raise ValueError("Canonical binary ledger requires exactly two logits")
            probabilities = probabilities_from_logits(logits)
            entropies = normalized_binary_entropy(probabilities)
            predictions = np.argmax(probabilities, axis=1)

            for row_index in range(logits.shape[0]):
                row = {
                    "run_id": run_id,
                    "sample_id": raw_batch["sample_id"][row_index],
                    "speaker_id": raw_batch["speaker_id"][row_index],
                    "fold": int(fold),
                    "seed": int(seed),
                    "model_id": model_id,
                    "split_role": split_role,
                    "onset_variant": onset_variant,
                    "pre_roll_ms": int(pre_roll_ms),
                    "prefix_ms": int(raw_batch["prefix_ms"][row_index]),
                    "label_id": int(raw_batch["label_id"][row_index]),
                    "logit_control": float(logits[row_index, 0]),
                    "logit_dysarthric": float(logits[row_index, 1]),
                    "prob_control": float(probabilities[row_index, 0]),
                    "prob_dysarthric": float(probabilities[row_index, 1]),
                    "predicted_label": int(predictions[row_index]),
                    "normalized_entropy": float(entropies[row_index]),
                    "onset_detected": bool(raw_batch["onset_detected"][row_index]),
                    "onset_sample": int(raw_batch["onset_sample"][row_index]),
                    "confirmation_sample": int(raw_batch["confirmation_sample"][row_index]),
                    "onset_threshold": float(raw_batch["onset_threshold"][row_index]),
                    "source_sample_rate": int(raw_batch["source_sample_rate"][row_index]),
                    "source_channels": int(raw_batch["source_channels"][row_index]),
                    "input_num_samples": int(raw_batch["input_num_samples"][row_index]),
                }
                if "feature_lengths" in raw_batch:
                    row["feature_frames"] = int(raw_batch["feature_lengths"][row_index])
                elif "input_lengths" in raw_batch:
                    raw_input_samples = int(raw_batch["input_lengths"][row_index])
                    if raw_input_samples != row["input_num_samples"]:
                        raise ValueError(
                            "Raw collator input_lengths disagree with input_num_samples"
                        )
                    row["raw_input_samples"] = raw_input_samples
                rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Inference loader for {split_role!r} was empty")
    if frame.duplicated(["sample_id", "prefix_ms"]).any():
        raise ValueError("Inference produced duplicate sample/prefix rows")
    expected_set = frozenset(expected_prefixes)
    for sample_id, group in frame.groupby("sample_id", sort=False):
        observed = frozenset(group["prefix_ms"].astype(int))
        if observed != expected_set:
            raise ValueError(
                f"Sample {sample_id} has prefixes {sorted(observed)}, "
                f"expected {sorted(expected_set)}"
            )
    return frame


def export_prefix_ledger(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write Parquet when its optional engine exists, otherwise write CSV."""

    missing = set(CANONICAL_PREFIX_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prefix ledger is missing canonical columns: {sorted(missing)}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return path
    if suffix != ".parquet":
        raise ValueError("Ledger output must end in .parquet or .csv")
    try:
        frame.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback
