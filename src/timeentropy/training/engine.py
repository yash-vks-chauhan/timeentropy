from __future__ import annotations

import inspect
import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from timeentropy.provenance import sha256_json

DevicePreference = Literal["auto", "cpu", "mps"]
SelectionMetric = Literal["speaker_balanced_macro_f1", "pooled_nll"]


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.00015
    weight_decay: float = 0.0005
    label_smoothing: float = 0.05
    gradient_clip_norm: float = 5.0
    max_epochs: int = 35
    early_stopping_patience: int = 8
    minimum_improvement: float = 0.0
    selection_metric: SelectionMetric = "speaker_balanced_macro_f1"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.max_epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("epoch and patience counts must be positive")
        if self.minimum_improvement < 0.0:
            raise ValueError("minimum_improvement must be non-negative")
        if self.selection_metric not in ("speaker_balanced_macro_f1", "pooled_nll"):
            raise ValueError(f"Unsupported selection_metric: {self.selection_metric}")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_control_draws: int
    train_dysarthric_draws: int
    checkpoint_nll: float
    checkpoint_speaker_balanced_macro_f1: float
    maximum_gradient_norm: float
    improved: bool


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_checkpoint_nll: float
    best_checkpoint_speaker_balanced_macro_f1: float
    selection_metric: SelectionMetric
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: Path


def seed_everything(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Seed Python, NumPy, and Torch for a reproducible fold process."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)


def seed_data_loader_worker(worker_id: int) -> None:
    """Derive library seeds from the worker seed assigned by DataLoader."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(preference: DevicePreference = "auto") -> torch.device:
    """Select Apple MPS when requested/available, otherwise use CPU."""

    if preference not in ("auto", "cpu", "mps"):
        raise ValueError(f"Unsupported device preference: {preference}")
    if preference == "cpu":
        return torch.device("cpu")
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if preference == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available")
    return torch.device("mps" if mps_available else "cpu")


def _accepted_batch_keywords(model: nn.Module) -> set[str]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return set()
    if any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values()):
        return {"feature_lengths", "padding_mask"}
    return {name for name in ("feature_lengths", "padding_mask") if name in parameters}


def _direct_model_call(
    model: nn.Module,
    features: Tensor,
    feature_lengths: Tensor,
    padding_mask: Tensor,
    accepted_keywords: set[str],
) -> Tensor:
    keywords: dict[str, Tensor] = {}
    if "feature_lengths" in accepted_keywords:
        keywords["feature_lengths"] = feature_lengths
    if "padding_mask" in accepted_keywords:
        keywords["padding_mask"] = padding_mask
    logits = model(features, **keywords)
    if not isinstance(logits, Tensor) or logits.ndim != 2:
        raise ValueError("Model must return a B x C logit tensor")
    return logits


def _raw_model_call(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    input_values = batch["input_values"]
    input_lengths = batch["input_lengths"]
    attention_mask = batch["attention_mask"]
    sample_rates = batch["sample_rate"]
    if not isinstance(input_values, Tensor) or not isinstance(input_lengths, Tensor):
        raise TypeError("Raw batch input_values and input_lengths must be tensors")
    if not isinstance(attention_mask, Tensor) or not isinstance(sample_rates, Tensor):
        raise TypeError("Raw batch attention_mask and sample_rate must be tensors")
    if input_values.ndim != 2:
        raise ValueError("Raw batch input_values must have shape B x samples")
    unique_sample_rates = torch.unique(sample_rates.detach().cpu()).tolist()
    if len(unique_sample_rates) != 1:
        raise ValueError("All raw waveforms in a batch must have the same sample rate")
    logits = model(
        input_values,
        sample_lengths=input_lengths,
        attention_mask=attention_mask,
        sample_rate=int(unique_sample_rates[0]),
    )
    if not isinstance(logits, Tensor) or logits.ndim != 2:
        raise ValueError("Model must return a B x C logit tensor")
    return logits


def call_model_for_batch(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    """Dispatch spectral or raw-audio batches without padding leakage."""

    if "input_values" in batch:
        required = {"input_lengths", "attention_mask", "sample_rate"}
        missing = required - set(batch)
        if missing:
            raise KeyError(f"Raw batch is missing fields: {sorted(missing)}")
        return _raw_model_call(model, batch)
    if "features" not in batch:
        raise KeyError("Batch must contain either 'features' or 'input_values'")

    features = batch["features"]
    feature_lengths = batch["feature_lengths"]
    padding_mask = batch["padding_mask"]
    if not isinstance(features, Tensor) or not isinstance(feature_lengths, Tensor):
        raise TypeError("Batch features and feature_lengths must be tensors")
    if not isinstance(padding_mask, Tensor):
        raise TypeError("Batch padding_mask must be a tensor")

    accepted = _accepted_batch_keywords(model)
    if accepted:
        return _direct_model_call(
            model,
            features,
            feature_lengths,
            padding_mask,
            accepted,
        )

    # Frozen reconstructed models originally accepted only a feature tensor.
    # Group different lengths so their global/temporal pooling never sees the
    # right-padding introduced by the collator. Concatenation and reordering
    # preserve gradients back to every group.
    unique_lengths = torch.unique(feature_lengths.detach().cpu(), sorted=True).tolist()
    if len(unique_lengths) == 1:
        length = int(unique_lengths[0])
        return model(features[..., :length])

    group_logits: list[Tensor] = []
    group_indices: list[Tensor] = []
    for raw_length in unique_lengths:
        length = int(raw_length)
        indices = torch.nonzero(feature_lengths == length, as_tuple=False).flatten()
        selected = features.index_select(0, indices)[..., :length]
        logits = model(selected)
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError("Model must return a B x C logit tensor")
        group_logits.append(logits)
        group_indices.append(indices)
    concatenated_logits = torch.cat(group_logits, dim=0)
    concatenated_indices = torch.cat(group_indices, dim=0)
    inverse_order = torch.argsort(concatenated_indices)
    return concatenated_logits.index_select(0, inverse_order)


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields without disturbing string ledger metadata."""

    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def speaker_balanced_macro_f1(
    labels: Sequence[int],
    predictions: Sequence[int],
    speakers: Sequence[str],
) -> float:
    """Macro-F1 in which every speaker contributes equal total weight.

    Each TORGO speaker is entirely control or entirely dysarthric, so a per-speaker F1 is
    undefined. Instead every utterance is weighted by ``1 / n(speaker)``, which prevents a
    single speaker from dominating the statistic. With validation partitions that are
    38-77% one speaker, an unweighted metric mostly measures that speaker.
    """

    if not (len(labels) == len(predictions) == len(speakers)):
        raise ValueError("labels, predictions, and speakers must have equal length")
    if not labels:
        raise ValueError("Checkpoint validation loader is empty")
    counts = Counter(speakers)
    weights = [1.0 / counts[speaker] for speaker in speakers]

    scores = []
    for target in (0, 1):
        true_positive = sum(
            weight
            for weight, label, prediction in zip(weights, labels, predictions, strict=True)
            if label == target and prediction == target
        )
        false_positive = sum(
            weight
            for weight, label, prediction in zip(weights, labels, predictions, strict=True)
            if label != target and prediction == target
        )
        false_negative = sum(
            weight
            for weight, label, prediction in zip(weights, labels, predictions, strict=True)
            if label == target and prediction != target
        )
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
    return sum(scores) / len(scores)


def evaluate_checkpoint_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return unsmoothed mean NLL and speaker-balanced macro-F1 over checkpoint data."""

    model.eval()
    total_loss = 0.0
    total_items = 0
    labels: list[int] = []
    predictions: list[int] = []
    speakers: list[str] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            logits = call_model_for_batch(model, batch)
            batch_labels = batch["label_id"]
            loss = F.cross_entropy(logits, batch_labels, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_items += int(batch_labels.shape[0])
            labels.extend(int(value) for value in batch_labels.detach().cpu())
            predictions.extend(int(value) for value in logits.argmax(dim=-1).detach().cpu())
            speakers.extend(str(value) for value in raw_batch["speaker_id"])
    if total_items == 0:
        raise ValueError("Checkpoint validation loader is empty")
    return total_loss / total_items, speaker_balanced_macro_f1(labels, predictions, speakers)


def evaluate_nll(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute unsmoothed mean negative log likelihood over checkpoint data."""

    return evaluate_checkpoint_metrics(model, loader, device)[0]


def _to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    checkpoint_nll: float,
    training_config: TrainingConfig,
    metadata: Mapping[str, Any],
    provenance_hashes: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_record = asdict(training_config)
    metadata_record = dict(metadata)
    hashes = dict(provenance_hashes)
    hashes["training_config_sha256"] = sha256_json(config_record)
    hashes["checkpoint_metadata_sha256"] = sha256_json(metadata_record)
    payload = {
        "schema_version": 1,
        "epoch": epoch,
        "checkpoint_validation_nll": checkpoint_nll,
        "model_state_dict": _to_cpu(model.state_dict()),
        "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
        "training_config": config_record,
        "metadata": metadata_record,
        "hashes": hashes,
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    checkpoint_loader: DataLoader,
    *,
    device: torch.device,
    config: TrainingConfig,
    checkpoint_path: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    provenance_hashes: Mapping[str, str],
) -> TrainingResult:
    """Train with AdamW and restore the minimum checkpoint-subset NLL state."""

    model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters")
    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    resolved_checkpoint = Path(checkpoint_path)
    history: list[EpochRecord] = []
    best_nll = float("inf")
    best_macro_f1 = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(config.max_epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_items = 0
        sampled_class_counts = torch.zeros(2, dtype=torch.long)
        maximum_gradient_norm = 0.0
        for raw_batch in train_loader:
            batch = move_batch_to_device(raw_batch, device)
            labels = batch["label_id"]
            optimizer.zero_grad(set_to_none=True)
            logits = call_model_for_batch(model, batch)
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=config.label_smoothing,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=config.gradient_clip_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Gradient norm became non-finite")
            maximum_gradient_norm = max(
                maximum_gradient_norm,
                float(gradient_norm.detach().cpu()),
            )
            optimizer.step()
            batch_items = int(labels.shape[0])
            sampled_class_counts += torch.bincount(labels.detach().cpu(), minlength=2)[:2]
            total_loss += float(loss.detach().cpu()) * batch_items
            total_items += batch_items
        if total_items == 0:
            raise ValueError("Training loader is empty")

        checkpoint_nll, checkpoint_macro_f1 = evaluate_checkpoint_metrics(
            model, checkpoint_loader, device
        )
        if config.selection_metric == "speaker_balanced_macro_f1":
            improved = checkpoint_macro_f1 > best_macro_f1 + config.minimum_improvement
        else:
            improved = checkpoint_nll < best_nll - config.minimum_improvement
        if improved:
            best_nll = checkpoint_nll
            best_macro_f1 = checkpoint_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                resolved_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                checkpoint_nll=checkpoint_nll,
                training_config=config,
                metadata=checkpoint_metadata,
                provenance_hashes=provenance_hashes,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=total_loss / total_items,
                train_control_draws=int(sampled_class_counts[0]),
                train_dysarthric_draws=int(sampled_class_counts[1]),
                checkpoint_nll=checkpoint_nll,
                checkpoint_speaker_balanced_macro_f1=checkpoint_macro_f1,
                maximum_gradient_norm=maximum_gradient_norm,
                improved=improved,
            )
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            stopped_early = True
            break

    if best_epoch < 0 or not resolved_checkpoint.is_file():
        raise RuntimeError("Training completed without a valid checkpoint")
    checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_checkpoint_nll=best_nll,
        best_checkpoint_speaker_balanced_macro_f1=best_macro_f1,
        selection_metric=config.selection_metric,
        epochs_completed=len(history),
        stopped_early=stopped_early,
        checkpoint_path=resolved_checkpoint,
    )
