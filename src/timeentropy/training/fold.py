from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from timeentropy.config import PROJECT_ROOT, load_yaml, project_path
from timeentropy.data import (
    AugmentationConfig,
    PrefixFeatureDataset,
    attach_validation_partitions,
    collate_prefix_features,
    make_class_balanced_sampler,
)
from timeentropy.evaluation.ledger import validate_prefix_ledger
from timeentropy.models import CNNBiLSTM, ShortPrefixTransformer, SmallCNN
from timeentropy.provenance import environment_record, sha256_file, sha256_json
from timeentropy.signal.onset import OnsetConfig
from timeentropy.training.engine import (
    DevicePreference,
    TrainingConfig,
    seed_data_loader_worker,
    seed_everything,
    select_device,
    train_with_early_stopping,
)
from timeentropy.training.inference import evaluate_all_prefixes, export_prefix_ledger


@dataclass(frozen=True)
class FoldRunResult:
    run_id: str
    fold: int
    model_id: str
    seed: int
    device: str
    checkpoint_path: Path
    ledger_path: Path
    summary_path: Path
    best_epoch: int
    best_checkpoint_nll: float


def fold_run_id(
    config: dict[str, Any],
    *,
    fold: int,
    model_id: str,
    seed: int,
) -> str:
    """Return the canonical run identifier shared by all fold launchers.

    Keeping this small identity function separate prevents orchestration plans
    from drifting away from the paths produced by the actual training code.
    """

    if fold < 0:
        raise ValueError("fold must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not model_id:
        raise ValueError("model_id must be non-empty")
    return (
        f"{config['experiment_name']}-fold{fold}-{model_id}-seed{seed}-"
        f"{config['onset']['primary_mode']}"
    )


def create_model(model_id: str, *, dropout: float) -> nn.Module:
    """Instantiate a frozen primary architecture from its config identifier."""

    factories: dict[str, type[nn.Module]] = {
        "small_cnn": SmallCNN,
        "cnn_bilstm": CNNBiLSTM,
        "transformer": ShortPrefixTransformer,
    }
    try:
        factory = factories[model_id]
    except KeyError as error:
        raise ValueError(f"Unsupported primary model: {model_id}") from error
    return factory(dropout=dropout)


def _make_loader(
    dataset: PrefixFeatureDataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    balanced: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = (
        make_class_balanced_sampler(dataset.labels, seed=seed) if balanced else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_prefix_features,
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        # Workers are recreated at each epoch so they receive the training
        # dataset's updated deterministic epoch-prefix assignment.
        persistent_workers=False,
    )


def _dataset_options(config: dict[str, Any], *, seed: int) -> dict[str, Any]:
    onset = config["onset"]
    audio = config["audio"]
    features = config["features"]
    augmentation = config["training"]["augmentation"]
    return {
        "prefixes_ms": tuple(int(value) for value in config["prefixes"]["training_ms"]),
        "target_sample_rate": int(audio["sample_rate"]),
        "pre_roll_ms": int(onset["pre_roll_ms"]),
        "onset_config": OnsetConfig(
            sample_rate=int(audio["sample_rate"]),
            frame_ms=int(onset["frame_ms"]),
            hop_ms=int(onset["hop_ms"]),
            absolute_rms_floor=float(onset["absolute_rms_floor"]),
            recovered_relative_db_below_peak=float(
                onset["recovered_relative_db_below_peak"]
            ),
            causal_noise_reference_ms=int(onset["causal_noise_reference_ms"]),
            causal_snr_db=float(onset["causal_snr_db"]),
            consecutive_active_frames=int(onset["consecutive_active_frames"]),
        ),
        "onset_mode": str(onset["primary_mode"]),
        "no_onset_policy": str(onset["no_onset_policy"]),
        "n_fft": int(features["n_fft"]),
        "window_ms": int(features["window_ms"]),
        "hop_ms": int(features["hop_ms"]),
        "normalization_floor": float(features["normalization_floor"]),
        "augmentation": AugmentationConfig(
            gain_db=tuple(float(value) for value in augmentation["gain_db"]),
            additive_noise_probability=float(
                augmentation["additive_noise_probability"]
            ),
            additive_noise_snr_db=tuple(
                float(value) for value in augmentation["additive_noise_snr_db"]
            ),
            spectrogram_mask_probability=float(
                augmentation["spectrogram_mask_probability"]
            ),
            frequency_mask_max_bins=int(augmentation["frequency_mask_max_bins"]),
            time_mask_max_frames=int(augmentation["time_mask_max_frames"]),
        ),
        "seed": seed,
    }


def _load_fold_frames(config: dict[str, Any], fold: int) -> tuple[pd.DataFrame, ...]:
    manifest_path = project_path(config["data"]["manifest"])
    fold_path = project_path(config["data"]["fold_manifest"])
    manifest = pd.read_csv(manifest_path)
    roles = pd.read_csv(fold_path)
    roles = roles.loc[roles["fold"].eq(fold), ["sample_id", "role"]]
    if roles.empty:
        raise ValueError(f"Fold {fold} is absent from the fold manifest")
    if roles["sample_id"].duplicated().any():
        raise ValueError(f"Fold {fold} contains duplicate role rows")
    frame = manifest.merge(roles, on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != len(manifest):
        raise ValueError(f"Fold {fold} does not cover the complete manifest")

    train = frame.loc[frame["role"].eq("train")].copy()
    validation = attach_validation_partitions(
        frame.loc[frame["role"].eq("validation")].copy()
    )
    test = frame.loc[frame["role"].eq("test")].copy()
    checkpoint = validation.loc[
        validation["validation_partition"].eq("checkpoint")
    ].copy()
    calibration = validation.loc[
        validation["validation_partition"].eq("calibration")
    ].copy()
    policy = validation.loc[validation["validation_partition"].eq("policy")].copy()
    for name, split in (
        ("train", train),
        ("checkpoint", checkpoint),
        ("calibration", calibration),
        ("policy_tuning", policy),
        ("test", test),
    ):
        if split.empty:
            raise ValueError(f"Fold {fold} has an empty {name} split")
    return train, checkpoint, calibration, policy, test


def run_fold(
    config_path: str | Path,
    *,
    fold: int,
    model_id: str,
    seed: int | None = None,
    device_preference: DevicePreference = "auto",
    output_root: str | Path = "artifacts/runs",
    num_workers: int = 0,
) -> FoldRunResult:
    """Train one frozen speaker fold and export all raw validation/test logits."""

    config = load_yaml(config_path)
    if config.get("status") != "frozen_before_test_evaluation":
        raise ValueError("Experiment configuration is not marked frozen")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    training = config["training"]
    resolved_seed = int(training["primary_seed"] if seed is None else seed)
    seed_everything(resolved_seed)
    device = select_device(device_preference)
    train, checkpoint, calibration, policy, test = _load_fold_frames(config, fold)

    dataset_options = _dataset_options(config, seed=resolved_seed)
    train_dataset = PrefixFeatureDataset(train, training=True, **dataset_options)
    checkpoint_dataset = PrefixFeatureDataset(checkpoint, training=False, **dataset_options)
    batch_size = int(training["batch_size"])
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        seed=resolved_seed,
        num_workers=num_workers,
        balanced=True,
    )
    checkpoint_loader = _make_loader(
        checkpoint_dataset,
        batch_size=batch_size,
        seed=resolved_seed,
        num_workers=num_workers,
        balanced=False,
    )

    model = create_model(model_id, dropout=float(training["dropout"]))
    model_metadata = asdict(model.metadata)
    run_id = fold_run_id(
        config,
        fold=fold,
        model_id=model_id,
        seed=resolved_seed,
    )
    run_directory = project_path(output_root) / run_id
    checkpoint_path = run_directory / "best_checkpoint.pt"
    config_path_resolved = project_path(config_path)
    manifest_path = project_path(config["data"]["manifest"])
    fold_manifest_path = project_path(config["data"]["fold_manifest"])
    hashes = {
        "experiment_config_sha256": sha256_json(config),
        "experiment_config_file_sha256": sha256_file(config_path_resolved),
        "manifest_sha256": sha256_file(manifest_path),
        "fold_manifest_sha256": sha256_file(fold_manifest_path),
        "model_architecture_sha256": sha256_json(model_metadata),
    }
    checkpoint_metadata = {
        "run_id": run_id,
        "fold": fold,
        "seed": resolved_seed,
        "model_id": model_id,
        "device": str(device),
        "onset_variant": config["onset"]["primary_mode"],
        "model": model_metadata,
        "original_train_class_counts": {
            "control": int(train["label_id"].eq(0).sum()),
            "dysarthric": int(train["label_id"].eq(1).sum()),
        },
        "sampling": {
            "method": "inverse_class_frequency_with_replacement",
            "draws_per_epoch": len(train),
            "expected_control_fraction": 0.5,
            "expected_dysarthric_fraction": 0.5,
        },
        "environment": environment_record(PROJECT_ROOT),
    }
    training_config = TrainingConfig(
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        label_smoothing=float(training["label_smoothing"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        max_epochs=int(training["max_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
    )
    result = train_with_early_stopping(
        model,
        train_loader,
        checkpoint_loader,
        device=device,
        config=training_config,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        provenance_hashes=hashes,
    )

    ledger_parts: list[pd.DataFrame] = []
    prefixes = tuple(int(value) for value in config["prefixes"]["training_ms"])
    for split_role, split_frame in (
        ("checkpoint", checkpoint),
        ("calibration", calibration),
        ("policy_tuning", policy),
        ("test", test),
    ):
        dataset = PrefixFeatureDataset(split_frame, training=False, **dataset_options)
        loader = _make_loader(
            dataset,
            batch_size=batch_size,
            seed=resolved_seed,
            num_workers=num_workers,
            balanced=False,
        )
        ledger_parts.append(
            evaluate_all_prefixes(
                model,
                loader,
                device=device,
                run_id=run_id,
                fold=fold,
                seed=resolved_seed,
                model_id=model_id,
                split_role=split_role,
                onset_variant=str(config["onset"]["primary_mode"]),
                pre_roll_ms=int(config["onset"]["pre_roll_ms"]),
                expected_prefixes_ms=prefixes,
            )
        )
    ledger = pd.concat(ledger_parts, ignore_index=True).sort_values(
        ["split_role", "sample_id", "prefix_ms"], ignore_index=True
    )
    validate_prefix_ledger(ledger, expected_prefixes_ms=prefixes)
    ledger_path = export_prefix_ledger(
        ledger,
        run_directory / "prefix_predictions.parquet",
    )

    artifact_hashes = {
        **hashes,
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "prefix_ledger_sha256": sha256_file(ledger_path),
    }
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "fold": fold,
        "model_id": model_id,
        "seed": resolved_seed,
        "device": str(device),
        "best_epoch": result.best_epoch,
        "best_checkpoint_nll": result.best_checkpoint_nll,
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        "checkpoint_path": str(checkpoint_path),
        "ledger_path": str(ledger_path),
        "hashes": artifact_hashes,
        "original_train_class_counts": checkpoint_metadata[
            "original_train_class_counts"
        ],
        "sampling": checkpoint_metadata["sampling"],
        "history": [asdict(record) for record in result.history],
    }
    summary_path = run_directory / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return FoldRunResult(
        run_id=run_id,
        fold=fold,
        model_id=model_id,
        seed=resolved_seed,
        device=str(device),
        checkpoint_path=checkpoint_path,
        ledger_path=ledger_path,
        summary_path=summary_path,
        best_epoch=result.best_epoch,
        best_checkpoint_nll=result.best_checkpoint_nll,
    )
