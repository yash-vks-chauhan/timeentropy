"""Reproducible raw-waveform fold runner for the WavLM reference model."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from timeentropy.config import PROJECT_ROOT, load_yaml, project_path
from timeentropy.data import (
    AugmentationConfig,
    RawPrefixDataset,
    collate_raw_prefix_batch,
    make_class_balanced_sampler,
)
from timeentropy.evaluation.ledger import validate_prefix_ledger
from timeentropy.models import WavLMReference
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
from timeentropy.training.fold import FoldRunResult, _load_fold_frames, fold_run_id
from timeentropy.training.inference import evaluate_all_prefixes, export_prefix_ledger


def _make_raw_loader(
    dataset: RawPrefixDataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    balanced: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = make_class_balanced_sampler(dataset.labels, seed=seed) if balanced else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_raw_prefix_batch,
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        persistent_workers=False,
    )


def _raw_dataset_options(config: dict[str, Any], *, seed: int) -> dict[str, Any]:
    onset = config["onset"]
    audio = config["audio"]
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
            recovered_relative_db_below_peak=float(onset["recovered_relative_db_below_peak"]),
            causal_noise_reference_ms=int(onset["causal_noise_reference_ms"]),
            causal_snr_db=float(onset["causal_snr_db"]),
            consecutive_active_frames=int(onset["consecutive_active_frames"]),
        ),
        "onset_mode": str(onset["primary_mode"]),
        "no_onset_policy": str(onset["no_onset_policy"]),
        "augmentation": AugmentationConfig(
            gain_db=tuple(float(value) for value in augmentation["gain_db"]),
            additive_noise_probability=float(augmentation["additive_noise_probability"]),
            additive_noise_snr_db=tuple(
                float(value) for value in augmentation["additive_noise_snr_db"]
            ),
            # Raw WavLM input has no spectrogram masking stage. These values are
            # retained in provenance through the frozen experiment config.
            spectrogram_mask_probability=float(augmentation["spectrogram_mask_probability"]),
            frequency_mask_max_bins=int(augmentation["frequency_mask_max_bins"]),
            time_mask_max_frames=int(augmentation["time_mask_max_frames"]),
        ),
        "seed": seed,
    }


def _modern_reference_config(config: dict[str, Any]) -> dict[str, Any]:
    try:
        reference = dict(config["models"]["modern_reference_config"])
    except (KeyError, TypeError) as error:
        raise ValueError("Config must define models.modern_reference_config") from error
    required = {
        "model_id",
        "checkpoint_id",
        "revision",
        "input_representation",
        "pooling",
        "fine_tuning",
        "unfreeze_top_n",
        "dropout",
        "sample_rate",
    }
    missing = required - set(reference)
    if missing:
        raise ValueError(f"Modern reference config is missing: {sorted(missing)}")
    if not isinstance(reference["checkpoint_id"], str) or not reference["checkpoint_id"]:
        raise ValueError("Modern reference checkpoint_id must be a non-empty string")
    if reference["revision"] is not None and (
        not isinstance(reference["revision"], str) or not reference["revision"]
    ):
        raise ValueError("Modern reference revision must be a string or null")
    if reference["input_representation"] != "raw_mono_waveform":
        raise ValueError("Modern reference input_representation must be raw_mono_waveform")
    if reference["pooling"] != "masked_mean":
        raise ValueError("Modern reference pooling must be masked_mean")
    if reference["fine_tuning"] not in ("frozen_backbone", "unfreeze_top_n"):
        raise ValueError("Unsupported modern reference fine_tuning mode")
    configured_top_n = int(reference["unfreeze_top_n"])
    if configured_top_n < 0:
        raise ValueError("Modern reference unfreeze_top_n cannot be negative")
    if reference["fine_tuning"] == "frozen_backbone" and configured_top_n != 0:
        raise ValueError("frozen_backbone requires unfreeze_top_n=0")
    if reference["fine_tuning"] == "unfreeze_top_n" and configured_top_n < 1:
        raise ValueError("unfreeze_top_n mode requires a positive layer count")
    if not 0.0 <= float(reference["dropout"]) < 1.0:
        raise ValueError("Modern reference dropout must be in [0, 1)")
    if int(reference["sample_rate"]) != 16_000:
        raise ValueError("WavLM modern reference requires sample_rate=16000")
    return reference


def _resolved_checkpoint_revision(model: WavLMReference) -> str | None:
    """Read the immutable revision recorded by Transformers after loading."""

    for owner in (model.backbone_config, model.backbone):
        for attribute in ("_commit_hash", "commit_hash"):
            value = getattr(owner, attribute, None)
            if isinstance(value, str) and value:
                return value
    return None


def create_ssl_reference_model(
    config: dict[str, Any],
    *,
    allow_download: bool = False,
    unfreeze_top_n: int | None = None,
    backbone: nn.Module | None = None,
    backbone_config: Any | None = None,
) -> tuple[WavLMReference, dict[str, Any]]:
    """Create the configured WavLM model without implicit network access."""

    reference = _modern_reference_config(config)
    requested_revision = reference["revision"]
    configured_top_n = int(reference["unfreeze_top_n"])
    resolved_top_n = configured_top_n if unfreeze_top_n is None else int(unfreeze_top_n)
    if resolved_top_n < 0:
        raise ValueError("unfreeze_top_n cannot be negative")
    fine_tuning = "unfreeze_top_n" if resolved_top_n else "frozen_backbone"

    model_arguments = {
        "checkpoint_id": str(reference["checkpoint_id"]),
        "dropout": float(reference["dropout"]),
        "pooling_method": str(reference["pooling"]),
        "fine_tuning": fine_tuning,
        "unfreeze_top_n": resolved_top_n,
    }
    if backbone is not None:
        model = WavLMReference(
            backbone,
            config=backbone_config,
            checkpoint_revision=requested_revision,
            backbone_source="injected_for_fold_run",
            **model_arguments,
        )
    else:
        try:
            model = WavLMReference.from_pretrained(
                revision=requested_revision,
                local_files_only=not allow_download,
                **model_arguments,
            )
        except OSError as error:
            if allow_download:
                raise
            raise RuntimeError(
                "The configured WavLM checkpoint is absent from the local Hugging Face "
                "cache. Re-run with --allow-download to authorize network retrieval."
            ) from error

    resolved_revision = _resolved_checkpoint_revision(model)
    if (
        resolved_revision is None
        and isinstance(requested_revision, str)
        and len(requested_revision) == 40
        and all(character in "0123456789abcdefABCDEF" for character in requested_revision)
    ):
        resolved_revision = requested_revision
    if (
        isinstance(requested_revision, str)
        and len(requested_revision) == 40
        and resolved_revision != requested_revision
    ):
        raise RuntimeError(
            "Loaded WavLM revision does not match the immutable revision requested by config: "
            f"requested {requested_revision}, resolved {resolved_revision}"
        )
    checkpoint_record = {
        "checkpoint_id": str(reference["checkpoint_id"]),
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "local_files_only": not allow_download,
        "download_authorized": bool(allow_download),
        "pooling_method": model.pooling_method,
        "fine_tuning_status": model.metadata.fine_tuning_status,
    }
    return model, checkpoint_record


def _split_membership_hashes(
    train: pd.DataFrame,
    checkpoint: pd.DataFrame,
    calibration: pd.DataFrame,
    policy: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, str]:
    frames = {
        "train": train,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "policy_tuning": policy,
        "test": test,
    }
    return {
        f"{name}_sample_ids_sha256": sha256_json(sorted(frame["sample_id"].astype(str).tolist()))
        for name, frame in frames.items()
    }


def run_ssl_fold(
    config_path: str | Path,
    *,
    fold: int,
    seed: int | None = None,
    device_preference: DevicePreference = "auto",
    output_root: str | Path = "artifacts/runs",
    num_workers: int = 0,
    allow_download: bool = False,
    unfreeze_top_n: int | None = None,
    batch_size: int | None = None,
    backbone: nn.Module | None = None,
    backbone_config: Any | None = None,
) -> FoldRunResult:
    """Train one WavLM speaker fold and export a canonical four-prefix ledger.

    ``allow_download=False`` is deliberately the default. Synthetic/offline
    tests pass an injected backbone and therefore never import Transformers.
    """

    config = load_yaml(config_path)
    if config.get("status") != "frozen_before_test_evaluation":
        raise ValueError("Experiment configuration is not marked frozen")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if int(config["audio"]["sample_rate"]) != 16_000:
        raise ValueError("WavLM fold runner requires audio.sample_rate=16000")
    reference = _modern_reference_config(config)
    if int(reference["sample_rate"]) != int(config["audio"]["sample_rate"]):
        raise ValueError("Modern reference and audio sample rates must agree")

    training = config["training"]
    resolved_seed = int(training["primary_seed"] if seed is None else seed)
    resolved_batch_size = int(training["batch_size"] if batch_size is None else batch_size)
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive")
    seed_everything(resolved_seed)
    device = select_device(device_preference)
    train, checkpoint, calibration, policy, test = _load_fold_frames(config, fold)

    dataset_options = _raw_dataset_options(config, seed=resolved_seed)
    train_dataset = RawPrefixDataset(train, training=True, **dataset_options)
    checkpoint_dataset = RawPrefixDataset(checkpoint, training=False, **dataset_options)
    train_loader = _make_raw_loader(
        train_dataset,
        batch_size=resolved_batch_size,
        seed=resolved_seed,
        num_workers=num_workers,
        balanced=True,
    )
    checkpoint_loader = _make_raw_loader(
        checkpoint_dataset,
        batch_size=resolved_batch_size,
        seed=resolved_seed,
        num_workers=num_workers,
        balanced=False,
    )

    model, checkpoint_record = create_ssl_reference_model(
        config,
        allow_download=allow_download,
        unfreeze_top_n=unfreeze_top_n,
        backbone=backbone,
        backbone_config=backbone_config,
    )
    model_metadata = asdict(model.metadata)
    model_id = str(reference["model_id"])
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
        "checkpoint_identity_sha256": sha256_json(checkpoint_record),
        **_split_membership_hashes(train, checkpoint, calibration, policy, test),
    }
    original_train_class_counts = {
        "control": int(train["label_id"].eq(0).sum()),
        "dysarthric": int(train["label_id"].eq(1).sum()),
    }
    sampling_record = {
        "method": "inverse_class_frequency_with_replacement",
        "draws_per_epoch": len(train),
        "expected_control_fraction": 0.5,
        "expected_dysarthric_fraction": 0.5,
    }
    checkpoint_metadata = {
        "run_id": run_id,
        "fold": fold,
        "seed": resolved_seed,
        "model_id": model_id,
        "device": str(device),
        "onset_variant": config["onset"]["primary_mode"],
        "input_representation": "raw_mono_waveform",
        "sample_rate": int(config["audio"]["sample_rate"]),
        "class_balance": "weighted_random_sampler",
        "original_train_class_counts": original_train_class_counts,
        "sampling": sampling_record,
        "validation_partition_method": "sha256_sample_id",
        "model": model_metadata,
        "pretrained_checkpoint": checkpoint_record,
        "environment": environment_record(PROJECT_ROOT),
    }
    training_config = TrainingConfig(
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        label_smoothing=float(training["label_smoothing"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        max_epochs=int(training["max_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        selection_metric=str(
            training.get("checkpoint_selection_metric", "speaker_balanced_macro_f1")
        ),
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
        dataset = RawPrefixDataset(split_frame, training=False, **dataset_options)
        loader = _make_raw_loader(
            dataset,
            batch_size=resolved_batch_size,
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
    samples_by_prefix = {
        str(prefix): sorted(
            int(value)
            for value in ledger.loc[ledger["prefix_ms"].eq(prefix), "raw_input_samples"].unique()
        )
        for prefix in prefixes
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
        "best_checkpoint_speaker_balanced_macro_f1": (
            result.best_checkpoint_speaker_balanced_macro_f1
        ),
        "checkpoint_selection_metric": result.selection_metric,
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        "checkpoint_path": str(checkpoint_path),
        "ledger_path": str(ledger_path),
        "model": model_metadata,
        "pretrained_checkpoint": checkpoint_record,
        "training": {
            "batch_size": resolved_batch_size,
            "class_balance": "weighted_random_sampler",
            "original_train_class_counts": original_train_class_counts,
            "sampling": sampling_record,
            "optimizer_parameter_scope": "requires_grad_only",
            "prefix_sampling": "uniform_sha256_seed_epoch_sample_id",
            "validation_partition_method": "sha256_sample_id",
        },
        "split_sample_counts": {
            "train": len(train),
            "checkpoint": len(checkpoint),
            "calibration": len(calibration),
            "policy_tuning": len(policy),
            "test": len(test),
        },
        "raw_input_sample_counts_by_prefix_ms": samples_by_prefix,
        "hashes": artifact_hashes,
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
