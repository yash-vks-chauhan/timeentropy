from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from timeentropy.config import PROJECT_ROOT, project_path
from timeentropy.evaluation.calibration import fit_temperature, probabilities_from_logits
from timeentropy.evaluation.decisions import (
    PREFIX_TIMES_MS,
    DecisionBatch,
    decide_from_probabilities,
    fixed_duration_predictions,
)
from timeentropy.evaluation.entropy import normalized_binary_entropy
from timeentropy.evaluation.ledger import validate_prefix_ledger
from timeentropy.evaluation.metrics import (
    accuracy_earliness_harmonic_mean,
    binary_classification_metrics,
    evaluate_early_classification,
)
from timeentropy.evaluation.pipeline import read_ledger
from timeentropy.evaluation.selection import select_policy_parameters
from timeentropy.models import create_model
from timeentropy.provenance import environment_record, sha256_file, sha256_json
from timeentropy.robustness.config import (
    RobustnessAnalysis,
    RobustnessProtocol,
    load_robustness_protocol,
)
from timeentropy.robustness.dataset import (
    RobustnessPrefixFeatureDataset,
    collate_robustness_prefix_features,
)
from timeentropy.signal.onset import OnsetConfig
from timeentropy.training.engine import (
    DevicePreference,
    call_model_for_batch,
    move_batch_to_device,
    seed_data_loader_worker,
    seed_everything,
    select_device,
)


@dataclass(frozen=True)
class ReferencePolicy:
    temperature: float
    theta: float
    epsilon: float
    temperature_fit: dict[str, Any]
    policy_selection: dict[str, Any]


@dataclass(frozen=True)
class TestTimeRobustnessResult:
    output_directory: Path
    prefix_ledger_path: Path
    decision_ledger_path: Path
    metrics_path: Path
    policy_path: Path
    summary_path: Path


def inclusive_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Construct an inclusive decimal grid and reject an unreachable stop."""

    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("Grid bounds must be finite")
    if step <= 0.0 or stop < start:
        raise ValueError("Invalid grid bounds")
    count = round((stop - start) / step)
    values = tuple(round(start + index * step, 10) for index in range(count + 1))
    if abs(values[-1] - stop) > 1.0e-8:
        raise ValueError("Grid stop must be reachable by an integer number of steps")
    return values


def _prefix_cube(
    frame: pd.DataFrame,
    probability_columns: tuple[str, str],
) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = frame.sort_values(["sample_id", "prefix_ms"], ignore_index=True)
    if ordered.empty or ordered.duplicated(["sample_id", "prefix_ms"]).any():
        raise ValueError("Prefix cube requires non-empty unique sample/prefix rows")
    observed = ordered.groupby("sample_id", sort=False)["prefix_ms"].agg(tuple)
    if not all(tuple(values) == PREFIX_TIMES_MS for values in observed):
        raise ValueError("Every sample must contain the registered ordered prefix sequence")
    constant = ["run_id", "sample_id", "speaker_id", "fold", "label_id"]
    if "reference_run_id" in ordered.columns:
        constant.append("reference_run_id")
    for column in constant:
        if ordered.groupby("sample_id")[column].nunique().gt(1).any():
            raise ValueError(f"{column} varies across prefixes")
    metadata = ordered.groupby("sample_id", sort=True)[constant].first().reset_index(drop=True)
    probabilities = ordered[list(probability_columns)].to_numpy(dtype=np.float64)
    return metadata, probabilities.reshape(len(metadata), len(PREFIX_TIMES_MS), 2)


def _select_clean_reference_policy(
    reference: pd.DataFrame,
    protocol: RobustnessProtocol,
) -> ReferencePolicy:
    """Fit/tune exclusively on clean calibration and policy-validation rows."""

    calibration = reference.loc[reference["split_role"].eq("calibration")]
    policy = reference.loc[reference["split_role"].eq("policy_tuning")].copy()
    if calibration.empty or policy.empty:
        raise ValueError("Reference ledger requires calibration and policy_tuning rows")
    fit = fit_temperature(
        calibration[["logit_control", "logit_dysarthric"]].to_numpy(dtype=np.float64),
        calibration["label_id"].to_numpy(dtype=np.int64),
    )
    logits = policy[["logit_control", "logit_dysarthric"]].to_numpy(dtype=np.float64)
    probabilities = probabilities_from_logits(logits, temperature=fit.temperature)
    policy["selected_prob_control"] = probabilities[:, 0]
    policy["selected_prob_dysarthric"] = probabilities[:, 1]
    metadata, cube = _prefix_cube(
        policy,
        ("selected_prob_control", "selected_prob_dysarthric"),
    )
    decision_config = protocol.parent_config["decision_policy"]
    theta_config = decision_config["theta_grid"]
    epsilon_config = decision_config["epsilon_grid"]
    selection = select_policy_parameters(
        metadata["label_id"].to_numpy(dtype=np.int64),
        cube,
        split_role="policy_tuning",
        policy_variant="complete",
        theta_grid=inclusive_grid(
            float(theta_config["start"]),
            float(theta_config["stop"]),
            float(theta_config["step"]),
        ),
        epsilon_grid=inclusive_grid(
            float(epsilon_config["start"]),
            float(epsilon_config["stop"]),
            float(epsilon_config["step"]),
        ),
        macro_f1_margin=float(decision_config["selection"]["macro_f1_margin_absolute"]),
    )
    return ReferencePolicy(
        temperature=fit.temperature,
        theta=selection.selected.theta,
        epsilon=selection.selected.epsilon,
        temperature_fit={
            **asdict(fit),
            "fit_split_role": "calibration",
            "fit_rows": len(calibration),
            "fit_unique_samples": int(calibration["sample_id"].nunique()),
            "test_rows_used": 0,
        },
        policy_selection={
            "policy_variant": selection.policy_variant,
            "selection_split_role": "policy_tuning",
            "selection_rows": len(policy),
            "selection_unique_samples": int(policy["sample_id"].nunique()),
            "test_rows_used": 0,
            "macro_f1_margin": selection.macro_f1_margin,
            "fixed_300_macro_f1": selection.fixed_300_macro_f1,
            "feasible_policy_found": selection.feasible_policy_found,
            "selected": asdict(selection.selected),
            "candidate_grid": [asdict(candidate) for candidate in selection.candidates],
        },
    )


def _load_reference_run(
    ledger_path: str | Path,
    *,
    run_id: str,
    fold: int,
    protocol: RobustnessProtocol,
) -> pd.DataFrame:
    ledger = read_ledger(ledger_path)
    validate_prefix_ledger(ledger)
    selected = ledger.loc[ledger["run_id"].astype(str).eq(run_id)].copy()
    if selected.empty:
        raise ValueError(f"Reference ledger does not contain checkpoint run_id {run_id!r}")
    validate_prefix_ledger(selected)
    checks = {
        "fold": (selected["fold"].astype(int), fold),
        "model_id": (selected["model_id"].astype(str), protocol.model_id),
        "seed": (selected["seed"].astype(int), protocol.seed),
        "onset_variant": (
            selected["onset_variant"].astype(str),
            str(protocol.parent_config["onset"]["primary_mode"]),
        ),
        "pre_roll_ms": (
            selected["pre_roll_ms"].astype(int),
            int(protocol.parent_config["onset"]["pre_roll_ms"]),
        ),
    }
    for name, (values, expected) in checks.items():
        if not values.eq(expected).all():
            raise ValueError(f"Reference ledger {name} does not match the frozen checkpoint")
    required_roles = {"calibration", "policy_tuning", "test"}
    missing_roles = required_roles - set(selected["split_role"].astype(str))
    if missing_roles:
        raise ValueError(f"Reference ledger is missing split roles: {sorted(missing_roles)}")
    return selected


def _load_primary_checkpoint(
    checkpoint_path: str | Path,
    *,
    fold: int,
    protocol: RobustnessProtocol,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint payload")
    metadata = payload.get("metadata")
    hashes = payload.get("hashes")
    state = payload.get("model_state_dict")
    if (
        not isinstance(metadata, dict)
        or not isinstance(hashes, dict)
        or not isinstance(state, dict)
    ):
        raise TypeError("Checkpoint lacks metadata, hashes, or model_state_dict")
    expected_metadata = {
        "fold": fold,
        "seed": protocol.seed,
        "model_id": protocol.model_id,
        "onset_variant": protocol.parent_config["onset"]["primary_mode"],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Checkpoint metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    if hashes.get("experiment_config_sha256") != protocol.parent_config_sha256:
        raise ValueError("Checkpoint was not trained from the frozen parent config")

    model = create_model(
        protocol.model_id,
        dropout=float(protocol.parent_config["training"]["dropout"]),
    )
    architecture = getattr(model, "metadata", None)
    expected_architecture_hash = None if architecture is None else sha256_json(asdict(architecture))
    if (
        expected_architecture_hash is not None
        and hashes.get("model_architecture_sha256") != expected_architecture_hash
    ):
        raise ValueError("Checkpoint model architecture hash does not match this implementation")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def _load_fold_test_frame(protocol: RobustnessProtocol, fold: int) -> pd.DataFrame:
    manifest_path = project_path(protocol.parent_config["data"]["manifest"])
    fold_path = project_path(protocol.parent_config["data"]["fold_manifest"])
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    roles = pd.read_csv(fold_path, dtype={"sample_id": str})
    roles = roles.loc[
        roles["fold"].astype(int).eq(fold) & roles["role"].eq("test"),
        ["sample_id", "role"],
    ]
    test = manifest.merge(roles, on="sample_id", how="inner", validate="one_to_one")
    if test.empty:
        raise ValueError(f"Fold {fold} has no test rows")
    test["path"] = test["path"].map(lambda value: str(project_path(str(value))))
    return test


def _dataset_options(protocol: RobustnessProtocol, analysis: RobustnessAnalysis) -> dict[str, Any]:
    parent = protocol.parent_config
    audio = parent["audio"]
    onset = parent["onset"]
    features = parent["features"]
    return {
        "prefixes_ms": tuple(int(value) for value in parent["prefixes"]["training_ms"]),
        "target_sample_rate": int(audio["sample_rate"]),
        "pre_roll_ms": analysis.pre_roll_ms,
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
        "onset_mode": analysis.onset_mode,
        "no_onset_policy": str(onset["no_onset_policy"]),
        "n_fft": int(features["n_fft"]),
        "window_ms": int(features["window_ms"]),
        "hop_ms": int(features["hop_ms"]),
        "normalization_floor": float(features["normalization_floor"]),
    }


def _condition_loader(
    test: pd.DataFrame,
    *,
    protocol: RobustnessProtocol,
    analysis: RobustnessAnalysis,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = RobustnessPrefixFeatureDataset(
        test,
        analysis=analysis,
        primary_seed=protocol.seed,
        **_dataset_options(protocol, analysis),
    )
    generator = torch.Generator().manual_seed(protocol.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_robustness_prefix_features,
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        persistent_workers=False,
    )


def _optional_value(values: Sequence[Any], index: int) -> float:
    value = values[index]
    return float("nan") if value is None else float(value)


def _infer_condition(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    protocol: RobustnessProtocol,
    analysis: RobustnessAnalysis,
    fold: int,
    reference_run_id: str,
    checkpoint_sha256: str,
) -> pd.DataFrame:
    run_id = f"robustness-v1-{reference_run_id}-{analysis.analysis_id}"
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            logits = call_model_for_batch(model, batch).detach().cpu().numpy().astype(np.float64)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Robustness inference requires binary logits")
            probabilities = probabilities_from_logits(logits)
            entropies = normalized_binary_entropy(probabilities)
            predictions = np.argmax(probabilities, axis=1)
            for index in range(len(logits)):
                rows.append(
                    {
                        "run_id": run_id,
                        "reference_run_id": reference_run_id,
                        "reference_checkpoint_sha256": checkpoint_sha256,
                        "sample_id": str(raw_batch["sample_id"][index]),
                        "speaker_id": str(raw_batch["speaker_id"][index]),
                        "fold": fold,
                        "seed": protocol.seed,
                        "model_id": protocol.model_id,
                        "split_role": "test",
                        "onset_variant": analysis.onset_mode,
                        "pre_roll_ms": analysis.pre_roll_ms,
                        "prefix_ms": int(raw_batch["prefix_ms"][index]),
                        "label_id": int(raw_batch["label_id"][index]),
                        "logit_control": float(logits[index, 0]),
                        "logit_dysarthric": float(logits[index, 1]),
                        "prob_control": float(probabilities[index, 0]),
                        "prob_dysarthric": float(probabilities[index, 1]),
                        "predicted_label": int(predictions[index]),
                        "normalized_entropy": float(entropies[index]),
                        "onset_detected": bool(raw_batch["onset_detected"][index]),
                        "onset_sample": int(raw_batch["onset_sample"][index]),
                        "crop_anchor_sample": int(raw_batch["crop_anchor_sample"][index]),
                        "confirmation_sample": int(raw_batch["confirmation_sample"][index]),
                        "onset_threshold": float(raw_batch["onset_threshold"][index]),
                        "source_sample_rate": int(raw_batch["source_sample_rate"][index]),
                        "source_channels": int(raw_batch["source_channels"][index]),
                        "input_num_samples": int(raw_batch["input_num_samples"][index]),
                        "feature_frames": int(raw_batch["feature_lengths"][index]),
                        "robustness_analysis_id": analysis.analysis_id,
                        "robustness_role": analysis.role,
                        "perturbation_kind": analysis.perturbation_kind,
                        "snr_db": _optional_value(raw_batch["snr_db"], index),
                        "onset_offset_ms": _optional_value(raw_batch["onset_offset_ms"], index),
                        "reference_onset_sample": int(raw_batch["reference_onset_sample"][index]),
                        "requested_crop_anchor_sample": int(
                            raw_batch["requested_crop_anchor_sample"][index]
                        ),
                        "onset_offset_applied": bool(raw_batch["onset_offset_applied"][index]),
                        "noise_seed_sha256": str(raw_batch["noise_seed_sha256"][index]),
                        "noise_applied": bool(raw_batch["noise_applied"][index]),
                        "realized_snr_db": _optional_value(raw_batch["realized_snr_db"], index),
                        "signal_rms": _optional_value(raw_batch["signal_rms"], index),
                        "noise_rms": _optional_value(raw_batch["noise_rms"], index),
                    }
                )
    result = pd.DataFrame(rows)
    validate_prefix_ledger(result)
    return result


def _clean_condition(
    reference: pd.DataFrame,
    *,
    analysis: RobustnessAnalysis,
    checkpoint_sha256: str,
) -> pd.DataFrame:
    clean = reference.loc[reference["split_role"].eq("test")].copy()
    reference_run_id = str(clean["run_id"].iloc[0])
    clean["reference_run_id"] = reference_run_id
    clean["run_id"] = f"robustness-v1-{reference_run_id}-{analysis.analysis_id}"
    clean["reference_checkpoint_sha256"] = checkpoint_sha256
    clean["robustness_analysis_id"] = analysis.analysis_id
    clean["robustness_role"] = analysis.role
    clean["perturbation_kind"] = analysis.perturbation_kind
    clean["snr_db"] = float("nan")
    clean["onset_offset_ms"] = float("nan")
    clean["reference_onset_sample"] = clean.get("onset_sample", -1)
    clean["requested_crop_anchor_sample"] = clean.get("onset_sample", -1)
    clean["onset_offset_applied"] = False
    clean["noise_seed_sha256"] = ""
    clean["noise_applied"] = False
    clean["realized_snr_db"] = float("nan")
    clean["signal_rms"] = float("nan")
    clean["noise_rms"] = float("nan")
    calibrated_columns = {
        "temperature",
        "calibrated_prob_control",
        "calibrated_prob_dysarthric",
        "calibrated_predicted_label",
        "calibrated_normalized_entropy",
    }
    clean = clean.drop(columns=list(calibrated_columns & set(clean.columns)))
    validate_prefix_ledger(clean)
    return clean


def _apply_frozen_temperature(frame: pd.DataFrame, temperature: float) -> pd.DataFrame:
    result = frame.copy()
    logits = result[["logit_control", "logit_dysarthric"]].to_numpy(dtype=np.float64)
    probabilities = probabilities_from_logits(logits, temperature=temperature)
    result["temperature"] = temperature
    result["calibrated_prob_control"] = probabilities[:, 0]
    result["calibrated_prob_dysarthric"] = probabilities[:, 1]
    result["calibrated_predicted_label"] = np.argmax(probabilities, axis=1).astype(np.int64)
    result["calibrated_normalized_entropy"] = normalized_binary_entropy(probabilities)
    validate_prefix_ledger(result)
    return result


def _decision_row(
    sample: pd.Series,
    decision: DecisionBatch,
    index: int,
    *,
    analysis: RobustnessAnalysis,
    policy: ReferencePolicy,
) -> dict[str, Any]:
    row = {
        "run_id": sample["run_id"],
        "reference_run_id": sample["reference_run_id"],
        "sample_id": sample["sample_id"],
        "speaker_id": sample["speaker_id"],
        "fold": int(sample["fold"]),
        "robustness_analysis_id": analysis.analysis_id,
        "robustness_role": analysis.role,
        "perturbation_kind": analysis.perturbation_kind,
        "snr_db": analysis.snr_db,
        "onset_offset_ms": analysis.onset_offset_ms,
        "policy_id": "complete-temperature_scaled",
        "policy_variant": "complete",
        "calibration_state": "temperature_scaled",
        "temperature": policy.temperature,
        "theta": policy.theta,
        "epsilon": policy.epsilon,
        "selection_split_role": "policy_tuning",
        "calibration_split_role": "calibration",
        "label_id": int(sample["label_id"]),
        "decision_time_ms": int(decision.decision_time_ms[index]),
        "predicted_label": int(decision.predicted_label[index]),
        "accepted": bool(decision.accepted[index]),
        "accepted_at_endpoint": bool(decision.accepted_at_endpoint[index]),
        "forced_fallback": bool(decision.forced_fallback[index]),
    }
    for candidate_index, candidate_ms in enumerate((200, 250, 300)):
        row[f"threshold_pass_{candidate_ms}"] = bool(
            decision.threshold_pass[index, candidate_index]
        )
        row[f"class_stable_{candidate_ms}"] = bool(decision.class_stable[index, candidate_index])
        row[f"entropy_stable_{candidate_ms}"] = bool(
            decision.entropy_stable[index, candidate_index]
        )
        row[f"candidate_rule_pass_{candidate_ms}"] = bool(
            decision.candidate_rule_pass[index, candidate_index]
        )
    return row


def _evaluate_condition(
    frame: pd.DataFrame,
    *,
    analysis: RobustnessAnalysis,
    policy: ReferencePolicy,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    metadata, probabilities = _prefix_cube(
        frame,
        ("calibrated_prob_control", "calibrated_prob_dysarthric"),
    )
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    decision = decide_from_probabilities(
        probabilities,
        theta=policy.theta,
        epsilon=policy.epsilon,
        policy_variant="complete",
    )
    decision_rows = [
        _decision_row(
            sample,
            decision,
            index,
            analysis=analysis,
            policy=policy,
        )
        for index, sample in metadata.iterrows()
    ]
    early = evaluate_early_classification(labels, decision)
    metrics: list[dict[str, Any]] = [
        {
            "fold": int(metadata["fold"].iloc[0]),
            "robustness_analysis_id": analysis.analysis_id,
            "robustness_role": analysis.role,
            "perturbation_kind": analysis.perturbation_kind,
            "snr_db": analysis.snr_db,
            "onset_offset_ms": analysis.onset_offset_ms,
            "policy_id": "complete-temperature_scaled",
            "temperature": policy.temperature,
            "theta": policy.theta,
            "epsilon": policy.epsilon,
            "calibration_source": "clean_reference:calibration",
            "selection_source": "clean_reference:policy_tuning",
            "test_rows_used_for_tuning": 0,
            **early.as_dict(),
        }
    ]
    for prefix_ms in PREFIX_TIMES_MS:
        predictions = fixed_duration_predictions(probabilities, prefix_ms)
        classification = binary_classification_metrics(labels, predictions)
        consumed = prefix_ms / 300.0
        metrics.append(
            {
                "fold": int(metadata["fold"].iloc[0]),
                "robustness_analysis_id": analysis.analysis_id,
                "robustness_role": analysis.role,
                "perturbation_kind": analysis.perturbation_kind,
                "snr_db": analysis.snr_db,
                "onset_offset_ms": analysis.onset_offset_ms,
                "policy_id": f"fixed_{prefix_ms}",
                "temperature": policy.temperature,
                "theta": float("nan"),
                "epsilon": float("nan"),
                "calibration_source": "clean_reference:calibration",
                "selection_source": "not_applicable",
                "test_rows_used_for_tuning": 0,
                **classification.as_dict(),
                "mean_latency_ms": float(prefix_ms),
                "forced_fallback_rate": 0.0,
                "consumed_fraction": consumed,
                "timeliness": 1.0 - consumed,
                "accuracy_earliness_harmonic_mean": (
                    accuracy_earliness_harmonic_mean(classification.accuracy, consumed)
                ),
                "macro_f1_earliness_harmonic_mean": (
                    accuracy_earliness_harmonic_mean(classification.macro_f1, consumed)
                ),
            }
        )
        prefix_index = PREFIX_TIMES_MS.index(prefix_ms)
        for index, sample in metadata.iterrows():
            decision_rows.append(
                {
                    "run_id": sample["run_id"],
                    "reference_run_id": sample["reference_run_id"],
                    "sample_id": sample["sample_id"],
                    "speaker_id": sample["speaker_id"],
                    "fold": int(sample["fold"]),
                    "robustness_analysis_id": analysis.analysis_id,
                    "robustness_role": analysis.role,
                    "perturbation_kind": analysis.perturbation_kind,
                    "snr_db": analysis.snr_db,
                    "onset_offset_ms": analysis.onset_offset_ms,
                    "policy_id": f"fixed_{prefix_ms}",
                    "policy_variant": "fixed_duration",
                    "calibration_state": "temperature_scaled",
                    "temperature": policy.temperature,
                    "theta": float("nan"),
                    "epsilon": float("nan"),
                    "selection_split_role": "not_applicable",
                    "calibration_split_role": "calibration",
                    "label_id": int(sample["label_id"]),
                    "decision_time_ms": prefix_ms,
                    "predicted_label": int(np.argmax(probabilities[index, prefix_index, :])),
                    "accepted": True,
                    "accepted_at_endpoint": False,
                    "forced_fallback": False,
                    **{
                        f"{flag}_{candidate_ms}": False
                        for flag in (
                            "threshold_pass",
                            "class_stable",
                            "entropy_stable",
                            "candidate_rule_pass",
                        )
                        for candidate_ms in (200, 250, 300)
                    },
                }
            )
    decision_frame = pd.DataFrame(decision_rows)
    for column in ("snr_db", "onset_offset_ms"):
        decision_frame[column] = pd.to_numeric(decision_frame[column], errors="coerce")
    return decision_frame, metrics


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
        return path
    try:
        frame.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback


def _validate_robustness_decisions(
    decisions: pd.DataFrame,
    prefix_ledger: pd.DataFrame,
) -> None:
    required = {
        "run_id",
        "sample_id",
        "policy_id",
        "label_id",
        "decision_time_ms",
        "predicted_label",
        "forced_fallback",
        "accepted_at_endpoint",
        "robustness_analysis_id",
    }
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"Robustness decision ledger is missing columns: {sorted(missing)}")
    if decisions.empty or decisions.duplicated(["run_id", "sample_id", "policy_id"]).any():
        raise ValueError("Robustness decisions must be non-empty and uniquely keyed")
    if not decisions["decision_time_ms"].isin(PREFIX_TIMES_MS).all():
        raise ValueError("Robustness decision time is not a registered prefix")
    if (
        not decisions["label_id"].isin((0, 1)).all()
        or not decisions["predicted_label"].isin((0, 1)).all()
    ):
        raise ValueError("Robustness decisions require binary labels")
    forced = decisions["forced_fallback"].astype(bool)
    endpoint = decisions["accepted_at_endpoint"].astype(bool)
    if (forced & decisions["decision_time_ms"].ne(300)).any() or (forced & endpoint).any():
        raise ValueError("Robustness fallback/endpoint flags are inconsistent")
    lookup = prefix_ledger[
        ["run_id", "sample_id", "prefix_ms", "calibrated_predicted_label"]
    ].rename(
        columns={
            "prefix_ms": "decision_time_ms",
            "calibrated_predicted_label": "prefix_predicted_label",
        }
    )
    merged = decisions.merge(
        lookup,
        on=["run_id", "sample_id", "decision_time_ms"],
        how="left",
        validate="many_to_one",
    )
    if merged["prefix_predicted_label"].isna().any() or not np.array_equal(
        merged["predicted_label"].to_numpy(dtype=np.int64),
        merged["prefix_predicted_label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Robustness decisions do not link to calibrated prefix predictions")


def _resolve_analyses(
    protocol: RobustnessProtocol,
    analysis_ids: Iterable[str] | None,
) -> tuple[RobustnessAnalysis, ...]:
    if analysis_ids is None:
        return protocol.test_time_analyses
    requested = tuple(str(value) for value in analysis_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("analysis_ids must be non-empty and unique when supplied")
    selected: list[RobustnessAnalysis] = []
    for analysis_id in requested:
        analysis = protocol.analysis(analysis_id)
        if analysis.family != "test_time":
            raise ValueError(
                f"{analysis_id} requires condition-matched training; it is not test-time"
            )
        selected.append(analysis)
    return tuple(selected)


def run_test_time_robustness(
    robustness_config_path: str | Path,
    *,
    fold: int,
    checkpoint_path: str | Path,
    reference_ledger_path: str | Path,
    output_directory: str | Path,
    analysis_ids: Iterable[str] | None = None,
    device_preference: DevicePreference = "auto",
    batch_size: int | None = None,
    num_workers: int = 0,
) -> TestTimeRobustnessResult:
    """Evaluate frozen test-time conditions without fitting anything on test rows."""

    protocol = load_robustness_protocol(robustness_config_path)
    if fold not in protocol.folds:
        raise ValueError(f"Fold {fold} is not registered in the robustness protocol")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    resolved_batch_size = (
        int(protocol.parent_config["training"]["batch_size"])
        if batch_size is None
        else int(batch_size)
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive")
    analyses = _resolve_analyses(protocol, analysis_ids)
    seed_everything(protocol.seed)
    device = select_device(device_preference)
    model, checkpoint = _load_primary_checkpoint(
        checkpoint_path,
        fold=fold,
        protocol=protocol,
        device=device,
    )
    metadata = checkpoint["metadata"]
    reference_run_id = str(metadata.get("run_id", ""))
    if not reference_run_id:
        raise ValueError("Checkpoint metadata lacks run_id")
    reference = _load_reference_run(
        reference_ledger_path,
        run_id=reference_run_id,
        fold=fold,
        protocol=protocol,
    )
    policy = _select_clean_reference_policy(reference, protocol)
    test = _load_fold_test_frame(protocol, fold)
    reference_test_ids = frozenset(
        reference.loc[reference["split_role"].eq("test"), "sample_id"].astype(str)
    )
    manifest_test_ids = frozenset(test["sample_id"].astype(str))
    if reference_test_ids != manifest_test_ids:
        raise ValueError("Reference ledger test samples differ from the frozen fold manifest")

    checkpoint_hash = sha256_file(checkpoint_path)
    ledger_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    zero_shift_audit: dict[str, Any] | None = None
    for analysis in analyses:
        if analysis.perturbation_kind == "none":
            raw = _clean_condition(
                reference,
                analysis=analysis,
                checkpoint_sha256=checkpoint_hash,
            )
        else:
            loader = _condition_loader(
                test,
                protocol=protocol,
                analysis=analysis,
                batch_size=resolved_batch_size,
                num_workers=num_workers,
            )
            raw = _infer_condition(
                model,
                loader,
                device=device,
                protocol=protocol,
                analysis=analysis,
                fold=fold,
                reference_run_id=reference_run_id,
                checkpoint_sha256=checkpoint_hash,
            )
        calibrated = _apply_frozen_temperature(raw, policy.temperature)
        decisions, metrics = _evaluate_condition(
            calibrated,
            analysis=analysis,
            policy=policy,
        )
        ledger_parts.append(calibrated)
        decision_parts.append(decisions)
        metric_rows.extend(metrics)

        if analysis.perturbation_kind == "onset_shift" and analysis.onset_offset_ms == 0:
            baseline = reference.loc[
                reference["split_role"].eq("test"),
                ["sample_id", "prefix_ms", "logit_control", "logit_dysarthric"],
            ]
            compared = calibrated.merge(
                baseline,
                on=["sample_id", "prefix_ms"],
                suffixes=("_new", "_reference"),
                validate="one_to_one",
            )
            differences = np.abs(
                compared[["logit_control_new", "logit_dysarthric_new"]].to_numpy(dtype=np.float64)
                - compared[["logit_control_reference", "logit_dysarthric_reference"]].to_numpy(
                    dtype=np.float64
                )
            )
            maximum = float(differences.max(initial=0.0))
            zero_shift_audit = {
                "rows": len(compared),
                "maximum_absolute_logit_difference": maximum,
                "informational_only": True,
                "note": (
                    "Device-dependent floating-point differences are recorded but are not "
                    "used to accept, tune, or reject the condition."
                ),
            }

    prefix_ledger = pd.concat(ledger_parts, ignore_index=True).sort_values(
        ["robustness_analysis_id", "sample_id", "prefix_ms"], ignore_index=True
    )
    decisions = pd.concat(decision_parts, ignore_index=True).sort_values(
        ["robustness_analysis_id", "policy_id", "sample_id"], ignore_index=True
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["robustness_analysis_id", "policy_id"], ignore_index=True
    )
    validate_prefix_ledger(prefix_ledger)
    _validate_robustness_decisions(decisions, prefix_ledger)
    if not metrics["test_rows_used_for_tuning"].eq(0).all():
        raise RuntimeError("Internal invariant failed: test rows entered robustness tuning")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    prefix_path = _write_table(prefix_ledger, output / "robustness_prefix_predictions.parquet")
    decision_path = _write_table(decisions, output / "robustness_decisions.parquet")
    metrics_path = _write_table(metrics, output / "robustness_metrics.csv")
    policy_path = output / "clean_reference_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fold": fold,
                "reference_run_id": reference_run_id,
                "temperature": policy.temperature,
                "theta": policy.theta,
                "epsilon": policy.epsilon,
                "temperature_fit": policy.temperature_fit,
                "policy_selection": policy.policy_selection,
                "calibration_source": "clean_reference:calibration",
                "selection_source": "clean_reference:policy_tuning",
                "test_rows_used": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "schema_version": 1,
        "execution_stage": "test_inference_only",
        "fold": fold,
        "model_id": protocol.model_id,
        "seed": protocol.seed,
        "device": str(device),
        "analyses": [asdict(analysis) for analysis in analyses],
        "test_set_tuning_permitted": False,
        "test_rows_used_for_temperature_fit": 0,
        "test_rows_used_for_policy_selection": 0,
        "clean_reference_sources": {
            "temperature": "calibration",
            "policy": "policy_tuning",
        },
        "inputs": {
            "robustness_config": {
                "path": str(protocol.config_path),
                "file_sha256": protocol.config_file_sha256,
                "content_sha256": protocol.config_sha256,
            },
            "parent_config": {
                "path": str(protocol.parent_config_path),
                "file_sha256": protocol.parent_config_file_sha256,
                "content_sha256": protocol.parent_config_sha256,
            },
            "checkpoint": {
                "path": str(Path(checkpoint_path)),
                "sha256": checkpoint_hash,
                "run_id": reference_run_id,
            },
            "reference_ledger": {
                "path": str(Path(reference_ledger_path)),
                "sha256": sha256_file(reference_ledger_path),
            },
            "manifest_sha256": sha256_file(
                project_path(protocol.parent_config["data"]["manifest"])
            ),
            "fold_manifest_sha256": sha256_file(
                project_path(protocol.parent_config["data"]["fold_manifest"])
            ),
        },
        "rows": {
            "prefix_predictions": len(prefix_ledger),
            "decisions": len(decisions),
            "metrics": len(metrics),
        },
        "zero_shift_reproduction_audit": zero_shift_audit,
        "environment": environment_record(PROJECT_ROOT),
        "outputs": {
            "prefix_ledger": {"path": str(prefix_path), "sha256": sha256_file(prefix_path)},
            "decision_ledger": {
                "path": str(decision_path),
                "sha256": sha256_file(decision_path),
            },
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        },
    }
    summary_path = output / "robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return TestTimeRobustnessResult(
        output_directory=output,
        prefix_ledger_path=prefix_path,
        decision_ledger_path=decision_path,
        metrics_path=metrics_path,
        policy_path=policy_path,
        summary_path=summary_path,
    )
