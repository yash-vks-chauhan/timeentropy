from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timeentropy.evaluation.calibration import (
    calibration_metrics,
    fit_temperature,
    probabilities_from_logits,
)
from timeentropy.evaluation.decisions import (
    PREFIX_TIMES_MS,
    PolicyVariant,
    decide_from_maximum_confidence,
    decide_from_probabilities,
    fixed_duration_predictions,
)
from timeentropy.evaluation.entropy import (
    maximum_confidence_threshold_for_entropy,
    normalized_binary_entropy,
)
from timeentropy.evaluation.ledger import (
    validate_decision_ledger,
    validate_oof_test_coverage,
    validate_prefix_ledger,
)
from timeentropy.evaluation.metrics import binary_classification_metrics
from timeentropy.evaluation.selection import (
    DEFAULT_EPSILON_GRID,
    DEFAULT_THETA_GRID,
    select_policy_parameters,
)
from timeentropy.provenance import sha256_file, sha256_json

DEFAULT_POLICY_VARIANTS: tuple[PolicyVariant, ...] = (
    "entropy_only",
    "entropy_class_stability",
    "entropy_change_stability",
    "complete",
)


@dataclass(frozen=True)
class EvaluationArtifacts:
    calibrated_prefix_ledger: pd.DataFrame
    decision_ledger: pd.DataFrame
    fixed_predictions: pd.DataFrame
    selections: pd.DataFrame
    candidate_grid: pd.DataFrame
    temperature_fits: pd.DataFrame
    calibration_summary: pd.DataFrame
    reliability_bins: pd.DataFrame


def read_ledger(path: str | Path) -> pd.DataFrame:
    """Read one canonical CSV or Parquet prefix ledger."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix.lower() == ".csv":
        return pd.read_csv(resolved)
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    raise ValueError(f"Unsupported ledger extension: {resolved.suffix}")


def combine_prefix_ledgers(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read, concatenate, and validate independently generated fold ledgers."""

    resolved = tuple(Path(path) for path in paths)
    if not resolved:
        raise ValueError("At least one prefix ledger is required")
    frame = pd.concat([read_ledger(path) for path in resolved], ignore_index=True)
    validate_prefix_ledger(frame)
    return frame.sort_values(
        ["model_id", "seed", "onset_variant", "fold", "split_role", "sample_id", "prefix_ms"],
        ignore_index=True,
    )


def _metric_rows(
    frame: pd.DataFrame,
    *,
    probability_columns: tuple[str, str],
    calibration_state: str,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    grouping = ["run_id", "fold", "model_id", "seed", "split_role", "prefix_ms"]
    for key, group in frame.groupby(grouping, sort=True, dropna=False):
        probabilities = group[list(probability_columns)].to_numpy(dtype=np.float64)
        result = calibration_metrics(
            probabilities,
            group["label_id"].to_numpy(dtype=np.int64),
            bins=bins,
        )
        identity = dict(zip(grouping, key, strict=True))
        summaries.append(
            {
                **identity,
                "calibration_state": calibration_state,
                "samples": result.samples,
                "negative_log_likelihood": result.negative_log_likelihood,
                "binary_brier_score": result.binary_brier_score,
                "expected_calibration_error": result.expected_calibration_error,
            }
        )
        for bin_index, reliability in enumerate(result.reliability_bins):
            reliability_rows.append(
                {
                    **identity,
                    "calibration_state": calibration_state,
                    "bin_index": bin_index,
                    **asdict(reliability),
                }
            )
    return summaries, reliability_rows


def calibrate_prefix_ledger(
    frame: pd.DataFrame,
    *,
    bins: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one temperature per run on calibration rows and apply it everywhere.

    Every registered prefix contributes one logit/label pair to the scalar-temperature
    objective, matching the frozen protocol.  The identity temperature is retained if the
    numerical optimum does not improve calibration-set NLL.
    """

    validate_prefix_ledger(frame)
    if bins < 2:
        raise ValueError("bins must be at least two")
    calibrated = frame.copy()
    temperatures: dict[str, float] = {}
    fit_rows: list[dict[str, Any]] = []
    for run_id, run in calibrated.groupby("run_id", sort=True):
        calibration = run.loc[run["split_role"].eq("calibration")]
        if calibration.empty:
            raise ValueError(f"Run {run_id!r} has no calibration rows")
        fit = fit_temperature(
            calibration[["logit_control", "logit_dysarthric"]].to_numpy(dtype=np.float64),
            calibration["label_id"].to_numpy(dtype=np.int64),
        )
        temperatures[str(run_id)] = fit.temperature
        first = run.iloc[0]
        fit_rows.append(
            {
                "run_id": run_id,
                "fold": int(first["fold"]),
                "model_id": str(first["model_id"]),
                "seed": int(first["seed"]),
                **asdict(fit),
                "fit_split_role": "calibration",
                "fit_rows": len(calibration),
                "fit_unique_samples": int(calibration["sample_id"].nunique()),
            }
        )

    calibrated["temperature"] = calibrated["run_id"].astype(str).map(temperatures)
    if calibrated["temperature"].isna().any():
        raise RuntimeError("A run did not receive a fitted temperature")
    logits = calibrated[["logit_control", "logit_dysarthric"]].to_numpy(dtype=np.float64)
    scaled_probabilities = probabilities_from_logits(
        logits / calibrated["temperature"].to_numpy(dtype=np.float64)[:, None]
    )
    calibrated["calibrated_prob_control"] = scaled_probabilities[:, 0]
    calibrated["calibrated_prob_dysarthric"] = scaled_probabilities[:, 1]
    calibrated["calibrated_predicted_label"] = np.argmax(
        scaled_probabilities, axis=1
    ).astype(np.int64)
    calibrated["calibrated_normalized_entropy"] = normalized_binary_entropy(
        scaled_probabilities
    )
    validate_prefix_ledger(calibrated)

    summary_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for columns, state in (
        (("prob_control", "prob_dysarthric"), "raw"),
        (("calibrated_prob_control", "calibrated_prob_dysarthric"), "temperature_scaled"),
    ):
        summary, reliability = _metric_rows(
            calibrated,
            probability_columns=columns,
            calibration_state=state,
            bins=bins,
        )
        summary_rows.extend(summary)
        reliability_rows.extend(reliability)
    return (
        calibrated,
        pd.DataFrame(fit_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(reliability_rows),
    )


def _prefix_cube(
    frame: pd.DataFrame,
    *,
    probability_columns: tuple[str, str],
    expected_prefixes_ms: Sequence[int] = PREFIX_TIMES_MS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return one metadata row per sample and an N x P x 2 probability cube."""

    if frame.empty:
        raise ValueError("Cannot create a prefix cube from an empty frame")
    prefixes = tuple(int(value) for value in expected_prefixes_ms)
    ordered = frame.sort_values(["sample_id", "prefix_ms"], ignore_index=True)
    if ordered.duplicated(["sample_id", "prefix_ms"]).any():
        raise ValueError("Duplicate sample/prefix rows in one run split")
    counts = ordered.groupby("sample_id")["prefix_ms"].agg(tuple)
    if not all(tuple(sorted(values)) == prefixes for values in counts):
        raise ValueError("Every sample must contain the registered prefix sequence")
    metadata_columns = ["run_id", "sample_id", "speaker_id", "fold", "label_id"]
    metadata = ordered.groupby("sample_id", sort=True)[metadata_columns].first()
    for column in ("run_id", "speaker_id", "fold", "label_id"):
        if ordered.groupby("sample_id")[column].nunique().gt(1).any():
            raise ValueError(f"{column} varies across prefixes for a sample")
    probabilities = ordered[list(probability_columns)].to_numpy(dtype=np.float64)
    probabilities = probabilities.reshape(len(metadata), len(prefixes), 2)
    return metadata.reset_index(drop=True), probabilities


def _decision_rows(
    metadata: pd.DataFrame,
    decisions: Any,
    *,
    policy_id: str,
    policy_variant: str,
    calibration_state: str,
    theta: float,
    epsilon: float,
    confidence_threshold: float = float("nan"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample in metadata.reset_index(drop=True).iterrows():
        row: dict[str, Any] = {
            "run_id": sample["run_id"],
            "sample_id": sample["sample_id"],
            "speaker_id": sample["speaker_id"],
            "fold": int(sample["fold"]),
            "policy_id": policy_id,
            "policy_variant": policy_variant,
            "calibration_state": calibration_state,
            "theta": float(theta),
            "epsilon": float(epsilon),
            "confidence_threshold": float(confidence_threshold),
            "label_id": int(sample["label_id"]),
            "decision_time_ms": int(decisions.decision_time_ms[index]),
            "predicted_label": int(decisions.predicted_label[index]),
            "accepted": bool(decisions.accepted[index]),
            "accepted_at_endpoint": bool(decisions.accepted_at_endpoint[index]),
            "forced_fallback": bool(decisions.forced_fallback[index]),
        }
        for candidate_index, candidate_ms in enumerate((200, 250, 300)):
            row[f"threshold_pass_{candidate_ms}"] = bool(
                decisions.threshold_pass[index, candidate_index]
            )
            row[f"class_stable_{candidate_ms}"] = bool(
                decisions.class_stable[index, candidate_index]
            )
            row[f"entropy_stable_{candidate_ms}"] = bool(
                decisions.entropy_stable[index, candidate_index]
            )
            row[f"candidate_rule_pass_{candidate_ms}"] = bool(
                decisions.candidate_rule_pass[index, candidate_index]
            )
        rows.append(row)
    return rows


def build_policy_artifacts(
    calibrated_prefix_ledger: pd.DataFrame,
    *,
    policy_variants: Sequence[PolicyVariant] = DEFAULT_POLICY_VARIANTS,
    theta_grid: Iterable[float] = DEFAULT_THETA_GRID,
    epsilon_grid: Iterable[float] = DEFAULT_EPSILON_GRID,
    macro_f1_margin: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select each policy on validation rows and apply it once to test rows."""

    validate_prefix_ledger(calibrated_prefix_ledger)
    decision_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    variants = tuple(policy_variants)
    if not variants:
        raise ValueError("At least one policy variant is required")

    for run_id, run in calibrated_prefix_ledger.groupby("run_id", sort=True):
        policy_frame = run.loc[run["split_role"].eq("policy_tuning")]
        test_frame = run.loc[run["split_role"].eq("test")]
        if policy_frame.empty or test_frame.empty:
            raise ValueError(f"Run {run_id!r} lacks policy-tuning or test rows")

        # Scalar temperature scaling preserves the class argmax, so fixed-prefix
        # predictions are emitted once from raw probabilities.
        test_metadata, test_raw = _prefix_cube(
            test_frame,
            probability_columns=("prob_control", "prob_dysarthric"),
        )
        for prefix_index, prefix_ms in enumerate(PREFIX_TIMES_MS):
            predictions = fixed_duration_predictions(test_raw, prefix_ms)
            for row_index, sample in test_metadata.iterrows():
                fixed_rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": sample["sample_id"],
                        "speaker_id": sample["speaker_id"],
                        "fold": int(sample["fold"]),
                        "policy_id": f"fixed_{prefix_ms}",
                        "prefix_ms": int(prefix_ms),
                        "label_id": int(sample["label_id"]),
                        "predicted_label": int(predictions[row_index]),
                        "prob_control": float(test_raw[row_index, prefix_index, 0]),
                        "prob_dysarthric": float(test_raw[row_index, prefix_index, 1]),
                    }
                )
                if prefix_ms in (200, 250, 300):
                    decision_rows.append(
                        {
                            "run_id": run_id,
                            "sample_id": sample["sample_id"],
                            "speaker_id": sample["speaker_id"],
                            "fold": int(sample["fold"]),
                            "policy_id": f"fixed_{prefix_ms}",
                            "policy_variant": "fixed_duration",
                            "calibration_state": "not_applicable",
                            "theta": float("nan"),
                            "epsilon": float("nan"),
                            "confidence_threshold": float("nan"),
                            "label_id": int(sample["label_id"]),
                            "decision_time_ms": int(prefix_ms),
                            "predicted_label": int(predictions[row_index]),
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

        for calibration_state, columns in (
            ("raw", ("prob_control", "prob_dysarthric")),
            (
                "temperature_scaled",
                ("calibrated_prob_control", "calibrated_prob_dysarthric"),
            ),
        ):
            policy_metadata, policy_probabilities = _prefix_cube(
                policy_frame,
                probability_columns=columns,
            )
            test_metadata, test_probabilities = _prefix_cube(
                test_frame,
                probability_columns=columns,
            )
            policy_targets = policy_metadata["label_id"].to_numpy(dtype=np.int64)
            for variant in variants:
                selection = select_policy_parameters(
                    policy_targets,
                    policy_probabilities,
                    split_role="policy_tuning",
                    policy_variant=variant,
                    theta_grid=theta_grid,
                    epsilon_grid=epsilon_grid,
                    macro_f1_margin=macro_f1_margin,
                )
                policy_id = f"{variant}-{calibration_state}"
                selected = selection.selected
                test_decisions = decide_from_probabilities(
                    test_probabilities,
                    theta=selected.theta,
                    epsilon=selected.epsilon,
                    policy_variant=variant,
                )
                decision_rows.extend(
                    _decision_rows(
                        test_metadata,
                        test_decisions,
                        policy_id=policy_id,
                        policy_variant=variant,
                        calibration_state=calibration_state,
                        theta=selected.theta,
                        epsilon=selected.epsilon,
                    )
                )
                selection_rows.append(
                    {
                        "run_id": run_id,
                        "fold": int(test_metadata.iloc[0]["fold"]),
                        "policy_id": policy_id,
                        "policy_variant": variant,
                        "calibration_state": calibration_state,
                        "confidence_threshold": float("nan"),
                        "macro_f1_margin": selection.macro_f1_margin,
                        "fixed_300_policy_macro_f1": selection.fixed_300_macro_f1,
                        "feasible_policy_found": selection.feasible_policy_found,
                        **asdict(selected),
                    }
                )
                for candidate in selection.candidates:
                    candidate_rows.append(
                        {
                            "run_id": run_id,
                            "fold": int(test_metadata.iloc[0]["fold"]),
                            "policy_id": policy_id,
                            "policy_variant": variant,
                            "calibration_state": calibration_state,
                            "confidence_threshold": float("nan"),
                            "selected": (
                                candidate.theta == selected.theta
                                and candidate.epsilon == selected.epsilon
                            ),
                            **asdict(candidate),
                        }
                    )

                if variant == "entropy_only":
                    confidence_threshold = maximum_confidence_threshold_for_entropy(
                        selected.theta
                    )
                    confidence_decisions = decide_from_maximum_confidence(
                        test_probabilities,
                        confidence_threshold=confidence_threshold,
                    )
                    if not (
                        np.array_equal(
                            confidence_decisions.decision_time_ms,
                            test_decisions.decision_time_ms,
                        )
                        and np.array_equal(
                            confidence_decisions.predicted_label,
                            test_decisions.predicted_label,
                        )
                        and np.array_equal(
                            confidence_decisions.forced_fallback,
                            test_decisions.forced_fallback,
                        )
                    ):
                        raise RuntimeError(
                            "Binary entropy-only and mapped maximum-confidence policies diverged"
                        )
                    confidence_policy_id = (
                        f"max_softmax_confidence-{calibration_state}"
                    )
                    decision_rows.extend(
                        _decision_rows(
                            test_metadata,
                            confidence_decisions,
                            policy_id=confidence_policy_id,
                            policy_variant="max_softmax_confidence",
                            calibration_state=calibration_state,
                            theta=selected.theta,
                            epsilon=0.0,
                            confidence_threshold=confidence_threshold,
                        )
                    )
                    selection_rows.append(
                        {
                            "run_id": run_id,
                            "fold": int(test_metadata.iloc[0]["fold"]),
                            "policy_id": confidence_policy_id,
                            "policy_variant": "max_softmax_confidence",
                            "calibration_state": calibration_state,
                            "confidence_threshold": confidence_threshold,
                            "equivalent_entropy_theta": selected.theta,
                            "equivalence": "exact_for_binary_probabilities",
                            "macro_f1_margin": selection.macro_f1_margin,
                            "fixed_300_policy_macro_f1": selection.fixed_300_macro_f1,
                            "feasible_policy_found": selection.feasible_policy_found,
                            **asdict(selected),
                        }
                    )
                    for candidate in selection.candidates:
                        candidate_rows.append(
                            {
                                "run_id": run_id,
                                "fold": int(test_metadata.iloc[0]["fold"]),
                                "policy_id": confidence_policy_id,
                                "policy_variant": "max_softmax_confidence",
                                "calibration_state": calibration_state,
                                "confidence_threshold": (
                                    maximum_confidence_threshold_for_entropy(candidate.theta)
                                ),
                                "equivalent_entropy_theta": candidate.theta,
                                "equivalence": "exact_for_binary_probabilities",
                                "selected": (
                                    candidate.theta == selected.theta
                                    and candidate.epsilon == selected.epsilon
                                ),
                                **asdict(candidate),
                            }
                        )

    decisions = pd.DataFrame(decision_rows).sort_values(
        ["policy_id", "fold", "sample_id"], ignore_index=True
    )
    fixed = pd.DataFrame(fixed_rows).sort_values(
        ["prefix_ms", "fold", "sample_id"], ignore_index=True
    )
    selections = pd.DataFrame(selection_rows).sort_values(
        ["policy_id", "fold"], ignore_index=True
    )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["policy_id", "fold", "theta", "epsilon"], ignore_index=True
    )

    for state, prediction_column in (
        ("raw", "predicted_label"),
        ("temperature_scaled", "calibrated_predicted_label"),
    ):
        selected_decisions = decisions.loc[decisions["calibration_state"].eq(state)]
        validate_decision_ledger(
            selected_decisions,
            prefix_ledger=calibrated_prefix_ledger,
            prefix_prediction_column=prediction_column,
        )
    fixed_decisions = decisions.loc[decisions["policy_variant"].eq("fixed_duration")]
    validate_decision_ledger(
        fixed_decisions,
        prefix_ledger=calibrated_prefix_ledger,
        prefix_prediction_column="predicted_label",
    )
    return decisions, fixed, selections, candidates


def evaluate_prefix_ledgers(
    frame: pd.DataFrame,
    *,
    expected_oof_sample_ids: Iterable[str] | None = None,
    bins: int = 15,
    policy_variants: Sequence[PolicyVariant] = DEFAULT_POLICY_VARIANTS,
    theta_grid: Iterable[float] = DEFAULT_THETA_GRID,
    epsilon_grid: Iterable[float] = DEFAULT_EPSILON_GRID,
    macro_f1_margin: float = 0.01,
) -> EvaluationArtifacts:
    """Run the complete frozen post-training evaluation pipeline in memory."""

    validate_prefix_ledger(frame)
    if expected_oof_sample_ids is not None:
        validate_oof_test_coverage(frame, expected_oof_sample_ids)
    calibrated, fits, calibration, reliability = calibrate_prefix_ledger(frame, bins=bins)
    decisions, fixed, selections, candidates = build_policy_artifacts(
        calibrated,
        policy_variants=policy_variants,
        theta_grid=theta_grid,
        epsilon_grid=epsilon_grid,
        macro_f1_margin=macro_f1_margin,
    )
    return EvaluationArtifacts(
        calibrated_prefix_ledger=calibrated,
        decision_ledger=decisions,
        fixed_predictions=fixed,
        selections=selections,
        candidate_grid=candidates,
        temperature_fits=fits,
        calibration_summary=calibration,
        reliability_bins=reliability,
    )


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


def write_evaluation_artifacts(
    artifacts: EvaluationArtifacts,
    output_directory: str | Path,
    *,
    input_paths: Sequence[str | Path] = (),
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist ledgers/tables plus one hash-addressed provenance summary."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "calibrated_prefix_ledger": _write_table(
            artifacts.calibrated_prefix_ledger,
            output / "calibrated_prefix_predictions.parquet",
        ),
        "decision_ledger": _write_table(
            artifacts.decision_ledger,
            output / "decisions.parquet",
        ),
        "fixed_predictions": _write_table(
            artifacts.fixed_predictions,
            output / "fixed_predictions.parquet",
        ),
        "selections": _write_table(artifacts.selections, output / "policy_selections.csv"),
        "candidate_grid": _write_table(
            artifacts.candidate_grid, output / "policy_candidate_grid.csv"
        ),
        "temperature_fits": _write_table(
            artifacts.temperature_fits, output / "temperature_fits.csv"
        ),
        "calibration_summary": _write_table(
            artifacts.calibration_summary, output / "calibration_summary.csv"
        ),
        "reliability_bins": _write_table(
            artifacts.reliability_bins, output / "reliability_bins.csv"
        ),
    }
    summary = {
        "schema_version": 1,
        "input_ledgers": [
            {"path": str(Path(path)), "sha256": sha256_file(path)} for path in input_paths
        ],
        "config_sha256": None if config is None else sha256_json(config),
        "rows": {
            "calibrated_prefix_ledger": len(artifacts.calibrated_prefix_ledger),
            "decision_ledger": len(artifacts.decision_ledger),
            "fixed_predictions": len(artifacts.fixed_predictions),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    summary_path = output / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    paths["summary"] = summary_path
    return paths


def fixed_prefix_metrics(fixed_predictions: pd.DataFrame) -> pd.DataFrame:
    """Convenience summary used for immediate consistency checks."""

    required = {"prefix_ms", "label_id", "predicted_label"}
    missing = required - set(fixed_predictions.columns)
    if missing:
        raise ValueError(f"Fixed predictions are missing columns: {sorted(missing)}")
    rows = []
    for prefix_ms, group in fixed_predictions.groupby("prefix_ms", sort=True):
        metrics = binary_classification_metrics(group["label_id"], group["predicted_label"])
        rows.append({"prefix_ms": int(prefix_ms), **metrics.as_dict()})
    return pd.DataFrame(rows)
