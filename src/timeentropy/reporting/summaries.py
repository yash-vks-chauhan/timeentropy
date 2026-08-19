from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from timeentropy.evaluation.calibration import calibration_metrics
from timeentropy.evaluation.ledger import validate_decision_ledger, validate_prefix_ledger
from timeentropy.evaluation.metrics import (
    accuracy_earliness_harmonic_mean,
    binary_classification_metrics,
)

SYSTEM_COLUMNS = ("model_id", "seed", "onset_variant", "pre_roll_ms", "policy_id")
CALIBRATION_SYSTEM_COLUMNS = ("model_id", "seed", "onset_variant", "pre_roll_ms")


@dataclass(frozen=True)
class SummaryTables:
    systems: pd.DataFrame
    pooled_metrics: pd.DataFrame
    confusion_matrices: pd.DataFrame
    class_metrics: pd.DataFrame
    error_breakdown: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_summary: pd.DataFrame
    latency_counts: pd.DataFrame
    latency_summary: pd.DataFrame


@dataclass(frozen=True)
class CalibrationTables:
    metrics: pd.DataFrame
    reliability_bins: pd.DataFrame


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _system_id(values: dict[str, object], columns: tuple[str, ...] = SYSTEM_COLUMNS) -> str:
    return "|".join(f"{column}={values[column]}" for column in columns)


def enrich_decision_ledger(prefix: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Validate ledgers and attach test-speaker/fold/model metadata to each decision."""

    prefix_report = validate_prefix_ledger(prefix)
    prediction_column = (
        "calibrated_predicted_label" if prefix_report.calibrated else "predicted_label"
    )
    validate_decision_ledger(
        decisions,
        prefix_ledger=prefix,
        prefix_prediction_column=prediction_column,
    )

    metadata_columns = [
        "run_id",
        "sample_id",
        "speaker_id",
        "fold",
        "seed",
        "model_id",
        "split_role",
        "onset_variant",
        "pre_roll_ms",
        "label_id",
    ]
    metadata = prefix[metadata_columns].drop_duplicates()
    if metadata.duplicated(["run_id", "sample_id"]).any():
        raise ValueError("Prefix metadata is inconsistent within at least one run/sample")

    enriched = decisions.merge(
        metadata,
        how="left",
        on=["run_id", "sample_id"],
        suffixes=("", "_prefix"),
        validate="many_to_one",
    )
    if enriched["speaker_id"].isna().any():
        raise ValueError("At least one decision lacks matching prefix-ledger metadata")
    if not np.array_equal(
        enriched["label_id"].to_numpy(dtype=np.int64),
        enriched["label_id_prefix"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Decision and prefix ledgers disagree on label_id")
    if not enriched["split_role"].eq("test").all():
        raise ValueError("Reporting accepts out-of-fold test decisions only")
    enriched = enriched.drop(columns=["label_id_prefix"])

    enriched["system_id"] = [
        _system_id(record) for record in enriched[list(SYSTEM_COLUMNS)].to_dict(orient="records")
    ]
    if enriched.duplicated(["system_id", "sample_id"]).any():
        raise ValueError("A system has more than one out-of-fold decision for a sample")
    return enriched.sort_values(["system_id", "fold", "sample_id"]).reset_index(drop=True)


def _classification_record(group: pd.DataFrame) -> dict[str, int | float]:
    metrics = binary_classification_metrics(group["label_id"], group["predicted_label"])
    return metrics.as_dict()


def _latency_record(group: pd.DataFrame, *, maximum_observation_ms: int = 300) -> dict[str, float]:
    times = group["decision_time_ms"].to_numpy(dtype=np.float64)
    if maximum_observation_ms <= 0 or np.any(times > maximum_observation_ms):
        raise ValueError("Decision times exceed the registered observation horizon")
    return {
        "mean_latency_ms": float(times.mean()),
        "latency_sd_ms": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        "median_latency_ms": float(np.median(times)),
        "p95_latency_ms": float(np.quantile(times, 0.95)),
        "forced_fallback_rate": float(group["forced_fallback"].astype(bool).mean()),
        "accepted_at_endpoint_rate": float(group["accepted_at_endpoint"].astype(bool).mean()),
    }


def _joint_earliness_record(
    classification: dict[str, int | float],
    latency: dict[str, float],
    *,
    maximum_observation_ms: int = 300,
) -> dict[str, float]:
    consumed = latency["mean_latency_ms"] / maximum_observation_ms
    return {
        "maximum_observation_ms": float(maximum_observation_ms),
        "consumed_fraction": consumed,
        "timeliness": 1.0 - consumed,
        "accuracy_earliness_harmonic_mean": accuracy_earliness_harmonic_mean(
            float(classification["accuracy"]), consumed
        ),
        "macro_f1_earliness_harmonic_mean": accuracy_earliness_harmonic_mean(
            float(classification["macro_f1"]), consumed
        ),
    }


def _system_metadata(group: pd.DataFrame) -> dict[str, object]:
    row = group.iloc[0]
    return {"system_id": row["system_id"], **{column: row[column] for column in SYSTEM_COLUMNS}}


def build_summary_tables(enriched: pd.DataFrame) -> SummaryTables:
    """Build pooled, fold-specific, class, error, and latency tables.

    Pooled out-of-fold metrics and unweighted fold averages are deliberately emitted in
    separate tables with explicit estimand labels.  They must never be substituted for
    one another.
    """

    required = {
        "system_id",
        "sample_id",
        "speaker_id",
        "fold",
        "label_id",
        "predicted_label",
        "decision_time_ms",
        "forced_fallback",
        "accepted_at_endpoint",
        *SYSTEM_COLUMNS,
    }
    missing = sorted(required - set(enriched.columns))
    if missing:
        raise ValueError(f"Enriched decisions lack required columns: {missing}")
    if enriched.empty:
        raise ValueError("Enriched decisions must not be empty")

    systems: list[dict[str, object]] = []
    pooled: list[dict[str, object]] = []
    confusion: list[dict[str, object]] = []
    classes: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    latency_counts: list[dict[str, object]] = []
    latency_summary: list[dict[str, object]] = []

    for _, group in enriched.groupby("system_id", sort=True):
        metadata = _system_metadata(group)
        systems.append(
            {
                **metadata,
                "samples": len(group),
                "speakers": int(group["speaker_id"].nunique()),
                "folds": int(group["fold"].nunique()),
            }
        )
        classification = _classification_record(group)
        latency = _latency_record(group)
        joint = _joint_earliness_record(classification, latency)
        pooled.append(
            {
                **metadata,
                "estimand": "pooled_out_of_fold_utterances",
                "aggregation_note": "One pooled confusion matrix over all OOF utterances",
                **classification,
                **latency,
                **joint,
            }
        )

        tn = int(classification["true_control"])
        fp = int(classification["false_dysarthric"])
        fn = int(classification["false_control"])
        tp = int(classification["true_dysarthric"])
        for actual, predicted, count in (
            ("control", "control", tn),
            ("control", "dysarthric", fp),
            ("dysarthric", "control", fn),
            ("dysarthric", "dysarthric", tp),
        ):
            confusion.append(
                {**metadata, "actual_class": actual, "predicted_class": predicted, "count": count}
            )

        control_precision = _safe_divide(tn, tn + fn)
        control_recall = _safe_divide(tn, tn + fp)
        control_f1 = _safe_divide(2 * tn, 2 * tn + fp + fn)
        dysarthric_precision = _safe_divide(tp, tp + fp)
        dysarthric_recall = _safe_divide(tp, tp + fn)
        dysarthric_f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
        classes.extend(
            [
                {
                    **metadata,
                    "class_label": "control",
                    "label_id": 0,
                    "support": tn + fp,
                    "precision": control_precision,
                    "recall": control_recall,
                    "f1": control_f1,
                },
                {
                    **metadata,
                    "class_label": "dysarthric",
                    "label_id": 1,
                    "support": tp + fn,
                    "precision": dysarthric_precision,
                    "recall": dysarthric_recall,
                    "f1": dysarthric_f1,
                },
            ]
        )
        errors.append(
            {
                **metadata,
                "control_support": tn + fp,
                "dysarthric_support": tp + fn,
                "false_positive_dysarthric_count": fp,
                "false_positive_rate": _safe_divide(fp, tn + fp),
                "false_negative_dysarthric_count": fn,
                "false_negative_rate": _safe_divide(fn, tp + fn),
            }
        )

        latency_summary.append({**metadata, **latency})
        total = len(group)
        for decision_time, time_group in group.groupby("decision_time_ms", sort=True):
            latency_counts.append(
                {
                    **metadata,
                    "decision_time_ms": int(decision_time),
                    "count": len(time_group),
                    "proportion": float(len(time_group) / total),
                    "forced_fallback_count": int(time_group["forced_fallback"].astype(bool).sum()),
                    "accepted_at_endpoint_count": int(
                        time_group["accepted_at_endpoint"].astype(bool).sum()
                    ),
                }
            )

        for fold, fold_group in group.groupby("fold", sort=True):
            fold_classification = _classification_record(fold_group)
            fold_latency = _latency_record(fold_group)
            folds.append(
                {
                    **metadata,
                    "fold": int(fold),
                    "estimand": "fold_specific",
                    **fold_classification,
                    **fold_latency,
                    **_joint_earliness_record(fold_classification, fold_latency),
                }
            )

    fold_frame = pd.DataFrame(folds)
    fold_summary: list[dict[str, object]] = []
    summary_metrics = (
        "accuracy",
        "macro_f1",
        "dysarthric_recall",
        "specificity",
        "dysarthric_precision",
        "balanced_accuracy",
        "mean_latency_ms",
        "forced_fallback_rate",
        "accuracy_earliness_harmonic_mean",
        "macro_f1_earliness_harmonic_mean",
    )
    for _, group in fold_frame.groupby("system_id", sort=True):
        metadata = {column: group.iloc[0][column] for column in ("system_id", *SYSTEM_COLUMNS)}
        record: dict[str, object] = {
            **metadata,
            "estimand": "unweighted_fold_mean_and_sample_sd",
            "aggregation_note": (
                "Arithmetic mean across folds; not the pooled OOF confusion-matrix estimand"
            ),
            "fold_count": len(group),
        }
        for metric in summary_metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        fold_summary.append(record)

    return SummaryTables(
        systems=pd.DataFrame(systems),
        pooled_metrics=pd.DataFrame(pooled),
        confusion_matrices=pd.DataFrame(confusion),
        class_metrics=pd.DataFrame(classes),
        error_breakdown=pd.DataFrame(errors),
        fold_metrics=fold_frame,
        fold_summary=pd.DataFrame(fold_summary),
        latency_counts=pd.DataFrame(latency_counts),
        latency_summary=pd.DataFrame(latency_summary),
    )


def build_calibration_tables(prefix: pd.DataFrame, bins: int = 15) -> CalibrationTables:
    """Build raw/calibrated calibration summaries at every prefix and fold aggregation."""

    report = validate_prefix_ledger(prefix)
    test = prefix[prefix["split_role"] == "test"].copy()
    if test.empty:
        raise ValueError("Prefix ledger contains no out-of-fold test rows")
    test["calibration_system_id"] = [
        _system_id(record, CALIBRATION_SYSTEM_COLUMNS)
        for record in test[list(CALIBRATION_SYSTEM_COLUMNS)].to_dict(orient="records")
    ]

    metric_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    probability_sets = [
        ("raw", "prob_control", "prob_dysarthric"),
    ]
    if report.calibrated:
        probability_sets.append(
            ("temperature_scaled", "calibrated_prob_control", "calibrated_prob_dysarthric")
        )

    pooled_groups = [
        ("pooled_out_of_fold_utterances", None, group)
        for _, group in test.groupby(["calibration_system_id", "prefix_ms"], sort=True)
    ]
    fold_groups = [
        ("fold_specific", int(key[2]), group)
        for key, group in test.groupby(["calibration_system_id", "prefix_ms", "fold"], sort=True)
    ]
    for aggregation, fold, group in [*pooled_groups, *fold_groups]:
        first = group.iloc[0]
        metadata = {
            "calibration_system_id": first["calibration_system_id"],
            **{column: first[column] for column in CALIBRATION_SYSTEM_COLUMNS},
            "aggregation": aggregation,
            "fold": fold,
            "prefix_ms": int(first["prefix_ms"]),
        }
        temperatures = (
            group[["run_id", "temperature"]]
            .drop_duplicates()["temperature"]
            .to_numpy(dtype=np.float64)
            if report.calibrated
            else np.asarray([], dtype=np.float64)
        )
        for state, control_column, dysarthric_column in probability_sets:
            probabilities = group[[control_column, dysarthric_column]].to_numpy(dtype=np.float64)
            result = calibration_metrics(probabilities, group["label_id"], bins=bins)
            metric_rows.append(
                {
                    **metadata,
                    "calibration_state": state,
                    "samples": result.samples,
                    "bins": result.bins,
                    "negative_log_likelihood": result.negative_log_likelihood,
                    "binary_brier_score": result.binary_brier_score,
                    "expected_calibration_error": result.expected_calibration_error,
                    "temperature_mean": (
                        float(temperatures.mean()) if state == "temperature_scaled" else None
                    ),
                    "temperature_sd": (
                        float(temperatures.std(ddof=1))
                        if state == "temperature_scaled" and len(temperatures) > 1
                        else (0.0 if state == "temperature_scaled" else None)
                    ),
                }
            )
            for bin_index, reliability in enumerate(result.reliability_bins):
                reliability_rows.append(
                    {
                        **metadata,
                        "calibration_state": state,
                        "bin_index": bin_index,
                        "lower_bound": reliability.lower_bound,
                        "upper_bound": reliability.upper_bound,
                        "samples": reliability.samples,
                        "mean_confidence": reliability.mean_confidence,
                        "accuracy": reliability.accuracy,
                    }
                )

    return CalibrationTables(
        metrics=pd.DataFrame(metric_rows),
        reliability_bins=pd.DataFrame(reliability_rows),
    )
