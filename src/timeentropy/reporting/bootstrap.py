from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_STAT_COLUMNS = (
    "true_control",
    "false_dysarthric",
    "false_control",
    "true_dysarthric",
    "samples",
    "latency_sum_ms",
    "fallback_count",
)


@dataclass(frozen=True)
class BootstrapResult:
    intervals: pd.DataFrame
    replicates: int
    seed: int


def _speaker_sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for speaker, group in frame.groupby("speaker_id", sort=True):
        labels = group["label_id"].unique()
        if len(labels) != 1:
            raise ValueError(f"Speaker {speaker} has inconsistent diagnosis labels")
        truth = group["label_id"].to_numpy(dtype=np.int64)
        prediction = group["predicted_label"].to_numpy(dtype=np.int64)
        records.append(
            {
                "speaker_id": speaker,
                "label_id": int(labels[0]),
                "true_control": int(np.sum((truth == 0) & (prediction == 0))),
                "false_dysarthric": int(np.sum((truth == 0) & (prediction == 1))),
                "false_control": int(np.sum((truth == 1) & (prediction == 0))),
                "true_dysarthric": int(np.sum((truth == 1) & (prediction == 1))),
                "samples": len(group),
                "latency_sum_ms": float(group["decision_time_ms"].sum()),
                "fallback_count": int(group["forced_fallback"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(records)


def _bootstrap_totals(
    speaker_statistics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    values = speaker_statistics[list(_STAT_COLUMNS)].to_numpy(dtype=np.float64)
    totals = np.zeros((replicates, len(_STAT_COLUMNS)), dtype=np.float64)
    for label in sorted(speaker_statistics["label_id"].unique()):
        indices = np.flatnonzero(speaker_statistics["label_id"].to_numpy() == label)
        if len(indices) == 0:
            continue
        sampled = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        totals += values[sampled].sum(axis=1)
    return totals


def _metric_arrays(totals: np.ndarray) -> dict[str, np.ndarray]:
    tn, fp, fn, tp, samples, latency_sum, fallback_count = totals.T

    def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator != 0,
        )

    control_f1 = divide(2 * tn, 2 * tn + fp + fn)
    dysarthric_f1 = divide(2 * tp, 2 * tp + fp + fn)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    return {
        "accuracy": divide(tn + tp, samples),
        "macro_f1": (control_f1 + dysarthric_f1) / 2.0,
        "dysarthric_recall": recall,
        "specificity": specificity,
        "dysarthric_precision": precision,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mean_latency_ms": divide(latency_sum, samples),
        "forced_fallback_rate": divide(fallback_count, samples),
    }


def _point_totals(speaker_statistics: pd.DataFrame) -> np.ndarray:
    return speaker_statistics[list(_STAT_COLUMNS)].to_numpy(dtype=np.float64).sum(axis=0)[None, :]


def clustered_speaker_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 20_260_819,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Return deterministic percentile CIs from diagnosis-stratified speaker resampling."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    required = {
        "speaker_id",
        "label_id",
        "predicted_label",
        "decision_time_ms",
        "forced_fallback",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Bootstrap frame lacks required columns: {missing}")
    statistics = _speaker_sufficient_statistics(frame)
    if len(statistics) < 2:
        raise ValueError("Speaker bootstrap requires at least two speakers")
    draws = _metric_arrays(_bootstrap_totals(statistics, replicates, seed))
    point = _metric_arrays(_point_totals(statistics))
    alpha = (1.0 - confidence) / 2.0
    rows = []
    for metric, values in draws.items():
        rows.append(
            {
                "metric": metric,
                "estimate": float(point[metric][0]),
                "ci_lower": float(np.quantile(values, alpha)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                "confidence": confidence,
                "method": "percentile_speaker_cluster_bootstrap_stratified_by_label",
                "replicates": replicates,
                "seed": seed,
                "speaker_clusters": len(statistics),
            }
        )
    return BootstrapResult(intervals=pd.DataFrame(rows), replicates=replicates, seed=seed)


def paired_cluster_bootstrap_differences(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    """Return left-minus-right metric differences using shared speaker resamples."""

    left_stats = (
        _speaker_sufficient_statistics(left).sort_values("speaker_id").reset_index(drop=True)
    )
    right_stats = (
        _speaker_sufficient_statistics(right).sort_values("speaker_id").reset_index(drop=True)
    )
    if not left_stats[["speaker_id", "label_id"]].equals(right_stats[["speaker_id", "label_id"]]):
        raise ValueError("Paired systems do not contain the same speaker clusters and labels")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    rng = np.random.default_rng(seed)
    labels = left_stats["label_id"].to_numpy(dtype=np.int64)
    left_values = left_stats[list(_STAT_COLUMNS)].to_numpy(dtype=np.float64)
    right_values = right_stats[list(_STAT_COLUMNS)].to_numpy(dtype=np.float64)
    left_totals = np.zeros((replicates, len(_STAT_COLUMNS)), dtype=np.float64)
    right_totals = np.zeros_like(left_totals)
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        sampled = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        left_totals += left_values[sampled].sum(axis=1)
        right_totals += right_values[sampled].sum(axis=1)

    left_metrics = _metric_arrays(left_totals)
    right_metrics = _metric_arrays(right_totals)
    left_point = _metric_arrays(left_values.sum(axis=0)[None, :])
    right_point = _metric_arrays(right_values.sum(axis=0)[None, :])
    alpha = (1.0 - confidence) / 2.0
    rows = []
    for metric in left_metrics:
        differences = left_metrics[metric] - right_metrics[metric]
        nonpositive = float(np.mean(differences <= 0.0))
        nonnegative = float(np.mean(differences >= 0.0))
        rows.append(
            {
                "metric": metric,
                "left_estimate": float(left_point[metric][0]),
                "right_estimate": float(right_point[metric][0]),
                "difference_left_minus_right": float(
                    left_point[metric][0] - right_point[metric][0]
                ),
                "ci_lower": float(np.quantile(differences, alpha)),
                "ci_upper": float(np.quantile(differences, 1.0 - alpha)),
                "bootstrap_two_sided_sign_probability": min(
                    1.0, 2.0 * min(nonpositive, nonnegative)
                ),
                "confidence": confidence,
                "method": "paired_percentile_speaker_cluster_bootstrap_stratified_by_label",
                "replicates": replicates,
                "seed": seed,
                "speaker_clusters": len(left_stats),
            }
        )
    return pd.DataFrame(rows)
