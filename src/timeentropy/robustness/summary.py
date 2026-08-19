from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timeentropy.config import PROJECT_ROOT, project_path
from timeentropy.evaluation.metrics import (
    accuracy_earliness_harmonic_mean,
    binary_classification_metrics,
)
from timeentropy.evaluation.pipeline import read_ledger
from timeentropy.provenance import environment_record, sha256_file
from timeentropy.robustness.config import load_robustness_protocol

METRIC_COLUMNS = (
    "accuracy",
    "macro_f1",
    "dysarthric_recall",
    "specificity",
    "dysarthric_precision",
    "balanced_accuracy",
    "mean_latency_ms",
    "forced_fallback_rate",
    "consumed_fraction",
    "timeliness",
    "accuracy_earliness_harmonic_mean",
    "macro_f1_earliness_harmonic_mean",
)


@dataclass(frozen=True)
class RobustnessSummaryResult:
    pooled_metrics_path: Path
    fold_metrics_path: Path
    fold_mean_sd_path: Path
    summary_path: Path


def _metric_row(group: pd.DataFrame) -> dict[str, Any]:
    classification = binary_classification_metrics(
        group["label_id"].to_numpy(dtype=np.int64),
        group["predicted_label"].to_numpy(dtype=np.int64),
    )
    latency = group["decision_time_ms"].to_numpy(dtype=np.float64)
    mean_latency = float(latency.mean())
    consumed = mean_latency / 300.0
    return {
        **classification.as_dict(),
        "mean_latency_ms": mean_latency,
        "forced_fallback_rate": float(group["forced_fallback"].astype(bool).mean()),
        "consumed_fraction": consumed,
        "timeliness": 1.0 - consumed,
        "accuracy_earliness_harmonic_mean": accuracy_earliness_harmonic_mean(
            classification.accuracy, consumed
        ),
        "macro_f1_earliness_harmonic_mean": accuracy_earliness_harmonic_mean(
            classification.macro_f1, consumed
        ),
    }


def _validate_inputs(
    decisions: pd.DataFrame,
    *,
    expected_by_fold: dict[int, frozenset[str]],
    expected_analyses: frozenset[str],
    require_all_conditions: bool,
) -> None:
    required = {
        "sample_id",
        "speaker_id",
        "fold",
        "robustness_analysis_id",
        "robustness_role",
        "perturbation_kind",
        "snr_db",
        "onset_offset_ms",
        "policy_id",
        "label_id",
        "predicted_label",
        "decision_time_ms",
        "forced_fallback",
        "calibration_split_role",
        "selection_split_role",
    }
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"Robustness decisions are missing columns: {sorted(missing)}")
    if decisions.empty:
        raise ValueError("Robustness decisions must not be empty")
    key = ["robustness_analysis_id", "policy_id", "sample_id"]
    if decisions.duplicated(key).any():
        raise ValueError("A condition/policy contains duplicate out-of-fold sample decisions")
    if not decisions["decision_time_ms"].isin((150, 200, 250, 300)).all():
        raise ValueError("A robustness decision uses an unregistered prefix")
    if (
        not decisions["label_id"].isin((0, 1)).all()
        or not decisions["predicted_label"].isin((0, 1)).all()
    ):
        raise ValueError("Robustness decisions require binary labels")
    if not decisions["calibration_split_role"].eq("calibration").all():
        raise ValueError("Every robustness decision must use calibration-only temperature fitting")
    if not decisions["selection_split_role"].isin(("policy_tuning", "not_applicable")).all():
        raise ValueError("A robustness decision has an invalid selection source")

    observed_analyses = frozenset(decisions["robustness_analysis_id"].astype(str))
    if not observed_analyses.issubset(expected_analyses):
        raise ValueError("Decision inputs contain an analysis absent from the frozen protocol")
    if require_all_conditions and observed_analyses != expected_analyses:
        raise ValueError("Final robustness summary requires every prespecified test-time condition")
    expected_all = frozenset().union(*expected_by_fold.values())
    group_columns = ["robustness_analysis_id", "policy_id"]
    policy_sets = decisions.groupby("robustness_analysis_id")["policy_id"].agg(
        lambda values: frozenset(values.astype(str))
    )
    if policy_sets.nunique() != 1:
        raise ValueError("Every robustness condition must contain the same policy set")
    for identity, group in decisions.groupby(group_columns, sort=False):
        observed_all = frozenset(group["sample_id"].astype(str))
        if observed_all != expected_all:
            raise ValueError(f"Condition/policy {identity} lacks complete OOF sample coverage")
        for fold, expected in expected_by_fold.items():
            observed = frozenset(
                group.loc[group["fold"].astype(int).eq(fold), "sample_id"].astype(str)
            )
            if observed != expected:
                raise ValueError(f"Condition/policy {identity} has wrong coverage for fold {fold}")
    if decisions.groupby("sample_id")["label_id"].nunique().gt(1).any():
        raise ValueError("A sample label differs across robustness conditions or policies")


def summarize_test_time_robustness(
    robustness_config_path: str | Path,
    *,
    decision_ledger_paths: Sequence[str | Path],
    output_directory: str | Path,
    require_all_conditions: bool = True,
) -> RobustnessSummaryResult:
    """Combine fold decisions without refitting or selecting any parameter."""

    if not decision_ledger_paths:
        raise ValueError("At least one robustness decision ledger is required")
    protocol = load_robustness_protocol(robustness_config_path)
    decisions = pd.concat([read_ledger(path) for path in decision_ledger_paths], ignore_index=True)
    manifest = pd.read_csv(
        project_path(protocol.parent_config["data"]["manifest"]),
        dtype={"sample_id": str},
    )
    fold_manifest = pd.read_csv(
        project_path(protocol.parent_config["data"]["fold_manifest"]),
        dtype={"sample_id": str},
    )
    expected_by_fold = {
        fold: frozenset(
            fold_manifest.loc[
                fold_manifest["fold"].astype(int).eq(fold) & fold_manifest["role"].eq("test"),
                "sample_id",
            ].astype(str)
        )
        for fold in protocol.folds
    }
    if any(not values for values in expected_by_fold.values()):
        raise ValueError("Every frozen fold must have test samples")
    expected_all = frozenset().union(*expected_by_fold.values())
    if expected_all != frozenset(manifest["sample_id"].astype(str)):
        raise ValueError("Frozen fold tests do not form exactly one complete OOF cohort")
    _validate_inputs(
        decisions,
        expected_by_fold=expected_by_fold,
        expected_analyses=frozenset(
            analysis.analysis_id for analysis in protocol.test_time_analyses
        ),
        require_all_conditions=require_all_conditions,
    )

    identity_columns = [
        "robustness_analysis_id",
        "robustness_role",
        "perturbation_kind",
        "snr_db",
        "onset_offset_ms",
        "policy_id",
    ]
    pooled_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for identity, group in decisions.groupby(identity_columns, sort=True, dropna=False):
        common = dict(zip(identity_columns, identity, strict=True))
        pooled_rows.append({**common, "fold": "pooled_oof", **_metric_row(group)})
        for fold, fold_group in group.groupby("fold", sort=True):
            fold_rows.append({**common, "fold": int(fold), **_metric_row(fold_group)})
    pooled = pd.DataFrame(pooled_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_summary_rows: list[dict[str, Any]] = []
    for identity, group in fold_metrics.groupby(identity_columns, sort=True, dropna=False):
        row = {
            **dict(zip(identity_columns, identity, strict=True)),
            "folds": int(group["fold"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_fold_mean"] = float(group[metric].mean())
            row[f"{metric}_fold_sample_sd"] = float(group[metric].std(ddof=1))
        fold_summary_rows.append(row)
    fold_summary = pd.DataFrame(fold_summary_rows)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pooled_path = output / "pooled_oof_metrics.csv"
    fold_path = output / "fold_metrics.csv"
    fold_summary_path = output / "fold_mean_sample_sd.csv"
    pooled.to_csv(pooled_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    fold_summary.to_csv(fold_summary_path, index=False)
    summary = {
        "schema_version": 1,
        "execution_stage": "post_inference_summary_only",
        "test_set_tuning_permitted": False,
        "parameters_fitted_during_summary": [],
        "require_all_prespecified_conditions": require_all_conditions,
        "robustness_config": {
            "path": str(protocol.config_path),
            "sha256": protocol.config_sha256,
        },
        "input_decision_ledgers": [
            {"path": str(Path(path)), "sha256": sha256_file(path)} for path in decision_ledger_paths
        ],
        "rows": {
            "decisions": len(decisions),
            "pooled_metrics": len(pooled),
            "fold_metrics": len(fold_metrics),
        },
        "estimands": {
            "pooled": "all out-of-fold sample decisions pooled once",
            "fold": "unweighted fold mean and sample standard deviation",
        },
        "environment": environment_record(PROJECT_ROOT),
        "outputs": {
            "pooled": {"path": str(pooled_path), "sha256": sha256_file(pooled_path)},
            "fold": {"path": str(fold_path), "sha256": sha256_file(fold_path)},
            "fold_mean_sd": {
                "path": str(fold_summary_path),
                "sha256": sha256_file(fold_summary_path),
            },
        },
    }
    summary_path = output / "robustness_combined_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return RobustnessSummaryResult(
        pooled_metrics_path=pooled_path,
        fold_metrics_path=fold_path,
        fold_mean_sd_path=fold_summary_path,
        summary_path=summary_path,
    )
