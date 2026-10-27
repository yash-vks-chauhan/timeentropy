#!/usr/bin/env python3
"""Aggregate the primary and two frozen Transformer replication seeds."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from timeentropy.evaluation.ledger import (
    validate_decision_ledger,
    validate_oof_test_coverage,
    validate_prefix_ledger,
)
from timeentropy.provenance import environment_record, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts/evaluation/transformer-three-seed"
CONFIG_PATH = PROJECT_ROOT / "configs/rebuild_v1.yaml"
MANIFEST_PATH = PROJECT_ROOT / "data/metadata/torgo_manifest_v1.csv"
SEED_DIRECTORIES = {
    20260819: PROJECT_ROOT / "artifacts/evaluation/transformer-primary",
    20260820: PROJECT_ROOT / "artifacts/evaluation/transformer-seed20260820",
    20260821: PROJECT_ROOT / "artifacts/evaluation/transformer-seed20260821",
}
METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "dysarthric_recall",
    "specificity",
    "dysarthric_precision",
    "mean_latency_ms",
    "forced_fallback_rate",
)
PRIMARY_POLICY = "complete-temperature_scaled"


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["label_id"].to_numpy(dtype=np.int64)
    prediction = frame["predicted_label"].to_numpy(dtype=np.int64)
    return {
        "samples": len(frame),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "dysarthric_recall": float(recall_score(truth, prediction, pos_label=1)),
        "specificity": float(recall_score(truth, prediction, pos_label=0)),
        "dysarthric_precision": float(
            precision_score(truth, prediction, pos_label=1, zero_division=0)
        ),
        "mean_latency_ms": float(frame["decision_time_ms"].mean()),
        "forced_fallback_rate": float(frame["forced_fallback"].astype(bool).mean()),
    }


def load_seed(
    seed: int,
    directory: Path,
    expected_sample_ids: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prefix_path = directory / "calibrated_prefix_predictions.parquet"
    decision_path = directory / "decisions.parquet"
    evaluation_summary_path = directory / "evaluation_summary.json"
    report_manifest_path = (
        PROJECT_ROOT / f"artifacts/reports/transformer-seed{seed}/report_manifest.json"
        if seed != 20260819
        else PROJECT_ROOT / "artifacts/reports/transformer-primary/report_manifest.json"
    )
    for path in (prefix_path, decision_path, evaluation_summary_path, report_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    prefix = pd.read_parquet(prefix_path)
    decisions = pd.read_parquet(decision_path)
    validate_prefix_ledger(prefix)
    validate_oof_test_coverage(prefix, expected_sample_ids)
    validate_decision_ledger(
        decisions,
        prefix_ledger=prefix,
        prefix_prediction_column="calibrated_predicted_label",
    )
    if set(prefix["seed"].astype(int)) != {seed}:
        raise ValueError(f"Prefix ledger in {directory} does not belong to seed {seed}")
    # The decision ledger carries no seed column; every decision row names the run that
    # produced it, so its runs must be exactly the seed-verified runs of the prefix ledger.
    if not set(decisions["run_id"]).issubset(set(prefix["run_id"])):
        raise ValueError(f"Decision ledger in {directory} names runs absent from its prefix ledger")
    if any(f"seed{seed}" not in run_id for run_id in set(decisions["run_id"])):
        raise ValueError(f"Decision ledger in {directory} does not belong to seed {seed}")

    inputs = {
        "calibrated_prefix_ledger": {
            "path": str(prefix_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(prefix_path),
        },
        "decision_ledger": {
            "path": str(decision_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(decision_path),
        },
        "evaluation_summary": {
            "path": str(evaluation_summary_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(evaluation_summary_path),
        },
        "report_manifest": {
            "path": str(report_manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(report_manifest_path),
        },
    }
    return decisions, inputs


def seed_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    critical = float(t.ppf(0.975, df=len(SEED_DIRECTORIES) - 1))
    for policy_id, group in per_seed.groupby("policy_id", sort=True):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1))
            half_width = critical * sample_sd / math.sqrt(len(values))
            rows.append(
                {
                    "policy_id": policy_id,
                    "metric": metric,
                    "seeds": len(values),
                    "mean_across_seeds": mean,
                    "sample_sd_across_seeds": sample_sd,
                    "minimum_seed_value": float(values.min()),
                    "maximum_seed_value": float(values.max()),
                    "descriptive_t_interval_lower": mean - half_width,
                    "descriptive_t_interval_upper": mean + half_width,
                    "interval_warning": (
                        "Three-seed descriptive interval; not a speaker-generalization "
                        "confidence interval and not an independent-corpus uncertainty estimate."
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"sample_id": str})
    expected_sample_ids = manifest["sample_id"].astype(str)
    per_seed_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    input_records: dict[str, object] = {}
    reference_keys: dict[str, pd.DataFrame] = {}

    for seed, directory in SEED_DIRECTORIES.items():
        decisions, input_records[str(seed)] = load_seed(
            seed,
            directory,
            expected_sample_ids,
        )
        for policy_id, policy in decisions.groupby("policy_id", sort=True):
            if len(policy) != len(manifest):
                raise ValueError(
                    f"Seed {seed}, policy {policy_id} has {len(policy)} rows; "
                    f"expected {len(manifest)}"
                )
            ordered = policy.sort_values("sample_id").reset_index(drop=True)
            keys = ordered[["sample_id", "speaker_id", "fold", "label_id"]]
            if policy_id in reference_keys and not keys.equals(reference_keys[policy_id]):
                raise ValueError(f"Policy {policy_id} is not paired identically across seeds")
            reference_keys.setdefault(str(policy_id), keys)
            per_seed_rows.append(
                {"seed": seed, "policy_id": policy_id, **metrics(ordered)}
            )
            for fold, fold_frame in ordered.groupby("fold", sort=True):
                fold_rows.append(
                    {
                        "seed": seed,
                        "fold": int(fold),
                        "policy_id": policy_id,
                        **metrics(fold_frame),
                    }
                )

    per_seed = pd.DataFrame(per_seed_rows).sort_values(["policy_id", "seed"])
    per_fold = pd.DataFrame(fold_rows).sort_values(["policy_id", "seed", "fold"])
    across_seed = seed_summary(per_seed)
    per_seed_path = OUTPUT_DIR / "per_seed_policy_metrics.csv"
    per_fold_path = OUTPUT_DIR / "seed_fold_policy_metrics.csv"
    across_seed_path = OUTPUT_DIR / "across_seed_policy_summary.csv"
    per_seed.to_csv(per_seed_path, index=False)
    per_fold.to_csv(per_fold_path, index=False)
    across_seed.to_csv(across_seed_path, index=False)

    primary_seed_rows = per_seed.loc[per_seed["policy_id"].eq(PRIMARY_POLICY)]
    primary_summary_rows = across_seed.loc[
        across_seed["policy_id"].eq(PRIMARY_POLICY)
    ]
    primary_summary_path = OUTPUT_DIR / "primary_policy_three_seed_summary.json"
    primary_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "transformer-three-seed-summary-v1",
                "policy_id": PRIMARY_POLICY,
                "per_seed": primary_seed_rows.to_dict(orient="records"),
                "across_seed": primary_summary_rows.to_dict(orient="records"),
                "uncertainty_scope": (
                    "Variation across three training random seeds on identical frozen folds. "
                    "Speaker-cluster intervals in each full report remain the appropriate "
                    "generalization uncertainty analysis."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    output_paths = [
        per_seed_path,
        per_fold_path,
        across_seed_path,
        primary_summary_path,
    ]
    source_path = Path(__file__).resolve()
    aggregate_manifest = {
        "schema_version": "transformer-three-seed-aggregate-v1",
        "seeds": list(SEED_DIRECTORIES),
        "primary_policy_id": PRIMARY_POLICY,
        "expected_oof_samples_per_policy_seed": len(manifest),
        "policies": sorted(per_seed["policy_id"].astype(str).unique()),
        "inputs": input_records,
        "frozen_inputs": {
            "experiment_config": {
                "path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(CONFIG_PATH),
            },
            "manifest": {
                "path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(MANIFEST_PATH),
            },
        },
        "analysis_code": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(source_path),
        },
        "outputs": {
            path.name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in output_paths
        },
        "environment": environment_record(PROJECT_ROOT),
        "test_set_tuning_performed": False,
    }
    aggregate_manifest_path = OUTPUT_DIR / "aggregate_manifest.json"
    aggregate_manifest_path.write_text(
        json.dumps(aggregate_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(aggregate_manifest_path)


if __name__ == "__main__":
    main()
