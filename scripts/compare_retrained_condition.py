#!/usr/bin/env python3
"""Compare one condition-matched retrained system with a baseline on identical utterances.

Every retrained input-definition ablation asks the same question: on the same held-out
utterances, does changing one input definition change the decision? The primary inference is
a paired speaker-cluster bootstrap, because utterances within a speaker are not independent.
The utterance-level McNemar result is reported alongside it only as a descriptive check.

This reproduces the frozen zero-versus-30 ms pre-roll analysis and generalizes it to the
onset-mode comparators; it reads decision ledgers only and never trains or refits anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from timeentropy.evaluation.ledger import validate_decision_ledger
from timeentropy.provenance import environment_record, sha256_file
from timeentropy.reporting.bootstrap import paired_cluster_bootstrap_differences

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIR_KEYS = ["sample_id", "speaker_id", "fold", "label_id"]
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


def selected(path: Path, policy_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    validate_decision_ledger(frame)
    chosen = frame.loc[frame["policy_id"].eq(policy_id)].copy()
    if chosen.empty:
        raise SystemExit(f"No {policy_id!r} decisions in {path}")
    return chosen.sort_values("sample_id").reset_index(drop=True)


def point_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["label_id"].to_numpy()
    prediction = frame["predicted_label"].to_numpy()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-decisions", type=Path, required=True)
    parser.add_argument(
        "--baseline-decisions",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluation/transformer-primary/decisions.parquet",
    )
    parser.add_argument("--condition-label", required=True, help="Column name for the condition.")
    parser.add_argument("--baseline-label", default="primary", help="Column name for the baseline.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-id", default="complete-temperature_scaled")
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    condition = selected(arguments.condition_decisions, arguments.policy_id)
    baseline = selected(arguments.baseline_decisions, arguments.policy_id)
    if not condition[PAIR_KEYS].equals(baseline[PAIR_KEYS]):
        raise SystemExit("The two conditions are not paired on identical out-of-fold samples")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    contrast = f"{arguments.condition_label}_minus_{arguments.baseline_label}"
    difference_column = f"difference_{contrast}"

    bootstrap = paired_cluster_bootstrap_differences(
        condition,
        baseline,
        replicates=arguments.replicates,
        seed=arguments.bootstrap_seed,
        confidence=arguments.confidence,
    )
    bootstrap.insert(0, "contrast", contrast)
    bootstrap_path = arguments.output_dir / "paired_speaker_bootstrap.csv"
    bootstrap.to_csv(bootstrap_path, index=False)

    fold_rows: list[dict[str, object]] = []
    for fold in sorted(condition["fold"].astype(int).unique()):
        condition_fold = point_metrics(condition.loc[condition["fold"].astype(int).eq(fold)])
        baseline_fold = point_metrics(baseline.loc[baseline["fold"].astype(int).eq(fold)])
        for metric in METRICS:
            fold_rows.append(
                {
                    "fold": fold,
                    "metric": metric,
                    arguments.condition_label: condition_fold[metric],
                    arguments.baseline_label: baseline_fold[metric],
                    difference_column: condition_fold[metric] - baseline_fold[metric],
                    "samples": condition_fold["samples"],
                }
            )
    fold_path = arguments.output_dir / "paired_fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    pooled_rows = []
    condition_pooled = point_metrics(condition)
    baseline_pooled = point_metrics(baseline)
    for metric in METRICS:
        pooled_rows.append(
            {
                "metric": metric,
                arguments.condition_label: condition_pooled[metric],
                arguments.baseline_label: baseline_pooled[metric],
                difference_column: condition_pooled[metric] - baseline_pooled[metric],
            }
        )
    pooled_path = arguments.output_dir / "pooled_metrics.csv"
    pd.DataFrame(pooled_rows).to_csv(pooled_path, index=False)

    condition_correct = condition["predicted_label"].eq(condition["label_id"])
    baseline_correct = baseline["predicted_label"].eq(baseline["label_id"])
    condition_only = int((condition_correct & ~baseline_correct).sum())
    baseline_only = int((~condition_correct & baseline_correct).sum())
    discordant = condition_only + baseline_only
    mcnemar = {
        "condition_correct_baseline_wrong": condition_only,
        "condition_wrong_baseline_correct": baseline_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": (
            float(binomtest(min(condition_only, baseline_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
        "method": "exact_mcnemar_binomial",
        "scope_warning": (
            "Descriptive utterance-level sensitivity analysis; the paired speaker-cluster "
            "bootstrap is the primary inference."
        ),
    }
    mcnemar_path = arguments.output_dir / "mcnemar_sensitivity.json"
    mcnemar_path.write_text(json.dumps(mcnemar, indent=2, sort_keys=True) + "\n")

    outputs = [bootstrap_path, fold_path, pooled_path, mcnemar_path]
    summary = {
        "schema_version": "paired-condition-comparison-v1",
        "contrast": contrast,
        "policy_id": arguments.policy_id,
        "paired_rows": len(condition),
        "speaker_clusters": int(condition["speaker_id"].nunique()),
        "folds": sorted(int(value) for value in condition["fold"].unique()),
        "bootstrap": {
            "unit": "speaker",
            "diagnosis_stratified": True,
            "paired": True,
            "replicates": arguments.replicates,
            "seed": arguments.bootstrap_seed,
            "confidence": arguments.confidence,
        },
        "inputs": {
            arguments.condition_label: {
                "path": str(arguments.condition_decisions.resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(arguments.condition_decisions),
            },
            arguments.baseline_label: {
                "path": str(arguments.baseline_decisions.resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(arguments.baseline_decisions),
            },
        },
        "outputs": {
            path.name: {
                "path": str(path.resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in outputs
        },
        "environment": environment_record(PROJECT_ROOT),
    }
    summary_path = arguments.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for path in [*outputs, summary_path]:
        print(path)


if __name__ == "__main__":
    main()
