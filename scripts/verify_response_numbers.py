#!/usr/bin/env python3
"""Check that the numbers quoted in the reviewer response match the generated reports.

The response document is the artefact reviewers read. Every figure in it is copied by hand
from a report table, so it can silently drift whenever the pipeline is re-run. This script
re-derives each quoted value from the reports and fails if any disagrees.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Check:
    label: str
    quoted: float
    actual: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return abs(self.quoted - self.actual) <= self.tolerance


def build_checks(report_dir: Path) -> list[Check]:
    pooled = pd.read_csv(report_dir / "pooled_metrics.csv").set_index("policy_id")
    folds = pd.read_csv(report_dir / "fold_summary.csv").set_index("policy_id")
    calibration = pd.read_csv(report_dir / "calibration_metrics.csv")
    calibration = calibration[
        calibration["fold"].isna() & (calibration["prefix_ms"] == 300)
    ].set_index("calibration_state")
    matrix = pd.read_csv(report_dir / "confusion_matrices.csv")
    matrix = matrix[matrix["policy_id"] == "complete-temperature_scaled"].set_index(
        ["actual_class", "predicted_class"]
    )["count"]

    control_correct = int(matrix.loc[("control", "control")])
    control_as_dysarthric = int(matrix.loc[("control", "dysarthric")])
    dysarthric_as_control = int(matrix.loc[("dysarthric", "control")])
    dysarthric_correct = int(matrix.loc[("dysarthric", "dysarthric")])
    total = control_correct + control_as_dysarthric + dysarthric_as_control + dysarthric_correct

    def accuracy(policy: str) -> float:
        return float(pooled.loc[policy, "accuracy"]) * 100.0

    def fallback(policy: str) -> float:
        return float(pooled.loc[policy, "forced_fallback_rate"]) * 100.0

    return [
        Check("R1.11 pooled accuracy at fixed 200 ms", 43.07, accuracy("fixed_200"), 0.005),
        Check("R1.11 pooled accuracy at fixed 250 ms", 42.11, accuracy("fixed_250"), 0.005),
        Check("R1.11 pooled accuracy at fixed 300 ms", 41.94, accuracy("fixed_300"), 0.005),
        Check("R1.6 pooled accuracy, complete policy", 41.95, accuracy("complete-temperature_scaled"), 0.005),
        Check("R1.1 entropy-only accuracy", 42.33, accuracy("entropy_only-raw"), 0.005),
        Check("R1.1 max-softmax accuracy (must equal entropy-only)", 42.33, accuracy("max_softmax_confidence-raw"), 0.005),
        Check("R1.6 unweighted fold-mean accuracy", 43.90, float(folds.loc["complete-temperature_scaled", "accuracy_mean"]) * 100.0, 0.005),
        Check("R1.6 fold accuracy SD", 19.58, float(folds.loc["complete-temperature_scaled", "accuracy_sd"]) * 100.0, 0.02),
        Check("R1.13 raw expected calibration error", 0.429, float(calibration.loc["raw", "expected_calibration_error"]), 0.001),
        Check("R1.13 calibrated expected calibration error", 0.212, float(calibration.loc["temperature_scaled", "expected_calibration_error"]), 0.001),
        Check("R1.13 mean fitted temperature", 11.23, float(calibration.loc["temperature_scaled", "temperature_mean"]), 0.01),
        Check("R1.13 forced fallback, raw", 8.9, fallback("complete-raw"), 0.05),
        Check("R1.13 forced fallback, calibrated", 64.2, fallback("complete-temperature_scaled"), 0.05),
        Check("R1.20 confusion cell control -> control", 1669, control_correct, 0),
        Check("R1.20 confusion cell control -> dysarthric", 3528, control_as_dysarthric, 0),
        Check("R1.20 confusion cell dysarthric -> control", 1192, dysarthric_as_control, 0),
        Check("R1.20 confusion cell dysarthric -> dysarthric", 1742, dysarthric_correct, 0),
        Check("R1.20 accuracy derived from the matrix", 41.95, (control_correct + dysarthric_correct) / total * 100.0, 0.005),
        Check("R3.10 false-positive rate", 67.9, control_as_dysarthric / (control_correct + control_as_dysarthric) * 100.0, 0.05),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/reports/transformer-primary"),
        help="Reports for the headline model quoted in the response document.",
    )
    arguments = parser.parse_args()

    checks = build_checks(arguments.report_dir)
    for check in checks:
        marker = "ok" if check.ok else "MISMATCH"
        print(f"[{marker:>8}] {check.label}: quoted {check.quoted}, actual {check.actual:.4f}")

    failures = [check for check in checks if not check.ok]
    print(f"\n{len(checks) - len(failures)}/{len(checks)} quoted values verified")
    if failures:
        raise SystemExit(f"{len(failures)} quoted value(s) no longer match the reports")


if __name__ == "__main__":
    main()
