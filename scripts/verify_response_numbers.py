#!/usr/bin/env python3
"""Check that the numbers quoted in the submission documents match the generated artifacts.

The manuscript and the response letter are what reviewers read, and every figure in them is
copied by hand from an artifact table, so they can silently drift whenever the pipeline is
re-run. This script re-derives each quoted value from its source artifact and fails if any
disagrees. It also confirms the value is actually written in one of the documents, so a check
cannot keep passing after the sentence it guards has been rewritten.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SYSTEM = (
    "model_id=transformer|seed=20260819|onset_variant=causal_noise_floor"
    "|pre_roll_ms=30|policy_id=complete-temperature_scaled"
)


@dataclass(frozen=True)
class Check:
    label: str
    quoted: float
    actual: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return abs(self.quoted - self.actual) <= self.tolerance

    def written_forms(self) -> tuple[str, ...]:
        """Every spelling of the quoted value a document may legitimately use."""

        if float(self.quoted).is_integer():
            plain = str(int(self.quoted))
            return (plain, f"{int(self.quoted):,}")
        text = f"{self.quoted}"
        trimmed = text.rstrip("0").rstrip(".") if "." in text else text
        return tuple(dict.fromkeys((text, trimmed)))


def transformer_checks(report_dir: Path) -> list[Check]:
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
    ]


def wavlm_checks(metrics_path: Path) -> list[Check]:
    """The pretrained reference reverses the paper's conclusion, so its row is checked too."""

    metrics = pd.read_csv(metrics_path).set_index("prefix_ms")
    return [
        Check("R1.10 WavLM accuracy at fixed 200 ms", 55.36, float(metrics.loc[200, "accuracy"]) * 100.0, 0.005),
        Check("R1.10 WavLM accuracy at fixed 250 ms", 56.95, float(metrics.loc[250, "accuracy"]) * 100.0, 0.005),
        Check("R1.10 WavLM accuracy at fixed 300 ms", 59.29, float(metrics.loc[300, "accuracy"]) * 100.0, 0.005),
        Check("R1.10 WavLM macro-F1 at fixed 300 ms", 58.10, float(metrics.loc[300, "macro_f1"]) * 100.0, 0.005),
        Check("R1.10 WavLM dysarthric recall at fixed 300 ms", 58.79, float(metrics.loc[300, "dysarthric_recall"]) * 100.0, 0.005),
        Check("R1.10 WavLM balanced accuracy at fixed 300 ms", 59.18, float(metrics.loc[300, "balanced_accuracy"]) * 100.0, 0.005),
    ]


def onset_checks(summary_path: Path) -> list[Check]:
    """These are label-free detector diagnostics, not onset-localization error rates."""

    summary = json.loads(summary_path.read_text())
    return [
        Check("R1.4 causal_fixed sample-zero anchors", 71.7, summary["causal_fixed"]["fires_at_zero_rate"] * 100.0, 0.05),
        Check("R1.4 causal_fixed room-tone prefixes", 43.3, summary["causal_fixed"]["quiet_prefix_rate"] * 100.0, 0.05),
        Check("R1.4 offline-peak sample-zero anchors", 45.0, summary["recovered_offline_peak"]["fires_at_zero_rate"] * 100.0, 0.05),
        Check("R1.4 offline-peak room-tone prefixes", 27.5, summary["recovered_offline_peak"]["quiet_prefix_rate"] * 100.0, 0.05),
    ]


def error_audit_checks(strata_path: Path) -> list[Check]:
    strata = pd.read_csv(strata_path)
    primary = strata[(strata["system_id"] == PRIMARY_SYSTEM) & (strata["error_type"] == "FP")]

    def rate(stratifier: str, stratum: str) -> float:
        row = primary[
            (primary["stratifier"] == stratifier) & (primary["stratum"] == stratum)
        ]
        if len(row) != 1:
            raise SystemExit(f"expected one {stratifier}={stratum} row, found {len(row)}")
        return float(row["error_rate"].iloc[0]) * 100.0

    return [
        Check("R3.10 false-positive rate", 67.89, rate("overall", "all"), 0.005),
        Check("R3.10 false-positive rate, detected onset", 66.25, rate("onset_status", "detected"), 0.005),
        Check("R3.10 false-positive rate, registered detector failure", 91.87, rate("onset_status", "failure"), 0.005),
        Check("R3.10 false-positive rate, sentence prompts", 76.92, rate("prompt_type", "sentence"), 0.005),
        Check("R3.10 false-positive rate, isolated tokens", 64.69, rate("prompt_type", "isolated_token"), 0.005),
        Check("R3.10 false-positive rate, vocal exercises", 63.54, rate("prompt_type", "vocal_exercise"), 0.005),
    ]


def onset_mode_checks(comparison_root: Path) -> list[Check]:
    """The comparator rules classify better; the audit says the gain is a detector artifact."""

    checks = []
    for condition, label, accuracy, difference in (
        ("causal_fixed_preroll30", "causal_fixed", 59.28, 17.33),
        ("offline_peak_preroll30", "recovered_offline_peak", 67.37, 25.42),
    ):
        pooled = pd.read_csv(
            comparison_root / condition / "comparison_vs_primary/pooled_metrics.csv"
        ).set_index("metric")
        column = pooled.columns[0]
        checks.append(
            Check(
                f"R1.4 {label} pooled accuracy",
                accuracy,
                float(pooled.loc["accuracy", column]) * 100.0,
                0.005,
            )
        )
        bootstrap = pd.read_csv(
            comparison_root / condition / "comparison_vs_primary/paired_speaker_bootstrap.csv"
        ).set_index("metric")
        checks.append(
            Check(
                f"R1.4 {label} minus primary accuracy",
                difference,
                float(bootstrap.loc["accuracy", "difference_left_minus_right"]) * 100.0,
                0.005,
            )
        )
    return checks


def confound_checks(audit_dir: Path) -> list[Check]:
    """The class-dependent recording-start rate is what makes the comparator gain an artifact."""

    distribution = pd.read_csv(audit_dir / "anchor_distribution_by_rule.csv").set_index("rule")
    return [
        Check(
            "R1.4 causal_fixed recording-start rate, control",
            88.94,
            float(distribution.loc["causal_fixed", "recording_start_rate_control"]) * 100.0,
            0.005,
        ),
        Check(
            "R1.4 causal_fixed recording-start rate, dysarthric",
            59.78,
            float(distribution.loc["causal_fixed", "recording_start_rate_dysarthric"]) * 100.0,
            0.005,
        ),
        Check(
            "R1.4 offline-peak recording-start rate, control",
            72.73,
            float(distribution.loc["recovered_offline_peak", "recording_start_rate_control"]) * 100.0,
            0.005,
        ),
        Check(
            "R1.4 offline-peak recording-start rate, dysarthric",
            25.09,
            float(distribution.loc["recovered_offline_peak", "recording_start_rate_dysarthric"]) * 100.0,
            0.005,
        ),
    ]


def seed_checks(summary_path: Path) -> list[Check]:
    per_seed = pd.read_csv(summary_path).set_index(["policy_id", "seed"])
    policy = "complete-temperature_scaled"
    return [
        Check("R2.10 seed 20260820 accuracy", 48.41, float(per_seed.loc[(policy, 20260820), "accuracy"]) * 100.0, 0.005),
        Check("R2.10 seed 20260821 accuracy", 46.76, float(per_seed.loc[(policy, 20260821), "accuracy"]) * 100.0, 0.005),
        Check("R2.10 seed 20260820 balanced accuracy", 49.89, float(per_seed.loc[(policy, 20260820), "balanced_accuracy"]) * 100.0, 0.005),
        Check("R2.10 seed 20260821 balanced accuracy", 48.03, float(per_seed.loc[(policy, 20260821), "balanced_accuracy"]) * 100.0, 0.005),
    ]


def build_checks(
    report_dir: Path, wavlm_metrics: Path, onset_summary: Path, error_strata: Path
) -> list[Check]:
    checks = transformer_checks(report_dir)
    for path, builder in (
        (wavlm_metrics, wavlm_checks),
        (onset_summary, onset_checks),
        (error_strata, error_audit_checks),
        (PROJECT_ROOT / "artifacts/robustness/retrained", onset_mode_checks),
        (PROJECT_ROOT / "artifacts/onset_diagnosis/onset_mode_confound", confound_checks),
        (
            PROJECT_ROOT / "artifacts/evaluation/transformer-three-seed/per_seed_policy_metrics.csv",
            seed_checks,
        ),
    ):
        if path.exists():
            checks.extend(builder(path))
        else:
            print(f"[  absent] {path} is missing; its checks are skipped")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/reports/transformer-primary"),
        help="Reports for the headline model quoted in the submission documents.",
    )
    parser.add_argument(
        "--wavlm-metrics",
        type=Path,
        default=Path("artifacts/evaluation/wavlm-primary/fixed_prefix_metrics.csv"),
    )
    parser.add_argument(
        "--onset-summary",
        type=Path,
        default=Path("artifacts/onset_diagnosis/onset_mode_summary.json"),
    )
    parser.add_argument(
        "--error-strata",
        type=Path,
        default=Path("artifacts/error_audit/per_system_error_strata.csv"),
    )
    parser.add_argument(
        "--document",
        type=Path,
        action="append",
        default=None,
        help="Documents that must contain each quoted value; repeatable.",
    )
    arguments = parser.parse_args()
    documents = arguments.document or [
        Path("docs/revised_manuscript.md"),
        Path("docs/reviewer_response.md"),
    ]

    checks = build_checks(
        arguments.report_dir,
        arguments.wavlm_metrics,
        arguments.onset_summary,
        arguments.error_strata,
    )
    prose = "\n".join(path.read_text() for path in documents)

    failures = []
    for check in checks:
        written = any(form in prose for form in check.written_forms())
        marker = "ok" if check.ok and written else "MISMATCH" if not check.ok else "UNQUOTED"
        print(f"[{marker:>8}] {check.label}: quoted {check.quoted}, actual {check.actual:.4f}")
        if not check.ok:
            failures.append(f"{check.label}: quoted {check.quoted}, actual {check.actual:.4f}")
        elif not written:
            failures.append(f"{check.label}: {check.quoted} appears in no document")

    print(f"\n{len(checks) - len(failures)}/{len(checks)} quoted values verified")
    if failures:
        print("\n".join(failures))
        raise SystemExit(f"{len(failures)} quoted value(s) no longer match the artifacts")


if __name__ == "__main__":
    main()
