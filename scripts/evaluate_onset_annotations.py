#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timeentropy.analysis.onset_validation import (
    DEFAULT_BOOTSTRAP_SEED,
    completed_references,
    evaluate_onset_annotations,
    normalize_saved_estimates,
    run_onset_estimators,
    write_evaluation_outputs,
)
from timeentropy.config import project_path
from timeentropy.signal.onset import OnsetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate causal and recovered onset detectors against manual references."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--saved-estimates", type=Path)
    parser.add_argument(
        "--estimates-output",
        type=Path,
        default=Path("artifacts/onset_validation/onset_estimates.csv"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/onset_validation/onset_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/onset_validation/onset_summary.json"),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--hop-ms", type=int, default=10)
    parser.add_argument("--absolute-rms-floor", type=float, default=0.003)
    parser.add_argument("--consecutive-active-frames", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = pd.read_csv(project_path(args.annotations), keep_default_na=False)
    references, audit = completed_references(annotations)
    if args.saved_estimates is None:
        estimates = run_onset_estimators(
            references,
            onset_config=OnsetConfig(
                sample_rate=args.sample_rate,
                frame_ms=args.frame_ms,
                hop_ms=args.hop_ms,
                absolute_rms_floor=args.absolute_rms_floor,
                consecutive_active_frames=args.consecutive_active_frames,
            ),
        )
    else:
        estimates = normalize_saved_estimates(pd.read_csv(project_path(args.saved_estimates)))
    estimates_path = project_path(args.estimates_output)
    estimates_path.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(estimates_path, index=False)

    evaluation = evaluate_onset_annotations(
        annotations,
        estimates,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    metrics_path, summary_path = write_evaluation_outputs(
        evaluation,
        project_path(args.metrics_output),
        project_path(args.summary_output),
    )
    print(
        f"References: {audit.completed_rows} completed, {audit.incomplete_rows} incomplete, "
        f"{audit.excluded_rows} excluded, {audit.invalid_rows} invalid"
    )
    print(f"Wrote estimates to {estimates_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote JSON summary to {summary_path}")


if __name__ == "__main__":
    main()
