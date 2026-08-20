from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from timeentropy.config import PROJECT_ROOT
from timeentropy.evaluation.ledger import validate_oof_test_coverage
from timeentropy.provenance import environment_record, sha256_file, sha256_json
from timeentropy.reporting.bootstrap import clustered_speaker_bootstrap_ci
from timeentropy.reporting.comparison import (
    paired_model_comparisons,
    paired_policy_comparisons,
)
from timeentropy.reporting.summaries import (
    build_calibration_tables,
    build_summary_tables,
    enrich_decision_ledger,
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"Unsupported ledger format for {path}; use CSV, TSV, or Parquet")


def _stable_system_seed(base_seed: int, system_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{system_id}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _write_table(frame: pd.DataFrame, output_directory: Path, stem: str) -> None:
    frame.to_csv(output_directory / f"{stem}.csv", index=False)
    frame.to_json(output_directory / f"{stem}.json", orient="records", indent=2)


def generate_report_bundle(
    prefix_ledger_path: str | Path,
    decision_ledger_path: str | Path,
    output_directory: str | Path,
    *,
    config_path: str | Path | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_819,
    confidence: float = 0.95,
    calibration_bins: int = 15,
    include_model_comparisons: bool = True,
) -> dict[str, Any]:
    """Validate OOF ledgers and write the complete reviewer-auditable report bundle."""

    prefix_path = Path(prefix_ledger_path).resolve()
    decision_path = Path(decision_ledger_path).resolve()
    output_path = Path(output_directory).resolve()
    configuration_path = Path(config_path).resolve() if config_path is not None else None
    for path in (prefix_path, decision_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if configuration_path is not None and not configuration_path.is_file():
        raise FileNotFoundError(configuration_path)

    prefix = _read_table(prefix_path)
    decisions = _read_table(decision_path)
    enriched = enrich_decision_ledger(prefix, decisions)
    expected_samples = prefix.loc[prefix["split_role"] == "test", "sample_id"].unique()
    validate_oof_test_coverage(prefix, expected_samples)

    summaries = build_summary_tables(enriched)
    calibration = build_calibration_tables(prefix, bins=calibration_bins)
    bootstrap_rows: list[pd.DataFrame] = []
    for system_id, group in enriched.groupby("system_id", sort=True):
        system_seed = _stable_system_seed(bootstrap_seed, str(system_id))
        intervals = clustered_speaker_bootstrap_ci(
            group,
            replicates=bootstrap_replicates,
            seed=system_seed,
            confidence=confidence,
        ).intervals
        intervals.insert(0, "system_id", system_id)
        bootstrap_rows.append(intervals)
    bootstrap_frame = pd.concat(bootstrap_rows, ignore_index=True)

    if include_model_comparisons:
        comparison_frame, mcnemar_frame = paired_model_comparisons(
            enriched,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence=confidence,
        )
    else:
        comparison_frame, mcnemar_frame = pd.DataFrame(), pd.DataFrame()

    # Policy-versus-baseline testing is independent of how many models are present: it is the
    # comparison the adaptive policy is actually sold on, so it always runs.
    policy_frame, policy_mcnemar_frame = paired_policy_comparisons(
        enriched,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    tables = {
        **asdict(summaries),
        "bootstrap_confidence_intervals": bootstrap_frame,
        "calibration_metrics": calibration.metrics,
        "calibration_reliability_bins": calibration.reliability_bins,
        "paired_model_comparisons": comparison_frame,
        "mcnemar_sensitivity": mcnemar_frame,
        "paired_policy_comparisons": policy_frame,
        "policy_mcnemar_sensitivity": policy_mcnemar_frame,
    }
    for stem, frame in tables.items():
        _write_table(frame, output_path, stem)

    report_settings = {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence": confidence,
        "calibration_bins": calibration_bins,
        "include_model_comparisons": include_model_comparisons,
        "fold_summary_estimand": "unweighted_fold_mean_and_sample_sd",
        "pooled_estimand": "pooled_out_of_fold_utterances",
    }
    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "prefix_ledger": str(prefix_path),
        "prefix_ledger_sha256": sha256_file(prefix_path),
        "decision_ledger": str(decision_path),
        "decision_ledger_sha256": sha256_file(decision_path),
        "config": str(configuration_path) if configuration_path else None,
        "config_sha256": sha256_file(configuration_path) if configuration_path else None,
        "report_settings": report_settings,
        "report_settings_sha256": sha256_json(report_settings),
        "environment": environment_record(PROJECT_ROOT),
        "output_tables": sorted(tables),
    }
    with (output_path / "report_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate auditable statistics from validated OOF prediction ledgers."
    )
    parser.add_argument("--prefix-ledger", required=True, type=Path)
    parser.add_argument("--decision-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/rebuild_v1.yaml")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_819)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--no-model-comparisons", action="store_true")
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    manifest = generate_report_bundle(
        arguments.prefix_ledger,
        arguments.decision_ledger,
        arguments.output_dir,
        config_path=arguments.config,
        bootstrap_replicates=arguments.bootstrap_replicates,
        bootstrap_seed=arguments.bootstrap_seed,
        confidence=arguments.confidence,
        calibration_bins=arguments.calibration_bins,
        include_model_comparisons=not arguments.no_model_comparisons,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
