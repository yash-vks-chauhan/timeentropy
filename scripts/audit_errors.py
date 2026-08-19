#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from timeentropy.analysis.error_audit import (
    DEFAULT_DURATION_CUTPOINTS_SEC,
    DEFAULT_ENTROPY_CUTPOINTS,
    DEFAULT_REVIEW_PER_ERROR,
    DEFAULT_REVIEW_SEED,
    audit_classification_errors,
    write_error_audit,
)
from timeentropy.config import project_path


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    raise ValueError(f"Unsupported table format for {path}; use CSV or Parquet")


def _cutpoints(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Cutpoints must be comma-separated numbers") from error
    if not result:
        raise argparse.ArgumentTypeError("At least one cutpoint is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit false positives and false negatives from validated calibrated prefix and "
            "decision ledgers."
        )
    )
    parser.add_argument("--prefix-ledger", required=True, type=Path)
    parser.add_argument("--decision-ledger", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--onset-table", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/error_audit"))
    parser.add_argument(
        "--primary-system-id",
        action="append",
        help=(
            "Exact system_id to sample for blinded review; repeat as needed. By default, all "
            "complete-temperature_scaled systems are primary (or all systems if absent)."
        ),
    )
    parser.add_argument("--review-per-error", type=int, default=DEFAULT_REVIEW_PER_ERROR)
    parser.add_argument("--review-seed", type=int, default=DEFAULT_REVIEW_SEED)
    parser.add_argument(
        "--duration-cutpoints-sec",
        type=_cutpoints,
        default=DEFAULT_DURATION_CUTPOINTS_SEC,
        help="Comma-separated deterministic duration cutpoints (default: 0.5,1,2,5).",
    )
    parser.add_argument(
        "--entropy-cutpoints",
        type=_cutpoints,
        default=DEFAULT_ENTROPY_CUTPOINTS,
        help="Comma-separated normalized-entropy cutpoints (default: 0.2,0.4,0.6,0.8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix_path = project_path(args.prefix_ledger)
    decision_path = project_path(args.decision_ledger)
    manifest_path = project_path(args.manifest)
    onset_path = project_path(args.onset_table) if args.onset_table is not None else None

    audit = audit_classification_errors(
        _read_table(prefix_path),
        _read_table(decision_path),
        _read_table(manifest_path),
        onset_table=_read_table(onset_path) if onset_path is not None else None,
        primary_system_ids=args.primary_system_id,
        review_per_error=args.review_per_error,
        review_seed=args.review_seed,
        duration_cutpoints_sec=args.duration_cutpoints_sec,
        entropy_cutpoints=args.entropy_cutpoints,
    )
    input_paths = {
        "prefix_ledger": prefix_path,
        "decision_ledger": decision_path,
        "frozen_manifest": manifest_path,
    }
    if onset_path is not None:
        input_paths["onset_table"] = onset_path
    paths = write_error_audit(
        audit,
        project_path(args.output_dir),
        input_paths=input_paths,
        project_root=project_path(Path(".")),
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
