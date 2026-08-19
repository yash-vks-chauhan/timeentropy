#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from timeentropy.config import load_yaml, project_path
from timeentropy.evaluation.pipeline import (
    combine_prefix_ledgers,
    evaluate_prefix_ledgers,
    fixed_prefix_metrics,
    write_evaluation_artifacts,
)


def _inclusive_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    if step <= 0.0 or stop < start:
        raise ValueError("Invalid grid bounds")
    count = round((stop - start) / step)
    values = tuple(round(start + index * step, 10) for index in range(count + 1))
    if abs(values[-1] - stop) > 1.0e-8:
        raise ValueError("Grid stop must be reachable by an integer number of steps")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit fold temperatures, select validation-only entropy policies, and export "
            "canonical test decisions from raw prefix ledgers."
        )
    )
    parser.add_argument(
        "--prefix-ledger",
        action="append",
        required=True,
        type=Path,
        help="One fold ledger; repeat exactly once per fold/run being combined.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rebuild_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    ledger = combine_prefix_ledgers(args.prefix_ledger)
    manifest = pd.read_csv(project_path(config["data"]["manifest"]), dtype={"sample_id": str})
    theta = config["decision_policy"]["theta_grid"]
    epsilon = config["decision_policy"]["epsilon_grid"]
    artifacts = evaluate_prefix_ledgers(
        ledger,
        expected_oof_sample_ids=manifest["sample_id"].astype(str),
        bins=int(config["calibration"]["ece_bins"]),
        theta_grid=_inclusive_grid(
            float(theta["start"]), float(theta["stop"]), float(theta["step"])
        ),
        epsilon_grid=_inclusive_grid(
            float(epsilon["start"]), float(epsilon["stop"]), float(epsilon["step"])
        ),
        macro_f1_margin=float(
            config["decision_policy"]["selection"]["macro_f1_margin_absolute"]
        ),
    )
    paths = write_evaluation_artifacts(
        artifacts,
        args.output_dir,
        input_paths=args.prefix_ledger,
        config=config,
    )
    fixed_metrics_path = Path(args.output_dir) / "fixed_prefix_metrics.csv"
    fixed_prefix_metrics(artifacts.fixed_predictions).to_csv(fixed_metrics_path, index=False)
    paths["fixed_prefix_metrics"] = fixed_metrics_path
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
