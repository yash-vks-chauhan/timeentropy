#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from timeentropy.robustness.config import (
    load_robustness_protocol,
    write_retraining_configs,
)
from timeentropy.robustness.retrained import evaluate_retrained_ablation_ledgers
from timeentropy.robustness.summary import summarize_test_time_robustness
from timeentropy.robustness.test_time import run_test_time_robustness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan and evaluate the frozen robustness protocol. This CLI never launches "
            "training; condition-matched training remains an explicit run_fold.py step."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/robustness_v1.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "plan", help="Validate and export the bounded fold-by-condition execution plan."
    )
    plan.add_argument("--output-dir", type=Path)

    prepare = commands.add_parser(
        "prepare-retraining",
        help="Write condition-matched configs for ablations that require new training.",
    )
    prepare.add_argument("--output-dir", type=Path, required=True)

    test_time = commands.add_parser(
        "test-time",
        help="Apply frozen noise/onset perturbations to one primary fold checkpoint.",
    )
    test_time.add_argument("--fold", type=int, required=True)
    test_time.add_argument("--checkpoint", type=Path, required=True)
    test_time.add_argument("--reference-ledger", type=Path, required=True)
    test_time.add_argument("--output-dir", type=Path, required=True)
    test_time.add_argument(
        "--analysis-id",
        action="append",
        help="Optional frozen test-time analysis ID; repeat to run a subset.",
    )
    test_time.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    test_time.add_argument("--batch-size", type=int)
    test_time.add_argument("--num-workers", type=int, default=0)

    retrained = commands.add_parser(
        "evaluate-retrained",
        help=(
            "Verify and evaluate five condition-matched fold ledgers after explicit "
            "retraining has completed."
        ),
    )
    retrained.add_argument("--analysis-id", required=True)
    retrained.add_argument("--prefix-ledger", action="append", type=Path, required=True)
    retrained.add_argument("--run-summary", action="append", type=Path, required=True)
    retrained.add_argument("--output-dir", type=Path, required=True)

    summarize = commands.add_parser(
        "summarize-test-time",
        help="Combine all fold decisions into distinct pooled and fold-level estimands.",
    )
    summarize.add_argument("--decision-ledger", action="append", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument(
        "--allow-subset",
        action="store_true",
        help="Permit a condition subset for debugging; final reports must omit this flag.",
    )
    return parser.parse_args()


def _plan(args: argparse.Namespace) -> None:
    protocol = load_robustness_protocol(args.config)
    rows = protocol.plan_rows()
    summary = {
        "schema_version": 1,
        "robustness_config": str(protocol.config_path),
        "robustness_config_sha256": protocol.config_sha256,
        "parent_config": str(protocol.parent_config_path),
        "parent_config_sha256": protocol.parent_config_sha256,
        "model_id": protocol.model_id,
        "seed": protocol.seed,
        "folds": list(protocol.folds),
        "retrained_input_conditions": len(protocol.retrained_analyses),
        "test_time_conditions": len(protocol.test_time_analyses),
        "fold_condition_jobs": len(rows),
        "test_set_tuning_permitted": False,
    }
    if args.output_dir is not None:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        plan_path = output / "robustness_execution_plan.csv"
        summary_path = output / "robustness_plan_summary.json"
        pd.DataFrame(rows).to_csv(plan_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        summary["outputs"] = {"plan": str(plan_path), "summary": str(summary_path)}
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        _plan(args)
        return
    protocol = load_robustness_protocol(args.config)
    if args.command == "prepare-retraining":
        paths = write_retraining_configs(protocol, args.output_dir)
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return
    if args.command == "test-time":
        result = run_test_time_robustness(
            args.config,
            fold=args.fold,
            checkpoint_path=args.checkpoint,
            reference_ledger_path=args.reference_ledger,
            output_directory=args.output_dir,
            analysis_ids=args.analysis_id,
            device_preference=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(
            json.dumps(
                {
                    "prefix_ledger": str(result.prefix_ledger_path),
                    "decision_ledger": str(result.decision_ledger_path),
                    "metrics": str(result.metrics_path),
                    "policy": str(result.policy_path),
                    "summary": str(result.summary_path),
                },
                indent=2,
            )
        )
        return
    if args.command == "evaluate-retrained":
        result = evaluate_retrained_ablation_ledgers(
            args.config,
            analysis_id=args.analysis_id,
            prefix_ledger_paths=args.prefix_ledger,
            run_summary_paths=args.run_summary,
            output_directory=args.output_dir,
        )
        outputs = {name: str(path) for name, path in result.evaluation_paths.items()}
        outputs["fixed_metrics"] = str(result.fixed_metrics_path)
        outputs["robustness_summary"] = str(result.robustness_summary_path)
        print(json.dumps(outputs, indent=2))
        return
    if args.command == "summarize-test-time":
        result = summarize_test_time_robustness(
            args.config,
            decision_ledger_paths=args.decision_ledger,
            output_directory=args.output_dir,
            require_all_conditions=not args.allow_subset,
        )
        print(
            json.dumps(
                {
                    "pooled_metrics": str(result.pooled_metrics_path),
                    "fold_metrics": str(result.fold_metrics_path),
                    "fold_mean_sd": str(result.fold_mean_sd_path),
                    "summary": str(result.summary_path),
                },
                indent=2,
            )
        )
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
