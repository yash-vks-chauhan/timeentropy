#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from timeentropy.orchestration import (
    build_run_matrix,
    dependency_commit_drift,
    dependency_errors,
    execute_jobs,
    plan_record,
    run_preflight,
    select_jobs,
    verify_job_completion,
    write_plan,
)
from timeentropy.robustness.config import load_robustness_protocol


def _add_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/rebuild_v1.yaml"))
    parser.add_argument(
        "--robustness-config", type=Path, default=Path("configs/robustness_v1.yaml")
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--robustness-output-root", type=Path, default=Path("artifacts/robustness")
    )
    parser.add_argument(
        "--derived-config-root",
        type=Path,
        default=Path("artifacts/orchestration/retraining-configs"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-wavlm-download", action="store_true")
    parser.add_argument("--wavlm-batch-size", type=int)


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--fold", action="append", type=int, default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--analysis-id", action="append", default=[])
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument(
        "--with-dependencies",
        action="store_true",
        help="Also select the prerequisite jobs of every filtered job.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, check, and sequentially execute the frozen experiment matrix. Every "
            "stage is delegated to the existing per-stage CLIs; this runner only "
            "orders work, verifies artifacts, and refuses unsafe execution."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "plan", help="Expand the frozen matrix and export the job list without running it."
    )
    plan.add_argument("--output-dir", type=Path)
    plan.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Export the matrix without inspecting data, environment, or Git state.",
    )
    plan.add_argument("--git-policy", choices=("clean", "commit"), default="clean")

    preflight = commands.add_parser(
        "preflight", help="Check every prerequisite for the selected jobs and exit non-zero if unmet."
    )
    preflight.add_argument("--git-policy", choices=("clean", "commit"), default="clean")
    preflight.add_argument("--min-free-gib", type=float, default=10.0)

    status = commands.add_parser(
        "status", help="Classify each selected job as pending, complete, or invalid."
    )
    status.add_argument("--expected-git-commit")

    run = commands.add_parser(
        "run", help="Execute the selected jobs one at a time, verifying each before continuing."
    )
    run.add_argument("--git-policy", choices=("clean", "commit"), default="clean")
    run.add_argument("--min-free-gib", type=float, default=10.0)
    run.add_argument("--output-dir", type=Path, help="Optional directory for the execution plan.")
    run.add_argument(
        "--rerun-invalid",
        action="store_true",
        help="Permit overwriting a run whose existing artifacts fail verification.",
    )
    run.add_argument(
        "--force-complete",
        action="store_true",
        help="Re-execute jobs that already verify as complete.",
    )
    run.add_argument(
        "--allow-dependency-commit-drift",
        action="store_true",
        help=(
            "Consume dependencies produced at a different commit. Every content invariant is "
            "still enforced; the differing commits are printed and written into the plan."
        ),
    )

    for subparser in (plan, preflight, status, run):
        _add_matrix_arguments(subparser)
        _add_selection_arguments(subparser)
    return parser.parse_args()


def _matrix(args: argparse.Namespace):
    return build_run_matrix(
        args.config,
        args.robustness_config,
        output_root=args.output_root,
        robustness_output_root=args.robustness_output_root,
        derived_config_root=args.derived_config_root,
        device=args.device,
        num_workers=args.num_workers,
        allow_wavlm_download=args.allow_wavlm_download,
        wavlm_batch_size=args.wavlm_batch_size,
    )


def _selected(args: argparse.Namespace, matrix):
    return select_jobs(
        matrix,
        stages=args.stage,
        models=args.model,
        folds=args.fold,
        seeds=args.seed,
        analysis_ids=args.analysis_id,
        run_ids=args.run_id,
        with_dependencies=args.with_dependencies,
    )


def _print_preflight(report) -> None:
    for check in report.checks:
        marker = "ok" if check.ok else check.severity
        print(f"[{marker:>7}] {check.check_id}: {check.message}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    matrix = _matrix(args)
    selected = _selected(args, matrix)

    if args.command == "status":
        protocol = load_robustness_protocol(matrix.robustness_config_path)
        statuses = [
            verify_job_completion(
                job,
                expected_git_commit=args.expected_git_commit,
                robustness_protocol=protocol,
            )
            for job in selected
        ]
        counts: dict[str, int] = {}
        for item in statuses:
            counts[item.state] = counts.get(item.state, 0) + 1
        for item in statuses:
            print(f"[{item.state:>8}] {item.run_id}: {item.reason}")
        print(json.dumps({"selected_jobs": len(selected), "states": counts}, sort_keys=True))
        return

    if args.command == "plan":
        report = None
        if not args.skip_preflight:
            report, _ = run_preflight(
                matrix, selected, git_policy=args.git_policy,
                allow_wavlm_download=args.allow_wavlm_download,
            )
            _print_preflight(report)
        record = plan_record(matrix, selected, preflight=report)
        outputs = {}
        if args.output_dir is not None:
            json_path, csv_path = write_plan(record, args.output_dir)
            outputs = {"plan": str(json_path), "jobs": str(csv_path)}
        print(
            json.dumps(
                {
                    "matrix_sha256": matrix.matrix_sha256,
                    "total_jobs": len(matrix.jobs),
                    "selected_jobs": len(selected),
                    "preflight_ready": None if report is None else report.ready,
                    "outputs": outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    report, commit = run_preflight(
        matrix,
        selected,
        git_policy=args.git_policy,
        allow_wavlm_download=args.allow_wavlm_download,
        min_free_gib=args.min_free_gib,
    )
    _print_preflight(report)

    if args.command == "preflight":
        print(
            json.dumps(
                {
                    "ready": report.ready,
                    "selected_jobs": len(selected),
                    "git_commit": commit,
                    "unmet": [check.check_id for check in report.errors],
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(0 if report.ready else 1)

    if not report.ready:
        unmet = ", ".join(check.check_id for check in report.errors)
        raise SystemExit(f"Refusing to execute: unmet prerequisites ({unmet}).")
    if commit is None:
        raise SystemExit("Refusing to execute: no Git commit is available to stamp artifacts.")

    unmet_dependencies = dependency_errors(
        selected,
        matrix,
        expected_git_commit=commit,
        allow_commit_drift=args.allow_dependency_commit_drift,
    )
    if unmet_dependencies:
        raise SystemExit(
            "Refusing to execute: unmet dependencies.\n  " + "\n  ".join(unmet_dependencies)
        )

    drift = dependency_commit_drift(selected, matrix, expected_git_commit=commit)
    if drift and not args.allow_dependency_commit_drift:
        raise SystemExit(
            "Refusing to execute: dependencies were produced at other commits.\n  "
            + "\n  ".join(f"{run_id} produced at {value}" for run_id, value in drift.items())
            + "\n Pass --allow-dependency-commit-drift to consume them; the difference will "
            "be recorded."
        )
    for run_id, value in drift.items():
        print(f"[ drift ] {run_id} was produced at {value}, not {commit}", file=sys.stderr)

    statuses = execute_jobs(
        matrix,
        selected,
        expected_git_commit=commit,
        rerun_invalid=args.rerun_invalid,
        force_complete=args.force_complete,
    )
    if args.output_dir is not None:
        write_plan(plan_record(matrix, selected, statuses=statuses, preflight=report), args.output_dir)
    print(
        json.dumps(
            {
                "git_commit": commit,
                "dependency_commit_drift": drift,
                "executed_jobs": len(statuses),
                "states": {
                    state: sum(1 for item in statuses if item.state == state)
                    for state in sorted({item.state for item in statuses})
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
