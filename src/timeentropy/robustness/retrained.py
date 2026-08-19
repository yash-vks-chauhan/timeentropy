from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from timeentropy.config import project_path
from timeentropy.evaluation.pipeline import (
    combine_prefix_ledgers,
    evaluate_prefix_ledgers,
    fixed_prefix_metrics,
    write_evaluation_artifacts,
)
from timeentropy.provenance import sha256_file, sha256_json
from timeentropy.robustness.config import (
    RobustnessProtocol,
    derived_retraining_config,
    load_robustness_protocol,
)
from timeentropy.robustness.test_time import inclusive_grid


@dataclass(frozen=True)
class RetrainedRobustnessResult:
    output_directory: Path
    evaluation_paths: dict[str, Path]
    fixed_metrics_path: Path
    robustness_summary_path: Path


def _load_run_summaries(
    summary_paths: Sequence[str | Path],
) -> dict[int, tuple[Path, dict[str, Any]]]:
    by_fold: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in summary_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError(f"Unsupported run summary: {path}")
        fold = int(value.get("fold", -1))
        if fold in by_fold:
            raise ValueError(f"Duplicate run summaries for fold {fold}")
        by_fold[fold] = (path, value)
    return by_fold


def _validate_condition_ledgers(
    ledger: pd.DataFrame,
    *,
    protocol: RobustnessProtocol,
    analysis_id: str,
) -> None:
    analysis = protocol.analysis(analysis_id)
    expected_constants = {
        "model_id": protocol.model_id,
        "seed": protocol.seed,
        "onset_variant": analysis.onset_mode,
        "pre_roll_ms": analysis.pre_roll_ms,
    }
    for column, expected in expected_constants.items():
        if not ledger[column].eq(expected).all():
            raise ValueError(
                f"Retrained analysis {analysis_id} has unexpected {column}; expected {expected!r}"
            )
    observed_folds = frozenset(ledger["fold"].astype(int))
    if observed_folds != frozenset(protocol.folds):
        raise ValueError(
            f"Retrained ledgers cover folds {sorted(observed_folds)}, "
            f"expected {sorted(protocol.folds)}"
        )
    run_fold_counts = ledger.groupby("run_id")["fold"].nunique()
    if not run_fold_counts.eq(1).all():
        raise ValueError("A retrained robustness run_id spans multiple folds")
    fold_run_counts = ledger.groupby("fold")["run_id"].nunique()
    if not fold_run_counts.eq(1).all():
        raise ValueError("Each robustness fold must be represented by exactly one run_id")


def _validate_run_summary_provenance(
    ledger: pd.DataFrame,
    summaries: dict[int, tuple[Path, dict[str, Any]]],
    *,
    protocol: RobustnessProtocol,
    expected_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if frozenset(summaries) != frozenset(protocol.folds):
        raise ValueError("Run summaries must cover every frozen robustness fold exactly once")
    expected_config_hash = sha256_json(expected_config)
    records: list[dict[str, Any]] = []
    for fold in protocol.folds:
        summary_path, summary = summaries[fold]
        run_ids = set(ledger.loc[ledger["fold"].astype(int).eq(fold), "run_id"].astype(str))
        if run_ids != {str(summary.get("run_id"))}:
            raise ValueError(f"Fold {fold} ledger run_id does not match its run summary")
        if summary.get("model_id") != protocol.model_id:
            raise ValueError(f"Fold {fold} run summary has the wrong model_id")
        if int(summary.get("seed", -1)) != protocol.seed:
            raise ValueError(f"Fold {fold} run summary has the wrong seed")
        hashes = summary.get("hashes")
        if not isinstance(hashes, dict):
            raise TypeError(f"Fold {fold} run summary lacks provenance hashes")
        if hashes.get("experiment_config_sha256") != expected_config_hash:
            raise ValueError(
                f"Fold {fold} was not trained from the frozen condition-matched config"
            )
        checkpoint_path = project_path(str(summary.get("checkpoint_path", "")))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} condition-matched checkpoint is unavailable: {checkpoint_path}"
            )
        records.append(
            {
                "fold": fold,
                "run_id": summary["run_id"],
                "run_summary_path": str(summary_path),
                "run_summary_sha256": sha256_file(summary_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "experiment_config_sha256": hashes["experiment_config_sha256"],
            }
        )
    return records


def evaluate_retrained_ablation_ledgers(
    robustness_config_path: str | Path,
    *,
    analysis_id: str,
    prefix_ledger_paths: Sequence[str | Path],
    run_summary_paths: Sequence[str | Path],
    output_directory: str | Path,
) -> RetrainedRobustnessResult:
    """Evaluate a condition-matched ablation after all frozen fold runs exist.

    This function never trains a model. It refuses evaluation-only perturbation IDs,
    requires one ledger and run summary per frozen fold, verifies the exact training
    config hash, and delegates calibration/policy selection to the standard validation-
    partition pipeline.
    """

    protocol = load_robustness_protocol(robustness_config_path)
    analysis = protocol.analysis(analysis_id)
    if analysis.family != "retrained_input":
        raise ValueError(
            f"{analysis_id} is a test-time perturbation and must reuse the primary checkpoint"
        )
    if len(prefix_ledger_paths) != len(protocol.folds):
        raise ValueError("Exactly one prefix ledger per frozen fold is required")
    if len(run_summary_paths) != len(protocol.folds):
        raise ValueError("Exactly one run summary per frozen fold is required")
    expected_config = derived_retraining_config(protocol, analysis_id)
    ledger = combine_prefix_ledgers(prefix_ledger_paths)
    _validate_condition_ledgers(
        ledger,
        protocol=protocol,
        analysis_id=analysis_id,
    )
    summaries = _load_run_summaries(run_summary_paths)
    run_records = _validate_run_summary_provenance(
        ledger,
        summaries,
        protocol=protocol,
        expected_config=expected_config,
    )

    manifest = pd.read_csv(
        project_path(protocol.parent_config["data"]["manifest"]),
        dtype={"sample_id": str},
    )
    decision_config = protocol.parent_config["decision_policy"]
    theta = decision_config["theta_grid"]
    epsilon = decision_config["epsilon_grid"]
    artifacts = evaluate_prefix_ledgers(
        ledger,
        expected_oof_sample_ids=manifest["sample_id"].astype(str),
        bins=int(protocol.parent_config["calibration"]["ece_bins"]),
        policy_variants=("complete",),
        theta_grid=inclusive_grid(
            float(theta["start"]), float(theta["stop"]), float(theta["step"])
        ),
        epsilon_grid=inclusive_grid(
            float(epsilon["start"]),
            float(epsilon["stop"]),
            float(epsilon["step"]),
        ),
        macro_f1_margin=float(decision_config["selection"]["macro_f1_margin_absolute"]),
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    evaluation_paths = write_evaluation_artifacts(
        artifacts,
        output,
        input_paths=prefix_ledger_paths,
        config=expected_config,
    )
    fixed_metrics_path = output / "fixed_prefix_metrics.csv"
    fixed_prefix_metrics(artifacts.fixed_predictions).to_csv(fixed_metrics_path, index=False)
    summary = {
        "schema_version": 1,
        "analysis": asdict(analysis),
        "execution_stage": "condition_matched_retrained_evaluation",
        "requires_condition_matched_checkpoint": True,
        "test_set_tuning_permitted": False,
        "temperature_fit_split": "calibration",
        "policy_selection_split": "policy_tuning",
        "test_rows_used_for_tuning": 0,
        "robustness_config": {
            "path": str(protocol.config_path),
            "sha256": protocol.config_sha256,
        },
        "condition_training_config_sha256": sha256_json(expected_config),
        "input_ledgers": [
            {"path": str(Path(path)), "sha256": sha256_file(path)} for path in prefix_ledger_paths
        ],
        "verified_runs": run_records,
        "outputs": {name: str(path) for name, path in evaluation_paths.items()},
    }
    summary_path = output / "robustness_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return RetrainedRobustnessResult(
        output_directory=output,
        evaluation_paths=evaluation_paths,
        fixed_metrics_path=fixed_metrics_path,
        robustness_summary_path=summary_path,
    )
