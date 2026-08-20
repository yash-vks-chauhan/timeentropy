"""Frozen, sequential, artifact-verified experiment orchestration.

This module plans commands around the repository's existing CLIs.  It does not
implement training or robustness inference itself and never launches work from
an import side effect.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from timeentropy.config import PROJECT_ROOT, load_yaml, project_path
from timeentropy.evaluation.ledger import validate_prefix_ledger
from timeentropy.provenance import sha256_file, sha256_json
from timeentropy.robustness.config import (
    RobustnessProtocol,
    derived_retraining_config,
    load_robustness_protocol,
    write_retraining_configs,
)
from timeentropy.training.fold import fold_run_id

Stage = Literal[
    "primary_stft",
    "seed_replication",
    "wavlm_reference",
    "robustness_retrain",
    "robustness_test_time",
]
GitPolicy = Literal["clean", "commit"]
JobState = Literal["pending", "complete", "invalid"]

STAGE_ORDER: tuple[Stage, ...] = (
    "primary_stft",
    "seed_replication",
    "wavlm_reference",
    "robustness_retrain",
    "robustness_test_time",
)
STAGE_RANK = {stage: index for index, stage in enumerate(STAGE_ORDER)}


@dataclass(frozen=True)
class ExperimentJob:
    """One sequentially executable unit with deterministic artifact identity."""

    stage: Stage
    run_id: str
    fold: int
    model_id: str
    seed: int
    analysis_id: str | None
    config_path: Path
    config_content_sha256: str
    output_directory: Path
    summary_path: Path
    checkpoint_path: Path | None
    ledger_path: Path | None
    dependency_run_ids: tuple[str, ...]
    command: tuple[str, ...]

    def identity_record(self) -> dict[str, Any]:
        """Return the stable fields covered by the run-matrix hash."""

        return {
            "stage": self.stage,
            "run_id": self.run_id,
            "fold": self.fold,
            "model_id": self.model_id,
            "seed": self.seed,
            "analysis_id": self.analysis_id,
            "config_path": _portable_path(self.config_path),
            "config_content_sha256": self.config_content_sha256,
            "output_directory": _portable_path(self.output_directory),
            "summary_path": _portable_path(self.summary_path),
            "checkpoint_path": _portable_optional_path(self.checkpoint_path),
            "ledger_path": _portable_optional_path(self.ledger_path),
            "dependency_run_ids": list(self.dependency_run_ids),
            "command": _portable_command(self.command),
        }


@dataclass(frozen=True)
class RunMatrix:
    """The complete frozen execution matrix and its source identities."""

    config_path: Path
    config_file_sha256: str
    config_content_sha256: str
    robustness_config_path: Path
    robustness_config_file_sha256: str
    robustness_config_content_sha256: str
    primary_seed: int
    robustness_seeds: tuple[int, ...]
    folds: tuple[int, ...]
    primary_robustness_analysis_id: str
    primary_robustness_run_ids: tuple[str, ...]
    jobs: tuple[ExperimentJob, ...]
    matrix_sha256: str


@dataclass(frozen=True)
class CompletionStatus:
    run_id: str
    state: JobState
    reason: str


@dataclass(frozen=True)
class PrerequisiteCheck:
    check_id: str
    ok: bool
    severity: Literal["error", "warning", "info"]
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PrerequisiteCheck, ...]

    @property
    def errors(self) -> tuple[PrerequisiteCheck, ...]:
        return tuple(item for item in self.checks if not item.ok and item.severity == "error")

    @property
    def ready(self) -> bool:
        return not self.errors


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _portable_optional_path(path: Path | None) -> str | None:
    return None if path is None else _portable_path(path)


def _portable_command(command: Sequence[str]) -> list[str]:
    values = list(command)
    if values:
        values[0] = "${PYTHON}"
    for index, value in enumerate(values[1:], start=1):
        candidate = Path(value)
        if candidate.is_absolute():
            values[index] = _portable_path(candidate)
    return values


def _resolved(value: str | Path) -> Path:
    return project_path(value).resolve()


def _script(name: str) -> Path:
    path = PROJECT_ROOT / "scripts" / name
    if not path.is_file():
        raise FileNotFoundError(f"Required launcher is absent: {path}")
    return path.resolve()


def _require_frozen_config(config: dict[str, Any], *, name: str) -> None:
    if config.get("schema_version") != 1:
        raise ValueError(f"{name} must use schema_version=1")
    if config.get("status") != "frozen_before_test_evaluation":
        raise ValueError(f"{name} is not marked frozen_before_test_evaluation")


def _fold_ids(config: dict[str, Any]) -> tuple[int, ...]:
    fold_path = _resolved(config["data"]["fold_manifest"])
    frame = pd.read_csv(fold_path, usecols=["fold"])
    folds = tuple(sorted(int(value) for value in frame["fold"].unique()))
    if folds != (0, 1, 2, 3, 4):
        raise ValueError(f"Frozen experiment requires folds (0, 1, 2, 3, 4), found {folds}")
    return folds


def _fold_job(
    *,
    stage: Stage,
    config: dict[str, Any],
    config_path: Path,
    fold: int,
    model_id: str,
    seed: int,
    output_root: Path,
    python_executable: str,
    device: str,
    num_workers: int,
    ssl_reference: bool,
    allow_wavlm_download: bool,
    wavlm_batch_size: int | None,
    analysis_id: str | None = None,
) -> ExperimentJob:
    run_id = fold_run_id(config, fold=fold, model_id=model_id, seed=seed)
    output = output_root / run_id
    script = _script("run_ssl_fold.py" if ssl_reference else "run_fold.py")
    command = [
        python_executable,
        str(script),
        "--config",
        str(config_path),
        "--fold",
        str(fold),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-root",
        str(output_root),
        "--num-workers",
        str(num_workers),
    ]
    if ssl_reference:
        if allow_wavlm_download:
            command.append("--allow-download")
        if wavlm_batch_size is not None:
            command.extend(("--batch-size", str(wavlm_batch_size)))
    else:
        command.extend(("--model", model_id))
    return ExperimentJob(
        stage=stage,
        run_id=run_id,
        fold=fold,
        model_id=model_id,
        seed=seed,
        analysis_id=analysis_id,
        config_path=config_path,
        config_content_sha256=sha256_json(config),
        output_directory=output,
        summary_path=output / "run_summary.json",
        checkpoint_path=output / "best_checkpoint.pt",
        ledger_path=output / "prefix_predictions.parquet",
        dependency_run_ids=(),
        command=tuple(command),
    )


def _test_time_job(
    *,
    protocol: RobustnessProtocol,
    analysis_id: str,
    fold: int,
    reference: ExperimentJob,
    output_root: Path,
    python_executable: str,
    device: str,
    num_workers: int,
) -> ExperimentJob:
    run_id = (
        f"{protocol.parent_config['experiment_name']}-robustness-test-time-"
        f"{analysis_id}-fold{fold}-seed{protocol.seed}"
    )
    output = output_root / "test-time" / analysis_id / f"fold{fold}"
    if reference.checkpoint_path is None or reference.ledger_path is None:
        raise RuntimeError("Primary robustness reference lacks checkpoint or ledger paths")
    command = (
        python_executable,
        str(_script("run_robustness.py")),
        "--config",
        str(protocol.config_path),
        "test-time",
        "--fold",
        str(fold),
        "--checkpoint",
        str(reference.checkpoint_path),
        "--reference-ledger",
        str(reference.ledger_path),
        "--output-dir",
        str(output),
        "--analysis-id",
        analysis_id,
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    )
    return ExperimentJob(
        stage="robustness_test_time",
        run_id=run_id,
        fold=fold,
        model_id=protocol.model_id,
        seed=protocol.seed,
        analysis_id=analysis_id,
        config_path=protocol.config_path,
        config_content_sha256=protocol.config_sha256,
        output_directory=output,
        summary_path=output / "robustness_summary.json",
        checkpoint_path=None,
        ledger_path=output / "robustness_prefix_predictions.parquet",
        dependency_run_ids=(reference.run_id,),
        command=command,
    )


def build_run_matrix(
    config_path: str | Path = "configs/rebuild_v1.yaml",
    robustness_config_path: str | Path = "configs/robustness_v1.yaml",
    *,
    output_root: str | Path = "artifacts/runs",
    robustness_output_root: str | Path = "artifacts/robustness",
    derived_config_root: str | Path = "artifacts/orchestration/retraining-configs",
    # Not resolved: under a virtualenv, sys.executable is a symlink into the base
    # interpreter, and resolving it yields a Python without the project installed.
    python_executable: str = os.fspath(Path(os.sys.executable)),
    device: str = "auto",
    num_workers: int = 0,
    allow_wavlm_download: bool = False,
    wavlm_batch_size: int | None = None,
) -> RunMatrix:
    """Expand and validate every frozen executable unit without launching it."""

    if device not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if wavlm_batch_size is not None and wavlm_batch_size < 1:
        raise ValueError("wavlm_batch_size must be positive")

    resolved_config_path = _resolved(config_path)
    config = load_yaml(resolved_config_path)
    _require_frozen_config(config, name="Experiment config")
    protocol = load_robustness_protocol(robustness_config_path)
    if protocol.parent_config_path != resolved_config_path:
        raise ValueError("Robustness protocol does not reference the selected experiment config")
    folds = _fold_ids(config)
    if tuple(protocol.folds) != folds:
        raise ValueError("Robustness folds differ from the parent five-fold matrix")

    primary_models = tuple(str(value) for value in config["models"]["primary"])
    if not primary_models or len(set(primary_models)) != len(primary_models):
        raise ValueError("models.primary must contain unique model identifiers")
    supported_primary = {"small_cnn", "cnn_bilstm", "transformer"}
    if set(primary_models) != supported_primary:
        raise ValueError(
            f"Primary model set must be {sorted(supported_primary)}, found {sorted(primary_models)}"
        )
    headline_model = str(config["models"]["primary_headline_model"])
    if headline_model not in primary_models or protocol.model_id != headline_model:
        raise ValueError("Headline and robustness model identities must agree")

    primary_seed = int(config["training"]["primary_seed"])
    robustness_seeds = tuple(int(value) for value in config["training"]["robustness_seeds"])
    if (
        not robustness_seeds
        or len(set(robustness_seeds)) != len(robustness_seeds)
        or primary_seed in robustness_seeds
        or any(seed < 0 for seed in robustness_seeds)
    ):
        raise ValueError("training.robustness_seeds must be unique non-negative extra seeds")

    modern_models = tuple(str(value) for value in config["models"]["modern_reference"])
    reference_config = dict(config["models"]["modern_reference_config"])
    if modern_models != (str(reference_config["model_id"]),):
        raise ValueError("Exactly one modern reference matching modern_reference_config is required")
    wavlm_model_id = modern_models[0]

    resolved_output_root = _resolved(output_root)
    resolved_robustness_root = _resolved(robustness_output_root)
    resolved_derived_root = _resolved(derived_config_root)
    jobs: list[ExperimentJob] = []

    for model_id in primary_models:
        for fold in folds:
            jobs.append(
                _fold_job(
                    stage="primary_stft",
                    config=config,
                    config_path=resolved_config_path,
                    fold=fold,
                    model_id=model_id,
                    seed=primary_seed,
                    output_root=resolved_output_root,
                    python_executable=python_executable,
                    device=device,
                    num_workers=num_workers,
                    ssl_reference=False,
                    allow_wavlm_download=False,
                    wavlm_batch_size=None,
                )
            )
    for seed in robustness_seeds:
        for fold in folds:
            jobs.append(
                _fold_job(
                    stage="seed_replication",
                    config=config,
                    config_path=resolved_config_path,
                    fold=fold,
                    model_id=headline_model,
                    seed=seed,
                    output_root=resolved_output_root,
                    python_executable=python_executable,
                    device=device,
                    num_workers=num_workers,
                    ssl_reference=False,
                    allow_wavlm_download=False,
                    wavlm_batch_size=None,
                )
            )
    for fold in folds:
        jobs.append(
            _fold_job(
                stage="wavlm_reference",
                config=config,
                config_path=resolved_config_path,
                fold=fold,
                model_id=wavlm_model_id,
                seed=primary_seed,
                output_root=resolved_output_root,
                python_executable=python_executable,
                device=device,
                num_workers=num_workers,
                ssl_reference=True,
                allow_wavlm_download=allow_wavlm_download,
                wavlm_batch_size=wavlm_batch_size,
            )
        )

    primary_analysis = next(item for item in protocol.retrained_analyses if item.role == "primary")
    primary_references = {
        job.fold: job
        for job in jobs
        if job.stage == "primary_stft"
        and job.model_id == headline_model
        and job.seed == primary_seed
    }
    if set(primary_references) != set(folds):
        raise RuntimeError("Primary headline reference jobs are incomplete")

    for analysis in protocol.retrained_analyses:
        if not analysis.requires_new_training:
            continue
        condition_config = derived_retraining_config(protocol, analysis.analysis_id)
        condition_path = resolved_derived_root / f"{analysis.analysis_id}.yaml"
        for fold in folds:
            jobs.append(
                _fold_job(
                    stage="robustness_retrain",
                    config=condition_config,
                    config_path=condition_path,
                    fold=fold,
                    model_id=headline_model,
                    seed=protocol.seed,
                    output_root=resolved_output_root,
                    python_executable=python_executable,
                    device=device,
                    num_workers=num_workers,
                    ssl_reference=False,
                    allow_wavlm_download=False,
                    wavlm_batch_size=None,
                    analysis_id=analysis.analysis_id,
                )
            )
    for analysis in protocol.test_time_analyses:
        for fold in folds:
            jobs.append(
                _test_time_job(
                    protocol=protocol,
                    analysis_id=analysis.analysis_id,
                    fold=fold,
                    reference=primary_references[fold],
                    output_root=resolved_robustness_root,
                    python_executable=python_executable,
                    device=device,
                    num_workers=num_workers,
                )
            )

    jobs.sort(
        key=lambda item: (
            STAGE_RANK[item.stage],
            item.analysis_id or "",
            item.model_id,
            item.seed,
            item.fold,
        )
    )
    run_ids = [job.run_id for job in jobs]
    if len(set(run_ids)) != len(run_ids):
        duplicates = sorted({value for value in run_ids if run_ids.count(value) > 1})
        raise ValueError(f"Run matrix contains duplicate run IDs: {duplicates}")
    output_paths = [job.output_directory for job in jobs]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Run matrix contains colliding output directories")

    reference_ids = tuple(primary_references[fold].run_id for fold in folds)
    matrix_record = {
        "schema_version": 1,
        "config_file_sha256": sha256_file(resolved_config_path),
        "config_content_sha256": sha256_json(config),
        "robustness_config_file_sha256": protocol.config_file_sha256,
        "robustness_config_content_sha256": protocol.config_sha256,
        "primary_seed": primary_seed,
        "robustness_seeds": list(robustness_seeds),
        "folds": list(folds),
        "primary_robustness_analysis_id": primary_analysis.analysis_id,
        "primary_robustness_run_ids": list(reference_ids),
        "jobs": [job.identity_record() for job in jobs],
    }
    return RunMatrix(
        config_path=resolved_config_path,
        config_file_sha256=matrix_record["config_file_sha256"],
        config_content_sha256=matrix_record["config_content_sha256"],
        robustness_config_path=protocol.config_path,
        robustness_config_file_sha256=protocol.config_file_sha256,
        robustness_config_content_sha256=protocol.config_sha256,
        primary_seed=primary_seed,
        robustness_seeds=robustness_seeds,
        folds=folds,
        primary_robustness_analysis_id=primary_analysis.analysis_id,
        primary_robustness_run_ids=reference_ids,
        jobs=tuple(jobs),
        matrix_sha256=sha256_json(matrix_record),
    )


def select_jobs(
    matrix: RunMatrix,
    *,
    stages: Sequence[str] = (),
    models: Sequence[str] = (),
    folds: Sequence[int] = (),
    seeds: Sequence[int] = (),
    analysis_ids: Sequence[str] = (),
    run_ids: Sequence[str] = (),
    with_dependencies: bool = False,
) -> tuple[ExperimentJob, ...]:
    """Select an intersection of filters and optionally include dependencies."""

    unknown_stages = set(stages) - set(STAGE_ORDER)
    if unknown_stages:
        raise ValueError(f"Unknown stages: {sorted(unknown_stages)}")
    known_ids = {job.run_id for job in matrix.jobs}
    unknown_ids = set(run_ids) - known_ids
    if unknown_ids:
        raise ValueError(f"Unknown run IDs: {sorted(unknown_ids)}")
    stage_set = set(stages)
    model_set = set(models)
    fold_set = set(folds)
    seed_set = set(seeds)
    analysis_set = set(analysis_ids)
    run_set = set(run_ids)
    selected = {
        job.run_id
        for job in matrix.jobs
        if (not stage_set or job.stage in stage_set)
        and (not model_set or job.model_id in model_set)
        and (not fold_set or job.fold in fold_set)
        and (not seed_set or job.seed in seed_set)
        and (not analysis_set or job.analysis_id in analysis_set)
        and (not run_set or job.run_id in run_set)
    }
    if with_dependencies:
        lookup = {job.run_id: job for job in matrix.jobs}
        frontier = list(selected)
        while frontier:
            current = lookup[frontier.pop()]
            for dependency in current.dependency_run_ids:
                if dependency not in selected:
                    selected.add(dependency)
                    frontier.append(dependency)
    return tuple(job for job in matrix.jobs if job.run_id in selected)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _existing_artifacts(job: ExperimentJob) -> tuple[Path, ...]:
    if not job.output_directory.exists():
        return ()
    return tuple(path for path in job.output_directory.rglob("*") if path.is_file())


def _resolve_summary_artifact(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("Summary artifact path is not a non-empty string")
    return _resolved(raw)


def _require_summary_hash(
    hashes: dict[str, Any], key: str, path: Path, errors: list[str]
) -> None:
    expected = hashes.get(key)
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(f"missing {key}")
    elif not path.is_file():
        errors.append(f"missing artifact {path}")
    elif sha256_file(path) != expected:
        errors.append(f"{key} differs from artifact")


def _validate_fold_completion(
    job: ExperimentJob,
    *,
    config: dict[str, Any],
    expected_git_commit: str | None,
) -> list[str]:
    errors: list[str] = []
    try:
        summary = json.loads(job.summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read run summary: {error}"]
    expected_fields = {
        "schema_version": 1,
        "run_id": job.run_id,
        "fold": job.fold,
        "model_id": job.model_id,
        "seed": job.seed,
    }
    for key, expected in expected_fields.items():
        if summary.get(key) != expected:
            errors.append(f"summary {key}={summary.get(key)!r}, expected {expected!r}")
    if job.checkpoint_path is None or job.ledger_path is None:
        return errors + ["fold job lacks canonical checkpoint or ledger path"]
    try:
        summary_checkpoint = _resolve_summary_artifact(summary.get("checkpoint_path"))
        summary_ledger = _resolve_summary_artifact(summary.get("ledger_path"))
    except ValueError as error:
        return errors + [str(error)]
    if summary_checkpoint != job.checkpoint_path.resolve():
        errors.append("summary checkpoint path differs from frozen matrix")
    if summary_ledger != job.ledger_path.resolve():
        errors.append("summary ledger path differs from frozen matrix")
    hashes = summary.get("hashes")
    if not isinstance(hashes, dict):
        return errors + ["summary hashes is not a mapping"]
    expected_hashes = {
        "experiment_config_sha256": job.config_content_sha256,
        "experiment_config_file_sha256": sha256_file(job.config_path),
        "manifest_sha256": sha256_file(_resolved(config["data"]["manifest"])),
        "fold_manifest_sha256": sha256_file(_resolved(config["data"]["fold_manifest"])),
    }
    for key, expected in expected_hashes.items():
        if hashes.get(key) != expected:
            errors.append(f"summary {key} differs from current frozen input")
    _require_summary_hash(hashes, "checkpoint_file_sha256", job.checkpoint_path, errors)
    _require_summary_hash(hashes, "prefix_ledger_sha256", job.ledger_path, errors)
    if errors:
        return errors

    try:
        ledger = _read_table(job.ledger_path)
        validate_prefix_ledger(ledger)
    except Exception as error:  # noqa: BLE001  # Validation reports the underlying invariant.
        return [f"prefix ledger validation failed: {error}"]
    constants = {
        "run_id": job.run_id,
        "fold": job.fold,
        "model_id": job.model_id,
        "seed": job.seed,
        "onset_variant": config["onset"]["primary_mode"],
        "pre_roll_ms": int(config["onset"]["pre_roll_ms"]),
    }
    for column, expected in constants.items():
        if column not in ledger or set(ledger[column].unique()) != {expected}:
            errors.append(f"ledger {column} does not equal {expected!r}")

    fold_manifest = pd.read_csv(_resolved(config["data"]["fold_manifest"]))
    membership = fold_manifest.loc[fold_manifest["fold"].eq(job.fold), ["sample_id", "role"]]
    expected_membership = membership.loc[
        membership["role"].isin(["validation", "test"])
    ].copy()
    observed_ids = set(ledger["sample_id"].astype(str))
    expected_ids = set(expected_membership["sample_id"].astype(str))
    if observed_ids != expected_ids:
        errors.append(
            f"ledger sample coverage differs: missing={len(expected_ids - observed_ids)}, "
            f"extra={len(observed_ids - expected_ids)}"
        )
    observed_test = set(ledger.loc[ledger["split_role"].eq("test"), "sample_id"].astype(str))
    expected_test = set(
        expected_membership.loc[
            expected_membership["role"].eq("test"), "sample_id"
        ].astype(str)
    )
    if observed_test != expected_test:
        errors.append("ledger test membership differs from the frozen fold manifest")

    try:
        import torch

        checkpoint = torch.load(
            job.checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as error:  # noqa: BLE001
        errors.append(f"checkpoint validation failed: {error}")
        return errors
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        errors.append("checkpoint schema is invalid")
        return errors
    metadata_record = checkpoint.get("metadata", {})
    checkpoint_hashes = checkpoint.get("hashes", {})
    for key, expected in (
        ("run_id", job.run_id),
        ("fold", job.fold),
        ("model_id", job.model_id),
        ("seed", job.seed),
    ):
        if metadata_record.get(key) != expected:
            errors.append(f"checkpoint metadata {key} differs")
    if checkpoint_hashes.get("experiment_config_sha256") != job.config_content_sha256:
        errors.append("checkpoint config hash differs")
    recorded_commit = metadata_record.get("environment", {}).get("git_commit")
    if expected_git_commit is not None and recorded_commit != expected_git_commit:
        errors.append(
            f"checkpoint git commit {recorded_commit!r} differs from {expected_git_commit!r}"
        )
    return errors


def _validate_robustness_completion(
    job: ExperimentJob,
    *,
    protocol: RobustnessProtocol,
    expected_git_commit: str | None,
) -> list[str]:
    errors: list[str] = []
    try:
        summary = json.loads(job.summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read robustness summary: {error}"]
    if summary.get("schema_version") != 1:
        errors.append("robustness summary schema is invalid")
    for key, expected in (
        ("fold", job.fold),
        ("model_id", job.model_id),
        ("seed", job.seed),
        ("test_set_tuning_permitted", False),
        ("test_rows_used_for_temperature_fit", 0),
        ("test_rows_used_for_policy_selection", 0),
    ):
        if summary.get(key) != expected:
            errors.append(f"robustness summary {key} differs")
    analyses = summary.get("analyses")
    if (
        not isinstance(analyses, list)
        or len(analyses) != 1
        or analyses[0].get("analysis_id") != job.analysis_id
    ):
        errors.append("robustness summary does not contain exactly the selected analysis")
    inputs = summary.get("inputs", {})
    robust_input = inputs.get("robustness_config", {})
    parent_input = inputs.get("parent_config", {})
    if robust_input.get("content_sha256") != protocol.config_sha256:
        errors.append("robustness content hash differs")
    if parent_input.get("content_sha256") != protocol.parent_config_sha256:
        errors.append("robustness parent content hash differs")
    if inputs.get("manifest_sha256") != sha256_file(
        _resolved(protocol.parent_config["data"]["manifest"])
    ):
        errors.append("robustness manifest hash differs")
    if inputs.get("fold_manifest_sha256") != sha256_file(
        _resolved(protocol.parent_config["data"]["fold_manifest"])
    ):
        errors.append("robustness fold-manifest hash differs")
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        return errors + ["robustness outputs is not a mapping"]
    for name in ("prefix_ledger", "decision_ledger", "metrics", "policy"):
        record = outputs.get(name)
        if not isinstance(record, dict):
            errors.append(f"missing robustness output record {name}")
            continue
        try:
            path = _resolve_summary_artifact(record.get("path"))
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"missing robustness output {path}")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"robustness output hash differs for {name}")
    if errors:
        return errors
    prefix_path = _resolve_summary_artifact(outputs["prefix_ledger"]["path"])
    try:
        ledger = _read_table(prefix_path)
        validate_prefix_ledger(ledger)
    except Exception as error:  # noqa: BLE001
        errors.append(f"robustness prefix ledger validation failed: {error}")
    else:
        if set(ledger.get("robustness_analysis_id", pd.Series(dtype=str)).astype(str)) != {
            job.analysis_id
        }:
            errors.append("robustness ledger analysis identity differs")
        if set(ledger["fold"].unique()) != {job.fold}:
            errors.append("robustness ledger fold differs")
    recorded_commit = summary.get("environment", {}).get("git_commit")
    if expected_git_commit is not None and recorded_commit != expected_git_commit:
        errors.append(
            f"robustness git commit {recorded_commit!r} differs from {expected_git_commit!r}"
        )
    return errors


def verify_job_completion(
    job: ExperimentJob,
    *,
    expected_git_commit: str | None = None,
    robustness_protocol: RobustnessProtocol | None = None,
) -> CompletionStatus:
    """Classify a job as pending, verified complete, or invalid/partial."""

    existing = _existing_artifacts(job)
    if not existing:
        return CompletionStatus(job.run_id, "pending", "no output artifacts")
    if not job.summary_path.is_file():
        return CompletionStatus(
            job.run_id,
            "invalid",
            "output directory is partial: completion summary is absent",
        )
    if job.stage == "robustness_test_time":
        protocol = robustness_protocol or load_robustness_protocol(job.config_path)
        errors = _validate_robustness_completion(
            job,
            protocol=protocol,
            expected_git_commit=expected_git_commit,
        )
    else:
        try:
            config = load_yaml(job.config_path)
        except (OSError, TypeError) as error:
            return CompletionStatus(job.run_id, "invalid", f"cannot load run config: {error}")
        errors = _validate_fold_completion(
            job,
            config=config,
            expected_git_commit=expected_git_commit,
        )
    if errors:
        return CompletionStatus(job.run_id, "invalid", "; ".join(errors[:8]))
    return CompletionStatus(job.run_id, "complete", "all artifacts and invariants verified")


def _check(
    check_id: str,
    ok: bool,
    message: str,
    *,
    severity: Literal["error", "warning", "info"] = "error",
    **details: Any,
) -> PrerequisiteCheck:
    return PrerequisiteCheck(check_id, ok, severity, message, details)


def _git_state(git_policy: GitPolicy) -> tuple[PrerequisiteCheck, str | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0:
        return (
            _check(
                "git_commit",
                False,
                "No Git commit exists; commit the frozen code and configs before execution.",
            ),
            None,
        )
    commit = revision.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        return _check("git_status", False, "Could not inspect Git worktree status."), commit
    dirty_lines = tuple(line for line in status.stdout.splitlines() if line)
    if git_policy == "clean":
        return (
            _check(
                "git_clean",
                not dirty_lines,
                (
                    f"Git commit {commit} is clean."
                    if not dirty_lines
                    else "Git worktree is not clean under the required clean policy."
                ),
                dirty_entries=list(dirty_lines[:25]),
            ),
            commit,
        )
    return (
        _check(
            "git_commit",
            True,
            (
                f"Git commit {commit} exists; {len(dirty_lines)} dirty entries are allowed "
                "by the commit-only policy and will be recorded as a warning."
            ),
            severity="warning" if dirty_lines else "info",
            dirty_entries=list(dirty_lines[:25]),
        ),
        commit,
    )


def _validate_manifest_and_audio(config: dict[str, Any]) -> tuple[PrerequisiteCheck, ...]:
    manifest_path = _resolved(config["data"]["manifest"])
    fold_path = _resolved(config["data"]["fold_manifest"])
    checks: list[PrerequisiteCheck] = []
    try:
        manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "speaker_id": str})
    except (OSError, pd.errors.ParserError) as error:
        return (_check("manifest", False, f"Cannot read frozen manifest: {error}"),)
    required = {
        "sample_id",
        "speaker_id",
        "label",
        "label_id",
        "path",
        "sample_rate",
        "num_frames",
    }
    missing = sorted(required - set(manifest.columns))
    expected = config["data"]["expected"]
    counts = {
        "utterances": len(manifest),
        "speakers": int(manifest["speaker_id"].nunique()) if "speaker_id" in manifest else 0,
        "control": int(manifest["label"].eq("control").sum()) if "label" in manifest else 0,
        "dysarthric": (
            int(manifest["label"].eq("dysarthric").sum()) if "label" in manifest else 0
        ),
    }
    expected_counts = {key: int(expected[key]) for key in counts}
    manifest_ok = (
        not missing
        and counts == expected_counts
        and not manifest["sample_id"].duplicated().any()
    )
    checks.append(
        _check(
            "manifest",
            manifest_ok,
            (
                "Frozen manifest columns, uniqueness, and cohort counts are valid."
                if manifest_ok
                else "Frozen manifest columns, uniqueness, or cohort counts differ."
            ),
            path=str(manifest_path),
            missing_columns=missing,
            observed_counts=counts,
            expected_counts=expected_counts,
        )
    )
    if missing:
        return tuple(checks)

    try:
        roles = pd.read_csv(fold_path, dtype={"sample_id": str, "speaker_id": str})
    except (OSError, pd.errors.ParserError) as error:
        checks.append(_check("fold_manifest", False, f"Cannot read fold manifest: {error}"))
        return tuple(checks)
    fold_ids = tuple(sorted(int(value) for value in roles.get("fold", pd.Series()).unique()))
    valid_roles = {"train", "validation", "test"}
    fold_ok = (
        {"fold", "sample_id", "role"}.issubset(roles.columns)
        and fold_ids == (0, 1, 2, 3, 4)
        and not roles.duplicated(["fold", "sample_id"]).any()
        and set(roles["role"].astype(str)).issubset(valid_roles)
        and all(
            set(roles.loc[roles["fold"].eq(fold), "sample_id"]) == set(manifest["sample_id"])
            for fold in fold_ids
        )
    )
    checks.append(
        _check(
            "fold_manifest",
            fold_ok,
            (
                "Every frozen fold covers every manifest sample exactly once."
                if fold_ok
                else "Fold manifest coverage or role invariants differ."
            ),
            path=str(fold_path),
            folds=list(fold_ids),
            rows=len(roles),
        )
    )

    missing_audio: list[str] = []
    invalid_audio: list[str] = []
    for row in manifest.itertuples(index=False):
        path = _resolved(row.path)
        if not path.is_file():
            if len(missing_audio) < 25:
                missing_audio.append(str(path))
            continue
        try:
            with wave.open(str(path), "rb") as handle:
                observed_rate = handle.getframerate()
                observed_frames = handle.getnframes()
            mismatched = observed_rate != int(row.sample_rate) or observed_frames != int(
                row.num_frames
            )
            if mismatched and len(invalid_audio) < 25:
                invalid_audio.append(str(path))
        except (OSError, wave.Error) as error:
            if len(invalid_audio) < 25:
                invalid_audio.append(f"{path}: {error}")
    present = len(manifest) - sum(
        not _resolved(value).is_file() for value in manifest["path"].astype(str)
    )
    audio_ok = present == len(manifest) and not invalid_audio
    checks.append(
        _check(
            "audio_coverage",
            audio_ok,
            (
                f"All {len(manifest)} WAVs exist and match frozen header metadata."
                if audio_ok
                else (
                    f"Audio recovery is incomplete or invalid: {present}/{len(manifest)} "
                    "manifest WAVs are present."
                )
            ),
            present=present,
            expected=len(manifest),
            first_missing=missing_audio,
            first_invalid=invalid_audio,
        )
    )
    return tuple(checks)


def _package_check(selected: Sequence[ExperimentJob]) -> PrerequisiteCheck:
    distributions = [
        "numpy",
        "pandas",
        "pyarrow",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "soundfile",
        "torch",
    ]
    if any(job.stage == "wavlm_reference" for job in selected):
        distributions.extend(("torchaudio", "transformers"))
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    missing = sorted(name for name, version in versions.items() if version is None)
    return _check(
        "python_dependencies",
        not missing,
        (
            "All dependencies required by the selected stages are installed."
            if not missing
            else f"Selected stages require missing distributions: {missing}."
        ),
        versions=versions,
    )


def _wavlm_cache_path(config: dict[str, Any]) -> Path:
    reference = config["models"]["modern_reference_config"]
    checkpoint_id = str(reference["checkpoint_id"]).replace("/", "--")
    revision = str(reference["revision"])
    explicit_cache = os.environ.get("HF_HUB_CACHE")
    if explicit_cache:
        hub = Path(explicit_cache).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub = hf_home.expanduser() / "hub"
    return hub / f"models--{checkpoint_id}" / "snapshots" / revision


def _wavlm_check(
    config: dict[str, Any],
    selected: Sequence[ExperimentJob],
    *,
    allow_download: bool,
) -> PrerequisiteCheck | None:
    if not any(job.stage == "wavlm_reference" for job in selected):
        return None
    snapshot = _wavlm_cache_path(config)
    config_present = (snapshot / "config.json").is_file()
    weights_present = any(
        (snapshot / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    cached = config_present and weights_present
    ok = cached or allow_download
    return _check(
        "wavlm_checkpoint",
        ok,
        (
            f"Pinned WavLM snapshot is cached at {snapshot}."
            if cached
            else (
                "Pinned WavLM snapshot is absent, but explicit download authorization is set."
                if allow_download
                else "Pinned WavLM snapshot is absent and downloads are not authorized."
            )
        ),
        severity="warning" if allow_download and not cached else "error",
        snapshot=str(snapshot),
        download_authorized=allow_download,
    )


def run_preflight(
    matrix: RunMatrix,
    selected: Sequence[ExperimentJob],
    *,
    git_policy: GitPolicy = "clean",
    allow_wavlm_download: bool = False,
    min_free_gib: float = 10.0,
) -> tuple[PreflightReport, str | None]:
    """Check exact prerequisites without creating artifacts or launching work."""

    if git_policy not in {"clean", "commit"}:
        raise ValueError("git_policy must be clean or commit")
    if min_free_gib < 0.0:
        raise ValueError("min_free_gib must be non-negative")
    config = load_yaml(matrix.config_path)
    _require_frozen_config(config, name="Experiment config")
    protocol = load_robustness_protocol(matrix.robustness_config_path)
    checks: list[PrerequisiteCheck] = [
        _check(
            "run_matrix",
            bool(selected),
            (
                f"Selected {len(selected)} of {len(matrix.jobs)} frozen jobs; matrix "
                f"SHA-256 is {matrix.matrix_sha256}."
                if selected
                else "No jobs match the supplied filters."
            ),
            selected_jobs=len(selected),
            total_jobs=len(matrix.jobs),
            matrix_sha256=matrix.matrix_sha256,
        ),
        _check(
            "frozen_configs",
            (
                sha256_file(matrix.config_path) == matrix.config_file_sha256
                and sha256_json(config) == matrix.config_content_sha256
                and sha256_file(matrix.robustness_config_path)
                == matrix.robustness_config_file_sha256
                and protocol.config_sha256 == matrix.robustness_config_content_sha256
            ),
            "Frozen experiment and robustness configs match the planned hashes.",
        ),
    ]
    git_check, commit = _git_state(git_policy)
    checks.append(git_check)
    checks.extend(_validate_manifest_and_audio(config))
    checks.append(_package_check(selected))
    wavlm = _wavlm_check(config, selected, allow_download=allow_wavlm_download)
    if wavlm is not None:
        checks.append(wavlm)

    output_parent = next(
        (parent for parent in matrix.jobs[0].output_directory.parents if parent.exists()),
        PROJECT_ROOT,
    )
    free_bytes = shutil.disk_usage(output_parent).free
    threshold = int(min_free_gib * 1024**3)
    checks.append(
        _check(
            "free_disk",
            free_bytes >= threshold,
            (
                f"Free disk is {free_bytes / 1024**3:.2f} GiB; "
                f"required safety floor is {min_free_gib:.2f} GiB."
            ),
            path=str(output_parent),
            free_bytes=free_bytes,
            required_bytes=threshold,
        )
    )
    return PreflightReport(tuple(checks)), commit


def materialize_retraining_configs(
    matrix: RunMatrix,
    selected: Sequence[ExperimentJob],
) -> dict[str, Path]:
    """Write only selected-condition configs through the frozen robustness helper."""

    retraining_jobs = [job for job in selected if job.stage == "robustness_retrain"]
    if not retraining_jobs:
        return {}
    roots = {job.config_path.parent for job in retraining_jobs}
    if len(roots) != 1:
        raise ValueError("Selected retraining jobs do not share one derived-config root")
    protocol = load_robustness_protocol(matrix.robustness_config_path)
    paths = write_retraining_configs(protocol, next(iter(roots)))
    for job in retraining_jobs:
        generated = paths.get(str(job.analysis_id))
        if generated is None or generated.resolve() != job.config_path.resolve():
            raise RuntimeError(f"Derived config path differs for {job.run_id}")
        if sha256_json(load_yaml(generated)) != job.config_content_sha256:
            raise RuntimeError(f"Derived config content differs for {job.run_id}")
    return paths


def _recorded_git_commit(job: ExperimentJob) -> str | None:
    """Return the commit an artifact records, wherever that stage happens to store it.

    Robustness summaries carry ``environment.git_commit`` directly. Fold summaries do not:
    training writes the environment into the checkpoint metadata instead. Reading only the
    summary silently reports "no drift" for every fold dependency.
    """

    if job.summary_path.is_file():
        try:
            summary = json.loads(job.summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        recorded = summary.get("environment", {}).get("git_commit")
        if recorded is not None:
            return str(recorded)

    if job.checkpoint_path is not None and job.checkpoint_path.is_file():
        import torch

        try:
            checkpoint = torch.load(
                job.checkpoint_path, map_location="cpu", weights_only=False, mmap=True
            )
        except Exception:  # noqa: BLE001  # An unreadable checkpoint is reported elsewhere.
            return None
        if isinstance(checkpoint, dict):
            metadata = checkpoint.get("metadata")
            if isinstance(metadata, dict):
                recorded = metadata.get("environment", {}).get("git_commit")
                if recorded is not None:
                    return str(recorded)
    return None


def dependency_commit_drift(
    selected: Sequence[ExperimentJob],
    matrix: RunMatrix,
    *,
    expected_git_commit: str | None,
) -> dict[str, str]:
    """Map each already-complete dependency to the commit it was actually produced at.

    A pipeline stage legitimately runs at a later commit than the stage it consumes, so a
    dependency's commit is recorded rather than required to match. Every other invariant --
    config content hash, architecture hash, manifest hashes, ledger schema -- is still
    enforced by ``verify_job_completion``.
    """

    if expected_git_commit is None:
        return {}
    selected_ids = {job.run_id for job in selected}
    lookup = {job.run_id: job for job in matrix.jobs}
    drift: dict[str, str] = {}
    for job in selected:
        for dependency_id in job.dependency_run_ids:
            if dependency_id in selected_ids or dependency_id in drift:
                continue
            dependency = lookup[dependency_id]
            recorded = _recorded_git_commit(dependency)
            if recorded is not None and recorded != expected_git_commit:
                drift[dependency_id] = recorded
    return drift


def dependency_errors(
    selected: Sequence[ExperimentJob],
    matrix: RunMatrix,
    *,
    expected_git_commit: str | None,
    allow_commit_drift: bool = False,
) -> tuple[str, ...]:
    """Return unmet dependencies that are neither selected nor already complete.

    ``allow_commit_drift`` permits a dependency that was produced at a different commit, so
    long as it still satisfies every content invariant. Use it when a later stage consumes an
    earlier stage's artifacts across an intervening commit; the drift is reported by
    ``dependency_commit_drift`` and must be recorded alongside the results.
    """

    selected_ids = {job.run_id for job in selected}
    lookup = {job.run_id: job for job in matrix.jobs}
    protocol = load_robustness_protocol(matrix.robustness_config_path)
    dependency_commit = None if allow_commit_drift else expected_git_commit
    errors: list[str] = []
    for job in selected:
        for dependency_id in job.dependency_run_ids:
            if dependency_id in selected_ids:
                continue
            dependency = lookup[dependency_id]
            status = verify_job_completion(
                dependency,
                expected_git_commit=dependency_commit,
                robustness_protocol=protocol,
            )
            if status.state != "complete":
                errors.append(
                    f"{job.run_id} requires {dependency_id}, which is {status.state}: "
                    f"{status.reason}"
                )
    return tuple(errors)


def execute_jobs(
    matrix: RunMatrix,
    selected: Sequence[ExperimentJob],
    *,
    expected_git_commit: str,
    rerun_invalid: bool = False,
    force_complete: bool = False,
) -> tuple[CompletionStatus, ...]:
    """Run selected jobs one at a time and verify every completed artifact set."""

    materialize_retraining_configs(matrix, selected)
    protocol = load_robustness_protocol(matrix.robustness_config_path)
    statuses: list[CompletionStatus] = []
    for job in selected:
        before = verify_job_completion(
            job,
            expected_git_commit=expected_git_commit,
            robustness_protocol=protocol,
        )
        if before.state == "complete" and not force_complete:
            statuses.append(before)
            continue
        if before.state == "invalid" and not rerun_invalid:
            raise RuntimeError(
                f"Refusing to overwrite invalid/partial run {job.run_id}: {before.reason}. "
                "Inspect or preserve it, then pass --rerun-invalid to retry in place."
            )
        subprocess.run(job.command, cwd=PROJECT_ROOT, check=True)
        after = verify_job_completion(
            job,
            expected_git_commit=expected_git_commit,
            robustness_protocol=protocol,
        )
        if after.state != "complete":
            raise RuntimeError(
                f"Launcher returned successfully but {job.run_id} did not verify: {after.reason}"
            )
        statuses.append(after)
    return tuple(statuses)


def plan_record(
    matrix: RunMatrix,
    selected: Sequence[ExperimentJob],
    *,
    statuses: Sequence[CompletionStatus] = (),
    preflight: PreflightReport | None = None,
) -> dict[str, Any]:
    status_lookup = {item.run_id: item for item in statuses}
    return {
        "schema_version": 1,
        "matrix_sha256": matrix.matrix_sha256,
        "config": {
            "path": _portable_path(matrix.config_path),
            "file_sha256": matrix.config_file_sha256,
            "content_sha256": matrix.config_content_sha256,
        },
        "robustness_config": {
            "path": _portable_path(matrix.robustness_config_path),
            "file_sha256": matrix.robustness_config_file_sha256,
            "content_sha256": matrix.robustness_config_content_sha256,
        },
        "folds": list(matrix.folds),
        "primary_seed": matrix.primary_seed,
        "robustness_seeds": list(matrix.robustness_seeds),
        "primary_robustness_reuse": {
            "analysis_id": matrix.primary_robustness_analysis_id,
            "run_ids": list(matrix.primary_robustness_run_ids),
        },
        "total_jobs": len(matrix.jobs),
        "selected_jobs": len(selected),
        "sequential_execution": True,
        "jobs": [
            {
                **job.identity_record(),
                "completion": (
                    asdict(status_lookup[job.run_id]) if job.run_id in status_lookup else None
                ),
            }
            for job in selected
        ],
        "preflight": (
            None
            if preflight is None
            else {
                "ready": preflight.ready,
                "checks": [asdict(check) for check in preflight.checks],
            }
        ),
    }


def write_plan(record: dict[str, Any], output_directory: str | Path) -> tuple[Path, Path]:
    """Atomically write the JSON audit plan and a flat CSV job matrix."""

    output = _resolved(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "experiment_plan.json"
    csv_path = output / "experiment_jobs.csv"
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary_json.replace(json_path)
    rows = []
    for job in record["jobs"]:
        completion = job.pop("completion", None)
        rows.append(
            {
                **job,
                "dependency_run_ids": json.dumps(job["dependency_run_ids"]),
                "command": json.dumps(job["command"]),
                "completion_state": None if completion is None else completion["state"],
                "completion_reason": None if completion is None else completion["reason"],
            }
        )
    frame = pd.DataFrame(rows)
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    return json_path, csv_path


def with_python_executable(job: ExperimentJob, executable: str) -> ExperimentJob:
    """Replace only argv[0], preserving the frozen job and output identity."""

    return replace(job, command=(executable, *job.command[1:]))
