from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from timeentropy.config import load_yaml
from timeentropy.orchestration import (
    STAGE_ORDER,
    build_run_matrix,
    dependency_commit_drift,
    dependency_errors,
    plan_record,
    run_preflight,
    select_jobs,
    verify_job_completion,
    with_python_executable,
    write_plan,
)


@pytest.fixture(scope="module")
def matrix():
    return build_run_matrix()


def test_matrix_expansion_is_deterministic_and_unique(matrix) -> None:
    rebuilt = build_run_matrix()
    assert rebuilt.matrix_sha256 == matrix.matrix_sha256

    run_ids = [job.run_id for job in matrix.jobs]
    assert len(set(run_ids)) == len(run_ids)
    directories = [job.output_directory for job in matrix.jobs]
    assert len(set(directories)) == len(directories)


def test_stage_counts_match_the_frozen_protocol(matrix) -> None:
    counts: dict[str, int] = {}
    for job in matrix.jobs:
        counts[job.stage] = counts.get(job.stage, 0) + 1

    config = load_yaml(matrix.config_path)
    folds = len(matrix.folds)
    primary_models = len(config["models"]["primary"])

    assert counts["primary_stft"] == primary_models * folds
    assert counts["seed_replication"] == len(matrix.robustness_seeds) * folds
    assert counts["wavlm_reference"] == folds
    assert sum(counts.values()) == len(matrix.jobs)


def test_jobs_are_emitted_in_frozen_stage_order(matrix) -> None:
    ranks = [STAGE_ORDER.index(job.stage) for job in matrix.jobs]
    assert ranks == sorted(ranks)


def test_test_time_jobs_depend_on_the_primary_headline_run(matrix) -> None:
    primary_ids = {job.run_id for job in matrix.jobs if job.stage == "primary_stft"}
    test_time = [job for job in matrix.jobs if job.stage == "robustness_test_time"]
    assert test_time

    for job in test_time:
        assert job.dependency_run_ids
        assert set(job.dependency_run_ids) <= primary_ids


def test_selection_filters_and_dependency_closure(matrix) -> None:
    only_primary = select_jobs(matrix, stages=["primary_stft"])
    assert {job.stage for job in only_primary} == {"primary_stft"}

    fold_zero = select_jobs(matrix, stages=["primary_stft"], folds=[0])
    assert {job.fold for job in fold_zero} == {0}

    test_time_fold_zero = select_jobs(matrix, stages=["robustness_test_time"], folds=[0])
    with_dependencies = select_jobs(
        matrix, stages=["robustness_test_time"], folds=[0], with_dependencies=True
    )
    assert len(with_dependencies) > len(test_time_fold_zero)
    assert "primary_stft" in {job.stage for job in with_dependencies}

    with pytest.raises(ValueError):
        select_jobs(matrix, stages=["not_a_stage"])
    with pytest.raises(ValueError):
        select_jobs(matrix, run_ids=["not-a-run"])


def test_dependency_errors_reports_unsatisfied_prerequisites(tmp_path) -> None:
    # An isolated output root: asserting against artifacts/runs would flip this test as
    # soon as the real primary stage completes.
    isolated = build_run_matrix(output_root=tmp_path / "runs")
    test_time = select_jobs(isolated, stages=["robustness_test_time"], folds=[0])
    errors = dependency_errors(test_time, isolated, expected_git_commit=None)
    assert errors
    assert all("requires" in message for message in errors)

    closed = select_jobs(
        isolated, stages=["robustness_test_time"], folds=[0], with_dependencies=True
    )
    assert dependency_errors(closed, isolated, expected_git_commit=None) == ()


def test_completion_states_distinguish_pending_from_partial(matrix, tmp_path) -> None:
    # Both cases are relocated under tmp_path: asserting against artifacts/runs would
    # make the result depend on whether a real run happens to be in flight.
    selected = select_jobs(matrix, stages=["primary_stft"], folds=[0])[0]
    empty = tmp_path / "empty" / selected.run_id
    job = replace(
        selected,
        output_directory=empty,
        summary_path=empty / "run_summary.json",
        ledger_path=empty / "prefix_predictions.parquet",
        checkpoint_path=empty / "best_checkpoint.pt",
    )
    assert verify_job_completion(job).state == "pending"

    output = tmp_path / "partial" / job.run_id
    output.mkdir(parents=True)
    (output / "prefix_predictions.parquet").write_bytes(b"partial")
    partial = replace(
        job,
        output_directory=output,
        summary_path=output / "run_summary.json",
        ledger_path=output / "prefix_predictions.parquet",
        checkpoint_path=output / "best_checkpoint.pt",
    )
    status = verify_job_completion(partial)
    assert status.state == "invalid"
    assert "summary" in status.reason


def test_with_python_executable_replaces_only_the_interpreter(matrix) -> None:
    job = matrix.jobs[0]
    swapped = with_python_executable(job, "/usr/bin/python3")
    assert swapped.command[0] == "/usr/bin/python3"
    assert swapped.command[1:] == job.command[1:]
    assert swapped.run_id == job.run_id
    assert swapped.output_directory == job.output_directory
    # The interpreter is normalised out of the portable record, so swapping it must not
    # change the job identity that the run-matrix hash covers.
    assert swapped.identity_record() == job.identity_record()


def test_build_run_matrix_rejects_unfrozen_and_invalid_settings(matrix, tmp_path) -> None:
    config = load_yaml(matrix.config_path)
    config["status"] = "draft"
    thawed = tmp_path / "thawed.yaml"
    thawed.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="frozen_before_test_evaluation"):
        build_run_matrix(thawed, matrix.robustness_config_path)

    with pytest.raises(ValueError, match="device"):
        build_run_matrix(matrix.config_path, matrix.robustness_config_path, device="cuda")
    with pytest.raises(ValueError, match="num_workers"):
        build_run_matrix(matrix.config_path, matrix.robustness_config_path, num_workers=-1)


def test_preflight_reports_unmet_prerequisites_without_creating_artifacts(
    matrix, tmp_path
) -> None:
    selected = select_jobs(matrix, stages=["primary_stft"], folds=[0])
    before = sorted(path.name for path in tmp_path.iterdir())
    report, _ = run_preflight(matrix, selected, git_policy="commit")

    check_ids = {check.check_id for check in report.checks}
    assert {"run_matrix", "frozen_configs", "manifest", "fold_manifest", "free_disk"} <= check_ids
    assert all(check.severity in {"error", "warning", "info"} for check in report.checks)
    assert sorted(path.name for path in tmp_path.iterdir()) == before

    with pytest.raises(ValueError, match="git_policy"):
        run_preflight(matrix, selected, git_policy="dirty")


def test_plan_record_round_trips_to_disk(matrix, tmp_path) -> None:
    selected = select_jobs(matrix, stages=["primary_stft"], folds=[0])
    record = plan_record(matrix, selected)
    assert record["selected_jobs"] == len(selected)
    assert record["sequential_execution"] is True
    assert record["matrix_sha256"] == matrix.matrix_sha256

    json_path, csv_path = write_plan(record, tmp_path / "plan")
    assert json.loads(Path(json_path).read_text())["matrix_sha256"] == matrix.matrix_sha256
    assert csv_path.is_file()


def test_default_interpreter_can_import_the_project(matrix) -> None:
    """The launcher interpreter must be the running one, symlinks intact.

    Resolving ``sys.executable`` dereferences a virtualenv's symlink into the base
    interpreter, which has no project install, so every launched job dies with
    ModuleNotFoundError before writing an artifact.
    """

    interpreter = matrix.jobs[0].command[0]
    assert Path(interpreter) == Path(sys.executable)

    completed = subprocess.run(
        [interpreter, "-c", "import timeentropy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_matrix_identity_is_independent_of_the_interpreter(matrix) -> None:
    other = build_run_matrix(python_executable="/somewhere/else/python")
    assert other.matrix_sha256 == matrix.matrix_sha256


def test_dependency_commit_drift_is_reported_and_gated(tmp_path) -> None:
    """A later stage may consume an earlier stage's artifacts, but the drift must surface."""

    isolated = build_run_matrix(output_root=tmp_path / "runs")
    consumer = select_jobs(isolated, stages=["robustness_test_time"], folds=[0])[:1]
    dependency_id = consumer[0].dependency_run_ids[0]
    dependency = next(job for job in isolated.jobs if job.run_id == dependency_id)

    # No artifacts yet: nothing to drift, and the dependency is simply unmet.
    assert dependency_commit_drift(consumer, isolated, expected_git_commit="aaa") == {}
    assert dependency_errors(consumer, isolated, expected_git_commit="aaa")

    dependency.output_directory.mkdir(parents=True)
    dependency.summary_path.write_text(
        json.dumps({"environment": {"git_commit": "bbb"}}),
        encoding="utf-8",
    )

    drift = dependency_commit_drift(consumer, isolated, expected_git_commit="aaa")
    assert drift == {dependency_id: "bbb"}

    # A matching commit is not drift, and no expectation means nothing to compare.
    assert dependency_commit_drift(consumer, isolated, expected_git_commit="bbb") == {}
    assert dependency_commit_drift(consumer, isolated, expected_git_commit=None) == {}


def test_drift_is_read_from_the_checkpoint_when_the_summary_omits_it(tmp_path) -> None:
    """Fold summaries carry no environment block; training records it in the checkpoint.

    Reading only the summary reports "no drift" for every fold dependency, which would make
    the recorded audit trail silently empty exactly where it matters.
    """

    import torch

    isolated = build_run_matrix(output_root=tmp_path / "runs")
    consumer = select_jobs(isolated, stages=["robustness_test_time"], folds=[0])[:1]
    dependency_id = consumer[0].dependency_run_ids[0]
    dependency = next(job for job in isolated.jobs if job.run_id == dependency_id)

    dependency.output_directory.mkdir(parents=True)
    # A fold summary as training actually writes it: no "environment" key at all.
    dependency.summary_path.write_text(json.dumps({"run_id": dependency_id}), encoding="utf-8")
    assert dependency_commit_drift(consumer, isolated, expected_git_commit="aaa") == {}

    assert dependency.checkpoint_path is not None
    torch.save(
        {"metadata": {"environment": {"git_commit": "ccc"}}},
        dependency.checkpoint_path,
    )
    assert dependency_commit_drift(consumer, isolated, expected_git_commit="aaa") == {
        dependency_id: "ccc"
    }
    assert dependency_commit_drift(consumer, isolated, expected_git_commit="ccc") == {}


def test_allow_commit_drift_still_enforces_content_invariants(tmp_path) -> None:
    """Relaxing the commit check must not relax anything else."""

    isolated = build_run_matrix(output_root=tmp_path / "runs")
    consumer = select_jobs(isolated, stages=["robustness_test_time"], folds=[0])[:1]
    dependency_id = consumer[0].dependency_run_ids[0]
    dependency = next(job for job in isolated.jobs if job.run_id == dependency_id)

    dependency.output_directory.mkdir(parents=True)
    dependency.summary_path.write_text(
        json.dumps({"environment": {"git_commit": "bbb"}}),
        encoding="utf-8",
    )

    # The summary exists but the run is otherwise incomplete, so the dependency stays
    # unmet even with the commit check relaxed.
    errors = dependency_errors(
        consumer, isolated, expected_git_commit="aaa", allow_commit_drift=True
    )
    assert errors
    assert all("git commit" not in message for message in errors)
