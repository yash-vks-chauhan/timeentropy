#!/usr/bin/env python3
"""Build a privacy-screened, deterministic reproducibility archive.

`--draft` permits a dirty worktree and artifacts from runs that have not finished; final
mode refuses both, so an archive without the `--draft` marker is one whose every claimed
result exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = PROJECT_ROOT / "artifacts/reproducibility-package"
DEFAULT_ARCHIVE = PROJECT_ROOT / "artifacts/timeentropy-reproducibility.zip"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

FINAL_REQUIRED = (
    "Time_Entropy_Dysarthric_Speech_Classification_REVISED.docx",
    "Response_to_Reviewers.docx",
    "artifacts/evaluation/transformer-seed20260820/decisions.parquet",
    "artifacts/evaluation/transformer-seed20260821/decisions.parquet",
    "artifacts/evaluation/transformer-three-seed/aggregate_manifest.json",
    "artifacts/reports/transformer-seed20260820/report_manifest.json",
    "artifacts/reports/transformer-seed20260821/report_manifest.json",
    "artifacts/robustness/retrained/causal_noise_floor_preroll0/decisions.parquet",
    "artifacts/robustness/retrained/causal_fixed_preroll30/decisions.parquet",
    "artifacts/robustness/retrained/offline_peak_preroll30/decisions.parquet",
    "artifacts/robustness/retrained/pre_roll_comparison/paired_speaker_bootstrap.csv",
    "artifacts/error_audit/per_system_error_strata.csv",
    "artifacts/sensitivity/transformer_complete_calibrated/theta_epsilon_surface_exact.csv",
)

SOURCE_PATHS = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "src",
    "scripts",
    "tests",
    "configs",
    "docs",
    "data/metadata/README.md",
    "data/metadata/torgo_manifest_v1.csv",
    "data/metadata/torgo_manifest_v1.meta.json",
    "data/metadata/torgo_folds_v1.csv",
    "data/annotations/onset_annotation_sheet_v1.schema.json",
)

ERROR_AUDIT_ALLOWLIST = (
    "audit_manifest.json",
    "error_audit_summary.json",
    "overall_error_strata.csv",
    "per_system_error_strata.csv",
)

FORBIDDEN_PATH_MARKERS = (
    b"/Users/",
    b"yashchauhan",
    b"Desktop/timeentropy",
    b"timeentropy_runtime",
    b"103081",
)

FORBIDDEN_TABLE_COLUMNS = {
    "absolute_path",
    "annotator",
    "audio_path",
    "notes",
    "prompt_text",
    "recovered_path",
    "review_notes",
    "reviewer",
    "source_path",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(*command: str, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_tree(source: Path, destination: Path, *, exclude: Iterable[str] = ()) -> None:
    excluded = set(exclude)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in excluded for part in relative.parts):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pt"} or path.name == ".DS_Store":
            continue
        copy_file(path, destination / relative)


def portable_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: portable_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_json(item) for item in value]
    if not isinstance(value, str):
        return value

    candidate = Path(value)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return "<redacted-local-path>"
    return value


def sanitize_json(
    path: Path, redactions: list[dict[str, str]], *, stage: Path
) -> None:
    original_hash = sha256(path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    portable = portable_json(parsed)
    if portable == parsed:
        return
    encoded = json.dumps(portable, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    redactions.append(
        {
            "file": path.relative_to(stage).as_posix(),
            "original_sha256": original_hash,
            "packaged_sha256": sha256(path),
            "transformation": (
                "Absolute filesystem strings were made repository-relative when possible; "
                "external local paths were replaced without retaining their values."
            ),
        }
    )


def sanitize_manifest_csv(
    path: Path, redactions: list[dict[str, str]], *, stage: Path
) -> None:
    original_hash = sha256(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = [
            name
            for name in (reader.fieldnames or [])
            if name not in {"recovered_path", "prompt_text"}
        ]
    temporary = path.with_suffix(".portable.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)
    os.replace(temporary, path)
    redactions.append(
        {
            "file": path.relative_to(stage).as_posix(),
            "original_sha256": original_hash,
            "packaged_sha256": sha256(path),
            "transformation": (
                "Dropped recovered_path (private forensic path) and prompt_text "
                "(not required to audit the quantitative cohort/fold claims)."
            ),
        }
    )


def sanitize_plain_text(
    path: Path, redactions: list[dict[str, str]], *, stage: Path
) -> None:
    original = path.read_text(encoding="utf-8")
    updated = original
    updated = updated.replace(
        "/Users/yashchauhan/timeentropy_runtime/artifacts/metadata/torgo_index.csv",
        "data/metadata/torgo_index.csv",
    )
    updated = updated.replace(
        "/Users/yashchauhan/Desktop/timeentropy", "${PROJECT_ROOT}"
    )
    if updated == original:
        return
    original_hash = sha256(path)
    path.write_text(updated, encoding="utf-8")
    redactions.append(
        {
            "file": path.relative_to(stage).as_posix(),
            "original_sha256": original_hash,
            "packaged_sha256": sha256(path),
            "transformation": "Replaced local-only path literals with portable placeholders.",
        }
    )


def copy_primary_sources(stage: Path) -> None:
    for relative in SOURCE_PATHS:
        source = PROJECT_ROOT / relative
        destination = stage / relative
        if source.is_dir():
            copy_tree(source, destination, exclude={"__pycache__", ".pytest_cache"})
        elif source.is_file():
            copy_file(source, destination)

    for path in sorted(PROJECT_ROOT.glob("Time_Entropy_Dysarthric_Speech_Classification_REVISED*.docx")):
        copy_file(path, stage / "documents" / path.name)
    response = PROJECT_ROOT / "Response_to_Reviewers.docx"
    if response.is_file():
        copy_file(response, stage / "documents" / response.name)


def complete_evaluation_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    complete: list[Path] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if (directory / "decisions.parquet").is_file() and (
            directory / "evaluation_summary.json"
        ).is_file():
            complete.append(directory)
    return complete


def copy_artifacts(stage: Path) -> None:
    artifacts = PROJECT_ROOT / "artifacts"
    destination = stage / "artifacts"

    for name in ("onset_diagnosis", "sensitivity", "profiling", "figures"):
        source = artifacts / name
        if source.is_dir():
            copy_tree(source, destination / name, exclude={"__pycache__"})

    for directory in complete_evaluation_directories(artifacts / "evaluation"):
        copy_tree(directory, destination / "evaluation" / directory.name)
        report = artifacts / "reports" / directory.name
        if report.is_dir() and (report / "report_manifest.json").is_file():
            copy_tree(report, destination / "reports" / report.name)

    seed_aggregate = artifacts / "evaluation/transformer-three-seed"
    if seed_aggregate.is_dir():
        copy_tree(seed_aggregate, destination / "evaluation/transformer-three-seed")

    audit = artifacts / "error_audit"
    for name in ERROR_AUDIT_ALLOWLIST:
        source = audit / name
        if source.is_file():
            copy_file(source, destination / "error_audit" / name)

    combined = artifacts / "robustness/test-time/combined"
    if combined.is_dir():
        copy_tree(combined, destination / "robustness/test-time/combined")
    test_time = artifacts / "robustness/test-time"
    if test_time.is_dir():
        for condition in sorted(path for path in test_time.iterdir() if path.is_dir()):
            if condition.name == "combined":
                continue
            for fold in sorted(path for path in condition.iterdir() if path.is_dir()):
                for name in (
                    "clean_reference_policy.json",
                    "robustness_decisions.parquet",
                    "robustness_metrics.csv",
                ):
                    source = fold / name
                    if source.is_file():
                        copy_file(
                            source,
                            destination
                            / "robustness/test-time"
                            / condition.name
                            / fold.name
                            / name,
                        )

    retrained = artifacts / "robustness/retrained"
    if retrained.is_dir():
        for directory in complete_evaluation_directories(retrained):
            copy_tree(directory, destination / "robustness/retrained" / directory.name)
        comparisons = retrained / "pre_roll_comparison"
        if comparisons.is_dir():
            copy_tree(comparisons, destination / "robustness/retrained/pre_roll_comparison")

    run_summaries = destination / "run_summaries"
    for summary in sorted((artifacts / "runs").glob("*/run_summary.json")):
        if (summary.parent / "prefix_predictions.parquet").is_file():
            copy_file(summary, run_summaries / f"{summary.parent.name}.json")



def normalize_and_record_redactions(stage: Path) -> list[dict[str, str]]:
    redactions: list[dict[str, str]] = []
    manifest = stage / "data/metadata/torgo_manifest_v1.csv"
    if manifest.is_file():
        sanitize_manifest_csv(manifest, redactions, stage=stage)

    for relative in (
        "README.md",
        "configs/rebuild_v1.yaml",
        "tests/test_recovered_manifest.py",
    ):
        path = stage / relative
        if path.is_file():
            sanitize_plain_text(path, redactions, stage=stage)

    for path in sorted(stage.rglob("*.json")):
        sanitize_json(path, redactions, stage=stage)

    return redactions


def write_environment(stage: Path) -> None:
    output = stage / "environment/requirements-freeze.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen = run(sys.executable, "-m", "pip", "freeze").splitlines()
    normalized = []
    for line in frozen:
        if line.startswith("-e ") and "timeentropy" in line:
            normalized.append("timeentropy==0.1.0")
        else:
            normalized.append(line)
    output.write_text("\n".join(sorted(normalized, key=str.casefold)) + "\n", encoding="utf-8")


def write_package_readme(stage: Path, *, draft: bool, missing: list[str]) -> None:
    status = "DRAFT — NOT FOR SUBMISSION" if draft else "FINAL EVIDENCE PACKAGE"
    missing_lines = (
        "\n".join(f"- `{item}`" for item in missing)
        if missing
        else "- None of the package's required generated artifacts are missing."
    )
    content = f"""# Time Entropy reproducibility package

Status: **{status}**

This package contains the code, frozen protocol, privacy-screened cohort metadata,
prediction/decision ledgers, aggregate reports, robustness results, profiling outputs,
figures, and submission documents supporting the revised manuscript. It does **not**
contain TORGO audio or archives, model checkpoints, local filesystem paths, the
unblinded qualitative-review key, or incomplete human-review worksheets.

## Data access

Obtain TORGO from its authorized distributor and place the selected head-mounted WAV
files under `data/raw/torgo/<speaker>/<session>/wav_headMic/`. The packaged manifest
uses repository-relative paths. Its local forensic `recovered_path` column and prompt
text were removed; `REDACTIONS.json` preserves the original file hash and records the
transformation. The public-release author must verify the current TORGO redistribution
terms; this archive deliberately redistributes no corpus audio.

## Audit existing results without retraining

Create Python 3.12 environment, install the project, and run:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,ssl]'
.venv/bin/python scripts/verify_response_numbers.py
.venv/bin/python -m pytest -q
```

The main auditable ledgers are in `artifacts/evaluation/`; report tables and
speaker-clustered intervals are in `artifacts/reports/`. The complete theta/epsilon
surface is in `artifacts/sensitivity/`. Test-time noise/onset-shift decisions and
summaries are in `artifacts/robustness/test-time/`; retrained onset/pre-roll ablations
are in `artifacts/robustness/retrained/`. Quantitative false-positive/false-negative
strata are in `artifacts/error_audit/`.

## Scope limits

No human reference onsets or completed human qualitative cause labels were available.
Accordingly, this package supports only the label-free onset diagnostic, imposed
onset-shift experiments, and descriptive error associations; it does not support a
manual onset-error or causal error-attribution claim. Cross-corpus, far-field, and
severity-stratified validation were not performed.

## Missing final-stage files

{missing_lines}

## Integrity

`MANIFEST.sha256` hashes every packaged file except itself. `REDACTIONS.json` records
every content-bearing path/privacy transformation with pre- and post-transformation
hashes. ZIP members are ordered lexicographically with fixed timestamps and modes.
"""
    (stage / "README_PACKAGE.md").write_text(content, encoding="utf-8")


def write_metadata(stage: Path, *, draft: bool, missing: list[str]) -> None:
    commit = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain", "--untracked-files=no"))
    record = {
        "schema_version": "timeentropy-reproducibility-package-v1",
        "status": "draft" if draft else "final",
        "source_git_commit": commit,
        "tracked_worktree_dirty_at_build": dirty,
        "missing_final_required_files": missing,
        "raw_audio_included": False,
        "model_checkpoints_included": False,
        "human_review_key_included": False,
        "manual_onset_annotations_completed": False,
        "qualitative_cause_annotations_completed": False,
        "zip_member_timestamp": "1980-01-01T00:00:00",
    }
    (stage / "PACKAGE_METADATA.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_csv_headers(stage: Path) -> None:
    violations: list[str] = []
    for path in sorted(stage.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        forbidden = sorted(set(header) & FORBIDDEN_TABLE_COLUMNS)
        if forbidden:
            violations.append(f"{path.relative_to(stage)}: {forbidden}")
    if violations:
        raise RuntimeError("Forbidden table columns remain:\n" + "\n".join(violations))


def validate_parquet_schemas(stage: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - final builder uses the project venv
        raise RuntimeError("pyarrow is required to privacy-check Parquet schemas") from exc

    violations: list[str] = []
    for path in sorted(stage.rglob("*.parquet")):
        names = set(pq.read_schema(path).names)
        forbidden = sorted(names & FORBIDDEN_TABLE_COLUMNS)
        if forbidden:
            violations.append(f"{path.relative_to(stage)}: {forbidden}")
    if violations:
        raise RuntimeError("Forbidden Parquet columns remain:\n" + "\n".join(violations))


def validate_private_strings(stage: Path) -> None:
    violations: list[str] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker.lower() in data.lower():
                violations.append(f"{path.relative_to(stage)} contains {marker!r}")
    if violations:
        raise RuntimeError("Private path/identifier scan failed:\n" + "\n".join(violations))


def validate_no_forbidden_payloads(stage: Path) -> None:
    violations: list[str] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        lowered_parts = {part.lower() for part in path.relative_to(stage).parts}
        if path.suffix.lower() in {".wav", ".flac", ".mp3", ".pt", ".ckpt"}:
            violations.append(path.relative_to(stage).as_posix())
        if "qualitative_review_key.csv" in lowered_parts:
            violations.append(path.relative_to(stage).as_posix())
        if {"raw", "archives"} & lowered_parts:
            violations.append(path.relative_to(stage).as_posix())
    if violations:
        raise RuntimeError("Forbidden payloads remain:\n" + "\n".join(violations))


def write_hash_manifest(stage: Path) -> None:
    rows = []
    for path in sorted(path for path in stage.rglob("*") if path.is_file()):
        if path.name == "MANIFEST.sha256":
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(stage).as_posix()}")
    (stage / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_deterministic_zip(stage: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    root_name = "timeentropy-reproducibility"
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as output:
        for path in sorted(path for path in stage.rglob("*") if path.is_file()):
            relative = path.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(f"{root_name}/{relative}", date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if path.suffix in {".py", ".sh"} else 0o644
            info.external_attr = (0o100000 | mode) << 16
            output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(temporary, archive)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage.resolve()
    archive = args.archive.resolve()
    missing = [relative for relative in FINAL_REQUIRED if not (PROJECT_ROOT / relative).is_file()]
    tracked_dirty = bool(run("git", "status", "--porcelain", "--untracked-files=no"))
    if not args.draft and tracked_dirty:
        raise SystemExit("Refusing final package build from a dirty tracked worktree")
    if not args.draft and missing:
        raise SystemExit("Missing final required files:\n  " + "\n  ".join(missing))

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    copy_primary_sources(stage)
    copy_artifacts(stage)
    write_environment(stage)
    redactions = normalize_and_record_redactions(stage)
    (stage / "REDACTIONS.json").write_text(
        json.dumps(
            {
                "schema_version": "package-redactions-v1",
                "records": redactions,
                "note": (
                    "Original hashes permit identification of source artifacts without "
                    "retaining private local path values in this package."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_package_readme(stage, draft=args.draft, missing=missing)
    write_metadata(stage, draft=args.draft, missing=missing)

    # Sanitize JSON generated above only if future metadata gains path-valued fields.
    for path in sorted(stage.rglob("*.json")):
        if path.name != "REDACTIONS.json":
            sanitize_json(path, redactions=[], stage=stage)

    validate_csv_headers(stage)
    validate_parquet_schemas(stage)
    validate_private_strings(stage)
    validate_no_forbidden_payloads(stage)
    write_hash_manifest(stage)
    write_deterministic_zip(stage, archive)

    print(
        json.dumps(
            {
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "files": sum(1 for path in stage.rglob("*") if path.is_file()),
                "missing_final_required_files": missing,
                "stage": str(stage),
                "status": "draft" if args.draft else "final",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
