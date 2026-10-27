from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timeentropy.analysis.onset_validation import classify_prompt_type
from timeentropy.evaluation.ledger import (
    LedgerInvariantError,
    validate_decision_ledger,
    validate_prefix_ledger,
)
from timeentropy.provenance import environment_record, sha256_file, sha256_json

AUDIT_SCHEMA_VERSION = "classification-error-audit-v1"
SYSTEM_COLUMNS = ("model_id", "seed", "onset_variant", "pre_roll_ms", "policy_id")
DEFAULT_DURATION_CUTPOINTS_SEC = (0.5, 1.0, 2.0, 5.0)
DEFAULT_ENTROPY_CUTPOINTS = (0.2, 0.4, 0.6, 0.8)
DEFAULT_REVIEW_SEED = 20260819
DEFAULT_REVIEW_PER_ERROR = 20

MANIFEST_REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "speaker_id",
        "label_id",
        "prompt_text",
        "duration_sec",
        "path",
    }
)
SELECTED_PREFIX_COLUMNS = (
    "calibrated_prob_control",
    "calibrated_prob_dysarthric",
    "calibrated_predicted_label",
    "calibrated_normalized_entropy",
)
STRATIFIERS = (
    "speaker_id",
    "diagnosis",
    "prompt_type",
    "decision_time_ms",
    "duration_bin",
    "onset_status",
    "entropy_bin",
)
REVIEW_TAXONOMY_COLUMNS = (
    "review_category",
    "suspected_cause",
    "review_notes",
    "reviewer",
    "review_status",
)


@dataclass(frozen=True)
class ErrorAudit:
    enriched_decisions: pd.DataFrame
    overall_errors: pd.DataFrame
    per_system_errors: pd.DataFrame
    qualitative_review: pd.DataFrame
    qualitative_review_key: pd.DataFrame
    summary: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: frozenset[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _stable_digest(seed: int, *parts: object) -> str:
    material = "\0".join((str(seed), *(str(part) for part in parts))).encode()
    return hashlib.sha256(material).hexdigest()


def _system_id(row: Mapping[str, object]) -> str:
    return "|".join(f"{column}={row[column]}" for column in SYSTEM_COLUMNS)


def _as_optional_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "detected"}:
        return True
    if normalized in {"false", "0", "no", "n", "failed", "failure"}:
        return False
    raise ValueError(f"Cannot interpret onset detected value {value!r} as Boolean")


def _prompt_type_or_missing(prompt_text: object) -> str:
    if prompt_text is None or pd.isna(prompt_text) or not str(prompt_text).strip():
        return "missing_prompt_text"
    return classify_prompt_type(prompt_text)


def _interval_labels(cutpoints: Sequence[float], *, suffix: str = "") -> tuple[list[float], list[str]]:
    values = [float(value) for value in cutpoints]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Cutpoints must be finite and positive")
    if values != sorted(set(values)):
        raise ValueError("Cutpoints must be strictly increasing and unique")
    edges = [-np.inf, *values, np.inf]

    def display(value: float) -> str:
        return f"{value:g}{suffix}"

    labels = [f"<{display(values[0])}"]
    labels.extend(
        f"[{display(lower)},{display(upper)})"
        for lower, upper in pairwise(values)
    )
    labels.append(f">={display(values[-1])}")
    return edges, labels


def _add_bins(
    frame: pd.DataFrame,
    *,
    duration_cutpoints_sec: Sequence[float],
    entropy_cutpoints: Sequence[float],
) -> pd.DataFrame:
    enriched = frame.copy()
    duration_edges, duration_labels = _interval_labels(duration_cutpoints_sec, suffix="s")
    entropy_edges, entropy_labels = _interval_labels(entropy_cutpoints)
    enriched["duration_bin"] = pd.cut(
        pd.to_numeric(enriched["duration_sec"], errors="raise"),
        bins=duration_edges,
        labels=duration_labels,
        right=False,
    ).astype(str)
    enriched["entropy_bin"] = pd.cut(
        pd.to_numeric(enriched["calibrated_normalized_entropy"], errors="raise"),
        bins=entropy_edges,
        labels=entropy_labels,
        right=False,
    ).astype(str)
    return enriched


def _prepare_onset_table(onset_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if "sample_id" not in onset_table.columns:
        raise ValueError("Onset table is missing required column 'sample_id'")
    prepared = onset_table.copy()
    total_rows = len(prepared)

    if "reference_onset_sec" in prepared.columns:
        reference = pd.to_numeric(prepared["reference_onset_sec"], errors="coerce")
        completed = reference.notna() & np.isfinite(reference) & reference.ge(0.0)
        # Reference rows that are incomplete are never filled or treated as labels.
        prepared = prepared.loc[completed].copy()
        prepared["reference_onset_sec"] = reference.loc[completed].astype(float)
        incomplete_rows = int((~completed).sum())
    else:
        incomplete_rows = 0

    rename: dict[str, str] = {}
    if "onset_mode" in prepared.columns and "onset_variant" not in prepared.columns:
        rename["onset_mode"] = "onset_variant"
    if "detected" in prepared.columns and "onset_detected" not in prepared.columns:
        rename["detected"] = "onset_detected"
    if "estimated_onset_sec" in prepared.columns:
        rename["estimated_onset_sec"] = "onset_estimated_sec"
    prepared = prepared.rename(columns=rename)

    if "onset_detected" in prepared.columns:
        prepared["onset_detected"] = prepared["onset_detected"].map(_as_optional_bool)
    if "onset_estimated_sec" in prepared.columns:
        prepared["onset_estimated_sec"] = pd.to_numeric(
            prepared["onset_estimated_sec"], errors="coerce"
        )
    if {"onset_estimated_sec", "reference_onset_sec"}.issubset(prepared.columns):
        detected = (
            prepared["onset_detected"].fillna(False).astype(bool)
            if "onset_detected" in prepared.columns
            else prepared["onset_estimated_sec"].notna()
        )
        prepared["onset_error_ms"] = np.where(
            detected & prepared["onset_estimated_sec"].notna(),
            (prepared["onset_estimated_sec"] - prepared["reference_onset_sec"]) * 1000.0,
            np.nan,
        )

    join_keys = ["sample_id"]
    if "onset_variant" in prepared.columns:
        join_keys.append("onset_variant")
    if prepared.duplicated(join_keys).any():
        raise ValueError(f"Onset table has duplicate rows for join keys {join_keys}")
    retained = [
        *join_keys,
        *(
            column
            for column in (
                "onset_detected",
                "onset_estimated_sec",
                "reference_onset_sec",
                "onset_error_ms",
            )
            if column in prepared.columns
        ),
    ]
    return prepared[retained].copy(), {
        "input_rows": total_rows,
        "completed_reference_rows": len(prepared) if "reference_onset_sec" in prepared else 0,
        "incomplete_reference_rows_excluded": incomplete_rows,
    }


def enrich_error_decisions(
    prefix_ledger: pd.DataFrame,
    decision_ledger: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    onset_table: pd.DataFrame | None = None,
    duration_cutpoints_sec: Sequence[float] = DEFAULT_DURATION_CUTPOINTS_SEC,
    entropy_cutpoints: Sequence[float] = DEFAULT_ENTROPY_CUTPOINTS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate and link decisions to their calibrated selected-prefix evidence."""

    prefix_report = validate_prefix_ledger(prefix_ledger)
    if not prefix_report.calibrated:
        raise LedgerInvariantError("Error audit requires a calibrated prefix ledger")
    validate_decision_ledger(
        decision_ledger,
        prefix_ledger=prefix_ledger,
        prefix_prediction_column="calibrated_predicted_label",
    )
    _require_columns(manifest, MANIFEST_REQUIRED_COLUMNS, "Frozen manifest")
    if manifest.empty:
        raise ValueError("Frozen manifest must not be empty")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Frozen manifest contains duplicate sample_id rows")

    prefix_columns = [
        "run_id",
        "sample_id",
        "prefix_ms",
        "speaker_id",
        "fold",
        "seed",
        "model_id",
        "split_role",
        "onset_variant",
        "pre_roll_ms",
        "label_id",
        *SELECTED_PREFIX_COLUMNS,
    ]
    if "onset_detected" in prefix_ledger.columns:
        prefix_columns.append("onset_detected")
    selected_prefix = prefix_ledger[prefix_columns].rename(
        columns={
            "prefix_ms": "decision_time_ms",
            "speaker_id": "prefix_speaker_id",
            "fold": "prefix_fold",
            "label_id": "prefix_label_id",
            "onset_detected": "prefix_onset_detected",
        }
    )
    enriched = decision_ledger.merge(
        selected_prefix,
        how="left",
        on=["run_id", "sample_id", "decision_time_ms"],
        validate="many_to_one",
    )
    if enriched["calibrated_normalized_entropy"].isna().any():
        raise ValueError("At least one decision lacks a matching selected prefix")
    for column in ("speaker_id", "fold"):
        prefix_column = f"prefix_{column}"
        if column in decision_ledger.columns:
            if not enriched[column].astype(str).eq(enriched[prefix_column].astype(str)).all():
                raise ValueError(
                    f"Decision and prefix ledgers disagree on {column}"
                )
        else:
            enriched[column] = enriched[prefix_column]
    if not np.array_equal(
        enriched["label_id"].to_numpy(dtype=np.int64),
        enriched["prefix_label_id"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Decision and prefix ledgers disagree on label_id")
    if not enriched["split_role"].eq("test").all():
        raise ValueError("Error audit accepts out-of-fold test decisions only")
    enriched = enriched.drop(columns=["prefix_speaker_id", "prefix_fold", "prefix_label_id"])

    manifest_columns = [
        column
        for column in (
            "sample_id",
            "speaker_id",
            "label_id",
            "label",
            "diagnosis",
            "prompt_text",
            "duration_sec",
            "path",
            "recovered_path",
            "session",
            "utterance_id",
        )
        if column in manifest.columns
    ]
    manifest_metadata = manifest[manifest_columns].rename(
        columns={
            "speaker_id": "manifest_speaker_id",
            "label_id": "manifest_label_id",
            "label": "manifest_label",
            "diagnosis": "manifest_diagnosis",
        }
    )
    enriched = enriched.merge(
        manifest_metadata,
        how="left",
        on="sample_id",
        validate="many_to_one",
    )
    if enriched["manifest_speaker_id"].isna().any():
        raise ValueError("At least one decision sample is absent from the frozen manifest")
    if not enriched["speaker_id"].astype(str).eq(
        enriched["manifest_speaker_id"].astype(str)
    ).all():
        raise ValueError("Prefix ledger and frozen manifest disagree on speaker_id")
    if not np.array_equal(
        enriched["label_id"].to_numpy(dtype=np.int64),
        enriched["manifest_label_id"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Decision ledger and frozen manifest disagree on label_id")

    if "manifest_diagnosis" in enriched.columns:
        diagnosis = enriched["manifest_diagnosis"]
    elif "manifest_label" in enriched.columns:
        diagnosis = enriched["manifest_label"]
    else:
        diagnosis = enriched["label_id"].map({0: "control", 1: "dysarthric"})
    enriched["diagnosis"] = diagnosis.fillna(
        enriched["label_id"].map({0: "control", 1: "dysarthric"})
    ).astype(str)
    enriched["prompt_type"] = enriched["prompt_text"].map(_prompt_type_or_missing)
    enriched["system_id"] = [
        _system_id(record)
        for record in enriched[list(SYSTEM_COLUMNS)].to_dict(orient="records")
    ]
    if enriched.duplicated(["system_id", "sample_id"]).any():
        raise ValueError("A system has more than one out-of-fold decision for a sample")

    onset_audit = {
        "input_rows": 0,
        "completed_reference_rows": 0,
        "incomplete_reference_rows_excluded": 0,
    }
    if onset_table is not None:
        prepared_onset, onset_audit = _prepare_onset_table(onset_table)
        onset_keys = ["sample_id"]
        if "onset_variant" in prepared_onset.columns:
            onset_keys.append("onset_variant")
        rename = {
            column: f"audit_{column}"
            for column in prepared_onset.columns
            if column not in onset_keys
        }
        prepared_onset = prepared_onset.rename(columns=rename)
        enriched = enriched.merge(
            prepared_onset,
            how="left",
            on=onset_keys,
            validate="many_to_one",
        )

    prefix_detected = (
        enriched["prefix_onset_detected"].map(_as_optional_bool)
        if "prefix_onset_detected" in enriched.columns
        else pd.Series([None] * len(enriched), index=enriched.index, dtype=object)
    )
    audit_detected = (
        enriched["audit_onset_detected"].map(_as_optional_bool)
        if "audit_onset_detected" in enriched.columns
        else pd.Series([None] * len(enriched), index=enriched.index, dtype=object)
    )
    detected = pd.Series(
        [
            audit_value
            if audit_value is not None and not pd.isna(audit_value)
            else prefix_value
            for audit_value, prefix_value in zip(
                audit_detected, prefix_detected, strict=True
            )
        ],
        index=enriched.index,
        dtype="boolean",
    )
    enriched["onset_detected"] = detected
    enriched["onset_status"] = detected.map(
        lambda value: "unavailable" if value is None or pd.isna(value) else (
            "detected" if bool(value) else "failure"
        )
    )
    enriched = _add_bins(
        enriched,
        duration_cutpoints_sec=duration_cutpoints_sec,
        entropy_cutpoints=entropy_cutpoints,
    )
    enriched["error_type"] = np.select(
        [
            enriched["label_id"].eq(0) & enriched["predicted_label"].eq(1),
            enriched["label_id"].eq(1) & enriched["predicted_label"].eq(0),
        ],
        ["FP", "FN"],
        default="correct",
    )
    enriched = enriched.drop(columns=["manifest_speaker_id", "manifest_label_id"])
    return enriched.sort_values(["system_id", "sample_id"]).reset_index(drop=True), onset_audit


def _stratum_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _error_records(frame: pd.DataFrame, *, scope: str, system_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    groups: list[tuple[str, object, pd.DataFrame]] = [("overall", "all", frame)]
    for stratifier in STRATIFIERS:
        groups.extend(
            (stratifier, value, group)
            for value, group in frame.groupby(stratifier, dropna=False, sort=True, observed=True)
        )
    for stratifier, stratum, group in groups:
        for error_type, eligible_label, error_prediction, denominator_name in (
            ("FP", 0, 1, "actual_control_decisions"),
            ("FN", 1, 0, "actual_dysarthric_decisions"),
        ):
            eligible = group["label_id"].eq(eligible_label)
            errors = eligible & group["predicted_label"].eq(error_prediction)
            denominator = int(eligible.sum())
            error_count = int(errors.sum())
            records.append(
                {
                    "scope": scope,
                    "system_id": system_id,
                    "stratifier": stratifier,
                    "stratum": _stratum_value(stratum),
                    "error_type": error_type,
                    "error_count": error_count,
                    "eligible_denominator": denominator,
                    "error_rate": error_count / denominator if denominator else np.nan,
                    "correct_count": denominator - error_count,
                    "total_decisions_in_stratum": len(group),
                    "unique_samples_in_stratum": int(group["sample_id"].nunique()),
                    "systems_in_stratum": int(group["system_id"].nunique()),
                    "rate_denominator_definition": denominator_name,
                }
            )
    return records


def build_error_tables(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pooled-opportunity and per-system FP/FN stratum tables."""

    if enriched.empty:
        raise ValueError("Enriched decisions must not be empty")
    required = {
        "sample_id",
        "system_id",
        "label_id",
        "predicted_label",
        *STRATIFIERS,
    }
    _require_columns(enriched, frozenset(required), "Enriched decisions")
    overall = pd.DataFrame(_error_records(enriched, scope="all_systems", system_id="all"))
    per_system_records: list[dict[str, Any]] = []
    for system_id, group in enriched.groupby("system_id", sort=True):
        per_system_records.extend(
            _error_records(group, scope="per_system", system_id=str(system_id))
        )
    per_system = pd.DataFrame(per_system_records)
    return overall, per_system


def _default_primary_systems(enriched: pd.DataFrame) -> list[str]:
    preferred = enriched[enriched["policy_id"].astype(str).isin(
        {"complete-temperature_scaled", "complete-calibrated", "complete_calibrated"}
    )]
    source = preferred if not preferred.empty else enriched
    return sorted(source["system_id"].astype(str).unique())


def create_blinded_review_sample(
    enriched: pd.DataFrame,
    *,
    primary_system_ids: Sequence[str] | None = None,
    per_error: int = DEFAULT_REVIEW_PER_ERROR,
    seed: int = DEFAULT_REVIEW_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Select errors deterministically, then remove outcome/system fields from review rows."""

    if per_error < 1:
        raise ValueError("per_error must be positive")
    known = set(enriched["system_id"].astype(str))
    primary = (
        _default_primary_systems(enriched)
        if primary_system_ids is None
        else sorted({str(value) for value in primary_system_ids})
    )
    unknown = sorted(set(primary) - known)
    if unknown:
        raise ValueError(f"Unknown primary system_id values: {unknown}")

    selected_groups: list[pd.DataFrame] = []
    sampling: list[dict[str, Any]] = []
    for system_id in primary:
        system = enriched[enriched["system_id"].astype(str).eq(system_id)]
        for error_type in ("FP", "FN"):
            available = system[system["error_type"].eq(error_type)].copy()
            available["selection_hash"] = [
                _stable_digest(seed, system_id, error_type, run_id, sample_id)
                for run_id, sample_id in zip(
                    available["run_id"], available["sample_id"], strict=True
                )
            ]
            chosen = available.sort_values(["selection_hash", "sample_id"]).head(per_error)
            chosen = chosen.copy()
            chosen["review_item_id"] = [
                f"review-{digest[:16]}" for digest in chosen["selection_hash"]
            ]
            selected_groups.append(chosen)
            sampling.append(
                {
                    "system_id": system_id,
                    "error_type": error_type,
                    "available": len(available),
                    "requested": per_error,
                    "selected": len(chosen),
                }
            )
    selected = (
        pd.concat(selected_groups, ignore_index=True)
        if selected_groups
        else enriched.iloc[0:0].copy()
    )
    selected = selected.sort_values("review_item_id").reset_index(drop=True)

    review_columns = [
        "review_item_id",
        "sample_id",
        *(
            column
            for column in (
                "path",
                "recovered_path",
                "prompt_text",
                "prompt_type",
                "duration_sec",
                "decision_time_ms",
            )
            if column in selected.columns
        ),
    ]
    review = selected[review_columns].copy()
    for column in REVIEW_TAXONOMY_COLUMNS:
        review[column] = ""

    key_columns = [
        "review_item_id",
        "sample_id",
        "system_id",
        *SYSTEM_COLUMNS,
        "run_id",
        "error_type",
        "label_id",
        "predicted_label",
        "decision_time_ms",
        "calibrated_prob_control",
        "calibrated_prob_dysarthric",
        "calibrated_normalized_entropy",
        "selection_hash",
    ]
    key = selected[key_columns].copy()
    if any(review[column].astype(str).str.strip().ne("").any() for column in REVIEW_TAXONOMY_COLUMNS):
        raise AssertionError("Review taxonomy columns must be blank at selection time")
    return review, key, sampling, primary


def audit_classification_errors(
    prefix_ledger: pd.DataFrame,
    decision_ledger: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    onset_table: pd.DataFrame | None = None,
    primary_system_ids: Sequence[str] | None = None,
    review_per_error: int = DEFAULT_REVIEW_PER_ERROR,
    review_seed: int = DEFAULT_REVIEW_SEED,
    duration_cutpoints_sec: Sequence[float] = DEFAULT_DURATION_CUTPOINTS_SEC,
    entropy_cutpoints: Sequence[float] = DEFAULT_ENTROPY_CUTPOINTS,
) -> ErrorAudit:
    enriched, onset_audit = enrich_error_decisions(
        prefix_ledger,
        decision_ledger,
        manifest,
        onset_table=onset_table,
        duration_cutpoints_sec=duration_cutpoints_sec,
        entropy_cutpoints=entropy_cutpoints,
    )
    overall, per_system = build_error_tables(enriched)
    review, key, sampling, primary = create_blinded_review_sample(
        enriched,
        primary_system_ids=primary_system_ids,
        per_error=review_per_error,
        seed=review_seed,
    )
    summary: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decision_rows": len(enriched),
        "unique_samples": int(enriched["sample_id"].nunique()),
        "systems": int(enriched["system_id"].nunique()),
        "primary_system_ids": primary,
        "review_seed": review_seed,
        "review_per_error_requested": review_per_error,
        "review_sampling": sampling,
        "duration_cutpoints_sec": [float(value) for value in duration_cutpoints_sec],
        "entropy_cutpoints": [float(value) for value in entropy_cutpoints],
        "onset_table_audit": onset_audit,
        "rate_semantics": {
            "FP": "false positives / all actual-control decision opportunities in stratum",
            "FN": "false negatives / all actual-dysarthric decision opportunities in stratum",
            "all_systems_scope": (
                "pooled decision opportunities; the same utterance contributes once per system"
            ),
        },
        "qualitative_review_note": (
            "Selection is deterministic and precedes human taxonomy labeling. "
            "No error cause or category is inferred from metadata."
        ),
        "missing_prompt_policy": (
            "Nonblank prompts use classify_prompt_type from onset validation; missing prompts "
            "are retained as the explicit missing_prompt_text audit stratum."
        ),
    }
    return ErrorAudit(
        enriched_decisions=enriched,
        overall_errors=overall,
        per_system_errors=per_system,
        qualitative_review=review,
        qualitative_review_key=key,
        summary=summary,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def write_error_audit(
    audit: ErrorAudit,
    output_dir: str | Path,
    *,
    input_paths: Mapping[str, str | Path] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write CSV/JSON audit artifacts and a hash-bearing provenance manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "enriched_decisions": destination / "enriched_decisions.csv",
        "overall_errors": destination / "overall_error_strata.csv",
        "per_system_errors": destination / "per_system_error_strata.csv",
        "qualitative_review": destination / "qualitative_review_blinded.csv",
        "qualitative_review_key": destination / "qualitative_review_key.csv",
        "summary": destination / "error_audit_summary.json",
        "audit_manifest": destination / "audit_manifest.json",
    }
    for name, frame in (
        ("enriched_decisions", audit.enriched_decisions),
        ("overall_errors", audit.overall_errors),
        ("per_system_errors", audit.per_system_errors),
        ("qualitative_review", audit.qualitative_review),
        ("qualitative_review_key", audit.qualitative_review_key),
    ):
        _write_csv(frame, paths[name])
    paths["summary"].write_text(
        json.dumps(audit.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inputs = {
        name: {"path": str(Path(path)), "sha256": sha256_file(path)}
        for name, path in sorted((input_paths or {}).items())
    }
    output_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if name != "audit_manifest"
    }
    root = Path(project_root) if project_root is not None else Path.cwd()
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "configuration_sha256": sha256_json(
            {
                key: audit.summary[key]
                for key in (
                    "primary_system_ids",
                    "review_seed",
                    "review_per_error_requested",
                    "duration_cutpoints_sec",
                    "entropy_cutpoints",
                    "rate_semantics",
                )
            }
        ),
        "inputs": inputs,
        "outputs": output_hashes,
        "row_counts": {
            "enriched_decisions": len(audit.enriched_decisions),
            "overall_errors": len(audit.overall_errors),
            "per_system_errors": len(audit.per_system_errors),
            "qualitative_review": len(audit.qualitative_review),
            "qualitative_review_key": len(audit.qualitative_review_key),
        },
        "environment": environment_record(root),
        "causal_inference_performed": False,
    }
    paths["audit_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
