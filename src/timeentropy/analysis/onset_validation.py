from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from timeentropy.data.audio import load_audio
from timeentropy.signal.onset import OnsetConfig, OnsetMode, detect_onset

ANNOTATION_SCHEMA_VERSION = "onset-annotation-v1"
PROMPT_TYPES = ("vocal_exercise", "isolated_token", "sentence")
ONSET_MODES: tuple[OnsetMode, ...] = (
    "causal_fixed",
    "causal_noise_floor",
    "recovered_offline_peak",
)
PERTURBATION_OFFSETS_MS = (-50, -30, -10, 0, 10, 30, 50)
DEFAULT_SELECTION_SEED = 20260819
DEFAULT_BOOTSTRAP_SEED = 20260819

ANNOTATION_COLUMNS = (
    "annotation_schema_version",
    "selection_seed",
    "selection_rank_within_speaker",
    "sample_id",
    "speaker_id",
    "diagnosis",
    "label_id",
    "session",
    "utterance_id",
    "prompt_text",
    "prompt_type",
    "path",
    "duration_sec",
    "sample_rate",
    "reference_onset_sec",
    "annotator",
    "annotation_status",
    "notes",
)

ERROR_METRICS = (
    "failure_rate",
    "signed_bias_ms",
    "mae_ms",
    "median_absolute_error_ms",
    "p90_absolute_error_ms",
    "p95_absolute_error_ms",
    "within_20_ms_rate",
    "within_50_ms_rate",
)
ESTIMATE_COLUMNS = (
    "sample_id",
    "onset_mode",
    "detected",
    "estimated_onset_sec",
    "estimated_onset_sample",
    "confirmation_sample",
    "threshold",
    "analysis_sample_rate",
    "source_sample_rate",
    "source_channels",
)
METRIC_COLUMNS = (
    "group_dimension",
    "group_value",
    "onset_mode",
    "speaker_clusters",
    "reference_count",
    "detected_count",
    "failure_count",
    *ERROR_METRICS,
    *(f"{name}_{bound}" for name in ERROR_METRICS for bound in ("ci_low", "ci_high")),
)


@dataclass(frozen=True)
class PerturbedOnset:
    requested_offset_ms: int
    requested_onset_sample: int
    clipped_onset_sample: int
    applied_offset_ms: float
    clipped: bool


@dataclass(frozen=True)
class ReferenceAudit:
    total_rows: int
    completed_rows: int
    incomplete_rows: int
    excluded_rows: int
    invalid_rows: int


@dataclass(frozen=True)
class OnsetEvaluation:
    metrics: pd.DataFrame
    reference_audit: ReferenceAudit
    estimates: pd.DataFrame
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float


def classify_prompt_type(prompt_text: object) -> Literal[
    "vocal_exercise", "isolated_token", "sentence"
]:
    """Derive one transparent TORGO prompt stratum from prompt text.

    A prompt wholly enclosed in square brackets is an elicited vocal/oral
    exercise. For other prompts, bracketed or parenthesized pronunciation
    clarifiers are removed before counting spoken alphanumeric tokens. One or
    zero counted tokens is an isolated token; two or more form a sentence.
    """

    if prompt_text is None or (isinstance(prompt_text, float) and math.isnan(prompt_text)):
        raise ValueError("prompt_text must not be missing")
    normalized = " ".join(str(prompt_text).split())
    if not normalized:
        raise ValueError("prompt_text must not be blank")
    if re.fullmatch(r"\[.*\]", normalized):
        return "vocal_exercise"
    spoken = re.sub(r"\[[^\]]*\]|\([^\)]*\)", " ", normalized)
    tokens = re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", spoken)
    return "isolated_token" if len(tokens) <= 1 else "sentence"


def _stable_digest(seed: int, *parts: object) -> str:
    material = "\0".join((str(seed), *(str(part) for part in parts))).encode()
    return hashlib.sha256(material).hexdigest()


def _prompt_quotas(available: Mapping[str, int], total: int) -> dict[str, int]:
    """Allocate nearly equal prompt strata, redistributing shortages evenly."""

    if total < 1:
        raise ValueError("per-speaker annotation count must be positive")
    if sum(int(available.get(name, 0)) for name in PROMPT_TYPES) < total:
        raise ValueError(f"Fewer than {total} eligible samples are available")

    # For the default total of 20 this gives 6 vocal exercises, 7 isolated
    # tokens, and 7 sentences. The difference between targets is at most one.
    base, remainder = divmod(total, len(PROMPT_TYPES))
    desired = {
        "vocal_exercise": base,
        "isolated_token": base + (1 if remainder >= 1 else 0),
        "sentence": base + (1 if remainder >= 2 else 0),
    }
    quotas = {
        name: min(int(available.get(name, 0)), desired[name]) for name in PROMPT_TYPES
    }
    while sum(quotas.values()) < total:
        candidates = [
            name for name in PROMPT_TYPES if quotas[name] < int(available.get(name, 0))
        ]
        if not candidates:
            raise RuntimeError("Prompt quota redistribution exhausted available samples")
        selected = min(candidates, key=lambda name: (quotas[name], PROMPT_TYPES.index(name)))
        quotas[selected] += 1
    return quotas


def create_annotation_sample(
    manifest: pd.DataFrame,
    *,
    per_speaker: int = 20,
    expected_speakers: int = 15,
    seed: int = DEFAULT_SELECTION_SEED,
) -> pd.DataFrame:
    """Create a deterministic, speaker-balanced sheet with blank manual labels."""

    required = {"sample_id", "speaker_id", "prompt_text", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing annotation columns: {sorted(missing)}")
    if manifest.empty or manifest["sample_id"].astype(str).duplicated().any():
        raise ValueError("Manifest must be nonempty with unique sample_id values")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    speakers = sorted(manifest["speaker_id"].astype(str).unique())
    if len(speakers) != expected_speakers:
        raise ValueError(
            f"Expected {expected_speakers} speakers for the annotation sample, "
            f"found {len(speakers)}"
        )

    prompt_eligible = manifest["prompt_text"].notna() & manifest[
        "prompt_text"
    ].astype(str).str.strip().ne("")
    frame = manifest.loc[prompt_eligible].copy()
    if set(frame["speaker_id"].astype(str)) != set(speakers):
        raise ValueError("At least one speaker has no nonblank prompt text")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["speaker_id"] = frame["speaker_id"].astype(str)
    frame["prompt_type"] = frame["prompt_text"].map(classify_prompt_type)
    selected_parts: list[pd.DataFrame] = []
    for speaker_id in speakers:
        speaker = frame.loc[frame["speaker_id"].eq(speaker_id)].copy()
        available = speaker["prompt_type"].value_counts().to_dict()
        quotas = _prompt_quotas(available, per_speaker)
        for prompt_type in PROMPT_TYPES:
            stratum = speaker.loc[speaker["prompt_type"].eq(prompt_type)].copy()
            stratum["_selection_hash"] = stratum["sample_id"].map(
                lambda sample_id, speaker=speaker_id, prompt=prompt_type: _stable_digest(
                    seed, speaker, prompt, sample_id
                )
            )
            stratum = stratum.sort_values(
                ["_selection_hash", "sample_id"], kind="mergesort"
            ).head(quotas[prompt_type])
            selected_parts.append(stratum)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected["_prompt_order"] = selected["prompt_type"].map(
        {name: index for index, name in enumerate(PROMPT_TYPES)}
    )
    selected = selected.sort_values(
        ["speaker_id", "_prompt_order", "_selection_hash", "sample_id"],
        kind="mergesort",
        ignore_index=True,
    )
    selected["selection_rank_within_speaker"] = (
        selected.groupby("speaker_id", sort=False).cumcount() + 1
    )
    if len(selected) != expected_speakers * per_speaker:
        raise RuntimeError("Annotation sampler produced an unexpected total")
    if not selected.groupby("speaker_id").size().eq(per_speaker).all():
        raise RuntimeError("Annotation sampler did not preserve exact speaker balance")

    diagnosis = _normalize_diagnosis(selected)
    if "label_id" in selected.columns:
        label_id = selected["label_id"]
    else:
        label_id = diagnosis.map({"control": 0, "dysarthric": 1})
    if label_id.isna().any():
        raise ValueError("Diagnosis values cannot be converted to binary label_id")

    def optional(column: str, default: object = "") -> pd.Series:
        if column in selected.columns:
            return selected[column]
        return pd.Series([default] * len(selected), index=selected.index)

    sheet = pd.DataFrame(
        {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "selection_seed": int(seed),
            "selection_rank_within_speaker": selected[
                "selection_rank_within_speaker"
            ].astype(int),
            "sample_id": selected["sample_id"],
            "speaker_id": selected["speaker_id"],
            "diagnosis": diagnosis,
            "label_id": label_id.astype(int),
            "session": optional("session"),
            "utterance_id": optional("utterance_id"),
            "prompt_text": selected["prompt_text"],
            "prompt_type": selected["prompt_type"],
            "path": selected["path"].astype(str),
            "duration_sec": optional("duration_sec", np.nan),
            "sample_rate": optional("sample_rate", np.nan),
            # These fields are intentionally blank. Only a human annotation
            # workflow may populate them.
            "reference_onset_sec": "",
            "annotator": "",
            "annotation_status": "",
            "notes": "",
        },
        columns=ANNOTATION_COLUMNS,
    )
    return sheet


def annotation_schema() -> dict[str, Any]:
    """Return a machine-readable schema and the prompt-stratification rules."""

    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "reference_time_unit": "seconds from the beginning of the decoded/resampled file",
        "manual_reference_policy": (
            "reference_onset_sec, annotator, annotation_status, and notes are blank on "
            "creation; the software never imputes a manual onset"
        ),
        "completed_reference_rule": (
            "a finite non-negative reference_onset_sec not marked excluded/invalid"
        ),
        "prompt_type_rules": {
            "vocal_exercise": "the entire normalized prompt is enclosed in square brackets",
            "isolated_token": (
                "after removing bracketed/parenthesized clarifiers, at most one "
                "alphanumeric spoken token remains"
            ),
            "sentence": "after clarifier removal, at least two spoken tokens remain",
        },
        "default_prompt_targets_per_speaker": {
            "vocal_exercise": 6,
            "isolated_token": 7,
            "sentence": 7,
        },
        "shortage_rule": (
            "take all available rows in a deficient stratum and redistribute the deficit "
            "as evenly as possible among strata with remaining samples"
        ),
        "prompt_eligibility_rule": (
            "rows with missing or blank prompt_text are excluded rather than assigned an "
            "invented prompt type"
        ),
        "columns": {
            "annotation_schema_version": "constant schema identifier",
            "selection_seed": "integer SHA-256 selection seed",
            "selection_rank_within_speaker": "stable one-based row rank",
            "sample_id": "unique utterance identifier",
            "speaker_id": "TORGO speaker code",
            "diagnosis": "control or dysarthric corpus label",
            "label_id": "0 control, 1 dysarthric",
            "session": "corpus session identifier",
            "utterance_id": "utterance identifier within session/channel",
            "prompt_text": "preserved corpus prompt",
            "prompt_type": "derived three-level annotation stratum",
            "path": "audio path used for detector evaluation",
            "duration_sec": "indexed file duration in seconds",
            "sample_rate": "indexed source sample rate",
            "reference_onset_sec": "manual acoustic onset in seconds; initially blank",
            "annotator": "human annotator identifier; initially blank",
            "annotation_status": "optional workflow status; initially blank",
            "notes": "free-form annotation notes; initially blank",
        },
    }


def write_annotation_sheet(
    sheet: pd.DataFrame,
    output_csv: str | Path,
    schema_json: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write an annotation CSV and its schema sidecar."""

    missing = set(ANNOTATION_COLUMNS) - set(sheet.columns)
    if missing:
        raise ValueError(f"Annotation sheet is missing columns: {sorted(missing)}")
    csv_path = Path(output_csv)
    schema_path = (
        Path(schema_json)
        if schema_json is not None
        else csv_path.with_suffix(".schema.json")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.loc[:, ANNOTATION_COLUMNS].to_csv(csv_path, index=False)
    schema_path.write_text(
        json.dumps(annotation_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, schema_path


def onset_perturbation_schedule(
    onset_sample: int,
    *,
    num_samples: int,
    sample_rate: int,
    offsets_ms: Sequence[int] = PERTURBATION_OFFSETS_MS,
) -> tuple[PerturbedOnset, ...]:
    """Apply the registered onset shifts and clip each to waveform boundaries."""

    if num_samples < 1 or sample_rate <= 0:
        raise ValueError("num_samples and sample_rate must be positive")
    if onset_sample < 0 or onset_sample >= num_samples:
        raise ValueError("onset_sample must lie inside the waveform")
    results: list[PerturbedOnset] = []
    for raw_offset in offsets_ms:
        offset = int(raw_offset)
        requested = onset_sample + round(offset * sample_rate / 1000.0)
        clipped = min(max(requested, 0), num_samples - 1)
        results.append(
            PerturbedOnset(
                requested_offset_ms=offset,
                requested_onset_sample=requested,
                clipped_onset_sample=clipped,
                applied_offset_ms=(clipped - onset_sample) * 1000.0 / sample_rate,
                clipped=clipped != requested,
            )
        )
    return tuple(results)


def _normalize_diagnosis(frame: pd.DataFrame) -> pd.Series:
    if "diagnosis" in frame.columns:
        values = frame["diagnosis"]
    elif "label" in frame.columns:
        values = frame["label"]
    elif "label_id" in frame.columns:
        values = frame["label_id"].map({0: "control", 1: "dysarthric"})
    else:
        raise ValueError("Annotations require diagnosis, label, or label_id")
    missing = values.isna() | values.astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError("Diagnosis values must not be blank")
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"control", "dysarthric"}).all():
        unknown = sorted(set(normalized) - {"control", "dysarthric"})
        raise ValueError(f"Unknown diagnosis values: {unknown}")
    return normalized


def completed_references(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, ReferenceAudit]:
    """Select only genuine completed manual references and audit every exclusion."""

    required = {"sample_id", "speaker_id", "path", "reference_onset_sec"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotations are missing columns: {sorted(missing)}")
    if annotations["sample_id"].astype(str).duplicated().any():
        raise ValueError("Annotation sample_id values must be unique")

    frame = annotations.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["speaker_id"].isna().any() or frame["speaker_id"].astype(str).str.strip().eq("").any():
        raise ValueError("speaker_id values must not be blank")
    frame["speaker_id"] = frame["speaker_id"].astype(str)
    frame["diagnosis"] = _normalize_diagnosis(frame)
    if "prompt_type" not in frame.columns:
        if "prompt_text" not in frame.columns:
            raise ValueError("Annotations require prompt_type or prompt_text")
        frame["prompt_type"] = frame["prompt_text"].map(classify_prompt_type)
    else:
        missing_prompt = frame["prompt_type"].isna() | frame["prompt_type"].astype(str).str.strip().eq("")
        if missing_prompt.any():
            if "prompt_text" not in frame.columns:
                raise ValueError("Blank prompt_type rows require prompt_text")
            frame.loc[missing_prompt, "prompt_type"] = frame.loc[
                missing_prompt, "prompt_text"
            ].map(classify_prompt_type)
    frame["prompt_type"] = frame["prompt_type"].astype(str).str.strip()
    if not frame["prompt_type"].isin(PROMPT_TYPES).all():
        unknown = sorted(set(frame["prompt_type"]) - set(PROMPT_TYPES))
        raise ValueError(f"Unknown prompt_type values: {unknown}")

    reference = pd.to_numeric(frame["reference_onset_sec"], errors="coerce")
    status = frame.get("annotation_status", pd.Series("", index=frame.index))
    normalized_status = status.fillna("").astype(str).str.strip().str.lower()
    explicitly_excluded = normalized_status.isin(
        {"exclude", "excluded", "invalid", "reject", "rejected"}
    )
    blank_reference = reference.isna()
    invalid_reference = reference.notna() & (
        ~np.isfinite(reference.to_numpy(dtype=np.float64)) | reference.lt(0.0)
    )
    if "duration_sec" in frame.columns:
        duration = pd.to_numeric(frame["duration_sec"], errors="coerce")
        invalid_reference |= reference.notna() & duration.notna() & reference.gt(duration)

    complete = ~(explicitly_excluded | blank_reference | invalid_reference)
    selected = frame.loc[complete].copy()
    selected["reference_onset_sec"] = reference.loc[complete].astype(float)
    audit = ReferenceAudit(
        total_rows=len(frame),
        completed_rows=int(complete.sum()),
        incomplete_rows=int((blank_reference & ~explicitly_excluded).sum()),
        excluded_rows=int(explicitly_excluded.sum()),
        invalid_rows=int((invalid_reference & ~explicitly_excluded).sum()),
    )
    categorized = (
        audit.completed_rows
        + audit.incomplete_rows
        + audit.excluded_rows
        + audit.invalid_rows
    )
    if categorized != len(frame):
        raise RuntimeError("Reference audit categories are not exhaustive")
    return selected, audit


def asdict_reference_audit(audit: ReferenceAudit) -> dict[str, int]:
    return {
        "total_rows": audit.total_rows,
        "completed_rows": audit.completed_rows,
        "incomplete_rows": audit.incomplete_rows,
        "excluded_rows": audit.excluded_rows,
        "invalid_rows": audit.invalid_rows,
    }


def run_onset_estimators(
    references: pd.DataFrame,
    *,
    onset_config: OnsetConfig | None = None,
    modes: Sequence[OnsetMode] = ONSET_MODES,
) -> pd.DataFrame:
    """Run registered onset detectors for rows with completed references."""

    config = onset_config or OnsetConfig()
    required = {"sample_id", "path"}
    missing = required - set(references.columns)
    if missing:
        raise ValueError(f"References are missing estimator columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for record in references.to_dict(orient="records"):
        loaded = load_audio(
            record["path"], target_sample_rate=config.sample_rate, mono=True
        )
        for mode in modes:
            result = detect_onset(loaded.waveform, config=config, mode=mode)
            rows.append(
                {
                    "sample_id": str(record["sample_id"]),
                    "onset_mode": mode,
                    "detected": result.detected,
                    "estimated_onset_sec": (
                        np.nan
                        if result.onset_sample is None
                        else result.onset_sample / config.sample_rate
                    ),
                    "estimated_onset_sample": (
                        -1 if result.onset_sample is None else result.onset_sample
                    ),
                    "confirmation_sample": (
                        -1
                        if result.confirmation_sample is None
                        else result.confirmation_sample
                    ),
                    "threshold": result.threshold,
                    "analysis_sample_rate": config.sample_rate,
                    "source_sample_rate": loaded.source_sample_rate,
                    "source_channels": loaded.source_channels,
                }
            )
    return pd.DataFrame(rows, columns=ESTIMATE_COLUMNS)


def _parse_detected(values: pd.Series, estimates: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.copy()
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    parsed = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "detected": True,
            "false": False,
            "0": False,
            "no": False,
            "failed": False,
            "": np.nan,
        }
    )
    unknown = parsed.isna() & normalized.ne("")
    if unknown.any():
        raise ValueError(f"Unknown detected values: {sorted(normalized.loc[unknown].unique())}")
    parsed = parsed.astype("boolean")
    inferred = pd.Series(estimates.notna().to_numpy(), index=values.index, dtype="boolean")
    return parsed.fillna(inferred).astype(bool)


def normalize_saved_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    """Normalize long or `<mode>_onset_sec` wide saved estimates."""

    if "sample_id" not in estimates.columns:
        raise ValueError("Saved estimates require sample_id")
    if {"onset_mode", "estimated_onset_sec"}.issubset(estimates.columns):
        result = estimates.copy()
    else:
        rows: list[pd.DataFrame] = []
        for mode in ONSET_MODES:
            estimate_column = f"{mode}_onset_sec"
            if estimate_column not in estimates.columns:
                raise ValueError(
                    "Wide saved estimates require one `<mode>_onset_sec` column per mode"
                )
            columns = {"sample_id": estimates["sample_id"], "onset_mode": mode}
            columns["estimated_onset_sec"] = estimates[estimate_column]
            detected_column = f"{mode}_detected"
            if detected_column in estimates.columns:
                columns["detected"] = estimates[detected_column]
            rows.append(pd.DataFrame(columns))
        result = pd.concat(rows, ignore_index=True)

    result["sample_id"] = result["sample_id"].astype(str)
    result["onset_mode"] = result["onset_mode"].astype(str)
    if not result["onset_mode"].isin(ONSET_MODES).all():
        unknown = sorted(set(result["onset_mode"]) - set(ONSET_MODES))
        raise ValueError(f"Saved estimates contain unknown onset modes: {unknown}")
    if result.duplicated(["sample_id", "onset_mode"]).any():
        raise ValueError("Saved estimates contain duplicate sample/mode rows")
    result["estimated_onset_sec"] = pd.to_numeric(
        result["estimated_onset_sec"], errors="coerce"
    )
    detected_source = (
        result["detected"]
        if "detected" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    result["detected"] = _parse_detected(
        detected_source, result["estimated_onset_sec"]
    )
    inconsistent = result["detected"] & result["estimated_onset_sec"].isna()
    if inconsistent.any():
        raise ValueError("A detected saved estimate cannot have a blank onset time")
    stale_failure = ~result["detected"] & result["estimated_onset_sec"].notna()
    if stale_failure.any():
        raise ValueError("A failed saved estimate cannot retain an onset time")
    invalid = result["estimated_onset_sec"].notna() & (
        ~np.isfinite(result["estimated_onset_sec"].to_numpy(dtype=np.float64))
        | result["estimated_onset_sec"].lt(0.0)
    )
    if invalid.any():
        raise ValueError("Saved onset estimates must be finite and non-negative")
    return result


def _point_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    references = len(frame)
    detected = frame["detected"].to_numpy(dtype=bool)
    successful = frame.loc[detected]
    errors = successful["error_ms"].to_numpy(dtype=np.float64)
    absolute = np.abs(errors)
    values: dict[str, float | int] = {
        "reference_count": references,
        "detected_count": int(detected.sum()),
        "failure_count": int((~detected).sum()),
        "failure_rate": float((~detected).mean()) if references else float("nan"),
        "signed_bias_ms": float(errors.mean()) if len(errors) else float("nan"),
        "mae_ms": float(absolute.mean()) if len(absolute) else float("nan"),
        "median_absolute_error_ms": (
            float(np.median(absolute)) if len(absolute) else float("nan")
        ),
        "p90_absolute_error_ms": (
            float(np.quantile(absolute, 0.90)) if len(absolute) else float("nan")
        ),
        "p95_absolute_error_ms": (
            float(np.quantile(absolute, 0.95)) if len(absolute) else float("nan")
        ),
        # Detector failures count as outside tolerance rather than silently
        # disappearing from the denominator.
        "within_20_ms_rate": (
            float(np.count_nonzero(absolute <= 20.0) / references)
            if references
            else float("nan")
        ),
        "within_50_ms_rate": (
            float(np.count_nonzero(absolute <= 50.0) / references)
            if references
            else float("nan")
        ),
    }
    return values


def _bootstrap_seed(base_seed: int, *parts: object) -> int:
    digest = _stable_digest(base_seed, *parts)
    return int(digest[:16], 16) % (2**63 - 1)


def _cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    seed: int,
    replicates: int,
    confidence_level: float,
) -> dict[str, tuple[float, float]]:
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    speakers = sorted(frame["speaker_id"].astype(str).unique())
    if not speakers:
        return {name: (float("nan"), float("nan")) for name in ERROR_METRICS}
    clusters = {
        speaker: frame.loc[frame["speaker_id"].astype(str).eq(speaker)]
        for speaker in speakers
    }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in ERROR_METRICS}
    for _ in range(replicates):
        chosen = rng.choice(speakers, size=len(speakers), replace=True)
        replicate = pd.concat([clusters[str(speaker)] for speaker in chosen], ignore_index=True)
        metrics = _point_metrics(replicate)
        for name in ERROR_METRICS:
            samples[name].append(float(metrics[name]))
    tail = (1.0 - confidence_level) / 2.0
    intervals: dict[str, tuple[float, float]] = {}
    for name, values in samples.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            intervals[name] = (
                float(np.quantile(finite, tail)),
                float(np.quantile(finite, 1.0 - tail)),
            )
        else:
            intervals[name] = (float("nan"), float("nan"))
    return intervals


def evaluate_onset_annotations(
    annotations: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = 2_000,
    confidence_level: float = 0.95,
) -> OnsetEvaluation:
    """Evaluate all modes overall and by diagnosis, speaker, and prompt type."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    references, audit = completed_references(annotations)
    normalized = normalize_saved_estimates(estimates)
    if references.empty:
        empty = pd.DataFrame(columns=METRIC_COLUMNS)
        return OnsetEvaluation(
            metrics=empty,
            reference_audit=audit,
            estimates=normalized,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )

    expected = pd.MultiIndex.from_product(
        [references["sample_id"].astype(str), ONSET_MODES],
        names=["sample_id", "onset_mode"],
    )
    observed = pd.MultiIndex.from_frame(normalized[["sample_id", "onset_mode"]])
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"Saved estimates lack {len(missing)} completed sample/mode rows")
    selected_estimates = normalized.set_index(["sample_id", "onset_mode"]).loc[
        expected
    ].reset_index()
    joined = selected_estimates.merge(
        references[
            [
                "sample_id",
                "speaker_id",
                "diagnosis",
                "prompt_type",
                "reference_onset_sec",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    joined["error_ms"] = (
        joined["estimated_onset_sec"] - joined["reference_onset_sec"]
    ) * 1000.0
    joined.loc[~joined["detected"], "error_ms"] = np.nan

    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", joined)]
    for dimension, column in (
        ("diagnosis", "diagnosis"),
        ("speaker", "speaker_id"),
        ("prompt_type", "prompt_type"),
    ):
        for value in sorted(joined[column].astype(str).unique()):
            groups.append((dimension, value, joined.loc[joined[column].astype(str).eq(value)]))

    rows: list[dict[str, Any]] = []
    for dimension, value, group in groups:
        for mode in ONSET_MODES:
            mode_frame = group.loc[group["onset_mode"].eq(mode)].copy()
            point = _point_metrics(mode_frame)
            ci = _cluster_bootstrap_intervals(
                mode_frame,
                seed=_bootstrap_seed(bootstrap_seed, dimension, value, mode),
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
            )
            row: dict[str, Any] = {
                "group_dimension": dimension,
                "group_value": value,
                "onset_mode": mode,
                "speaker_clusters": int(mode_frame["speaker_id"].nunique()),
                **point,
            }
            for name in ERROR_METRICS:
                row[f"{name}_ci_low"] = ci[name][0]
                row[f"{name}_ci_high"] = ci[name][1]
            rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(
        ["group_dimension", "group_value", "onset_mode"], ignore_index=True
    )
    return OnsetEvaluation(
        metrics=metrics,
        reference_audit=audit,
        estimates=selected_estimates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def write_evaluation_outputs(
    evaluation: OnsetEvaluation,
    metrics_csv: str | Path,
    summary_json: str | Path,
) -> tuple[Path, Path]:
    """Write the long metrics table and a JSON audit/metrics record."""

    csv_path = Path(metrics_csv)
    json_path = Path(summary_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.metrics.to_csv(csv_path, index=False)
    summary = {
        "schema_version": "onset-evaluation-v1",
        "reference_audit": asdict_reference_audit(evaluation.reference_audit),
        "onset_modes": list(ONSET_MODES),
        "bootstrap": {
            "unit": "speaker",
            "seed": evaluation.bootstrap_seed,
            "replicates": evaluation.bootstrap_replicates,
            "confidence_level": evaluation.confidence_level,
        },
        "within_tolerance_denominator": (
            "all completed references; detector failures count as outside tolerance"
        ),
        "metrics": evaluation.metrics.to_dict(orient="records"),
    }
    json_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path
