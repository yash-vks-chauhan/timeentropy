from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from timeentropy.evaluation.calibration import probabilities_from_logits
from timeentropy.evaluation.decisions import CANDIDATE_TIMES_MS, PREFIX_TIMES_MS
from timeentropy.evaluation.entropy import normalized_binary_entropy

REQUIRED_PREFIX_COLUMNS = frozenset(
    {
        "run_id",
        "sample_id",
        "speaker_id",
        "fold",
        "seed",
        "model_id",
        "split_role",
        "onset_variant",
        "pre_roll_ms",
        "prefix_ms",
        "label_id",
        "logit_control",
        "logit_dysarthric",
        "prob_control",
        "prob_dysarthric",
        "predicted_label",
        "normalized_entropy",
    }
)

CALIBRATED_PREFIX_COLUMNS = frozenset(
    {
        "temperature",
        "calibrated_prob_control",
        "calibrated_prob_dysarthric",
        "calibrated_predicted_label",
        "calibrated_normalized_entropy",
    }
)

REQUIRED_DECISION_COLUMNS = frozenset(
    {
        "run_id",
        "sample_id",
        "policy_id",
        "label_id",
        "decision_time_ms",
        "predicted_label",
        "forced_fallback",
        "accepted_at_endpoint",
    }
)


class LedgerInvariantError(ValueError):
    """Raised when a canonical prediction or decision ledger is not self-consistent."""


@dataclass(frozen=True)
class PrefixLedgerReport:
    rows: int
    samples: int
    runs: int
    test_samples: int
    calibrated: bool


@dataclass(frozen=True)
class DecisionLedgerReport:
    rows: int
    samples: int
    policies: int


def _require_columns(frame: pd.DataFrame, required: frozenset[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LedgerInvariantError(f"{name} is missing required columns: {missing}")


def _raise_errors(errors: list[str]) -> None:
    if errors:
        raise LedgerInvariantError("; ".join(errors))


def validate_prefix_ledger(
    frame: pd.DataFrame,
    *,
    expected_prefixes_ms: Sequence[int] = PREFIX_TIMES_MS,
    tolerance: float = 1.0e-8,
) -> PrefixLedgerReport:
    """Validate the canonical long-form prefix-prediction ledger.

    Raw probabilities must reproduce from raw logits.  If any calibrated column is
    present, all calibrated columns are required and must reproduce from the same logits
    divided by one positive, run-level temperature.
    """

    _require_columns(frame, REQUIRED_PREFIX_COLUMNS, "Prefix ledger")
    if frame.empty:
        raise LedgerInvariantError("Prefix ledger must not be empty")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    calibrated_columns_present = CALIBRATED_PREFIX_COLUMNS & set(frame.columns)
    if calibrated_columns_present and calibrated_columns_present != CALIBRATED_PREFIX_COLUMNS:
        missing = sorted(CALIBRATED_PREFIX_COLUMNS - calibrated_columns_present)
        raise LedgerInvariantError(f"Calibrated ledger columns are incomplete: {missing}")
    calibrated = bool(calibrated_columns_present)
    errors: list[str] = []

    key = ["run_id", "sample_id", "prefix_ms"]
    if frame.duplicated(key).any():
        errors.append("duplicate (run_id, sample_id, prefix_ms) rows")

    if not frame["label_id"].isin([0, 1]).all():
        errors.append("label_id contains values other than 0 and 1")
    if not frame["predicted_label"].isin([0, 1]).all():
        errors.append("predicted_label contains values other than 0 and 1")

    numeric_columns = [
        "logit_control",
        "logit_dysarthric",
        "prob_control",
        "prob_dysarthric",
        "normalized_entropy",
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        errors.append("raw logits, probabilities, or entropies contain non-finite values")
    else:
        logits = frame[["logit_control", "logit_dysarthric"]].to_numpy(dtype=np.float64)
        probabilities = frame[["prob_control", "prob_dysarthric"]].to_numpy(dtype=np.float64)
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            errors.append("raw probabilities lie outside [0, 1]")
        elif not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
            errors.append("raw probabilities do not sum to one")
        else:
            expected_probabilities = probabilities_from_logits(logits)
            if not np.allclose(
                probabilities,
                expected_probabilities,
                rtol=0.0,
                atol=tolerance,
            ):
                errors.append("raw probabilities do not reproduce from raw logits")
            expected_predictions = np.argmax(probabilities, axis=1)
            if not np.array_equal(
                expected_predictions,
                frame["predicted_label"].to_numpy(dtype=np.int64),
            ):
                errors.append("predicted_label does not equal the raw probability argmax")
            expected_entropy = normalized_binary_entropy(probabilities)
            if not np.allclose(
                expected_entropy,
                frame["normalized_entropy"].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=tolerance,
            ):
                errors.append("normalized_entropy does not reproduce from raw probabilities")

    expected_prefixes = frozenset(int(value) for value in expected_prefixes_ms)
    actual_prefixes = set(frame["prefix_ms"].astype(int))
    if not actual_prefixes.issubset(expected_prefixes):
        errors.append(
            f"unregistered prefix times found: {sorted(actual_prefixes - expected_prefixes)}"
        )
    incomplete: list[str] = []
    for (run_id, sample_id), group in frame.groupby(["run_id", "sample_id"], sort=False):
        if frozenset(group["prefix_ms"].astype(int)) != expected_prefixes:
            incomplete.append(f"{run_id}/{sample_id}")
            if len(incomplete) == 5:
                break
    if incomplete:
        errors.append(f"samples without exactly the registered prefixes: {incomplete}")

    for column in ("speaker_id", "label_id"):
        if frame.groupby("sample_id")[column].nunique().gt(1).any():
            errors.append(f"{column} is inconsistent across rows for at least one sample")

    run_constant_columns = ["model_id", "fold", "seed", "onset_variant", "pre_roll_ms"]
    for column in run_constant_columns:
        if frame.groupby("run_id")[column].nunique().gt(1).any():
            errors.append(f"{column} is not constant within run_id")

    if calibrated:
        temperatures = frame["temperature"].to_numpy(dtype=np.float64)
        calibrated_numeric = frame[
            [
                "calibrated_prob_control",
                "calibrated_prob_dysarthric",
                "calibrated_normalized_entropy",
            ]
        ].to_numpy(dtype=np.float64)
        if (
            not np.all(np.isfinite(temperatures))
            or np.any(temperatures <= 0.0)
            or not np.all(np.isfinite(calibrated_numeric))
        ):
            errors.append("calibrated values or temperatures are invalid")
        else:
            if frame.groupby("run_id")["temperature"].nunique().gt(1).any():
                errors.append("temperature is not constant within run_id")
            logits = frame[["logit_control", "logit_dysarthric"]].to_numpy(dtype=np.float64)
            scaled_logits = logits / temperatures[:, None]
            expected = probabilities_from_logits(scaled_logits)
            stored = frame[["calibrated_prob_control", "calibrated_prob_dysarthric"]].to_numpy(
                dtype=np.float64
            )
            if not np.allclose(stored, expected, rtol=0.0, atol=tolerance):
                errors.append("calibrated probabilities do not reproduce from logits/temperature")
            elif not np.array_equal(
                np.argmax(stored, axis=1),
                frame["calibrated_predicted_label"].to_numpy(dtype=np.int64),
            ):
                errors.append("calibrated_predicted_label does not equal probability argmax")
            elif not np.allclose(
                normalized_binary_entropy(stored),
                frame["calibrated_normalized_entropy"].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=tolerance,
            ):
                errors.append("calibrated entropy does not reproduce from probabilities")

    _raise_errors(errors)
    test_rows = frame.loc[frame["split_role"] == "test", "sample_id"]
    return PrefixLedgerReport(
        rows=len(frame),
        samples=int(frame["sample_id"].nunique()),
        runs=int(frame["run_id"].nunique()),
        test_samples=int(test_rows.nunique()),
        calibrated=calibrated,
    )


def validate_oof_test_coverage(
    frame: pd.DataFrame,
    expected_sample_ids: Iterable[str],
    *,
    experiment_columns: Sequence[str] = (
        "model_id",
        "seed",
        "onset_variant",
        "pre_roll_ms",
    ),
) -> None:
    """Require complete, exactly-once out-of-fold test coverage per experiment."""

    _require_columns(frame, REQUIRED_PREFIX_COLUMNS, "Prefix ledger")
    expected = frozenset(str(value) for value in expected_sample_ids)
    if not expected:
        raise ValueError("expected_sample_ids must not be empty")
    test = frame[frame["split_role"] == "test"].copy()
    if test.empty:
        raise LedgerInvariantError("Prefix ledger has no test rows")

    errors: list[str] = []
    for experiment, group in test.groupby(list(experiment_columns), dropna=False, sort=False):
        observed = frozenset(group["sample_id"].astype(str))
        if observed != expected:
            errors.append(
                f"experiment {experiment} coverage differs: "
                f"missing={len(expected - observed)}, extra={len(observed - expected)}"
            )
        fold_counts = group.groupby("sample_id")["fold"].nunique()
        if not fold_counts.eq(1).all():
            errors.append(f"experiment {experiment} tests at least one sample in multiple folds")
        run_counts = group.groupby("sample_id")["run_id"].nunique()
        if not run_counts.eq(1).all():
            errors.append(f"experiment {experiment} has multiple test runs for at least one sample")
        row_counts = group.groupby("sample_id").size()
        prefix_counts = group.groupby("sample_id")["prefix_ms"].nunique()
        if (
            not row_counts.eq(len(PREFIX_TIMES_MS)).all()
            or not prefix_counts.eq(len(PREFIX_TIMES_MS)).all()
        ):
            errors.append(
                f"experiment {experiment} does not have exactly one row per registered prefix"
            )
    _raise_errors(errors)


def validate_decision_ledger(
    decisions: pd.DataFrame,
    *,
    prefix_ledger: pd.DataFrame | None = None,
    prefix_prediction_column: str = "calibrated_predicted_label",
) -> DecisionLedgerReport:
    """Validate decision times, flags, uniqueness, and optional prefix-label linkage."""

    _require_columns(decisions, REQUIRED_DECISION_COLUMNS, "Decision ledger")
    if decisions.empty:
        raise LedgerInvariantError("Decision ledger must not be empty")
    errors: list[str] = []
    key = ["run_id", "sample_id", "policy_id"]
    if decisions.duplicated(key).any():
        errors.append("duplicate (run_id, sample_id, policy_id) decision rows")
    if not decisions["label_id"].isin([0, 1]).all():
        errors.append("decision label_id contains values other than 0 and 1")
    if not decisions["predicted_label"].isin([0, 1]).all():
        errors.append("decision predicted_label contains values other than 0 and 1")
    if not decisions["decision_time_ms"].isin(CANDIDATE_TIMES_MS).all():
        errors.append("decision_time_ms contains an ineligible time")
    forced = decisions["forced_fallback"].astype(bool)
    endpoint = decisions["accepted_at_endpoint"].astype(bool)
    if (forced & endpoint).any():
        errors.append("a decision cannot be both forced fallback and accepted at endpoint")
    if (forced & decisions["decision_time_ms"].ne(CANDIDATE_TIMES_MS[-1])).any():
        errors.append("forced fallbacks must occur at 300 ms")

    if prefix_ledger is not None:
        _require_columns(prefix_ledger, REQUIRED_PREFIX_COLUMNS, "Prefix ledger")
        if prefix_prediction_column not in prefix_ledger.columns:
            raise LedgerInvariantError(
                f"Prefix ledger lacks prediction column {prefix_prediction_column!r}"
            )
        lookup = prefix_ledger[
            ["run_id", "sample_id", "prefix_ms", prefix_prediction_column]
        ].rename(
            columns={
                "prefix_ms": "decision_time_ms",
                prefix_prediction_column: "prefix_predicted_label",
            }
        )
        merged = decisions.merge(
            lookup,
            how="left",
            on=["run_id", "sample_id", "decision_time_ms"],
            validate="many_to_one",
        )
        if merged["prefix_predicted_label"].isna().any():
            errors.append("at least one decision has no corresponding prefix prediction")
        elif not np.array_equal(
            merged["predicted_label"].to_numpy(dtype=np.int64),
            merged["prefix_predicted_label"].to_numpy(dtype=np.int64),
        ):
            errors.append("decision labels do not match their selected prefix predictions")

    _raise_errors(errors)
    return DecisionLedgerReport(
        rows=len(decisions),
        samples=int(decisions["sample_id"].nunique()),
        policies=int(decisions["policy_id"].nunique()),
    )
