from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from timeentropy.evaluation.decisions import DecisionBatch


@dataclass(frozen=True)
class BinaryMetrics:
    samples: int
    true_control: int
    false_dysarthric: int
    false_control: int
    true_dysarthric: int
    accuracy: float
    macro_f1: float
    dysarthric_recall: float
    specificity: float
    dysarthric_precision: float
    balanced_accuracy: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class EarlyMetrics:
    classification: BinaryMetrics
    mean_latency_ms: float
    forced_fallback_rate: float
    consumed_fraction: float
    timeliness: float
    accuracy_earliness_harmonic_mean: float
    macro_f1_earliness_harmonic_mean: float

    def as_dict(self) -> dict[str, int | float]:
        result = self.classification.as_dict()
        result.update(
            {
                "mean_latency_ms": self.mean_latency_ms,
                "forced_fallback_rate": self.forced_fallback_rate,
                "consumed_fraction": self.consumed_fraction,
                "timeliness": self.timeliness,
                "accuracy_earliness_harmonic_mean": (self.accuracy_earliness_harmonic_mean),
                "macro_f1_earliness_harmonic_mean": (self.macro_f1_earliness_harmonic_mean),
            }
        )
        return result


def _validated_binary_vector(values: ArrayLike, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    return array.astype(np.int64)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_classification_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> BinaryMetrics:
    """Compute an auditable binary confusion matrix and associated metrics.

    Label zero is control and label one is dysarthric.  The field names avoid the
    ambiguous positive/negative terminology that caused confusion in the manuscript.
    """

    targets = _validated_binary_vector(y_true, "y_true")
    predictions = _validated_binary_vector(y_pred, "y_pred")
    if targets.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    true_control = int(np.sum((targets == 0) & (predictions == 0)))
    false_dysarthric = int(np.sum((targets == 0) & (predictions == 1)))
    false_control = int(np.sum((targets == 1) & (predictions == 0)))
    true_dysarthric = int(np.sum((targets == 1) & (predictions == 1)))
    samples = len(targets)

    control_f1 = _safe_divide(
        2 * true_control,
        2 * true_control + false_dysarthric + false_control,
    )
    dysarthric_f1 = _safe_divide(
        2 * true_dysarthric,
        2 * true_dysarthric + false_dysarthric + false_control,
    )
    dysarthric_recall = _safe_divide(true_dysarthric, true_dysarthric + false_control)
    specificity = _safe_divide(true_control, true_control + false_dysarthric)

    return BinaryMetrics(
        samples=samples,
        true_control=true_control,
        false_dysarthric=false_dysarthric,
        false_control=false_control,
        true_dysarthric=true_dysarthric,
        accuracy=_safe_divide(true_control + true_dysarthric, samples),
        macro_f1=(control_f1 + dysarthric_f1) / 2.0,
        dysarthric_recall=dysarthric_recall,
        specificity=specificity,
        dysarthric_precision=_safe_divide(
            true_dysarthric,
            true_dysarthric + false_dysarthric,
        ),
        balanced_accuracy=(dysarthric_recall + specificity) / 2.0,
    )


def accuracy_earliness_harmonic_mean(score: float, consumed_fraction: float) -> float:
    """Return the standard harmonic mean of score and ``1 - earliness``."""

    if not 0.0 <= score <= 1.0:
        raise ValueError("score must lie in [0, 1]")
    if not 0.0 <= consumed_fraction <= 1.0:
        raise ValueError("consumed_fraction must lie in [0, 1]")
    timeliness = 1.0 - consumed_fraction
    denominator = score + timeliness
    return 2.0 * score * timeliness / denominator if denominator else 0.0


def evaluate_early_classification(
    y_true: ArrayLike,
    decisions: DecisionBatch,
    maximum_observation_ms: int = 300,
) -> EarlyMetrics:
    """Combine classification, latency, fallback, and early-classification metrics."""

    if maximum_observation_ms <= 0:
        raise ValueError("maximum_observation_ms must be positive")
    times = np.asarray(decisions.decision_time_ms, dtype=np.float64)
    fallback = np.asarray(decisions.forced_fallback, dtype=bool)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("decision times must be a non-empty one-dimensional array")
    if len(times) != len(fallback):
        raise ValueError("decision times and fallback flags must have the same length")
    if np.any(times <= 0.0) or np.any(times > maximum_observation_ms):
        raise ValueError("decision times must lie within the observation horizon")

    classification = binary_classification_metrics(y_true, decisions.predicted_label)
    if classification.samples != len(times):
        raise ValueError("The number of targets and decisions must match")
    mean_latency = float(times.mean())
    consumed = mean_latency / maximum_observation_ms
    return EarlyMetrics(
        classification=classification,
        mean_latency_ms=mean_latency,
        forced_fallback_rate=float(fallback.mean()),
        consumed_fraction=consumed,
        timeliness=1.0 - consumed,
        accuracy_earliness_harmonic_mean=accuracy_earliness_harmonic_mean(
            classification.accuracy,
            consumed,
        ),
        macro_f1_earliness_harmonic_mean=accuracy_earliness_harmonic_mean(
            classification.macro_f1,
            consumed,
        ),
    )
