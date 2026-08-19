from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class TemperatureFit:
    """Result of deterministic scalar temperature fitting."""

    temperature: float
    nll_before: float
    nll_after: float
    optimizer_success: bool
    optimizer_iterations: int


@dataclass(frozen=True)
class ReliabilityBin:
    lower_bound: float
    upper_bound: float
    samples: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationMetrics:
    samples: int
    bins: int
    negative_log_likelihood: float
    binary_brier_score: float
    expected_calibration_error: float
    reliability_bins: tuple[ReliabilityBin, ...]


def _validated_logits(logits: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("logits must have shape (samples, classes) with at least two classes")
    if len(values) == 0:
        raise ValueError("logits must contain at least one sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    return values


def _validated_labels(labels: ArrayLike, samples: int, classes: int) -> NDArray[np.int64]:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != samples:
        raise ValueError("labels must be one-dimensional and match the number of logits")
    if not np.issubdtype(values.dtype, np.integer) and not np.all(
        np.equal(values, np.floor(values))
    ):
        raise ValueError("labels must contain integer class indices")
    result = values.astype(np.int64)
    if np.any(result < 0) or np.any(result >= classes):
        raise ValueError("labels contain an out-of-range class index")
    return result


def apply_temperature(logits: ArrayLike, temperature: float) -> NDArray[np.float64]:
    """Divide logits by a positive scalar temperature."""

    values = _validated_logits(logits)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return values / temperature


def probabilities_from_logits(
    logits: ArrayLike,
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    """Compute a numerically stable softmax after scalar temperature scaling."""

    scaled = apply_temperature(logits, temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    logits: ArrayLike,
    labels: ArrayLike,
    temperature: float = 1.0,
) -> float:
    """Mean multiclass negative log likelihood from logits."""

    values = _validated_logits(logits)
    targets = _validated_labels(labels, len(values), values.shape[1])
    probabilities = probabilities_from_logits(values, temperature)
    selected = np.clip(probabilities[np.arange(len(targets)), targets], 1.0e-15, 1.0)
    return float(-np.log(selected).mean())


def fit_temperature(
    logits: ArrayLike,
    labels: ArrayLike,
    temperature_bounds: tuple[float, float] = (0.05, 20.0),
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 500,
) -> TemperatureFit:
    """Fit one deterministic temperature by minimizing calibration-set NLL.

    Optimization is performed over log-temperature with bounded scalar minimization.  If
    numerical optimization produces a value worse than the identity temperature, the
    identity is retained.  The caller is responsible for supplying only the dedicated
    calibration split.
    """

    values = _validated_logits(logits)
    targets = _validated_labels(labels, len(values), values.shape[1])
    lower, upper = temperature_bounds
    if not 0.0 < lower < 1.0 < upper:
        raise ValueError("temperature_bounds must be positive and contain 1.0")
    if tolerance <= 0.0 or maximum_iterations < 1:
        raise ValueError("tolerance and maximum_iterations must be positive")

    before = negative_log_likelihood(values, targets, temperature=1.0)

    def objective(log_temperature: float) -> float:
        return negative_log_likelihood(values, targets, temperature=math.exp(log_temperature))

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(math.log(lower), math.log(upper)),
        options={"xatol": tolerance, "maxiter": maximum_iterations},
    )
    candidate_temperature = math.exp(float(result.x))
    candidate_nll = objective(float(result.x))
    if not math.isfinite(candidate_nll) or candidate_nll > before:
        temperature = 1.0
        after = before
    else:
        temperature = candidate_temperature
        after = candidate_nll

    return TemperatureFit(
        temperature=float(temperature),
        nll_before=before,
        nll_after=float(after),
        optimizer_success=bool(result.success),
        optimizer_iterations=int(result.nfev),
    )


def calibration_metrics(
    probabilities: ArrayLike,
    labels: ArrayLike,
    bins: int = 15,
) -> CalibrationMetrics:
    """Measure binary calibration with fixed equal-width confidence bins.

    ECE uses top-label confidence and correctness.  The binary Brier score is the mean
    squared error between dysarthric-class probability and the binary target.  Empty
    reliability bins are retained so figures use the same 15-bin geometry at every prefix.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("probabilities must have shape (samples, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("probabilities must sum to one")
    targets = _validated_labels(labels, len(values), classes=2)
    if bins < 2:
        raise ValueError("bins must be at least two")

    row_index = np.arange(len(targets))
    selected = np.clip(values[row_index, targets], 1.0e-15, 1.0)
    nll = float(-np.log(selected).mean())
    brier = float(np.mean(np.square(values[:, 1] - targets)))
    predictions = np.argmax(values, axis=1)
    confidence = np.max(values, axis=1)
    correct = predictions == targets
    bin_indices = np.minimum((confidence * bins).astype(np.int64), bins - 1)

    reliability: list[ReliabilityBin] = []
    ece = 0.0
    for bin_index in range(bins):
        members = bin_indices == bin_index
        count = int(members.sum())
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if count:
            mean_confidence = float(confidence[members].mean())
            accuracy = float(correct[members].mean())
            ece += count / len(values) * abs(accuracy - mean_confidence)
        else:
            mean_confidence = float("nan")
            accuracy = float("nan")
        reliability.append(
            ReliabilityBin(
                lower_bound=lower,
                upper_bound=upper,
                samples=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )

    return CalibrationMetrics(
        samples=len(values),
        bins=bins,
        negative_log_likelihood=nll,
        binary_brier_score=brier,
        expected_calibration_error=float(ece),
        reliability_bins=tuple(reliability),
    )
