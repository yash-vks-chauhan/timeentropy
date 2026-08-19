from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from timeentropy.evaluation.entropy import normalized_binary_entropy

PREFIX_TIMES_MS = (150, 200, 250, 300)
CANDIDATE_TIMES_MS = (200, 250, 300)

PolicyVariant = Literal[
    "entropy_only",
    "entropy_class_stability",
    "entropy_change_stability",
    "complete",
]

_POLICY_VARIANTS = {
    "entropy_only",
    "entropy_class_stability",
    "entropy_change_stability",
    "complete",
}


@dataclass(frozen=True)
class DecisionBatch:
    """Per-sample output and audit flags for one early-decision policy."""

    policy_variant: str
    theta: float
    epsilon: float
    decision_time_ms: NDArray[np.int64]
    predicted_label: NDArray[np.int64]
    accepted: NDArray[np.bool_]
    accepted_at_endpoint: NDArray[np.bool_]
    forced_fallback: NDArray[np.bool_]
    threshold_pass: NDArray[np.bool_]
    class_stable: NDArray[np.bool_]
    entropy_stable: NDArray[np.bool_]
    candidate_rule_pass: NDArray[np.bool_]


def _validated_prefix_values(
    predicted_labels: ArrayLike,
    entropies: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    predictions = np.asarray(predicted_labels)
    values = np.asarray(entropies, dtype=np.float64)
    expected_columns = len(PREFIX_TIMES_MS)
    if predictions.ndim != 2 or predictions.shape[1] != expected_columns:
        raise ValueError(f"predicted_labels must have shape (samples, {expected_columns})")
    if values.shape != predictions.shape:
        raise ValueError("entropies must have the same shape as predicted_labels")
    if len(predictions) == 0:
        raise ValueError("At least one sample is required")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("entropies must be finite and lie in [0, 1]")
    if not np.all(np.isin(predictions, (0, 1))):
        raise ValueError("predicted_labels must contain only binary labels 0 and 1")
    return predictions.astype(np.int64), values


def apply_decision_rule(
    predicted_labels: ArrayLike,
    entropies: ArrayLike,
    theta: float,
    epsilon: float,
    policy_variant: PolicyVariant = "complete",
) -> DecisionBatch:
    """Apply a registered early-decision variant to four prefix predictions.

    Column zero is the 150 ms warm-up and is never eligible for output.  Columns one,
    two, and three correspond to the 200/250/300 ms candidates.  Stability at 200 ms
    is measured against the 150 ms warm-up.  If no candidate passes, the 300 ms label
    is emitted as a forced fallback.
    """

    predictions, entropy = _validated_prefix_values(predicted_labels, entropies)
    if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be finite and lie in [0, 1]")
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and lie in [0, 1]")
    if policy_variant not in _POLICY_VARIANTS:
        raise ValueError(f"Unsupported policy_variant: {policy_variant}")

    current_entropy = entropy[:, 1:]
    previous_entropy = entropy[:, :-1]
    threshold_pass = current_entropy <= theta
    class_stable = predictions[:, 1:] == predictions[:, :-1]
    entropy_stable = np.abs(current_entropy - previous_entropy) <= epsilon

    candidate_pass = threshold_pass.copy()
    if policy_variant in {"entropy_class_stability", "complete"}:
        candidate_pass &= class_stable
    if policy_variant in {"entropy_change_stability", "complete"}:
        candidate_pass &= entropy_stable

    accepted = candidate_pass.any(axis=1)
    first_candidate = np.argmax(candidate_pass, axis=1)
    candidate_times = np.asarray(CANDIDATE_TIMES_MS, dtype=np.int64)
    chosen_candidate = np.where(accepted, first_candidate, len(CANDIDATE_TIMES_MS) - 1)
    decision_time = candidate_times[chosen_candidate]
    prefix_index = chosen_candidate + 1
    row_index = np.arange(len(predictions))
    labels = predictions[row_index, prefix_index]
    accepted_at_endpoint = accepted & (decision_time == CANDIDATE_TIMES_MS[-1])
    forced_fallback = ~accepted

    return DecisionBatch(
        policy_variant=policy_variant,
        theta=float(theta),
        epsilon=float(epsilon),
        decision_time_ms=decision_time,
        predicted_label=labels,
        accepted=accepted,
        accepted_at_endpoint=accepted_at_endpoint,
        forced_fallback=forced_fallback,
        threshold_pass=threshold_pass,
        class_stable=class_stable,
        entropy_stable=entropy_stable,
        candidate_rule_pass=candidate_pass,
    )


def decide_from_probabilities(
    probabilities: ArrayLike,
    theta: float,
    epsilon: float,
    policy_variant: PolicyVariant = "complete",
) -> DecisionBatch:
    """Compute predictions and normalized entropy, then apply the policy."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(PREFIX_TIMES_MS), 2):
        raise ValueError("probabilities must have shape (samples, 4, 2)")
    entropies = normalized_binary_entropy(values)
    predictions = np.argmax(values, axis=-1)
    return apply_decision_rule(predictions, entropies, theta, epsilon, policy_variant)


def decide_from_maximum_confidence(
    probabilities: ArrayLike,
    confidence_threshold: float,
) -> DecisionBatch:
    """Apply conventional maximum-softmax confidence at 200/250/300 ms.

    This comparator intentionally has no class- or entropy-change stability condition.
    For binary probabilities, a confidence threshold obtained from
    :func:`maximum_confidence_threshold_for_entropy` is analytically equivalent to the
    entropy-only policy at the corresponding entropy threshold.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(PREFIX_TIMES_MS), 2):
        raise ValueError("probabilities must have shape (samples, 4, 2)")
    normalized_binary_entropy(values)
    if (
        not np.isfinite(confidence_threshold)
        or confidence_threshold < 0.5
        or confidence_threshold > 1.0
    ):
        raise ValueError("confidence_threshold must be finite and lie in [0.5, 1]")
    predictions = np.argmax(values, axis=-1).astype(np.int64)
    threshold_pass = np.max(values[:, 1:, :], axis=-1) >= confidence_threshold
    class_stable = predictions[:, 1:] == predictions[:, :-1]
    entropy_stable = np.ones_like(threshold_pass, dtype=bool)
    accepted = threshold_pass.any(axis=1)
    first_candidate = np.argmax(threshold_pass, axis=1)
    chosen = np.where(accepted, first_candidate, len(CANDIDATE_TIMES_MS) - 1)
    candidate_times = np.asarray(CANDIDATE_TIMES_MS, dtype=np.int64)
    decision_time = candidate_times[chosen]
    rows = np.arange(len(values))
    labels = predictions[rows, chosen + 1]
    return DecisionBatch(
        policy_variant="max_softmax_confidence",
        theta=float(confidence_threshold),
        epsilon=0.0,
        decision_time_ms=decision_time,
        predicted_label=labels,
        accepted=accepted,
        accepted_at_endpoint=accepted & (decision_time == CANDIDATE_TIMES_MS[-1]),
        forced_fallback=~accepted,
        threshold_pass=threshold_pass,
        class_stable=class_stable,
        entropy_stable=entropy_stable,
        candidate_rule_pass=threshold_pass.copy(),
    )


def fixed_duration_predictions(probabilities: ArrayLike, prefix_ms: int) -> NDArray[np.int64]:
    """Return labels forced at one registered prefix duration."""

    if prefix_ms not in PREFIX_TIMES_MS:
        raise ValueError(f"prefix_ms must be one of {PREFIX_TIMES_MS}")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(PREFIX_TIMES_MS), 2):
        raise ValueError("probabilities must have shape (samples, 4, 2)")
    # Entropy validation also checks range and probability sums.
    normalized_binary_entropy(values)
    prefix_index = PREFIX_TIMES_MS.index(prefix_ms)
    return np.argmax(values[:, prefix_index, :], axis=1).astype(np.int64)
