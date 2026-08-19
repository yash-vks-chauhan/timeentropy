from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from timeentropy.evaluation.decisions import (
    DecisionBatch,
    PolicyVariant,
    decide_from_probabilities,
    fixed_duration_predictions,
)
from timeentropy.evaluation.metrics import (
    binary_classification_metrics,
    evaluate_early_classification,
)

DEFAULT_THETA_GRID = tuple(round(0.30 + 0.05 * index, 2) for index in range(14))
DEFAULT_EPSILON_GRID = tuple(round(0.02 * index, 2) for index in range(11))


@dataclass(frozen=True)
class CandidateScore:
    theta: float
    epsilon: float
    macro_f1: float
    accuracy: float
    mean_latency_ms: float
    forced_fallback_rate: float
    feasible: bool


@dataclass(frozen=True)
class PolicySelection:
    policy_variant: str
    macro_f1_margin: float
    fixed_300_macro_f1: float
    feasible_policy_found: bool
    selected: CandidateScore
    selected_decisions: DecisionBatch
    candidates: tuple[CandidateScore, ...]


def _validated_grid(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(sorted({float(value) for value in values}))
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)) or result[0] < 0.0 or result[-1] > 1.0:
        raise ValueError(f"{name} values must be finite and lie in [0, 1]")
    return result


def select_policy_parameters(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    split_role: str,
    policy_variant: PolicyVariant = "complete",
    theta_grid: Iterable[float] = DEFAULT_THETA_GRID,
    epsilon_grid: Iterable[float] = DEFAULT_EPSILON_GRID,
    macro_f1_margin: float = 0.01,
) -> PolicySelection:
    """Select theta/epsilon under the frozen validation non-inferiority rule.

    Only the dedicated ``policy_tuning`` split is accepted.  Feasible policies remain
    within ``macro_f1_margin`` of fixed 300 ms macro-F1.  The deterministic ordering is:
    lowest latency, highest macro-F1, lowest fallback rate, lowest theta, lowest epsilon.
    If no candidate is feasible, macro-F1 is maximized before applying the other criteria.
    """

    if split_role != "policy_tuning":
        raise ValueError("Policy parameters may only be selected on split_role='policy_tuning'")
    if not np.isfinite(macro_f1_margin) or not 0.0 <= macro_f1_margin <= 1.0:
        raise ValueError("macro_f1_margin must be finite and lie in [0, 1]")
    theta_values = _validated_grid(theta_grid, "theta_grid")
    variants_with_epsilon = {"entropy_change_stability", "complete"}
    epsilon_values = (
        _validated_grid(epsilon_grid, "epsilon_grid")
        if policy_variant in variants_with_epsilon
        else (0.0,)
    )

    values = np.asarray(probabilities, dtype=np.float64)
    fixed_predictions = fixed_duration_predictions(values, 300)
    fixed_metrics = binary_classification_metrics(y_true, fixed_predictions)
    floor = fixed_metrics.macro_f1 - macro_f1_margin

    scored: list[CandidateScore] = []
    decisions_by_key: dict[tuple[float, float], DecisionBatch] = {}
    for theta in theta_values:
        for epsilon in epsilon_values:
            decisions = decide_from_probabilities(
                values,
                theta=theta,
                epsilon=epsilon,
                policy_variant=policy_variant,
            )
            metrics = evaluate_early_classification(y_true, decisions)
            candidate = CandidateScore(
                theta=theta,
                epsilon=epsilon,
                macro_f1=metrics.classification.macro_f1,
                accuracy=metrics.classification.accuracy,
                mean_latency_ms=metrics.mean_latency_ms,
                forced_fallback_rate=metrics.forced_fallback_rate,
                feasible=metrics.classification.macro_f1 + 1.0e-12 >= floor,
            )
            scored.append(candidate)
            decisions_by_key[(theta, epsilon)] = decisions

    feasible = [candidate for candidate in scored if candidate.feasible]
    if feasible:
        selected = min(
            feasible,
            key=lambda item: (
                item.mean_latency_ms,
                -item.macro_f1,
                item.forced_fallback_rate,
                item.theta,
                item.epsilon,
            ),
        )
    else:
        selected = min(
            scored,
            key=lambda item: (
                -item.macro_f1,
                item.mean_latency_ms,
                item.forced_fallback_rate,
                item.theta,
                item.epsilon,
            ),
        )

    return PolicySelection(
        policy_variant=policy_variant,
        macro_f1_margin=float(macro_f1_margin),
        fixed_300_macro_f1=fixed_metrics.macro_f1,
        feasible_policy_found=bool(feasible),
        selected=selected,
        selected_decisions=decisions_by_key[(selected.theta, selected.epsilon)],
        candidates=tuple(scored),
    )
