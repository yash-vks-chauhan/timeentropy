import numpy as np
import pytest

from timeentropy.evaluation.decisions import (
    apply_decision_rule,
    decide_from_maximum_confidence,
    decide_from_probabilities,
)
from timeentropy.evaluation.entropy import maximum_confidence_threshold_for_entropy
from timeentropy.evaluation.selection import select_policy_parameters


def test_complete_rule_uses_150_ms_only_as_warmup() -> None:
    predictions = np.asarray(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 1],
        ]
    )
    entropies = np.asarray(
        [
            [0.60, 0.40, 0.35, 0.30],
            [0.90, 0.80, 0.70, 0.60],
            [0.42, 0.40, 0.35, 0.30],
        ]
    )
    result = apply_decision_rule(
        predictions,
        entropies,
        theta=0.50,
        epsilon=0.10,
        policy_variant="complete",
    )
    np.testing.assert_array_equal(result.decision_time_ms, [250, 300, 250])
    np.testing.assert_array_equal(result.forced_fallback, [False, True, False])
    assert not np.any(result.decision_time_ms == 150)


def test_entropy_only_ablation_does_not_require_class_stability() -> None:
    predictions = np.asarray([[0, 1, 1, 1]])
    entropies = np.asarray([[0.45, 0.40, 0.35, 0.30]])
    result = apply_decision_rule(
        predictions,
        entropies,
        theta=0.50,
        epsilon=0.0,
        policy_variant="entropy_only",
    )
    assert result.decision_time_ms.tolist() == [200]
    assert result.predicted_label.tolist() == [1]


def test_selection_uses_margin_then_deterministic_latency_ties() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray(
        [
            [[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05]],
            [[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05]],
            [[0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95]],
            [[0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95]],
        ]
    )
    selection = select_policy_parameters(
        y_true,
        probabilities,
        split_role="policy_tuning",
        theta_grid=[0.20, 0.30, 0.40],
        epsilon_grid=[0.00, 0.10],
    )
    assert selection.feasible_policy_found
    assert selection.selected.theta == 0.30
    assert selection.selected.epsilon == 0.00
    assert selection.selected.mean_latency_ms == 200.0


def test_selection_refuses_test_predictions() -> None:
    probabilities = np.tile(np.asarray([[[0.9, 0.1]] * 4]), (2, 1, 1))
    with pytest.raises(ValueError, match="policy_tuning"):
        select_policy_parameters(
            [0, 1],
            probabilities,
            split_role="test",
            theta_grid=[0.5],
            epsilon_grid=[0.1],
        )


def test_binary_entropy_only_is_exactly_maximum_confidence() -> None:
    probabilities = np.asarray(
        [
            [[0.55, 0.45], [0.70, 0.30], [0.85, 0.15], [0.90, 0.10]],
            [[0.51, 0.49], [0.52, 0.48], [0.53, 0.47], [0.54, 0.46]],
        ]
    )
    theta = 0.75
    entropy = decide_from_probabilities(
        probabilities,
        theta=theta,
        epsilon=0.0,
        policy_variant="entropy_only",
    )
    confidence = decide_from_maximum_confidence(
        probabilities,
        maximum_confidence_threshold_for_entropy(theta),
    )
    np.testing.assert_array_equal(entropy.decision_time_ms, confidence.decision_time_ms)
    np.testing.assert_array_equal(entropy.predicted_label, confidence.predicted_label)
    np.testing.assert_array_equal(entropy.forced_fallback, confidence.forced_fallback)
