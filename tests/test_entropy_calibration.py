import numpy as np

from timeentropy.evaluation.calibration import (
    calibration_metrics,
    fit_temperature,
    negative_log_likelihood,
    probabilities_from_logits,
)
from timeentropy.evaluation.entropy import (
    maximum_confidence_threshold_for_entropy,
    normalized_binary_entropy,
)


def test_normalized_binary_entropy_has_expected_endpoints() -> None:
    values = normalized_binary_entropy([[0.5, 0.5], [1.0, 0.0], [0.9, 0.1]])
    np.testing.assert_allclose(values[:2], [1.0, 0.0], atol=1.0e-12)
    assert 0.0 < values[2] < 1.0


def test_binary_entropy_threshold_maps_exactly_to_confidence() -> None:
    theta = 0.50
    confidence = maximum_confidence_threshold_for_entropy(theta)
    entropy = float(normalized_binary_entropy([confidence, 1.0 - confidence]))
    assert abs(entropy - theta) < 1.0e-12


def test_temperature_fit_reduces_overconfident_nll_deterministically() -> None:
    labels = np.asarray([0] * 75 + [1] * 25, dtype=np.int64)
    logits = np.tile(np.asarray([[5.0, -5.0]]), (100, 1))
    first = fit_temperature(logits, labels)
    second = fit_temperature(logits, labels)
    assert first == second
    assert first.temperature > 1.0
    assert first.nll_after < first.nll_before
    assert negative_log_likelihood(logits, labels, first.temperature) == first.nll_after
    probabilities = probabilities_from_logits(logits, first.temperature)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_calibration_metrics_use_fixed_reliability_bins() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])
    result = calibration_metrics(probabilities, [0, 1, 1, 1], bins=5)
    assert result.samples == 4
    assert len(result.reliability_bins) == 5
    assert sum(item.samples for item in result.reliability_bins) == 4
    assert 0.0 <= result.expected_calibration_error <= 1.0
    assert result.binary_brier_score == np.mean(np.square(probabilities[:, 1] - [0, 1, 1, 1]))
