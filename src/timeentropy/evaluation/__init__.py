"""Calibration, early-decision, metric, and ledger utilities.

The evaluation package intentionally operates on saved logits and predictions.  Model
training must not be coupled to entropy-policy selection or test-set aggregation.
"""

from timeentropy.evaluation.calibration import (
    CalibrationMetrics,
    ReliabilityBin,
    TemperatureFit,
    apply_temperature,
    calibration_metrics,
    fit_temperature,
    negative_log_likelihood,
    probabilities_from_logits,
)
from timeentropy.evaluation.decisions import (
    CANDIDATE_TIMES_MS,
    PREFIX_TIMES_MS,
    DecisionBatch,
    PolicyVariant,
    apply_decision_rule,
    decide_from_maximum_confidence,
    decide_from_probabilities,
    fixed_duration_predictions,
)
from timeentropy.evaluation.entropy import (
    maximum_confidence_threshold_for_entropy,
    normalized_binary_entropy,
)
from timeentropy.evaluation.metrics import (
    BinaryMetrics,
    EarlyMetrics,
    binary_classification_metrics,
    evaluate_early_classification,
)
from timeentropy.evaluation.pipeline import (
    DEFAULT_POLICY_VARIANTS,
    EvaluationArtifacts,
    build_policy_artifacts,
    calibrate_prefix_ledger,
    combine_prefix_ledgers,
    evaluate_prefix_ledgers,
    fixed_prefix_metrics,
    read_ledger,
    write_evaluation_artifacts,
)
from timeentropy.evaluation.selection import (
    DEFAULT_EPSILON_GRID,
    DEFAULT_THETA_GRID,
    CandidateScore,
    PolicySelection,
    select_policy_parameters,
)

__all__ = [
    "CANDIDATE_TIMES_MS",
    "DEFAULT_EPSILON_GRID",
    "DEFAULT_POLICY_VARIANTS",
    "DEFAULT_THETA_GRID",
    "PREFIX_TIMES_MS",
    "BinaryMetrics",
    "CalibrationMetrics",
    "CandidateScore",
    "DecisionBatch",
    "EarlyMetrics",
    "EvaluationArtifacts",
    "PolicySelection",
    "PolicyVariant",
    "ReliabilityBin",
    "TemperatureFit",
    "apply_decision_rule",
    "apply_temperature",
    "binary_classification_metrics",
    "build_policy_artifacts",
    "calibrate_prefix_ledger",
    "calibration_metrics",
    "combine_prefix_ledgers",
    "decide_from_maximum_confidence",
    "decide_from_probabilities",
    "evaluate_early_classification",
    "evaluate_prefix_ledgers",
    "fit_temperature",
    "fixed_duration_predictions",
    "fixed_prefix_metrics",
    "maximum_confidence_threshold_for_entropy",
    "negative_log_likelihood",
    "normalized_binary_entropy",
    "probabilities_from_logits",
    "read_ledger",
    "select_policy_parameters",
    "write_evaluation_artifacts",
]
