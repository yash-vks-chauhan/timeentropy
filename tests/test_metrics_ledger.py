import numpy as np
import pandas as pd
import pytest

from timeentropy.evaluation.calibration import probabilities_from_logits
from timeentropy.evaluation.entropy import normalized_binary_entropy
from timeentropy.evaluation.ledger import (
    LedgerInvariantError,
    validate_decision_ledger,
    validate_oof_test_coverage,
    validate_prefix_ledger,
)
from timeentropy.evaluation.metrics import binary_classification_metrics


def _prefix_ledger() -> pd.DataFrame:
    rows = []
    for sample_id, speaker_id, label_id, fold in (
        ("a", "FC01", 0, 0),
        ("b", "F01", 1, 1),
    ):
        for prefix_ms in (150, 200, 250, 300):
            logits = np.asarray([[2.0, -1.0]]) if label_id == 0 else np.asarray([[-1.0, 2.0]])
            probabilities = probabilities_from_logits(logits)[0]
            rows.append(
                {
                    "run_id": f"run-fold-{fold}",
                    "sample_id": sample_id,
                    "speaker_id": speaker_id,
                    "fold": fold,
                    "seed": 20260819,
                    "model_id": "transformer",
                    "split_role": "test",
                    "onset_variant": "causal_fixed",
                    "pre_roll_ms": 30,
                    "prefix_ms": prefix_ms,
                    "label_id": label_id,
                    "logit_control": logits[0, 0],
                    "logit_dysarthric": logits[0, 1],
                    "prob_control": probabilities[0],
                    "prob_dysarthric": probabilities[1],
                    "predicted_label": int(np.argmax(probabilities)),
                    "normalized_entropy": float(normalized_binary_entropy(probabilities)),
                }
            )
    return pd.DataFrame(rows)


def test_binary_metrics_reproduce_confusion_counts() -> None:
    metrics = binary_classification_metrics(
        [0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1],
    )
    assert (metrics.true_control, metrics.false_dysarthric) == (2, 1)
    assert (metrics.false_control, metrics.true_dysarthric) == (1, 1)
    assert metrics.accuracy == 0.6
    assert metrics.dysarthric_recall == 0.5


def test_prefix_and_oof_ledger_invariants_pass() -> None:
    ledger = _prefix_ledger()
    report = validate_prefix_ledger(ledger)
    assert report.rows == 8
    assert report.samples == 2
    assert report.test_samples == 2
    validate_oof_test_coverage(ledger, expected_sample_ids=["a", "b"])


def test_corrupt_entropy_is_rejected() -> None:
    ledger = _prefix_ledger()
    ledger.loc[0, "normalized_entropy"] = 0.0
    with pytest.raises(LedgerInvariantError, match="normalized_entropy"):
        validate_prefix_ledger(ledger)


def test_duplicate_oof_run_is_rejected() -> None:
    ledger = _prefix_ledger()
    duplicate = ledger[ledger["sample_id"] == "a"].copy()
    duplicate["run_id"] = "duplicate-fold-0"
    combined = pd.concat([ledger, duplicate], ignore_index=True)
    with pytest.raises(LedgerInvariantError, match="multiple test runs"):
        validate_oof_test_coverage(combined, expected_sample_ids=["a", "b"])


def test_decision_label_must_match_selected_prefix() -> None:
    prefix = _prefix_ledger()
    decisions = pd.DataFrame(
        [
            {
                "run_id": "run-fold-0",
                "sample_id": "a",
                "policy_id": "complete-calibrated",
                "label_id": 0,
                "decision_time_ms": 200,
                "predicted_label": 0,
                "forced_fallback": False,
                "accepted_at_endpoint": False,
            }
        ]
    )
    report = validate_decision_ledger(
        decisions,
        prefix_ledger=prefix,
        prefix_prediction_column="predicted_label",
    )
    assert report.rows == 1
    decisions.loc[0, "predicted_label"] = 1
    with pytest.raises(LedgerInvariantError, match="do not match"):
        validate_decision_ledger(
            decisions,
            prefix_ledger=prefix,
            prefix_prediction_column="predicted_label",
        )
