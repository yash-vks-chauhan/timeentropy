import numpy as np
import pandas as pd

from timeentropy.reporting.summaries import build_summary_tables


def _allocate(total: int, weights: np.ndarray) -> np.ndarray:
    raw = total * weights / weights.sum()
    allocated = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - allocated))[: total - int(allocated.sum())]:
        allocated[index] += 1
    return allocated


def _old_manuscript_mismatch() -> pd.DataFrame:
    # These fold sizes are the recovered OOF fold sizes.  The correct counts reproduce
    # the old 75.89% unweighted fold average while their pooled total reproduces the
    # Figure 6 accuracy of 74.96%.
    fold_sizes = np.asarray([1689, 1940, 1657, 1288, 1557])
    fold_correct = np.asarray([1282, 1247, 1257, 1127, 1182])
    fold_errors = fold_sizes - fold_correct
    true_control = _allocate(3947, fold_correct)
    true_dysarthric = fold_correct - true_control
    false_dysarthric = _allocate(1250, fold_errors)
    false_control = fold_errors - false_dysarthric

    rows = []
    sample_number = 0
    for fold in range(5):
        cells = (
            (0, 0, int(true_control[fold])),
            (0, 1, int(false_dysarthric[fold])),
            (1, 0, int(false_control[fold])),
            (1, 1, int(true_dysarthric[fold])),
        )
        for label, prediction, count in cells:
            for _ in range(count):
                sample_number += 1
                rows.append(
                    {
                        "system_id": "transformer-primary",
                        "model_id": "transformer",
                        "seed": 20260819,
                        "onset_variant": "causal_fixed",
                        "pre_roll_ms": 30,
                        "policy_id": "complete_calibrated",
                        "sample_id": f"sample-{sample_number:05d}",
                        "speaker_id": f"speaker-{fold}-{label}",
                        "fold": fold,
                        "label_id": label,
                        "predicted_label": prediction,
                        "decision_time_ms": 300,
                        "forced_fallback": False,
                        "accepted_at_endpoint": True,
                    }
                )
    return pd.DataFrame(rows)


def test_pooled_and_unweighted_fold_estimands_cannot_be_confused() -> None:
    tables = build_summary_tables(_old_manuscript_mismatch())
    pooled = tables.pooled_metrics.iloc[0]
    fold_summary = tables.fold_summary.iloc[0]

    assert pooled["true_control"] == 3947
    assert pooled["false_dysarthric"] == 1250
    assert pooled["false_control"] == 786
    assert pooled["true_dysarthric"] == 2148
    assert round(100 * pooled["accuracy"], 2) == 74.96
    assert round(100 * fold_summary["accuracy_mean"], 2) == 75.89
    assert pooled["accuracy"] != fold_summary["accuracy_mean"]
    assert pooled["estimand"] == "pooled_out_of_fold_utterances"
    assert fold_summary["estimand"] == "unweighted_fold_mean_and_sample_sd"


def test_class_and_error_tables_reproduce_the_pooled_confusion_matrix() -> None:
    tables = build_summary_tables(_old_manuscript_mismatch())
    classes = tables.class_metrics.set_index("class_label")
    errors = tables.error_breakdown.iloc[0]
    assert classes.loc["control", "support"] == 5197
    assert classes.loc["dysarthric", "support"] == 2934
    assert errors["false_positive_dysarthric_count"] == 1250
    assert errors["false_negative_dysarthric_count"] == 786
    assert np.isclose(classes.loc["dysarthric", "recall"], 2148 / 2934)


def test_latency_counts_exactly_reproduce_mean_latency() -> None:
    frame = _old_manuscript_mismatch().iloc[:10].copy()
    frame["decision_time_ms"] = [200] * 2 + [250] * 3 + [300] * 5
    frame["forced_fallback"] = [False] * 5 + [True] * 5
    frame["accepted_at_endpoint"] = False
    tables = build_summary_tables(frame)
    counts = tables.latency_counts
    reconstructed = float(
        (counts["decision_time_ms"] * counts["count"]).sum() / counts["count"].sum()
    )
    assert reconstructed == tables.latency_summary.iloc[0]["mean_latency_ms"] == 265.0
