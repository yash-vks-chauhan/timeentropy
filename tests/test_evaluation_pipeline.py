from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timeentropy.evaluation.calibration import probabilities_from_logits
from timeentropy.evaluation.entropy import normalized_binary_entropy
from timeentropy.evaluation.pipeline import (
    calibrate_prefix_ledger,
    evaluate_prefix_ledgers,
    fixed_prefix_metrics,
    write_evaluation_artifacts,
)
from timeentropy.reporting.pipeline import generate_report_bundle


def _ledger() -> pd.DataFrame:
    rows = []
    samples = (
        ("calibration", "c0", "FC01", 0),
        ("calibration", "c1", "F01", 1),
        ("policy_tuning", "p0", "FC01", 0),
        ("policy_tuning", "p1", "F01", 1),
        ("test", "t0", "FC02", 0),
        ("test", "t1", "F03", 1),
    )
    for split_role, sample_id, speaker_id, label_id in samples:
        for prefix_ms in (150, 200, 250, 300):
            strength = 0.8 + prefix_ms / 500.0
            logits = (
                np.asarray([[strength, -strength]])
                if label_id == 0
                else np.asarray([[-strength, strength]])
            )
            probabilities = probabilities_from_logits(logits)[0]
            rows.append(
                {
                    "run_id": "run-fold-0",
                    "sample_id": sample_id,
                    "speaker_id": speaker_id,
                    "fold": 0,
                    "seed": 20260819,
                    "model_id": "transformer",
                    "split_role": split_role,
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


def test_complete_pipeline_calibrates_selects_and_preserves_audit_rows() -> None:
    artifacts = evaluate_prefix_ledgers(
        _ledger(),
        bins=5,
        policy_variants=("complete",),
        theta_grid=(0.9,),
        epsilon_grid=(1.0,),
    )
    assert len(artifacts.calibrated_prefix_ledger) == 24
    assert artifacts.calibrated_prefix_ledger["temperature"].nunique() == 1
    assert len(artifacts.temperature_fits) == 1
    assert len(artifacts.selections) == 2  # raw and temperature-scaled
    # Two adaptive states plus fixed 200/250/300, for two test samples.
    assert len(artifacts.decision_ledger) == 10
    adaptive = artifacts.decision_ledger.loc[
        artifacts.decision_ledger["policy_variant"].eq("complete")
    ]
    assert set(adaptive["decision_time_ms"]) == {200}
    assert len(artifacts.fixed_predictions) == 8
    assert len(artifacts.calibration_summary) == 3 * 4 * 2
    assert len(artifacts.reliability_bins) == 3 * 4 * 2 * 5


def test_fixed_prefix_metrics_recompute_confusion_values() -> None:
    artifacts = evaluate_prefix_ledgers(
        _ledger(),
        policy_variants=("entropy_only",),
        theta_grid=(1.0,),
        epsilon_grid=(0.0,),
    )
    metrics = fixed_prefix_metrics(artifacts.fixed_predictions)
    assert metrics["prefix_ms"].tolist() == [150, 200, 250, 300]
    assert metrics["accuracy"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_calibration_requires_dedicated_rows() -> None:
    ledger = _ledger().loc[lambda frame: ~frame["split_role"].eq("calibration")]
    with pytest.raises(ValueError, match="no calibration rows"):
        calibrate_prefix_ledger(ledger)


def test_artifact_writer_emits_hash_summary(tmp_path) -> None:
    artifacts = evaluate_prefix_ledgers(
        _ledger(),
        policy_variants=("complete",),
        theta_grid=(1.0,),
        epsilon_grid=(1.0,),
    )
    paths = write_evaluation_artifacts(artifacts, tmp_path, config={"frozen": True})
    assert paths["summary"].is_file()
    assert paths["decision_ledger"].is_file()
    report_directory = tmp_path / "reports"
    generate_report_bundle(
        paths["calibrated_prefix_ledger"],
        paths["decision_ledger"],
        report_directory,
        bootstrap_replicates=10,
        include_model_comparisons=False,
    )
    pooled = pd.read_csv(report_directory / "pooled_metrics.csv")
    assert {"fixed_300", "complete-raw", "complete-temperature_scaled"}.issubset(
        set(pooled["policy_id"])
    )
