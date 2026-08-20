import pandas as pd

from timeentropy.reporting.bootstrap import clustered_speaker_bootstrap_ci
from timeentropy.reporting.comparison import paired_model_comparisons


def _paired_models() -> pd.DataFrame:
    rows = []
    for model in ("model_a", "model_b"):
        for speaker_number, label in enumerate((0, 0, 1, 1)):
            for utterance in range(4):
                prediction = label
                if model == "model_b" and utterance == 0:
                    prediction = 1 - label
                rows.append(
                    {
                        "system_id": f"{model}-system",
                        "model_id": model,
                        "seed": 20260819,
                        "onset_variant": "causal_fixed",
                        "pre_roll_ms": 30,
                        "policy_id": "complete",
                        "sample_id": f"speaker-{speaker_number}-utt-{utterance}",
                        "speaker_id": f"speaker-{speaker_number}",
                        "fold": speaker_number,
                        "label_id": label,
                        "predicted_label": prediction,
                        "decision_time_ms": 250 if model == "model_a" else 300,
                        "forced_fallback": model == "model_b",
                        "accepted_at_endpoint": False,
                    }
                )
    return pd.DataFrame(rows)


def test_speaker_bootstrap_is_deterministic() -> None:
    frame = _paired_models().query("model_id == 'model_a'")
    first = clustered_speaker_bootstrap_ci(frame, replicates=200, seed=71).intervals
    second = clustered_speaker_bootstrap_ci(frame, replicates=200, seed=71).intervals
    pd.testing.assert_frame_equal(first, second)
    assert set(first["method"]) == {"percentile_speaker_cluster_bootstrap_stratified_by_label"}


def test_paired_comparison_reports_cluster_bootstrap_and_mcnemar() -> None:
    comparisons, mcnemar = paired_model_comparisons(
        _paired_models(),
        replicates=200,
        seed=91,
    )
    accuracy = comparisons[comparisons["metric"] == "accuracy"].iloc[0]
    assert accuracy["difference_left_minus_right"] > 0
    assert "speaker_cluster_bootstrap" in accuracy["method"]
    assert mcnemar.iloc[0]["discordant_pairs"] == 4
    assert "sensitivity" in mcnemar.iloc[0]["scope_warning"]


def test_paired_policy_comparisons_tests_adaptive_against_fixed_baselines() -> None:
    """R1.12/R2.12 need policy-vs-baseline testing, which the model comparison never does."""

    from timeentropy.reporting.comparison import (
        paired_model_comparisons,
        paired_policy_comparisons,
    )

    rows = []
    for policy_id, correct_run in (("complete-raw", 8), ("fixed_300", 4)):
        for index in range(12):
            label = index % 2
            rows.append(
                {
                    "system_id": f"m=transformer|policy={policy_id}",
                    "model_id": "transformer",
                    "seed": 20260819,
                    "onset_variant": "causal_noise_floor",
                    "pre_roll_ms": 30,
                    "policy_id": policy_id,
                    "sample_id": f"s{index:03d}",
                    "speaker_id": f"SPK{index % 4}",
                    "fold": index % 5,
                    "label_id": label,
                    "predicted_label": label if index < correct_run else 1 - label,
                    "decision_time_ms": 200 if policy_id.startswith("complete") else 300,
                    "forced_fallback": not policy_id.startswith("complete"),
                }
            )
    enriched = pd.DataFrame(rows)

    # The model comparison finds nothing: there is only one model.
    model_frame, model_mcnemar = paired_model_comparisons(enriched, replicates=50)
    assert model_frame.empty and model_mcnemar.empty

    frame, mcnemar = paired_policy_comparisons(enriched, replicates=200)
    assert not frame.empty
    assert set(mcnemar["adaptive_policy_id"]) == {"complete-raw"}
    assert set(mcnemar["baseline_policy_id"]) == {"fixed_300"}

    record = mcnemar.iloc[0]
    # complete-raw is correct on 8 of 12, fixed_300 on 4 of 12, and the disagreements are
    # exactly the four rows where they differ.
    assert record["left_correct_right_wrong"] == 4
    assert record["left_wrong_right_correct"] == 0
    assert record["discordant_pairs"] == 4
    assert 0.0 < record["exact_two_sided_p_value"] <= 1.0
    assert "not evidence of equivalence" in str(record["scope_warning"])


def test_paired_policy_comparisons_handles_empty_input() -> None:
    from timeentropy.reporting.comparison import paired_policy_comparisons

    frame, mcnemar = paired_policy_comparisons(pd.DataFrame())
    assert frame.empty and mcnemar.empty
