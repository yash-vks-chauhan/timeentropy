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
