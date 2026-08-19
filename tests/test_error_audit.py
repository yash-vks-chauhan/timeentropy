from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from timeentropy.analysis.error_audit import (
    REVIEW_TAXONOMY_COLUMNS,
    audit_classification_errors,
    write_error_audit,
)
from timeentropy.evaluation.calibration import probabilities_from_logits
from timeentropy.evaluation.entropy import normalized_binary_entropy
from timeentropy.provenance import sha256_file


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = [f"c{index}" for index in range(4)] + [f"d{index}" for index in range(4)]
    labels = {sample_id: int(sample_id.startswith("d")) for sample_id in samples}
    speakers = {
        "c0": "C1",
        "c1": "C1",
        "c2": "C2",
        "c3": "C2",
        "d0": "D1",
        "d1": "D1",
        "d2": "D2",
        "d3": "D2",
    }
    predictions = {
        "run-a": {"c0": 1, "c1": 1, "c2": 0, "c3": 0, "d0": 0, "d1": 1, "d2": 1, "d3": 1},
        "run-b": {"c0": 1, "c1": 0, "c2": 0, "c3": 0, "d0": 0, "d1": 0, "d2": 1, "d3": 1},
    }
    model_ids = {"run-a": "model-a", "run-b": "model-b"}

    prefix_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for run_id, run_predictions in predictions.items():
        for sample_id in samples:
            predicted = run_predictions[sample_id]
            logits = np.asarray([[2.0, -2.0] if predicted == 0 else [-2.0, 2.0]])
            raw = probabilities_from_logits(logits)[0]
            temperature = 2.0
            calibrated = probabilities_from_logits(logits / temperature)[0]
            for prefix_ms in (150, 200, 250, 300):
                prefix_rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "speaker_id": speakers[sample_id],
                        "fold": 0,
                        "seed": 7,
                        "model_id": model_ids[run_id],
                        "split_role": "test",
                        "onset_variant": "causal_fixed",
                        "pre_roll_ms": 30,
                        "prefix_ms": prefix_ms,
                        "label_id": labels[sample_id],
                        "logit_control": logits[0, 0],
                        "logit_dysarthric": logits[0, 1],
                        "prob_control": raw[0],
                        "prob_dysarthric": raw[1],
                        "predicted_label": predicted,
                        "normalized_entropy": normalized_binary_entropy(raw[None, :])[0],
                        "temperature": temperature,
                        "calibrated_prob_control": calibrated[0],
                        "calibrated_prob_dysarthric": calibrated[1],
                        "calibrated_predicted_label": predicted,
                        "calibrated_normalized_entropy": normalized_binary_entropy(
                            calibrated[None, :]
                        )[0],
                        "onset_detected": sample_id != "c0",
                    }
                )
            decision_rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "policy_id": "complete-temperature_scaled",
                    "label_id": labels[sample_id],
                    "decision_time_ms": 200,
                    "predicted_label": predicted,
                    "forced_fallback": False,
                    "accepted_at_endpoint": False,
                }
            )

    prompts: list[object] = ["[say ah]", "cat", "this is a sentence", np.nan] * 2
    manifest = pd.DataFrame(
        {
            "sample_id": samples,
            "speaker_id": [speakers[value] for value in samples],
            "label_id": [labels[value] for value in samples],
            "diagnosis": ["dysarthric" if labels[value] else "control" for value in samples],
            "prompt_text": prompts,
            "duration_sec": [0.3, 0.7, 1.5, 5.2, 0.4, 0.8, 2.2, 4.8],
            "path": [f"audio/{value}.wav" for value in samples],
        }
    )
    return pd.DataFrame(prefix_rows), pd.DataFrame(decision_rows), manifest


def _overall_row(table: pd.DataFrame, model_id: str, error_type: str) -> pd.Series:
    selected = table[
        table["system_id"].str.contains(f"model_id={model_id}|", regex=False)
        & table["stratifier"].eq("overall")
        & table["error_type"].eq(error_type)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_rates_use_class_eligible_denominators_not_all_rows() -> None:
    prefix, decisions, manifest = _synthetic_inputs()
    audit = audit_classification_errors(prefix, decisions, manifest, review_per_error=1)

    fp_a = _overall_row(audit.per_system_errors, "model-a", "FP")
    fn_a = _overall_row(audit.per_system_errors, "model-a", "FN")
    assert fp_a["error_count"] == 2
    assert fp_a["eligible_denominator"] == 4
    assert fp_a["error_rate"] == 0.5
    assert fp_a["total_decisions_in_stratum"] == 8
    assert fn_a["error_count"] == 1
    assert fn_a["eligible_denominator"] == 4
    assert fn_a["error_rate"] == 0.25

    pooled_fp = audit.overall_errors.query(
        "stratifier == 'overall' and error_type == 'FP'"
    ).iloc[0]
    assert pooled_fp["error_count"] == 3
    assert pooled_fp["eligible_denominator"] == 8
    assert pooled_fp["error_rate"] == 0.375

    impossible_fn = audit.per_system_errors[
        audit.per_system_errors["system_id"].str.contains("model_id=model-a|", regex=False)
        & audit.per_system_errors["stratifier"].eq("diagnosis")
        & audit.per_system_errors["stratum"].eq("control")
        & audit.per_system_errors["error_type"].eq("FN")
    ].iloc[0]
    assert impossible_fn["eligible_denominator"] == 0
    assert np.isnan(impossible_fn["error_rate"])


def test_links_calibrated_prefix_and_derives_prespecified_strata() -> None:
    prefix, decisions, manifest = _synthetic_inputs()
    onset = pd.DataFrame(
        {
            "sample_id": ["c0", "c1", "c2"],
            "onset_mode": ["causal_fixed"] * 3,
            "detected": [True, False, True],
            "estimated_onset_sec": [0.12, np.nan, 0.20],
            "reference_onset_sec": [0.10, 0.11, np.nan],
        }
    )
    audit = audit_classification_errors(
        prefix,
        decisions,
        manifest,
        onset_table=onset,
        review_per_error=1,
    )
    enriched = audit.enriched_decisions
    c0 = enriched[enriched["sample_id"].eq("c0")]
    c1 = enriched[enriched["sample_id"].eq("c1")]
    c2 = enriched[enriched["sample_id"].eq("c2")]
    assert c0["onset_status"].eq("detected").all()
    assert c1["onset_status"].eq("failure").all()
    # The incomplete manual reference is excluded; prefix evidence remains available.
    assert c2["onset_status"].eq("detected").all()
    assert audit.summary["onset_table_audit"]["incomplete_reference_rows_excluded"] == 1
    assert np.allclose(c0["audit_onset_error_ms"], 20.0)

    assert set(enriched["prompt_type"]) == {
        "vocal_exercise",
        "isolated_token",
        "sentence",
        "missing_prompt_text",
    }
    assert enriched["duration_bin"].notna().all()
    assert enriched["entropy_bin"].notna().all()
    assert np.allclose(
        enriched["calibrated_prob_control"] + enriched["calibrated_prob_dysarthric"],
        1.0,
    )


def test_review_selection_is_deterministic_blinded_and_has_blank_taxonomy() -> None:
    prefix, decisions, manifest = _synthetic_inputs()
    first = audit_classification_errors(prefix, decisions, manifest, review_per_error=1)
    second = audit_classification_errors(
        prefix.sample(frac=1.0, random_state=3).reset_index(drop=True),
        decisions.sample(frac=1.0, random_state=4).reset_index(drop=True),
        manifest.sample(frac=1.0, random_state=5).reset_index(drop=True),
        review_per_error=1,
    )
    pdt.assert_frame_equal(first.qualitative_review, second.qualitative_review)
    pdt.assert_frame_equal(first.qualitative_review_key, second.qualitative_review_key)
    assert len(first.qualitative_review) == 4
    for column in REVIEW_TAXONOMY_COLUMNS:
        assert first.qualitative_review[column].eq("").all()
    forbidden = {"error_type", "label_id", "predicted_label", "system_id"}
    assert forbidden.isdisjoint(first.qualitative_review.columns)
    assert forbidden.issubset(first.qualitative_review_key.columns)
    assert first.summary["qualitative_review_note"].endswith(
        "No error cause or category is inferred from metadata."
    )


def test_output_manifest_hashes_inputs_and_outputs(tmp_path: Path) -> None:
    prefix, decisions, manifest = _synthetic_inputs()
    prefix_path = tmp_path / "prefix.csv"
    decision_path = tmp_path / "decisions.csv"
    frozen_path = tmp_path / "manifest.csv"
    prefix.to_csv(prefix_path, index=False)
    decisions.to_csv(decision_path, index=False)
    manifest.to_csv(frozen_path, index=False)

    audit = audit_classification_errors(prefix, decisions, manifest, review_per_error=1)
    paths = write_error_audit(
        audit,
        tmp_path / "audit",
        input_paths={
            "prefix_ledger": prefix_path,
            "decision_ledger": decision_path,
            "frozen_manifest": frozen_path,
        },
        project_root=tmp_path,
    )
    audit_manifest = json.loads(paths["audit_manifest"].read_text())
    assert audit_manifest["inputs"]["prefix_ledger"]["sha256"] == sha256_file(prefix_path)
    assert audit_manifest["causal_inference_performed"] is False
    for record in audit_manifest["outputs"].values():
        assert record["sha256"] == sha256_file(record["path"])
