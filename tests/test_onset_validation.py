from __future__ import annotations

import json

import numpy as np
import pandas as pd
import soundfile as sf

from timeentropy.analysis.onset_validation import (
    ANNOTATION_COLUMNS,
    ONSET_MODES,
    classify_prompt_type,
    completed_references,
    create_annotation_sample,
    evaluate_onset_annotations,
    normalize_saved_estimates,
    onset_perturbation_schedule,
    run_onset_estimators,
    write_annotation_sheet,
    write_evaluation_outputs,
)
from timeentropy.signal.onset import OnsetConfig


def _selection_manifest() -> pd.DataFrame:
    rows = []
    for speaker_index in range(15):
        speaker = f"S{speaker_index:02d}"
        prompts = (
            [("vocal", f"[say Ah {index} times]") for index in range(10)]
            + [("token", f"token{index}") for index in range(15)]
            + [("sentence", f"This is sentence number {index}") for index in range(15)]
        )
        for row_index, (_, prompt) in enumerate(prompts):
            rows.append(
                {
                    "sample_id": f"{speaker}-{row_index:03d}",
                    "speaker_id": speaker,
                    "label": "control" if speaker_index < 7 else "dysarthric",
                    "label_id": 0 if speaker_index < 7 else 1,
                    "session": "Session1",
                    "utterance_id": f"{row_index:04d}",
                    "prompt_text": prompt,
                    "path": f"/audio/{speaker}/{row_index:04d}.wav",
                    "duration_sec": 1.0,
                    "sample_rate": 16_000,
                }
            )
    return pd.DataFrame(rows)


def test_prompt_rules_and_deterministic_exact_300_row_sampling(tmp_path) -> None:
    assert classify_prompt_type("[say Ah for five seconds]") == "vocal_exercise"
    assert classify_prompt_type('read [as in "I can read"]') == "isolated_token"
    assert classify_prompt_type("The quick fox jumps") == "sentence"

    manifest = _selection_manifest()
    missing_prompt_id = manifest.loc[0, "sample_id"]
    manifest.loc[0, "prompt_text"] = np.nan
    first = create_annotation_sample(manifest, seed=41)
    shuffled = create_annotation_sample(manifest.sample(frac=1.0, random_state=3), seed=41)
    pd.testing.assert_frame_equal(first, shuffled)
    assert len(first) == 300
    assert missing_prompt_id not in set(first["sample_id"])
    assert first.groupby("speaker_id").size().eq(20).all()
    strata = first.groupby(["speaker_id", "prompt_type"]).size().unstack()
    assert strata["vocal_exercise"].eq(6).all()
    assert strata["isolated_token"].eq(7).all()
    assert strata["sentence"].eq(7).all()
    assert first["reference_onset_sec"].eq("").all()
    assert first[["annotator", "annotation_status", "notes"]].eq("").all().all()
    assert tuple(first.columns) == ANNOTATION_COLUMNS

    csv_path, schema_path = write_annotation_sheet(first, tmp_path / "annotations.csv")
    assert csv_path.is_file() and schema_path.is_file()
    schema = json.loads(schema_path.read_text())
    assert schema["manual_reference_policy"].startswith("reference_onset_sec")


def test_prompt_shortage_is_redistributed_without_breaking_speaker_balance() -> None:
    manifest = _selection_manifest()
    speaker = "S00"
    vocal = manifest["prompt_text"].str.startswith("[")
    keep_one = manifest.loc[vocal & manifest["speaker_id"].eq(speaker)].index[:1]
    drop = manifest.index[vocal & manifest["speaker_id"].eq(speaker)].difference(keep_one)
    sample = create_annotation_sample(manifest.drop(drop), seed=9)
    counts = sample.loc[sample["speaker_id"].eq(speaker), "prompt_type"].value_counts()
    assert counts.sum() == 20
    assert counts["vocal_exercise"] == 1
    assert sorted([counts["isolated_token"], counts["sentence"]]) == [9, 10]


def _annotation_fixture() -> pd.DataFrame:
    rows = []
    for index in range(7):
        rows.append(
            {
                "sample_id": f"a{index}",
                "speaker_id": f"S{index % 3}",
                "diagnosis": "control" if index % 2 == 0 else "dysarthric",
                "prompt_text": "one" if index % 2 == 0 else "A complete sentence",
                "prompt_type": "isolated_token" if index % 2 == 0 else "sentence",
                "path": f"/audio/a{index}.wav",
                "duration_sec": 1.0,
                "reference_onset_sec": [0.10, 0.20, 0.30, 0.40, "", -0.1, 0.6][index],
                "annotation_status": "excluded" if index == 6 else "",
            }
        )
    return pd.DataFrame(rows)


def _estimate_fixture() -> pd.DataFrame:
    rows = []
    offsets = {
        "causal_fixed": 0.010,
        "causal_noise_floor": -0.020,
        "recovered_offline_peak": 0.050,
    }
    references = {"a0": 0.10, "a1": 0.20, "a2": 0.30, "a3": 0.40}
    for sample_id, reference in references.items():
        for mode, offset in offsets.items():
            failed = sample_id == "a3" and mode == "causal_fixed"
            rows.append(
                {
                    "sample_id": sample_id,
                    "onset_mode": mode,
                    "detected": not failed,
                    "estimated_onset_sec": np.nan if failed else reference + offset,
                }
            )
    return pd.DataFrame(rows)


def test_incomplete_manual_rows_are_counted_not_imputed_and_metrics_are_grouped(tmp_path) -> None:
    annotations = _annotation_fixture()
    references, audit = completed_references(annotations)
    assert references["sample_id"].tolist() == ["a0", "a1", "a2", "a3"]
    assert (audit.total_rows, audit.completed_rows) == (7, 4)
    assert (audit.incomplete_rows, audit.invalid_rows, audit.excluded_rows) == (1, 1, 1)

    first = evaluate_onset_annotations(
        annotations,
        _estimate_fixture(),
        bootstrap_seed=123,
        bootstrap_replicates=100,
    )
    second = evaluate_onset_annotations(
        annotations.sample(frac=1.0, random_state=5),
        _estimate_fixture().sample(frac=1.0, random_state=6),
        bootstrap_seed=123,
        bootstrap_replicates=100,
    )
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    assert set(first.metrics["group_dimension"]) == {
        "overall",
        "diagnosis",
        "speaker",
        "prompt_type",
    }
    causal = first.metrics.loc[
        first.metrics["group_dimension"].eq("overall")
        & first.metrics["onset_mode"].eq("causal_fixed")
    ].iloc[0]
    assert causal["reference_count"] == 4
    assert causal["failure_count"] == 1
    assert causal["failure_rate"] == 0.25
    assert abs(causal["signed_bias_ms"] - 10.0) < 1.0e-9
    assert causal["within_20_ms_rate"] == 0.75
    assert causal["failure_rate_ci_low"] <= causal["failure_rate"]
    assert causal["failure_rate_ci_high"] >= causal["failure_rate"]

    csv_path, json_path = write_evaluation_outputs(
        first, tmp_path / "metrics.csv", tmp_path / "summary.json"
    )
    assert csv_path.is_file() and json_path.is_file()
    summary = json.loads(json_path.read_text())
    assert summary["reference_audit"]["incomplete_rows"] == 1


def test_wide_saved_estimates_are_normalized() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a"],
            "causal_fixed_onset_sec": [0.1],
            "causal_noise_floor_onset_sec": [np.nan],
            "causal_noise_floor_detected": [False],
            "recovered_offline_peak_onset_sec": [0.12],
        }
    )
    normalized = normalize_saved_estimates(frame)
    assert len(normalized) == 3
    failed = normalized.loc[normalized["onset_mode"].eq("causal_noise_floor")].iloc[0]
    assert not failed["detected"]


def test_entirely_blank_reference_sheet_stays_empty_and_is_fully_audited() -> None:
    annotations = _annotation_fixture().iloc[:3].copy()
    annotations["reference_onset_sec"] = ""
    references, audit = completed_references(annotations)
    assert references.empty and audit.incomplete_rows == 3
    estimates = run_onset_estimators(references, onset_config=OnsetConfig(sample_rate=1_000))
    evaluation = evaluate_onset_annotations(
        annotations,
        estimates,
        bootstrap_replicates=10,
    )
    assert evaluation.metrics.empty
    assert len(evaluation.metrics.columns) > 10


def test_estimators_run_all_modes_on_synthetic_audio(tmp_path) -> None:
    path = tmp_path / "onset.wav"
    waveform = np.concatenate(
        [np.zeros(100, dtype=np.float32), np.ones(300, dtype=np.float32) * 0.1]
    )
    sf.write(path, waveform, 1_000, subtype="FLOAT")
    references = pd.DataFrame({"sample_id": ["x"], "path": [str(path)]})
    estimates = run_onset_estimators(
        references,
        onset_config=OnsetConfig(sample_rate=1_000),
    )
    assert tuple(estimates["onset_mode"]) == ONSET_MODES
    assert estimates["detected"].all()
    assert estimates["estimated_onset_sample"].eq(90).all()


def test_registered_perturbations_clip_at_waveform_boundaries() -> None:
    schedule = onset_perturbation_schedule(
        10,
        num_samples=40,
        sample_rate=1_000,
    )
    assert [item.requested_offset_ms for item in schedule] == [-50, -30, -10, 0, 10, 30, 50]
    assert [item.clipped_onset_sample for item in schedule] == [0, 0, 0, 10, 20, 39, 39]
    assert [item.clipped for item in schedule] == [True, True, False, False, False, True, True]
