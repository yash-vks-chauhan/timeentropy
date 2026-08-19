from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import yaml

from timeentropy.config import load_yaml
from timeentropy.data.dataset import PrefixFeatureDataset
from timeentropy.evaluation.calibration import probabilities_from_logits
from timeentropy.evaluation.entropy import normalized_binary_entropy
from timeentropy.models import create_model
from timeentropy.provenance import sha256_json
from timeentropy.robustness.config import (
    RobustnessAnalysis,
    derived_retraining_config,
    load_robustness_protocol,
    write_retraining_configs,
)
from timeentropy.robustness.dataset import (
    RobustnessPrefixFeatureDataset,
    collate_robustness_prefix_features,
)
from timeentropy.robustness.perturbations import (
    add_white_gaussian_noise,
    shifted_crop_anchor,
)
from timeentropy.robustness.retrained import evaluate_retrained_ablation_ledgers
from timeentropy.robustness.summary import summarize_test_time_robustness
from timeentropy.robustness.test_time import (
    _select_clean_reference_policy,
    run_test_time_robustness,
)
from timeentropy.signal.onset import OnsetConfig


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def _reference_ledger(run_id: str = "reference-fold0") -> pd.DataFrame:
    rows: list[dict] = []
    samples = (
        ("calibration", "c0", "C0", 0),
        ("calibration", "c1", "D0", 1),
        ("policy_tuning", "p0", "C1", 0),
        ("policy_tuning", "p1", "D1", 1),
        ("test", "t0", "C2", 0),
        ("test", "t1", "D2", 1),
    )
    for split_role, sample_id, speaker_id, label_id in samples:
        for prefix_ms in (150, 200, 250, 300):
            strength = 0.3 + prefix_ms / 300.0
            logits = np.asarray(
                [[strength, -strength] if label_id == 0 else [-strength, strength]],
                dtype=np.float64,
            )
            probabilities = probabilities_from_logits(logits)[0]
            rows.append(
                {
                    "run_id": run_id,
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
                    "onset_detected": True,
                    "onset_sample": 100,
                    "confirmation_sample": 130,
                    "onset_threshold": 0.003,
                    "source_sample_rate": 16_000,
                    "source_channels": 1,
                    "input_num_samples": (prefix_ms + 30) * 16,
                }
            )
    return pd.DataFrame(rows)


def _temporary_protocol(tmp_path: Path) -> tuple[Path, dict, Path, Path]:
    parent = load_yaml("configs/rebuild_v1.yaml")
    manifest_path = tmp_path / "manifest.csv"
    fold_path = tmp_path / "folds.csv"
    audio_paths: list[str] = []
    for index, sample_id in enumerate(("t0", "t1")):
        silence = np.zeros(1_600 + index * 80, dtype=np.float32)
        speech = 0.1 * np.sin(np.linspace(0.0, 100.0, 6_400, dtype=np.float32))
        path = tmp_path / f"{sample_id}.wav"
        sf.write(path, np.concatenate([silence, speech]), 16_000, subtype="FLOAT")
        audio_paths.append(str(path))
    pd.DataFrame(
        {
            "sample_id": ["t0", "t1"],
            "path": audio_paths,
            "speaker_id": ["C2", "D2"],
            "label_id": [0, 1],
        }
    ).to_csv(manifest_path, index=False)
    pd.DataFrame(
        {
            "fold": [0, 0],
            "sample_id": ["t0", "t1"],
            "role": ["test", "test"],
        }
    ).to_csv(fold_path, index=False)
    parent["data"]["manifest"] = str(manifest_path)
    parent["data"]["fold_manifest"] = str(fold_path)
    parent_path = tmp_path / "parent.yaml"
    parent_path.write_text(yaml.safe_dump(parent, sort_keys=False))

    robustness = load_yaml("configs/robustness_v1.yaml")
    robustness["parent_experiment_config"] = str(parent_path)
    robustness["scope"]["folds"] = [0]
    robustness_path = tmp_path / "robustness.yaml"
    robustness_path.write_text(yaml.safe_dump(robustness, sort_keys=False))
    return robustness_path, parent, manifest_path, fold_path


def test_frozen_protocol_separates_training_and_test_time_conditions(tmp_path) -> None:
    protocol = load_robustness_protocol("configs/robustness_v1.yaml")
    assert len(protocol.retrained_analyses) == 4
    assert len(protocol.test_time_analyses) == 11
    assert sum(item.requires_new_training for item in protocol.retrained_analyses) == 3
    assert all(not item.requires_new_training for item in protocol.test_time_analyses)
    assert len(protocol.plan_rows()) == 75
    assert not any(row["test_set_tuning_permitted"] for row in protocol.plan_rows())

    derived = derived_retraining_config(protocol, "causal_fixed_preroll0")
    assert derived["onset"]["pre_roll_ms"] == 0
    assert derived["onset"]["primary_mode"] == "causal_fixed"
    assert derived["robustness_retrained_input_ablation"]["requires_condition_matched_checkpoint"]
    with pytest.raises(ValueError, match="must not receive"):
        derived_retraining_config(protocol, "clean_reference")

    paths = write_retraining_configs(protocol, tmp_path / "derived")
    assert set(paths) == {
        "causal_fixed_preroll0",
        "offline_peak_preroll30",
        "causal_noise_floor_preroll30",
        "manifest",
    }
    assert write_retraining_configs(protocol, tmp_path / "derived") == paths


def test_protocol_rejects_changed_randomization_contract(tmp_path) -> None:
    robustness_path, _, _, _ = _temporary_protocol(tmp_path)
    robustness = yaml.safe_load(robustness_path.read_text())
    robustness["evaluation_only_noise"]["randomization"] = "row_order_rng"
    robustness_path.write_text(yaml.safe_dump(robustness, sort_keys=False))
    with pytest.raises(ValueError, match="randomization contract"):
        load_robustness_protocol(robustness_path)


def test_white_noise_is_deterministic_and_has_requested_snr() -> None:
    waveform = np.sin(np.linspace(0.0, 40.0, 4_000, dtype=np.float64)).astype(np.float32)
    first = add_white_gaussian_noise(
        waveform,
        primary_seed=19,
        sample_id="sample-a",
        snr_db=10.0,
    )
    second = add_white_gaussian_noise(
        waveform,
        primary_seed=19,
        sample_id="sample-a",
        snr_db=10.0,
    )
    other = add_white_gaussian_noise(
        waveform,
        primary_seed=19,
        sample_id="sample-b",
        snr_db=10.0,
    )
    np.testing.assert_array_equal(first.waveform, second.waveform)
    assert first.seed_sha256 == second.seed_sha256
    assert first.realized_snr_db == pytest.approx(10.0, abs=1.0e-12)
    assert not np.array_equal(first.waveform, other.waveform)

    silent = add_white_gaussian_noise(
        np.zeros(100, dtype=np.float32),
        primary_seed=19,
        sample_id="silent",
        snr_db=0.0,
    )
    assert not silent.applied and silent.realized_snr_db is None
    assert not silent.waveform.any()


def test_shifted_anchor_clips_at_both_boundaries() -> None:
    assert shifted_crop_anchor(
        reference_anchor_sample=10,
        waveform_samples=100,
        offset_ms=-50,
        sample_rate=1_000,
    ) == (-40, 0)
    assert shifted_crop_anchor(
        reference_anchor_sample=90,
        waveform_samples=100,
        offset_ms=50,
        sample_rate=1_000,
    ) == (140, 100)


def test_robustness_dataset_preserves_clean_path_and_audits_shift(tmp_path) -> None:
    waveform = np.concatenate(
        [
            np.zeros(100, dtype=np.float32),
            0.1 * np.sin(np.linspace(0.0, 30.0, 500, dtype=np.float32)),
        ]
    )
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, waveform, 1_000, subtype="FLOAT")
    manifest = pd.DataFrame(
        {
            "sample_id": ["fixture"],
            "path": [str(audio_path)],
            "speaker_id": ["S1"],
            "label_id": [1],
        }
    )
    options = {
        "prefixes_ms": (150, 200),
        "target_sample_rate": 1_000,
        "pre_roll_ms": 30,
        "onset_config": OnsetConfig(sample_rate=1_000),
        "onset_mode": "causal_fixed",
        "n_fft": 64,
    }
    clean_analysis = RobustnessAnalysis(
        analysis_id="clean",
        family="test_time",
        role="clean_reference",
        reference_checkpoint="primary",
        onset_mode="causal_fixed",
        pre_roll_ms=30,
    )
    standard = PrefixFeatureDataset(manifest, training=False, **options)
    robust_clean = RobustnessPrefixFeatureDataset(
        manifest,
        analysis=clean_analysis,
        primary_seed=7,
        **options,
    )
    torch.testing.assert_close(standard[0]["features"], robust_clean[0]["features"])

    shift_analysis = RobustnessAnalysis(
        analysis_id="shift50",
        family="test_time",
        role="onset_sensitivity",
        reference_checkpoint="primary",
        onset_mode="causal_fixed",
        pre_roll_ms=30,
        perturbation_kind="onset_shift",
        onset_offset_ms=50,
    )
    shifted = RobustnessPrefixFeatureDataset(
        manifest,
        analysis=shift_analysis,
        primary_seed=7,
        **options,
    )[0]
    assert shifted["reference_onset_sample"] == 90
    assert shifted["requested_crop_anchor_sample"] == 140
    assert shifted["crop_anchor_sample"] == 140
    assert shifted["onset_offset_applied"]
    batch = collate_robustness_prefix_features([shifted])
    assert batch["robustness_analysis_id"] == ["shift50"]
    assert batch["reference_onset_sample"].tolist() == [90]


def test_clean_policy_is_invariant_to_test_rows() -> None:
    protocol = load_robustness_protocol("configs/robustness_v1.yaml")
    original = _reference_ledger()
    changed = original.copy()
    test = changed["split_role"].eq("test")
    changed.loc[test, "label_id"] = 1 - changed.loc[test, "label_id"]
    changed.loc[test, "logit_control"] = 100.0
    changed.loc[test, "logit_dysarthric"] = -100.0
    assert _select_clean_reference_policy(original, protocol) == (
        _select_clean_reference_policy(changed, protocol)
    )


def test_clean_test_time_run_records_zero_test_tuning(tmp_path) -> None:
    robustness_path, parent, _, _ = _temporary_protocol(tmp_path)
    protocol = load_robustness_protocol(robustness_path)
    reference = _reference_ledger()
    ledger_path = tmp_path / "reference.csv"
    reference.to_csv(ledger_path, index=False)

    model = create_model("transformer", dropout=float(parent["training"]["dropout"]))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "metadata": {
                "run_id": "reference-fold0",
                "fold": 0,
                "seed": 20260819,
                "model_id": "transformer",
                "onset_variant": "causal_fixed",
            },
            "hashes": {
                "experiment_config_sha256": protocol.parent_config_sha256,
                "model_architecture_sha256": sha256_json(asdict(model.metadata)),
            },
        },
        checkpoint_path,
    )
    result = run_test_time_robustness(
        robustness_path,
        fold=0,
        checkpoint_path=checkpoint_path,
        reference_ledger_path=ledger_path,
        output_directory=tmp_path / "output",
        analysis_ids=("clean_reference",),
        device_preference="cpu",
        batch_size=2,
    )
    prefix = _read_table(result.prefix_ledger_path)
    decisions = _read_table(result.decision_ledger_path)
    metrics = pd.read_csv(result.metrics_path)
    summary = json.loads(result.summary_path.read_text())
    assert len(prefix) == 8
    assert len(decisions) == 10  # two samples x (adaptive + four fixed policies)
    assert len(metrics) == 5
    assert metrics["test_rows_used_for_tuning"].eq(0).all()
    assert summary["test_rows_used_for_temperature_fit"] == 0
    assert summary["test_rows_used_for_policy_selection"] == 0
    assert summary["clean_reference_sources"] == {
        "temperature": "calibration",
        "policy": "policy_tuning",
    }
    combined = summarize_test_time_robustness(
        robustness_path,
        decision_ledger_paths=(result.decision_ledger_path,),
        output_directory=tmp_path / "combined",
        require_all_conditions=False,
    )
    assert len(pd.read_csv(combined.pooled_metrics_path)) == 5
    assert len(pd.read_csv(combined.fold_metrics_path)) == 5
    combined_summary = json.loads(combined.summary_path.read_text())
    assert combined_summary["parameters_fitted_during_summary"] == []
    with pytest.raises(ValueError, match="duplicate"):
        summarize_test_time_robustness(
            robustness_path,
            decision_ledger_paths=(
                result.decision_ledger_path,
                result.decision_ledger_path,
            ),
            output_directory=tmp_path / "duplicates",
            require_all_conditions=False,
        )


def test_perturbed_run_exports_noise_and_zero_shift_audit(tmp_path) -> None:
    robustness_path, parent, _, _ = _temporary_protocol(tmp_path)
    protocol = load_robustness_protocol(robustness_path)
    ledger_path = tmp_path / "reference.csv"
    _reference_ledger().to_csv(ledger_path, index=False)
    model = create_model("transformer", dropout=float(parent["training"]["dropout"]))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "metadata": {
                "run_id": "reference-fold0",
                "fold": 0,
                "seed": 20260819,
                "model_id": "transformer",
                "onset_variant": "causal_fixed",
            },
            "hashes": {
                "experiment_config_sha256": protocol.parent_config_sha256,
                "model_architecture_sha256": sha256_json(asdict(model.metadata)),
            },
        },
        checkpoint_path,
    )
    result = run_test_time_robustness(
        robustness_path,
        fold=0,
        checkpoint_path=checkpoint_path,
        reference_ledger_path=ledger_path,
        output_directory=tmp_path / "perturbed",
        analysis_ids=("white_noise_snr_p20_db", "onset_shift_p0_ms"),
        device_preference="cpu",
        batch_size=3,
    )
    prefix = _read_table(result.prefix_ledger_path)
    noise = prefix.loc[prefix["perturbation_kind"].eq("white_noise")]
    zero_shift = prefix.loc[prefix["robustness_analysis_id"].eq("onset_shift_p0_ms")]
    assert len(prefix) == 16
    assert noise["noise_applied"].all()
    assert noise["noise_seed_sha256"].str.len().eq(64).all()
    assert np.allclose(noise["realized_snr_db"], 20.0, atol=1.0e-10)
    assert zero_shift["onset_offset_applied"].all()
    assert zero_shift["onset_sample"].eq(zero_shift["reference_onset_sample"]).all()
    summary = json.loads(result.summary_path.read_text())
    assert summary["zero_shift_reproduction_audit"]["rows"] == 8
    assert summary["zero_shift_reproduction_audit"]["informational_only"]


def test_retrained_evaluator_refuses_test_time_analysis_before_file_access() -> None:
    with pytest.raises(ValueError, match="test-time perturbation"):
        evaluate_retrained_ablation_ledgers(
            "configs/robustness_v1.yaml",
            analysis_id="clean_reference",
            prefix_ledger_paths=(),
            run_summary_paths=(),
            output_directory="unused",
        )


def test_retrained_evaluator_verifies_run_and_uses_standard_validation_pipeline(
    tmp_path,
) -> None:
    robustness_path, _, _, _ = _temporary_protocol(tmp_path)
    protocol = load_robustness_protocol(robustness_path)
    ledger_path = tmp_path / "primary-ledger.csv"
    _reference_ledger().to_csv(ledger_path, index=False)
    checkpoint_path = tmp_path / "primary-checkpoint.pt"
    checkpoint_path.write_bytes(b"synthetic checkpoint provenance fixture")
    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "reference-fold0",
                "fold": 0,
                "model_id": "transformer",
                "seed": 20260819,
                "checkpoint_path": str(checkpoint_path),
                "ledger_path": str(ledger_path),
                "hashes": {
                    "experiment_config_sha256": protocol.parent_config_sha256,
                },
            }
        )
    )
    result = evaluate_retrained_ablation_ledgers(
        robustness_path,
        analysis_id="primary_causal_fixed_preroll30",
        prefix_ledger_paths=(ledger_path,),
        run_summary_paths=(run_summary_path,),
        output_directory=tmp_path / "retrained-evaluation",
    )
    summary = json.loads(result.robustness_summary_path.read_text())
    assert summary["temperature_fit_split"] == "calibration"
    assert summary["policy_selection_split"] == "policy_tuning"
    assert summary["test_rows_used_for_tuning"] == 0
    selections = pd.read_csv(result.evaluation_paths["selections"])
    assert set(selections["policy_variant"]) == {"complete"}
