#!/usr/bin/env python3
"""Ask where the condition-matched onset-mode accuracy difference actually comes from.

Condition-matched retraining shows the two comparator onset rules classifying better than
the adaptive primary rule. Those comparators frequently anchor at sample zero, so their
crops begin at the start of the recording rather than at speech. This audit tests whether
the accuracy difference concentrates in exactly those recording-start crops, and whether it
falls on control or dysarthric utterances.

It is a stratification, not a causal test: it can show that the difference lives in the
recording-start stratum, which would mean the gain is not evidence about speech onset, but
it cannot prove what in that stratum carries the signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from timeentropy.evaluation.ledger import validate_decision_ledger
from timeentropy.provenance import environment_record, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "complete-temperature_scaled"
PRIMARY_LABEL = "causal_noise_floor"
CONDITIONS = {
    "causal_fixed": "artifacts/robustness/retrained/causal_fixed_preroll30",
    "recovered_offline_peak": "artifacts/robustness/retrained/offline_peak_preroll30",
}
PRIMARY_DIR = "artifacts/evaluation/transformer-primary"


def decisions(directory: Path) -> pd.DataFrame:
    frame = pd.read_parquet(directory / "decisions.parquet")
    validate_decision_ledger(frame)
    chosen = frame.loc[frame["policy_id"].eq(POLICY_ID)].copy()
    chosen["correct"] = chosen["predicted_label"].eq(chosen["label_id"])
    return chosen.set_index("sample_id").sort_index()


def anchors(directory: Path) -> pd.DataFrame:
    """One anchor row per utterance: the anchor is a property of the rule, not the prefix.

    A registered detector failure stores `onset_sample = -1` as a sentinel, but the crop it
    produces starts at sample zero, exactly like a detected zero anchor. `effective_anchor`
    is therefore what the model actually saw, and the two cases are kept distinguishable.
    """

    prefix = pd.read_parquet(directory / "calibrated_prefix_predictions.parquet")
    test = prefix.loc[prefix["split_role"].eq("test")]
    unique = test.groupby("sample_id").agg(
        onset_sample=("onset_sample", "first"),
        onset_detected=("onset_detected", "first"),
        onset_variant=("onset_variant", "first"),
    )
    detected = unique["onset_detected"].astype(bool)
    unique["effective_anchor"] = unique["onset_sample"].where(detected, 0)
    if (unique.loc[detected, "onset_sample"] < 0).any():
        raise SystemExit(f"Negative anchor on a detected row in {directory}")
    unique["starts_at_recording_start"] = unique["effective_anchor"].eq(0)
    return unique.sort_index()


def stratified_gain(
    primary: pd.DataFrame, condition: pd.DataFrame, anchor: pd.DataFrame, label: str
) -> pd.DataFrame:
    joined = pd.DataFrame(
        {
            "label_id": primary["label_id"],
            "speaker_id": primary["speaker_id"],
            "primary_correct": primary["correct"],
            "condition_correct": condition["correct"],
            "anchor_sample": anchor["effective_anchor"],
            "anchor_detected": anchor["onset_detected"].astype(bool),
        }
    )
    joined["stratum"] = [
        "anchor after sample 0"
        if sample > 0
        else "detected anchor at sample 0"
        if detected
        else "registered failure, crop from sample 0"
        for sample, detected in zip(joined["anchor_sample"], joined["anchor_detected"])
    ]
    joined["true_class"] = joined["label_id"].map({0: "control", 1: "dysarthric"})

    rows = []
    for keys, group in joined.groupby(["stratum", "true_class"], dropna=False):
        rows.append(
            {
                "condition": label,
                "stratum": keys[0],
                "true_class": keys[1],
                "utterances": len(group),
                "primary_accuracy": float(group["primary_correct"].mean()),
                "condition_accuracy": float(group["condition_correct"].mean()),
                "accuracy_difference": float(
                    group["condition_correct"].mean() - group["primary_correct"].mean()
                ),
            }
        )
    for stratum, group in joined.groupby("stratum", dropna=False):
        rows.append(
            {
                "condition": label,
                "stratum": stratum,
                "true_class": "all",
                "utterances": len(group),
                "primary_accuracy": float(group["primary_correct"].mean()),
                "condition_accuracy": float(group["condition_correct"].mean()),
                "accuracy_difference": float(
                    group["condition_correct"].mean() - group["primary_correct"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def anchor_distribution(anchor: pd.DataFrame, primary: pd.DataFrame, label: str) -> dict[str, float]:
    joined = anchor.join(primary["label_id"])
    at_zero = anchor["starts_at_recording_start"]
    return {
        "rule": label,
        "utterances": len(anchor),
        "recording_start_rate": float(at_zero.mean()),
        "recording_start_rate_control": float(at_zero[joined["label_id"].eq(0)].mean()),
        "recording_start_rate_dysarthric": float(at_zero[joined["label_id"].eq(1)].mean()),
        "class_gap_points": float(
            100.0
            * (
                at_zero[joined["label_id"].eq(0)].mean()
                - at_zero[joined["label_id"].eq(1)].mean()
            )
        ),
        "median_effective_anchor_ms": float(anchor["effective_anchor"].median() / 16.0),
        "detector_failure_rate": float((~anchor["onset_detected"].astype(bool)).mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/onset_diagnosis/onset_mode_confound"
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    primary_dir = PROJECT_ROOT / PRIMARY_DIR
    primary = decisions(primary_dir)
    primary_anchor = anchors(primary_dir)

    gains, distributions, speaker_rows = [], [anchor_distribution(primary_anchor, primary, PRIMARY_LABEL)], []
    for label, relative in CONDITIONS.items():
        directory = PROJECT_ROOT / relative
        condition = decisions(directory)
        anchor = anchors(directory)
        if not primary.index.equals(condition.index):
            raise SystemExit(f"{label} is not paired with the primary system on identical samples")
        gains.append(stratified_gain(primary, condition, anchor, label))
        distributions.append(anchor_distribution(anchor, primary, label))
        for speaker, group in pd.DataFrame(
            {
                "speaker_id": primary["speaker_id"],
                "label_id": primary["label_id"],
                "primary_correct": primary["correct"],
                "condition_correct": condition["correct"],
            }
        ).groupby("speaker_id"):
            speaker_rows.append(
                {
                    "condition": label,
                    "speaker_id": speaker,
                    "true_class": "dysarthric" if group["label_id"].iloc[0] == 1 else "control",
                    "utterances": len(group),
                    "primary_accuracy": float(group["primary_correct"].mean()),
                    "condition_accuracy": float(group["condition_correct"].mean()),
                    "accuracy_difference": float(
                        group["condition_correct"].mean() - group["primary_correct"].mean()
                    ),
                }
            )

    gain_path = arguments.output_dir / "accuracy_difference_by_anchor_stratum.csv"
    pd.concat(gains, ignore_index=True).to_csv(gain_path, index=False)
    distribution_path = arguments.output_dir / "anchor_distribution_by_rule.csv"
    pd.DataFrame(distributions).to_csv(distribution_path, index=False)
    speaker_path = arguments.output_dir / "accuracy_difference_by_speaker.csv"
    pd.DataFrame(speaker_rows).to_csv(speaker_path, index=False)

    outputs = [gain_path, distribution_path, speaker_path]
    summary = {
        "schema_version": "onset-mode-confound-audit-v1",
        "policy_id": POLICY_ID,
        "primary_rule": PRIMARY_LABEL,
        "compared_rules": sorted(CONDITIONS),
        "test_utterances": len(primary),
        "interpretation_limits": (
            "Strata are associations. A difference concentrated in recording-start crops shows "
            "that the comparator advantage is not evidence about speech-onset localization, but "
            "this audit does not identify what in those crops carries the signal."
        ),
        "outputs": {
            path.name: {
                "path": str(path.resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in outputs
        },
        "environment": environment_record(PROJECT_ROOT),
    }
    summary_path = arguments.output_dir / "confound_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for path in [*outputs, summary_path]:
        print(path)


if __name__ == "__main__":
    main()
