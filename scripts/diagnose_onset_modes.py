#!/usr/bin/env python3
"""Label-free comparison of the frozen onset-detector modes.

This diagnostic uses only the waveform. It never reads a label, a model output, or a
test prediction, so it may be run before test evaluation without leaking anything into
the frozen protocol. It answers one question: does a mode actually localise speech
onset, or does it fire on the recording's noise floor?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from timeentropy.config import load_yaml, project_path
from timeentropy.signal.onset import OnsetConfig, detect_onset, frame_rms

MODES = ("causal_fixed", "causal_noise_floor", "recovered_offline_peak")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def diagnose_utterance(wav: np.ndarray, sample_rate: int, config: OnsetConfig,
                       window_ms: int) -> list[dict]:
    frames = frame_rms(wav, 320, 160)
    noise_floor = float(np.median(frames[:10]))
    window = int(window_ms / 1000.0 * sample_rate)
    rows = []
    for mode in MODES:
        result = detect_onset(wav, config, mode=mode)
        onset = result.onset_sample if result.detected else 0
        post = _rms(wav[onset : onset + window])
        pre = _rms(wav[max(0, onset - window) : onset])
        rows.append(
            {
                "mode": mode,
                "detected": result.detected,
                "onset_ms": onset / sample_rate * 1000.0,
                "fires_at_zero": onset == 0,
                "noise_floor_rms": noise_floor,
                "threshold": result.threshold,
                "post_onset_rms": post,
                "pre_onset_rms": pre,
                "post_over_pre_db": (
                    np.nan if onset == 0 else 20.0 * np.log10(max(post, 1e-9) / max(pre, 1e-9))
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rebuild_v1.yaml"))
    parser.add_argument("--per-speaker", type=int, default=40)
    parser.add_argument("--window-ms", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_yaml(args.config)
    manifest = pd.read_csv(project_path(config["data"]["manifest"]))
    onset_config = OnsetConfig()

    sampled = manifest.groupby("speaker_id", group_keys=False).apply(
        lambda group: group.sample(min(args.per_speaker, len(group)), random_state=args.seed),
        include_groups=False,
    ).join(manifest[["speaker_id"]])

    records = []
    for row in sampled.itertuples(index=False):
        wav, sample_rate = sf.read(project_path(row.path), dtype="float32")
        for record in diagnose_utterance(wav, sample_rate, onset_config, args.window_ms):
            record.update({"sample_id": row.sample_id, "speaker_id": row.speaker_id})
            records.append(record)

    frame = pd.DataFrame(records)
    quiet_threshold = 0.01
    summary = {}
    for mode, group in frame.groupby("mode"):
        finite = group[np.isfinite(group["post_over_pre_db"])]
        summary[mode] = {
            "utterances": len(group),
            "fires_at_zero_rate": float(group["fires_at_zero"].mean()),
            "median_onset_ms": float(group["onset_ms"].median()),
            "median_post_onset_rms": float(group["post_onset_rms"].median()),
            "quiet_prefix_rate": float((group["post_onset_rms"] < quiet_threshold).mean()),
            "median_post_over_pre_db": (
                float(finite["post_over_pre_db"].median()) if len(finite) else None
            ),
        }

    summary["_meta"] = {
        "uses_labels": False,
        "uses_model_outputs": False,
        "quiet_prefix_rms_threshold": quiet_threshold,
        "window_ms": args.window_ms,
        "per_speaker": args.per_speaker,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_dir / "onset_mode_diagnostic.csv", index=False)
        (args.output_dir / "onset_mode_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
