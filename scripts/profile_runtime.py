from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from timeentropy.config import PROJECT_ROOT
from timeentropy.models import create_model
from timeentropy.profiling import profile_pipeline, write_profile_outputs
from timeentropy.signal.onset import OnsetConfig


def _synthetic_waveform(sample_rate: int, seed: int = 20260819) -> np.ndarray:
    rng = np.random.default_rng(seed)
    silence = np.zeros(round(0.10 * sample_rate), dtype=np.float32)
    time_axis = np.arange(round(0.70 * sample_rate), dtype=np.float32) / sample_rate
    speech = 0.08 * np.sin(2.0 * np.pi * 180.0 * time_axis)
    speech += rng.normal(0.0, 0.005, size=len(time_axis)).astype(np.float32)
    return np.concatenate([silence, speech.astype(np.float32)])


def _load_waveform(path: Path | None, sample_rate: int) -> np.ndarray:
    if path is None:
        return _synthetic_waveform(sample_rate)
    values, actual_rate = sf.read(path, dtype="float32", always_2d=True)
    if actual_rate != sample_rate:
        raise ValueError(
            f"Audio sample rate is {actual_rate}; profiler requires frozen rate {sample_rate}"
        )
    return values.mean(axis=1, dtype=np.float32)


def _load_checkpoint(model: torch.nn.Module, path: Path | None) -> None:
    if path is None:
        return
    value = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = value
    if isinstance(value, dict):
        # Training writes the weights under "model_state_dict" alongside optimizer state,
        # provenance hashes, and metadata; "state_dict" is accepted for bare exports.
        for key in ("model_state_dict", "state_dict"):
            if key in value:
                state_dict = value[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    model.load_state_dict(state_dict, strict=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile practical waveform-to-decision compute latency and complexity."
    )
    parser.add_argument(
        "--model", choices=("small_cnn", "cnn_bilstm", "transformer"), default="transformer"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--audio", type=Path, help="Optional 16 kHz audio; synthetic if omitted")
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--theta", type=float, default=0.90)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument(
        "--policy-variant",
        choices=(
            "entropy_only",
            "entropy_class_stability",
            "entropy_change_stability",
            "complete",
        ),
        default="complete",
    )
    parser.add_argument(
        "--onset-mode", choices=("causal_fixed", "causal_noise_floor"), default="causal_fixed"
    )
    parser.add_argument("--pre-roll-ms", type=int, default=30)
    parser.add_argument("--flop-backend", choices=("none", "thop"), default="none")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/rebuild_v1.yaml")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    onset_config = OnsetConfig()
    waveform = _load_waveform(arguments.audio, onset_config.sample_rate)
    model = create_model(arguments.model)
    _load_checkpoint(model, arguments.checkpoint)
    profile = profile_pipeline(
        model,
        waveform,
        onset_config=onset_config,
        onset_mode=arguments.onset_mode,
        pre_roll_ms=arguments.pre_roll_ms,
        theta=arguments.theta,
        epsilon=arguments.epsilon,
        policy_variant=arguments.policy_variant,
        device=arguments.device,
        batch_size=arguments.batch_size,
        warmup_repetitions=arguments.warmup,
        measured_repetitions=arguments.repetitions,
        flop_backend=arguments.flop_backend,
    )
    payload = write_profile_outputs(
        profile,
        arguments.output_dir,
        config_path=arguments.config,
        source_audio_path=arguments.audio,
        checkpoint_path=arguments.checkpoint,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
