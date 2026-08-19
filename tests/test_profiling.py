import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from timeentropy.profiling import profile_pipeline, resolve_device, write_profile_outputs
from timeentropy.provenance import sha256_file


class TinySpectrogramClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        pooled = values.mean(dim=(1, 2, 3)).unsqueeze(1)
        return self.classifier(pooled)


def _synthetic_waveform() -> np.ndarray:
    silence = np.zeros(1_600, dtype=np.float32)
    active = np.full(11_200, 0.10, dtype=np.float32)
    return np.concatenate([silence, active])


def test_profile_separates_acoustic_observation_and_component_compute() -> None:
    profile = profile_pipeline(
        TinySpectrogramClassifier(),
        _synthetic_waveform(),
        theta=1.0,
        epsilon=1.0,
        warmup_repetitions=1,
        measured_repetitions=3,
        device="cpu",
    )
    assert profile.device == "cpu"
    assert profile.batch_size == 1
    assert profile.acoustic_observation_ms == 200
    assert profile.input_span_at_decision_ms == 230
    # The left-aligned 90--110 ms frame already contains speech, so two-frame
    # confirmation is available at 120 ms for this waveform.
    assert profile.onset_timing_input_ms == 120.0
    assert profile.onset_timing_scope.startswith("causal samples")
    assert profile.onset_confirmation_delay_ms == 30.0
    assert profile.feature_shapes_frequency_by_time == (
        (257, 16),
        (257, 21),
    )

    component_mean = sum(item.mean_ms for item in profile.component_timings.values())
    assert profile.end_to_end_compute.mean_ms == pytest.approx(
        component_mean + profile.timing_accounting_overhead.mean_ms,
        abs=1.0e-9,
    )
    assert profile.estimated_observation_plus_compute.mean_ms == pytest.approx(
        profile.acoustic_observation_ms + profile.end_to_end_compute.mean_ms,
        abs=1.0e-9,
    )
    for statistics in (
        *profile.component_timings.values(),
        profile.end_to_end_compute,
        profile.timing_accounting_overhead,
    ):
        assert statistics.repetitions == 3
        assert statistics.median_ms >= 0.0
        assert statistics.p95_ms >= statistics.median_ms
        assert statistics.sd_ms >= 0.0

    assert profile.complexity.total_parameters == 4
    assert profile.complexity.trainable_parameters == 4
    assert profile.complexity.state_dict_bytes > 0
    assert not profile.complexity.flops_available
    assert profile.complexity.multiply_accumulates is None
    assert "not requested" in profile.complexity.flop_unavailable_reason
    assert "maximum 300 ms" in profile.complexity.flop_scope
    assert profile.complexity.representative_input_shape == (1, 1, 257, 31)


def test_device_resolution_supports_cpu_and_conditionally_mps() -> None:
    assert resolve_device("cpu").type == "cpu"
    if torch.backends.mps.is_available():
        assert resolve_device("mps").type == "mps"
    else:
        with pytest.raises(RuntimeError, match="not available"):
            resolve_device("mps")


def test_profile_runs_only_prefixes_needed_before_forced_fallback() -> None:
    profile = profile_pipeline(
        TinySpectrogramClassifier(),
        _synthetic_waveform(),
        theta=0.0,
        epsilon=0.0,
        warmup_repetitions=0,
        measured_repetitions=1,
    )
    assert profile.selected_decision_time_ms == 300
    assert profile.feature_shapes_frequency_by_time == (
        (257, 16),
        (257, 21),
        (257, 26),
        (257, 31),
    )


def test_profile_writer_emits_json_csv_and_config_hash(tmp_path) -> None:
    profile = profile_pipeline(
        TinySpectrogramClassifier(),
        _synthetic_waveform(),
        theta=1.0,
        epsilon=1.0,
        warmup_repetitions=0,
        measured_repetitions=1,
    )
    config_path = tmp_path / "frozen.yaml"
    config_path.write_text("protocol: frozen\n", encoding="utf-8")
    output_path = tmp_path / "profile"
    payload = write_profile_outputs(profile, output_path, config_path=config_path)

    assert (output_path / "runtime_profile.json").is_file()
    assert (output_path / "runtime_timings.csv").is_file()
    assert (output_path / "runtime_complexity.csv").is_file()
    written = json.loads((output_path / "runtime_profile.json").read_text(encoding="utf-8"))
    assert written == payload
    assert written["provenance"]["config_sha256"] == sha256_file(config_path)
    timings = pd.read_csv(output_path / "runtime_timings.csv")
    assert set(timings["config_sha256"]) == {sha256_file(config_path)}
    assert set(timings["device_name"]) == {profile.device_name}
    assert timings["environment_json"].str.contains('"torch"').all()
    assert {
        "onset_detection",
        "prefix_crop_until_decision",
        "stft_until_decision",
        "input_tensor_preparation_until_decision",
        "model_forward_until_decision",
        "logit_to_probability_until_decision",
        "policy_computation_until_decision",
        "end_to_end_compute",
        "estimated_observation_plus_compute",
    }.issubset(set(timings["component"]))
