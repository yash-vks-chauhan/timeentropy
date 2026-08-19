from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from timeentropy.config import PROJECT_ROOT
from timeentropy.profiling.runtime import PipelineProfile
from timeentropy.provenance import environment_record, sha256_file, sha256_json


def write_profile_outputs(
    profile: PipelineProfile,
    output_directory: str | Path,
    *,
    config_path: str | Path,
    source_audio_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write JSON/CSV runtime artifacts with config, input, and environment provenance."""

    output_path = Path(output_directory).resolve()
    configuration = Path(config_path).resolve()
    if not configuration.is_file():
        raise FileNotFoundError(configuration)
    audio = Path(source_audio_path).resolve() if source_audio_path is not None else None
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
    for optional_path in (audio, checkpoint):
        if optional_path is not None and not optional_path.is_file():
            raise FileNotFoundError(optional_path)
    output_path.mkdir(parents=True, exist_ok=True)

    profile_dict = profile.as_dict()
    provenance = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(configuration),
        "config_sha256": sha256_file(configuration),
        "source_audio_path": str(audio) if audio else None,
        "source_audio_sha256": sha256_file(audio) if audio else None,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "profile_settings_sha256": sha256_json(
            {
                "model_name": profile.model_name,
                "device": profile.device,
                "batch_size": profile.batch_size,
                "warmup_repetitions": profile.warmup_repetitions,
                "measured_repetitions": profile.measured_repetitions,
                "onset_mode": profile.onset_mode,
                "pre_roll_ms": profile.pre_roll_ms,
                "theta": profile.theta,
                "epsilon": profile.epsilon,
                "policy_variant": profile.policy_variant,
            }
        ),
        "environment": environment_record(PROJECT_ROOT),
    }
    payload = {"schema_version": 1, "profile": profile_dict, "provenance": provenance}
    # Normalize tuples and NumPy-adjacent scalar containers to their JSON representation so
    # the returned value is byte-semantically consistent with the written artifact.
    payload = json.loads(json.dumps(payload))
    with (output_path / "runtime_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    csv_provenance = {
        "generated_at_utc": provenance["generated_at_utc"],
        "config_sha256": provenance["config_sha256"],
        "source_audio_sha256": provenance["source_audio_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "profile_settings_sha256": provenance["profile_settings_sha256"],
        "device_name": profile.device_name,
        "environment_json": json.dumps(provenance["environment"], sort_keys=True),
    }
    timing_rows = []
    all_timings = {
        **profile.component_timings,
        "end_to_end_compute": profile.end_to_end_compute,
        "timing_accounting_overhead": profile.timing_accounting_overhead,
        "estimated_observation_plus_compute": profile.estimated_observation_plus_compute,
    }
    for component, statistics in all_timings.items():
        timing_rows.append(
            {
                "model_name": profile.model_name,
                "device": profile.device,
                "batch_size": profile.batch_size,
                "component": component,
                **csv_provenance,
                **asdict(statistics),
                "acoustic_observation_ms": profile.acoustic_observation_ms,
                "input_span_at_decision_ms": profile.input_span_at_decision_ms,
                "onset_timing_scope": profile.onset_timing_scope,
                "onset_timing_input_ms": profile.onset_timing_input_ms,
            }
        )
    pd.DataFrame(timing_rows).to_csv(output_path / "runtime_timings.csv", index=False)

    complexity_row = {
        "model_name": profile.model_name,
        "device": profile.device,
        **csv_provenance,
        **asdict(profile.complexity),
        **asdict(profile.memory),
    }
    pd.DataFrame([complexity_row]).to_csv(output_path / "runtime_complexity.csv", index=False)
    return payload
