from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from timeentropy.config import load_yaml, project_path
from timeentropy.provenance import sha256_file, sha256_json
from timeentropy.signal.onset import OnsetMode

AnalysisFamily = Literal["retrained_input", "test_time"]
PerturbationKind = Literal["none", "white_noise", "onset_shift"]

SUPPORTED_ONSET_MODES: frozenset[str] = frozenset(
    {"causal_fixed", "causal_noise_floor", "recovered_offline_peak"}
)
MAX_TEST_TIME_CONDITIONS = 64


@dataclass(frozen=True)
class RobustnessAnalysis:
    """One prespecified condition and its execution boundary."""

    analysis_id: str
    family: AnalysisFamily
    role: str
    reference_checkpoint: str
    onset_mode: OnsetMode
    pre_roll_ms: int
    perturbation_kind: PerturbationKind = "none"
    snr_db: float | None = None
    onset_offset_ms: int | None = None
    requires_condition_matched_checkpoint: bool = False
    requires_new_training: bool = False


@dataclass(frozen=True)
class RobustnessProtocol:
    """Validated robustness config joined to its frozen parent experiment."""

    config_path: Path
    parent_config_path: Path
    config: dict[str, Any]
    parent_config: dict[str, Any]
    config_file_sha256: str
    config_sha256: str
    parent_config_file_sha256: str
    parent_config_sha256: str
    model_id: str
    folds: tuple[int, ...]
    seed: int
    retrained_analyses: tuple[RobustnessAnalysis, ...]
    test_time_analyses: tuple[RobustnessAnalysis, ...]

    @property
    def analyses(self) -> tuple[RobustnessAnalysis, ...]:
        return self.retrained_analyses + self.test_time_analyses

    def analysis(self, analysis_id: str) -> RobustnessAnalysis:
        matches = [item for item in self.analyses if item.analysis_id == analysis_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown robustness analysis_id: {analysis_id}")
        return matches[0]

    def plan_rows(self) -> list[dict[str, Any]]:
        """Expand conditions by fold without performing data access or training."""

        rows: list[dict[str, Any]] = []
        for analysis in self.analyses:
            for fold in self.folds:
                rows.append(
                    {
                        **asdict(analysis),
                        "fold": fold,
                        "model_id": self.model_id,
                        "seed": self.seed,
                        "execution_stage": (
                            "train_then_evaluate"
                            if analysis.family == "retrained_input"
                            else "test_inference_only"
                        ),
                        "selection_source": (
                            "condition_matched_validation"
                            if analysis.family == "retrained_input"
                            else "clean_reference_validation"
                        ),
                        "test_set_tuning_permitted": False,
                    }
                )
        return rows


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty list")
    return value


def _slug_number(value: float) -> str:
    magnitude = f"{abs(float(value)):g}".replace(".", "p")
    return f"m{magnitude}" if value < 0 else f"p{magnitude}"


def _validate_retrained_analyses(
    config: dict[str, Any], parent: dict[str, Any]
) -> tuple[RobustnessAnalysis, ...]:
    raw_analyses = _require_sequence(
        config.get("retrained_input_ablations"), "retrained_input_ablations"
    )
    analyses: list[RobustnessAnalysis] = []
    for index, raw in enumerate(raw_analyses):
        item = _require_mapping(raw, f"retrained_input_ablations[{index}]")
        analysis_id = str(item.get("analysis_id", "")).strip()
        role = str(item.get("role", "")).strip()
        onset_mode = str(item.get("onset_mode", ""))
        if not analysis_id or not role:
            raise ValueError("Every retrained ablation requires analysis_id and role")
        if onset_mode not in SUPPORTED_ONSET_MODES:
            raise ValueError(f"Unsupported onset_mode in {analysis_id}: {onset_mode}")
        pre_roll_ms = int(item.get("pre_roll_ms", -1))
        if pre_roll_ms < 0:
            raise ValueError(f"pre_roll_ms must be non-negative in {analysis_id}")
        is_reference = role == "primary"
        analyses.append(
            RobustnessAnalysis(
                analysis_id=analysis_id,
                family="retrained_input",
                role=role,
                reference_checkpoint=analysis_id,
                onset_mode=onset_mode,  # type: ignore[arg-type]
                pre_roll_ms=pre_roll_ms,
                requires_condition_matched_checkpoint=True,
                requires_new_training=not is_reference,
            )
        )

    reference = [analysis for analysis in analyses if analysis.role == "primary"]
    if len(reference) != 1:
        raise ValueError("Exactly one retrained input analysis must have role='primary'")
    expected_mode = str(parent["onset"]["primary_mode"])
    expected_pre_roll = int(parent["onset"]["pre_roll_ms"])
    if reference[0].onset_mode != expected_mode or reference[0].pre_roll_ms != expected_pre_roll:
        raise ValueError("Primary robustness analysis must match the parent onset configuration")
    return tuple(analyses)


def _validate_test_time_analyses(
    config: dict[str, Any],
    parent: dict[str, Any],
    retrained: tuple[RobustnessAnalysis, ...],
) -> tuple[RobustnessAnalysis, ...]:
    reference = next(analysis for analysis in retrained if analysis.role == "primary")
    noise = _require_mapping(config.get("evaluation_only_noise"), "evaluation_only_noise")
    onset = _require_mapping(
        config.get("evaluation_only_onset_perturbation"),
        "evaluation_only_onset_perturbation",
    )
    for section_name, section in (("evaluation_only_noise", noise), ("onset", onset)):
        if str(section.get("reference_checkpoint")) != reference.analysis_id:
            raise ValueError(f"{section_name} must reference the primary checkpoint")

    if noise.get("noise_type") != "zero_mean_white_gaussian":
        raise ValueError("Only zero_mean_white_gaussian noise is implemented")
    if noise.get("insertion_stage") != "after_onset_crop_before_stft":
        raise ValueError("Noise must be inserted after onset crop and before STFT")
    if noise.get("randomization") != "sha256(primary_seed,sample_id,snr_db)":
        raise ValueError("Noise randomization contract differs from the implementation")
    raw_snr = _require_sequence(noise.get("snr_db"), "evaluation_only_noise.snr_db")
    snr_values = tuple(float(value) for value in raw_snr)
    if not all(math.isfinite(value) for value in snr_values):
        raise ValueError("All SNR values must be finite")
    if len(set(snr_values)) != len(snr_values):
        raise ValueError("SNR values must be unique")

    if onset.get("sign_convention") != "estimated_onset = primary_onset + offset":
        raise ValueError("Onset-offset sign convention differs from the implementation")
    expected_boundary = (
        "clip crop anchor to available waveform; preserve exact crop length by padding"
    )
    if onset.get("boundary_policy") != expected_boundary:
        raise ValueError("Onset boundary policy differs from the implementation")
    raw_offsets = _require_sequence(
        onset.get("offset_ms"), "evaluation_only_onset_perturbation.offset_ms"
    )
    offsets = tuple(int(value) for value in raw_offsets)
    if any(float(raw) != offset for raw, offset in zip(raw_offsets, offsets, strict=True)):
        raise ValueError("Onset offsets must be integer milliseconds")
    if len(set(offsets)) != len(offsets):
        raise ValueError("Onset offsets must be unique")

    analyses: list[RobustnessAnalysis] = []
    if bool(noise.get("include_clean_reference")):
        analyses.append(
            RobustnessAnalysis(
                analysis_id="clean_reference",
                family="test_time",
                role="clean_reference",
                reference_checkpoint=reference.analysis_id,
                onset_mode=reference.onset_mode,
                pre_roll_ms=reference.pre_roll_ms,
            )
        )
    analyses.extend(
        RobustnessAnalysis(
            analysis_id=f"white_noise_snr_{_slug_number(snr)}_db",
            family="test_time",
            role="noise_stress_test",
            reference_checkpoint=reference.analysis_id,
            onset_mode=reference.onset_mode,
            pre_roll_ms=reference.pre_roll_ms,
            perturbation_kind="white_noise",
            snr_db=snr,
        )
        for snr in snr_values
    )
    analyses.extend(
        RobustnessAnalysis(
            analysis_id=f"onset_shift_{_slug_number(offset)}_ms",
            family="test_time",
            role="onset_sensitivity",
            reference_checkpoint=reference.analysis_id,
            onset_mode=reference.onset_mode,
            pre_roll_ms=reference.pre_roll_ms,
            perturbation_kind="onset_shift",
            onset_offset_ms=offset,
        )
        for offset in offsets
    )
    if len(analyses) > MAX_TEST_TIME_CONDITIONS:
        raise ValueError(f"At most {MAX_TEST_TIME_CONDITIONS} test-time conditions are permitted")
    if not analyses:
        raise ValueError("The robustness protocol contains no test-time conditions")
    if len({analysis.analysis_id for analysis in analyses}) != len(analyses):
        raise ValueError("Generated test-time analysis IDs are not unique")
    if reference.onset_mode != parent["onset"]["primary_mode"]:
        raise ValueError("Test-time analyses must use the parent primary onset detector")
    return tuple(analyses)


def load_robustness_protocol(path: str | Path) -> RobustnessProtocol:
    """Load and strictly validate the frozen robustness and parent configs."""

    config_path = project_path(path).resolve()
    config = load_yaml(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported robustness schema_version")
    if config.get("status") != "frozen_before_test_evaluation":
        raise ValueError("Robustness configuration is not frozen before test evaluation")
    parent_value = config.get("parent_experiment_config")
    if not isinstance(parent_value, str) or not parent_value:
        raise ValueError("parent_experiment_config must be a non-empty path")
    parent_path = project_path(parent_value).resolve()
    parent = load_yaml(parent_path)
    if parent.get("status") != "frozen_before_test_evaluation":
        raise ValueError("Parent experiment configuration is not frozen")

    scope = _require_mapping(config.get("scope"), "scope")
    model_id = str(scope.get("model", ""))
    primary_models = tuple(str(value) for value in parent["models"]["primary"])
    if model_id not in primary_models:
        raise ValueError("Robustness model must be a registered parent primary model")
    if model_id != str(parent["models"]["primary_headline_model"]):
        raise ValueError("Robustness model must equal the prospectively designated headline model")
    folds = tuple(int(value) for value in _require_sequence(scope.get("folds"), "scope.folds"))
    if len(set(folds)) != len(folds) or any(fold < 0 for fold in folds):
        raise ValueError("scope.folds must contain unique non-negative fold IDs")
    seed = int(scope.get("seed", -1))
    if seed != int(parent["training"]["primary_seed"]):
        raise ValueError("Robustness seed must equal the frozen parent primary seed")
    prefixes = tuple(int(value) for value in parent["prefixes"]["training_ms"])
    if prefixes != (150, 200, 250, 300):
        raise ValueError("Robustness implementation requires prefixes (150, 200, 250, 300)")

    retrained = _validate_retrained_analyses(config, parent)
    test_time = _validate_test_time_analyses(config, parent, retrained)
    all_ids = [analysis.analysis_id for analysis in retrained + test_time]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Every robustness analysis_id must be globally unique")
    return RobustnessProtocol(
        config_path=config_path,
        parent_config_path=parent_path,
        config=config,
        parent_config=parent,
        config_file_sha256=sha256_file(config_path),
        config_sha256=sha256_json(config),
        parent_config_file_sha256=sha256_file(parent_path),
        parent_config_sha256=sha256_json(parent),
        model_id=model_id,
        folds=folds,
        seed=seed,
        retrained_analyses=retrained,
        test_time_analyses=test_time,
    )


def derived_retraining_config(protocol: RobustnessProtocol, analysis_id: str) -> dict[str, Any]:
    """Return the exact condition-matched training config for one input ablation."""

    analysis = protocol.analysis(analysis_id)
    if analysis.family != "retrained_input":
        raise ValueError("Test-time perturbations must not receive derived training configs")
    if analysis.role == "primary":
        return copy.deepcopy(protocol.parent_config)

    derived = copy.deepcopy(protocol.parent_config)
    derived["experiment_name"] = (
        f"{protocol.parent_config['experiment_name']}-robustness-{analysis.analysis_id}"
    )
    derived["onset"]["primary_mode"] = analysis.onset_mode
    derived["onset"]["pre_roll_ms"] = analysis.pre_roll_ms
    derived["robustness_retrained_input_ablation"] = {
        "schema_version": 1,
        "analysis_id": analysis.analysis_id,
        "role": analysis.role,
        "execution_stage": "train_then_evaluate",
        "requires_condition_matched_checkpoint": True,
        "test_set_tuning_permitted": False,
        "parent_experiment_config": str(protocol.parent_config_path),
        "parent_experiment_config_sha256": protocol.parent_config_sha256,
        "robustness_config": str(protocol.config_path),
        "robustness_config_sha256": protocol.config_sha256,
    }
    return derived


def write_retraining_configs(
    protocol: RobustnessProtocol, output_directory: str | Path
) -> dict[str, Path]:
    """Materialize non-primary retraining configs; never launch training."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for analysis in protocol.retrained_analyses:
        if not analysis.requires_new_training:
            continue
        payload = derived_retraining_config(protocol, analysis.analysis_id)
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        path = output / f"{analysis.analysis_id}.yaml"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite a different frozen config: {path}")
        path.write_text(text, encoding="utf-8")
        paths[analysis.analysis_id] = path

    manifest = {
        "schema_version": 1,
        "robustness_config": str(protocol.config_path),
        "robustness_config_sha256": protocol.config_sha256,
        "parent_experiment_config": str(protocol.parent_config_path),
        "parent_experiment_config_sha256": protocol.parent_config_sha256,
        "generated_configs": {
            analysis_id: {"path": str(path), "sha256": sha256_file(path)}
            for analysis_id, path in paths.items()
        },
        "note": "Configuration generation only; no model training was launched.",
    }
    manifest_path = output / "retraining_config_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths["manifest"] = manifest_path
    return paths
