from __future__ import annotations

import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from timeentropy.evaluation.calibration import probabilities_from_logits
from timeentropy.evaluation.decisions import (
    PREFIX_TIMES_MS,
    DecisionBatch,
    PolicyVariant,
    decide_from_probabilities,
)
from timeentropy.profiling.complexity import ComplexityReport, FlopBackend, model_complexity
from timeentropy.signal.features import log_magnitude_stft
from timeentropy.signal.onset import (
    OnsetConfig,
    OnsetMode,
    crop_post_onset_with_preroll,
    detect_onset,
)

DeviceRequest = Literal["cpu", "mps", "auto"]
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class TimingStatistics:
    median_ms: float
    mean_ms: float
    sd_ms: float
    p95_ms: float
    repetitions: int


@dataclass(frozen=True)
class MemoryReport:
    peak_process_rss_bytes_before: int | None
    peak_process_rss_bytes_after: int | None
    peak_process_rss_increment_bytes: int | None
    measurement_method: str | None
    reliability_note: str


@dataclass(frozen=True)
class PipelineProfile:
    model_name: str
    device: str
    device_name: str
    batch_size: int
    warmup_repetitions: int
    measured_repetitions: int
    sample_rate: int
    onset_mode: str
    pre_roll_ms: int
    prefix_times_ms: tuple[int, ...]
    policy_variant: str
    theta: float
    epsilon: float
    selected_decision_time_ms: int
    acoustic_observation_ms: int
    input_span_at_decision_ms: int
    onset_timing_scope: str
    onset_timing_input_ms: float
    onset_confirmation_delay_ms: float
    feature_shapes_frequency_by_time: tuple[tuple[int, int], ...]
    component_timings: dict[str, TimingStatistics]
    end_to_end_compute: TimingStatistics
    timing_accounting_overhead: TimingStatistics
    estimated_observation_plus_compute: TimingStatistics
    accounting_identity: str
    complexity: ComplexityReport
    memory: MemoryReport

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _RunMeasurement:
    component_ms: dict[str, float]
    end_to_end_compute_ms: float
    accounting_overhead_ms: float
    decision_time_ms: int
    onset_confirmation_delay_ms: float
    feature_shapes: tuple[tuple[int, int], ...]
    representative_input: Tensor


def resolve_device(request: DeviceRequest) -> torch.device:
    if request == "cpu":
        return torch.device("cpu")
    if request == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if request == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    raise ValueError("device must be 'cpu', 'mps', or 'auto'")


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _timed(callable_object: Callable[[], _Result], device: torch.device) -> tuple[_Result, float]:
    _synchronize(device)
    start = time.perf_counter_ns()
    result = callable_object()
    _synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return result, elapsed_ms


def _peak_process_rss_bytes() -> tuple[int | None, str | None]:
    try:
        import resource
    except ImportError:  # pragma: no cover - resource is present on supported Unix targets
        return None, None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value, "resource.getrusage(RUSAGE_SELF).ru_maxrss in bytes on macOS"
    if sys.platform.startswith("linux"):
        return value * 1024, "resource.getrusage(RUSAGE_SELF).ru_maxrss converted from KiB"
    return None, None


def _device_name(device: torch.device) -> str:
    processor = platform.processor() or platform.machine() or "CPU"
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            processor = result.stdout.strip()
    if device.type == "mps":
        return f"Apple Metal Performance Shaders ({processor})"
    return processor


def _timing_statistics(values: list[float]) -> TimingStatistics:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Timing samples must be a non-empty finite vector")
    if np.any(array < 0.0):
        raise ValueError("Timing samples must be non-negative")
    return TimingStatistics(
        median_ms=float(np.median(array)),
        mean_ms=float(array.mean()),
        sd_ms=float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        p95_ms=float(np.quantile(array, 0.95)),
        repetitions=len(array),
    )


def _onset_timing_signal(
    waveform: NDArray[np.float32],
    config: OnsetConfig,
    mode: OnsetMode,
) -> tuple[NDArray[np.float32], str]:
    """Bound causal onset timing to samples available when the detector can answer."""

    reference = detect_onset(waveform, config, mode=mode)
    if not reference.detected or reference.confirmation_sample is None:
        raise RuntimeError("Synthetic/profile waveform did not produce a usable onset")
    if mode == "recovered_offline_peak":
        return waveform, "full supplied waveform required by offline peak-relative threshold"

    timing_end = reference.confirmation_sample
    if mode == "causal_noise_floor":
        reference_frames = max(1, int(np.ceil(config.causal_noise_reference_ms / config.hop_ms)))
        frame_samples = round(config.frame_ms * config.sample_rate / 1000.0)
        hop_samples = round(config.hop_ms * config.sample_rate / 1000.0)
        noise_threshold_ready = frame_samples + (reference_frames - 1) * hop_samples
        timing_end = max(timing_end, noise_threshold_ready)
    timing_end = min(len(waveform), timing_end)
    return (
        waveform[:timing_end],
        "causal samples through detector confirmation/threshold availability",
    )


def _run_pipeline_once(
    model: nn.Module,
    waveform: NDArray[np.float32],
    onset_timing_waveform: NDArray[np.float32],
    onset_config: OnsetConfig,
    onset_mode: OnsetMode,
    pre_roll_ms: int,
    batch_size: int,
    theta: float,
    epsilon: float,
    policy_variant: PolicyVariant,
    device: torch.device,
) -> _RunMeasurement:
    _synchronize(device)
    total_start = time.perf_counter_ns()

    onset_object, onset_ms = _timed(
        lambda: detect_onset(onset_timing_waveform, onset_config, mode=onset_mode), device
    )
    onset = onset_object
    if not onset.detected or onset.onset_sample is None or onset.confirmation_sample is None:
        raise RuntimeError("Synthetic/profile waveform did not produce a usable onset")

    component = {
        "onset_detection": onset_ms,
        "prefix_crop_until_decision": 0.0,
        "stft_until_decision": 0.0,
        "input_tensor_preparation_until_decision": 0.0,
        "model_forward_until_decision": 0.0,
        "logit_to_probability_until_decision": 0.0,
        "policy_computation_until_decision": 0.0,
    }
    probability_list: list[NDArray[np.float64]] = []
    feature_shapes: list[tuple[int, int]] = []
    representative_input: Tensor | None = None
    decisions: DecisionBatch | None = None

    for prefix_ms in PREFIX_TIMES_MS:
        crop, crop_ms = _timed(
            lambda prefix_ms=prefix_ms: crop_post_onset_with_preroll(
                waveform,
                onset_sample=onset.onset_sample,
                post_onset_ms=prefix_ms,
                pre_roll_ms=pre_roll_ms,
                sample_rate=onset_config.sample_rate,
            ),
            device,
        )
        component["prefix_crop_until_decision"] += crop_ms
        feature, stft_ms = _timed(
            lambda crop=crop: log_magnitude_stft(
                crop,
                sample_rate=onset_config.sample_rate,
            ),
            device,
        )
        component["stft_until_decision"] += stft_ms
        feature_shapes.append(tuple(int(value) for value in feature.shape))

        def prepare_input_tensor(
            feature_at_prefix: NDArray[np.float32] = feature,
        ) -> Tensor:
            values = torch.from_numpy(feature_at_prefix).unsqueeze(0).unsqueeze(0)
            if batch_size > 1:
                values = values.repeat(batch_size, 1, 1, 1)
            return values.to(device=device, dtype=torch.float32)

        representative_input, preparation_ms = _timed(prepare_input_tensor, device)
        component["input_tensor_preparation_until_decision"] += preparation_ms
        with torch.inference_mode():
            logits, forward_ms = _timed(
                lambda values=representative_input: model(values),
                device,
            )
        if not isinstance(logits, Tensor) or logits.shape != (batch_size, 2):
            raise RuntimeError(
                f"Model must return B x 2 logits; received {getattr(logits, 'shape', None)}"
            )
        component["model_forward_until_decision"] += forward_ms

        def convert_logits_to_probability(
            logits_at_prefix: Tensor = logits,
        ) -> NDArray[np.float64]:
            logits_numpy = logits_at_prefix.detach().cpu().numpy().astype(np.float64)
            return probabilities_from_logits(logits_numpy)

        probabilities_at_prefix, conversion_ms = _timed(
            convert_logits_to_probability,
            device,
        )
        probability_list.append(probabilities_at_prefix)
        component["logit_to_probability_until_decision"] += conversion_ms

        # The 150 ms observation is a warm-up used only for stability at 200 ms.
        if prefix_ms == PREFIX_TIMES_MS[0]:
            continue

        def compute_policy() -> DecisionBatch:
            observed = np.stack(probability_list, axis=1)
            missing_prefixes = len(PREFIX_TIMES_MS) - observed.shape[1]
            if missing_prefixes:
                # Future columns are placeholders.  A decision is accepted below only when
                # the canonical rule chooses a time no later than the prefix actually observed.
                future_placeholders = np.repeat(
                    observed[:, -1:, :],
                    repeats=missing_prefixes,
                    axis=1,
                )
                observed = np.concatenate((observed, future_placeholders), axis=1)
            return decide_from_probabilities(
                observed,
                theta=theta,
                epsilon=epsilon,
                policy_variant=policy_variant,
            )

        decisions, policy_ms = _timed(compute_policy, device)
        component["policy_computation_until_decision"] += policy_ms
        decision_times = np.asarray(decisions.decision_time_ms, dtype=np.int64)
        if not np.all(decision_times == decision_times[0]):
            raise RuntimeError("Repeated batch items produced inconsistent decision times")
        if np.all(decision_times <= prefix_ms):
            break

    if decisions is None or representative_input is None:
        raise RuntimeError("No candidate decision or representative input was generated")
    decision_times = np.asarray(decisions.decision_time_ms, dtype=np.int64)

    _synchronize(device)
    end_to_end_ms = (time.perf_counter_ns() - total_start) / 1_000_000.0
    overhead = end_to_end_ms - sum(component.values())
    if overhead < -1.0e-6:
        raise RuntimeError("Component timings exceed the enclosing end-to-end measurement")
    overhead = max(0.0, overhead)
    confirmation_delay_ms = (
        (onset.confirmation_sample - onset.onset_sample) * 1000.0 / onset_config.sample_rate
    )
    return _RunMeasurement(
        component_ms=component,
        end_to_end_compute_ms=end_to_end_ms,
        accounting_overhead_ms=overhead,
        decision_time_ms=int(decision_times[0]),
        onset_confirmation_delay_ms=confirmation_delay_ms,
        feature_shapes=tuple(feature_shapes),
        representative_input=representative_input,
    )


def profile_pipeline(
    model: nn.Module,
    waveform: NDArray[np.float32],
    *,
    onset_config: OnsetConfig | None = None,
    onset_mode: OnsetMode = "causal_fixed",
    pre_roll_ms: int = 30,
    theta: float = 0.90,
    epsilon: float = 0.08,
    policy_variant: PolicyVariant = "complete",
    device: DeviceRequest = "cpu",
    batch_size: int = 1,
    warmup_repetitions: int = 10,
    measured_repetitions: int = 100,
    flop_backend: FlopBackend = "none",
) -> PipelineProfile:
    """Profile waveform-to-decision compute separately from acoustic observation time."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if batch_size < 1:
        raise ValueError("batch_size must be positive; batch size 1 is the primary condition")
    if warmup_repetitions < 0 or measured_repetitions < 1:
        raise ValueError(
            "warmup_repetitions must be non-negative and measured_repetitions positive"
        )
    if pre_roll_ms < 0:
        raise ValueError("pre_roll_ms must be non-negative")
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if len(signal) == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("waveform must be a non-empty finite vector")
    config = onset_config or OnsetConfig()
    resolved_device = resolve_device(device)
    onset_timing_waveform, onset_timing_scope = _onset_timing_signal(
        signal,
        config,
        onset_mode,
    )

    # Complexity is deliberately standardized at the maximum registered prefix rather than
    # changing with a particular sample's exit time.  This is one model forward, not the
    # cumulative cost of the sequential early-exit pipeline.
    maximum_crop_samples = round((pre_roll_ms + PREFIX_TIMES_MS[-1]) * config.sample_rate / 1000.0)
    maximum_feature = log_magnitude_stft(
        np.zeros(maximum_crop_samples, dtype=np.float32),
        sample_rate=config.sample_rate,
    )
    placeholder = torch.zeros(
        (batch_size, 1, *maximum_feature.shape),
        dtype=torch.float32,
    )
    complexity = model_complexity(model, placeholder, flop_backend=flop_backend)
    rss_before, memory_method = _peak_process_rss_bytes()
    model = model.to(resolved_device).eval()

    for _ in range(warmup_repetitions):
        _run_pipeline_once(
            model,
            signal,
            onset_timing_waveform,
            config,
            onset_mode,
            pre_roll_ms,
            batch_size,
            theta,
            epsilon,
            policy_variant,
            resolved_device,
        )

    measurements = [
        _run_pipeline_once(
            model,
            signal,
            onset_timing_waveform,
            config,
            onset_mode,
            pre_roll_ms,
            batch_size,
            theta,
            epsilon,
            policy_variant,
            resolved_device,
        )
        for _ in range(measured_repetitions)
    ]
    decision_times = {item.decision_time_ms for item in measurements}
    feature_shapes = {item.feature_shapes for item in measurements}
    confirmation_delays = {item.onset_confirmation_delay_ms for item in measurements}
    if len(decision_times) != 1 or len(feature_shapes) != 1 or len(confirmation_delays) != 1:
        raise RuntimeError("Profile repetitions produced inconsistent deterministic metadata")

    components = {
        name: _timing_statistics([item.component_ms[name] for item in measurements])
        for name in measurements[0].component_ms
    }
    end_values = [item.end_to_end_compute_ms for item in measurements]
    overhead_values = [item.accounting_overhead_ms for item in measurements]
    acoustic_observation = next(iter(decision_times))
    estimated_values = [acoustic_observation + value for value in end_values]
    rss_after, after_method = _peak_process_rss_bytes()
    if memory_method != after_method:
        memory_method = after_method or memory_method
    rss_increment = (
        max(0, rss_after - rss_before) if rss_before is not None and rss_after is not None else None
    )
    memory = MemoryReport(
        peak_process_rss_bytes_before=rss_before,
        peak_process_rss_bytes_after=rss_after,
        peak_process_rss_increment_bytes=rss_increment,
        measurement_method=memory_method,
        reliability_note=(
            "Process-lifetime resident-set high-water mark; not isolated model memory and may "
            "include Python/runtime allocations"
            if memory_method
            else "Peak process RSS is unavailable on this platform"
        ),
    )
    model_name = getattr(getattr(model, "metadata", None), "name", model.__class__.__name__)
    confirmation_delay = next(iter(confirmation_delays))
    return PipelineProfile(
        model_name=str(model_name),
        device=resolved_device.type,
        device_name=_device_name(resolved_device),
        batch_size=batch_size,
        warmup_repetitions=warmup_repetitions,
        measured_repetitions=measured_repetitions,
        sample_rate=config.sample_rate,
        onset_mode=onset_mode,
        pre_roll_ms=pre_roll_ms,
        prefix_times_ms=PREFIX_TIMES_MS,
        policy_variant=policy_variant,
        theta=float(theta),
        epsilon=float(epsilon),
        selected_decision_time_ms=acoustic_observation,
        acoustic_observation_ms=acoustic_observation,
        input_span_at_decision_ms=acoustic_observation + pre_roll_ms,
        onset_timing_scope=onset_timing_scope,
        onset_timing_input_ms=len(onset_timing_waveform) * 1000.0 / config.sample_rate,
        onset_confirmation_delay_ms=float(confirmation_delay),
        feature_shapes_frequency_by_time=next(iter(feature_shapes)),
        component_timings=components,
        end_to_end_compute=_timing_statistics(end_values),
        timing_accounting_overhead=_timing_statistics(overhead_values),
        estimated_observation_plus_compute=_timing_statistics(estimated_values),
        accounting_identity=(
            "end_to_end_compute = sum(component timings) + timing_accounting_overhead; "
            "estimated_observation_plus_compute = acoustic_observation + end_to_end_compute"
        ),
        complexity=complexity,
        memory=memory,
    )
