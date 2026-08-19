from __future__ import annotations

import importlib
import io
from dataclasses import dataclass
from importlib import metadata
from typing import Literal

import torch
from torch import Tensor, nn

FlopBackend = Literal["none", "thop"]


@dataclass(frozen=True)
class ComplexityReport:
    total_parameters: int
    trainable_parameters: int
    state_dict_bytes: int
    state_dict_serialization: str
    flops_available: bool
    flop_backend: str | None
    flop_backend_version: str | None
    multiply_accumulates: float | None
    floating_point_operations: float | None
    flop_convention: str | None
    flop_scope: str
    flop_unavailable_reason: str | None
    representative_input_shape: tuple[int, ...]


def _serialized_state_dict_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def _optional_flops(
    model: nn.Module,
    representative_input: Tensor,
    backend: FlopBackend,
) -> tuple[bool, str | None, str | None, float | None, float | None, str | None, str | None]:
    if backend == "none":
        return (
            False,
            None,
            None,
            None,
            None,
            None,
            "FLOP/MAC counting was not requested; no estimate was fabricated",
        )
    if backend != "thop":
        raise ValueError("flop_backend must be 'none' or 'thop'")
    try:
        thop = importlib.import_module("thop")
    except ImportError:
        return (
            False,
            "thop",
            None,
            None,
            None,
            "THOP MAC convention; FLOPs would be reported as 2 x MACs",
            "Optional backend 'thop' is not installed",
        )

    try:
        macs, _ = thop.profile(model, inputs=(representative_input,), verbose=False)
    except Exception as error:  # noqa: BLE001  # pragma: no cover
        return (
            False,
            "thop",
            metadata.version("thop"),
            None,
            None,
            "THOP MAC convention; FLOPs would be reported as 2 x MACs",
            f"THOP could not profile this model: {type(error).__name__}: {error}",
        )
    mac_value = float(macs)
    return (
        True,
        "thop",
        metadata.version("thop"),
        mac_value,
        2.0 * mac_value,
        "THOP counts multiply-accumulates; one MAC is reported as two FLOPs",
        None,
    )


def model_complexity(
    model: nn.Module,
    representative_input: Tensor,
    *,
    flop_backend: FlopBackend = "none",
) -> ComplexityReport:
    """Count parameters/state bytes and optionally request backend-defined MACs/FLOPs."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(representative_input, Tensor):
        raise TypeError("representative_input must be a torch.Tensor")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    state_bytes = _serialized_state_dict_bytes(model)
    (
        available,
        backend,
        backend_version,
        macs,
        flops,
        convention,
        reason,
    ) = _optional_flops(model, representative_input, flop_backend)
    return ComplexityReport(
        total_parameters=total,
        trainable_parameters=trainable,
        state_dict_bytes=state_bytes,
        state_dict_serialization="torch.save(state_dict) into an in-memory io.BytesIO buffer",
        flops_available=available,
        flop_backend=backend,
        flop_backend_version=backend_version,
        multiply_accumulates=macs,
        floating_point_operations=flops,
        flop_convention=convention,
        flop_scope="One model forward at the configured maximum 300 ms post-onset prefix",
        flop_unavailable_reason=reason,
        representative_input_shape=tuple(representative_input.shape),
    )
