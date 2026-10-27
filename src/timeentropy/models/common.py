"""Shared contracts for the reconstructed short-prefix classifiers.

The original experiment code was lost.  The model modules in this package are a
frozen reconstruction: every trainable layer is declared explicitly and each
model exposes machine-readable metadata, including its parameter count.  This
makes the new experiment auditable without implying that the deleted source was
recovered.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

INPUT_FREQUENCY_BINS = 257
NUM_CLASSES = 2
DEFAULT_DROPOUT = 0.30
ARCHITECTURE_VERSION = "rebuild-v1"


@dataclass(frozen=True)
class ArchitectureMetadata:
    """Serializable description of a frozen classifier architecture."""

    name: str
    version: str
    input_layout: str
    output_layout: str
    variable_time_axis: bool
    time_axis_behavior: str
    dropout_probability: float
    trainable_parameters: int
    expected_trainable_parameters: int
    stages: tuple[str, ...]
    implementation: tuple[str, ...] = ()


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of scalar parameters updated by optimization."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def validate_spectrogram_input(values: Tensor, *, minimum_time_frames: int = 1) -> None:
    """Validate the common ``B x 1 x 257 x T`` model-input contract."""

    if not isinstance(values, Tensor):
        raise TypeError("model input must be a torch.Tensor")
    if values.ndim != 4:
        raise ValueError(
            f"model input must have layout B x 1 x 257 x T; received shape {tuple(values.shape)}"
        )
    if values.shape[0] < 1:
        raise ValueError("model input must contain at least one batch item")
    if values.shape[1] != 1:
        raise ValueError(f"model input must have one channel; received {values.shape[1]}")
    if values.shape[2] != INPUT_FREQUENCY_BINS:
        raise ValueError(
            f"model input must have {INPUT_FREQUENCY_BINS} frequency bins; "
            f"received {values.shape[2]}"
        )
    if values.shape[3] < minimum_time_frames:
        raise ValueError(
            f"model input must have at least {minimum_time_frames} time frames; "
            f"received {values.shape[3]}"
        )
    if not values.is_floating_point():
        raise TypeError("model input must use a floating-point dtype")


def resolve_padding(
    values: Tensor,
    *,
    feature_lengths: Tensor | None = None,
    padding_mask: Tensor | None = None,
    minimum_valid_time_frames: int = 1,
) -> tuple[Tensor, Tensor]:
    """Validate lengths/mask and return a consistent right-padding description.

    ``padding_mask`` follows PyTorch's attention convention: ``True`` marks a
    padded position.  When both inputs are supplied, they must encode exactly
    the same right-padded layout.
    """

    batch_size, time_steps = values.shape[0], values.shape[-1]
    if feature_lengths is not None:
        if not isinstance(feature_lengths, Tensor):
            raise TypeError("feature_lengths must be a torch.Tensor")
        if feature_lengths.ndim != 1 or feature_lengths.shape[0] != batch_size:
            raise ValueError(
                f"feature_lengths must have shape ({batch_size},); "
                f"received {tuple(feature_lengths.shape)}"
            )
        if feature_lengths.dtype == torch.bool or feature_lengths.is_floating_point():
            raise TypeError("feature_lengths must use an integer dtype")
        lengths = feature_lengths.to(device=values.device, dtype=torch.long)
    elif padding_mask is not None:
        if not isinstance(padding_mask, Tensor):
            raise TypeError("padding_mask must be a torch.Tensor")
        if padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must use torch.bool")
        if padding_mask.shape != (batch_size, time_steps):
            raise ValueError(
                f"padding_mask must have shape ({batch_size}, {time_steps}); "
                f"received {tuple(padding_mask.shape)}"
            )
        lengths = (~padding_mask.to(device=values.device)).sum(dim=1, dtype=torch.long)
    else:
        lengths = torch.full(
            (batch_size,),
            time_steps,
            dtype=torch.long,
            device=values.device,
        )

    if bool((lengths < minimum_valid_time_frames).any().item()):
        raise ValueError(
            f"every item must have at least {minimum_valid_time_frames} valid time frames"
        )
    if bool((lengths > time_steps).any().item()):
        raise ValueError(f"feature_lengths cannot exceed the padded time dimension {time_steps}")

    positions = torch.arange(time_steps, device=values.device).unsqueeze(0)
    expected_mask = positions >= lengths.unsqueeze(1)
    if padding_mask is not None:
        if not isinstance(padding_mask, Tensor):
            raise TypeError("padding_mask must be a torch.Tensor")
        if padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must use torch.bool")
        if padding_mask.shape != (batch_size, time_steps):
            raise ValueError(
                f"padding_mask must have shape ({batch_size}, {time_steps}); "
                f"received {tuple(padding_mask.shape)}"
            )
        supplied_mask = padding_mask.to(device=values.device)
        if not torch.equal(supplied_mask, expected_mask):
            raise ValueError(
                "padding_mask must be right-padded and consistent with feature_lengths"
            )
    return lengths, expected_mask


def mask_padded_spectrogram(values: Tensor, padding_mask: Tensor) -> Tensor:
    """Replace all padded spectrogram frames with exact zeros."""

    return values.masked_fill(padding_mask[:, None, None, :], 0.0)


def masked_temporal_mean(sequence: Tensor, padding_mask: Tensor) -> Tensor:
    """Mean-pool a ``B x T x D`` sequence over valid positions only."""

    valid = (~padding_mask).unsqueeze(-1).to(dtype=sequence.dtype)
    denominator = valid.sum(dim=1).clamp_min(1.0)
    return (sequence * valid).sum(dim=1) / denominator


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """Batch normalization whose training statistics exclude padded time frames."""

    def forward(self, values: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if padding_mask is None:
            return super().forward(values)

        expanded_mask = padding_mask[:, None, None, :]
        if not self.training:
            return super().forward(values).masked_fill(expanded_mask, 0.0)

        valid = (~expanded_mask).to(dtype=values.dtype)
        count = valid.sum() * values.shape[2]
        mean = (values * valid).sum(dim=(0, 2, 3)) / count
        centered = values - mean[None, :, None, None]
        variance = (centered.square() * valid).sum(dim=(0, 2, 3)) / count

        if self.track_running_stats:
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    factor = 1.0 / float(self.num_batches_tracked.item())
                else:
                    factor = self.momentum
                if count.item() > 1:
                    unbiased_variance = variance * count / (count - 1.0)
                else:
                    unbiased_variance = variance
                self.running_mean.lerp_(mean.detach().to(self.running_mean.dtype), factor)
                self.running_var.lerp_(
                    unbiased_variance.detach().to(self.running_var.dtype), factor
                )

        normalized = centered * torch.rsqrt(variance[None, :, None, None] + self.eps)
        if self.affine:
            normalized = (
                normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            )
        return normalized.masked_fill(expanded_mask, 0.0)


class ConvNormActivation(nn.Module):
    """A fully specified 3x3 convolution, normalization, activation and pool block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        pool_kernel: tuple[int, int],
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.normalization = MaskedBatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)

    def forward(self, values: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        values = self.convolution(values)
        values = self.normalization(values, padding_mask)
        values = self.activation(values)
        if padding_mask is not None:
            values = values.masked_fill(padding_mask[:, None, None, :], 0.0)
        return self.pool(values)
