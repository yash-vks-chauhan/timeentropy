"""Compact three-block CNN for onset-aligned log-STFT images.

Layer specification for the default two-class model::

    B x 1 x 257 x T
      -> Conv(1,16,3x3)-BN-SiLU-MaxPool(2x2)
      -> Conv(16,32,3x3)-BN-SiLU-MaxPool(2x2)
      -> Conv(32,64,3x3)-BN-SiLU-MaxPool(2x2)
      -> global mean over frequency and time
      -> Dropout(0.30) -> Linear(64,2)

The three pools require at least eight input frames.  All documented prefix
representations (150-300 ms) exceed that minimum.  The architecture contains
23,538 trainable parameters, matching the count recorded in the surviving
manuscript while making every layer explicit here. For right-padded batches,
only valid downsampled time positions contribute to global pooling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import (
    ARCHITECTURE_VERSION,
    DEFAULT_DROPOUT,
    NUM_CLASSES,
    ArchitectureMetadata,
    ConvNormActivation,
    count_trainable_parameters,
    mask_padded_spectrogram,
    resolve_padding,
    validate_spectrogram_input,
)


class SmallCNN(nn.Module):
    """Three convolutional blocks followed by global time-frequency pooling."""

    EXPECTED_TRAINABLE_PARAMETERS = 23_538
    MINIMUM_TIME_FRAMES = 8

    def __init__(self, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the half-open interval [0, 1)")
        self.dropout_probability = float(dropout)
        self.features = nn.Sequential(
            ConvNormActivation(1, 16, pool_kernel=(2, 2)),
            ConvNormActivation(16, 32, pool_kernel=(2, 2)),
            ConvNormActivation(32, 64, pool_kernel=(2, 2)),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=self.dropout_probability)
        self.classifier = nn.Linear(64, NUM_CLASSES)

    def _extract_masked_feature_map(
        self,
        values: Tensor,
        feature_lengths: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        values = mask_padded_spectrogram(values, padding_mask)
        lengths = feature_lengths
        current_mask = padding_mask
        for block in self.features:
            values = block(values, current_mask)
            lengths = torch.div(lengths, 2, rounding_mode="floor")
            positions = torch.arange(values.shape[-1], device=values.device).unsqueeze(0)
            current_mask = positions >= lengths.unsqueeze(1)
            values = values.masked_fill(current_mask[:, None, None, :], 0.0)
        return values, lengths, current_mask

    def extract_feature_map(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the convolutional map before global pooling.

        Each block halves both frequency and time, so an input with ``T`` time
        frames produces ``floor(floor(floor(T/2)/2)/2)`` map frames.
        """

        validate_spectrogram_input(values, minimum_time_frames=self.MINIMUM_TIME_FRAMES)
        if feature_lengths is not None or padding_mask is not None:
            lengths, resolved_mask = resolve_padding(
                values,
                feature_lengths=feature_lengths,
                padding_mask=padding_mask,
                minimum_valid_time_frames=self.MINIMUM_TIME_FRAMES,
            )
            feature_map, _, _ = self._extract_masked_feature_map(values, lengths, resolved_mask)
            return feature_map
        return self.features(values)

    def forward_features(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Pool frequency and only the valid downsampled time positions."""

        validate_spectrogram_input(values, minimum_time_frames=self.MINIMUM_TIME_FRAMES)
        if feature_lengths is None and padding_mask is None:
            return self.global_pool(self.features(values)).flatten(start_dim=1)

        lengths, resolved_mask = resolve_padding(
            values,
            feature_lengths=feature_lengths,
            padding_mask=padding_mask,
            minimum_valid_time_frames=self.MINIMUM_TIME_FRAMES,
        )
        feature_map, downsampled_lengths, downsampled_mask = self._extract_masked_feature_map(
            values, lengths, resolved_mask
        )
        valid = (~downsampled_mask)[:, None, None, :].to(dtype=feature_map.dtype)
        summed = (feature_map * valid).sum(dim=(2, 3))
        denominator = (
            downsampled_lengths.to(dtype=feature_map.dtype) * feature_map.shape[2]
        ).unsqueeze(1)
        return summed / denominator

    def forward(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return unnormalized binary-class logits with shape ``B x 2``."""

        features = self.forward_features(
            values,
            feature_lengths=feature_lengths,
            padding_mask=padding_mask,
        )
        return self.classifier(self.dropout(features))

    @property
    def metadata(self) -> ArchitectureMetadata:
        """Describe the exact architecture instantiated by this module."""

        return ArchitectureMetadata(
            name="small_cnn",
            version=ARCHITECTURE_VERSION,
            input_layout="B x 1 x 257 x T (T >= 8)",
            output_layout="B x 2 logits",
            variable_time_axis=True,
            time_axis_behavior=(
                "three 2x downsamplings, then length-aware mean pooling over valid positions"
            ),
            dropout_probability=self.dropout_probability,
            trainable_parameters=count_trainable_parameters(self),
            expected_trainable_parameters=self.EXPECTED_TRAINABLE_PARAMETERS,
            stages=(
                "Conv2d(1,16,3x3,s1,p1,bias=False)-BN-SiLU-MaxPool2d(2x2)",
                "Conv2d(16,32,3x3,s1,p1,bias=False)-BN-SiLU-MaxPool2d(2x2)",
                "Conv2d(32,64,3x3,s1,p1,bias=False)-BN-SiLU-MaxPool2d(2x2)",
                "AdaptiveAvgPool2d(1x1)-Dropout-Linear(64,2)",
            ),
        )
