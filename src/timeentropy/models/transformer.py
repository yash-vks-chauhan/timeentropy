"""Two-layer short-prefix Transformer with explicit frame tokenization.

Layer specification for the default two-class model::

    B x 1 x 257 x T
      -> Conv(1,24,3x3)-BN-SiLU-MaxPool(2x1)
      -> Conv(24,48,3x3)-BN-SiLU-MaxPool(2x1)
      -> adaptive frequency pooling to two bins
      -> flatten 48 channels x 2 bins = 96 features at each time step
      -> bias-free Linear(96,96) + sinusoidal position encoding
      -> 2 x TransformerEncoderLayer(d=96, heads=4, FFN=256, GELU)
      -> mean over T
      -> Dropout(0.30)-Linear(96,24)-SiLU-Dropout(0.30)-Linear(24,2)

All convolutional pooling is frequency-only, and adaptive pooling changes only
frequency.  Consequently tokenization preserves exactly ``T`` temporal tokens.
Right-padding is excluded from attention and temporal pooling. The architecture
contains 196,594 trainable parameters, matching the count recorded in the
surviving manuscript.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import (
    ARCHITECTURE_VERSION,
    DEFAULT_DROPOUT,
    NUM_CLASSES,
    ArchitectureMetadata,
    ConvNormActivation,
    count_trainable_parameters,
    mask_padded_spectrogram,
    masked_temporal_mean,
    resolve_padding,
    validate_spectrogram_input,
)


class SinusoidalPositionEncoding(nn.Module):
    """Parameter-free sinusoidal positions generated for the current ``T``."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 2 or embedding_dim % 2:
            raise ValueError("embedding_dim must be a positive even integer")
        self.embedding_dim = embedding_dim

    def forward(self, values: Tensor) -> Tensor:
        """Add deterministic positions to a ``B x T x D`` token tensor."""

        if values.ndim != 3 or values.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"position input must have shape B x T x {self.embedding_dim}; "
                f"received {tuple(values.shape)}"
            )
        time_steps = values.shape[1]
        positions = torch.arange(time_steps, dtype=values.dtype, device=values.device).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=values.dtype, device=values.device)
            * (-math.log(10_000.0) / self.embedding_dim)
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.empty(
            (time_steps, self.embedding_dim), dtype=values.dtype, device=values.device
        )
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        return values + encoding.unsqueeze(0)


class ShortPrefixTransformer(nn.Module):
    """Convolutional frame tokenizer followed by a two-layer Transformer encoder."""

    EXPECTED_TRAINABLE_PARAMETERS = 196_594
    CONV_CHANNELS = 48
    FREQUENCY_TOKENS = 2
    EMBEDDING_DIM = 96
    ATTENTION_HEADS = 4
    ENCODER_LAYERS = 2
    FEEDFORWARD_DIM = 256
    CLASSIFIER_HIDDEN = 24

    def __init__(self, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the half-open interval [0, 1)")
        self.dropout_probability = float(dropout)
        self.spectral_frontend = nn.Sequential(
            ConvNormActivation(1, 24, pool_kernel=(2, 1)),
            ConvNormActivation(24, self.CONV_CHANNELS, pool_kernel=(2, 1)),
        )
        self.token_projection = nn.Linear(
            self.CONV_CHANNELS * self.FREQUENCY_TOKENS,
            self.EMBEDDING_DIM,
            bias=False,
        )
        self.position_encoding = SinusoidalPositionEncoding(self.EMBEDDING_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.EMBEDDING_DIM,
            nhead=self.ATTENTION_HEADS,
            dim_feedforward=self.FEEDFORWARD_DIM,
            dropout=self.dropout_probability,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.ENCODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_probability),
            nn.Linear(self.EMBEDDING_DIM, self.CLASSIFIER_HIDDEN),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.dropout_probability),
            nn.Linear(self.CLASSIFIER_HIDDEN, NUM_CLASSES),
        )

    def _tokenize_with_mask(self, values: Tensor, padding_mask: Tensor | None) -> Tensor:
        if padding_mask is None:
            feature_map = self.spectral_frontend(values)
        else:
            feature_map = mask_padded_spectrogram(values, padding_mask)
            for block in self.spectral_frontend:
                feature_map = block(feature_map, padding_mask)

        time_steps = feature_map.shape[-1]
        pooled = F.adaptive_avg_pool2d(
            feature_map,
            output_size=(self.FREQUENCY_TOKENS, time_steps),
        )
        # B x C x F' x T -> B x T x (C*F').  The time axis is never pooled.
        flattened = pooled.permute(0, 3, 1, 2).reshape(
            pooled.shape[0],
            time_steps,
            self.CONV_CHANNELS * self.FREQUENCY_TOKENS,
        )
        tokens = self.position_encoding(self.token_projection(flattened))
        if padding_mask is not None:
            tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return tokens

    def tokenize(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Convert a spectrogram to exactly ``T`` positional ``96-D`` tokens."""

        validate_spectrogram_input(values)
        if feature_lengths is None and padding_mask is None:
            return self._tokenize_with_mask(values, None)
        _, resolved_mask = resolve_padding(
            values,
            feature_lengths=feature_lengths,
            padding_mask=padding_mask,
        )
        return self._tokenize_with_mask(values, resolved_mask)

    def encode_tokens(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the complete encoded sequence with shape ``B x T x 96``."""

        validate_spectrogram_input(values)
        if feature_lengths is None and padding_mask is None:
            return self.encoder(self._tokenize_with_mask(values, None))

        _, resolved_mask = resolve_padding(
            values,
            feature_lengths=feature_lengths,
            padding_mask=padding_mask,
        )
        tokens = self._tokenize_with_mask(values, resolved_mask)
        encoded = self.encoder(tokens, src_key_padding_mask=resolved_mask)
        return encoded.masked_fill(resolved_mask.unsqueeze(-1), 0.0)

    def forward_features(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Mean-pool encoded tokens over valid time positions only."""

        if feature_lengths is None and padding_mask is None:
            return self.encode_tokens(values).mean(dim=1)

        _, resolved_mask = resolve_padding(
            values,
            feature_lengths=feature_lengths,
            padding_mask=padding_mask,
        )
        encoded = self.encode_tokens(
            values,
            feature_lengths=feature_lengths,
            padding_mask=padding_mask,
        )
        return masked_temporal_mean(encoded, resolved_mask)

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
        return self.classifier(features)

    @property
    def metadata(self) -> ArchitectureMetadata:
        """Describe the exact architecture instantiated by this module."""

        return ArchitectureMetadata(
            name="short_prefix_transformer",
            version=ARCHITECTURE_VERSION,
            input_layout="B x 1 x 257 x T (T >= 1)",
            output_layout="B x 2 logits",
            variable_time_axis=True,
            time_axis_behavior=(
                "preserved by CNN/tokenizer, padding-masked in attention, then masked-mean pooled"
            ),
            dropout_probability=self.dropout_probability,
            trainable_parameters=count_trainable_parameters(self),
            expected_trainable_parameters=self.EXPECTED_TRAINABLE_PARAMETERS,
            stages=(
                "Conv2d(1,24,3x3,s1,p1,bias=False)-BN-SiLU-MaxPool2d(2x1)",
                "Conv2d(24,48,3x3,s1,p1,bias=False)-BN-SiLU-MaxPool2d(2x1)",
                "AdaptiveAvgPool frequency to 2; flatten 48x2 per time step",
                "Linear(96,96,bias=False)-sinusoidal position encoding",
                "2x masked TransformerEncoderLayer(d=96,heads=4,ff=256,GELU,dropout=0.30)",
                "masked temporal mean-Dropout-Linear(96,24)-SiLU-Dropout-Linear(24,2)",
            ),
        )
