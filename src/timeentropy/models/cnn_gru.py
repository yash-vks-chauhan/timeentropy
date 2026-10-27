"""Frequency-compressing CNN with a unidirectional GRU temporal encoder.

This is deliberately isomorphic to :mod:`cnn_bilstm`: the convolutional
front end, hidden width, pooling, dropout, and classifier are identical.  The
recurrent cell is the only architectural difference, making the newly
specified baseline directly interpretable against CNN-BiLSTM.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .common import (
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


class CNNGRU(nn.Module):
    """Two-stage spectral CNN feeding a one-layer bidirectional GRU."""

    EXPECTED_TRAINABLE_PARAMETERS = 54_762
    FRAME_FEATURES = 48
    HIDDEN_SIZE = 64

    def __init__(self, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the half-open interval [0, 1)")
        self.dropout_probability = float(dropout)
        self.spectral_frontend = nn.Sequential(
            ConvNormActivation(1, 24, pool_kernel=(2, 1)),
            ConvNormActivation(24, self.FRAME_FEATURES, pool_kernel=(2, 1)),
        )
        self.temporal_encoder = nn.GRU(
            input_size=self.FRAME_FEATURES,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=self.dropout_probability)
        self.classifier = nn.Linear(2 * self.HIDDEN_SIZE, NUM_CLASSES)

    def _extract_masked_frame_features(self, values: Tensor, padding_mask: Tensor) -> Tensor:
        feature_map = mask_padded_spectrogram(values, padding_mask)
        for block in self.spectral_frontend:
            feature_map = block(feature_map, padding_mask)
        sequence = feature_map.mean(dim=2).transpose(1, 2).contiguous()
        return sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def extract_frame_features(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Convert ``B x 1 x 257 x T`` images to ``B x T x 48`` tokens."""

        validate_spectrogram_input(values)
        if feature_lengths is not None or padding_mask is not None:
            _, resolved_mask = resolve_padding(
                values, feature_lengths=feature_lengths, padding_mask=padding_mask
            )
            return self._extract_masked_frame_features(values, resolved_mask)
        return self.spectral_frontend(values).mean(dim=2).transpose(1, 2).contiguous()

    def encode_sequence(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the complete ``B x T x 128`` bidirectional state sequence."""

        validate_spectrogram_input(values)
        if feature_lengths is None and padding_mask is None:
            encoded, _ = self.temporal_encoder(self.extract_frame_features(values))
            return encoded
        lengths, resolved_mask = resolve_padding(
            values, feature_lengths=feature_lengths, padding_mask=padding_mask
        )
        sequence = self._extract_masked_frame_features(values, resolved_mask)
        packed = pack_padded_sequence(
            sequence, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_encoded, _ = self.temporal_encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_encoded, batch_first=True, total_length=values.shape[-1]
        )
        return encoded.masked_fill(resolved_mask.unsqueeze(-1), 0.0)

    def forward_features(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Mean-pool only valid bidirectional states over the time axis."""

        if feature_lengths is None and padding_mask is None:
            return self.encode_sequence(values).mean(dim=1)
        _, resolved_mask = resolve_padding(
            values, feature_lengths=feature_lengths, padding_mask=padding_mask
        )
        return masked_temporal_mean(
            self.encode_sequence(
                values, feature_lengths=feature_lengths, padding_mask=padding_mask
            ),
            resolved_mask,
        )

    def forward(
        self,
        values: Tensor,
        *,
        feature_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return unnormalized binary-class logits with shape ``B x 2``."""

        features = self.forward_features(
            values, feature_lengths=feature_lengths, padding_mask=padding_mask
        )
        return self.classifier(self.dropout(features))

    @property
    def metadata(self) -> ArchitectureMetadata:
        return ArchitectureMetadata(
            name="cnn_gru",
            version="baseline-v1",
            input_layout="B x 1 x 257 x T (T >= 1)",
            output_layout="B x 2 logits ordered (control, dysarthric)",
            variable_time_axis=True,
            time_axis_behavior="preserved by CNN, packed for BiGRU, then pooled over valid states",
            dropout_probability=self.dropout_probability,
            trainable_parameters=count_trainable_parameters(self),
            expected_trainable_parameters=self.EXPECTED_TRAINABLE_PARAMETERS,
            stages=(
                "Conv2d(1,24,3x3,s1,p1,bias=False)-BN-SiLU-MaxPool2d(2x1)",
                "Conv2d(24,48,3x3,s1,p1,bias=False)-BN-SiLU-MaxPool2d(2x1)",
                "frequency mean -> BxTx48",
                "packed BiGRU(input=48,hidden=64,layers=1,bidirectional=True)",
                "masked temporal mean-Dropout-Linear(128,2)",
            ),
            implementation=("recurrent_cell=nn.GRU", "comparison_model=cnn_bilstm"),
        )
