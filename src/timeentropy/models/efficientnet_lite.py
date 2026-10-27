"""Specified EfficientNet-Lite0 baseline for log-STFT classification."""

from __future__ import annotations

from importlib import metadata as importlib_metadata

from torch import Tensor, nn

from .common import (
    DEFAULT_DROPOUT,
    ArchitectureMetadata,
    count_trainable_parameters,
    validate_spectrogram_input,
)


class MissingBaselineDependencyError(ImportError):
    """Raised when an optional baseline dependency was not installed."""


class EfficientNetLite0(nn.Module):
    """``timm`` EfficientNet-Lite0, newly specified for one-channel log-STFT input."""

    MODEL_NAME = "efficientnet_lite0"
    EXPECTED_TRAINABLE_PARAMETERS = 3_372_994

    def __init__(self, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the half-open interval [0, 1)")
        try:
            import timm
        except ImportError as error:
            raise MissingBaselineDependencyError(
                "EfficientNetLite0 requires the 'baselines' dependency group; "
                "install with `pip install -e '.[baselines]'`."
            ) from error
        self.dropout_probability = float(dropout)
        self.timm_version = importlib_metadata.version("timm")
        self.backbone = timm.create_model(
            self.MODEL_NAME,
            pretrained=False,
            in_chans=1,
            num_classes=2,
            drop_rate=self.dropout_probability,
        )

    def forward(self, values: Tensor) -> Tensor:
        """Return logits ordered ``(control, dysarthric)`` for ``B x 1 x 257 x T`` input."""

        validate_spectrogram_input(values)
        logits = self.backbone(values)
        if logits.shape != (values.shape[0], 2):
            raise RuntimeError(f"EfficientNet-Lite0 returned unexpected logits shape {tuple(logits.shape)}")
        return logits

    @property
    def metadata(self) -> ArchitectureMetadata:
        return ArchitectureMetadata(
            name="efficientnet_lite0",
            version="baseline-v1",
            input_layout="B x 1 x 257 x T (T >= 1)",
            output_layout="B x 2 logits ordered (control, dysarthric)",
            variable_time_axis=True,
            time_axis_behavior="EfficientNet convolutional reductions followed by global pooling",
            dropout_probability=self.dropout_probability,
            trainable_parameters=count_trainable_parameters(self),
            expected_trainable_parameters=self.EXPECTED_TRAINABLE_PARAMETERS,
            stages=("timm EfficientNet-Lite0 feature extractor", "global pooling-dropout-linear(2)"),
            implementation=(
                f"timm_version={self.timm_version}",
                (
                    "timm.create_model(model_name='efficientnet_lite0', pretrained=False, "
                    f"in_chans=1, num_classes=2, drop_rate={self.dropout_probability:.2f})"
                ),
            ),
        )
