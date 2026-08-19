"""Optional WavLM reference model for raw 16 kHz speech.

The primary feasible self-supervised baseline keeps the pretrained
``microsoft/wavlm-base-plus`` backbone frozen and trains only a dropout/linear
binary head over a padding-masked mean of the final hidden states.  A controlled
``unfreeze_top_n`` mode can fine-tune only the last N Transformer encoder
layers.  Importing this module never imports Transformers or downloads weights;
``from_pretrained`` performs the optional dependency import lazily.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .common import DEFAULT_DROPOUT, NUM_CLASSES

CANONICAL_WAVLM_CHECKPOINT = "microsoft/wavlm-base-plus"
DEFAULT_CHECKPOINT_REVISION = "main"
REQUIRED_SAMPLE_RATE = 16_000
FineTuningMode = Literal["frozen_backbone", "unfreeze_top_n"]


class MissingSSLDependencyError(RuntimeError):
    """Raised when pretrained WavLM loading is requested without Transformers."""


@dataclass(frozen=True)
class SSLReferenceMetadata:
    """Auditable configuration and parameter totals for the WavLM reference."""

    name: str
    checkpoint_id: str
    checkpoint_revision: str | None
    backbone_source: str
    input_layout: str
    sample_rate: int
    pooling_method: str
    fine_tuning_status: str
    unfreeze_top_n: int
    hidden_size: int
    num_classes: int
    dropout_probability: float
    total_parameters: int
    trainable_parameters: int
    backbone_total_parameters: int
    backbone_trainable_parameters: int
    head_total_parameters: int
    head_trainable_parameters: int


def _count_parameters(module: nn.Module, *, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _import_wavlm_model() -> type[nn.Module]:
    """Import Hugging Face WavLM only when pretrained loading is requested."""

    try:
        transformers = importlib.import_module("transformers")
        return transformers.WavLMModel
    except (ImportError, AttributeError) as error:
        raise MissingSSLDependencyError(
            "WavLM requires the optional SSL dependencies. Install them with "
            "`pip install -e '.[ssl]'` before loading pretrained weights."
        ) from error


def _validate_raw_audio(
    input_values: Tensor,
    *,
    sample_lengths: Tensor | None,
    attention_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if not isinstance(input_values, Tensor):
        raise TypeError("input_values must be a torch.Tensor")
    if input_values.ndim != 2:
        raise ValueError(
            f"input_values must have shape B x samples; received {tuple(input_values.shape)}"
        )
    if input_values.shape[0] < 1 or input_values.shape[1] < 1:
        raise ValueError("input_values must contain at least one item and one sample")
    if not input_values.is_floating_point():
        raise TypeError("input_values must use a floating-point dtype")

    batch_size, padded_samples = input_values.shape
    if sample_lengths is not None:
        if not isinstance(sample_lengths, Tensor):
            raise TypeError("sample_lengths must be a torch.Tensor")
        if sample_lengths.ndim != 1 or sample_lengths.shape[0] != batch_size:
            raise ValueError(
                f"sample_lengths must have shape ({batch_size},); "
                f"received {tuple(sample_lengths.shape)}"
            )
        if sample_lengths.dtype == torch.bool or sample_lengths.is_floating_point():
            raise TypeError("sample_lengths must use an integer dtype")
        lengths = sample_lengths.to(device=input_values.device, dtype=torch.long)
    elif attention_mask is not None:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")
        if attention_mask.shape != (batch_size, padded_samples):
            raise ValueError(
                f"attention_mask must have shape ({batch_size}, {padded_samples}); "
                f"received {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            valid_mask = attention_mask.to(device=input_values.device)
        elif not attention_mask.is_floating_point() and not attention_mask.is_complex():
            valid_mask = attention_mask.to(device=input_values.device) != 0
        else:
            raise TypeError("attention_mask must use bool or an integer dtype")
        lengths = valid_mask.sum(dim=1, dtype=torch.long)
    else:
        lengths = torch.full(
            (batch_size,),
            padded_samples,
            dtype=torch.long,
            device=input_values.device,
        )

    if bool((lengths < 1).any().item()):
        raise ValueError("every waveform must contain at least one valid sample")
    if bool((lengths > padded_samples).any().item()):
        raise ValueError(f"sample_lengths cannot exceed padded size {padded_samples}")

    positions = torch.arange(padded_samples, device=input_values.device).unsqueeze(0)
    expected_valid_mask = positions < lengths.unsqueeze(1)
    if attention_mask is not None:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")
        if attention_mask.shape != (batch_size, padded_samples):
            raise ValueError(
                f"attention_mask must have shape ({batch_size}, {padded_samples}); "
                f"received {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            supplied_valid_mask = attention_mask.to(device=input_values.device)
        elif not attention_mask.is_floating_point() and not attention_mask.is_complex():
            supplied_valid_mask = attention_mask.to(device=input_values.device) != 0
        else:
            raise TypeError("attention_mask must use bool or an integer dtype")
        if not torch.equal(supplied_valid_mask, expected_valid_mask):
            raise ValueError(
                "attention_mask must use 1/True for valid samples, be right-padded, "
                "and agree with sample_lengths"
            )
    return lengths, expected_valid_mask


class WavLMReference(nn.Module):
    """WavLM last-hidden-state pooling with a small binary classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        config: Any | None = None,
        checkpoint_id: str = CANONICAL_WAVLM_CHECKPOINT,
        checkpoint_revision: str | None = DEFAULT_CHECKPOINT_REVISION,
        dropout: float = DEFAULT_DROPOUT,
        pooling_method: Literal["masked_mean"] = "masked_mean",
        fine_tuning: FineTuningMode | None = None,
        unfreeze_top_n: int = 0,
        backbone_source: str = "injected",
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the half-open interval [0, 1)")
        if pooling_method != "masked_mean":
            raise ValueError("pooling_method must be 'masked_mean'")
        if unfreeze_top_n < 0:
            raise ValueError("unfreeze_top_n cannot be negative")
        resolved_fine_tuning: FineTuningMode
        if fine_tuning is None:
            resolved_fine_tuning = "unfreeze_top_n" if unfreeze_top_n > 0 else "frozen_backbone"
        elif fine_tuning in ("frozen_backbone", "unfreeze_top_n"):
            resolved_fine_tuning = fine_tuning
        else:
            raise ValueError("fine_tuning must be 'frozen_backbone' or 'unfreeze_top_n'")
        if resolved_fine_tuning == "frozen_backbone" and unfreeze_top_n != 0:
            raise ValueError("frozen_backbone requires unfreeze_top_n=0")
        if resolved_fine_tuning == "unfreeze_top_n" and unfreeze_top_n < 1:
            raise ValueError("unfreeze_top_n mode requires a positive layer count")

        resolved_config = config if config is not None else getattr(backbone, "config", None)
        hidden_size = getattr(resolved_config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("backbone config must define a positive integer hidden_size")

        self.backbone = backbone
        self.backbone_config = resolved_config
        self.checkpoint_id = checkpoint_id
        self.checkpoint_revision = checkpoint_revision
        self.backbone_source = backbone_source
        self.dropout_probability = float(dropout)
        self.pooling_method = pooling_method
        self.fine_tuning = resolved_fine_tuning
        self.unfreeze_top_n = int(unfreeze_top_n)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=self.dropout_probability)
        self.classifier = nn.Linear(self.hidden_size, NUM_CLASSES)

        self._configure_backbone_trainability()
        if self.fine_tuning == "frozen_backbone":
            self.backbone.eval()

    @classmethod
    def from_pretrained(
        cls,
        *,
        checkpoint_id: str = CANONICAL_WAVLM_CHECKPOINT,
        revision: str | None = DEFAULT_CHECKPOINT_REVISION,
        local_files_only: bool = False,
        dropout: float = DEFAULT_DROPOUT,
        pooling_method: Literal["masked_mean"] = "masked_mean",
        fine_tuning: FineTuningMode | None = None,
        unfreeze_top_n: int = 0,
        **pretrained_kwargs: Any,
    ) -> WavLMReference:
        """Explicitly load a Hugging Face WavLM checkpoint.

        This is the only path that can access the Hugging Face cache or network.
        Tests should inject a backbone into ``__init__`` instead.
        """

        wavlm_model = _import_wavlm_model()
        backbone = wavlm_model.from_pretrained(
            checkpoint_id,
            revision=revision,
            local_files_only=local_files_only,
            **pretrained_kwargs,
        )
        return cls(
            backbone,
            checkpoint_id=checkpoint_id,
            checkpoint_revision=revision,
            dropout=dropout,
            pooling_method=pooling_method,
            fine_tuning=fine_tuning,
            unfreeze_top_n=unfreeze_top_n,
            backbone_source="huggingface_pretrained",
        )

    def _encoder_layers(self) -> nn.ModuleList:
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("backbone must expose encoder.layers as torch.nn.ModuleList")
        return layers

    def _configure_backbone_trainability(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if self.fine_tuning == "unfreeze_top_n":
            layers = self._encoder_layers()
            if self.unfreeze_top_n > len(layers):
                raise ValueError(
                    f"cannot unfreeze {self.unfreeze_top_n} layers; backbone has {len(layers)}"
                )
            for layer in layers[-self.unfreeze_top_n :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def train(self, mode: bool = True) -> WavLMReference:
        """Keep a completely frozen backbone deterministic during head training."""

        super().train(mode)
        if self.fine_tuning == "frozen_backbone":
            self.backbone.eval()
        return self

    def _extract_last_hidden_state(self, outputs: Any) -> Tensor:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise ValueError("backbone output must provide last_hidden_state with shape B x T x D")
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"backbone hidden size is {hidden.shape[-1]}, expected {self.hidden_size}"
            )
        return hidden

    def _fallback_feature_lengths(self, sample_lengths: Tensor) -> Tensor:
        kernels = getattr(self.backbone_config, "conv_kernel", None)
        strides = getattr(self.backbone_config, "conv_stride", None)
        if kernels is None or strides is None or len(kernels) != len(strides):
            raise ValueError(
                "backbone must provide _get_feature_vector_attention_mask or config "
                "conv_kernel/conv_stride for padding-safe pooling"
            )
        feature_lengths = sample_lengths
        for kernel, stride in zip(kernels, strides, strict=True):
            feature_lengths = (
                torch.div(
                    feature_lengths - int(kernel),
                    int(stride),
                    rounding_mode="floor",
                )
                + 1
            )
        return feature_lengths.clamp_min(0)

    def _feature_attention_mask(
        self,
        *,
        hidden_time_steps: int,
        sample_lengths: Tensor,
        sample_attention_mask: Tensor,
    ) -> Tensor:
        mask_helper = getattr(self.backbone, "_get_feature_vector_attention_mask", None)
        if callable(mask_helper):
            feature_mask = mask_helper(hidden_time_steps, sample_attention_mask.to(torch.long))
            if not isinstance(feature_mask, Tensor):
                raise TypeError("backbone feature-mask helper must return a torch.Tensor")
            feature_mask = feature_mask.to(device=sample_attention_mask.device, dtype=torch.bool)
        else:
            feature_lengths = self._fallback_feature_lengths(sample_lengths)
            positions = torch.arange(hidden_time_steps, device=sample_lengths.device).unsqueeze(0)
            feature_mask = positions < feature_lengths.unsqueeze(1)
        if feature_mask.shape != (sample_attention_mask.shape[0], hidden_time_steps):
            raise ValueError(
                "feature attention mask shape does not match last_hidden_state time dimension"
            )
        if bool((feature_mask.sum(dim=1) < 1).any().item()):
            raise ValueError("waveform is too short to produce a valid WavLM feature frame")
        return feature_mask

    def forward_features(
        self,
        input_values: Tensor,
        *,
        sample_lengths: Tensor | None = None,
        attention_mask: Tensor | None = None,
        sample_rate: int = REQUIRED_SAMPLE_RATE,
    ) -> Tensor:
        """Return a padding-safe masked mean of final WavLM hidden states."""

        if sample_rate != REQUIRED_SAMPLE_RATE:
            raise ValueError(
                f"WavLM reference requires {REQUIRED_SAMPLE_RATE} Hz audio; received {sample_rate}"
            )
        lengths, valid_samples = _validate_raw_audio(
            input_values,
            sample_lengths=sample_lengths,
            attention_mask=attention_mask,
        )
        masked_input = input_values.masked_fill(~valid_samples, 0.0)
        backbone_outputs = self.backbone(
            input_values=masked_input,
            attention_mask=valid_samples.to(torch.long),
        )
        hidden = self._extract_last_hidden_state(backbone_outputs)
        feature_mask = self._feature_attention_mask(
            hidden_time_steps=hidden.shape[1],
            sample_lengths=lengths,
            sample_attention_mask=valid_samples,
        )
        valid_features = feature_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denominator = valid_features.sum(dim=1).clamp_min(1.0)
        return (hidden * valid_features).sum(dim=1) / denominator

    def forward(
        self,
        input_values: Tensor,
        *,
        sample_lengths: Tensor | None = None,
        attention_mask: Tensor | None = None,
        sample_rate: int = REQUIRED_SAMPLE_RATE,
    ) -> Tensor:
        """Return unnormalized control/dysarthric logits with shape ``B x 2``."""

        pooled = self.forward_features(
            input_values,
            sample_lengths=sample_lengths,
            attention_mask=attention_mask,
            sample_rate=sample_rate,
        )
        return self.classifier(self.dropout(pooled))

    @property
    def metadata(self) -> SSLReferenceMetadata:
        """Return checkpoint, pooling, tuning and current parameter totals."""

        if self.fine_tuning == "frozen_backbone":
            status = "frozen_backbone"
        else:
            status = f"unfreeze_top_n:{self.unfreeze_top_n}"
        return SSLReferenceMetadata(
            name="wavlm_reference",
            checkpoint_id=self.checkpoint_id,
            checkpoint_revision=self.checkpoint_revision,
            backbone_source=self.backbone_source,
            input_layout="B x samples raw mono waveform",
            sample_rate=REQUIRED_SAMPLE_RATE,
            pooling_method=self.pooling_method,
            fine_tuning_status=status,
            unfreeze_top_n=self.unfreeze_top_n,
            hidden_size=self.hidden_size,
            num_classes=NUM_CLASSES,
            dropout_probability=self.dropout_probability,
            total_parameters=_count_parameters(self, trainable_only=False),
            trainable_parameters=_count_parameters(self, trainable_only=True),
            backbone_total_parameters=_count_parameters(self.backbone, trainable_only=False),
            backbone_trainable_parameters=_count_parameters(self.backbone, trainable_only=True),
            head_total_parameters=_count_parameters(self.classifier, trainable_only=False),
            head_trainable_parameters=_count_parameters(self.classifier, trainable_only=True),
        )
