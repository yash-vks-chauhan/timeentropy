from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from timeentropy.models import (
    CANONICAL_WAVLM_CHECKPOINT,
    MissingSSLDependencyError,
    WavLMReference,
    create_model,
    ssl_reference,
)


class _TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(layers))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = torch.tanh(layer(values))
        return values


class _TinyWavLMBackbone(nn.Module):
    """Offline stand-in exposing the subset of the WavLMModel contract used here."""

    def __init__(self, hidden_size: int = 8, layers: int = 3) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=hidden_size,
            conv_kernel=(1,),
            conv_stride=(1,),
            num_hidden_layers=layers,
        )
        self.input_projection = nn.Linear(1, hidden_size)
        self.encoder = _TinyEncoder(hidden_size, layers)
        self.last_attention_mask: torch.Tensor | None = None

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> types.SimpleNamespace:
        self.last_attention_mask = attention_mask
        hidden = self.input_projection(input_values.unsqueeze(-1))
        return types.SimpleNamespace(last_hidden_state=self.encoder(hidden))

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return attention_mask[:, :feature_vector_length].to(torch.bool)


def _padded_waveforms() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(59)
    short = torch.randn(11)
    long = torch.randn(19)
    waveforms = torch.full((2, 19), 8_000.0)
    waveforms[0, :11] = short
    waveforms[1] = long
    lengths = torch.tensor([11, 19], dtype=torch.long)
    attention_mask = torch.arange(19).unsqueeze(0) < lengths.unsqueeze(1)
    return short, long, waveforms, attention_mask


def test_frozen_reference_has_binary_output_and_auditable_metadata() -> None:
    backbone = _TinyWavLMBackbone()
    model = WavLMReference(backbone, dropout=0.0).train()
    waveforms = torch.randn(3, 17)
    logits = model(waveforms)

    assert logits.shape == (3, 2)
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())

    metadata = model.metadata
    assert metadata.checkpoint_id == CANONICAL_WAVLM_CHECKPOINT
    assert metadata.checkpoint_revision == "main"
    assert metadata.backbone_source == "injected"
    assert metadata.pooling_method == "masked_mean"
    assert metadata.fine_tuning_status == "frozen_backbone"
    assert metadata.backbone_trainable_parameters == 0
    assert metadata.head_trainable_parameters == metadata.head_total_parameters
    assert metadata.trainable_parameters == metadata.head_trainable_parameters
    assert metadata.total_parameters == (
        metadata.backbone_total_parameters + metadata.head_total_parameters
    )


def test_padding_and_masked_mean_do_not_change_logits() -> None:
    backbone = _TinyWavLMBackbone()
    model = WavLMReference(backbone, dropout=0.0).eval()
    short, long, padded, attention_mask = _padded_waveforms()
    lengths = attention_mask.sum(dim=1)

    with torch.inference_mode():
        expected = torch.cat(
            [
                model(short.unsqueeze(0)),
                model(long.unsqueeze(0)),
            ]
        )
        from_lengths = model(padded, sample_lengths=lengths)
        from_mask = model(padded, attention_mask=attention_mask)
        from_both = model(
            padded,
            sample_lengths=lengths,
            attention_mask=attention_mask.to(torch.long),
        )

    torch.testing.assert_close(from_lengths, expected)
    torch.testing.assert_close(from_mask, expected)
    torch.testing.assert_close(from_both, expected)
    assert backbone.last_attention_mask is not None
    assert backbone.last_attention_mask.tolist() == attention_mask.to(torch.long).tolist()


def test_unfreeze_top_n_changes_only_requested_encoder_layers() -> None:
    backbone = _TinyWavLMBackbone(layers=4)
    model = WavLMReference(
        backbone,
        dropout=0.0,
        unfreeze_top_n=2,
    ).train()

    assert backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.input_projection.parameters())
    for layer in backbone.encoder.layers[:2]:
        assert all(not parameter.requires_grad for parameter in layer.parameters())
    for layer in backbone.encoder.layers[2:]:
        assert all(parameter.requires_grad for parameter in layer.parameters())
    assert model.metadata.fine_tuning_status == "unfreeze_top_n:2"
    assert model.metadata.backbone_trainable_parameters > 0
    assert model.metadata.trainable_parameters > model.metadata.head_trainable_parameters


def test_factory_keeps_wavlm_loading_explicit_and_dependency_lazy() -> None:
    backbone = _TinyWavLMBackbone()
    model = create_model("wavlm", backbone=backbone, dropout=0.0)
    assert isinstance(model, WavLMReference)
    with pytest.raises(ValueError, match="injected backbone or explicit"):
        create_model("wavlm_reference")


def test_missing_transformers_dependency_has_actionable_error(monkeypatch) -> None:
    def missing_import(name: str):
        assert name == "transformers"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(ssl_reference.importlib, "import_module", missing_import)
    with pytest.raises(MissingSSLDependencyError, match=r"pip install -e '.\[ssl\]'"):
        WavLMReference.from_pretrained(local_files_only=True)


def test_attention_mask_must_be_right_padded_and_match_lengths() -> None:
    model = WavLMReference(_TinyWavLMBackbone(), dropout=0.0)
    waveforms = torch.randn(2, 12)
    interior_gap = torch.ones(2, 12, dtype=torch.bool)
    interior_gap[0, 4] = False
    with pytest.raises(ValueError, match="right-padded"):
        model(waveforms, attention_mask=interior_gap)

    inconsistent = torch.ones(2, 12, dtype=torch.bool)
    with pytest.raises(ValueError, match="agree with sample_lengths"):
        model(
            waveforms,
            sample_lengths=torch.tensor([7, 12]),
            attention_mask=inconsistent,
        )


def test_reference_rejects_wrong_sample_rate() -> None:
    model = WavLMReference(_TinyWavLMBackbone(), dropout=0.0)
    with pytest.raises(ValueError, match="requires 16000 Hz"):
        model(torch.randn(2, 12), sample_rate=8_000)
