from __future__ import annotations

import copy

import pytest
import torch

from timeentropy.models import (
    CNNGRU,
    CNNBiLSTM,
    EfficientNetLite0,
    ShortPrefixTransformer,
    SmallCNN,
    available_model_names,
    count_trainable_parameters,
    create_model,
)


def _mixed_length_batch(
    short: torch.Tensor,
    long: torch.Tensor,
    *,
    padded_time: int,
    padded_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([short.shape[-1], long.shape[-1]], dtype=torch.long)
    values = torch.full(
        (2, 1, 257, padded_time),
        fill_value=padded_value,
        dtype=short.dtype,
    )
    values[0, ..., : short.shape[-1]] = short
    values[1, ..., : long.shape[-1]] = long
    positions = torch.arange(padded_time).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    return values, lengths, padding_mask


@pytest.mark.parametrize(
    ("model_type", "expected_parameters"),
    [
        (SmallCNN, 23_538),
        (CNNBiLSTM, 69_354),
        (CNNGRU, 54_762),
        (EfficientNetLite0, 3_372_994),
        (ShortPrefixTransformer, 196_594),
    ],
)
def test_parameter_counts_match_frozen_architecture(model_type, expected_parameters: int) -> None:
    model = model_type()
    assert count_trainable_parameters(model) == expected_parameters
    assert model.metadata.trainable_parameters == expected_parameters
    assert model.metadata.expected_trainable_parameters == expected_parameters
    assert model.metadata.dropout_probability == pytest.approx(0.30)
    assert model.metadata.variable_time_axis


@pytest.mark.parametrize("time_frames", [13, 18, 23, 28, 37])
@pytest.mark.parametrize("model_type", [SmallCNN, CNNBiLSTM, CNNGRU, ShortPrefixTransformer])
def test_models_accept_multiple_prefix_lengths(model_type, time_frames: int) -> None:
    torch.manual_seed(17)
    model = model_type().eval()
    values = torch.randn(2, 1, 257, time_frames)
    with torch.inference_mode():
        logits = model(values)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("time_frames", [13, 28, 41])
def test_cnn_bilstm_preserves_time_until_temporal_pooling(time_frames: int) -> None:
    model = CNNBiLSTM().eval()
    values = torch.randn(2, 1, 257, time_frames)
    with torch.inference_mode():
        frame_features = model.extract_frame_features(values)
        encoded = model.encode_sequence(values)
        pooled = model.forward_features(values)
    assert frame_features.shape == (2, time_frames, 48)
    assert encoded.shape == (2, time_frames, 128)
    assert pooled.shape == (2, 128)


@pytest.mark.parametrize("time_frames", [13, 28, 41])
def test_cnn_gru_preserves_time_until_temporal_pooling(time_frames: int) -> None:
    model = CNNGRU().eval()
    values = torch.randn(2, 1, 257, time_frames)
    with torch.inference_mode():
        frame_features = model.extract_frame_features(values)
        encoded = model.encode_sequence(values)
        pooled = model.forward_features(values)
    assert frame_features.shape == (2, time_frames, 48)
    assert encoded.shape == (2, time_frames, 128)
    assert pooled.shape == (2, 128)


@pytest.mark.parametrize("time_frames", [13, 28, 41])
def test_transformer_preserves_time_during_tokenization_and_encoding(time_frames: int) -> None:
    model = ShortPrefixTransformer().eval()
    values = torch.randn(2, 1, 257, time_frames)
    with torch.inference_mode():
        tokens = model.tokenize(values)
        encoded = model.encode_tokens(values)
        pooled = model.forward_features(values)
    assert tokens.shape == (2, time_frames, 96)
    assert encoded.shape == (2, time_frames, 96)
    assert pooled.shape == (2, 96)


def test_small_cnn_documents_time_downsampling_before_global_pooling() -> None:
    model = SmallCNN().eval()
    values = torch.randn(2, 1, 257, 28)
    with torch.inference_mode():
        feature_map = model.extract_feature_map(values)
        pooled = model.forward_features(values)
    assert feature_map.shape == (2, 64, 32, 3)
    assert pooled.shape == (2, 64)


@pytest.mark.parametrize("model_type", [SmallCNN, CNNBiLSTM, CNNGRU, ShortPrefixTransformer])
def test_models_reject_wrong_frequency_dimension(model_type) -> None:
    model = model_type()
    with pytest.raises(ValueError, match="257 frequency bins"):
        model(torch.randn(2, 1, 256, 28))


def test_small_cnn_rejects_time_axis_too_short_for_three_pools() -> None:
    model = SmallCNN()
    with pytest.raises(ValueError, match="at least 8 time frames"):
        model(torch.randn(2, 1, 257, 7))


@pytest.mark.parametrize("model_type", [SmallCNN, CNNBiLSTM, CNNGRU, ShortPrefixTransformer])
def test_right_padding_does_not_change_eval_logits(model_type) -> None:
    torch.manual_seed(31)
    model = model_type().eval()
    short = torch.randn(1, 257, 13)
    long = torch.randn(1, 257, 28)
    padded, lengths, padding_mask = _mixed_length_batch(
        short,
        long,
        padded_time=28,
        padded_value=10_000.0,
    )

    with torch.inference_mode():
        expected = torch.cat(
            [
                model(short.unsqueeze(0)),
                model(long.unsqueeze(0)),
            ]
        )
        from_lengths = model(padded, feature_lengths=lengths)
        from_mask = model(padded, padding_mask=padding_mask)
        from_both = model(
            padded,
            feature_lengths=lengths,
            padding_mask=padding_mask,
        )

    torch.testing.assert_close(from_lengths, expected, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(from_mask, expected, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(from_both, expected, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize("model_type", [SmallCNN, CNNBiLSTM, ShortPrefixTransformer])
def test_extra_right_padding_does_not_change_training_logits(model_type) -> None:
    """Masked batch normalization also excludes padding during optimization."""

    torch.manual_seed(47)
    short = torch.randn(1, 257, 13)
    long = torch.randn(1, 257, 28)
    compact, compact_lengths, compact_mask = _mixed_length_batch(
        short,
        long,
        padded_time=28,
        padded_value=91.0,
    )
    extended, extended_lengths, extended_mask = _mixed_length_batch(
        short,
        long,
        padded_time=40,
        padded_value=-73.0,
    )
    compact_model = model_type(dropout=0.0).train()
    extended_model = copy.deepcopy(compact_model).train()

    compact_logits = compact_model(
        compact,
        feature_lengths=compact_lengths,
        padding_mask=compact_mask,
    )
    extended_logits = extended_model(
        extended,
        feature_lengths=extended_lengths,
        padding_mask=extended_mask,
    )
    torch.testing.assert_close(compact_logits, extended_logits, rtol=1.0e-5, atol=1.0e-6)


def test_masked_intermediate_sequences_zero_padded_positions() -> None:
    values = torch.randn(2, 1, 257, 28)
    lengths = torch.tensor([13, 28])
    padding_mask = torch.arange(28).unsqueeze(0) >= lengths.unsqueeze(1)

    bilstm = CNNBiLSTM().eval()
    transformer = ShortPrefixTransformer().eval()
    with torch.inference_mode():
        recurrent_states = bilstm.encode_sequence(values, feature_lengths=lengths)
        transformer_states = transformer.encode_tokens(values, padding_mask=padding_mask)

    assert not recurrent_states[0, :13].eq(0).all()
    assert recurrent_states[0, 13:].eq(0).all()
    assert not transformer_states[0, :13].eq(0).all()
    assert transformer_states[0, 13:].eq(0).all()


def test_lengths_and_padding_mask_must_be_consistent() -> None:
    values = torch.randn(2, 1, 257, 28)
    lengths = torch.tensor([13, 28])
    inconsistent_mask = torch.zeros(2, 28, dtype=torch.bool)
    with pytest.raises(ValueError, match="consistent with feature_lengths"):
        CNNBiLSTM()(
            values,
            feature_lengths=lengths,
            padding_mask=inconsistent_mask,
        )


def test_padding_mask_must_describe_right_padding() -> None:
    values = torch.randn(2, 1, 257, 28)
    interior_gap = torch.zeros(2, 28, dtype=torch.bool)
    interior_gap[0, 6] = True
    with pytest.raises(ValueError, match="right-padded"):
        ShortPrefixTransformer()(values, padding_mask=interior_gap)


def test_small_cnn_checks_each_items_valid_length() -> None:
    values = torch.randn(2, 1, 257, 28)
    with pytest.raises(ValueError, match="at least 8 valid time frames"):
        SmallCNN()(values, feature_lengths=torch.tensor([7, 28]))


def test_model_factory_supports_canonical_names_and_manuscript_aliases() -> None:
    assert available_model_names() == (
        "small_cnn",
        "cnn_bilstm",
        "cnn_gru",
        "efficientnet_lite0",
        "transformer",
        "wavlm_reference",
    )
    assert isinstance(create_model("small-cnn"), SmallCNN)
    assert isinstance(create_model("cnn-lstm"), CNNBiLSTM)
    assert isinstance(create_model("cnn-gru"), CNNGRU)
    assert isinstance(create_model("efficientnet-lite0"), EfficientNetLite0)
    assert isinstance(create_model("transformer"), ShortPrefixTransformer)
    with pytest.raises(ValueError, match="unknown model"):
        create_model("efficientnet")
