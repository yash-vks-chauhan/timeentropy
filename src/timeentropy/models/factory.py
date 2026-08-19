"""Stable name-to-model construction for training and evaluation entry points."""

from __future__ import annotations

from torch import nn

from .cnn_bilstm import CNNBiLSTM
from .common import DEFAULT_DROPOUT
from .small_cnn import SmallCNN
from .transformer import ShortPrefixTransformer

_MODEL_TYPES: dict[str, type[nn.Module]] = {
    "small_cnn": SmallCNN,
    "cnn_bilstm": CNNBiLSTM,
    "transformer": ShortPrefixTransformer,
}
_SSL_MODEL_NAME = "wavlm_reference"
_ALIASES = {
    "small-cnn": "small_cnn",
    "cnn_lstm": "cnn_bilstm",
    "cnn-lstm": "cnn_bilstm",
    "cnn-bilstm": "cnn_bilstm",
    "short_prefix_transformer": "transformer",
    "short-prefix-transformer": "transformer",
    "ssl_reference": _SSL_MODEL_NAME,
    "wavlm": _SSL_MODEL_NAME,
    "wavlm-base-plus": _SSL_MODEL_NAME,
}


def available_model_names() -> tuple[str, ...]:
    """Return canonical model identifiers in deterministic order."""

    return (*_MODEL_TYPES, _SSL_MODEL_NAME)


def create_model(
    name: str,
    *,
    dropout: float = DEFAULT_DROPOUT,
    **model_kwargs: object,
) -> nn.Module:
    """Instantiate a model without importing optional dependencies prematurely.

    WavLM requires either an injected ``backbone`` or the explicit
    ``load_pretrained=True`` flag, preventing accidental downloads.
    """

    normalized = name.strip().lower()
    canonical = _ALIASES.get(normalized, normalized)
    if canonical == _SSL_MODEL_NAME:
        from .ssl_reference import WavLMReference

        load_pretrained = model_kwargs.pop("load_pretrained", False)
        if load_pretrained not in (True, False):
            raise TypeError("load_pretrained must be a bool")
        if load_pretrained:
            if "backbone" in model_kwargs:
                raise ValueError("cannot combine backbone injection with load_pretrained=True")
            return WavLMReference.from_pretrained(dropout=dropout, **model_kwargs)
        if "backbone" not in model_kwargs:
            raise ValueError(
                "wavlm_reference requires an injected backbone or explicit load_pretrained=True"
            )
        return WavLMReference(dropout=dropout, **model_kwargs)

    try:
        model_type = _MODEL_TYPES[canonical]
    except KeyError as error:
        choices = ", ".join(available_model_names())
        raise ValueError(f"unknown model {name!r}; expected one of: {choices}") from error
    return model_type(dropout=dropout)
