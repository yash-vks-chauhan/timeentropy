from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

_LOG_TWO = math.log(2.0)


def _validated_binary_probabilities(probabilities: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("Binary probabilities must have a final dimension of size two")
    if not np.all(np.isfinite(values)):
        raise ValueError("Probabilities must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Probabilities must lie in [0, 1]")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("Binary probabilities must sum to one")
    return values


def normalized_binary_entropy(
    probabilities: ArrayLike,
    probability_floor: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Return Shannon entropy normalized by ``log(2)``.

    The final input dimension is ordered as control and dysarthric probability.  The
    normalization gives zero for a deterministic posterior and one for ``[0.5, 0.5]``.
    Clipping is used only for the logarithm; input probabilities are otherwise validated
    and are never silently renormalized.
    """

    if not 0.0 < probability_floor < 0.5:
        raise ValueError("probability_floor must lie strictly between zero and 0.5")
    values = _validated_binary_probabilities(probabilities)
    safe = np.clip(values, probability_floor, 1.0)
    entropy = -np.sum(values * np.log(safe), axis=-1) / _LOG_TWO
    return np.clip(entropy, 0.0, 1.0)


def maximum_confidence_threshold_for_entropy(theta: float) -> float:
    """Map a normalized binary-entropy threshold to maximum softmax confidence.

    On ``q in [0.5, 1]``, binary entropy is strictly decreasing.  Consequently,
    ``H(p) <= theta`` is exactly equivalent to ``max(p) >= returned_threshold``.
    """

    if not math.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be finite and lie in [0, 1]")
    if theta == 1.0:
        return 0.5
    if theta == 0.0:
        return 1.0

    lower = 0.5
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        value = float(normalized_binary_entropy([midpoint, 1.0 - midpoint]))
        if value > theta:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0
