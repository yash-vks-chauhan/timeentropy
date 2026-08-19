from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def class_balanced_weights(labels: Sequence[int]) -> torch.Tensor:
    """Return inverse-frequency sample weights with equal mass per class."""

    values = [int(label) for label in labels]
    if not values:
        raise ValueError("At least one label is required")
    counts = Counter(values)
    if len(counts) < 2:
        raise ValueError("Class balancing requires at least two observed classes")
    return torch.tensor([1.0 / counts[label] for label in values], dtype=torch.double)


def make_class_balanced_sampler(
    labels: Sequence[int],
    num_samples: int | None = None,
    seed: int = 20260819,
) -> WeightedRandomSampler:
    """Build the frozen inverse-frequency, with-replacement training sampler."""

    weights = class_balanced_weights(labels)
    draws = len(weights) if num_samples is None else int(num_samples)
    if draws <= 0:
        raise ValueError("num_samples must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=draws,
        replacement=True,
        generator=generator,
    )
