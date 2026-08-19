from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal

import pandas as pd

ValidationPartition = Literal["checkpoint", "calibration", "policy"]


def validation_partition(sample_id: str) -> ValidationPartition:
    """Map a sample ID to the frozen 50/25/25 validation subpartition.

    The two least-significant bits of the complete SHA-256 digest select one of
    four equally sized regions of the hash space. Two regions are assigned to
    checkpoint stopping and one each to calibration and policy tuning.
    """

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest, byteorder="big", signed=False) % 4
    if bucket < 2:
        return "checkpoint"
    if bucket == 2:
        return "calibration"
    return "policy"


def assign_validation_partitions(
    sample_ids: Iterable[str],
) -> tuple[ValidationPartition, ...]:
    """Assign validation partitions without depending on row order."""

    return tuple(validation_partition(sample_id) for sample_id in sample_ids)


def attach_validation_partitions(
    manifest: pd.DataFrame,
    sample_id_column: str = "sample_id",
    output_column: str = "validation_partition",
) -> pd.DataFrame:
    """Return a copy of a validation manifest with deterministic subpartitions."""

    if sample_id_column not in manifest.columns:
        raise ValueError(f"Manifest is missing {sample_id_column!r}")
    result = manifest.copy()
    result[output_column] = assign_validation_partitions(
        result[sample_id_column].astype(str).tolist()
    )
    return result
