from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from timeentropy.reporting.bootstrap import paired_cluster_bootstrap_differences

_MATCHED_CONDITION_COLUMNS = ("seed", "onset_variant", "pre_roll_ms", "policy_id")
_MATCHED_POLICY_COLUMNS = ("model_id", "seed", "onset_variant", "pre_roll_ms")


def _derived_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _paired_frames(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["sample_id", "speaker_id", "fold", "label_id"]
    left_sorted = left.sort_values("sample_id").reset_index(drop=True)
    right_sorted = right.sort_values("sample_id").reset_index(drop=True)
    if not left_sorted[key].equals(right_sorted[key]):
        raise ValueError("Paired systems must have identical sample, speaker, fold, and label rows")
    return left_sorted, right_sorted


def paired_model_comparisons(
    enriched: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 20_260_819,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare model pairs under identical seed, onset, pre-roll, and policy conditions."""

    comparison_rows: list[pd.DataFrame] = []
    mcnemar_rows: list[dict[str, object]] = []
    grouped = enriched.groupby(list(_MATCHED_CONDITION_COLUMNS), dropna=False, sort=True)
    for condition, condition_group in grouped:
        systems = [group for _, group in condition_group.groupby("system_id", sort=True)]
        for left, right in combinations(systems, 2):
            left_model = str(left.iloc[0]["model_id"])
            right_model = str(right.iloc[0]["model_id"])
            if left_model == right_model:
                continue
            left, right = _paired_frames(left, right)
            left_id = str(left.iloc[0]["system_id"])
            right_id = str(right.iloc[0]["system_id"])
            pair_seed = _derived_seed(seed, f"{left_id}|{right_id}")
            differences = paired_cluster_bootstrap_differences(
                left,
                right,
                replicates=replicates,
                seed=pair_seed,
                confidence=confidence,
            )
            differences.insert(0, "right_system_id", right_id)
            differences.insert(0, "left_system_id", left_id)
            for column, value in zip(_MATCHED_CONDITION_COLUMNS, condition, strict=True):
                differences[column] = value
            comparison_rows.append(differences)

            left_correct = left["predicted_label"].to_numpy() == left["label_id"].to_numpy()
            right_correct = right["predicted_label"].to_numpy() == right["label_id"].to_numpy()
            left_only = int(np.sum(left_correct & ~right_correct))
            right_only = int(np.sum(~left_correct & right_correct))
            discordant = left_only + right_only
            p_value = (
                float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            mcnemar_rows.append(
                {
                    "left_system_id": left_id,
                    "right_system_id": right_id,
                    **dict(zip(_MATCHED_CONDITION_COLUMNS, condition, strict=True)),
                    "left_correct_right_wrong": left_only,
                    "left_wrong_right_correct": right_only,
                    "discordant_pairs": discordant,
                    "exact_two_sided_p_value": p_value,
                    "method": "exact_mcnemar_binomial",
                    "scope_warning": (
                        "Descriptive utterance-level sensitivity analysis; speaker-cluster "
                        "bootstrap is the primary inference"
                    ),
                }
            )

    comparisons = (
        pd.concat(comparison_rows, ignore_index=True) if comparison_rows else pd.DataFrame()
    )
    return comparisons, pd.DataFrame(mcnemar_rows)


def _mcnemar_record(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object]:
    """Exact McNemar counts and p-value for two paired prediction sets."""

    left_correct = left["predicted_label"].to_numpy() == left["label_id"].to_numpy()
    right_correct = right["predicted_label"].to_numpy() == right["label_id"].to_numpy()
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    p_value = (
        float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue) if discordant else 1.0
    )
    return {
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": p_value,
        "method": "exact_mcnemar_binomial",
        "scope_warning": (
            "Descriptive utterance-level sensitivity analysis; speaker-cluster bootstrap is "
            "the primary inference. A non-significant result is not evidence of equivalence."
        ),
    }


def paired_policy_comparisons(
    enriched: pd.DataFrame,
    *,
    baseline_policy_prefix: str = "fixed_",
    replicates: int = 10_000,
    seed: int = 20_260_819,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each adaptive policy against every fixed-duration baseline of the same model.

    ``paired_model_comparisons`` holds ``policy_id`` fixed and varies the model, so it never
    answers whether the adaptive policy differs from the fixed-duration baseline it is sold
    against. Both reviewers asked for exactly that comparison, on identical utterances.
    """

    if enriched.empty:
        return pd.DataFrame(), pd.DataFrame()

    comparison_rows: list[pd.DataFrame] = []
    mcnemar_rows: list[dict[str, object]] = []
    grouped = enriched.groupby(list(_MATCHED_POLICY_COLUMNS), dropna=False, sort=True)
    for condition, condition_group in grouped:
        policies = {
            str(policy_id): group
            for policy_id, group in condition_group.groupby("policy_id", sort=True)
        }
        baselines = sorted(p for p in policies if p.startswith(baseline_policy_prefix))
        adaptives = sorted(p for p in policies if not p.startswith(baseline_policy_prefix))
        for adaptive_id in adaptives:
            for baseline_id in baselines:
                left, right = _paired_frames(policies[adaptive_id], policies[baseline_id])
                pair_seed = _derived_seed(seed, f"{condition}|{adaptive_id}|{baseline_id}")
                differences = paired_cluster_bootstrap_differences(
                    left,
                    right,
                    replicates=replicates,
                    seed=pair_seed,
                    confidence=confidence,
                )
                differences.insert(0, "baseline_policy_id", baseline_id)
                differences.insert(0, "adaptive_policy_id", adaptive_id)
                for column, value in zip(_MATCHED_POLICY_COLUMNS, condition, strict=True):
                    differences[column] = value
                comparison_rows.append(differences)

                mcnemar_rows.append(
                    {
                        "adaptive_policy_id": adaptive_id,
                        "baseline_policy_id": baseline_id,
                        **dict(zip(_MATCHED_POLICY_COLUMNS, condition, strict=True)),
                        **_mcnemar_record(left, right),
                    }
                )

    comparisons = (
        pd.concat(comparison_rows, ignore_index=True) if comparison_rows else pd.DataFrame()
    )
    return comparisons, pd.DataFrame(mcnemar_rows)
