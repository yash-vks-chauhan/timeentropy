from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from timeentropy.config import load_yaml


@dataclass(frozen=True)
class Fold:
    fold: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def load_folds(path: str | Path) -> tuple[Fold, ...]:
    raw = load_yaml(path)
    universe = tuple(str(item) for item in raw["speaker_universe"])
    universe_set = set(universe)
    folds: list[Fold] = []

    for item in raw["folds"]:
        validation = tuple(str(value) for value in item["validation"])
        test = tuple(str(value) for value in item["test"])
        if set(validation) & set(test):
            raise ValueError(f"Fold {item['fold']} has validation/test speaker overlap")
        if not set(validation).issubset(universe_set) or not set(test).issubset(universe_set):
            raise ValueError(f"Fold {item['fold']} contains an unknown speaker")
        train = tuple(value for value in universe if value not in set(validation) | set(test))
        folds.append(
            Fold(
                fold=int(item["fold"]),
                train=train,
                validation=validation,
                test=test,
            )
        )

    tested = [speaker for fold in folds for speaker in fold.test]
    if sorted(tested) != sorted(universe):
        raise ValueError("Every speaker must appear in exactly one test fold")
    return tuple(folds)


def attach_fold_roles(manifest: pd.DataFrame, folds: tuple[Fold, ...]) -> pd.DataFrame:
    """Return one row per sample and fold, with train/validation/test role."""

    records: list[pd.DataFrame] = []
    speakers = set(manifest["speaker_id"].astype(str))
    expected = set(folds[0].train) | set(folds[0].validation) | set(folds[0].test)
    if speakers != expected:
        raise ValueError(f"Manifest speakers differ from fold universe: {speakers ^ expected}")

    for fold in folds:
        roles = {speaker: "train" for speaker in fold.train}
        roles.update({speaker: "validation" for speaker in fold.validation})
        roles.update({speaker: "test" for speaker in fold.test})
        frame = manifest[["sample_id", "speaker_id", "label", "label_id"]].copy()
        frame.insert(0, "fold", fold.fold)
        frame["role"] = frame["speaker_id"].map(roles)
        records.append(frame)

    result = pd.concat(records, ignore_index=True)
    if result["role"].isna().any():
        raise ValueError("At least one sample did not receive a fold role")
    return result
