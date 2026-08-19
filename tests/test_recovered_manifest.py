from pathlib import Path

import pandas as pd

from timeentropy.data.manifest import EXPECTED_COUNTS, build_manifest
from timeentropy.splits import attach_fold_roles, load_folds

RECOVERED_INDEX = Path(
    "/Users/yashchauhan/timeentropy_runtime/artifacts/metadata/torgo_index.csv"
)


def test_recovered_index_reproduces_locked_cohort_when_available() -> None:
    if not RECOVERED_INDEX.exists():
        return
    manifest = build_manifest(RECOVERED_INDEX)
    assert len(manifest) == EXPECTED_COUNTS["utterances"]
    assert manifest["sample_id"].nunique() == EXPECTED_COUNTS["utterances"]
    assert manifest["speaker_id"].nunique() == EXPECTED_COUNTS["speakers"]
    assert int(manifest["label"].eq("control").sum()) == EXPECTED_COUNTS["control"]
    assert int(manifest["label"].eq("dysarthric").sum()) == EXPECTED_COUNTS["dysarthric"]


def test_recovered_fold_counts_match_documented_values() -> None:
    manifest = pd.read_csv("data/metadata/torgo_manifest_v1.csv")
    folded = attach_fold_roles(manifest, load_folds("configs/folds_v1.yaml"))
    expected = {
        0: {"train": (3094, 1408), "validation": (1290, 650), "test": (813, 876)},
        1: {"train": (2799, 1735), "validation": (1108, 549), "test": (1290, 650)},
        2: {"train": (3095, 2091), "validation": (994, 294), "test": (1108, 549)},
        3: {"train": (3211, 2075), "validation": (992, 565), "test": (994, 294)},
        4: {"train": (3392, 1493), "validation": (813, 876), "test": (992, 565)},
    }
    for fold, roles in expected.items():
        subset = folded.loc[folded["fold"].eq(fold)]
        for role, (control, dysarthric) in roles.items():
            selected = subset.loc[subset["role"].eq(role)]
            assert int(selected["label"].eq("control").sum()) == control
            assert int(selected["label"].eq("dysarthric").sum()) == dysarthric
