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


def test_manifest_channel_selection_is_not_hard_coded(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "sample_id": ["head", "array"],
            "path": ["/tmp/head.wav", "/tmp/array.wav"],
            "rel_path": ["S1/Session1/wav_headMic/a.wav", "S1/Session1/wav_arrayMic/a.wav"],
            "speaker_id": ["S1", "S1"],
            "gender": ["F", "F"],
            "label": ["control", "control"],
            "label_id": [0, 0],
            "session": ["Session1", "Session1"],
            "mic": ["headMic", "arrayMic"],
            "utterance_id": ["a", "a"],
            "prompt_text": ["hello", "hello"],
            "usable_prompt": [True, True],
            "duration_sec": [1.0, 1.0],
            "sample_rate": [16_000, 44_100],
            "num_frames": [16_000, 44_100],
        }
    )
    source_path = tmp_path / "source.csv"
    source.to_csv(source_path, index=False)

    array = build_manifest(source_path, mic="arrayMic", expected_counts=None)
    assert array["sample_id"].tolist() == ["array"]
    assert array["mic"].tolist() == ["arrayMic"]
