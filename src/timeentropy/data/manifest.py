from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from timeentropy.config import PROJECT_ROOT, project_path
from timeentropy.provenance import sha256_file
from timeentropy.splits import attach_fold_roles, load_folds

EXPECTED_COUNTS = {
    "utterances": 8131,
    "speakers": 15,
    "control": 5197,
    "dysarthric": 2934,
}


def _observed_counts(selected: pd.DataFrame) -> dict[str, int]:
    return {
        "utterances": len(selected),
        "speakers": selected["speaker_id"].nunique(),
        "control": int(selected["label"].eq("control").sum()),
        "dysarthric": int(selected["label"].eq("dysarthric").sum()),
    }


def build_manifest(
    source_index: str | Path,
    *,
    mic: str = "headMic",
    expected_counts: dict[str, int] | None = EXPECTED_COUNTS,
) -> pd.DataFrame:
    """Select one microphone channel without changing the locked head-mic cohort.

    Non-head-mic manifests deliberately do not inherit ``EXPECTED_COUNTS``: TORGO
    sessions are not one-to-one across channels, so their observed cohort must be
    recorded rather than forced to the 8,131 head-mic utterances.
    """

    if not mic.strip():
        raise ValueError("mic must be a non-empty channel identifier")
    source = pd.read_csv(
        source_index,
        dtype={
            "sample_id": "string",
            "path": "string",
            "rel_path": "string",
            "speaker_id": "string",
            "gender": "string",
            "label": "string",
            "session": "string",
            "mic": "string",
            "utterance_id": "string",
            "prompt_text": "string",
        },
    )
    required = {
        "sample_id",
        "path",
        "rel_path",
        "speaker_id",
        "gender",
        "label",
        "label_id",
        "session",
        "mic",
        "utterance_id",
        "prompt_text",
        "usable_prompt",
        "duration_sec",
        "sample_rate",
        "num_frames",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Recovered index is missing columns: {sorted(missing)}")

    usable = source["usable_prompt"].astype(str).str.lower().eq("true")
    selected = source.loc[usable & source["duration_sec"].ge(0.05) & source["mic"].eq(mic)].copy()
    selected = selected.sort_values("sample_id").reset_index(drop=True)

    observed = _observed_counts(selected)
    if expected_counts is not None and observed != expected_counts:
        raise ValueError(f"Recovered manifest counts differ: {observed} != {expected_counts}")
    if selected["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    return selected


def write_manifests(
    source_index: str | Path,
    manifest_output: str | Path,
    fold_output: str | Path,
    folds_path: str | Path,
    raw_root: str | Path = "data/raw/torgo",
    mic: str = "headMic",
) -> tuple[Path, Path]:
    expected_counts = EXPECTED_COUNTS if mic == "headMic" else None
    manifest = build_manifest(source_index, mic=mic, expected_counts=expected_counts)
    manifest.insert(2, "recovered_path", manifest["path"])
    manifest["path"] = manifest["rel_path"].map(
        lambda value: str(Path(raw_root) / str(value))
    )
    folds = load_folds(folds_path)
    fold_manifest = attach_fold_roles(manifest, folds)

    manifest_path = project_path(manifest_output)
    fold_path = project_path(fold_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    fold_manifest.to_csv(fold_path, index=False)
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_index": str(Path(source_index).resolve()),
        "source_index_sha256": sha256_file(source_index),
        "manifest_sha256": sha256_file(manifest_path),
        "fold_manifest_sha256": sha256_file(fold_path),
        "selection": {
            "mic": mic,
            "usable_prompt": True,
            "minimum_duration_sec": 0.05,
        },
        "counts": _observed_counts(manifest),
    }
    metadata_path = manifest_path.with_suffix(".meta.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path, fold_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        "--manifest-output",
        dest="manifest_output",
        type=Path,
        default=Path("data/metadata/torgo_manifest_v1.csv"),
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=PROJECT_ROOT / "configs/folds_v1.yaml",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/torgo"))
    parser.add_argument(
        "--mic",
        default="headMic",
        help="TORGO microphone channel to select (for example headMic or arrayMic).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_output = args.fold_output
    if fold_output is None:
        if args.mic == "headMic":
            fold_output = Path("data/metadata/torgo_folds_v1.csv")
        else:
            name = args.manifest_output.name.replace("manifest", "folds", 1)
            fold_output = args.manifest_output.with_name(name)
    manifest, folds = write_manifests(
        args.source_index,
        args.manifest_output,
        fold_output,
        args.folds,
        args.raw_root,
        args.mic,
    )
    print(f"Wrote {manifest}")
    print(f"Wrote {folds}")


if __name__ == "__main__":
    main()
