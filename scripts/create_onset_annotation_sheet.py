#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timeentropy.analysis.onset_validation import (
    DEFAULT_SELECTION_SEED,
    create_annotation_sample,
    write_annotation_sheet,
)
from timeentropy.config import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic speaker-balanced onset annotation sheet."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/torgo_manifest_v1.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/onset_annotation_sheet_v1.csv"),
    )
    parser.add_argument("--schema-output", type=Path)
    parser.add_argument("--per-speaker", type=int, default=20)
    parser.add_argument("--expected-speakers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(project_path(args.manifest))
    sheet = create_annotation_sample(
        manifest,
        per_speaker=args.per_speaker,
        expected_speakers=args.expected_speakers,
        seed=args.seed,
    )
    csv_path, schema_path = write_annotation_sheet(
        sheet,
        project_path(args.output),
        None if args.schema_output is None else project_path(args.schema_output),
    )
    print(f"Wrote {len(sheet)} blank annotation rows to {csv_path}")
    print(f"Wrote annotation schema to {schema_path}")
    print(sheet.groupby(["speaker_id", "prompt_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
