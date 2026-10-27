"""Build a verified array-microphone source index matched to the head-mic cohort.

The official TORGO archives use the same speaker/session/utterance filename for
both channels.  This script preserves the frozen head-mic ``sample_id`` so
cross-channel decisions can be paired exactly, while reading each array WAV's
actual header rather than assuming a sample rate or duration.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-index", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/torgo"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.head_index, dtype={"sample_id": "string", "rel_path": "string"})
    required = {"sample_id", "rel_path", "mic", "duration_sec", "sample_rate", "num_frames"}
    missing = required - set(source.columns)
    if missing:
        raise SystemExit(f"Head index is missing columns: {sorted(missing)}")
    head = source.loc[source["mic"].eq("headMic")].copy()
    rows: list[dict[str, object]] = []
    for row in head.itertuples(index=False):
        rel_path = str(row.rel_path).replace("wav_headMic", "wav_arrayMic")
        path = args.raw_root / rel_path
        if not path.is_file():
            continue
        try:
            with wave.open(str(path), "rb") as handle:
                sample_rate = handle.getframerate()
                num_frames = handle.getnframes()
                channels = handle.getnchannels()
        except wave.Error as error:
            raise SystemExit(f"Invalid array WAV {path}: {error}") from error
        if channels < 1:
            raise SystemExit(f"Array WAV has no channels: {path}")
        value = row._asdict()
        value.update(
            {
                "path": str(path.resolve()),
                "rel_path": rel_path,
                "mic": "arrayMic",
                "duration_sec": num_frames / sample_rate,
                "sample_rate": sample_rate,
                "num_frames": num_frames,
            }
        )
        rows.append(value)
    result = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    if result.empty:
        raise SystemExit("No array WAVs matched the head-mic source index")
    if result["sample_id"].duplicated().any():
        raise SystemExit("Array source index has duplicate sample_id values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(result)} matched array-channel WAVs")


if __name__ == "__main__":
    main()
