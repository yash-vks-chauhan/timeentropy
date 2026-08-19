from __future__ import annotations

import argparse
import bz2
import csv
import shutil
import subprocess
import tarfile
import wave
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class ArchiveSpec:
    group: str
    url: str
    size_bytes: int
    etag: str


ARCHIVES = {
    "F": ArchiveSpec(
        "F",
        "https://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2",
        1_139_746_435,
        "43ef2683-5200ec2b7a700",
    ),
    "FC": ArchiveSpec(
        "FC",
        "https://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2",
        2_532_029_976,
        "96ebb618-5200f0818e200",
    ),
    "M": ArchiveSpec(
        "M",
        "https://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2",
        2_506_634_222,
        "956833ee-5200f55954f00",
    ),
    "MC": ArchiveSpec(
        "MC",
        "https://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2",
        3_398_783_713,
        "ca9552e1-5200f9b521780",
    ),
}


def speaker_group(speaker_id: str) -> str:
    if speaker_id.startswith("FC"):
        return "FC"
    if speaker_id.startswith("MC"):
        return "MC"
    if speaker_id.startswith("F"):
        return "F"
    if speaker_id.startswith("M"):
        return "M"
    raise ValueError(f"Unknown TORGO speaker prefix: {speaker_id}")


def load_targets(manifest_path: Path, group: str) -> dict[str, dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    targets: dict[str, dict[str, str]] = {}
    for row in rows:
        if speaker_group(row["speaker_id"]) != group:
            continue
        rel_path = PurePosixPath(row["rel_path"]).as_posix()
        targets[rel_path] = row
    if not targets:
        raise ValueError(f"No manifest targets belong to archive group {group}")
    return targets


def download_archive(spec: ArchiveSpec, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("aria2c"):
        command = [
            "aria2c",
            "--continue=true",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=1M",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--console-log-level=warn",
            "--summary-interval=0",
            f"--dir={archive_path.parent}",
            f"--out={archive_path.name}",
            spec.url,
        ]
    else:
        command = [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--continue-at",
            "-",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--output",
            str(archive_path),
            spec.url,
        ]
    subprocess.run(command, check=True)
    size = archive_path.stat().st_size
    if size != spec.size_bytes:
        raise RuntimeError(f"{archive_path} has {size} bytes; expected {spec.size_bytes}")
    # Read the complete bzip2 stream once before extraction so corruption is caught early.
    with bz2.open(archive_path, "rb") as handle:
        while handle.read(8 * 1024 * 1024):
            pass


def _member_suffix(member_name: str) -> str | None:
    parts = PurePosixPath(member_name).parts
    for index, part in enumerate(parts):
        if part == "wav_headMic" and index >= 2 and index + 1 < len(parts):
            return PurePosixPath(*parts[index - 2 : index + 2]).as_posix()
    return None


def extract_selected(
    archive_path: Path,
    raw_root: Path,
    targets: dict[str, dict[str, str]],
) -> set[str]:
    raw_root.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = _member_suffix(member.name)
            if suffix is None or suffix not in targets:
                continue
            destination = raw_root / suffix
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member {member.name}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.add(suffix)
    missing = set(targets) - extracted
    if missing:
        preview = sorted(missing)[:10]
        raise RuntimeError(f"Archive lacked {len(missing)} indexed WAVs; first: {preview}")
    return extracted


def validate_wavs(raw_root: Path, targets: dict[str, dict[str, str]]) -> None:
    failures: list[str] = []
    for rel_path, row in targets.items():
        path = raw_root / rel_path
        try:
            with wave.open(str(path), "rb") as handle:
                sample_rate = handle.getframerate()
                num_frames = handle.getnframes()
            if sample_rate != int(row["sample_rate"]):
                failures.append(f"{rel_path}: sample rate {sample_rate}")
            if num_frames != int(row["num_frames"]):
                failures.append(f"{rel_path}: frames {num_frames}")
        except (FileNotFoundError, wave.Error) as exc:
            failures.append(f"{rel_path}: {exc}")
    if failures:
        raise RuntimeError("WAV validation failed:\n" + "\n".join(failures[:20]))


def recover_group(
    group: str,
    manifest_path: Path,
    archive_root: Path,
    raw_root: Path,
    delete_archive_after_verify: bool,
) -> None:
    spec = ARCHIVES[group]
    targets = load_targets(manifest_path, group)
    already_present = all((raw_root / rel_path).is_file() for rel_path in targets)
    if already_present:
        validate_wavs(raw_root, targets)
        print(f"{group}: {len(targets)} WAVs already present and valid")
        return

    archive_path = archive_root / f"{group}.tar.bz2"
    print(f"{group}: downloading {spec.size_bytes} bytes from {spec.url}")
    download_archive(spec, archive_path)
    print(f"{group}: extracting {len(targets)} indexed headMic WAVs")
    extract_selected(archive_path, raw_root, targets)
    validate_wavs(raw_root, targets)
    print(f"{group}: validated {len(targets)} WAVs")
    if delete_archive_after_verify:
        archive_path.unlink()
        print(f"{group}: removed verified temporary archive {archive_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume TORGO downloads and selectively extract indexed headMic WAVs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/torgo_manifest_v1.csv"),
    )
    parser.add_argument("--archive-root", type=Path, default=Path("data/archives"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/torgo"))
    parser.add_argument("--groups", nargs="+", choices=tuple(ARCHIVES), default=list(ARCHIVES))
    parser.add_argument("--delete-archive-after-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for group in args.groups:
        recover_group(
            group,
            args.manifest,
            args.archive_root,
            args.raw_root,
            args.delete_archive_after_verify,
        )


if __name__ == "__main__":
    main()
