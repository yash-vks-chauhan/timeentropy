#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from timeentropy.training.ssl_fold import run_ssl_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen-by-default WavLM reference on one TORGO speaker fold "
            "and export its calibration-ready four-prefix ledger."
        )
    )
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--config", type=Path, default=Path("configs/rebuild_v1.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override training.batch_size for memory-constrained WavLM runs.",
    )
    parser.add_argument(
        "--unfreeze-top-n",
        type=int,
        help="Fine-tune the final N WavLM encoder layers; default is frozen backbone.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Authorize Hugging Face network retrieval when the checkpoint is not cached. "
            "Without this flag, loading is strictly local-files-only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.unfreeze_top_n is not None and args.unfreeze_top_n < 0:
        raise SystemExit("--unfreeze-top-n must be non-negative")
    result = run_ssl_fold(
        args.config,
        fold=args.fold,
        seed=args.seed,
        device_preference=args.device,
        output_root=args.output_root,
        num_workers=args.num_workers,
        allow_download=args.allow_download,
        unfreeze_top_n=args.unfreeze_top_n,
        batch_size=args.batch_size,
    )
    print(f"Run: {result.run_id}")
    print(f"Best checkpoint NLL: {result.best_checkpoint_nll:.6f}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Prefix ledger: {result.ledger_path}")
    print(f"Summary: {result.summary_path}")


if __name__ == "__main__":
    main()
