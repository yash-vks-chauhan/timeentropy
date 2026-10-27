#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from timeentropy.models import available_model_names
from timeentropy.training.fold import run_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one frozen TORGO fold and export its prefix ledger."
    )
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(name for name in available_model_names() if name != "wavlm_reference"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rebuild_v1.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    result = run_fold(
        args.config,
        fold=args.fold,
        model_id=args.model,
        seed=args.seed,
        device_preference=args.device,
        output_root=args.output_root,
        num_workers=args.num_workers,
    )
    print(f"Run: {result.run_id}")
    print(f"Best checkpoint NLL: {result.best_checkpoint_nll:.6f}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Prefix ledger: {result.ledger_path}")
    print(f"Summary: {result.summary_path}")


if __name__ == "__main__":
    main()
