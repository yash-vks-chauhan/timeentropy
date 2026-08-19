"""Reproducible fold training, checkpoint selection, and raw-logit export."""

from timeentropy.training.engine import (
    EpochRecord,
    TrainingConfig,
    TrainingResult,
    call_model_for_batch,
    evaluate_nll,
    seed_everything,
    select_device,
    train_with_early_stopping,
)
from timeentropy.training.fold import FoldRunResult, create_model, run_fold
from timeentropy.training.inference import evaluate_all_prefixes, export_prefix_ledger
from timeentropy.training.ssl_fold import create_ssl_reference_model, run_ssl_fold

__all__ = [
    "EpochRecord",
    "FoldRunResult",
    "TrainingConfig",
    "TrainingResult",
    "call_model_for_batch",
    "create_model",
    "create_ssl_reference_model",
    "evaluate_all_prefixes",
    "evaluate_nll",
    "export_prefix_ledger",
    "run_fold",
    "run_ssl_fold",
    "seed_everything",
    "select_device",
    "train_with_early_stopping",
]
