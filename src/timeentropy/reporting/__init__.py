"""Auditable tables and speaker-clustered statistics from saved OOF ledgers."""

from timeentropy.reporting.bootstrap import clustered_speaker_bootstrap_ci
from timeentropy.reporting.comparison import paired_model_comparisons
from timeentropy.reporting.pipeline import generate_report_bundle
from timeentropy.reporting.summaries import (
    SYSTEM_COLUMNS,
    SummaryTables,
    build_calibration_tables,
    build_summary_tables,
    enrich_decision_ledger,
)

__all__ = [
    "SYSTEM_COLUMNS",
    "SummaryTables",
    "build_calibration_tables",
    "build_summary_tables",
    "clustered_speaker_bootstrap_ci",
    "enrich_decision_ledger",
    "generate_report_bundle",
    "paired_model_comparisons",
]
