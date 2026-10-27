#!/usr/bin/env bash
# Evaluate the stages that finish after the primary sweep: the two Transformer replication
# seeds, their three-seed aggregate, and the two condition-matched onset-mode retrainings.
#
# Run this only once `scripts/run_experiments.py status` reports every selected job complete.
# Each stage reads frozen per-fold prefix ledgers and writes a fresh evaluation/report tree;
# none of them trains.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
FOLDS="0 1 2 3 4"

for seed in 20260820 20260821; do
  echo "== evaluating seed ${seed}"
  ledgers=()
  for fold in $FOLDS; do
    ledgers+=(--prefix-ledger \
      "artifacts/runs/reviewer_rebuild_v1-fold${fold}-transformer-seed${seed}-causal_noise_floor/prefix_predictions.parquet")
  done
  "$PYTHON" scripts/evaluate_ledgers.py \
    --config configs/rebuild_v1.yaml \
    "${ledgers[@]}" \
    --output-dir "artifacts/evaluation/transformer-seed${seed}"
  "$PYTHON" scripts/generate_reports.py \
    --config configs/rebuild_v1.yaml \
    --prefix-ledger "artifacts/evaluation/transformer-seed${seed}/calibrated_prefix_predictions.parquet" \
    --decision-ledger "artifacts/evaluation/transformer-seed${seed}/decisions.parquet" \
    --output-dir "artifacts/reports/transformer-seed${seed}"
done

echo "== aggregating the three seeds"
"$PYTHON" scripts/aggregate_seed_replications.py

for condition in causal_fixed_preroll30:causal_fixed offline_peak_preroll30:recovered_offline_peak; do
  analysis=${condition%%:*}
  variant=${condition##*:}
  echo "== evaluating ${analysis}"
  arguments=()
  for fold in $FOLDS; do
    run="artifacts/runs/reviewer_rebuild_v1-robustness-${analysis}-fold${fold}-transformer-seed20260819-${variant}"
    arguments+=(--prefix-ledger "${run}/prefix_predictions.parquet")
    arguments+=(--run-summary "${run}/run_summary.json")
  done
  "$PYTHON" scripts/run_robustness.py evaluate-retrained \
    --analysis-id "$analysis" \
    "${arguments[@]}" \
    --output-dir "artifacts/robustness/retrained/${analysis}"
done
