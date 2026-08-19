# Frozen robustness execution protocol

`configs/robustness_v1.yaml` is a prospective protocol, not a menu from which to pick
favorable test results. The robustness runner validates that both this file and its parent
experiment are marked `frozen_before_test_evaluation` before it reads a checkpoint or test
waveform.

## Two execution classes

Input-definition ablations and test-time stress tests answer different questions and are kept
in separate code paths.

| Analysis family | Conditions | Checkpoint rule | Calibration and policy rule |
|---|---:|---|---|
| Condition-matched input ablation | 4 x 5 folds | The primary condition may reuse the parent checkpoint. Each of the other three conditions requires a newly trained checkpoint using its own onset detector and pre-roll. | Fit temperature on that condition's calibration partition and select the policy on that condition's policy-tuning partition. |
| Evaluation-only perturbation | 11 x 5 folds | Reuse the primary `causal_fixed`, 30 ms pre-roll checkpoint. Retraining is forbidden. | Fit temperature once on the clean calibration partition and select `theta`/`epsilon` once on the clean policy-tuning partition. Apply those fixed values to every perturbed test condition. |

The four input conditions are the primary causal detector with 30 ms pre-roll, a zero-pre-roll
ablation, the recovered noncausal peak-relative detector, and the secondary causal noise-floor
detector. The 11 test-time conditions are the clean reference, white noise at 20/10/0 dB SNR,
and onset-anchor shifts of -50/-30/-10/0/+10/+30/+50 ms.

The plan records `requires_condition_matched_checkpoint`, `requires_new_training`,
`execution_stage`, and `selection_source` for every fold-condition job. The test-time CLI rejects
an input-ablation ID, and the retrained-ablation CLI rejects a test-time ID.

## Test-time perturbation definitions

White Gaussian noise is added after onset-aligned cropping and before the STFT. Its random stream
is derived from SHA-256 of the primary seed, sample ID, and SNR, so row order, batch size, and
worker count do not change the waveform. The sampled noise is centered to zero mean and scaled to
the requested crop-level RMS SNR. An all-zero crop remains zero because a finite relative SNR is
undefined; this is explicitly recorded as `noise_applied=false`. Synthetic white noise is only a
stress test and is not evidence of cross-corpus or real far-field robustness.

For onset sensitivity, `estimated_onset = primary_onset + offset`. The requested anchor is clipped
to `[0, waveform_length]`; the crop is then boundary-padded to preserve the exact pre-roll plus
post-onset input length. If the primary detector fails, the registered no-onset fallback remains at
sample zero and no artificial shifted onset is invented. Both the reference onset and requested
and clipped anchors are retained in the prefix ledger.

The zero-shift condition is independently recomputed. Its maximum absolute logit difference from
the saved clean ledger is recorded as an informational numerical audit. It is not used to accept,
reject, tune, or remove a condition, because hardware backends can introduce small floating-point
differences.

## Leakage controls

For a test-time run, the only fitted quantities are:

1. scalar temperature from rows with `split_role=calibration`; and
2. the complete-policy `theta` and `epsilon` from rows with
   `split_role=policy_tuning`, using the frozen grid and macro-F1 constraint.

Perturbed test logits are generated only after those values are fixed. Every metrics row and the
JSON summary record zero test rows used for fitting or selection. Conditions are emitted in frozen
config order, not ranked or filtered by test performance.

For a retrained input ablation, the standard evaluation pipeline enforces the same split boundary
within each condition. The robustness wrapper additionally requires one ledger and one run summary
per frozen fold, checks the exact condition-training-config hash, verifies the model/onset/pre-roll
identity, and hashes every supplied checkpoint and ledger.

## Commands

Validate and inspect the bounded 75-job plan without reading audio:

```bash
.venv/bin/python scripts/run_robustness.py plan \
  --output-dir artifacts/robustness/plan
```

Generate the three non-primary condition-matched configs. This command only writes configs and
does not launch training:

```bash
.venv/bin/python scripts/run_robustness.py prepare-retraining \
  --output-dir artifacts/robustness/retraining-configs
```

Each generated config must then be passed explicitly to `scripts/run_fold.py` for every fold. For
example:

```bash
.venv/bin/python scripts/run_fold.py \
  --config artifacts/robustness/retraining-configs/causal_fixed_preroll0.yaml \
  --fold 0 --model transformer --device auto
```

After all five condition-matched runs exist, verify their summaries and evaluate the ablation:

```bash
.venv/bin/python scripts/run_robustness.py evaluate-retrained \
  --analysis-id causal_fixed_preroll0 \
  --prefix-ledger artifacts/runs/FOLD0/prefix_predictions.parquet \
  --prefix-ledger artifacts/runs/FOLD1/prefix_predictions.parquet \
  --prefix-ledger artifacts/runs/FOLD2/prefix_predictions.parquet \
  --prefix-ledger artifacts/runs/FOLD3/prefix_predictions.parquet \
  --prefix-ledger artifacts/runs/FOLD4/prefix_predictions.parquet \
  --run-summary artifacts/runs/FOLD0/run_summary.json \
  --run-summary artifacts/runs/FOLD1/run_summary.json \
  --run-summary artifacts/runs/FOLD2/run_summary.json \
  --run-summary artifacts/runs/FOLD3/run_summary.json \
  --run-summary artifacts/runs/FOLD4/run_summary.json \
  --output-dir artifacts/robustness/retrained/causal_fixed_preroll0
```

Run all frozen test-time conditions for one primary fold:

```bash
.venv/bin/python scripts/run_robustness.py test-time \
  --fold 0 \
  --checkpoint artifacts/runs/PRIMARY_FOLD0/best_checkpoint.pt \
  --reference-ledger artifacts/runs/PRIMARY_FOLD0/prefix_predictions.parquet \
  --output-dir artifacts/robustness/test-time/fold0 \
  --device auto
```

Repeat the last command for all five folds. `--analysis-id` may be repeated for bounded debugging,
but the manuscript robustness table must include every prespecified condition.

After all five complete test-time decision ledgers exist, compute pooled out-of-fold metrics and
unweighted fold mean plus sample standard deviation as separate estimands:

```bash
.venv/bin/python scripts/run_robustness.py summarize-test-time \
  --decision-ledger artifacts/robustness/test-time/fold0/robustness_decisions.parquet \
  --decision-ledger artifacts/robustness/test-time/fold1/robustness_decisions.parquet \
  --decision-ledger artifacts/robustness/test-time/fold2/robustness_decisions.parquet \
  --decision-ledger artifacts/robustness/test-time/fold3/robustness_decisions.parquet \
  --decision-ledger artifacts/robustness/test-time/fold4/robustness_decisions.parquet \
  --output-dir artifacts/robustness/test-time/combined
```

The summarizer verifies exact out-of-fold coverage for every condition and policy and fits no
parameters. `--allow-subset` is available only for debugging and must not be used for the final
robustness table.

## Outputs and audit trail

A test-time fold run writes:

- calibrated per-prefix logits/probabilities and perturbation metadata;
- adaptive and fixed-duration per-sample decisions;
- fold-condition metrics for the complete calibrated policy and 150/200/250/300 ms baselines;
- the clean validation-only temperature fit and complete policy candidate grid; and
- a JSON summary containing config, manifest, fold-manifest, checkpoint, reference-ledger, and
  environment provenance.

Parquet is preferred; CSV is used when a Parquet engine is unavailable. Reports must identify
fold-level and pooled out-of-fold estimands separately. The current primary ledger schema does not
store its checkpoint hash, so test-time binding uses the run ID, frozen experiment-config hash,
model-architecture hash, and the recorded zero-shift audit; the robustness output itself records
the checkpoint hash.
