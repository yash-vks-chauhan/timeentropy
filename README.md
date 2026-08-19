# Time Entropy Dysarthria Rebuild

This repository is a clean reconstruction of the experiments behind the manuscript
`Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy`.

The original training code, checkpoints, and prediction files were deleted. This rebuild
therefore treats the manuscript's old metrics as unverified historical values. It does not tune
the replacement implementation toward those old numbers. All revised results will be generated
from a canonical out-of-fold prediction ledger and will replace the old tables and figures.

## Recovered evidence

- The surviving TORGO index contains 8,216 head-mounted-microphone recordings.
- Applying `usable_prompt == True` and `duration_sec >= 0.05` reproduces the manuscript's
  8,131-utterance subset: 5,197 control and 2,934 dysarthric recordings from 15 speakers.
- An older manuscript preserves speaker-group folds and partial preprocessing/training
  settings. Those settings are recorded in `docs/reconstruction_spec.md` but are not treated
  as proof of the deleted implementation.

## Rebuild principles

1. Freeze preprocessing, folds, model definitions, calibration, and selection rules before
   inspecting test results.
2. Treat nominal prefix time as post-onset acoustic observation time. A 30 ms historical
   pre-roll may be included in the input but does not reduce the amount of post-onset audio.
3. Use 150 ms only as a warm-up observation. Eligible decision times are 200, 250, and
   300 ms, with a 300 ms forced fallback.
4. Fit calibration and select entropy-policy parameters on validation speakers only.
5. Save sample-, speaker-, fold-, seed-, model-, and prefix-level logits before computing any
   aggregate metric.
6. Report fold means and pooled out-of-fold metrics as distinct estimands.

## Environment

The project targets Python 3.11 or 3.12. Python 3.14 is intentionally excluded because the
scientific and PyTorch ecosystem may not provide stable wheels for it.

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

The optional WavLM reference uses the additional `ssl` dependency group. Installing that group
does not itself download model weights:

```bash
.venv/bin/python -m pip install -e '.[dev,ssl]'
```

Build and validate the recovered manifest:

```bash
.venv/bin/python scripts/build_manifests.py \
  --source-index /Users/yashchauhan/timeentropy_runtime/artifacts/metadata/torgo_index.csv
```

Recover only the selected head-mounted-microphone audio. The recovery command verifies each
official archive and extracted WAV against the frozen manifest, processes one group at a time,
and can remove the verified temporary archive to conserve disk space:

```bash
.venv/bin/python scripts/recover_torgo_audio.py \
  --groups F --delete-archive-after-verify
```

Train one frozen STFT-model fold and save raw logits at every registered prefix:

```bash
.venv/bin/python scripts/run_fold.py --fold 0 --model transformer --device auto
```

Run the modern frozen WavLM Base+ reference on the same fold. Model download is disabled unless
it is explicitly authorized; the configured Hugging Face revision is an immutable commit:

```bash
.venv/bin/python scripts/run_ssl_fold.py --fold 0 --device auto --allow-download
```

After all five ledgers for one model/seed are present, fit fold-specific temperatures, select
`theta` and `epsilon` only on policy-tuning validation rows, and emit final test decisions:

```bash
.venv/bin/python scripts/evaluate_ledgers.py \
  --prefix-ledger artifacts/runs/FOLD0/prefix_predictions.parquet \
  --prefix-ledger artifacts/runs/FOLD1/prefix_predictions.parquet \
  --prefix-ledger artifacts/runs/FOLD2/prefix_predictions.parquet \
  --prefix-ledger artifacts/runs/FOLD3/prefix_predictions.parquet \
  --prefix-ledger artifacts/runs/FOLD4/prefix_predictions.parquet \
  --output-dir artifacts/evaluation/transformer-primary
```

Generate pooled OOF, fold-mean, speaker-bootstrap, calibration, confusion-matrix, error, and
paired-comparison reports from the validated ledgers:

```bash
.venv/bin/python scripts/generate_reports.py \
  --prefix-ledger artifacts/evaluation/transformer-primary/calibrated_prefix_predictions.parquet \
  --decision-ledger artifacts/evaluation/transformer-primary/decisions.parquet \
  --output-dir artifacts/reports/transformer-primary
```

The prespecified robustness matrix can be validated without launching training:

```bash
.venv/bin/python scripts/run_robustness.py plan \
  --output-dir artifacts/robustness/plan
```

The repository also contains dedicated commands for deterministic onset-annotation sampling,
onset-error evaluation, blinded false-positive/false-negative auditing, and component-level
runtime profiling. Their protocols and interpretation rules are documented in
`docs/reconstruction_spec.md`, `docs/robustness_protocol.md`, and
`docs/reviewer_action_matrix.md`.

The placeholder fold directory names in the example must be replaced by the actual run IDs.

## Operating rule: do not commit while a sweep is running

`scripts/run_experiments.py run` captures the Git commit at preflight and stamps it into every
checkpoint and run summary, then refuses any artifact whose recorded commit differs. Committing
mid-sweep therefore splits the provenance chain and halts the run, **even when only documentation
changed** — the guard deliberately does not judge which files are scientifically relevant. Land
every commit before launching, and leave the worktree untouched until the sweep finishes.

No revised scientific result should be quoted until audio recovery, all frozen fold runs,
calibration/policy selection, ledger validation, and report generation have completed. Passing
unit tests establishes implementation invariants; it is not experimental evidence.
