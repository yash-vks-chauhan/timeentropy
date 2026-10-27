# Time Entropy dysarthria reconstruction

This repository reconstructs the experiments behind *Minimal-Time Dysarthric
Speech Detection Using Speech-Onset Time Entropy*. The original training code,
checkpoints, and predictions were deleted, so the submitted manuscript's numerical
results are treated as unverified historical claims. Revised results are generated
from this implementation and its frozen, speaker-independent protocol; they are not
presented as a recovery of the deleted outputs.

## Evidence and scope

The frozen cohort contains 8,131 TORGO head-mounted-microphone recordings from 15
speakers: 5,197 control and 2,934 dysarthric utterances. The five folds are exclusive
by speaker, and every speaker appears in the test role once. The exact sample and fold
manifests are in `data/metadata/`.

The experiment is utterance-level dysarthric-speech classification, not neurological
diagnosis or severity estimation. Its empirical scope is the selected TORGO head-mounted
channel under the recorded folds. Cross-corpus and real far-field/mobile validation were
not performed, and speaker etiology is unavailable.

[Joy, Umesh, and Abraham (2017)](https://doi.org/10.21437/Interspeech.2017-878)
publish speech-language-pathologist/Frenchay severity categories for the eight TORGO
dysarthric speakers: severe (F01, M01, M02, M04), severe-moderate (M05), moderate
(F03), and mild (F04, M03). Any severity-stratified classification result is descriptive:
severity is speaker-level, several strata contain only one or two speakers, and severity
cannot be disentangled from speaker identity in this cohort.

The repository provides:

- compact log-STFT models (small CNN, CNN–BiLSTM, and Transformer);
- a frozen WavLM Base+ backbone with a trained linear classification head;
- calibrated fixed-prefix and adaptive stopping-policy evaluation;
- speaker-clustered uncertainty and paired comparisons;
- label-free onset diagnostics and imposed onset-shift/noise stress tests;
- condition-matched onset/pre-roll retraining;
- multi-seed Transformer replication;
- quantitative classification-error strata; and
- component-level runtime and compact-model complexity profiling.

Human reference-onset annotation and human qualitative error-cause coding were not
completed. No result may be described as manual onset accuracy, inter-annotator
agreement, or causal error attribution. Synthetic white noise is not a substitute for
real microphone or cross-corpus validation.

## Core design

1. Prefix time is post-onset acoustic observation time. Up to 30 ms of already
   observed historical pre-roll may be included without reducing post-onset evidence.
2. A shared model produces logits at 150, 200, 250, and 300 ms. The 150 ms prefix is
   a warm-up predecessor; eligible decisions are 200, 250, and 300 ms, with forced
   fallback at 300 ms.
3. Checkpoint selection, temperature fitting, and policy tuning use three disjoint
   row subsets of validation-speaker utterances. Test-speaker rows are never used for
   fitting or selection.
4. Pooled out-of-fold metrics and unweighted fold mean ± sample SD are separate
   estimands.
5. Speaker-clustered bootstrap is the primary uncertainty analysis. McNemar-style
   utterance counts are an unclustered sensitivity analysis.
6. For this binary task, normalized entropy is a monotone transform of maximum
   softmax probability. The evaluated object is the calibrated sequential policy,
   not a novel single-prefix uncertainty statistic.

## Environment

Python 3.11 or 3.12 is required (`pyproject.toml` excludes 3.13 and later).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,ssl]'
```

The `ssl` group installs WavLM dependencies but does not authorize or trigger a model
download by itself.

## Data preparation

TORGO audio is not distributed with this repository. Obtain it through an authorized
source and confirm its current license/redistribution terms. The frozen manifest uses
repository-relative paths under `data/raw/torgo/`.

If rebuilding the manifest from the recovered index, provide its location explicitly:

```bash
.venv/bin/python scripts/build_manifests.py \
  --source-index /path/to/torgo_index.csv
```

Recover only the selected head-mounted files and verify them against the manifest:

```bash
.venv/bin/python scripts/recover_torgo_audio.py \
  --groups F --delete-archive-after-verify
```

The tracked frozen config contains a historical machine-local `recovered_index` field
used when the cohort manifest was constructed. Training reads the frozen manifest, not
that forensic source path. A public reproducibility package must redact the local path
in a staged copy while recording both source and packaged hashes; silently rewriting
the frozen config would break artifact provenance.

## Plan, validate, run, and inspect the experiment matrix

The orchestration layer expands the frozen protocol into 100 jobs: 15 primary compact
folds, 10 Transformer seed-replication folds, five WavLM reference folds, 15 retrained
input-ablation folds, and 55 test-time robustness fold-conditions.

```bash
.venv/bin/python scripts/run_experiments.py plan \
  --output-dir artifacts/orchestration/plan

.venv/bin/python scripts/run_experiments.py preflight \
  --device auto

.venv/bin/python scripts/run_experiments.py status
```

To execute missing jobs sequentially, use `run`. WavLM download is disabled unless
explicitly authorized:

```bash
.venv/bin/python scripts/run_experiments.py run \
  --device auto \
  --allow-wavlm-download \
  --output-dir artifacts/orchestration/execution
```

The runner verifies each job's config, manifests, architecture identity, ledger schema,
and producing commit. Consult the final JSON line from `status`; tests alone do not
establish that corpus experiments are complete.

## Evaluate ledgers and generate reports

`scripts/evaluate_ledgers.py` fits one scalar temperature per fold on calibration rows,
selects `theta` and `epsilon` on policy-tuning rows, and emits fixed and adaptive test
decisions. `scripts/generate_reports.py` produces pooled, fold, speaker, calibration,
latency, bootstrap, and paired-comparison tables. Both CLIs take explicit input ledgers;
use `--help` for their complete interfaces.

Canonical generated locations are:

- primary evaluation: `artifacts/evaluation/{small_cnn,cnn_bilstm,transformer,wavlm}-primary/`
- primary reports: `artifacts/reports/<system>/`
- seed evaluations and aggregate: `artifacts/evaluation/transformer-seed*/` and
  `artifacts/evaluation/transformer-three-seed/`
- test-time robustness: `artifacts/robustness/test-time/`
- retrained onset/pre-roll comparisons: `artifacts/robustness/retrained/`
- onset diagnostic: `artifacts/onset_diagnosis/`
- threshold sensitivity: `artifacts/sensitivity/`
- quantitative error audit: `artifacts/error_audit/`
- runtime profiles: `artifacts/profiling/`

Generated artifacts are intentionally ignored by Git because they are large. The
submission reproducibility archive uses an explicit allowlist, hashes every included
file, and excludes raw audio, checkpoints, private paths, incomplete human-review
worksheets, and the unblinded qualitative-review key.

## Analysis, figures, and submission documents

These stages read frozen artifacts. None of them trains, refits, calibrates, or selects a
policy, so each can be re-run at any time without changing a result.

```bash
.venv/bin/python scripts/audit_errors.py                    # error strata and blinded review sheet
.venv/bin/python scripts/generate_sensitivity_artifacts.py  # theta/epsilon surface, accuracy-earliness
bash scripts/evaluate_completion_stages.sh                  # seed replications, aggregate, onset-mode retrains
.venv/bin/python scripts/generate_figures.py                # manuscript figures
.venv/bin/python scripts/build_submission_documents.py      # the two DOCX files plus a marked copy
.venv/bin/python scripts/build_reproducibility_package.py   # privacy-screened archive
```

`evaluate_completion_stages.sh` is the only entry above that requires the full matrix: run it
after `run_experiments.py status` reports every job complete, and only then replace the
`{{PENDING_...}}` tokens in the two Markdown sources with its measured results.

`build_submission_documents.py` renders through pandoc against a reference document it
generates in the venue's A4 geometry, then applies the deterministic Word layout pass in
`src/timeentropy/reporting/docx_layout.py`: duplicate image captions removed, content-aware
column widths, repeated header rows, unsplit rows, captions kept with their content. It
refuses to build a document containing an unresolved `{{PLACEHOLDER}}` token, a missing
bibliography entry, a missing declaration section, or a caption sequence with a gap. Pass
`--allow-placeholders` only for a draft that is not going to a reviewer.

`build_reproducibility_package.py --draft` permits a dirty worktree and unfinished runs;
final mode refuses both, so an archive without the draft marker is one whose every claimed
result exists.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python scripts/verify_response_numbers.py
```

The numerical verifier must be updated whenever the final manuscript or response
changes. Report-generation manifests record their own inputs, hashes, environment,
and producing commit; results were not all produced from one physical file or one Git
commit.

## Provenance rule during long runs

Do not commit while `scripts/run_experiments.py run` is active. The runner captures the
preflight commit and refuses artifacts stamped with a different commit, even for a
documentation-only change. Commit before launching a sweep and keep tracked files
unchanged until it exits.

A later stage may deliberately consume verified artifacts from an earlier commit with
`--allow-dependency-commit-drift`. This bypasses only commit equality; content hashes,
model/config identity, manifests, and ledger invariants remain enforced, and the drift
is recorded.

## Reproducibility limits

The primary training backend was Apple Silicon MPS. Seeds and complete ledgers are
recorded, but at least one required MPS backward operation is nondeterministic; bit-exact
retraining is therefore not claimed. Multi-seed variation is reported separately from
speaker-generalization uncertainty.

Compact-model CPU/MPS profiles are batch-1 warm measurements on the recorded hardware.
They separate acoustic observation from detector compute, crop, STFT, model, posterior,
and policy compute. WavLM runtime/FLOPs were not profiled and must not be inferred from
the compact-model measurements.
