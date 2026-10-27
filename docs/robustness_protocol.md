# Frozen robustness protocol and evidence map

`configs/robustness_v1.yaml` is a prespecified protocol. It is not a menu from which
favorable conditions may be selected. Its parent experiment and the robustness config
must both be marked `frozen_before_test_evaluation`.

## Analysis families

Input-definition ablations and test-time perturbations answer different questions and
use separate execution paths.

| Family | Conditions | Checkpoint rule | Calibration/policy rule |
|---|---:|---|---|
| Condition-matched input definition | Four definitions × five folds | The primary adaptive/30 ms condition is the parent result. Zero pre-roll, fixed causal floor, and recovered offline peak each require newly trained checkpoints. | Fit a temperature and select `theta`/`epsilon` separately from that condition's validation partitions. |
| Evaluation-only perturbation | Eleven conditions × five folds | Reuse the primary adaptive/30 ms checkpoint; retraining is forbidden. | Fit temperature once on clean calibration rows and select policy once on clean policy-tuning rows. Apply both unchanged to every perturbed test condition. |

The 20 conceptual input-condition folds include five primary folds already counted in
the parent experiment, so the robustness-specific retraining stage has 15 jobs. The
test-time stage has 55 jobs. `run_robustness.py plan` therefore describes 75
fold-condition entries, while `run_experiments.py` expands the complete study to 100
jobs after also counting primary compact models, seed replications, and WavLM.

## Condition-matched input definitions

1. `primary_causal_noise_floor_preroll30`: adaptive causal detector, 30 ms pre-roll;
2. `causal_noise_floor_preroll0`: the same detector with no pre-roll;
3. `causal_fixed_preroll30`: fixed threshold 0.003 with 30 ms pre-roll; and
4. `offline_peak_preroll30`: whole-utterance peak-relative, noncausal alignment with
   30 ms pre-roll.

The last two quantify sensitivity to onset-definition choice. The offline rule cannot
support a real-time claim because its threshold uses the complete waveform.

## Evaluation-only perturbations

The eleven conditions are:

- clean reference;
- zero-mean white Gaussian noise at 20, 10, and 0 dB crop-level RMS SNR; and
- anchor shifts of −50, −30, −10, 0, +10, +30, and +50 ms.

### Noise definition

Noise is added after onset-aligned cropping and before STFT. Its random stream is a
SHA-256 function of primary seed, sample ID, and SNR, so row order, batch size, and
worker count do not change it. The sampled stream is centered and scaled to requested
RMS SNR. An all-zero crop remains zero and records `noise_applied=false`, because finite
relative SNR is undefined.

White noise is a synthetic stress test. It is not an independent corpus, real far-field
microphone, or mobile-channel experiment. A measured improvement under noise is a
counterintuitive shift in the classifier's class bias/decision behavior, not evidence
that noise improves dysarthric speech or a proven regularization mechanism.

### Anchor-shift definition

`requested_anchor = primary_anchor + offset`. The requested anchor is clipped to the
waveform boundary, and padding preserves exact pre-roll plus post-onset length. If the
primary detector has no active run, the registered zero fallback remains and no
artificial “true” onset is invented.

The zero-shift condition is independently recomputed. Its maximum absolute logit
difference from the clean saved ledger is recorded as an informational floating-point
audit only; it is never used to tune, accept, or reject a condition.

Artificial offsets quantify sensitivity to anchor changes. They do not estimate the
real detector's error distribution because no human reference onset is available.

## Leakage and identity controls

For each primary fold, the only fitted test-time quantities are derived from clean
validation rows:

1. scalar temperature from `split_role=calibration`; and
2. complete-policy `theta`/`epsilon` from `split_role=policy_tuning`.

Perturbed test logits are generated after those values are fixed. Every output records
zero test rows used for temperature fitting or policy selection. Conditions are emitted
in frozen config order and are neither ranked nor filtered by test performance.

For each retrained condition, the standard evaluation pipeline enforces the same split
boundary. The wrapper requires five ledgers and five run summaries, verifies condition
config hash, model/onset/pre-roll identity, and out-of-fold coverage, and hashes all
inputs.

## Authoritative commands

Inspect the robustness-only plan:

```bash
.venv/bin/python scripts/run_robustness.py plan \
  --output-dir artifacts/robustness/plan
```

Inspect the complete frozen matrix and current completion state:

```bash
.venv/bin/python scripts/run_experiments.py plan \
  --output-dir artifacts/orchestration/plan

.venv/bin/python scripts/run_experiments.py status
```

Execute only the robustness stages when their dependencies are present:

```bash
.venv/bin/python scripts/run_experiments.py run \
  --stage robustness_retrain \
  --stage robustness_test_time \
  --device auto \
  --output-dir artifacts/orchestration/robustness-execution
```

If dependencies were produced at a verified earlier commit, use
`--allow-dependency-commit-drift`; this bypasses only commit equality and records the
drift. Do not use `--rerun-invalid` to overwrite a scientifically valid run merely
because a later documentation commit changed.

Low-level `run_robustness.py test-time`, `evaluate-retrained`, and
`summarize-test-time` commands remain available for bounded diagnostics. Their inputs
must be explicit; do not use placeholder `FOLD0` paths in a reproducibility record.

## Output map

Test-time evidence:

- per condition/fold: `artifacts/robustness/test-time/<analysis_id>/fold<k>/`
- combined pooled and fold summaries: `artifacts/robustness/test-time/combined/`

Each fold-condition directory contains calibrated prefix predictions, decision rows,
fixed/adaptive metrics, clean validation policy identity, and a hash-bearing summary.
The combined summarizer verifies exact out-of-fold coverage for every condition and
policy and fits no parameters. `--allow-subset` is debugging-only and is forbidden for
the final table.

Retrained evidence:

- `artifacts/robustness/retrained/causal_noise_floor_preroll0/`
- `artifacts/robustness/retrained/causal_fixed_preroll30/`
- `artifacts/robustness/retrained/offline_peak_preroll30/`
- paired pre-roll inference: `artifacts/robustness/retrained/pre_roll_comparison/`

## Completed zero-pre-roll finding

The condition-matched zero-pre-roll experiment does not support the old statement that
30 ms pre-roll makes no material difference.

| Complete calibrated policy | Accuracy | Macro-F1 | Balanced accuracy | Mean decision time |
|---|---:|---:|---:|---:|
| 30 ms primary | 41.95% | 41.95% | 45.74% | 269.39 ms |
| 0 ms retrained | 50.98% | 50.29% | 51.72% | 269.25 ms |

The paired speaker-cluster bootstrap estimate for zero minus 30 ms is +9.03 accuracy
points (95% interval +0.10 to +19.79), +8.35 macro-F1 points (−0.38 to +19.95), and
+5.98 balanced-accuracy points (−1.62 to +15.23). Specificity rises while dysarthric
recall falls, and fold differences are heterogeneous. The 30 ms condition remains the
prespecified primary definition, but the revision must not claim that the pre-roll is
beneficial or neutral.

## Interpretation rules

- Always report pooled OOF and unweighted fold mean ± sample SD separately.
- Report complete calibrated policy and fixed prefixes from the same condition.
- Keep calibration/policy selection condition-specific for retrained definitions.
- State whether a result is condition-matched retraining or evaluation-only stress.
- Treat additive-noise improvements, onset-shift changes, and pre-roll changes as
  descriptive effects on this classifier/cohort.
- Do not infer recording-noise cause, articulatory mechanism, or clinical benefit.
- Do not use a non-significant interval as proof of equivalence.

## Unavailable robustness evidence

- no completed manual onset references or real onset-error bins;
- no cross-corpus evaluation;
- no real far-field/mobile microphone recordings;
- no verified speaker-level neurological etiology; and
- no human qualitative cause coding.

A published speech-language-pathologist/Frenchay severity mapping is available for all
eight dysarthric speakers (Joy, Umesh, and Abraham, 2017;
<https://doi.org/10.21437/Interspeech.2017-878>). It can support descriptive
severity-stratified dysarthric recall/error reporting, but it is constant within speaker
and does not turn a robustness comparison into independent severity validation.

These are declared limitations. They are not replaced by synthetic noise, artificial
anchor shifts, or an empty annotation template.
