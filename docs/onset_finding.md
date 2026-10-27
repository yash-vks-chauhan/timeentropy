# Speech-onset detector diagnostic

## Scope of this evidence

Reviewer scrutiny exposed a defect in the submitted onset rule. The reconstruction
therefore compared three frozen RMS modes on a deterministic, speaker-balanced sample
of 600 TORGO waveforms (40 per speaker). This comparison is **label-free**: it uses no
diagnosis label, model output, or test metric.

It is not human onset validation. The 300-row manual annotation worksheet has no
completed reference onsets, so this study cannot report onset error, agreement, or the
fraction of anchors within a human tolerance. “Returned sample zero” and “quiet 300 ms
crop” are diagnostic heuristics, not ground-truth localization errors.

## Three evaluated modes

All modes use left-aligned 20 ms RMS frames, 10 ms hops, and the first run of two
consecutive active frames.

### Fixed causal floor

`causal_fixed` uses `tau=0.003` (approximately −50.5 dBFS). The floor is below the
initial signal level in many head-mounted recordings, so the first two frames are often
already active and the returned anchor is frame zero.

### Recovered whole-utterance peak rule

`recovered_offline_peak` uses
`tau=max(0.003, max_m r[m] × 10^(−35/20))`. Because the maximum is taken over the
complete recording, the rule reads future audio and cannot support an online-latency
claim. It also frequently returns frame zero.

### Adaptive causal noise-floor rule

`causal_noise_floor` uses

```text
tau = max(0.003, median(r[0], ..., r[9]) × 10^(15/20)).
```

The ten frames start at 0, 10, ..., 90 ms. With 20 ms frames, the last reference frame
ends at 110 ms, so the threshold is acoustically available only after 110 ms. Online
confirmation is the later of threshold availability and the end of the required
two-active-frame run. The returned anchor may refer to an earlier frame; anchor time,
confirmation/availability time, and CPU detector execution time are distinct.

The adaptive reference can be contaminated when speech begins inside the first 110 ms.
Without human references, its absolute localization accuracy remains unknown.

## Label-free diagnostic results

`scripts/diagnose_onset_modes.py` generated
`artifacts/onset_diagnosis/onset_mode_diagnostic.csv` and
`onset_mode_summary.json`. The script uses sample zero as the registered crop fallback
when no active run is detected, so the combined “returned zero” rate must be separated
into detected-at-frame-zero and no-onset cases.

| Mode | Detected at frame 0 | No active run; zero fallback | Returned sample 0 (combined) | Median returned anchor | Median 300 ms crop RMS | Quiet crop (`RMS<0.01`) | Median post/pre RMS ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `causal_fixed` | 429/600 (71.5%) | 1/600 (0.2%) | 71.7% | 0 ms | 0.0119 | 43.3% | +19.5 dB |
| `causal_noise_floor` | 0/600 (0.0%) | 28/600 (4.7%) | 4.7% | 1060 ms | 0.1713 | 3.5% | +25.3 dB |
| `recovered_offline_peak` | 269/600 (44.8%) | 1/600 (0.2%) | 45.0% | 40 ms | 0.0149 | 27.5% | +15.2 dB |

Relative to the fixed-floor mode, the adaptive mode has approximately 14.4 times
larger median crop RMS and reduces the quiet-crop heuristic from 43.3% to 3.5%. This is
an amplitude/RMS comparison, not “14× more acoustic energy” (energy is proportional to
RMS squared).

The median initial-reference RMS in the sampled recordings is approximately 0.00850
(−41.4 dBFS), already above the 0.003 fixed floor. Initial windows vary widely and can
contain immediate speech, so no single “room-tone range” should be asserted from this
diagnostic.

## Why the primary mode changed

The primary changed from `causal_fixed` to `causal_noise_floor` because of the
prespecified label-free criteria: fewer sample-zero fallbacks/anchors, fewer quiet
300 ms crops, and substantially larger post-anchor RMS. The decision did not use class
labels or classifier performance. The adaptive mode was already a registered
alternative; it was not invented after viewing test accuracy.

The earlier causal-fixed single-fold run is superseded evidence and must not be used as
a current performance result. The final input-definition comparison uses all five
condition-matched folds with condition-specific calibration and policy selection under
`artifacts/robustness/retrained/`.

## Downstream stress test

The prespecified test-time experiment shifts the primary anchor by −50, −30, −10, 0,
+10, +30, and +50 ms while keeping checkpoint, calibration, and policy fixed. For the
complete calibrated Transformer policy, pooled accuracy is 41.95% at zero shift,
38.89% at −10 ms, 35.27% at −30 ms, 35.84% at −50 ms, 43.44% at +10 ms, 41.63% at
+30 ms, and 40.97% at +50 ms.

These imposed-shift results show asymmetric sensitivity to the crop anchor. They do not
measure real detector error, prove the adaptive anchor is correct, or identify a causal
acoustic mechanism. The complete fold-level and pooled evidence is in
`artifacts/robustness/test-time/combined/`.

## Correct reporting language

Use:

- “sample-zero anchor/fallback rate,” not “onset failure rate,” unless the no-active-run
  flag itself is being reported;
- “quiet-crop heuristic,” not “prefix proven to contain no speech”;
- “label-free onset-mode diagnostic,” not “manual onset validation”;
- “noncausal whole-utterance comparator,” not a deployable detector; and
- “descriptive association/sensitivity,” not causal error propagation.

The submitted peak-relative rule returned sample zero for 45.0% of the diagnostic
sample and produced a quiet 300 ms crop for 27.5%. This plausibly affects the validity
of the submitted onset-aligned interpretation, but the deleted predictions prevent a
quantitative correction of the historical results.

## Condition-matched retraining reverses the diagnostic's implication

Each comparator was retrained on all five folds with its own temperature and
validation-selected policy, then evaluated on the identical 8,131 out-of-fold utterances.
Both rejected rules classify better than the primary rule: pooled accuracy 59.28% for
`causal_fixed` and 67.37% for `recovered_offline_peak`, against 41.95%. The paired
speaker-clustered accuracy differences are +17.33 points [+3.85, +28.97] and +25.42 points
[+12.42, +37.14]; both exclude zero.

The advantage is an artifact of where each rule fails. Stratifying the difference by whether
the crop begins at the recording start puts the whole effect in that stratum: `causal_fixed`
is 14.13 points *worse* than the primary rule wherever it anchors after sample zero. Inside
the recording-start stratum the gain lands on control utterances (+33.02 and +47.08) rather
than dysarthric ones (+7.49 and −1.23). The reason is that the comparators reach the
recording start at very different rates per class — 88.94% of control versus 59.78% of
dysarthric utterances for `causal_fixed`, and 72.73% versus 25.09% for
`recovered_offline_peak`, against 6.39% versus 2.15% for the primary rule.

A detector whose failure rate depends on the class supplies a shortcut: "this input looks
like the start of a recording" becomes a proxy for "control". The measured accuracy gain is
therefore evidence about the recording, not about speech-onset localization, and it does not
justify reinstating either rule. The primary rule was fixed from waveform diagnostics before
any test evaluation; changing it on these test numbers would be selection on the test set.

The stratification establishes where the difference lives. It does not identify what inside
those crops carries the signal, and no causal claim is made. Artifacts:
`artifacts/robustness/retrained/{causal_fixed_preroll30,offline_peak_preroll30}/` and
`artifacts/onset_diagnosis/onset_mode_confound/`.

## Remaining limitations

- No human reference onsets or inter-annotator agreement.
- Possible contamination of the adaptive 110 ms noise-reference horizon by immediate
  speech.
- One corpus and one head-mounted channel.
- A crop-energy heuristic cannot distinguish low-energy speech from room tone.
- Test-time artificial shifts do not reproduce the joint distribution of real onset
  errors.
- The confound stratification shows that the comparator advantage lives in recording-start
  crops; it does not isolate the acoustic or channel property responsible.
