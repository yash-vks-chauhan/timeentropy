# Onset-detector finding

## Summary

The prespecified onset validation that Reviewers 1 and 2 requested (R1.4, R2.4, R3.1) found that
the frozen **primary** onset detector did not localise speech onset on most of the corpus. The
primary mode has been changed from `causal_fixed` to `causal_noise_floor` on the strength of that
validation. The **original manuscript's** detector is affected by the same problem to a lesser
but still substantial degree.

## The defect

`causal_fixed` thresholds short-time RMS at a fixed absolute floor of **0.003**, which is
**−50 dBFS**. Measured room tone in the TORGO head-mounted-microphone recordings sits at
**−22 to −41 dBFS** — that is, *above* the threshold. The first analysis frame therefore already
exceeds it, the two-consecutive-frame criterion is satisfied immediately, and the detector
returns an onset at sample 0.

Concrete example (MC04, `[say 'Eee-P-Ah' repeatedly]`, 4.92 s): room tone holds near −41 dBFS for
about 1.4 s, then speech reaches −9 dBFS. All ten of the first ten frames exceed the threshold,
so the reported "onset" is sample 0 and the nominal 300 ms onset-aligned prefix is entirely room
tone.

The failure is speaker-dependent, because it depends on each speaker's recording level. M05's
recording opens directly on speech, so an anchor at sample 0 is approximately correct there.

## Measured effect

Label-free diagnostic over 600 utterances, 40 per speaker, using only waveforms — no label, no
model output, no test metric (`scripts/diagnose_onset_modes.py`):

| Mode | Fires at sample 0 | Median onset | Median post-onset RMS | Prefix is room tone | Post/pre energy |
|---|---:|---:|---:|---:|---:|
| `causal_fixed` (former primary) | **71.7%** | 0 ms | 0.0119 | **43.3%** | +19.5 dB |
| `causal_noise_floor` (**new primary**) | 4.7% | 1060 ms | 0.1713 | 3.5% | +25.3 dB |
| `recovered_offline_peak` (original manuscript) | **45.0%** | 40 ms | 0.0149 | **27.5%** | +15.2 dB |

The adaptive causal rule places roughly **14×** more acoustic energy inside the prefix and
reduces room-tone prefixes from 43.3% to 3.5%.

## Downstream consequence

The single completed fold under `causal_fixed` (CNN–BiLSTM, fold 0) scored **77.4–82.2%** on its
three validation partitions and **52.9%** on test, against a 51.9% majority-class baseline. The
per-speaker breakdown identifies the mechanism: control speaker **MC04** — 650 utterances, 38% of
that fold's test set, all with onsets at sample 0 — was classified dysarthric on **100%** of
utterances, while control speaker FC01 scored 85.9% and dysarthric speaker M05, whose recording
opens on speech, scored 99.6%.

This is a direct empirical demonstration of the concern raised in R1.7 and R1.18: with 15
speakers, a single speaker can dominate a fold. It is also a direct demonstration of R1.4 — onset
localisation error propagates into the classification result, not merely into the latency claim.

## Why this is not test-set tuning

The decision to change the primary mode rests **only** on waveform statistics: the relationship
between each recording's noise floor and the detector threshold, and the acoustic energy present
in the resulting prefix. No label, model output, or test metric enters the diagnostic, and it can
be reproduced on the raw corpus with no trained model. `causal_noise_floor` was already one of the
three prespecified modes in the frozen configuration; it was selected among prespecified
alternatives on a prespecified label-free criterion, not introduced afterwards.

The test result above is reported as a *consequence* of the defect, not as its evidence. Had the
onset diagnostic been run before any training — as it should have been — the same conclusion would
have followed from the waveform statistics alone.

## Disposition

- `configs/rebuild_v1.yaml`: `onset.primary_mode` → `causal_noise_floor`; `causal_fixed` demoted
  to `secondary_mode`.
- `configs/robustness_v1.yaml`: the primary retrained condition becomes
  `primary_causal_noise_floor_preroll30`; the pre-roll ablation becomes
  `causal_noise_floor_preroll0`; `causal_fixed_preroll30` is retained as a
  `fixed_threshold_comparator` so the failure is quantified in the manuscript rather than merely
  described.
- The completed `causal_fixed` fold is retained under
  `artifacts/evidence/superseded_causal_fixed/` and is not a current result.

## Reporting obligation

The revision reports that the original manuscript's non-causal peak-relative detector fires at
sample 0 on approximately **45%** of utterances, leaving about **27.5%** of prefixes containing
room tone rather than speech. This plausibly affected the submitted results and is disclosed
directly in the response to R1.4 and R2.4 rather than left for a reviewer to discover.
