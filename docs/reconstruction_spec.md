# Reconstruction specification and evidence register

## Status

The original source code, checkpoints, validation logits, test predictions, and plotting
scripts are unavailable. Old numerical results are therefore historical claims rather than
reproduced results. The rebuild must regenerate every result from raw audio.

## Directly recovered and independently checked

- `torgo_index.csv` contains 8,216 head-mounted-microphone rows from 15 speakers.
- Filtering `usable_prompt == True` and `duration_sec >= 0.05` yields exactly 8,131 rows:
  5,197 control and 2,934 dysarthric.
- Selected audio totals approximately 7.07 hours and about 0.76 GiB of PCM16 sample payload.
- The five validation/test speaker groups in `configs/folds_v1.yaml` reproduce the utterance
  counts printed in the older manuscript.
- Each speaker appears in a test group exactly once.

## Recovered narrative settings, not independently verified against code

- Audio resampling to mono 16 kHz.
- Recovered onset detector: 20 ms RMS frame, 10 ms hop, threshold equal to the larger of
  0.003 and 35 dB below whole-utterance peak RMS, followed by 30 ms pre-roll.
- STFT: 512-point FFT, 25 ms Hann window, 10 ms hop, no centered padding, log magnitude,
  per-sample standardization.
- AdamW, learning rate 0.00015, weight decay 0.0005, dropout 0.30, label smoothing 0.05,
  gradient clipping at 5.0, maximum 35 epochs, early-stopping patience 8.
- Training augmentation: gain between -3 and +3 dB, additive noise at 20-35 dB SNR,
  frequency masks up to 16 bins, and time masks up to four frames.
- Historical entropy grid 0.30-0.95 and epsilon 0.08. A table reported Transformer fold
  thresholds of 0.95, 0.90, 0.85, 0.85, and 0.90, but the exact old objective was not defined.
- Narrative parameter counts were 23,538 (small CNN), 69,354 (CNN-LSTM), 54,762
  (CNN-GRU), 57,998 (custom EfficientNet-like model), and 196,594 (Transformer). Exact
  layer definitions were not preserved.

## Contradictions that the rebuild resolves

1. The recovered onset threshold depends on whole-utterance peak RMS and is therefore
   non-causal. It may be retained only as an offline-alignment comparator. Label-free validation
   subsequently showed it also fails to localise onset on about 45% of utterances, because its
   threshold falls below the recordings' room tone; the primary detector is the level-adaptive
   causal noise-floor rule. See `docs/onset_finding.md`.
2. The old wording appears to include the 30 ms pre-roll inside a nominal prefix, while also
   claiming the full nominal duration occurs after onset.
3. One old algorithm applied stability only "if a previous prefix exists," permitting a
   200 ms entropy-only decision. The current equation implies comparison against 150 ms.
4. Old Table 3 likely reports unweighted fold means while the confusion matrix and latency
   counts pool 8,131 out-of-fold utterances. The estimands were not labelled.
5. A fixed-duration arithmetic mean at 225 ms was presented like a deployable baseline.
6. The manuscript claimed five model families but reported only three.

## Frozen rebuild decisions

- Nominal time `t` means `t` milliseconds of post-onset audio. The input also includes up to
  30 ms of already-observed historical pre-roll, so its maximum duration is `t + 30 ms`.
- The same trained model produces logits for 150, 200, 250, and 300 ms prefixes.
- The 150 ms result is a warm-up predecessor. Decisions are eligible at 200/250/300 ms.
- A 200 ms decision must satisfy stability relative to the 150 ms posterior.
- Calibration and entropy-policy selection use validation speakers only.
- The primary policy minimizes validation observation time subject to macro-F1 remaining
  within 0.01 absolute of that fold's fixed-300-ms validation macro-F1.
- The validation speakers are deterministically partitioned by hashed sample ID into
  checkpoint-stopping (50%), calibration (25%), and policy-tuning (25%) subsets.
- The Transformer is the prospectively designated headline architecture; test performance is
  not used to choose the headline model or random seed.
- Test results are inspected only after configuration freezing and validation selection.
- CNN-GRU and the undefined custom EfficientNet-lite are removed from the revised primary
  comparison. A modern pretrained speech representation is added instead.
