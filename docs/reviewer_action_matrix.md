# Reviewer action matrix

## Governing rule

The original code, checkpoints, and prediction files were deleted. None of the numerical
claims in the submitted manuscript can therefore be defended as reproduced results. The
revision will replace every result table and figure with outputs generated from the frozen
rebuild protocol and one canonical out-of-fold prediction ledger. Historical values may be
mentioned only to explain why the revision was necessary.

## Priority 1: experimental integrity

| Issue | Required action | Evidence/output | Status |
|---|---|---|---|
| Figure 6 does not reproduce Table 3 | Regenerate predictions, decisions, metrics, and confusion matrices from one ledger | Validated prefix and decision ledgers; machine-readable metrics JSON | Code implemented; awaiting full rerun |
| Pooled and fold-averaged summaries were conflated | Report pooled OOF metrics and unweighted fold mean ± SD as separate estimands | Fold table plus pooled table with explicit captions | Code implemented; awaiting full rerun |
| Mean latency did not exactly reproduce from Table 4 counts | Recompute latency from the same decision rows used for classification metrics | Decision-time counts and row-level decision ledger | Code implemented; awaiting full rerun |
| Old fixed-duration “mean” is not a deployable baseline | Compare fixed 150, 200, 250, and 300 ms policies separately | Fixed-prefix table and accuracy–earliness curve | Protocol frozen |
| Test set may have influenced historical threshold selection | Fit checkpoints, temperature, and policy on disjoint validation subsets only | Split-role column and selection audit file | Implemented |

## Priority 2: fully specified method

| Reviewer request | Frozen revision action |
|---|---|
| RMS onset frame, hop, threshold, and criterion | State 20 ms RMS frames, 10 ms hop, threshold formula, and two-consecutive-frame confirmation. The recovered whole-utterance peak-relative rule is non-causal and is retained only as an offline comparator; a fixed-threshold causal rule is primary. |
| Quantitative onset validation | Manually annotate a prespecified stratified subset, report detection-failure rate and signed/absolute onset error, and repeat the main policy under onset perturbations. |
| Meaning of 30 ms pre-roll | Define prefix `t` as `t` ms of post-onset signal plus up to 30 ms of already observed past context. Report an ablation with zero pre-roll. |
| Complete small-CNN architecture | Publish every convolution, normalization, activation, pooling operation, dropout, classifier layer, input/output shape, and verified parameter count. |
| CNN-LSTM frame tokens | The spectral CNN pools frequency only; the remaining `T` positions form a `B × T × 48` sequence for a one-layer bidirectional LSTM. |
| Transformer frame tokens | The spectral CNN pools frequency only, adaptively reduces frequency to two bins, and flattens `48 × 2` values into each 96-dimensional time token before positional encoding. |
| EfficientNet-Lite variant | No exact historical variant can be recovered. Remove the unsupported EfficientNet-Lite result rather than invent a variant. |
| CNN-GRU architecture/result | No exact historical architecture or prediction file survives. Remove it from the primary revision rather than claim reproducibility. |
| Entropy threshold and stability tolerance | Search `theta = 0.30…0.95` in steps of 0.05 and `epsilon = 0.00…0.20` in steps of 0.02 on policy-tuning validation rows only. Select minimum mean observation time subject to macro-F1 being within 0.01 absolute of fixed 300 ms. Publish fold-specific selected values. |
| Value of `C` and `log C` | `C = 2`. Division by `log(2)` maps binary Shannon entropy to `[0,1]`, making the threshold scale explicit. Also state that for two classes entropy thresholding is a monotone transform of maximum softmax confidence, so the contribution is the validated sequential policy rather than a novel uncertainty statistic. |
| Value of `Delta` and first predecessor | `Delta = 50 ms`. Compute the 150 ms prediction as a warm-up predecessor for the 200 ms stability tests; never emit a 150 ms decision. |
| Why candidates start at 200 ms | The 150 ms prefix initializes stability. Eligible outputs are 200, 250, and 300 ms. Update the equation, algorithm, and figure consistently. |

## Priority 3: validation and analysis

| Requested analysis | Planned output |
|---|---|
| Calibration | Raw versus temperature-scaled NLL, Brier score, 15-bin ECE, reliability diagrams, and policy performance. |
| Policy-component ablation | Fixed 300 ms, entropy only, entropy plus class stability, entropy plus entropy-change stability, and complete policy. |
| Threshold sensitivity | Validation response surfaces and test reporting only at validation-selected parameters; no test retuning. |
| Accuracy–earliness trade-off | Fixed-prefix curve, consumed fraction, timeliness, and prespecified accuracy–earliness and macro-F1–earliness harmonic means with their exact definition. |
| Fold and speaker variation | All five fold results, mean ± SD, pooled OOF metrics, per-speaker metrics, and speaker-clustered bootstrap 95% confidence intervals. |
| Statistical comparison | Paired utterance correctness analysis plus speaker-clustered paired bootstrap effect intervals. Avoid interpreting a non-significant result as equivalence. |
| False positives in Figure 6 | Regenerate the matrix first. Audit false positives by speaker, prompt type, duration, onset error/failure, entropy, SNR proxy, and decision time. Include representative spectrograms only after a blinded error taxonomy is defined. |
| Robustness | Prespecified additive-noise/SNR and onset-shift tests; causal versus recovered offline onset; zero versus 30 ms pre-roll. |
| Practical latency | Separately time acoustic observation, onset processing, feature extraction, model inference, policy computation, and total wall time. Report warm/cold CPU and available accelerator timing with hardware and batch size. |
| Complexity | Parameters, serialized weights, peak memory, multiply–accumulate/FLOP convention, and prefix-specific runtime. |
| Modern speech baseline | Add a clearly specified pretrained/self-supervised speech representation using the identical speaker folds and input durations. Mark frozen-feature versus fine-tuned use explicitly. |

The calibration, stopping-policy, reporting, onset-validation, error-audit, profiling,
modern-reference, and robustness execution layers are now implemented and tested. They remain
scientifically **pending** until the recovered audio has been validated and the frozen fold runs
have completed; implementation tests are not substitutes for corpus results.

## Priority 4: claims and scope

- Replace “diagnosis” language with “utterance-level dysarthric-speech classification.” The
  experiment does not establish neurological diagnosis or clinical severity.
- Narrow generalization claims to head-mounted-microphone TORGO speech under the evaluated
  speaker-independent folds.
- Add a literature table with corpus, microphone, task, segment duration, speaker separation,
  metric, and latency definition. Do not rank non-comparable numbers as if they share a test
  protocol.
- Describe class balancing exactly: original fold counts, inverse-frequency weighted sampling
  with replacement, epoch sample count, and the resulting expected 50/50 draw distribution.
- Report the exact speaker IDs in train, validation, and test for every fold.
- Report sex, diagnosis/etiology, severity, and prompt type only where the TORGO metadata or an
  authoritative corpus source supports them; label missing fields rather than infer them.

## Requests that cannot be honestly completed from TORGO alone

Cross-corpus testing on UA-Speech and far-field/mobile microphone validation are new-data
experiments, not manuscript edits. They should be run only if legally accessible data and a
compatible task mapping can be established. If they cannot be completed within the revision,
the response must say so directly, narrow the claims, and list them as limitations rather than
claiming that noise injection substitutes for an independent corpus or real microphones.

## Deliverables before resubmission

1. Frozen configuration, environment manifest, source hashes, and exact fold manifest.
2. Raw prefix-logit ledger for every validation and OOF test utterance at 150/200/250/300 ms.
3. Calibration, threshold-selection, and final decision ledgers.
4. Recomputed tables, figures, confidence intervals, and profiling results from those ledgers.
5. Revised manuscript with tracked or highlighted changes.
6. Point-by-point response document using the supplied response sample only as a formatting
   guide; every response must cite a manuscript page/line and an artifact or analysis.
