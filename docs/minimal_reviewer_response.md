# Response to Reviewers

**Manuscript:** Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy

We thank the editor and reviewers for identifying substantial reproducibility and reporting problems. The most important issue was that the submitted confusion matrix did not reconcile with the reported accuracy and macro-F1. The original code, checkpoints, and per-utterance predictions could not be recovered, so we did not adjust the old values to fit the figure. We rebuilt the experiment from raw audio under a frozen, version-controlled protocol and replaced the unreproducible results.

To keep the revision readable, the main manuscript retains the original title, section sequence, clinical background, and venue formatting wherever those remain accurate. Exact architectures, folds, threshold surfaces, robustness conditions, error strata, and runtime details are placed in the Supplementary Material. The marked copy highlights the substantive additions and corrected values without applying blanket yellow highlighting to unchanged prose.

## Reviewer 1

### R1.1 Novelty relative to confidence-based early classification and entropy rejection

**Author Response:** We agree. The Introduction and Section 3.2 now state that normalized binary entropy is a monotone transformation of maximum-softmax confidence and is not a new uncertainty statistic. The evaluated contribution is the complete sequential operating protocol: onset-aligned prefixes, held-out calibration, validation-only threshold selection, adjacent-prefix class/entropy stability, and a bounded endpoint.

**Manuscript location:** Introduction; Section 3.2; Section 4.3.

### R1.2 Exact architectures and parameter configurations

**Author Response:** Section 3.3 gives a compact architecture summary and Supplementary Tables S2–S4 give every retained layer, tokenizer, output shape, and parameter count. The small CNN has 23,538 trainable parameters, CNN–BiLSTM 69,354, and Transformer 196,594. Historical CNN-GRU and EfficientNet-lite claims were removed because no exact definition or prediction record survives; inventing a variant retrospectively would be misleading.

**Manuscript location:** Section 3.3; Supplementary Tables S1–S4.

### R1.3 Entropy threshold and stability-tolerance tuning

**Author Response:** We now report exhaustive search over $\theta=0.30,0.35,\ldots,0.95$ and $\epsilon=0.00,0.02,\ldots,0.20$ (154 pairs per fold). Selection minimizes validation observation time subject to macro-F1 remaining within 0.01 of that fold's fixed-300-ms validation macro-F1, with stated tie-breakers. The complete 770-row surface accompanies the supplement.

**Manuscript location:** Section 3.2; Supplementary Table S12 and the accompanying CSV.

### R1.4 Quantitative validation of RMS onset localization

**Author Response:** No human reference-onset labels exist, so we do not report localization error or agreement. We instead report a label-free 600-utterance diagnostic, condition-matched onset-rule retraining, and imposed ±10/30/50-ms shifts, consistently describing these as sensitivity analyses rather than ground-truth onset validation.

**Manuscript location:** Sections 3.1, 4.1, and 4.5; Supplementary Tables S7–S9 and S14.

### R1.5 Meaning and effect of the 30-ms pre-roll

**Author Response:** A nominal $t$-ms prefix now means 30 ms of already-observed history plus the full $t$ ms after the returned anchor. Pre-roll does not add future waiting. The condition-matched comparison did not establish a balanced-accuracy benefit for 30 ms, so the earlier performance-preservation claim was withdrawn.

**Manuscript location:** Sections 3.1 and 4.5; Supplementary Table S11.

### R1.6 Pooled versus fold-averaged results

**Author Response:** The revision distinguishes pooled out-of-fold estimates from unweighted fold means. The complete adaptive Transformer results are reported separately for each fold and as mean ± sample SD.

**Manuscript location:** Sections 3.4 and 4.3; Table 5; Supplementary Table S19.

### R1.7 Variation and confidence intervals with 15 speakers

**Author Response:** We added per-fold variation, per-speaker results, and 10,000-replicate speaker-clustered bootstrap intervals. The wide intervals and fold spread are now central to the interpretation.

**Manuscript location:** Sections 4.2–4.4; Table 5; Supplementary Tables S10, S13, and S16.

### R1.8 Original and effective class distributions and balancing

**Author Response:** The selected data contain 5,197 control (63.92%) and 2,934 dysarthric (36.08%) utterances. Training uses inverse-frequency weighted sampling with replacement, giving equal expected class mass; test evaluation retains the original distribution. The distinction explains why raw accuracy and balanced accuracy are both reported.

**Manuscript location:** Section 3.4; Supplementary Table S5.

### R1.9 Comparability with prior methods

**Author Response:** Table 1 now compares corpus, input duration, speaker protocol, endpoint, and the principal reason each result is not directly comparable. We do not rank headline scores across incompatible protocols.

**Manuscript location:** Section 2.3; Table 1.

### R1.10 Modern pretrained/self-supervised baseline

**Author Response:** We added frozen WavLM Base+ using the same folds, onset rule, pre-roll, prefixes, balancing, and validation partitions. It reaches 59.18% pooled balanced accuracy at 300 ms, but its speaker-clustered interval includes chance; this is reported as directional rather than conclusive evidence.

**Manuscript location:** Sections 3.3, 4.2, and 5; Tables 2–3.

### R1.11 Standardized accuracy–earliness measure

**Author Response:** Section 3.4 defines consumed fraction, timeliness, and an accuracy–timeliness harmonic score while preserving accuracy and time as separate outcomes. Table 4 shows that fixed 150 ms is the strongest evaluated point on this scale and that the adaptive policy does not improve the fixed-duration frontier.

**Manuscript location:** Sections 3.4 and 4.3; Table 4; Supplementary Table S20.

### R1.12 Statistical support for improved sensitivity

**Author Response:** The blanket sensitivity-improvement claim was withdrawn. Speaker-clustered paired intervals show that, relative to fixed 300 ms, the adaptive policy changes dysarthric recall by −0.68 points [−1.54, −0.03] and accuracy by +0.01 [−0.41, +0.44]. Non-significant differences are not called equivalent.

**Manuscript location:** Section 4.3.

### R1.13 Posterior calibration

**Author Response:** We added raw and temperature-scaled negative log likelihood, Brier score, 15-bin ECE, and policy results. The Transformer's raw ECE is 0.429 and falls to 0.212 after scaling; calibration changes fallback behavior without improving accuracy.

**Manuscript location:** Section 4.3.

### R1.14 Cross-corpus validation

**Author Response:** This was not performed because no compatible licensed external corpus was available. We restrict all empirical claims to the selected TORGO head-mounted channel and state cross-corpus validation as required future work rather than implying generalization.

**Manuscript location:** Section 5; Supplementary Section S8.

### R1.15 Practical latency components

**Author Response:** The revision separates post-anchor acoustic observation, onset-threshold availability, buffered pre-roll, crop/STFT/model/posterior/policy computation, and their measured sum. A 200-ms decision uses 200 ms post-anchor observation and 3.629 ms warm CPU compute; 200 ms is not elapsed time from recording-session start.

**Manuscript location:** Sections 3.1 and 4.5; Supplementary Table S18.

### R1.16 Complexity, size, memory, and inference time

**Author Response:** Table 7 reports compact-model parameter counts, serialized weights, MACs, FLOPs, CPU compute, and peak-process RSS increments. WavLM deployment complexity was not profiled and is explicitly marked unavailable.

**Manuscript location:** Section 4.5; Table 7; Supplementary Table S18.

### R1.17 Head-mounted microphone limitation

**Author Response:** Real mobile, conversational, array-microphone, and far-field evaluations were not completed. Synthetic white noise is not presented as a microphone proxy. The limitation and required matched-channel validation are explicit.

**Manuscript location:** Sections 4.5 and 5; Supplementary Section S8.

### R1.18 Per-speaker or fold-wise confusion information

**Author Response:** We report complete fold metrics and per-speaker accuracy/prediction rates. These show that performance is highly heterogeneous and not attributable to a single speaker.

**Manuscript location:** Tables 5–6; Supplementary Tables S10 and S16.

### R1.19 Detection versus clinical diagnosis

**Author Response:** The manuscript consistently describes corpus-defined utterance classification. It explicitly states that the task neither diagnoses neurological disease nor estimates clinical severity.

**Manuscript location:** Abstract; Sections 5 and 6.

### R1.20 Cross-checking tables and Figure 6

**Author Response:** The submitted Figure 6 and its metrics were withdrawn. The corrected matrix is generated from the same 8,131 decision rows as all pooled metrics and gives $(1669+1742)/8131=41.95\%$ accuracy and 269.39-ms mean observation.

**Manuscript location:** Section 4.4; Table 6; Figure 2.

## Reviewer 1 — Required Experimental Validation

### R1-EV.1 Recompute all metrics from the same prediction file

**Author Response:** Completed. Accuracy, macro-F1, class metrics, confusion matrix, mean observation, and fallback are generated from canonical out-of-fold ledgers with count and linkage invariants. The primary complete Transformer policy gives 41.95% accuracy, 41.95% macro-F1, 59.37% dysarthric recall, 269.39 ms mean observation, and 64.25% fallback.

**Manuscript location:** Sections 3.4 and 4.3–4.4; Tables 5–6.

### R1-EV.2 Five-fold statistics with mean, SD, and confidence intervals

**Author Response:** Completed. Table 5 reports every fold and mean ± SD; speaker-clustered intervals and per-speaker results are provided in the text and supplement.

**Manuscript location:** Table 5; Supplementary Tables S10, S13, S16, and S19.

### R1-EV.3 Threshold sensitivity

**Author Response:** Completed on validation-only policy rows. The exact grid contains 154 cells per fold, and the pooled policy combines fold-specific selected cells rather than one global threshold.

**Manuscript location:** Section 3.2 and Section 4.3; Supplementary Table S12 and accompanying CSV.

### R1-EV.4 Raw versus calibrated posterior entropy

**Author Response:** Completed. Raw and temperature-scaled calibration metrics and policy outputs are reported. Scaling reduced ECE but increased validation-selected fallback and did not improve classification.

**Manuscript location:** Section 4.3.

### R1-EV.5 Policy-component ablation

**Author Response:** Completed. Fixed policies, entropy-only, maximum-softmax, individual stability conditions, raw complete policy, and scaled complete policy were evaluated. Entropy-only and maximum-softmax were identical, and no component improved the fixed frontier.

**Manuscript location:** Section 4.3; Table 4; Supplementary Table S20.

### R1-EV.6 Fixed 200, 250, and 300 ms and standardized trade-off

**Author Response:** Completed, with fixed 150 ms also retained as a valid baseline and mandatory predecessor. Table 4 reports accuracy, macro-F1, observation time, fallback, and harmonic accuracy–timeliness.

**Manuscript location:** Section 4.3; Table 4; Supplementary Table S20.

### R1-EV.7 Cross-corpus test

**Author Response:** Not completed. The revised claims are restricted accordingly, and cross-corpus testing is identified as a necessary next step.

**Manuscript location:** Section 5; Supplementary Section S8.

### R1-EV.8 Noise, onset errors, microphones, speech type, and severity

**Author Response:** We completed white-noise and imposed-onset-shift sensitivity, prompt-type, duration, speaker, detector-status, and descriptive severity stratification. Real microphone/channel transfer, continuous-conversation evaluation, and human onset-error measurement were not performed and are not implied.

**Manuscript location:** Sections 4.1, 4.4, 4.5, and 5; Supplementary Tables S14–S17.

### R1-EV.9 Parameters, FLOPs, timing, and total latency

**Author Response:** Completed for compact models. Parameter count, serialized size, MACs, FLOPs, CPU inference path, RMS, STFT, policy, total warm compute, and acoustic observation are separated. WavLM deployment profiling remains unavailable.

**Manuscript location:** Section 4.5; Table 7; Supplementary Table S18.

## Reviewer 2

### R2.1 Choice and sensitivity of $\theta$ and $\epsilon$

**Author Response:** Exact ranges, resolution, objective, tie-breakers, fold selections, and the full validation surface are now reported. The result shows that no universal threshold was supported.

**Manuscript location:** Sections 3.2 and 4.3; Supplementary Table S12.

### R2.2 Background and theoretical uniqueness

**Author Response:** The related-work section now connects the method to early classification, posterior confidence, entropy deferral, and calibration. The revision explicitly withdraws novelty for binary entropy itself and limits the contribution to the combined operating protocol and its empirical audit.

**Manuscript location:** Introduction; Sections 2.2 and 3.2.

### R2.3 Systematic comparison table

**Author Response:** Table 1 compares input duration/unit, corpus, speaker protocol, endpoint, and the principal comparability limitation. It avoids ranking incompatible headline scores.

**Manuscript location:** Section 2.3; Table 1.

### R2.4 Complete onset-detection framework

**Author Response:** Section 3.1 specifies the RMS equation, 20-ms left-aligned frames, 10-ms hops, adaptive threshold, two-consecutive-frame rule, lack of smoothing, failure handling, threshold availability, and confirmation timing. Label-free diagnostics and imposed shifts are reported separately.

**Manuscript location:** Sections 3.1, 4.1, and 4.5; Supplementary Tables S7–S9 and S14.

### R2.5 Posterior calibration

**Author Response:** Raw and scaled NLL, Brier score, ECE, temperatures, and policy outcomes are now included. The original assumption of reliable softmax entropy is not retained.

**Manuscript location:** Section 4.3.

### R2.6 Exact $\theta$ and $\epsilon$ values

**Author Response:** The selected pairs are reported for all folds in Supplementary Table S12. Folds 0 and 2 selected (0.95, 0.20); folds 1, 3, and 4 selected (0.30, 0.00).

**Manuscript location:** Section 4.3; Supplementary Table S12.

### R2.7 CNN-GRU and EfficientNet-lite missing from results

**Author Response:** The historical claims were removed. No exact architecture, EfficientNet-Lite variant, checkpoint, or prediction file survives, so retaining names without reproducible results would compound the problem. A future newly specified baseline would be labelled as a new experiment, not a recovered historical result.

**Manuscript location:** Introduction and Section 3.3; Supplementary Section S8.

### R2.8 Demographic and clinical distributions

**Author Response:** The revision reports class, sex, selected utterance count, prompt type, and published speaker-level severity. Neurological etiology is not assigned to speaker IDs because the verified source reports only cohort-level totals. Severity results are explicitly descriptive and confounded with speaker identity.

**Manuscript location:** Section 3.4; Supplementary Table S17.

### R2.9 Exact five-fold speaker assignments

**Author Response:** All train, validation, and test speaker IDs and test counts are now provided.

**Manuscript location:** Supplementary Table S6.

### R2.10 Architecture and training reproducibility

**Author Response:** Complete retained architectures, parameters, optimizer, learning rate, weight decay, batch size, epochs, patience, clipping, dropout, label smoothing, augmentation, balancing, checkpoint metric, and seeds are reported.

**Manuscript location:** Sections 3.3–3.4; Supplementary Tables S1–S5.

### R2.11 Fold variability, speaker results, and significance

**Author Response:** The revision includes every fold, mean ± SD, per-speaker behavior, and speaker-clustered paired bootstrap intervals. Utterance-level McNemar results are treated only as descriptive sensitivity because utterances within a speaker are not independent.

**Manuscript location:** Sections 3.4 and 4.2–4.4; Table 5; Supplementary Tables S10, S13, S16, and S19.

### R2.12 Fair fixed-duration comparison

**Author Response:** The arithmetic mean of fixed durations was removed as a comparator. The adaptive policy is now compared separately with fixed 150, 200, 250, and 300 ms on identical utterances. It does not improve the observed accuracy–earliness frontier.

**Manuscript location:** Section 4.3; Table 4; Supplementary Table S20.

## Reviewer 3

### R3.1 RMS onset threshold, frame, hop, and criterion

**Author Response:** The detector uses 20-ms left-aligned frames, a 10-ms hop, no smoothing, an adaptive threshold equal to the larger of 0.003 and the first-ten-frame median multiplied by $10^{15/20}$, and the first run of two active frames. Failure handling and signal-availability timing are also stated.

**Manuscript location:** Section 3.1.

### R3.2 Why 30-ms pre-roll and its performance effect

**Author Response:** Thirty milliseconds is a bounded history slot intended to reduce truncation after a late threshold crossing; it adds no future waiting. Condition-matched retraining did not establish a balanced-accuracy advantage and revealed a class trade-off, so no performance-benefit claim remains.

**Manuscript location:** Sections 3.1 and 4.5; Supplementary Table S11.

### R3.3 Complete small CNN architecture

**Author Response:** The three 3×3 convolution/BatchNorm/SiLU/2×2-pooling stages, length-aware mean, dropout, 64→2 head, shapes, and 23,538-parameter total are now reported.

**Manuscript location:** Section 3.3; Supplementary Table S2.

### R3.4 Exact EfficientNet-lite variant

**Author Response:** The exact historical variant cannot be recovered. We therefore removed EfficientNet-lite from the evaluated systems instead of inventing an answer. Any future EfficientNet-Lite0 experiment will be presented as newly specified and newly trained.

**Manuscript location:** Introduction and Section 3.3.

### R3.5 Conversion of log-STFT maps to frame-wise tokens

**Author Response:** Both sequence front ends pool only frequency. CNN–BiLSTM forms one 48-dimensional token per time frame; the Transformer pools frequency to two bins, flattens 48×2 values into a 96-dimensional token, projects 96→96, and adds positional encoding. Masks exclude padded states.

**Manuscript location:** Section 3.3; Supplementary Tables S3–S4.

### R3.6 Selection of $\theta$ and $\epsilon$

**Author Response:** The complete ranges, resolution, 154-pair grid, validation partition, constrained objective, and tie-breakers are reported. The exact fold selections and complete surface accompany the supplement.

**Manuscript location:** Sections 3.2 and 4.3; Supplementary Table S12.

### R3.7 Value of $C$ and normalization by $\log C$

**Author Response:** $C=2$. Division by $\log 2$ maps Shannon entropy to $[0,1]$; for a binary posterior it is a monotone reparameterization of maximum-softmax confidence.

**Manuscript location:** Section 3.2.

### R3.8 Value of $\Delta$ and the first predecessor

**Author Response:** $\Delta=50$ ms. The stored 150-ms prediction supplies the predecessor for both class and entropy stability at 200 ms; there is no entropy-only exemption at the first candidate.

**Manuscript location:** Section 3.2.

### R3.9 Why candidates begin at 200 ms although 150 ms is evaluated

**Author Response:** The 150-ms prefix is a valid fixed-duration baseline but initializes temporal stability. Eligible adaptive outputs are 200, 250, and 300 ms, with forced fallback at 300 ms.

**Manuscript location:** Figure 1 and Section 3.2.

### R3.10 False-positive control samples in the corrected confusion matrix

**Author Response:** The corrected Transformer false-positive rate is 67.89%. Six of seven control speakers are predicted dysarthric on more than half their utterances. Detector-failure rows (91.87%) and sentence prompts (76.92%) have especially high rates, but overlapping speaker/channel structure prevents a unique causal claim.

**Manuscript location:** Section 4.4; Table 6; Figure 2; Supplementary Tables S15–S16.

## Requests not completed or not recoverable

The following are reported transparently as limitations: human reference-onset annotation and inter-annotator agreement; external cross-corpus evaluation; real mobile/conversational/far-field or array-microphone testing; historical CNN-GRU and EfficientNet-lite definitions; completed human qualitative error labels; and WavLM deployment profiling. None is represented by an inferred, simulated, or retrospectively invented result.
