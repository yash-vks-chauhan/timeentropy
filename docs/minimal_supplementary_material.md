# Supplementary Material

**Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy**

**Suresh Kumar Chelliah, Arivazhagan N\***

Department of Computational Intelligence, School of Computing, SRM Institute of Science and Technology, Kattankulathur-603203, Chennai, India.

This supplement contains the detailed architecture, partition, sensitivity, robustness, error, and runtime information referenced by the concise revised manuscript. Every classification result is generated from canonical out-of-fold prediction or decision ledgers. Percentages use all 8,131 utterances unless another denominator is stated.

## S1. Architecture specifications

**Table S1.** Compact-model input and output conventions.

| Item | Specification |
|---|---|
| Log-STFT input | $B\times1\times257\times T$; $T=16,21,26,31$ at 150, 200, 250, 300 ms |
| Convolution bias | Disabled when followed by batch normalization |
| Activation | SiLU in compact convolutional blocks; GELU in Transformer FFN |
| Dropout | 0.30 |
| Output | Two logits ordered control, dysarthric |
| Padding | Exact waveform boundary padding; sequence masks exclude padded positions |

**Table S2.** Complete small CNN architecture (23,538 trainable parameters).

| Stage | Exact operation | Output channels | Parameters |
|---|---|---:|---:|
| 1 | Conv2d 1→16, 3×3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2×2 | 16 | 176 |
| 2 | Conv2d 16→32, 3×3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2×2 | 32 | 4,672 |
| 3 | Conv2d 32→64, 3×3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2×2 | 64 | 18,560 |
| Pool | Length-aware mean over valid frequency/time positions | 64 | 0 |
| Head | Dropout 0.30; Linear 64→2 | 2 | 130 |

**Table S3.** Complete CNN–BiLSTM architecture (69,354 trainable parameters).

| Stage | Exact operation | Output shape | Parameters |
|---|---|---|---:|
| 1 | Conv2d 1→24, 3×3, padding 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×24×128×T | 264 |
| 2 | Conv2d 24→48, 3×3, padding 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×48×64×T | 10,464 |
| Tokens | Mean over 64 frequency bins; transpose | B×T×48 | 0 |
| Sequence | One-layer bidirectional LSTM; input 48; hidden 64 per direction | B×T×128 | 58,368 |
| Head | Masked temporal mean; Dropout 0.30; Linear 128→2 | 2 | 258 |

**Table S4.** Complete short-prefix Transformer architecture (196,594 trainable parameters).

| Stage | Exact operation | Output shape | Parameters |
|---|---|---|---:|
| 1 | Conv2d 1→24, 3×3, padding 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×24×128×T | 264 |
| 2 | Conv2d 24→48, 3×3, padding 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×48×64×T | 10,464 |
| Tokens | Adaptive frequency mean-pool to two bins; flatten 48×2; bias-free Linear 96→96; sinusoidal positions | B×T×96 | 9,216 |
| Encoder | Two post-norm layers; $d_{model}=96$; four heads; FFN 256; GELU; dropout 0.30 | B×T×96 | 174,272 |
| Head | Masked temporal mean; dropout; Linear 96→24; SiLU; dropout; Linear 24→2 | 2 | 2,378 |

Both sequence models preserve the time dimension. The CNN–BiLSTM forms one 48-dimensional token per valid STFT frame. The Transformer forms one 96-dimensional token by pooling frequency to two bins and flattening 48×2 values. Padded states are cleared and excluded from masked temporal means.

## S2. Dataset, folds, and training

**Table S5.** Training configuration frozen before primary test evaluation.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate / schedule | $1.5\times10^{-4}$ / constant |
| Weight decay | $5\times10^{-4}$ |
| Batch size | 64 |
| Maximum epochs / patience | 35 / 8 |
| Gradient clipping | 5.0 global norm |
| Dropout / label smoothing | 0.30 / 0.05 |
| Class balancing | Inverse-frequency weighted sampling with replacement |
| Training augmentation | Waveform gain −3 to +3 dB; additive Gaussian noise at 20–35 dB SNR ($p=0.5$); compact log-STFT masks; frequency mask up to 16 bins and time mask up to four frames ($p=0.5$) |
| Checkpoint metric | Speaker-balanced macro-F1 on checkpoint-selection validation rows |
| Primary / replication seeds | 20260819 / 20260820, 20260821 |

**Table S6.** Five speaker-independent fold assignments. Every speaker appears in exactly one test set and has only one role per fold.

| Fold | Train | Validation | Test | Test utterances |
|---:|---|---|---|---:|
| 0 | F01, F03, FC02, FC03, M03, M04, MC01 | F04, M02, MC02, MC03 | FC01, M01, M05, MC04 | 1,689 |
| 1 | F03, FC01, FC02, FC03, M01, M04, M05, MC04 | F01, M03, MC01 | F04, M02, MC02, MC03 | 1,940 |
| 2 | F03, F04, FC01, FC03, M01, M02, M05, MC02, MC03, MC04 | FC02, M04 | F01, M03, MC01 | 1,657 |
| 3 | F01, F04, FC01, M01, M02, M03, M05, MC01, MC02, MC03, MC04 | F03, FC03 | FC02, M04 | 1,288 |
| 4 | F01, F04, FC02, M02, M03, M04, MC01, MC02, MC03 | FC01, M01, M05, MC04 | F03, FC03 | 1,557 |

## S3. Onset diagnostics and onset-rule confound audit

**Table S7.** Label-free onset-detector behavior on 600 utterances. “Room-tone prefix” means a 300-ms crop RMS below 0.01; it is not a human onset-error label.

| Mode | Anchor at sample zero | Median anchor | Median crop RMS | Room-tone prefix | Post/pre energy |
|---|---:|---:|---:|---:|---:|
| `causal_fixed` | 71.7% | 0 ms | 0.0119 | 43.3% | +19.5 dB |
| **`causal_noise_floor`** | **4.7%** | 1,060 ms | **0.1713** | **3.5%** | **+25.3 dB** |
| `recovered_offline_peak` | 45.0% | 40 ms | 0.0149 | 27.5% | +15.2 dB |

**Table S8.** Condition-matched onset-rule retraining, pooled out of fold. Each condition has its own validation-only temperature and policy selection.

| Onset rule | Accuracy | Macro-F1 | Balanced accuracy | Dysarthric recall | Specificity | Mean time | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| `causal_noise_floor` (primary) | 41.95% | 41.95% | 45.74% | 59.37% | 32.11% | 269.39 ms | 64.25% |
| `causal_fixed` | 59.28% | 58.09% | 59.17% | 58.76% | 59.57% | 248.16 ms | 41.85% |
| `recovered_offline_peak` | 67.37% | 66.52% | 68.21% | 71.20% | 65.21% | 253.46 ms | 50.81% |

**Table S9.** Recording-start crop rates and location of comparator accuracy differences.

| Rule / contrast | Recording-start crops overall | Control | Dysarthric | Class gap | Δ accuracy within recording-start crops | Δ accuracy when anchor is after zero |
|---|---:|---:|---:|---:|---:|---:|
| Primary rule | 4.86% | 6.39% | 2.15% | 4.24 points | — | — |
| `causal_fixed` | 78.42% | 88.94% | 59.78% | 29.15 points | +26.00 points | −14.13 points |
| `recovered_offline_peak` | 55.54% | 72.73% | 25.09% | 47.65 points | +39.22 points | +8.24 points |

The comparator gain is concentrated in recording-start crops and primarily in control utterances. The table localizes the association; it does not identify which acoustic or session feature causes it.

## S4. Fold, pre-roll, threshold, and seed results

**Table S10.** Per-fold raw accuracy at fixed 300 ms.

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Range |
|---|---:|---:|---:|---:|---:|---:|
| Small CNN | 46.8% | 21.4% | 55.0% | 76.7% | 32.0% | 55.3 points |
| CNN–BiLSTM | 60.9% | 28.0% | 53.3% | 53.9% | 41.2% | 32.8 points |
| Transformer | 48.0% | 24.4% | 39.3% | 75.3% | 32.4% | 50.9 points |
| WavLM Base+ | 72.3% | 60.4% | 62.7% | 51.2% | 46.9% | 25.3 points |

**Table S11.** Condition-matched complete Transformer policy with 0-ms versus 30-ms pre-roll. Differences are 0 ms minus 30 ms with paired speaker-clustered 95% intervals.

| Metric | 0-ms pre-roll | 30-ms pre-roll | Difference, 95% interval |
|---|---:|---:|---:|
| Accuracy | 50.98% | 41.95% | +9.03 points [0.09, 19.79] |
| Macro-F1 | 50.29% | 41.95% | +8.35 points [−0.37, 19.95] |
| Dysarthric recall | 54.40% | 59.37% | −4.98 points [−11.90, 0.14] |
| Specificity | 49.05% | 32.11% | +16.93 points [3.42, 34.40] |
| Balanced accuracy | 51.72% | 45.74% | +5.98 points [−1.62, 15.23] |
| Mean time | 269.25 ms | 269.39 ms | −0.14 ms [−1.86, 1.06] |
| Fallback | 63.24% | 64.25% | −1.01 points [−2.73, 0.51] |

**Table S12.** Complete validation-only policy selection by fold. Search: $\theta=0.30$–0.95 by 0.05 and $\epsilon=0.00$–0.20 by 0.02.

| Fold | Fixed-300 validation macro-F1 | Macro-F1 range over 154 cells | Mean-time range | Selected $\theta$ | Selected $\epsilon$ | Selected macro-F1 | Selected mean time | Fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.787122 | 0.784278–0.789126 | 212.626–300.000 ms | 0.95 | 0.20 | 0.786278 | 212.626 ms | 0.068687 |
| 1 | 0.200000 | 0.200000–0.200000 | 300.000–300.000 ms | 0.30 | 0.00 | 0.200000 | 300.000 ms | 1.000000 |
| 2 | 0.655207 | 0.655207–0.664027 | 228.451–300.000 ms | 0.95 | 0.20 | 0.664027 | 228.451 ms | 0.168350 |
| 3 | 0.465923 | 0.465923–0.465923 | 300.000–300.000 ms | 0.30 | 0.00 | 0.465923 | 300.000 ms | 1.000000 |
| 4 | 0.538156 | 0.538156–0.538156 | 300.000–300.000 ms | 0.30 | 0.00 | 0.538156 | 300.000 ms | 1.000000 |

The exact 770-row validation surface is supplied as `artifacts/sensitivity/transformer_complete_calibrated/theta_epsilon_surface_exact.csv`. No test rows were used to generate or select from it.

**Table S13.** Complete calibrated Transformer policy under three training seeds. The interval is descriptive over three seed means and is not a speaker-generalization interval.

| Seed | Accuracy | Macro-F1 | Balanced accuracy | Dysarthric recall | Specificity | Mean time | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260819 | 41.95% | 41.95% | 45.74% | 59.37% | 32.11% | 269.39 ms | 64.25% |
| 20260820 | 48.41% | 48.03% | 49.89% | 55.21% | 44.56% | 269.79 ms | 62.55% |
| 20260821 | 46.76% | 46.34% | 48.03% | 52.59% | 43.47% | 270.72 ms | 64.11% |
| Mean ± SD | 45.71 ± 3.35% | 45.44 ± 3.14% | 47.89 ± 2.08% | 55.73 ± 3.42% | 40.05 ± 6.89% | 269.97 ± 0.68 ms | 63.64 ± 0.94% |

## S5. Robustness and error stratification

**Table S14.** Pooled test-time perturbation sensitivity. Clean-validation parameters are held fixed.

| Condition | Accuracy | Macro-F1 | Dysarthric recall | Accuracy change from clean |
|---|---:|---:|---:|---:|
| Clean reference | 41.95% | 41.95% | 59.37% | — |
| White noise, 20 dB SNR | 43.09% | 43.09% | 60.26% | +1.14 points |
| White noise, 10 dB SNR | 44.57% | 44.24% | 72.49% | +2.62 points |
| White noise, 0 dB SNR | 38.05% | 35.82% | 78.56% | −3.90 points |
| Anchor shift −50 ms | 35.84% | 33.60% | 75.12% | −6.11 points |
| Anchor shift −30 ms | 35.27% | 33.55% | 71.17% | −6.68 points |
| Anchor shift −10 ms | 38.89% | 38.61% | 63.26% | −3.06 points |
| Anchor shift 0 ms | 41.95% | 41.95% | 59.37% | 0.00 points |
| Anchor shift +10 ms | 43.44% | 42.39% | 56.00% | +1.49 points |
| Anchor shift +30 ms | 41.63% | 41.60% | 54.74% | −0.32 points |
| Anchor shift +50 ms | 40.97% | 40.96% | 55.11% | −0.98 points |

**Table S15.** Descriptive false-positive strata for the complete calibrated Transformer. Denominators contain actual-control utterances; strata overlap.

| Stratifier | Stratum | False positives / actual control | False-positive rate |
|---|---|---:|---:|
| Overall | All | 3,528 / 5,197 | 67.89% |
| Detector status | Detected | 3,223 / 4,865 | 66.25% |
| Detector status | Registered failure | 305 / 332 | 91.87% |
| Prompt | Isolated token | 2,416 / 3,735 | 64.69% |
| Prompt | Sentence | 1,050 / 1,365 | 76.92% |
| Prompt | Vocal exercise | 61 / 96 | 63.54% |
| Decision time | 200 ms | 910 / 1,236 | 73.62% |
| Decision time | 250 ms | 198 / 246 | 80.49% |
| Decision time | 300 ms | 2,420 / 3,715 | 65.14% |
| Duration | 1–<2 s | 1,243 / 1,895 | 65.59% |
| Duration | 2–<5 s | 2,050 / 2,981 | 68.77% |
| Duration | ≥5 s | 229 / 309 | 74.11% |

**Table S16.** Per-speaker behavior under the complete calibrated Transformer policy.

| Speaker | Class | Utterances | Accuracy | Predicted dysarthric |
|---|---|---:|---:|---:|
| MC02 | Control | 464 | 1.7% | 98.3% |
| MC03 | Control | 826 | 8.1% | 91.9% |
| MC04 | Control | 650 | 20.9% | 79.1% |
| F04 | Dysarthric | 246 | 24.0% | 24.0% |
| FC01 | Control | 163 | 24.5% | 75.5% |
| F03 | Dysarthric | 565 | 28.0% | 28.0% |
| MC01 | Control | 1,108 | 31.7% | 68.3% |
| FC03 | Control | 992 | 35.0% | 65.0% |
| M03 | Dysarthric | 417 | 43.6% | 43.6% |
| M05 | Dysarthric | 490 | 58.2% | 58.2% |
| FC02 | Control | 994 | 72.4% | 27.6% |
| M02 | Dysarthric | 404 | 84.2% | 84.2% |
| M04 | Dysarthric | 294 | 85.0% | 85.0% |
| M01 | Dysarthric | 386 | 90.2% | 90.2% |
| F01 | Dysarthric | 132 | 90.9% | 90.9% |

**Table S17.** Dysarthric recall by published speaker-level severity. Categories follow the external speaker mapping; severity is constant within speaker.

| Published severity | Speakers | Dysarthric utterances | Correctly predicted dysarthric | Recall |
|---|---|---:|---:|---:|
| Severe | F01, M01, M02, M04 | 1,216 | 1,058 | 87.01% |
| Severe–moderate | M05 | 490 | 285 | 58.16% |
| Moderate | F03 | 565 | 158 | 27.96% |
| Mild | F04, M03 | 663 | 241 | 36.35% |
| All | Eight dysarthric speakers | 2,934 | 1,742 | 59.37% |

Severity-stratified values are descriptive because each category contains only one to four speakers and cannot separate severity from speaker identity.

## S6. Runtime accounting

**Table S18.** Measured compute for one 200-ms Transformer decision after warm-up, 100 repetitions. RMS computation is algorithm execution time, not acoustic confirmation delay.

| Component | CPU | Apple MPS |
|---|---:|---:|
| RMS detector computation | 0.079 ms | 0.089 ms |
| Prefix crop | 0.008 ms | 0.011 ms |
| STFT and standardization | 0.342 ms | 0.360 ms |
| Input tensor preparation | 0.015 ms | 0.580 ms |
| Model forward | 3.003 ms | 5.921 ms |
| Logit to probability | 0.038 ms | 0.584 ms |
| Policy computation | 0.119 ms | 0.140 ms |
| Timing-accounting overhead | 0.035 ms | 0.046 ms |
| **Total measured compute** | **3.629 ms** | **7.732 ms** |
| Post-anchor acoustic observation | 200 ms | 200 ms |
| **Observation plus measured compute** | **203.629 ms** | **207.732 ms** |

The primary onset threshold is available by 110 ms from recording start for 8,127 of 8,131 files, and two-frame confirmation is available 30 ms after the returned anchor. The 200-ms candidate occurs after both bounds for a nonnegative anchor. Nevertheless, 200 ms denotes post-anchor observation, not elapsed time from recording-session start. The 30-ms pre-roll is buffered history and adds no future wait.

## S7. Consolidated statistical and policy validation

**Table S19.** Complete temperature-scaled Transformer fold statistics and pooled speaker-clustered intervals. Panel A reports each held-out fold and the unweighted five-fold mean ± sample SD. Panel B reports pooled out-of-fold estimates with 95% percentile speaker-clustered bootstrap intervals from 10,000 replicates over the 15 speakers, stratified by class.

**Panel A. Five-fold results and unweighted mean ± sample SD.**

| Fold | $n$ | Accuracy | Macro-F1 | Dysarthric recall | Mean observation | Fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,689 | 47.90% | 43.78% | 72.26% | 226.11 ms | 10.95% |
| 1 | 1,940 | 24.43% | 22.26% | 61.38% | 300.00 ms | 100.00% |
| 2 | 1,657 | 39.41% | 39.36% | 55.01% | 225.11 ms | 15.33% |
| 3 | 1,288 | 75.31% | 71.52% | 85.03% | 300.00 ms | 100.00% |
| 4 | 1,557 | 32.43% | 31.42% | 27.96% | 300.00 ms | 100.00% |
| Mean±SD | — | 43.90 ± 19.58% | 41.67 ± 18.59% | 60.33 ± 21.39% | 270.24 ± 40.75 ms | 65.26 ± 47.60% |

**Panel B. Pooled estimates and 95% intervals.**

| Metric | Pooled estimate | Speaker-clustered 95% interval |
|---|---:|---:|
| Accuracy | 41.95% | 30.42–55.19% |
| Macro-F1 | 41.95% | 29.49–54.73% |
| Dysarthric recall | 59.37% | 42.13–79.10% |
| Specificity | 32.11% | 14.47–50.52% |
| Dysarthric precision | 33.06% | 23.75–44.72% |
| Balanced accuracy | 45.74% | 33.21–58.91% |
| Mean observation | 269.39 ms | 247.57–288.77 ms |
| Forced fallback | 64.25% | 39.20–86.65% |

The wide intervals show that pooled utterance-level precision does not remove the uncertainty caused by having only 15 independent speaker clusters.

**Table S20.** Transformer policy ablation and standardized accuracy–earliness comparison, pooled out of fold. $H$ is the harmonic mean of accuracy and timeliness, where timeliness is $1-\text{mean observation}/300$ ms.

| Policy | Accuracy | Macro-F1 | Dysarthric recall | Mean observation | Fallback | Accuracy–timeliness $H$ |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 150 ms | 43.94% | 43.89% | 56.78% | 150.00 ms | 0.00% | 0.468 |
| Fixed 200 ms | 43.07% | 43.04% | 56.65% | 200.00 ms | 0.00% | 0.376 |
| Fixed 250 ms | 42.11% | 42.11% | 57.91% | 250.00 ms | 0.00% | 0.239 |
| Fixed 300 ms | 41.94% | 41.93% | 60.05% | 300.00 ms | 0.00% | 0.000 |
| Entropy only, raw | 42.33% | 42.33% | 57.33% | 209.06 ms | 4.22% | 0.353 |
| Maximum softmax, raw | 42.33% | 42.33% | 57.33% | 209.06 ms | 4.22% | 0.353 |
| Entropy + class stability, raw | 42.28% | 42.28% | 57.53% | 212.90 ms | 5.28% | 0.344 |
| Entropy + entropy stability, raw | 41.99% | 41.99% | 58.15% | 221.44 ms | 7.32% | 0.323 |
| Complete policy, raw | 41.99% | 41.99% | 58.35% | 224.38 ms | 8.88% | 0.315 |
| Complete policy, temperature-scaled | 41.95% | 41.95% | 59.37% | 269.39 ms | 64.25% | 0.164 |

Entropy-only and maximum-softmax are identical; the stability conditions increase observation time without improving accuracy or the fixed-duration frontier.

## S8. Items not completed or recoverable

The following remain uncompleted: human reference-onset annotation and inter-annotator agreement; cross-corpus evaluation; real mobile, conversational, array-microphone, and far-field tests; recovery of historical CNN-GRU and EfficientNet-lite definitions; human qualitative error labels; and WavLM deployment runtime, FLOPs, and memory. No missing result is inferred, simulated, or retrospectively invented.
