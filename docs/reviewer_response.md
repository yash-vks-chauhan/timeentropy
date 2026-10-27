# Point-by-point response to reviewers

**Paper ID:** _to be completed by the corresponding author from the editorial system_

**Paper title:** Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy: A Reproducibility Study

**Authors:** Suresh Kumar Chelliah, Arivazhagan N

We thank the organizers and all three reviewers for their careful evaluation. The reviews
identified missing implementation detail, incomplete uncertainty reporting, and a mathematical
inconsistency between the submitted confusion matrix and reported metrics. Those comments led us
to reconstruct the experiment from the surviving corpus and metadata rather than defend results
that could not be regenerated.

The original training code, checkpoints, and per-utterance prediction files had been deleted.
Accordingly, the submitted numerical results have been withdrawn. The revised manuscript reports
a version-controlled reconstruction with saved per-prefix predictions, speaker-independent folds,
validation-only calibration and policy selection, and explicit provenance. Classification tables
within a system are generated from that system's canonical fixed-prefix or decision ledger;
onset diagnostics, robustness tests, profiling, pre-roll retraining, and error stratification use
their own separately hashed artifacts. We do not claim that every result comes from one file.

The reconstruction materially changes the paper's conclusion. None of the three compact models
exceeds chance reliably on held-out speakers, and the calibrated adaptive Transformer does not
improve the fixed-duration accuracy-earliness frontier. A frozen WavLM Base+ reference performs
better directionally, although its speaker-clustered confidence interval still includes chance.
The condition-matched onset-rule retraining requested by the reviewers produced a result we did
not expect and report against our own framing: the two rules our waveform diagnostic rejected
classify *better*, and the entire advantage sits in crops that begin at the recording start,
which those rules produce at sharply class-dependent rates. Detector failure is itself a
class-correlated signal here. The paper is therefore framed as a reproducibility study and a
report of negative as well as positive findings.

The reviewers also requested validation of the RMS onset detector. No manual reference-onset
annotation has been completed. We therefore report only a label-free waveform diagnostic and
imposed anchor-shift sensitivity. The diagnostic shows sample-zero anchors and low-energy crops;
these are warning indicators, not ground-truth localization-error rates. Likewise, a deterministic
40-example blinded error-review set was generated, but its human-review fields remain blank. We
report reproducible quantitative strata and do not claim a completed qualitative audit or a
human-assigned cause of the errors.

Claims are limited to the evaluated TORGO head-mounted-microphone subset, the stated speaker
folds, and the stated utterance-level binary classification task. Cross-corpus, genuine
far-field/mobile, spontaneous conversational, severity-prediction, and diagnostic claims are not
made. We answer all twenty Reviewer-1 comments, the nine Reviewer-1 experimental-validation
requirements, all twelve Reviewer-2 comments, all ten Reviewer-3 comments, and the three minor
comments. Where reviewers raised the same point we answer it once and cross-reference it. Every
answer ends with a **Manuscript location** line giving the revised manuscript's final section,
table, and figure numbers for that change. Page and line numbers are not cited because
they depend on the journal's final typesetting; no equation-number, algorithm, or appendix
reference is asserted, because the revised manuscript contains none.

## Reviewer-1 Comments with Author Response

**Reviewer Concern (R1.1):** The proposed entropy-based framework is interesting, but the novelty
should be more clearly distinguished from conventional confidence-based early classification and
entropy-based rejection methods.

**Author Response:** We agree and have narrowed the novelty claim. For two classes, normalized
posterior entropy is a strictly monotone decreasing transform of maximum-softmax confidence.
Consequently, an entropy threshold and a corresponding maximum-softmax threshold implement the
same gate. The revised manuscript no longer presents normalized entropy as a new uncertainty
statistic.

The object evaluated is instead a sequential stopping policy: temperature scaling is fitted on a
held-out validation calibration partition; predictions at adjacent prefixes are compared for
class and entropy stability; and fold-specific `theta` and `epsilon` are selected on policy-tuning
validation rows by minimizing observation time subject to macro-F1 remaining within 0.01 absolute
of the fold's fixed-300-ms value. Test speakers are evaluated once at those selected values.

The ablation directly includes maximum-softmax confidence:

| Policy | Accuracy | Macro-F1 | Mean observation | Forced fallback |
|---|---:|---:|---:|---:|
| Fixed 200 ms | 43.07% | 43.04% | 200.0 ms | 0% |
| Fixed 250 ms | 42.11% | 42.11% | 250.0 ms | 0% |
| Fixed 300 ms | 41.94% | 41.93% | 300.0 ms | 0% |
| Entropy only, raw | 42.33% | 42.33% | 209.1 ms | 4.2% |
| Maximum-softmax confidence, raw | 42.33% | 42.33% | 209.1 ms | 4.2% |
| Entropy plus class stability, raw | 42.28% | 42.28% | 212.9 ms | 5.3% |
| Entropy plus entropy stability, raw | 41.99% | 41.99% | 221.4 ms | 7.3% |
| Complete policy, raw | 41.99% | 41.99% | 224.4 ms | 8.9% |
| Complete policy, temperature-scaled | 41.95% | 41.95% | 269.4 ms | 64.2% |

The entropy-only and maximum-softmax rows match exactly, and the stability conditions add latency
without an accuracy gain. The revised Related Work section now places this result within early
time-series classification, uncertainty/rejection, posterior calibration, and short-segment
pathological-speech research.

**Manuscript location:** §1.2, §2.3, §3.3, §3.4, and Table 23.

**Reviewer Concern (R1.2):** The exact architecture and parameter configuration of the CNN,
CNN-LSTM, CNN-GRU, EfficientNet-lite, and Transformer models should be provided for
reproducibility.

**Author Response:** We provide the recoverable architectures and programmatically verified
parameter counts below and in the Model Architectures subsection. All compact models receive a
log-magnitude STFT tensor of shape `B x 1 x 257 x T` and emit two logits ordered control and
dysarthric. A convolution is bias-free when followed by batch normalization.

| Model/stage | Exact operation | Output | Parameters |
|---|---|---|---:|
| Small CNN 1 | Conv2d 1->16, 3x3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2x2 | 16 channels | 176 |
| Small CNN 2 | Conv2d 16->32, 3x3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2x2 | 32 channels | 4,672 |
| Small CNN 3 | Conv2d 32->64, 3x3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2x2 | 64 channels | 18,560 |
| Small CNN head | Length-aware mean over valid positions; Dropout 0.30; Linear 64->2 | 2 logits | 130 |
| CNN-BiLSTM front end | Conv2d 1->24 then 24->48, 3x3, padding 1; BatchNorm2d; SiLU; MaxPool 2x1 after each layer | `B x 48 x 64 x T` | 10,728 |
| CNN-BiLSTM sequence/head | Mean over frequency; one-layer bidirectional LSTM, input 48, hidden 64/direction; masked temporal mean; Dropout 0.30; Linear 128->2 | `B x T x 128`, then 2 logits | 58,626 |
| Transformer front end | Same two convolutional stages; adaptive frequency mean-pool to 2 bins; flatten 48x2; bias-free Linear 96->96; sinusoidal positions | `B x T x 96` | 19,944 |
| Transformer encoder/head | Two post-norm encoder layers, `d_model=96`, 4 heads, FFN 256, GELU, dropout 0.30; masked temporal mean; Linear 96->24; SiLU; dropout; Linear 24->2 | 2 logits | 176,650 |

Totals are 23,538 parameters for the small CNN, 69,354 for the CNN-BiLSTM, and 196,594 for the
Transformer. The frozen WavLM Base+ reference has 94,381,936 frozen backbone parameters and a
1,538-parameter trainable linear head, for 94,383,474 total parameters.

The exact CNN-GRU and EfficientNet-lite definitions cannot be recovered from the surviving
materials. We therefore withdraw those model names and their historical results rather than
invent variants. This disposition is also addressed at R2.7 and R3.4.

**Manuscript location:** §3, §3.5, Tables 3–5, and Figure 1.

**Reviewer Concern (R1.3):** The manuscript should report the exact hyperparameter values, search
ranges and tuning procedure used for the entropy threshold θ and stability tolerance ε.

**Author Response:** The complete search is now specified and visualized. For each outer fold,
`theta = 0.30, 0.35, ..., 0.95` and `epsilon = 0.00, 0.02, ..., 0.20`, giving 154 pairs. Values
are selected only on that fold's policy-tuning validation partition after temperature fitting on
a separate calibration partition. A pair is feasible when validation macro-F1 is at least the
same fold's fixed-300-ms macro-F1 minus 0.01. The primary objective is minimum mean observation
time; tie-breakers are higher macro-F1, lower fallback, lower `theta`, then lower `epsilon`.

| Fold | Fixed-300 validation macro-F1 | Range over 154 cells | Observation range | Selected `theta` | Selected `epsilon` | Selected macro-F1 | Selected mean | Fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.787122 | 0.784278-0.789126 | 212.626-300 ms | 0.95 | 0.20 | 0.786278 | 212.626 ms | 6.87% |
| 1 | 0.200000 | 0.200000-0.200000 | 300-300 ms | 0.30 | 0.00 | 0.200000 | 300 ms | 100% |
| 2 | 0.655207 | 0.655207-0.664027 | 228.451-300 ms | 0.95 | 0.20 | 0.664027 | 228.451 ms | 16.84% |
| 3 | 0.465923 | 0.465923-0.465923 | 300-300 ms | 0.30 | 0.00 | 0.465923 | 300 ms | 100% |
| 4 | 0.538156 | 0.538156-0.538156 | 300-300 ms | 0.30 | 0.00 | 0.538156 | 300 ms | 100% |

All 154 cells met the macro-F1 constraint in every fold. The surface is mostly degenerate: folds
1, 3, and 4 remain at 300 ms with 100% fallback throughout the grid, whereas folds 0 and 2 show
an earliness response and select the most permissive searched corner. This is evidence of strong
fold dependence, not a sharply identified universal threshold. The full 770-row validation table
and a five-panel surface are supplied; test data were not swept or retuned.

**Manuscript location:** §3.4, §4.3, Tables 9, 11 and 22, and Figure 7.

**Reviewer Concern (R1.4):** The speech-onset detection procedure requires quantitative
validation because errors in RMS-based onset localization can directly affect the claimed
200-300 ms latency.

**Author Response:** We agree that ground-truth onset validation is required for a definitive
localization claim. It has not been completed: the prespecified 300-row annotation sheet contains
no manual reference onsets. We have therefore removed claims of measured onset-localization error
and report two narrower analyses.

First, a label-free diagnostic samples 40 utterances per speaker, 600 total, and uses waveform
statistics only. The implementation identifiers and results are:

| Mode | Fires at sample zero | Median anchor | Median post-anchor RMS | 300-ms crop below RMS 0.01 |
|---|---:|---:|---:|---:|
| `causal_fixed` | 71.7% | 0 ms | 0.0119 | 43.3% |
| `causal_noise_floor` initial-level adaptive rule | 4.7% | 1,060 ms | 0.1713 | 3.5% |
| `recovered_offline_peak` earlier whole-utterance rule | 45.0% | 40 ms | 0.0149 | 27.5% |

These sample-zero and low-energy-crop rates are warning indicators; they are not false-onset rates
against human references. The earlier whole-utterance peak rule is non-causal because it requires
the utterance maximum. The initial-level adaptive rule uses only a bounded initial signal window
and is used for the reported classification experiment. Its historical implementation ID is
`causal_noise_floor`, but the first ten overlapping frames can contain speech and are not assumed
to be pure noise.

Second, the complete calibrated policy is evaluated after imposed anchor shifts of -50, -30,
-10, 0, +10, +30, and +50 ms. Accuracy is 35.84%, 35.27%, 38.89%, 41.95%, 43.44%, 41.63%, and
40.97%, respectively. These perturbations quantify sensitivity to controlled displacement; they
do not measure the detector's real error distribution.

Condition-matched retraining under the fixed and earlier peak-relative onset modes is now
complete. Each comparator was retrained on all five folds under configurations
`causal_fixed_preroll30` and `offline_peak_preroll30`, with its own temperature and its own
validation-selected policy, then evaluated on the identical 8,131 out-of-fold utterances:

| Onset rule | Accuracy | Macro-F1 | Balanced accuracy | Dysarthric recall | Specificity | Fallback |
|---|---:|---:|---:|---:|---:|---:|
| `causal_noise_floor` (primary) | 41.95% | 41.95% | 45.74% | 59.37% | 32.11% | 64.25% |
| `causal_fixed` | 59.28% | 58.09% | 59.17% | 58.76% | 59.57% | 41.85% |
| `recovered_offline_peak` | 67.37% | 66.52% | 68.21% | 71.20% | 65.21% | 50.81% |

Paired speaker-clustered accuracy differences are +17.33 points [+3.85, +28.97] for
`causal_fixed` and +25.42 points [+12.42, +37.14] for `recovered_offline_peak`; both exclude
zero. We report this against our own earlier framing: the waveform diagnostic identified these
rules as unreliable, and on classification accuracy they are better.

They are better for a reason that does not favour them. Stratifying the difference by what the
detector did shows the entire advantage sitting in crops that begin at the recording start.
Where `causal_fixed` anchors after sample zero it is 14.13 points *worse* than the primary rule.
Within the recording-start stratum the gain falls on control utterances (+33.02 points for
`causal_fixed`, +47.08 for `recovered_offline_peak`) and not on dysarthric ones (+7.49 and
−1.23). The cause is that the comparators reach the recording start at very different rates by
class: an 88.94%/59.78% control/dysarthric split for `causal_fixed` and 72.73%/25.09% for
`recovered_offline_peak`, against 6.39%/2.15% for the primary rule. A detector that fails
differentially by class supplies a shortcut, and "this looks like the start of a recording"
becomes a proxy for "control".

We therefore retain `causal_noise_floor`. It was fixed from waveform diagnostics recorded in the
frozen configuration before any test evaluation, and switching the primary rule on the strength
of these test numbers would be selection on the test set — the practice this reconstruction
exists to correct. The comparators are reported as the prespecified ablation they are. The
stratification shows where the difference lives; it does not identify what inside those crops
carries the signal, and we make no causal claim. Full results are in revised manuscript §5.1, Tables 12–15.

Manual reference-onset annotation remains an uncompleted human task and is listed explicitly at
the end of this response.

**Manuscript location:** §3.1, §5.1, §6, Tables 2 and 11–15, and Figure 2.

**Reviewer Concern (R1.5):** The 30 ms pre-roll introduces acoustic information before the
nominal speech onset, so the definition of "200 ms after speech onset" should explicitly account
for this additional context.

**Author Response:** We now define a nominal `t`-ms prefix as exactly `t` ms of post-anchor audio
preceded by at most 30 ms of already observed buffered context. Thus a nominal 200-ms input can
span at most 230 ms of samples but still contains the full 200 ms after the returned anchor.
Pre-roll does not require future waiting, although onset-threshold availability and confirmation
do impose their own signal-availability conditions, addressed at R1.15 and R2.4.

The 30-ms value was a bounded-context design choice, not a value selected from held-out
performance. The completed condition-matched retraining shows that it is not empirically
superior. With identical 8,131 held-out utterances, the 0-ms condition versus the 30-ms condition
gave:

| Metric | 0 ms | 30 ms | Difference, 0 minus 30 ms [speaker-clustered 95% CI] |
|---|---:|---:|---:|
| Accuracy | 50.98% | 41.95% | +9.03 points [ +0.09, +19.79 ] |
| Macro-F1 | 50.29% | 41.95% | +8.35 points [ -0.37, +19.95 ] |
| Balanced accuracy | 51.72% | 45.74% | +5.98 points [ -1.62, +15.23 ] |
| Dysarthric recall | 54.40% | 59.37% | -4.98 points [ -11.90, +0.14 ] |
| Specificity | 49.05% | 32.11% | +16.93 points [ +3.42, +34.40 ] |
| Mean observation | 269.253 ms | 269.389 ms | -0.135 ms [ -1.862, +1.063 ] |
| Forced fallback | 63.24% | 64.25% | -1.01 points [ -2.73, +0.51 ] |

The primary paired speaker-cluster analysis therefore shows higher raw accuracy and specificity
without pre-roll, while macro-F1, balanced accuracy, recall, and fallback intervals include zero.
The exact utterance-level McNemar result is also reported only as a descriptive sensitivity check
because utterances from one speaker are not independent. We have removed the earlier "no pre-roll
effect" statement and any claim that 30 ms captures proven discriminative articulatory cues.

**Manuscript location:** §3.1, §3.2, §5.4, and Tables 26–27.

**Reviewer Concern (R1.6):** The manuscript should clarify whether the five-fold results are
pooled across all test speakers or averaged across folds, and report fold-wise performance rather
than only aggregate values.

**Author Response:** We now distinguish two estimands throughout. “Pooled out-of-fold” means one
metric computed after concatenating the five disjoint test-fold prediction sets, so each of the
8,131 utterances contributes once. “Fold mean ± SD” is the unweighted mean and sample standard
deviation across the five fold-level metrics. Unequal test-fold sizes make these estimands
different; neither is presented as the other.

For the complete calibrated Transformer policy, the fold-level results are:

| Fold | Test n | Accuracy | Macro-F1 | Dysarthric recall | Balanced accuracy | Mean observation | Forced fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,689 | 47.90% | 43.78% | 72.26% | 46.95% | 226.11 ms | 10.95% |
| 1 | 1,940 | 24.43% | 22.26% | 61.38% | 33.60% | 300.00 ms | 100.00% |
| 2 | 1,657 | 39.41% | 39.36% | 55.01% | 43.34% | 225.11 ms | 15.33% |
| 3 | 1,288 | 75.31% | 71.52% | 85.03% | 78.73% | 300.00 ms | 100.00% |
| 4 | 1,557 | 32.43% | 31.42% | 27.96% | 31.47% | 300.00 ms | 100.00% |
| **Unweighted mean ± SD** | — | **43.90 ± 19.58%** | **41.67 ± 18.59%** | **60.33 ± 21.39%** | **46.82 ± 18.98%** | **270.24 ± 40.75 ms** | **65.26 ± 47.60%** |

The corresponding pooled accuracy and macro-F1 are both 41.95%, mean observation is 269.39 ms,
and forced fallback is 64.25%. The large between-fold variation is now treated as a central
result rather than hidden by an aggregate.

**Manuscript location:** §4.4, §5.2, §5.4, Tables 17–18 and 23–24, and Figure 4.

**Reviewer Concern (R1.7):** The dataset contains only 15 speakers, so speaker-independent
performance should be accompanied by fold-wise variation and confidence intervals.

**Author Response:** We agree. In addition to the fold table above, we report pooled
speaker-cluster bootstrap intervals obtained by resampling the 15 speakers, not individual
utterances. These intervals describe the pooled out-of-fold estimand and are not mislabeled as
“fold confidence intervals.”

| Metric | Pooled estimate | Speaker-clustered 95% CI |
|---|---:|---:|
| Accuracy | 41.95% | 30.42–55.19% |
| Macro-F1 | 41.95% | 29.49–54.73% |
| Dysarthric recall | 59.37% | 42.13–79.10% |
| Specificity | 32.11% | 14.47–50.52% |
| Balanced accuracy | 45.74% | 33.21–58.91% |
| Mean observation | 269.39 ms | 247.6–288.8 ms |
| Forced fallback | 64.25% | 39.20–86.65% |

The intervals are wide and balanced accuracy includes 50%. The revision therefore avoids a
claim of robust population-level discrimination from this 15-speaker experiment.

**Manuscript location:** §4.2, §4.4, §5.6, and Tables 8, 24 and 32.

**Reviewer Concern (R1.8):** Class-balanced sampling is used during training, but the manuscript
should report the original and effective class distributions and explain precisely how balancing
is implemented.

**Author Response:** The full selected corpus contains 5,197 control utterances from seven
speakers (63.92%) and 2,934 dysarthric utterances from eight speakers (36.08%), a control-to-
dysarthric utterance ratio of 1.771:1. Training alone uses inverse-class-frequency weights in a
sampler with replacement. Each epoch draws as many rows as the original training partition; the
expected class draw is 50/50. Validation and test partitions retain their natural distributions,
and augmentation is training-only.

The primary Transformer run logs the following realized sampling:

| Fold | Original training control/dysarthric | Draws per epoch | Mean realized control/dysarthric draws | Logged epochs |
|---:|---:|---:|---:|---:|
| 0 | 3,094 / 1,408 | 4,502 | 2,252.2 / 2,249.8 | 15 |
| 1 | 2,799 / 1,735 | 4,534 | 2,265.1 / 2,268.9 | 19 |
| 2 | 3,095 / 2,091 | 5,186 | 2,583.9 / 2,602.1 | 35 |
| 3 | 3,211 / 2,075 | 5,286 | 2,643.1 / 2,642.9 | 24 |
| 4 | 3,392 / 1,493 | 4,885 | 2,443.7 / 2,441.3 | 15 |

Because sampling is with replacement, these are draws, not necessarily distinct utterances. This
wording replaces the ambiguous claim that the data themselves were permanently rebalanced.

**Manuscript location:** §4.1, §4.3, and Tables 6 and 9–10.

**Reviewer Concern (R1.9):** The comparison with existing dysarthria-detection methods should use
comparable speaker-independent protocols and clearly identify differences in datasets,
microphones, utterance duration and evaluation metrics.

**Author Response:** We replaced the narrative ranking with a structured comparison covering the
corpus and participant count, microphone, input unit or duration, speaker-separation protocol,
reported metrics, and whether latency is defined. It includes representative TORGO and UA-Speech
studies from 2019–2025 and the present compact-model and frozen-WavLM experiments. The table marks
protocol differences explicitly; none of the cited accuracy values is treated as a direct
head-to-head comparison where corpus, channel, duration, or partitioning differs. The present
estimand is now stated narrowly as utterance-level classification on onset-aligned, head-mounted-
microphone TORGO prefixes under the specified speaker-disjoint folds.

**Manuscript location:** §2.4 and Table 1.

**Reviewer Concern (R1.10):** The paper should include a stronger baseline using a modern
pretrained or self-supervised speech representation.

**Author Response:** We added a completed frozen WavLM Base+ reference using checkpoint revision
4c66d4806a428f2e922ccfa1a962776e232d487b. The frozen backbone has 94,381,936 parameters and the
768-to-2 classification head has 1,538 trainable parameters, for 94,383,474 parameters in total.
It uses raw mono 16-kHz audio, an exact sample-level attention mask, valid-position mean pooling,
and the same five speaker folds and fixed prefix durations. Only the frozen-backbone
configuration was evaluated; WavLM fine-tuning was not performed.

| Prefix | Accuracy | Macro-F1 | Balanced accuracy | Dysarthric recall |
|---:|---:|---:|---:|---:|
| 150 ms | 52.50% | 52.45% | 55.87% | 67.96% |
| 200 ms | 55.36% | 54.99% | 57.29% | 64.25% |
| 250 ms | 56.95% | 56.40% | 58.32% | 63.22% |
| 300 ms | 59.29% | 58.10% | 59.18% | 58.79% |

The 300-ms speaker-clustered balanced-accuracy interval is 47.9–70.7%, so it includes chance.
These results are reported as a stronger directional baseline, not proof of superiority or a
mechanism such as speaker memorization. WavLM runtime and operation counts were not profiled and
are not inferred from the compact-model profiler.

**Manuscript location:** §3.6, §5.2, §5.3, and Tables 16 and 20.

**Reviewer Concern (R1.11):** The latency-accuracy analysis should report an explicit
area-under-the-accuracy-earliness curve or another standardized early-classification metric
instead of relying primarily on mean latency.

**Author Response:** We report the entire fixed-prefix curve together with consumed fraction
(mean observation divided by 300 ms), timeliness (one minus consumed fraction), and prespecified
harmonic means combining either accuracy or macro-F1 with timeliness. For the compact
Transformer:

| Operating point | Accuracy | Macro-F1 | Mean observation | Consumed fraction | Timeliness | Harmonic mean, accuracy/timeliness | Harmonic mean, macro-F1/timeliness |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed 150 ms | 43.94% | 43.89% | 150.0 ms | 0.500 | 0.500 | 0.4678 | 0.4675 |
| Fixed 200 ms | 43.07% | 43.04% | 200.0 ms | 0.667 | 0.333 | 0.3758 | 0.3757 |
| Fixed 250 ms | 42.11% | 42.11% | 250.0 ms | 0.833 | 0.167 | 0.2388 | 0.2388 |
| Fixed 300 ms | 41.94% | 41.93% | 300.0 ms | 1.000 | 0.000 | 0.0000 | 0.0000 |
| Complete adaptive policy | 41.95% | 41.95% | 269.39 ms | 0.898 | 0.102 | 0.1641 | 0.1641 |

Fixed 150 ms is the best standardized operating point among those evaluated. It is a valid
fixed-prefix operating point, while the adaptive candidate set remains 200, 250, and 300 ms for
the predecessor-based rule explained at R3.8–R3.9. The adaptive row combines the independently
selected fold-specific policies; it is not generated from one averaged θ/ε pair.

**Manuscript location:** §5.4, Table 21, and Figures 3 and 6.

**Reviewer Concern (R1.12):** The claim that the entropy criterion improves sensitivity should
be supported with statistical testing.

**Author Response:** We withdraw the blanket claim that entropy improves sensitivity. Paired
speaker-cluster bootstrap intervals are the primary inference. Exact McNemar tests are shown only
as descriptive, unclustered utterance-level sensitivity analyses because multiple utterances from
one speaker are dependent.

| Adaptive minus baseline | Δ accuracy [95% CI] | Δ dysarthric recall [95% CI] | Δ mean observation [95% CI] | Descriptive McNemar discordant pairs; p |
|---|---:|---:|---:|---:|
| Fixed 200 ms | -1.12 points [-3.67, +0.95] | +2.73 points [+0.59, +4.69] | +69.39 ms [+47.75, +88.82] | 321 adaptive-only correct / 412 fixed-only correct; 0.000875 |
| Fixed 250 ms | -0.16 points [-1.93, +1.17] | +1.47 points [-0.20, +2.96] | +19.39 ms [-2.29, +39.10] | 181 / 194; 0.536 |
| Fixed 300 ms | +0.01 points [-0.41, +0.44] | -0.68 points [-1.54, -0.03] | -30.61 ms [-51.94, -10.91] | 51 / 50; 1.000 |

Thus the policy has higher dysarthric recall than fixed 200 ms in the clustered interval, but is
substantially slower and has no supported accuracy gain. Against fixed 300 ms it saves observation
time, has no supported accuracy gain, and has slightly lower recall. The evidence is a trade-off,
not a general sensitivity improvement.

**Manuscript location:** §5.4 and Tables 22–23.

**Reviewer Concern (R1.13):** The manuscript should evaluate posterior calibration before using
entropy as a confidence measure.

**Author Response:** We added validation-only scalar temperature scaling and report negative
log-likelihood, Brier score, and 15-bin expected calibration error at the 300-ms Transformer
prefix:

| Posterior | NLL | Brier score | ECE |
|---|---:|---:|---:|
| Raw softmax | 1.393 | 0.446 | 0.429 |
| Temperature-scaled | 0.837 | 0.303 | 0.212 |

Fold temperatures are 1.159, 20.000, 2.196, 20.000, and 12.814 (mean 11.23 ± 9.21). Calibration
improves these probability-quality measures but raises the policy forced-fallback rate from 8.9%
to 64.2%; it does not improve policy accuracy. We therefore do not equate improved calibration
with improved classification, and we use the calibrated policy as the conservative primary
analysis. The frozen WavLM reference has raw ECE about 0.0445 and temperature-scaled ECE about
0.0593; temperature scaling is not assumed to help every model.

**Manuscript location:** §3.7, §5.3, Tables 19–20, and Figure 5.

**Reviewer Concern (R1.14):** The generalization claim is currently limited to TORGO;
cross-corpus validation using UA-Speech or another independent dysarthric-speech corpus is
required to support broader applicability.

**Author Response:** This new-data experiment was not completed. We do not substitute synthetic
noise or another within-TORGO split for cross-corpus evidence. All claims are therefore narrowed
to the selected head-mounted-microphone TORGO utterances and the stated speaker-disjoint folds.
UA-Speech or another independently acquired corpus remains necessary before making a
cross-corpus generalization claim and is listed among the incomplete requests at the end of this
response.

**Manuscript location:** §6 (listed as not completed).

**Reviewer Concern (R1.15):** The practical latency should distinguish model inference time from
the acoustic observation time, preprocessing time, STFT computation and speech-onset detection
time.

**Author Response:** We now separate signal availability from computation. The following
component profile is for one 200-ms Transformer decision, batch size one, after warm-up on an
Apple M2:

| Component | CPU mean | MPS mean |
|---|---:|---:|
| RMS-detector computation | 0.079 ms | 0.089 ms |
| Prefix crop | 0.008 ms | 0.011 ms |
| STFT plus standardization | 0.342 ms | 0.360 ms |
| Tensor preparation | 0.015 ms | 0.580 ms |
| Model forward | 3.047 ms | 5.921 ms |
| Logit-to-probability conversion | 0.037 ms | 0.584 ms |
| Policy computation | 0.120 ms | 0.140 ms |
| **Total computation** | **3.68 ms** | **7.73 ms** |
| Post-anchor acoustic observation | 200.00 ms | 200.00 ms |
| **Observation plus computation** | **203.68 ms** | **207.73 ms** |

The 0.079/0.089-ms entry is computation time, not the signal time needed to establish an onset.
With 20-ms frames, 10-ms hops, ten initial frames, and two-frame confirmation, the adaptive
threshold is available after 110 ms of the recording. An anchor at frame m is supported by two
frames after mH + 30 ms and cannot be emitted before max(mH + 30 ms, 110 ms), even though the
returned anchor is backdated to mH. The stored confirmation field records the two-frame edge and
does not by itself encode the 110-ms threshold-availability bound; this is now disclosed.

Eligible adaptive candidates begin at 200 ms; the 150-ms observation is their mandatory
predecessor. Both are later than the 110-ms initial-threshold bound. The buffered 30-ms pre-roll
is already available and adds no future wait. Latency from the beginning of the recording still
depends on when speech actually starts; our “observation time” is deliberately measured from the
returned anchor and is not presented as a fixed recording-start latency.

**Manuscript location:** §3.1, §5.7, and Tables 30 and 33.

**Reviewer Concern (R1.16):** The manuscript should report computational complexity, parameter
count, memory use, and real-time inference latency.

**Author Response:** Complexity is reported for each compact model at a 300-ms prefix, and the
component timing above is reported for the Transformer at 200 ms:

| Compact model | Parameters | Serialized weights | MACs | FLOPs | CPU inference | Peak RSS increment |
|---|---:|---:|---:|---:|---:|---:|
| Small CNN | 23,538 | 100 KiB | 18.26 M | 36.52 M | 2.14 ms | 5.2 MB |
| CNN-BiLSTM | 69,354 | 279 KiB | 44.70 M | 89.41 M | 3.69 ms | 14.9 MB |
| Transformer | 196,594 | 783 KiB | 46.24 M | 92.49 M | 3.63 ms | 14.5 MB |

The counting convention is one multiply-accumulate equal to two FLOPs. These measurements
support a narrow lightweight-compute description for the three compact networks; they do not
make the 200–300 ms acoustic observation disappear. The frozen WavLM model's total and trainable
parameter counts are reported at R1.10, but WavLM runtime, MACs, FLOPs, and peak memory were not
profiled.

**Manuscript location:** §5.7 and Tables 29–30.

**Reviewer Concern (R1.17):** The use of only head-mounted microphone recordings limits the
generalizability of the proposed method to mobile, conversational and far-field environments;
the limitation should be experimentally investigated rather than only acknowledged.

**Author Response:** We completed a deterministic test-time perturbation analysis, but it is not
a substitute for recordings from other microphones or environments:

| Condition | Accuracy | Macro-F1 | Dysarthric recall | Accuracy change from clean |
|---|---:|---:|---:|---:|
| Clean | 41.95% | 41.95% | 59.37% | — |
| Additive noise, 20 dB SNR | 43.09% | 43.09% | 60.26% | +1.14 points |
| Additive noise, 10 dB SNR | 44.57% | 44.24% | 72.49% | +2.62 points |
| Additive noise, 0 dB SNR | 38.05% | 35.82% | 78.56% | -3.90 points |
| Anchor shift, -50 ms | 35.84% | 33.60% | 75.12% | -6.11 points |
| Anchor shift, -30 ms | 35.27% | 33.55% | 71.17% | -6.68 points |
| Anchor shift, -10 ms | 38.89% | 38.61% | 63.26% | -3.06 points |
| Anchor shift, +10 ms | 43.44% | 43.39% | 56.00% | +1.49 points |
| Anchor shift, +30 ms | 41.63% | 41.60% | 54.74% | -0.32 points |
| Anchor shift, +50 ms | 40.97% | 40.96% | 55.11% | -0.98 points |

The improvement under moderate synthetic noise is counterintuitive and indicates instability or
class-bias sensitivity; this experiment does not identify a causal mechanism and cannot be read
as robustness. Genuine far-field, mobile, alternative-microphone, reverberant, and spontaneous
conversational evaluations were not completed. The revised scope and limitations say so
explicitly.

**Manuscript location:** §5.5, §6, Table 34, and Figure 8.

**Reviewer Concern (R1.18):** The manuscript should provide per-speaker results or at least
fold-wise confusion matrices to determine whether the overall performance is dominated by a
small number of speakers.

**Author Response:** We report per-speaker accuracy and predicted-dysarthric rate with each
speaker's utterance count, plus the pooled confusion matrix at R1.20. Because every TORGO speaker
in this binary subset belongs to one class, speaker-level macro-F1 is not an informative
two-class quantity and is not claimed.

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

The spread is extreme and most control speakers have a high predicted-dysarthric rate. This
supports the descriptive conclusion of broad dysarthric-class prediction bias. It does not by
itself establish why that bias occurs.

**Manuscript location:** §5.6, Tables 28 and 32, and Figure 10.

**Reviewer Concern (R1.19):** The distinction between dysarthric-speech detection and clinical
diagnosis should remain explicit.

**Author Response:** We agree. The revised terminology is “utterance-level dysarthric-speech
classification” or “dysarthric-speech detection.” This binary corpus-label task does not diagnose
a neurological disorder, quantify severity, establish etiology, or constitute a clinical
decision instrument. Those boundaries are stated in the title context, abstract, task
definition, discussion, and conclusion.

**Manuscript location:** §1, §1.1, §4.1, and §7.

**Reviewer Concern (R1.20):** The reported experimental tables and figures should be cross-checked
because the final confusion matrix does not reproduce the accuracy claimed for the best
entropy-based Transformer.

**Critical Experimental Validation Issue:** Figure 6 is mathematically inconsistent with the
reported result. The manuscript reports the best entropy-based Transformer as accuracy 75.89%,
macro-F1 72.82%, dysarthric recall 73.38%, and mean latency 264.6 ms. However, Figure 6 gives:

| | Predicted control | Predicted dysarthric |
|---|---:|---:|
| True control | 3,947 | 1,250 |
| True dysarthric | 786 | 2,148 |

These values total 8,131 samples, consistent with Table 4. But the accuracy calculated from this
confusion matrix is (3,947+2,148)/8,131=74.96%, not 75.89%. The dysarthric recall is approximately
2,148/(2,148+786)=73.21%, rather than 73.38%. The confusion matrix also gives a macro-F1 of
approximately 73.67%, rather than 72.82%.

Therefore, Figure 6 and Table 3/5 cannot represent exactly the same prediction set. The authors
must regenerate the confusion matrix and all metrics from the same saved test predictions. This
is a major experimental-validation issue, particularly because the manuscript explicitly claims
that all reported metrics are generated from saved prediction files and are mutually consistent.

**Author Response:** The reviewer is correct. The submitted matrix has true-control counts
3,947/1,250 and true-dysarthric counts 786/2,148. It gives 74.96% accuracy, 73.21% dysarthric
recall, and approximately 73.67% macro-F1, not the submitted 75.89%, 73.38%, and 72.82%.

The original code and prediction files no longer exist, so we cannot establish the historical
cause. A pooled-versus-unweighted-fold estimand mismatch is consistent with the unequal fold
sizes and opposite-signed metric discrepancies, but it is not proven and is not presented as a
causal diagnosis. The old result is withdrawn.

For the rebuilt complete calibrated Transformer policy, the classification ledger yields:

| | Predicted control | Predicted dysarthric | Total |
|---|---:|---:|---:|
| True control | 1,669 | 3,528 | 5,197 |
| True dysarthric | 1,192 | 1,742 | 2,934 |
| **Total** | **2,861** | **5,270** | **8,131** |

From those same decision rows, accuracy is 41.95%, dysarthric precision 33.06%, dysarthric recall
59.37%, specificity 32.11%, macro-F1 41.95%, balanced accuracy 45.74%, mean observation
269.389 ms, and forced fallback 64.25%. Decision counts are 2,285 at 200 ms, 408 at 250 ms, and
5,438 at 300 ms; 5,224 are forced fallbacks and 214 satisfy the rule at the endpoint.

Every classification-result table for this rebuilt system is derived from its canonical
fixed-prefix or decision ledger. Calibration, onset diagnostics, robustness, profiling, pre-roll
retraining, and the error audit have distinct, separately hashed source artifacts. This replaces
the unsupported claim that every result in the paper comes from one universal ledger.

**Manuscript location:** §4.4, §5.6, and Table 28.

## Reviewer-1 Required Experimental Validation

**Reviewer Concern (R1-EV.1):** Recompute from the same prediction file Accuracy, Precision,
Recall, Specificity, Macro-F1, Dysarthric recall, Confusion matrix, mean latency and
Forced-fallback rate.

**Author Response:** Completed for the rebuilt system. The matrix and all requested values are
reported at R1.20 and derived from the same 8,131 decision rows. Historical submitted values are
withdrawn rather than mixed with the reconstruction.

**Manuscript location:** §5.6 and Table 28.

**Reviewer Concern (R1-EV.2):** Five-fold statistical validation report for each fold Accuracy,
Macro-F1, Dysarthric recall, mean latency, Forced-fallback rate and provide mean and SD and 95%
confidence intervals.

**Author Response:** Completed at R1.6–R1.7. All five fold rows and unweighted mean ± SD are
reported, together with speaker-cluster bootstrap 95% intervals for the pooled estimand.

**Manuscript location:** §5.4 and Tables 23–24.

**Reviewer Concern (R1-EV.3):** Entropy-threshold sensitivity analysis must be done. This is
essential because the proposed method's main contribution is the entropy-based decision rule.

**Author Response:** Completed at R1.3. The exhaustive 14-by-11 validation grid contains 154
cells per fold and 770 rows total. Every cell satisfies the stated macro-F1 feasibility
constraint. Folds 1, 3, and 4 are completely degenerate at 300 ms/100% fallback; folds 0 and 2
respond to the grid and select θ=0.95, ε=0.20. The complete adaptive test point pools these
fold-specific, validation-selected policies; test rows are never swept.

**Manuscript location:** §5.4, Table 22, and Figure 7.

**Reviewer Concern (R1-EV.4):** Compare uncalibrated softmax, temperature scaling, calibrated
posterior entropy. Then determine whether calibration improves the early-decision policy.

**Author Response:** Completed at R1.13. Temperature scaling reduces NLL, Brier score, and ECE,
but policy accuracy does not improve and forced fallback rises from 8.9% to 64.2%. Calibration is
therefore useful for exposing the raw posterior's overconfidence, not for claiming a better
early-decision result here.

**Manuscript location:** §3.7, §5.3, Tables 19–20, and Figure 5.

**Reviewer Concern (R1-EV.5):** At minimum compare Fixed 300 ms Transformer, Entropy-only
decision, Entropy and class stability, Entropy with entropy stability, Complete entropy-based
framework. This would establish whether each condition in Equation 4 actually contributes to the
final performance.

**Author Response:** Completed in the ablation at R1.1, with fixed 200/250/300-ms references and
maximum-softmax confidence added. Entropy-only and maximum-softmax rows match exactly in this
binary task. Stability conditions add observation time without an accuracy improvement in this
reconstruction.

**Manuscript location:** §5.4 and Table 23.

**Reviewer Concern (R1-EV.6):** Report results at 200 ms, 250 ms and 300 ms and calculate a
standardized accuracy–earliness trade-off rather than reporting mean latency alone.

**Author Response:** Completed at R1.11. Fixed 150 ms is also shown as an evaluated fixed-prefix
reference. Consumed fraction, timeliness, and two harmonic trade-off scores make the conclusion
explicit: fixed 150 ms is the best standardized point; the calibrated adaptive policy does not
improve the fixed-prefix frontier.

**Manuscript location:** §5.2, §5.4, and Tables 16 and 21.

**Reviewer Concern (R1-EV.7):** The strongest validation would be Train on TORGO → test on
UA-Speech or an equivalent independent corpus. The authors themselves identify cross-corpus
evaluation as an important future direction.

**Author Response:** Not completed. This requires new corpus access and a separately specified
label/protocol mapping. Claims have been narrowed to TORGO; no within-corpus robustness test is
presented as a substitute.

**Manuscript location:** §6 (listed as not completed).

**Reviewer Concern (R1-EV.8):** Evaluate the method under background noise, reduced SNR,
speech-onset detection errors, different microphone conditions, continuous speech, different
speakers/severity levels.

**Author Response:** Partly completed. Deterministic additive-noise and imposed-anchor-shift
results are reported at R1.17; the 0-versus-30-ms retraining comparison is at R1.5; per-speaker
results are at R1.18; and the available speaker-level Frenchay severity strata are reported at
R2.8. Genuine microphone-domain, far-field/mobile, spontaneous continuous-speech, and
cross-corpus experiments were not completed. Etiology cannot be stratified because the
cohort-level TORGO cause counts are not mapped to speaker IDs in the verified source. Synthetic
noise and imposed shifts are not described as ground-truth onset validation or microphone-domain
evaluation.

**Manuscript location:** §5.5, §6, Table 34, and Figure 8.

**Reviewer Concern (R1-EV.9):** Report number of parameters, model size, FLOPs, CPU inference time,
GPU inference time, STFT preprocessing time, onset-detection time and total end-to-end latency.
This is important because 264.6 ms mean decision latency is not necessarily equivalent to 264.6
ms real-world system latency.

**Author Response:** Compact-model parameters, storage, MACs/FLOPs, CPU inference, and peak RSS
are reported at R1.16. CPU and Apple MPS component timing for the 200-ms Transformer path is at
R1.15. Apple MPS is reported as the available accelerator measurement, not mislabeled as a
discrete GPU result. WavLM runtime/FLOPs were not profiled. Acoustic observation, detector signal
availability, and compute are kept separate.

**Manuscript location:** §5.7 and Tables 30 and 33–35.

## Reviewer-2 Comments with Author Response

**Reviewer Concern (R2 summary of the submitted paper):** The proposed system provides a
latency-efficient architecture for detection of dysarthria in 200-300 ms after speech starts by
leveraging log-STFT spectrogram features and small neural networks. Using posterior entropy
confidence criterion, the best-performing model Transformer yields accuracy of 75.89% on average
latency of 264.6 ms with speaker-independent TORGO tests. In this case, the trade-off between
efficiency and accuracy is achieved, although almost half of the test samples need a 300 ms
timeout.

**Author Response:** This accurately summarizes the submitted claim, but those numerical values
cannot be supported after the inconsistency identified by Reviewer 1 and the deletion of the
original prediction files. They are withdrawn. The rebuilt calibrated Transformer gives 41.95%
accuracy at 269.39 ms mean observation with 64.25% forced fallback, and it does not improve the
fixed-prefix accuracy-earliness frontier. The revised title, abstract, results, and conclusion
state the reconstructed result rather than the submitted claim.

**Manuscript location:** §1.1 and §5.

**Reviewer Concern (R2.1):** Although it can be seen that there is a latency improvement realized
using the entropy-based early decision approach, there is no clear indication of how the entropy
threshold (θ) and stability tolerance (ε) have been chosen for the system. It would be good if the
authors clearly indicate how these have been chosen and if they are highly sensitive.

**Author Response:** The exact validation-only selection rule and all 770 grid rows are reported
at R1.3. Briefly, θ ranges from 0.30 to 0.95 in steps of 0.05 and ε from 0.00 to 0.20 in steps of
0.02. The minimum-observation feasible pair is chosen separately in each fold, subject to
validation macro-F1 being no more than 0.01 below the same fold's fixed-300-ms value. All cells
are feasible, and most of the surface is degenerate: three folds remain at 300 ms/100% fallback
for every pair, whereas two folds select θ=0.95, ε=0.20. The revised conclusion is therefore that
the policy is strongly fold-dependent and not governed by one stable universal pair.

**Manuscript location:** §3.4, §5.4, Table 22, and Figure 7.

**Reviewer Concern (R2.2):** Regarding the stated contribution on detection of dysarthric speech
in minimum time, there is inadequate background provided from previous literature in early time
series classification, uncertainty-based speech processing, and short segment pathological
speech processing. It is necessary for the authors to explain what is unique about the time-
entropy based methodology in terms of theory and methodology.

**Author Response:** We expanded the background in those three areas and narrowed the novelty
claim. Normalized entropy itself is not new, and for a binary posterior it is a monotone transform
of maximum-softmax confidence. The tested contribution is an auditable application-specific
sequential policy: onset-aligned short prefixes, validation-only temperature fitting, adjacent-
prefix class and entropy stability, a maximum duration, and fold-specific constrained
validation selection. The ablation at R1.1 shows that entropy and maximum-softmax gates are
equivalent here and that the added stability conditions do not improve accuracy. Those negative
results are part of the revised theoretical interpretation.

**Manuscript location:** §1.2 and §2.1.

**Reviewer Concern (R2.3):** Literature review in this context is mainly descriptive and lacks
any form of systematic comparison. It is very important that a comparative table of past
approaches to dysarthric speech recognition be added to the paper in relation to the size of
input/segmentation, corpus and validation approach used, latency definition, speaker
independence, and classification accuracy.

**Author Response:** A structured comparison table has been added, with one field for each
requested dimension: corpus and participant count, channel or microphone, input unit and
duration, speaker separation and validation protocol, latency definition, metric, and an explicit
comparability note. It covers representative recent TORGO and UA-Speech systems and the present
compact and WavLM references. Because the published systems use different corpora, channels,
durations, tasks, and splits, the table supports protocol comparison rather than an accuracy
ranking.

**Manuscript location:** §2.4 and Table 1.

**Reviewer Concern (R2.4):** The onset detection unit based on short-time RMS energy does not have
sufficient technical rigor. The paper needs to present the complete mathematical framework for
onset detection, along with the energy thresholding conditions, temporal smoothing settings,
decision rules at the frame level, and the method of detecting errors in onset localization.

**Author Response:** For a mono waveform x sampled at 16 kHz, frame m is x[mH+n],
n=0,…,N-1, with N=320 samples (20 ms) and H=160 samples (10 ms). Its RMS is

r[m] = sqrt((1/N) Σ from n=0 to N-1 of x[mH+n]^2).

The primary implementation, historically identified as causal_noise_floor, is described in prose
as an initial-level adaptive bounded-lookahead rule. It sets

τ = max(0.003, median(r[0],…,r[9]) × 10^(15/20)).

The first ten overlapping frames begin at 0–90 ms and require audio through 110 ms. They may
contain speech and are not called a pure-noise reference. No centered padding or temporal
smoothing is used. With K=2, the returned onset frame is the earliest m for which both r[m] and
r[m+1] are at least τ. If no such pair exists, the retained fallback anchor is sample zero. A
30-ms pre-roll starts the crop at max(0,mH-480 samples).

The two-frame evidence for an anchor mH is available at mH+30 ms; because τ itself needs the
initial ten frames, the streaming availability is max(mH+30 ms,110 ms). The implementation's
stored confirmation sample reflects the two-frame edge only, and this limitation is disclosed.

No manual reference onsets exist, so real localization error is not measured. The completed
label-free diagnostic and imposed ±10/30/50-ms anchor shifts are reported at R1.4 and R1.17 and
are explicitly not called validation against ground truth. The 300-row human-annotation sheet
remains blank and is listed as incomplete.

**Manuscript location:** §3.1, §5.1, and Tables 2 and 11–15.

**Reviewer Concern (R2.5):** This formulation depends on the assumption that normalized posterior
entropy is a good proxy for uncertainty in predictions. However, in practice, deep neural
networks produce uncalibrated overconfident posteriors. It is necessary for the authors to show
whether the posteriors generated by the model are calibrated using measures like expected
calibration error or reliability diagrams.

**Author Response:** We agree. The revision reports raw and temperature-scaled NLL, Brier score,
15-bin ECE, and reliability plots, with temperature fitted only on calibration validation rows.
For the Transformer at 300 ms, ECE falls from 0.429 to 0.212, NLL from 1.393 to 0.837, and Brier
score from 0.446 to 0.303. This correction does not improve policy accuracy and increases forced
fallback from 8.9% to 64.2%. Thus calibration exposes material overconfidence but does not rescue
the early-decision policy. Full values and fold temperatures are at R1.13.

**Manuscript location:** §3.7, §5.3, Tables 19–20, and Figure 5.

**Reviewer Concern (R2.6):** Equation 4 is the formalization of the three early stopping
conditions; however, no tuned value or distribution of θ and ε is shown in the manuscript. In
order to achieve reproducibility, the exact values of θ and ε, which have been obtained in the
validation splits, must be published.

**Author Response:** The historical equation reference is retained here only to identify the
reviewer's concern; the revised prose specifies the rule directly. Validation rows are assigned
deterministically by hashed sample ID to checkpoint-selection (50%), temperature-calibration
(25%), and policy-tuning (25%) subsets. θ and ε are selected only on the last subset, using the
grid and criterion at R1.3.

Selected Transformer pairs are fold 0: 0.95/0.20; fold 1: 0.30/0.00; fold 2: 0.95/0.20; fold 3:
0.30/0.00; and fold 4: 0.30/0.00. The low-corner choices in folds 1, 3, and 4 are tie-break
artifacts on entirely flat surfaces: all 154 settings yield 300 ms and 100% fallback. The
adaptive pooled result therefore combines fold-specific selected policies rather than one global
average setting.

**Manuscript location:** §3.4.

**Reviewer Concern (R2.7):** According to the authors, a total of five model families have been
considered in this paper, such as CNN-GRU and EfficientNet-Lite models. But it can be seen that
these two models have not been included in the fixed duration results in Sections 4 and 5; rather,
there are just three models included in these sections.

**Author Response:** The reviewer identified an unreproducible inconsistency. No exact CNN-GRU or
EfficientNet-Lite variant definition, checkpoint, or prediction file survives. We have therefore
withdrawn both names and all associated historical claims instead of inventing architectures or
adding results that cannot be regenerated. The reconstructed compact comparison contains the
small CNN, CNN-BiLSTM, and Transformer; the modern frozen WavLM Base+ reference is reported
separately.

**Manuscript location:** §3.5 and §6.

**Reviewer Concern (R2.8):** Despite the information presented by the dataset summary regarding
the number of utterances per speaker, which is 8,131 utterances from 15 speakers, important
demographic and clinical distributions have been left out. The authors need to elaborate on the
demographics and clinical data related to sex distribution of the speakers, dysarthria level
severity, neurological cause of the disease, and utterance type (isolated words versus continuous
sentences).

**Author Response:** The available speaker metadata support the following sex-by-class
description: female, three dysarthric and three control speakers, contributing 943 and 2,149
utterances respectively (3,092 total); male, five dysarthric and four control speakers,
contributing 1,991 and 3,048 utterances respectively (5,039 total).

Using the same deterministic prompt taxonomy as the error audit:

| Prompt type | Total | Share | Control / dysarthric |
|---|---:|---:|---:|
| Isolated | 5,876 | 72.27% | 3,735 / 2,141 |
| Sentence | 2,080 | 25.58% | 1,365 / 715 |
| Vocal exercise | 171 | 2.10% | 96 / 75 |
| Missing prompt text | 4 | 0.05% | 1 / 3 |

A published TORGO description based on Frenchay dysarthria assessment provides a speaker-level
severity mapping [32]. Applying those published speaker labels to the selected primary
Transformer decision rows gives:

| Published severity | Speakers | Selected dysarthric utterances | Correctly predicted dysarthric | Dysarthric recall |
|---|---|---:|---:|---:|
| Severe | F01, M01, M02, M04 | 1,216 | 1,058 | 87.01% |
| Severe–moderate | M05 | 490 | 285 | 58.16% |
| Moderate | F03 | 565 | 158 | 27.96% |
| Mild | F04, M03 | 663 | 241 | 36.35% |

The same stratification appears in the revised manuscript as Table 28. These are descriptive
recall strata, not a severity-prediction experiment: severity is constant within each of only
eight dysarthric speakers, so severity and speaker identity cannot be separated by this table. The original TORGO report gives a cohort-level total of seven speakers
with cerebral palsy and one with amyotrophic lateral sclerosis [1], but the verified source does
not map those etiologies to IDs. We therefore do not impute speaker-level etiology or report an
etiology-stratified metric. Prompt type is an elicitation taxonomy; the sentence category is not
spontaneous conversation.

**Manuscript location:** §4.1, §5.6, and Tables 6–7, 30 and 33.

**Reviewer Concern (R2.9):** The description of the five-fold speaker-independent partition
scheme lacks detail. In light of the small number of speakers (N=15), the actual speaker IDs
allocated to the training, validation, and testing sets for each of the five folds must be
provided.

**Author Response:** The frozen speaker assignments are:

| Fold | Training speakers | Validation speakers | Test speakers |
|---:|---|---|---|
| 0 | F01, F03, FC02, FC03, M03, M04, MC01 | F04, M02, MC02, MC03 | FC01, M01, M05, MC04 |
| 1 | F03, FC01, FC02, FC03, M01, M04, M05, MC04 | F01, M03, MC01 | F04, M02, MC02, MC03 |
| 2 | F03, F04, FC01, FC03, M01, M02, M05, MC02, MC03, MC04 | FC02, M04 | F01, M03, MC01 |
| 3 | F01, F04, FC01, M01, M02, M03, M05, MC01, MC02, MC03, MC04 | F03, FC03 | FC02, M04 |
| 4 | F01, F04, FC02, M02, M03, M04, MC01, MC02, MC03 | FC01, M01, M05, MC04 | F03, FC03 |

No speaker appears in more than one partition within a fold. Across the five outer tests, every
selected speaker is tested once.

**Manuscript location:** §4.2 and Table 8.

**Reviewer Concern (R2.10):** Important details needed for the independent reproduction of their
findings are left out by the authors. They need to provide full architecture details such as
layer settings, filter sizes, hidden dimensions, total number of parameters, training parameters
like initial learning rate, learning rate schedule, batch size, optimization details, max epoch,
early stopping patience, and random seed values.

**Author Response:** Full compact architecture definitions and counts are provided at R1.2.
Training uses AdamW, constant learning rate 1.5×10^-4, weight decay 5×10^-4, batch size 64,
maximum 35 epochs, early-stopping patience 8, gradient-norm clipping at 5, dropout 0.30, and label
smoothing 0.05. The checkpoint metric is speaker-balanced macro-F1 on the checkpoint-selection
validation rows. Training-only augmentation applies random gain from -3 to +3 dB, additive noise
at 20–35 dB SNR with probability 0.5, and frequency masking up to 16 bins and time masking up to
4 frames with probability 0.5. The inverse-frequency replacement sampler is detailed at R1.8.

The completed primary reconstruction uses seed 20260819. Apple MPS execution is deterministic as
far as the recorded software controls permit but is not claimed to be bit-exact across hardware
or software versions. The requested independent-seed replication is now complete. The Transformer
was retrained end to end under two further seeds on the identical folds, each with its own
calibration and validation-only policy selection:

| Seed | Accuracy | Macro-F1 | Balanced accuracy | Dysarthric recall | Specificity | Fallback |
|---|---:|---:|---:|---:|---:|---:|
| 20260819 (primary) | 41.95% | 41.95% | 45.74% | 59.37% | 32.11% | 64.25% |
| 20260820 | 48.41% | 48.03% | 49.89% | 55.21% | 44.56% | 62.55% |
| 20260821 | 46.76% | 46.34% | 48.03% | 52.59% | 43.47% | 64.11% |
| Mean ± SD | 45.71 ± 3.35% | 45.44 ± 3.14% | 47.89 ± 2.08% | 55.73 ± 3.42% | 40.05 ± 6.89% | 63.64 ± 0.94% |

Seed choice moves accuracy by 6.46 points and specificity by 12.45 points, which is large next to
the differences between compact architectures. It does not change the conclusion: the three-seed
mean balanced accuracy is 47.89% with a descriptive interval of 42.73–53.05%, so the calibrated
adaptive Transformer is at or below chance under every seed. Timing is stable — mean decision
time varies by 1.33 ms and fallback by 1.70 points — so the policy's latency behaviour is not
seed-sensitive even though its accuracy is. Seed 20260819 is the least favourable of the three on
accuracy, so reporting it as the headline is conservative rather than selective.

We report this as three-seed variability only. The interval is computed over three seed means and
is not a speaker-generalization interval; only the Transformer was replicated. Results are in
revised manuscript §5.4, Table 28.

**Manuscript location:** §3.5, §4.3, §6, and Tables 3–5, 9 and 11.

**Reviewer Concern (R2.11):** The reported experiment-based evaluation uses only the point
estimates of aggregate accuracy and macro F1 score. It is necessary for the authors to
incorporate fold-based variability statistics (such as standard deviation or 95% confidence
interval), speaker-wise performance results, and statistical significance tests (Wilcoxon
signed-rank test or paired t-test) to prove that the difference in performance of the fixed-
duration baselines versus the entropy-based approach is statistically significant.

**Author Response:** Fold mean ± SD is reported at R1.6, speaker-cluster bootstrap intervals at
R1.7, per-speaker results at R1.18, and paired policy-minus-baseline intervals at R1.12. We use
speaker-cluster resampling as the primary inference because utterances within a speaker are
dependent and five fold summaries provide very low power for a conventional fold-level t or
Wilcoxon test. The utterance-level McNemar result is retained only as an explicitly unclustered
descriptive sensitivity analysis.

These analyses do not prove a general advantage. Accuracy differences between the calibrated
adaptive policy and fixed 200, 250, or 300 ms all have clustered intervals containing zero. That
negative result is now stated directly.

**Manuscript location:** §4.4, §5.4, §5.6, and Tables 23–25 and 32.

**Reviewer Concern (R2.12):** The usefulness of the early-decision scheme is now greatly
exaggerated because it compares itself to the mean of the fixed-duration approach for all time
slices instead of an optimum fixed-duration baseline (fixed 200 ms or fixed 250 ms). In addition,
since about 48.5% to 57.2% of decisions are based on the 300 ms end point, it is important for the
authors to make a fair comparison regarding the effectiveness of the proposed method in terms of
reduced 35.4 ms latency.

**Author Response:** We agree and have withdrawn the submitted comparison and its 35.4-ms claim.
The reconstruction compares the same utterances separately at every fixed operating point. The
adaptive policy averages 269.39 ms: it is 69.39 ms slower than fixed 200 ms, 19.39 ms slower than
fixed 250 ms, and 30.61 ms faster than fixed 300 ms. Adaptive accuracy is 41.95%, versus 43.07%
at fixed 200 ms, 42.11% at fixed 250 ms, and 41.94% at fixed 300 ms; all paired
speaker-cluster accuracy intervals include zero. Forced fallback is 64.25%, with 66.88% of all
utterances reaching 300 ms when accepted endpoint decisions are included.

Fixed 150 ms is also a valid evaluated fixed-prefix operating point and has the best standardized
accuracy-earliness score. It is not an adaptive candidate because the sequential stability rule
uses it as the mandatory predecessor for the first eligible 200-ms decision. The revised
conclusion is that the calibrated adaptive policy does not improve the fixed-prefix frontier; its
only supported timing benefit is relative to waiting for 300 ms on every utterance.

**Manuscript location:** §5.2, §5.4, and Tables 16 and 21.

## Reviewer-3 Comments with Author Response

**Reviewer Concern (R3.1):** How is speech onset detected using the RMS detector, and what
threshold, frame length, hop size, and onset criterion are used?

**Author Response:** At 16 kHz, the detector uses 320-sample (20-ms) frames and a 160-sample
(10-ms) hop. For frame m,

r[m] = sqrt((1/320) Σ from n=0 to 319 of x[160m+n]^2).

The primary initial-level adaptive bounded-lookahead rule sets
τ=max(0.003,median(r[0],…,r[9])×10^(15/20)). Its historical implementation ID is
causal_noise_floor. The first ten overlapping frames may contain speech; they are not assumed to
be noise. There is no centered padding, temporal smoothing, or moving average. The onset anchor
is the earliest frame start mH for which two consecutive RMS values, r[m] and r[m+1], meet or
exceed τ. If no pair qualifies, the retained fallback anchor is zero. Threshold availability and
confirmation timing are detailed at R1.15/R2.4. No manual ground-truth onsets exist, so the rule
is not presented as quantitatively validated localization.

**Manuscript location:** §3.1, §5.1, Tables 2 and 11–15, and Figure 2.

**Reviewer Concern (R3.2):** Why was a fixed 30 ms pre-roll selected, and how does it affect
classification performance?

**Author Response:** The 30-ms value was retained as bounded, already observed context intended
to reduce signal loss if an energy anchor follows weak initial energy. It was not selected by
optimizing held-out performance, and no claim of a proven articulatory cue is made. A nominal
t-ms example contains t ms after the returned anchor plus at most 30 ms before it; pre-roll adds
input samples but no future waiting.

The completed condition-matched 0-versus-30-ms retraining is reported fully at R1.5. In summary,
0 ms gives 50.98% accuracy versus 41.95% at 30 ms, a +9.03-point paired difference with
speaker-clustered 95% CI +0.09 to +19.79. Specificity is also higher by 16.93 points
[+3.42,+34.40]. Macro-F1, balanced accuracy, recall, latency, and fallback difference intervals
include zero. Thus the data do not support retaining 30 ms as a performance-improving choice.

**Manuscript location:** §3.1, §3.2, §5.4, and Tables 26–27.

**Reviewer Concern (R3.3):** Provide the complete architecture of the small CNN, including
layers, kernel sizes, filters, pooling, activations, and fully connected layers.

**Author Response:** The input is B×1×257×T. The network is:

1. bias-free Conv2d 1→16, kernel 3×3, stride 1, padding 1; BatchNorm2d(16); SiLU; MaxPool2d 2×2;
2. bias-free Conv2d 16→32, kernel 3×3, stride 1, padding 1; BatchNorm2d(32); SiLU; MaxPool2d 2×2;
3. bias-free Conv2d 32→64, kernel 3×3, stride 1, padding 1; BatchNorm2d(64); SiLU; MaxPool2d 2×2;
4. length-aware mean pooling over valid remaining frequency/time positions; dropout 0.30; and
5. Linear 64→2, producing control/dysarthric logits.

The verified total is 23,538 parameters. “Small CNN” does not hide an additional fully connected
block.

**Manuscript location:** §3.5 and Table 3.

**Reviewer Concern (R3.4):** Which exact EfficientNet-lite variant is used?

**Author Response:** No exact variant can be identified from the surviving material. The width
and depth multipliers, resolution, implementation source, checkpoint, and predictions were
deleted or never recorded. We therefore remove EfficientNet-Lite and its historical result from
the revised manuscript rather than guess a variant. The same reproducibility disposition applies
to the unrecoverable CNN-GRU.

**Manuscript location:** §3.5 and §6.

**Reviewer Concern (R3.5):** How are the 2D log-STFT representations converted into frame-wise
token sequences for the Transformer and recurrent models?

**Author Response:** Both sequence models receive B×1×257×T log-magnitude STFT inputs and use
two convolutional stages: 1→24 and 24→48 channels, each a 3×3 bias-free convolution with padding
1, BatchNorm2d, SiLU, and 2×1 max pooling. Pooling halves only frequency, so the T time positions
remain aligned to STFT frames.

For the CNN-BiLSTM, the resulting B×48×64×T tensor is averaged over its 64 frequency positions,
transposed to B×T×48, and passed to a one-layer bidirectional LSTM with hidden size 64 per
direction. Masked temporal mean pooling produces 128 features, followed by dropout 0.30 and
Linear 128→2.

For the Transformer, the convolutional tensor is adaptively mean-pooled to two frequency bins.
At each time position, 48 channels×2 bins are flattened into a 96-dimensional vector, projected
by a bias-free Linear 96→96 layer, and given sinusoidal positional encoding. The T vectors are
the tokens for two Transformer encoder layers. Padding masks are propagated, and only valid
tokens enter the final masked mean. The head is Linear 96→24, SiLU, dropout 0.30, and
Linear 24→2.

**Manuscript location:** §3.5 and Tables 4–5.

**Reviewer Concern (R3.6):** How are the entropy threshold θ and stability tolerance ϵ selected,
including the search range, resolution, and optimization criterion?

**Author Response:** For each fold, after fitting temperature on its calibration-validation rows,
we exhaustively search θ=0.30,…,0.95 at resolution 0.05 and ε=0.00,…,0.20 at resolution 0.02 on
separate policy-tuning validation rows. This is 14×11=154 pairs per fold. A pair is feasible if
its validation macro-F1 is at least fixed-300-ms macro-F1 minus 0.01. We minimize mean
observation; ties prefer higher macro-F1, lower fallback, lower θ, then lower ε. Exact selected
pairs and the degenerate surface are reported at R1.3/R2.6. The adaptive test point combines
those fold-specific choices and is not tuned on test data.

**Manuscript location:** §3.4, §5.4, Table 22, and Figure 7.

**Reviewer Concern (R3.7):** What is the value of C, and why is normalization by logC used for
this binary classification task?

**Author Response:** C=2, for control and dysarthric. The normalized entropy is

H(p) = -Σ from c=1 to C of p[c] log p[c] / log C.

Division by log C maps entropy to [0,1] and makes the definition reusable for other class counts.
For C=2 it does not provide information beyond binary maximum-softmax confidence: the two are
monotonically related, as the matching ablation rows at R1.1 demonstrate. This limitation is now
stated explicitly.

**Manuscript location:** §3.3.

**Reviewer Concern (R3.8):** What is the value of Δ in Equation (4), and how is the previous
prediction handled at the first candidate decision point of 200 ms?

**Author Response:** Δ=50 ms. The revised rule has no special first-candidate exemption. A 150-ms
prediction is always computed as the mandatory predecessor. At the first eligible 200-ms point,
the system compares the 200-ms class and entropy with the 150-ms values. If the threshold, class-
stability, and entropy-stability conditions all pass, it may emit at 200 ms. Otherwise it
continues to 250 ms, comparing against 200 ms, and then to the forced 300-ms endpoint. This rule
is stated directly in prose and implementation rather than relying on a nonexistent revised
equation or algorithm number.

**Manuscript location:** §3.4.

**Reviewer Concern (R3.9):** Why does the early-decision candidate set begin at 200 ms when a
150 ms prefix is also evaluated?

**Author Response:** A stability test requires two adjacent observations. The 150-ms prefix is
therefore both a valid evaluated fixed-prefix operating point and the predecessor that makes the
first adaptive comparison possible; the eligible adaptive candidate set begins at 200 ms. The
fixed-150 result is reported in the accuracy-earliness curve and is in fact the best standardized
fixed point. It is not silently discarded or described as an adaptive emission.

**Manuscript location:** §3.4, §5.2, and Table 16.

**Reviewer Concern (R3.10):** Why are a relatively large number of control samples misclassified
as dysarthric in Figure 6, and what causes these false positives?

**Author Response:** The historical Figure 6 is withdrawn for the inconsistency at R1.20. In the
rebuilt primary calibrated Transformer, 3,528 of 5,197 control utterances are false positives
(67.89%), so the concern remains and is more severe. The output pattern establishes broad
dysarthric-class prediction bias, but the available artifacts do not establish one causal
mechanism.

The completed quantitative error audit reports the following descriptive strata:

| Control subset | False positives / control utterances | False-positive rate |
|---|---:|---:|
| All controls | 3,528 / 5,197 | 67.89% |
| Isolated prompts | 2,416 / 3,735 | 64.69% |
| Sentence prompts | 1,050 / 1,365 | 76.92% |
| Vocal exercises | 61 / 96 | 63.54% |
| Missing prompt text | 1 / 1 | 100% (n=1) |
| Detector reports onset | 3,223 / 4,865 | 66.25% |
| Detector fallback status | 305 / 332 | 91.87% |
| Duration 0.5–1 s | 6 / 12 | 50.00% |
| Duration 1–2 s | 1,243 / 1,895 | 65.59% |
| Duration 2–5 s | 2,050 / 2,981 | 68.77% |
| Duration at least 5 s | 229 / 309 | 74.11% |
| Decision at 200 ms | 910 / 1,236 | 73.62% |
| Decision at 250 ms | 198 / 246 | 80.49% |
| Decision at 300 ms | 2,420 / 3,715 | 65.14% |
| Entropy 0.8–1.0 | 2,709 / 4,191 | 64.64% |
| Entropy 0.6–0.8 | 435 / 584 | 74.49% |
| Entropy 0.4–0.6 | 214 / 248 | 86.29% |
| Entropy 0.2–0.4 | 170 / 174 | 97.70% |

There are no control observations below 0.2 entropy and no controls shorter than 0.5 s, so those
empty strata are not given percentages. “Detector fallback status” is an algorithm status, not a
human-confirmed onset error. The variations are associations and cannot show that prompt type,
duration, onset status, decision time, or entropy causes the errors.

A deterministic blinded set of 40 candidate examples exists, but all human-review fields are
blank. No human qualitative categories, manual onset references, SNR-proxy audit, or
representative spectrogram review is claimed. A causal explanation will require those annotations
and additional controlled data.

**Manuscript location:** §5.6, Tables 28 and 31–33, and Figures 9–10.

## Minor Comments with Author Response

**Reviewer Concern (Minor 1):** The manuscript requires minor English language editing to improve
grammar, clarity and overall readability.

**Author Response:** The manuscript was rewritten in full for this revision rather than edited in
place, and the new text was proofread for grammar, tense consistency, and readability. Hedged and
overstated phrasings were also removed, since several were inaccurate rather than merely unclear:
causal wording such as "symptom of speaker memorization" is now stated as an association, "five
independent measurements" is now "converging descriptive indicators," and every claim of a
completed analysis that had not been run has been withdrawn. Terminology is applied consistently
throughout: "utterance-level dysarthric-speech classification" rather than diagnosis, "anchor"
rather than "onset" where the detector's output is meant, and "label-free diagnostic" rather than
"validation."

**Manuscript location:** Throughout; the terminology decisions are stated in §1.1 and §4.1.

**Reviewer Concern (Minor 2):** References (1, 7-15, etc.) appear to be outdated, kindly update
the "literature review" with more recent and relevant references published between 2020 and 2026.

**Author Response:** Related Work was rewritten and now cites 27 references, 16 of them published
in 2020 or later, including work from 2024 and 2025 [20,24,29,31]. Added recent sources cover
pretrained and self-supervised representations for dysarthric-speech detection and severity
classification, speaker-independent evaluation of TORGO, confound auditing of this corpus, and
early time-series classification.

The 11 pre-2020 citations retained in Related Work are foundational or definitional rather than
literature-review filler, and replacing them with recent substitutes would misattribute the work:
[6] Shannon (1948) for entropy, [25] the 1999 voice-activity-detection standard that our RMS
detector follows, [7] the 2006 entropy-based confidence measure the method is distinguished
from, [1] the 2012 TORGO corpus paper, and [8,12,13,14,15,17,26] for the architecture families
and prior TORGO results the earlier version claimed. We have not removed a source merely for
being old, and we have not added a recent source that the text does not actually use.

**Manuscript location:** §2.

**Reviewer Concern (Minor 3):** References [5] and [18] contain inconsistent year suffix
formatting (e.g., "2023b"), please verify and correct the year labels according to the required
citation style.

**Author Response:** Corrected. The revised reference list contains no year-suffix labels at all:
no two references share both an author set and a year, so no disambiguating suffix is required,
and every entry now carries a plain four-digit year. All 32 entries follow one APA-style pattern
— authors, year in parentheses, title, italicized source, volume/issue, page range, and a
resolvable DOI or publisher URL — with numeric in-text citation labels. We verified
programmatically that every in-text citation resolves to a listed entry and that every listed
entry is cited at least once.

**Manuscript location:** References list.

## Additional Reference Added for the Revision

[32] Joy, N. M., Umesh, S., and Abraham, B. (2017). On improving acoustic models for TORGO
dysarthric speech database. In *Interspeech 2017* (pp. 2695–2699). ISCA.
https://doi.org/10.21437/Interspeech.2017-878

## Requests Not Completed or Not Recoverable

The following limitations remain after the reconstruction:

- Manual reference-onset annotation and ground-truth localization-error measurement were not
  completed; the prepared 300-row annotation sheet is blank.
- The 40-example blinded qualitative error review was not completed by human reviewers. Only the
  quantitative strata at R3.10 are reported.
- Cross-corpus evaluation on UA-Speech or another independent corpus was not completed.
- Genuine far-field, mobile, alternative-microphone, reverberant, and spontaneous conversational
  recordings were not evaluated.
- Published Frenchay severity categories are now reported descriptively at R2.8. A verified
  speaker-ID mapping for neurological etiology remains unavailable and was not imputed.
- Exact historical EfficientNet-Lite and CNN-GRU definitions and results are not recoverable and
  have been withdrawn.
- The independent-seed replication covers the Transformer only; the small CNN, CNN–BiLSTM, and
  WavLM reference were each run under seed 20260819 alone.
- A discrete-CUDA-GPU profile and WavLM runtime/MAC/FLOP/memory profile were not completed; Apple
  MPS timing is reported under its actual device name. WavLM fine-tuning was not evaluated.
