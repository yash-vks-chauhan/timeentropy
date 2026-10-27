# Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy

**Suresh Kumar Chelliah¹, Arivazhagan N²\***

¹,²Department of Computational Intelligence, School of Computing, SRM Institute of Science and Technology, Kattankulathur-603203, Chennai, India.

E-mail: ¹sureshkc@srmist.edu.in, ²arivazhn@srmist.edu.in

\* Corresponding author: arivazhn@srmist.edu.in

ORCID: Suresh Kumar Chelliah — 0009-0006-5641-8344; Arivazhagan N — 0000-0001-9515-4237

## Abstract

Automatic dysarthric speech detection can support assistive communication, but many systems wait for relatively long speech segments before producing a decision. This paper evaluates dysarthric-versus-control classification from 150–300 ms speech-onset prefixes and tests whether a calibrated posterior-entropy policy improves the accuracy–earliness trade-off within a 200–300 ms decision window. Experiments use 8,131 TORGO head-mounted-microphone utterances from 15 speakers and five speaker-exclusive folds. Reviewer scrutiny found that the confusion matrix in an earlier version did not reconcile with its reported metrics; because the original predictions were unavailable, all results were regenerated through a frozen pipeline and the earlier values were withdrawn.

Under a level-adaptive RMS onset rule, the three compact log-STFT models did not exceed chance balanced accuracy at 300 ms: 47.47% for the small CNN, 47.56% for the CNN–BiLSTM, and 45.88% for the Transformer. A frozen WavLM Base+ representation reached 59.18% balanced accuracy and 58.10% macro-F1 at 300 ms, but its speaker-clustered 95% balanced-accuracy interval (47.9–70.7%) included chance. The complete temperature-scaled Transformer policy produced 41.95% raw accuracy, 41.95% macro-F1, 59.37% dysarthric recall, 269.39 ms mean post-onset observation, and 64.25% forced fallback. It did not improve the fixed-duration accuracy–earliness frontier. The raw Transformer posterior was severely miscalibrated (15-bin expected calibration error 0.429), and normalized entropy was analytically equivalent to maximum-softmax confidence in this binary task. These findings support a reproducible negative result and identify onset definition, calibration, and speaker variability as central limitations; they do not support clinical deployment.

**Keywords:** dysarthric speech detection, early decision, speaker-independent evaluation, speech onset, posterior entropy, calibration, TORGO.

## 1. Introduction

Dysarthria is a motor speech disorder in which impaired neuromuscular control of the articulators affects articulation, phonation, resonance, prosody, and overall speech intelligibility. It arises from a wide range of neurological conditions, including cerebral palsy, amyotrophic lateral sclerosis, Parkinson's disease, and stroke, and it can vary substantially in type and severity from one speaker to another. Because the disorder degrades the same acoustic cues that listeners and recognizers rely on, people with dysarthria are frequently underserved by mainstream speech interfaces and may experience reduced access to assistive and communicative technology. Automatic detection systems are valuable when they provide objective, repeatable measurements from the speech signal, but to be useful in practice they must also cope with high inter-speaker variability and with the limited size of publicly available dysarthric-speech corpora.

The TORGO database contains time-aligned acoustic and articulatory recordings from speakers with dysarthria and matched control speakers [1]. The present work uses its head-mounted-microphone recordings and a speaker-independent protocol so that evaluation asks whether a model generalizes to unseen speakers rather than whether it recognizes speakers already represented during training.

Most pathological-speech systems optimize performance after receiving a complete utterance or a relatively long segment [2–5]. That is appropriate offline but less suited to interactive settings in which acoustic observation time matters. The motivating question is therefore narrow: how much utterance-level class information is present during the first 150–300 ms after a detected onset, and does a validation-selected confidence policy provide a better accuracy–earliness trade-off than fixed observation times?

The original manuscript described normalized posterior entropy as the confidence signal in a sequential wait-or-decide rule [6,7]. Entropy-based deferral is established rather than novel, and for two classes normalized entropy is a monotone transformation of maximum-softmax confidence. The contribution evaluated here is consequently the complete operating protocol: onset-aligned short prefixes, validation-only calibration and policy selection, adjacent-prefix class and entropy stability, and a bounded 300-ms endpoint. This positioning connects the method to calibration [8] and early time-series classification [9–11] without claiming a new uncertainty statistic.

Peer review also exposed a numerical inconsistency. The earlier confusion matrix implied 74.96% accuracy, whereas the accompanying table reported 75.89%; the original code, checkpoints, and per-utterance predictions no longer existed. We therefore rebuilt the experiment from raw audio under a frozen protocol and replaced, rather than cosmetically adjusted, every unreproducible result. Historical CNN-GRU and “EfficientNet-lite” results were removed because no exact architecture, variant, checkpoint, or prediction record survives.

The main contributions of the revised study are:

1. a fully specified RMS onset, prefix, log-STFT, model, calibration, and policy pipeline;
2. regenerated speaker-exclusive fixed-prefix and adaptive results with fold and speaker uncertainty;
3. a modern frozen WavLM Base+ reference under the same folds and prefixes;
4. calibration, threshold-sensitivity, policy-component, pre-roll, seed, onset-shift, and noise analyses; and
5. a corrected confusion matrix and quantitative false-positive audit, with unsupported causal and clinical claims removed.

Extensive fold partitions, parameter settings, validation surfaces, robustness tables, and speaker-level results are provided in the Supplementary Material so that the main manuscript remains focused.

## 2. Related Work

### 2.1. Corpora and speaker-independent evaluation

TORGO and UA-Speech are widely used English dysarthric-speech resources [1,15]. Because their speaker counts are small, utterance-level random splits can retain the same speakers in training and testing and answer a different question from speaker-independent evaluation. Subject-disjoint work has used acoustic, articulatory, bottleneck, and speaker-invariant representations [2,17,18]. A TORGO/UA-Speech confound audit further showed that recording condition can influence classification and that performance must not automatically be attributed to pathology [16]. The present study therefore reports speaker-exclusive folds, per-speaker variation, the selected microphone stream, and onset diagnostics.

### 2.2. Models, uncertainty, and early decisions

Convolutional models have been applied to pairwise dysarthria detection, temporal-envelope cues, STFT magnitude/phase representations, and clinically interpretable bottlenecks [3–5,19]. Recurrent systems model temporal dependencies [13,21,22], while Transformer encoders provide content-dependent temporal interaction [14]. Recent work increasingly uses domain adaptation and self-supervised speech representations for detection and severity assessment [20,23,24,28,29]. We include a frozen WavLM Base+ reference [30,31] because a modern pretrained representation is a stronger comparator than only training compact models from scratch.

Entropy and posterior confidence have long been used to detect uncertain predictions [6,7], and temperature scaling is a standard held-out correction for overconfident neural posteriors [8]. Early time-series classification treats predictive quality and earliness as separate outcomes [9–11]. Our entropy gate is therefore evaluated as one component of a sequential policy, not as a new theory of uncertainty.

### 2.3. Protocol-aware comparison

Table 1 summarizes representative studies using the dimensions that most affect comparability. It is descriptive rather than a ranking because corpora, microphones, segment durations, split units, prediction units, and metrics differ.

**Table 1.** Compact protocol comparison with representative dysarthria studies. NR: not reported in the verified source.

| Study | Corpus and input | Speaker protocol | Reported endpoint | Difference from this study |
|---|---|---|---|---|
| Millet and Zeghidour [26] | TORGO, raw speech padded to 2.5 s | Fixed disjoint speakers | UAR 82.4 ± 0.4 | Much longer fixed input; one split |
| Hernandez et al. [27] | TORGO, whole sentences | Speakers overlap train/test | Accuracy 85.7% | Does not estimate unseen-speaker generalization |
| Schu et al. [16] | TORGO, 500-ms windows and speaker aggregation | Leave-one-speaker-out | Speaker accuracy 71.1% for speech | Different prediction unit and aggregation |
| Javanmardi et al. [20] | TORGO array microphone, complete utterances | Leave-one-speaker-out | Accuracy 76.12% | Different channel and full-utterance input |
| Xiong et al. [29] | Large private source; TORGO cross-corpus test | Source speaker-disjoint | Accuracy 74.18% | Training data and transfer setting differ |
| Present study | TORGO head microphone, 150–300-ms onset prefixes | Five speaker-exclusive folds | Accuracy, balanced accuracy, macro-F1, and observation time | Short-prefix utterance classification with adaptive stopping |

No verified comparison uses the same corpus subset, microphone, onset rule, 150–300-ms inputs, speaker folds, and adaptive observation policy. Numerical scores are therefore interpreted within protocol.

## 3. Materials and Methods

![](artifacts/figures/figure_architecture.png){width=78%}

**Figure 1.** Reconstructed speech-onset time-entropy pipeline. The 150-ms prefix is evaluated as a fixed baseline and supplies the predecessor for the first eligible adaptive decision at 200 ms.

### 3.1. Speech onset, pre-roll, and log-STFT prefixes

Every waveform is converted to mono at 16 kHz. Following short-time energy practice in voice-activity detection [25], left-aligned RMS for waveform $x[n]$ is computed without centered padding:

$$
r[m]=\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x[mH+n]^2}.
$$

The frame length is $N=20$ ms (320 samples), the hop is $H=10$ ms (160 samples), and no temporal smoothing is applied. The primary threshold is

$$
\tau=\max\left(0.003,\operatorname{median}\{r[0],\ldots,r[q_x-1]\}\,10^{15/20}\right),
\qquad q_x=\min(10,M_x),
$$

where $M_x$ is the number of complete frames. An onset is the start of the first run of two consecutive frames satisfying $r[m]\ge\tau$. If no such run exists, detector failure is recorded and the registered crop fallback is sample zero. The ten threshold frames start at 0–90 ms and end by 110 ms; evidence for the two-frame run is available 30 ms after its returned anchor. These signal-availability conditions are distinguished from the detector's sub-millisecond computation time.

A nominal $t$-ms prefix contains a 30-ms pre-roll slot followed by the complete $t$-ms post-anchor slot, with zero padding only at waveform boundaries. The pre-roll is already-observed buffered history and therefore adds no future waiting; a 200-ms decision uses up to 230 ms of samples but still requires 200 ms of post-anchor observation. Thirty milliseconds was retained as a bounded design choice to reduce truncation when weak onset energy precedes the threshold crossing, not because it was proven optimal. Condition-matched zero-pre-roll retraining did not establish a balanced-accuracy benefit for 30 ms (Supplementary Table S11).

Each compact-model crop is converted to a log-magnitude STFT using a 512-point FFT, a 25-ms periodic Hann window, a 10-ms hop, no centered padding, and per-utterance standardization. Prefixes of 150, 200, 250, and 300 ms yield 257 frequency bins and 16, 21, 26, and 31 time frames, respectively.

### 3.2. Posterior entropy and sequential policy

Let $p(c\mid X_t)$ be the posterior for class $c$ after prefix $X_t$. Here $C=2$ (control and dysarthric), and normalized entropy is

$$
H_t=-\frac{1}{\log C}\sum_{c=1}^{C}p(c\mid X_t)\log p(c\mid X_t).
$$

Division by $\log C=\log 2$ maps entropy to $[0,1]$. For $C=2$, $H_t$ is a strictly decreasing transformation of $\max_c p(c\mid X_t)$; an entropy-only gate is therefore equivalent to a maximum-softmax-confidence gate.

The evaluated prefixes are $\{150,200,250,300\}$ ms with $\Delta=50$ ms. The eligible adaptive set is $\{200,250,300\}$ ms. The 150-ms prediction never emits an adaptive decision; it initializes the class and entropy stability tests at 200 ms. The earliest eligible $t$ is accepted when

$$
H_t\le\theta,\qquad \hat y_t=\hat y_{t-\Delta},\qquad |H_t-H_{t-\Delta}|\le\epsilon.
$$

If no candidate satisfies all conditions, the 300-ms prediction is emitted. Within each fold, validation-speaker utterances are deterministically divided into checkpoint-selection (50%), calibration (25%), and policy-tuning (25%) subsets. After temperature fitting on the calibration subset, exhaustive policy search uses $\theta=0.30,0.35,\ldots,0.95$ and $\epsilon=0.00,0.02,\ldots,0.20$ (154 pairs per fold). Selection minimizes validation observation time subject to macro-F1 remaining within 0.01 absolute of that fold's fixed-300-ms validation macro-F1; ties favor higher macro-F1, lower fallback, lower $\theta$, and then lower $\epsilon$. Test speakers are never used for fitting or selection.

### 3.3. Model architectures and frame-wise tokens

All compact models receive $B\times1\times257\times T$ log-STFT tensors and emit two logits ordered control/dysarthric. Bias is omitted from convolutions followed by batch normalization, and dropout is 0.30.

**Table 2.** Reproducible architecture summary. Complete layer and parameter accounting is in Supplementary Tables S2–S4.

| Model | Convolution/tokenizer | Sequence and head | Trainable parameters |
|---|---|---|---:|
| Small CNN | Conv 1→16, 16→32, 32→64; each 3×3, stride 1, pad 1, BatchNorm, SiLU, 2×2 max pool | Length-aware mean over valid time/frequency; dropout; Linear 64→2 | 23,538 |
| CNN–BiLSTM | Conv 1→24→48; 3×3, BatchNorm, SiLU, frequency-only 2×1 pool; mean frequency to one 48-D token per frame | One bidirectional LSTM, hidden 64/direction; masked temporal mean; dropout; Linear 128→2 | 69,354 |
| Transformer | Same 1→24→48 front end; adapt frequency to two bins, flatten 48×2, bias-free Linear 96→96, sinusoidal positions | Two post-norm layers, $d=96$, four heads, FFN 256, GELU; masked mean; Linear 96→24→2 | 196,594 |
| Frozen WavLM Base+ [30,31] | Raw 16-kHz waveform; frozen `microsoft/wavlm-base-plus`, immutable revision `4c66d4806a428f2e922ccfa1a962776e232d487b` | Masked mean; dropout; trained Linear 768→2 | 1,538 head; 94,381,936 frozen backbone |

The recurrent and Transformer tokenizers pool frequency only, never time. Consequently, each valid STFT time frame produces one sequence token: 48-dimensional for the CNN–BiLSTM and 96-dimensional for the Transformer. Padding masks exclude padded tokens from recurrence/attention pooling.

The historical manuscript named CNN-GRU and an “EfficientNet-lite” model motivated by efficient convolutional scaling [12], but their exact definitions and predictions are unrecoverable. They are omitted rather than assigning an EfficientNet-Lite variant retrospectively. Newly specifying and retraining such baselines would constitute a new experiment, not recovery of the historical models.

### 3.4. Dataset, folds, training, and evaluation

The selected TORGO head-mounted-microphone subset contains 8,131 utterances from 15 speakers: 5,197 control utterances from seven speakers and 2,934 dysarthric utterances from eight speakers. Isolated tokens account for 72.27%, sentences 25.58%, vocal exercises 2.10%, and four rows have missing prompt text. Published dysarthria severity is available only as a speaker-level mapping [32]; it is descriptive and cannot be separated from speaker identity in this small cohort. Speaker-level neurological etiology is not imputed because the verified corpus source reports only cohort totals.

Five predefined folds keep training, validation, and test speakers disjoint, and every speaker appears in exactly one test partition (Supplementary Table S6). Test-fold sizes range from 1,288 to 1,940 utterances. The original class distribution is 63.92% control and 36.08% dysarthric. During training, inverse-frequency weighted sampling with replacement gives equal expected class mass; evaluation always uses the untouched test distribution.

Training uses AdamW, learning rate $1.5\times10^{-4}$, weight decay $5\times10^{-4}$, batch size 64, at most 35 epochs, patience 8, gradient-norm clipping at 5.0, dropout 0.30, and label smoothing 0.05. Train-only augmentation combines waveform gain, additive Gaussian noise, compact log-STFT masks, and frequency masking. The primary seed is 20260819; Transformer seeds 20260820 and 20260821 assess seed sensitivity. Checkpoints are selected by speaker-balanced macro-F1 on the checkpoint-selection validation rows.

We report pooled out-of-fold accuracy, macro-F1, dysarthric recall, specificity, balanced accuracy, mean post-anchor observation time, and forced-fallback rate. Unweighted fold mean ± sample SD is kept distinct from pooled estimates. Primary uncertainty uses 10,000 speaker-clustered bootstrap replicates; utterance-level McNemar tests are retained only as descriptive sensitivity analyses. The accuracy–earliness display reports accuracy and time separately and also gives consumed fraction $t/300$, timeliness $1-t/300$, and their harmonic score.

## 4. Results and Discussion

### 4.1. Onset diagnostics

A label-free diagnostic of 600 utterances compared the primary level-adaptive rule with a fixed 0.003 threshold and the recovered whole-utterance peak-relative rule. The primary rule anchored at sample zero on 4.7% of files and produced 3.5% low-energy 300-ms crops, compared with 71.7%/43.3% for the fixed rule and 45.0%/27.5% for the peak-relative rule (Supplementary Table S7). These are waveform indicators, not errors against manually annotated onsets; reference-onset annotation remains incomplete.

Condition-matched retraining paradoxically raised pooled accuracy to 59.28% under the fixed rule and 67.37% under the recovered peak-relative rule, versus 41.95% for the primary rule. The gain was confined to recording-start crops, which the comparator rules produced much more often for control than dysarthric utterances (class gaps 29.15 and 47.65 points). The detector outcome therefore supplied a class-correlated recording-start shortcut. We retain the prespecified primary rule because reselecting an onset rule from test accuracy would itself use test data for design. Full stratification is in Supplementary Tables S8–S9.

### 4.2. Fixed-prefix models

**Table 3.** Pooled out-of-fold fixed-prefix results over all 8,131 utterances. Balanced accuracy is included because raw accuracy is affected by the 64/36 class prior.

| Model | 150-ms accuracy / balanced accuracy | 200-ms accuracy / balanced accuracy | 250-ms accuracy / balanced accuracy | 300-ms accuracy / balanced accuracy |
|---|---:|---:|---:|---:|
| Small CNN | 44.12 / 48.74% | 43.18 / 47.60% | 44.45 / 47.53% | 44.32 / 47.47% |
| CNN–BiLSTM | 41.43 / 43.89% | 42.66 / 44.28% | 44.73 / 46.06% | 46.61 / 47.56% |
| Transformer | 43.94 / 46.74% | 43.07 / 46.03% | 42.11 / 45.55% | 41.94 / 45.88% |
| Frozen WavLM Base+ | 52.50 / 55.87% | 55.36 / 57.29% | 56.95 / 58.32% | **59.29 / 59.18%** |

At 300 ms, none of the compact models exceeded chance balanced accuracy. WavLM improved monotonically with prefix length and reached 59.18% balanced accuracy and 58.10% macro-F1 at 300 ms. Its speaker-clustered 95% balanced-accuracy interval was 47.9–70.7%, however, so the evidence is directionally stronger but statistically unresolved. Fixed-300 raw accuracy varied substantially across folds: 24.4–75.3% for the Transformer and 46.9–72.3% for WavLM (Supplementary Table S10).

### 4.3. Calibration, policy selection, and earliness

The raw Transformer's 300-ms ECE was 0.429, with negative log likelihood 1.393 and Brier score 0.446. Temperature scaling reduced these to 0.212, 0.837, and 0.303, respectively, with mean fitted temperature 11.23. WavLM's raw ECE was 0.0445. Thus the compact Transformer was substantially overconfident, and calibration changed exit behavior even though it did not improve classification.

**Table 4.** Transformer fixed and adaptive policies, pooled out of fold. Adaptive thresholds are selected separately within each fold on validation-only rows.

| Policy | Accuracy | Macro-F1 | Mean observation | Forced fallback | Harmonic accuracy–timeliness |
|---|---:|---:|---:|---:|---:|
| Fixed 150 ms | **43.94%** | **43.89%** | 150.0 ms | 0.0% | **0.468** |
| Fixed 200 ms | 43.07% | 43.04% | 200.0 ms | 0.0% | 0.376 |
| Fixed 250 ms | 42.11% | 42.11% | 250.0 ms | 0.0% | 0.239 |
| Fixed 300 ms | 41.94% | 41.93% | 300.0 ms | 0.0% | 0.000 |
| Entropy only, raw | 42.33% | 42.33% | 209.1 ms | 4.2% | 0.353 |
| Complete policy, raw | 41.99% | 41.99% | 224.4 ms | 8.9% | 0.315 |
| Complete policy, temperature-scaled | 41.95% | 41.95% | 269.39 ms | 64.25% | 0.164 |

Entropy-only and converted maximum-softmax policies produced identical predictions. Neither the confidence gate nor the stability conditions improved accuracy over fixed 150 or 200 ms. Relative to fixed 300 ms, the complete scaled policy saved 30.61 ms (speaker-clustered 95% interval 10.91–51.94 ms) with an accuracy difference of +0.01 points (−0.41 to +0.44) and a dysarthric-recall difference of −0.68 points (−1.54 to −0.03). Relative to fixed 200 ms it was 69.39 ms slower and 1.12 accuracy points lower. The adaptive policy therefore does not improve the observed fixed-duration frontier.

The validation grid did not identify a universal threshold. Folds 1, 3, and 4 selected $\theta=0.30$, $\epsilon=0.00$ and fell back on every test utterance; folds 0 and 2 selected $\theta=0.95$, $\epsilon=0.20$. The full 770-row surface and fold selections are provided with Supplementary Table S12.

**Table 5.** Complete temperature-scaled Transformer policy by held-out test fold.

| Fold | n | Accuracy | Macro-F1 | Dysarthric recall | Mean observation | Fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,689 | 47.90% | 43.78% | 72.26% | 226.11 ms | 10.95% |
| 1 | 1,940 | 24.43% | 22.26% | 61.38% | 300.00 ms | 100.00% |
| 2 | 1,657 | 39.41% | 39.36% | 55.01% | 225.11 ms | 15.33% |
| 3 | 1,288 | 75.31% | 71.52% | 85.03% | 300.00 ms | 100.00% |
| 4 | 1,557 | 32.43% | 31.42% | 27.96% | 300.00 ms | 100.00% |
| Mean ± SD | — | 43.90 ± 19.58% | 41.67 ± 18.59% | 60.33 ± 21.39% | 270.24 ± 40.75 ms | 65.26 ± 47.60% |

Three Transformer seeds changed pooled accuracy by 6.46 points while mean observation varied by only 1.33 ms. The three-seed mean balanced accuracy was 47.89%, and its descriptive interval included 50% (Supplementary Table S13). Seed choice therefore affects classification more than timing but does not reverse the compact-model conclusion.

### 4.4. Corrected confusion matrix and false positives

**Table 6.** Pooled confusion matrix for the complete temperature-scaled Transformer policy.

| | Predicted control | Predicted dysarthric | Row total |
|---|---:|---:|---:|
| True control | 1,669 | **3,528** | 5,197 |
| True dysarthric | 1,192 | 1,742 | 2,934 |
| Total | 2,861 | 5,270 | **8,131** |

The matrix gives $(1669+1742)/8131=41.95\%$ accuracy, matching the canonical decision ledger and the 269.39-ms mean observation. The false-positive rate is 3,528/5,197 = 67.89%. False positives are not confined to one speaker: six of seven control speakers are predicted dysarthric for more than half their utterances, ranging from 65.0% to 98.3%; FC02 is the exception at 27.6%.

![](artifacts/figures/figure_confusion.png){width=70%}

**Figure 2.** Corrected pooled confusion matrices for the complete temperature-scaled Transformer and the frozen WavLM reference, shaded by true-class row proportion.

The strongest descriptive associations were registered onset-detector failure (91.87% false-positive rate versus 66.25% when detected) and sentence prompts (76.92% versus 64.69% for isolated tokens). These overlapping strata may reflect speaker, channel, or recording structure; they do not establish a unique cause. The uncompleted blinded listening sheet and missing human-reference onset labels preclude qualitative-cause or onset-error claims. Full strata and per-speaker results are in Supplementary Tables S15–S16.

### 4.5. Robustness, pre-roll, and computation

Imposed test-time anchor shifts showed asymmetric sensitivity. Shifting 30–50 ms early reduced pooled accuracy by 6.11–6.68 points; equal later shifts changed accuracy by −0.32 to −0.98 points. At 10 dB white-noise SNR, raw accuracy increased by 2.62 points. This counterintuitive improvement indicates instability or class-bias sensitivity, not robustness, and synthetic noise is not a substitute for a different microphone or corpus (Supplementary Table S14).

Removing pre-roll increased raw accuracy from 41.95% to 50.98% in one condition-matched seed, but the balanced-accuracy difference was +5.98 points with a 95% interval of −1.62 to +15.23, and dysarthric recall moved in the opposite direction. The study therefore does not show that 30 ms is empirically superior.

**Table 7.** Compact-model complexity and warm batch-one compute. MAC/FLOP counts use the 300-ms prefix; CPU compute uses the 200-ms decision path.

| Model | Trainable parameters | Serialized weights | MACs / FLOPs | CPU total compute | Peak RSS increment |
|---|---:|---:|---:|---:|---:|
| Small CNN | 23,538 | 100 KiB | 18.26 M / 36.52 M | 2.14 ms | 5.2 MB |
| CNN–BiLSTM | 69,354 | 279 KiB | 44.70 M / 89.41 M | 3.69 ms | 14.9 MB |
| Transformer | 196,594 | 783 KiB | 46.24 M / 92.49 M | 3.63 ms | 14.5 MB |

For one 200-ms Transformer decision, RMS computation, cropping, STFT/standardization, tensor preparation, model inference, posterior conversion, and policy logic total 3.629 ms on CPU after warm-up. Acoustic observation is 200 ms, so observation plus measured compute is 203.629 ms. This excludes elapsed time before the returned onset anchor; “200 ms” is explicitly a post-anchor observation measure. Full component timings and conventions are in Supplementary Table S18. WavLM runtime, memory, and FLOPs were not profiled.

## 5. Limitations and Interpretation

The findings support a narrow reconstruction, not a deployable detector. The selected corpus has 15 speakers, one head-mounted-microphone stream, and predominantly isolated-token prompts. Fold variation is large, validation speaker sets are small, and severity is inseparable from speaker identity. Cross-corpus, conversational, mobile, and real far-field validation were not performed; synthetic noise does not provide that evidence.

Onset localization has not been evaluated against human reference times. The label-free sample-zero, crop-energy, and post/pre-energy diagnostics identify threshold/recording interactions but are not localization-error statistics. The pre-roll comparison is single-seed and class-dependent. The error audit is quantitative but observational and lacks completed human qualitative labels.

The complete entropy policy does not improve the fixed-duration frontier, and normalized binary entropy adds no ordering information beyond maximum-softmax confidence. The compact models are at or below chance balanced accuracy and are unsuitable for clinical use. Frozen WavLM is directionally stronger and better calibrated, but its clustered interval includes chance and no fine-tuning or deployment profile was evaluated.

The task is corpus-defined utterance-level dysarthric-speech classification. It does not diagnose a neurological disease, estimate clinical severity, or replace clinical assessment. Broader claims require manual onset annotation, more speakers, external corpora, real channel transfer, prospective clinical validation, and independently replicated model selection.

## 6. Conclusion

This study evaluated short speech-onset prefixes and a calibrated entropy-and-stability stopping rule under speaker-exclusive TORGO folds. Peer review revealed that the earlier confusion matrix and headline metrics were inconsistent; because the original prediction records were unavailable, the experiment was reconstructed and the earlier 75.89% result was withdrawn.

The regenerated evidence changes the conclusion. Under the prespecified level-adaptive onset rule, compact CNN, CNN–BiLSTM, and Transformer models remain at or below chance balanced accuracy. The temperature-scaled Transformer policy reaches 41.95% accuracy at 269.39 ms mean post-anchor observation and does not improve the fixed-duration accuracy–earliness frontier. Its large false-positive rate, fold variation, and calibration error show that low entropy is not sufficient evidence of reliable speaker-independent classification.

Frozen WavLM Base+ reaches 59.18% balanced accuracy at 300 ms and improves with longer prefixes, but its speaker-clustered interval includes chance. The supported contribution is therefore a reproducible negative result and a transparent protocol for separating onset definition, calibration, classification quality, and observation time. Further claims require human onset references and external speaker/channel validation.

## Data and code availability

The reconstruction code, frozen configurations, fold manifests, per-utterance ledgers, policy grids, and analysis artifacts are maintained in the version-controlled project repository used for this revision. On acceptance, the authors intend to deposit a privacy-screened archive with a citable DOI. The archive will exclude TORGO audio, model checkpoints, and incomplete human-annotation worksheets; TORGO remains available from its distributors under its existing licence [1].

## Conflict of Interest / Competing Interests

The authors declare that they have no conflicts of interest or competing interests related to this work.

## Funding Information

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Ethical Statement

This study is a secondary analysis of the existing TORGO speech corpus [1]. No new participants were recruited, no intervention was performed, and no new speech recordings were collected for this study. The corpus was used in accordance with its applicable access and use conditions. The ethical approval and informed-consent arrangements for the original data collection are those documented by the corpus creators.

## Acknowledgement

The authors thank the SRM Institute of Science and Technology for providing the computational resources used in this study, and the creators and maintainers of the TORGO database for making the corpus available to the research community.

## References

[1] Rudzicz, F., Namasivayam, A. K., and Wolff, T. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. *Language Resources and Evaluation*, 46(4), 523–541. https://doi.org/10.1007/s10579-011-9145-0

[2] Kim, J., Kumar, N., Tsiartas, A., Li, M., and Narayanan, S. S. (2015). Automatic intelligibility classification of sentence-level pathological speech. *Computer Speech & Language*, 29(1), 132–144. https://doi.org/10.1016/j.csl.2014.02.001

[3] Janbakhshi, P., Kodrasi, I., and Bourlard, H. (2021). Automatic dysarthric speech detection exploiting pairwise distance-based convolutional neural networks. In *ICASSP 2021* (pp. 7328–7332). IEEE. https://doi.org/10.1109/ICASSP39728.2021.9413922

[4] Kodrasi, I. (2021). Temporal envelope and fine structure cues for dysarthric speech detection using CNNs. *IEEE Signal Processing Letters*, 28, 1853–1857. https://doi.org/10.1109/LSP.2021.3108509

[5] Janbakhshi, P., and Kodrasi, I. (2022). Experimental investigation on STFT phase representations for deep learning-based dysarthric speech detection. In *ICASSP 2022* (pp. 6477–6481). IEEE. https://doi.org/10.1109/ICASSP43922.2022.9747205

[6] Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

[7] Chen, T.-H., Chen, B., and Wang, H.-M. (2006). On using entropy information to improve posterior probability-based confidence measures. In *Chinese Spoken Language Processing: ISCSLP 2006* (pp. 454–463). Springer. https://doi.org/10.1007/11939993_48

[8] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

[9] Xing, Z., Pei, J., and Yu, P. S. (2012). Early classification on time series. *Knowledge and Information Systems*, 31(1), 105–127. https://doi.org/10.1007/s10115-011-0400-x

[10] Mori, U., Mendiburu, A., Dasgupta, S., and Lozano, J. A. (2018). Early classification of time series by simultaneously optimizing accuracy and earliness. *IEEE Transactions on Neural Networks and Learning Systems*, 29(10), 4569–4578. https://doi.org/10.1109/TNNLS.2017.2764939

[11] Schäfer, P., and Leser, U. (2020). TEASER: Early and accurate time series classification. *Data Mining and Knowledge Discovery*, 34(5), 1336–1362. https://doi.org/10.1007/s10618-020-00690-z

[12] Tan, M., and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 6105–6114). PMLR. https://proceedings.mlr.press/v97/tan19a.html

[13] Graves, A., Mohamed, A.-R., and Hinton, G. (2013). Speech recognition with deep recurrent neural networks. In *ICASSP 2013* (pp. 6645–6649). IEEE. https://doi.org/10.1109/ICASSP.2013.6638947

[14] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998–6008.

[15] Kim, H., Hasegawa-Johnson, M., Perlman, A., Gunderson, J., Huang, T. S., Watkin, K., and Frame, S. (2008). Dysarthric speech database for universal access research. In *Interspeech 2008* (pp. 1741–1744). ISCA. https://doi.org/10.21437/Interspeech.2008-480

[16] Schu, G., Janbakhshi, P., and Kodrasi, I. (2023). On using the UA-Speech and TORGO databases to validate automatic dysarthric speech classification approaches. In *ICASSP 2023* (pp. 1–5). IEEE. https://doi.org/10.1109/ICASSP49357.2023.10095981

[17] Yılmaz, E., Mitra, V., Sivaraman, G., and Franco, H. (2019). Articulatory and bottleneck features for speaker-independent ASR of dysarthric speech. *Computer Speech & Language*, 58, 319–334. https://doi.org/10.1016/j.csl.2019.05.002

[18] Janbakhshi, P., and Kodrasi, I. (2022). Adversarial-free speaker identity-invariant representation learning for automatic dysarthric speech classification. In *Interspeech 2022* (pp. 2138–2142). ISCA. https://doi.org/10.21437/Interspeech.2022-402

[19] Xu, L., Liss, J., and Berisha, V. (2023). Dysarthria detection based on a deep learning model with a clinically-interpretable layer. *JASA Express Letters*, 3(1), 015201. https://doi.org/10.1121/10.0016833

[20] Javanmardi, F., Kadiri, S. R., and Alku, P. (2024). Pre-trained models for detection and severity level classification of dysarthria from speech. *Speech Communication*, 158, 103047. https://doi.org/10.1016/j.specom.2024.103047

[21] Bhat, C., and Strik, H. (2020). Automatic assessment of sentence-level dysarthria intelligibility using BLSTM. *IEEE Journal of Selected Topics in Signal Processing*, 14(2), 322–330. https://doi.org/10.1109/JSTSP.2020.2967652

[22] Gurugubelli, K., and Vuppala, A. K. (2020). Analytic phase features for dysarthric speech detection and intelligibility assessment. *Speech Communication*, 121, 1–15. https://doi.org/10.1016/j.specom.2020.04.006

[23] Bhat, C., and Strik, H. (2025). Speech technology for automatic recognition and assessment of dysarthric speech: An overview. *Journal of Speech, Language, and Hearing Research*, 68(2), 547–577. https://doi.org/10.1044/2024_JSLHR-23-00740

[24] Kadirvelu, B., Stumpf, L., Waibel, S., and Faisal, A. A. (2025). Speaker-independent dysarthria severity classification using self-supervised transformers and multi-task learning. *PLOS Digital Health*, 4(11), e0001076. https://doi.org/10.1371/journal.pdig.0001076

[25] Sohn, J., Kim, N. S., and Sung, W. (1999). A statistical model-based voice activity detection. *IEEE Signal Processing Letters*, 6(1), 1–3. https://doi.org/10.1109/97.736233

[26] Millet, J., and Zeghidour, N. (2019). Learning to detect dysarthria from raw speech. In *ICASSP 2019* (pp. 5831–5835). IEEE. https://doi.org/10.1109/ICASSP.2019.8682324

[27] Hernandez, A., Yeo, E. J., Kim, S., and Chung, M. (2020). Dysarthria detection and severity assessment using rhythm-based metrics. In *Interspeech 2020* (pp. 2897–2901). ISCA. https://doi.org/10.21437/Interspeech.2020-2354

[28] Wang, D., Deng, L., Yeung, Y. T., Chen, X., Liu, X., and Meng, H. (2021). Unsupervised domain adaptation for dysarthric speech detection via domain adversarial training and mutual information minimization. In *Interspeech 2021* (pp. 2956–2960). ISCA. https://doi.org/10.21437/Interspeech.2021-2139

[29] Xiong, Y., Berisha, V., Liss, J., and Chakrabarti, C. (2025). Mitigating overfitting during speech foundation model fine-tuning: Applications to dysarthric speech detection. In *Interspeech 2025* (pp. 2133–2137). ISCA. https://doi.org/10.21437/Interspeech.2025-567

[30] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Wu, J., Zeng, M., Yu, X., and Wei, F. (2022). WavLM: Large-scale self-supervised pre-training for full stack speech processing. *IEEE Journal of Selected Topics in Signal Processing*, 16(6), 1505–1518. https://doi.org/10.1109/JSTSP.2022.3188113

[31] Microsoft. (2021). *WavLM Base+ model card*, revision `4c66d4806a428f2e922ccfa1a962776e232d487b`. Hugging Face. https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b (accessed August 24, 2026).

[32] Joy, N. M., Umesh, S., and Abraham, B. (2017). On improving acoustic models for TORGO dysarthric speech database. In *Interspeech 2017* (pp. 2695–2699). ISCA. https://doi.org/10.21437/Interspeech.2017-878
