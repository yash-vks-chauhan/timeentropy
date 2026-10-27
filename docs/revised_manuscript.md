# Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy: A Reproducibility Study

Suresh Kumar Chelliah¹, Arivazhagan N²*

¹,²Department of Computational Intelligence, School of Computing, SRM Institute of Science and Technology, Kattankulathur-603203, Chennai, India.

E-mail: ¹sureshkc@srmist.edu.in, ²arivazhn@srmist.edu.in

\* Corresponding author: arivazhn@srmist.edu.in

ORCID: Suresh Kumar Chelliah — 0009-0006-5641-8344; Arivazhagan N — 0000-0001-9515-4237

## Abstract

Automatic dysarthric-speech classification could support assistive applications, and a natural question is how early after speech onset a reliable utterance-level decision can be made. This paper asks whether dysarthric-versus-control speech can be classified from 200–300 ms post-onset prefixes of TORGO head-mounted-microphone recordings, and whether posterior entropy can reduce observation time within that window. Reviewer scrutiny of an earlier version revealed that its confusion matrix did not reconcile with its reported accuracy, and the original code and per-sample predictions no longer existed. We therefore rebuilt the experiment as a frozen, version-controlled pipeline and replaced the earlier numerical results.

Under a level-adaptive, bounded-lookahead onset rule and strictly speaker-independent five-fold evaluation, three compact architectures trained from scratch remained at or below chance: pooled balanced accuracy was 47.5% for the small CNN, 47.6% for the CNN–BiLSTM, and 45.9% for the Transformer, with Transformer fold accuracy spanning 24.4–75.3%. Three training seeds moved pooled accuracy by 6.5 points while leaving mean decision time within 1.4 ms.

The entropy stopping rule did not improve fixed-duration classification. For two classes, normalized posterior entropy is a monotone transform of maximum-softmax confidence, and the two ablation arms were numerically identical. The Transformer's raw posteriors were severely miscalibrated (15-bin expected calibration error 0.429; fitted temperature 11.2), and the temperature-scaled policy reached 64.2% forced fallback against 8.9% for the raw-posterior policy without improved accuracy. A frozen WavLM Base+ representation with a 1,538-parameter head reached 59.2% pooled balanced accuracy under the same folds, improved monotonically with prefix length, and was well calibrated (raw expected calibration error 0.044); its speaker-clustered 95% interval of 47.9–70.7% still includes chance.

The reconstruction also produced a methodological result. A label-free waveform diagnostic rejected two onset rules because they anchor at sample zero on 45.0% and 71.7% of utterances. Condition-matched retraining then showed those same rules *raising* pooled accuracy by 25.4 and 17.3 points. The entire gain lies in crops that begin at the recording start, which the rejected rules produce for 72.7% and 88.9% of control utterances but only 25.1% and 59.8% of dysarthric ones. A detector that fails differentially by class supplies a class-correlated shortcut, so detector diagnostics and downstream accuracy can point in opposite directions and must be reported separately. The system remains unsuitable for clinical use.

**Keywords:** dysarthric speech classification, early decision, speaker-independent evaluation, reproducibility, calibration, speech onset, posterior entropy, TORGO.

## 1. Introduction

Dysarthria is a motor speech disorder in which impaired neuromuscular control of the articulators affects articulation, phonation, resonance, prosody, and overall speech intelligibility. It arises from neurological conditions including cerebral palsy, amyotrophic lateral sclerosis, Parkinson's disease, and stroke, and varies substantially in type and severity between speakers. Because the disorder degrades the acoustic cues used by listeners and recognizers, people with dysarthria are frequently underserved by mainstream speech interfaces. Automatic systems may provide repeatable measurements, but they must contend with high inter-speaker variability and the small speaker counts of public dysarthric-speech corpora.

TORGO contains time-aligned acoustic and articulatory recordings from speakers with dysarthria and control speakers [1]. The present work uses its head-mounted-microphone recordings under a speaker-independent protocol, so the evaluation asks whether a model generalizes to unseen speakers rather than whether it recognizes speakers already observed during training.

Most prior systems optimize performance after receiving a complete utterance or a relatively long segment [2–5]. That is appropriate offline but less suited to interactive settings in which acoustic observation time matters. The motivating question is therefore narrow: how much utterance-level class information is present during the first 200–300 ms after a detected onset, and does a calibrated sequential confidence policy provide a better accuracy–earliness trade-off than fixed observation times?

Entropy and posterior confidence have long been used to quantify uncertainty [6,7], and temperature scaling provides a held-out correction for neural-posterior miscalibration [8]. Early time-series classification formalizes the joint accuracy–earliness problem [9–11]. Convolutional scaling, recurrent acoustic modelling, and Transformer encoders provide relevant architectural precedents [12–14]. These precedents motivate the evaluation protocol but do not by themselves establish effectiveness for short dysarthric-speech prefixes.

### 1.1. Why this paper is a reproducibility study

An earlier version reported 76.09% Transformer accuracy at 300 ms and 75.89% for an entropy-gated system at 264.6 ms mean decision time. During peer review, a reviewer recomputed the metrics implied by the published confusion matrix and found that they did not agree with the accompanying table: the matrix implied 74.96% accuracy rather than 75.89%, 73.21% dysarthric recall rather than 73.38%, and 73.67% macro-F1 rather than 72.82%.

The original training code, checkpoints, and per-sample prediction files no longer existed, so the discrepancy could not be traced and the values could not be regenerated. We therefore rebuilt the experiment from raw audio, froze the reconstruction protocol before inspecting test results, and replaced the earlier results. CNN-GRU and an “EfficientNet-lite” network named in the earlier version are withdrawn: no layer definition, variant identifier, checkpoint, or prediction file for either survives, and naming a variant after the fact would invent a record.

The rebuild also exposed a problem upstream of classification. The earlier detector used an absolute floor combined with a whole-utterance peak-relative threshold. On many head-mounted recordings this threshold was below the opening room tone, so it anchored at the first frame and a nominally onset-aligned prefix could contain little or no speech. Because no manual reference-onset annotation has been completed, the present paper reports this as a label-free detector diagnostic rather than a measured onset-localization error.

### 1.2. Contributions

1. A reproducible, label-free diagnostic of three RMS onset rules on TORGO, including exact framing, thresholds, confirmation logic, and the at-most-110-ms availability horizon of the adaptive threshold (Sections 3.1 and 5.1).
2. Evidence that detector diagnostics and classification accuracy can be anti-correlated: condition-matched retraining under the two rejected onset rules improves accuracy, but the entire improvement sits in recording-start crops that those rules produce at sharply class-dependent rates, making detector failure itself a leakage channel (Section 5.1).
3. A frozen speaker-independent reconstruction with per-utterance ledgers for three fully specified compact models and a frozen WavLM Base+ reference evaluated on identical folds and prefix durations (Sections 3–5).
4. Evidence that the compact spectral models do not exceed chance balanced accuracy under this protocol, whereas the pretrained reference shows a directionally better but statistically unresolved result (Section 5.2).
5. An explicit demonstration that normalized binary entropy and maximum-softmax confidence implement equivalent gates, together with held-out calibration, a complete `theta`–`epsilon` validation surface, fixed-duration comparisons, a three-seed replication, and a pre-roll ablation (Sections 3.3–3.4 and 5.3–5.4).
6. Robustness and error-structure analyses that quantify sensitivity to imposed onset shifts, additive noise, speaker, prompt type, duration, decision time, detector status, and entropy while avoiding unsupported causal explanations (Sections 5.5–5.6).

Section 2 reviews related work. Sections 3 and 4 describe methods and evaluation. Section 5 presents results, Section 6 defines the limits of inference, and Section 7 concludes.

## 2. Related Work

### 2.1. Dysarthric-speech corpora and speaker-independent evaluation

TORGO [1] and UA-Speech [15] are widely used dysarthric-speech corpora, but their small speaker counts make the evaluation protocol decisive. Results from utterance- or prompt-disjoint splits that retain the same speakers in training and testing do not estimate generalization to unseen speakers. A TORGO/UA-Speech audit showed that some classifiers performed as well or better on non-speech than on speech and identified recording-condition differences as a potential confound [16]. Speaker-independent work has used disjoint speaker sets, leave-one-speaker-out evaluation, articulatory and bottleneck features [17], and speaker-identity-invariant representations [18]. We consequently report exact speaker partitions, per-speaker results, pre-onset diagnostics, and the microphone stream.

### 2.2. Models and endpoints

Prior dysarthria systems include convolutional detectors based on pairwise distances [3], temporal-envelope and fine-structure cues [4], and clinically interpretable bottleneck layers [19]. Pretrained representations have been evaluated for detection and severity classification [20], recurrent models for sentence intelligibility [21], and STFT magnitude/phase features for detection and intelligibility [5,22]. A recent overview emphasizes that recognition, intelligibility, severity, and binary detection are distinct endpoints [23], while newer self-supervised work focuses on speaker-independent severity classification [24]. Efficient convolutional scaling [12], recurrent acoustic models [13], and Transformer encoders [14] motivated model families named in the earlier version; only architectures that can now be specified and executed are retained.

### 2.3. Early classification, uncertainty, and onset detection

The wait-or-decide problem is formalized in early time-series classification, where predictive quality and earliness are separate outcomes [9–11]. Entropy- or confidence-based deferral has precedent [6,7], while held-out temperature scaling addresses neural-posterior miscalibration [8]. For two classes, normalized entropy is a monotone transform of maximum-softmax confidence; the present contribution is therefore evaluated as a sequential stopping policy, not as a new uncertainty statistic. Short-time RMS energy is used only as an onset detector, following established voice-activity-detection practice [25].

### 2.4. Protocol-aware comparison

Table 1 compares representative studies along the dimensions that determine whether numerical results are comparable. `NR` means that the item was not reported or could not be verified from the evidence assembled for this revision; it is not inferred from another study using the same corpus.

**Table 1.** Protocol-aware comparison with representative prior dysarthria-detection studies. Results are descriptive within each source and are not ranked across rows. “Latency” denotes acoustic observation time unless a source explicitly reports compute time.

| Study and model | Corpus | Microphone | Segment / prediction unit | Speaker protocol | Latency definition | Reported result | Why not directly comparable |
|---|---|---|---|---|---|---|---|
| Millet and Zeghidour (2019) [26], raw-waveform filterbank, LSTM, attention | TORGO; 7 dysarthric, 7 control | NR | Recordings padded to 2.5 s; recording output | Fixed disjoint train/validation/test speakers; 3 initializations | Fixed 2.5-s input; no adaptive time | Test UAR 82.4 ± 0.4 | One fixed split, longer input, UAR, and no onset-aligned stopping policy |
| Hernandez et al. (2020) [27], RF/RBF-SVM/MLP with prosodic-rhythm features | TORGO, all 15 speakers, selected sentences; separate QoLT experiments | NR | Whole sentence; duration NR | Prompts separated, but every TORGO speaker occurs in train and test; 10-fold tuning plus test set | Whole sentence; not defined | Best TORGO detection accuracy 85.7%; F1 85.7% | Speaker-overlapping evaluation does not estimate unseen-speaker generalization |
| Wang et al. (2021) [28], CBRNN attention with domain-adversarial and mutual-information objectives | Labelled UA-Speech source; unlabelled TORGO target | NR | Whole-utterance mel spectrogram | Unsupervised domain adaptation using unlabelled target audio | Whole utterance; not defined | UDA improved utterance weighted recall 22.2 points and speaker accuracy 20.0 points over its baseline | Target-corpus access and improvement-from-baseline outcomes differ from held-out within-TORGO evaluation |
| Janbakhshi and Kodrasi (2022) [5], magnitude/phase CNNs | PC-GITA; 50 Parkinson's disease, 50 neurotypical | NR | 500-ms windows with 50% overlap; soft vote per speaker | Speaker-independent stratified 10-fold CV; 5 seeds and 5 speaker splits | No adaptive time; final decision pools windows | Magnitude + instantaneous-frequency CNN speaker ACC 93.68 ± 5.32; AUC 0.97 ± 0.05 | Different disease, corpus, language, and speaker-level aggregation |
| Schu et al. (2023) [16], SVM, Mel CNN/SRL, wav2vec 2.0 with confound audit | Matched TORGO subset; 7 dysarthric, 7 control; 62 utterances/speaker | NR | Mel models use 500-ms windows with 250-ms shift; speaker aggregation | Leave-one-speaker-out; training-speaker data split 90/10 | No online time; 500-ms windows feed speaker vote | SRL speaker accuracy: speech 71.1 ± 3.1, non-speech 100.0 ± 0.0, complete 91.1 ± 3.1 | Multiple windows per speaker and non-speech performance preclude a disease-only interpretation |
| Javanmardi et al. (2024) [20], frozen wav2vec 2.0/HuBERT embeddings with SVM/CNN | TORGO; 8 dysarthric, 7 control; non-words, words, sentences | Array microphone, 16 kHz | Complete variable-length utterance; temporal mean embedding | Leave-one-speaker-out; CNN reserves 10% of each training speaker's samples | Complete utterance; not defined | HuBERT L24 + SVM: ACC 76.12%, sensitivity 0.55, specificity 0.81, F1 0.57, EER 0.340 | Different microphone, full utterances, prompt mixture, and LOSO protocol |
| Xiong et al. (2025) [29], partially tuned wav2vec 2.0 with multitask learning | Private 200-h source; TORGO cross-corpus test | NR | Whole waveform; duration NR | Source partitions speaker-disjoint; no TORGO tuning | Whole sample; not defined | TORGO sample-level accuracy 74.18% | Cross-corpus training on a much larger private source corpus |
| **Present reconstruction**, compact log-STFT models and frozen WavLM Base+ [30,31] | TORGO LDC-release subset; 8 dysarthric, 7 control; 8,131 utterances | Head-mounted, mono 16 kHz | 200/250/300-ms post-anchor slots plus a fixed 30-ms pre-roll slot, with zero padding at waveform boundaries; utterance output | Five predefined speaker-disjoint folds; validation-speaker calibration and policy selection | Post-anchor acoustic observation separated from compute; candidates at 200/250/300 ms | Frozen WavLM fixed 300 ms: balanced ACC 59.18%, macro-F1 58.10%, raw ACC 59.29%; calibrated Transformer policy: raw ACC/macro-F1 41.95% at 269.4 ms, 64.2% fallback | Distinct short-prefix, channel, prediction-unit, and observation-time estimand; WavLM 95% speaker-clustered balanced-ACC interval includes chance |

No prior row is a matched numerical benchmark. The closest verified short-window studies use 500-ms segments and aggregate multiple segments to a speaker decision [5,16], while the closest modern TORGO self-supervised study uses complete array-microphone utterances [20]. Other high headline scores use a fixed 2.5-s input [26], speaker-overlapping TORGO splits [27], unlabelled target-corpus adaptation [28], or training on a much larger private corpus [29]. In the reviewed evidence, we did not identify a speaker-independent TORGO study using head-mounted-microphone, onset-aligned 200–300-ms prefixes and a validation-selected adaptive observation-time policy. This is a qualified protocol-gap statement, not a state-of-the-art claim.

## 3. Materials and Methods

Figure 1 summarises the reconstructed pipeline as eight stages, from the corpus through speaker-exclusive splitting, onset anchoring and cropping, log-STFT conversion, the four evaluated systems, calibration, and the two decision paths that are compared. It replaces the submitted overview diagram, which listed two model families that are withdrawn in this revision because no recoverable definition exists for them, omitted the pretrained reference, and showed 150 ms as a decision window rather than the mandatory predecessor it actually is.

![](artifacts/figures/figure_architecture.png){width=88%}

**Figure 1.** Overview of the reconstructed speech-onset time-entropy pipeline. Shaded stages are those the reviewers asked us to specify exactly: the onset rule and its eligible decision times (Section 3.1), and the four systems evaluated on identical folds and prefixes (Sections 3.5 and 3.6). The 150-ms prefix is a valid fixed-prefix evaluation but is never an adaptive emission point.


### 3.1. RMS onset detector and timing semantics

Every waveform is converted to mono at 16 kHz. For waveform $x[n]$, left-aligned short-time RMS is computed without centred padding or future reflection:

$$
r[m]=\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x[mH+n]^2}.
$$

The frame length is $N=20$ ms (320 samples) and the hop is $H=10$ ms (160 samples). Frame $m$ is active exactly when $r[m]\ge\tau$. With $K=2$ consecutive frames required, the anchor index is

$$
m^*=\min\{m:r[m]\ge\tau\ \text{and}\ r[m+1]\ge\tau\},
$$

and the returned onset anchor is $m^*H$. No temporal smoothing is applied. The two-frame run is the only protection against an isolated active frame.

The three detector modes are:

**Table 2.** Speech-onset thresholding modes and their role. The primary mode is used for every classification result; the other two appear only in the label-free waveform diagnostic of Section 5.1.

| Mode | Threshold | Role |
|---|---|---|
| `causal_noise_floor` (**primary**) | $\max(0.003,\operatorname{median}\{r[0],\ldots,r[q_x-1]\}\,10^{15/20})$, $q_x=\min(10,M_x)$ | Level-adaptive rule with bounded initial lookahead; $M_x$ is the number of available RMS frames; used for headline classification |
| `causal_fixed` | $0.003$ | Level-blind comparator in the label-free waveform diagnostic |
| `recovered_offline_peak` | $\max(0.003,(\max_m r[m])\,10^{-35/20})$ | Non-causal whole-utterance comparator corresponding to the earlier version |

The primary threshold is often described compactly as being estimated from the “first 100 ms.” Let $M_x$ be the number of complete RMS frames available for recording $x$ and $q_x=\min(\lceil100/10\rceil,M_x)=\min(10,M_x)$. For 8,127 of 8,131 recordings, $q_x=10$: the overlapping 20-ms frames start at 0, 10, ..., 90 ms and extend through 110 ms. Three short recordings provide four complete frames and one provides six, giving 50- and 70-ms threshold-availability times, respectively. In general,

$$
a_{\tau}(x)=(q_x-1)H+N\le110\ \text{ms}.
$$

The two-frame evidence for anchor $m^*H$ is available at $(m^*+K-1)H+N=m^*H+30$ ms. A streaming implementation therefore cannot emit the detected anchor before

$$
a^*=\max\{m^*H+30\ \text{ms},\ a_{\tau}(x)\}.
$$

This distinction matters: the anchor can be backdated to an earlier frame, but the threshold and two-frame confirmation are not available then. The released reconstruction's `confirmation_sample` field records the two-frame right edge and does not by itself include $a_{\tau}(x)$; it is therefore not used as a complete end-to-end availability timestamp for the adaptive mode. Section 5.7 separates post-anchor observation, detector computation, and this signal-availability condition.

If no two-frame run reaches the threshold, the row is recorded as a detector failure and a registered fallback anchor at sample zero is used. Such rows are retained in every denominator. Because reference onsets have not been manually annotated, detector evaluation in Section 5.1 is explicitly label-free: sample-zero anchors, crop energy, and post/pre-energy ratios are diagnostic indicators, not localization errors against ground truth.

### 3.2. Prefix definition and log-STFT representation

A nominal $t$-ms crop is a fixed $t+30$-ms input: a 30-ms pre-roll slot followed by a $t$-ms post-anchor slot. The loader fills those slots with available waveform samples and left- or right-zero-pads only when the requested interval crosses a waveform boundary. Away from a boundary, the pre-roll is already-observed buffered history, so it adds no future waiting; the nominal post-anchor observation time remains $t$.

The 30-ms value was retained as a bounded context design choice to reduce signal loss when an energy threshold anchors after weak onset energy; no performance benefit was assumed. Section 5.4 reports a condition-matched 0-ms versus 30-ms retraining comparison. Its balanced-accuracy interval includes zero, so the fixed value is not justified as an empirically superior operating point.

For the compact models, each crop is converted to a log-magnitude STFT with a 512-point FFT, a 25-ms periodic Hann window, a 10-ms hop, no centred padding, and per-utterance standardization. Prefixes of 150, 200, 250, and 300 ms yield 257 frequency bins and 16, 21, 26, and 31 time frames. The 150-ms prefix is a valid fixed-prefix evaluation and the warm-up input to the stability rule, but it is not an adaptive candidate.

### 3.3. Posterior entropy

Let $p(c\mid X_t)$ be the posterior probability of class $c$ after prefix $X_t$, and let $C$ be the class count. Here $C=2$ (control and dysarthric). Normalized entropy is

$$
H_t=-\frac{1}{\log C}\sum_{c=1}^{C}p(c\mid X_t)\log p(c\mid X_t).
$$

Division by $\log C=\log 2$ maps entropy exactly to $[0,1]$, where 0 denotes a fully concentrated posterior and 1 denotes a uniform posterior. This is an interpretive normalization on a fixed dimensionless scale.

For $C=2$, $H_t$ is a strictly decreasing function of $\max_c p(c\mid X_t)$. An entropy threshold is therefore equivalent to a maximum-softmax-confidence threshold. We retain the normalized form for interpretability but make no claim that it is an independent uncertainty statistic. The predicted class is $\hat{y}_t=\arg\max_c p(c\mid X_t)$.

### 3.4. Sequential decision rule and validation-only selection

The evaluated prefixes are $\{150,200,250,300\}$ ms with $\Delta=50$ ms. The eligible adaptive-decision set is $\{200,250,300\}$ ms. The 150-ms prediction is always computed and stored, but never emits an adaptive decision; it supplies the predecessor required at 200 ms.

The earliest eligible time $t$ is accepted only when all three conditions hold:

1. $H_t\le\theta$;
2. $\hat{y}_t=\hat{y}_{t-\Delta}$;
3. $|H_t-H_{t-\Delta}|\le\epsilon$.

If no candidate passes all conditions, the 300-ms prediction is emitted. There is no confidence-only exemption at 200 ms.

Within each fold, validation-speaker rows are deterministically divided by hashed sample identifier into checkpoint-selection (50%), calibration (25%), and policy-tuning (25%) subsets. After fitting temperature on the calibration subset, $\theta$ and $\epsilon$ are selected on policy-tuning rows only by exhaustive search over $\theta\in\{0.30,0.35,\ldots,0.95\}$ and $\epsilon\in\{0.00,0.02,\ldots,0.20\}$, for 154 cells per fold. Selection minimizes mean observation time subject to validation macro-F1 remaining within 0.01 absolute of that fold's fixed-300-ms validation macro-F1. Ties are resolved by higher macro-F1, lower fallback, lower $\theta$, and lower $\epsilon$, in that order. No test row participates in calibration or selection.

### 3.5. Compact model architectures and frame-wise tokens

All three compact models receive tensors shaped $B\times1\times257\times T$ and emit two logits ordered (control, dysarthric). Convolutions are bias-free when followed by batch normalization; the dropout probability is 0.30. Parameter counts were verified programmatically.

**Table 3.** Small CNN (23,538 trainable parameters).

| Stage | Exact operation | Output channels | Parameters |
|---|---|---:|---:|
| 1 | Conv2d 1→16, 3×3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2×2 | 16 | 176 |
| 2 | Conv2d 16→32, 3×3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2×2 | 32 | 4,672 |
| 3 | Conv2d 32→64, 3×3, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool 2×2 | 64 | 18,560 |
| Pool | Length-aware mean over valid frequency/time positions | 64 | 0 |
| Head | Dropout 0.30; Linear 64→2 | 2 | 130 |

**Table 4.** CNN–BiLSTM (69,354 trainable parameters).

| Stage | Exact operation | Output shape | Parameters |
|---|---|---|---:|
| 1 | Conv2d 1→24, 3×3, padding 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×24×128×T | 264 |
| 2 | Conv2d 24→48, 3×3, padding 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×48×64×T | 10,464 |
| Tokens | Mean over 64 remaining frequency bins; transpose | B×T×48 | 0 |
| Sequence | One-layer bidirectional LSTM, input 48, hidden 64 per direction | B×T×128 | 58,368 |
| Head | Masked temporal mean; Dropout 0.30; Linear 128→2 | 2 | 258 |

**Table 5.** Short-prefix Transformer (196,594 trainable parameters).

| Stage | Exact operation | Output shape | Parameters |
|---|---|---|---:|
| 1 | Conv2d 1→24, 3×3, padding 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×24×128×T | 264 |
| 2 | Conv2d 24→48, 3×3, padding 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×48×64×T | 10,464 |
| Tokens | Adaptive mean-pool frequency to 2 bins; flatten 48×2; bias-free Linear 96→96; sinusoidal position encoding | B×T×96 | 9,216 |
| Encoder | 2 encoder layers; dimension 96; 4 heads; feed-forward 256; GELU; dropout 0.30; post-norm | B×T×96 | 174,272 |
| Head | Masked temporal mean; Dropout; Linear 96→24; SiLU; Dropout; Linear 24→2 | 2 | 2,378 |

Both sequence front ends pool frequency only and never pool time. The CNN–BiLSTM therefore supplies one 48-dimensional token per valid STFT frame. The Transformer adaptively pools frequency to two bins, flattens 48×2 values into a 96-dimensional token, applies a bias-free 96→96 projection, and adds sinusoidal position encoding. It receives a key-padding mask; padded encoded states are cleared and excluded from temporal pooling. Thus prefix duration maps one-to-one to token-sequence length.

### 3.6. Frozen WavLM reference

WavLM Base+ is evaluated on the same folds and prefix durations [30,31]. The checkpoint is `microsoft/wavlm-base-plus`, pinned to immutable revision `4c66d4806a428f2e922ccfa1a962776e232d487b`. The input is the raw 16-kHz onset-aligned waveform with a sample-level attention mask. Masked-mean pooling feeds Dropout 0.30 and Linear 768→2. The backbone contains 94,381,936 parameters, all frozen; only the 1,538-parameter head is fitted. The resolved commit is recorded in each run summary. Fine-tuning was not evaluated.

The reference uses the same speaker folds, onset rule, pre-roll, prefix durations, class-balancing scheme, checkpoint metric, and calibration/policy partitions. The intended comparison changes representation, not the evaluation protocol.

### 3.7. Calibration

A scalar temperature is fitted per fold by minimizing negative log likelihood on the held-out calibration subset and is then applied to every prefix. Calibration is measured with negative log likelihood, Brier score, and 15-bin expected calibration error (ECE), before and after scaling. Policy results are reported for raw and scaled posteriors so calibration's effect is observed rather than assumed.

## 4. Experimental Setup

### 4.1. Dataset

The head-mounted-microphone subset of TORGO is selected by `usable_prompt=True` and duration at least 0.05 s, yielding 8,131 utterances from 15 speakers: 5,197 control (63.92%) and 2,934 dysarthric (36.08%), an imbalance ratio of 1.771:1.

**Table 6.** Speaker sex, published dysarthria severity, and selected utterance counts. Severity categories follow the Frenchay-based speaker mapping reported by Joy et al. [32]; utterance counts are recomputed from the reconstruction manifest.

| Class | Sex | Published severity | Speakers | Selected utterances |
|---|---|---|---|---:|
| Dysarthric | Female | Severe | F01 | 132 |
| Dysarthric | Female | Moderate | F03 | 565 |
| Dysarthric | Female | Mild | F04 | 246 |
| Dysarthric | Male | Severe | M01, M02, M04 | 1,084 |
| Dysarthric | Male | Severe–moderate | M05 | 490 |
| Dysarthric | Male | Mild | M03 | 417 |
| Control | Female | Not applicable | FC01, FC02, FC03 | 2,149 |
| Control | Male | Not applicable | MC01, MC02, MC03, MC04 | 3,048 |

**Table 7.** Prompt type derived with the same `classify_prompt_type` taxonomy used in the error audit.

| Prompt type | Control | Dysarthric | Total |
|---|---:|---:|---:|
| Isolated token | 3,735 | 2,141 | 5,876 (72.27%) |
| Sentence | 1,365 | 715 | 2,080 (25.58%) |
| Vocal exercise | 96 | 75 | 171 (2.10%) |
| Missing prompt text | 1 | 3 | 4 (0.05%) |

The index is dominated by isolated-token prompts. Per-speaker counts range from 132 (F01) to 1,108 (MC01). Female speakers contribute 943 dysarthric and 2,149 control utterances; male speakers contribute 1,991 and 3,048, respectively. Among the 2,934 dysarthric utterances, the published speaker labels yield 1,216 severe (41.45%), 490 severe–moderate (16.70%), 565 moderate (19.26%), and 663 mild (22.60%) rows. Severity is constant within each of only eight dysarthric speakers, so these are descriptive corpus strata rather than independent severity observations. The original TORGO report gives a cohort-level total of seven speakers with cerebral palsy and one with amyotrophic lateral sclerosis [1], but does not map those etiologies to speaker IDs; speaker-level etiology is therefore not imputed.

### 4.2. Speaker-independent folds

Every speaker appears in exactly one test partition and occupies only one role within a fold.

**Table 8.** Fold composition by speaker.

| Fold | Train | Validation | Test | Test utterances |
|---:|---|---|---|---:|
| 0 | F01, F03, FC02, FC03, M03, M04, MC01 | F04, M02, MC02, MC03 | FC01, M01, M05, MC04 | 1,689 |
| 1 | F03, FC01, FC02, FC03, M01, M04, M05, MC04 | F01, M03, MC01 | F04, M02, MC02, MC03 | 1,940 |
| 2 | F03, F04, FC01, FC03, M01, M02, M05, MC02, MC03, MC04 | FC02, M04 | F01, M03, MC01 | 1,657 |
| 3 | F01, F04, FC01, M01, M02, M03, M05, MC01, MC02, MC03, MC04 | F03, FC03 | FC02, M04 | 1,288 |
| 4 | F01, F04, FC02, M02, M03, M04, MC01, MC02, MC03 | FC01, M01, M05, MC04 | F03, FC03 | 1,557 |

Test partitions sum to 8,131 but are unequal (1,288–1,940). Pooled and unweighted-fold estimates are therefore reported separately.

### 4.3. Training configuration

**Table 9.** Training configuration frozen before primary test evaluation.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate / schedule | $1.5\times10^{-4}$, constant |
| Weight decay | $5\times10^{-4}$ |
| Batch size | 64 |
| Maximum epochs / early-stopping patience | 35 / 8 |
| Gradient clipping | 5.0 global norm |
| Dropout / label smoothing | 0.30 / 0.05 |
| Class balancing | Inverse-frequency weighted sampler with replacement, training partition only |
| Training augmentation | All models: waveform gain −3 to +3 dB and noise at 20–35 dB SNR ($p=0.5$); compact log-STFT models only: frequency mask up to 16 bins and time mask up to 4 frames ($p=0.5$) |
| Checkpoint metric | Speaker-balanced macro-F1 on checkpoint-selection validation rows |
| Reported seed | 20260819 |
| Replication seeds | 20260820 and 20260821, Transformer only; reported separately in Table 28 |

Inverse-frequency sampling draws one training-partition-sized epoch with equal expected class mass. Balancing and augmentation never affect validation or test denominators. Checkpoints are selected with speaker-balanced macro-F1 because the small validation partitions can otherwise be dominated by one speaker. The choice used training/validation curves only.

**Table 10.** Original training counts and realized per-epoch draws for the headline Transformer at seed 20260819. The weighted sampler draws with replacement; realized columns are means over the completed epochs in that fold.

| Fold | Original control / dysarthric | Draws per epoch | Completed epochs | Mean control draws | Mean dysarthric draws |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,094 / 1,408 | 4,502 | 15 | 2,252.2 | 2,249.8 |
| 1 | 2,799 / 1,735 | 4,534 | 19 | 2,265.1 | 2,268.9 |
| 2 | 3,095 / 2,091 | 5,186 | 35 | 2,583.9 | 2,602.1 |
| 3 | 3,211 / 2,075 | 5,286 | 24 | 2,643.1 | 2,642.9 |
| 4 | 3,392 / 1,493 | 4,885 | 15 | 2,443.7 | 2,441.3 |

The realized draw distribution is close to 50/50 in every fold, while the original training rows remain imbalanced. Sampling is applied only during training; validation and test retain their observed class frequencies.

Primary runs used Apple Silicon MPS, for which at least one backward operation has no deterministic implementation. Runs are seeded and artifacts are hashed, but the present paper does not claim bit-exact replay or completed multi-seed replication.

### 4.4. Estimands, uncertainty, and provenance

Two estimands are kept distinct:

- **Pooled out-of-fold:** one metric over all 8,131 test utterances, each represented once.
- **Unweighted fold mean ± sample SD:** equal-weight mean and sample standard deviation across five fold-specific values.

Speaker-clustered percentile bootstrap intervals use 10,000 replicates, resampling speakers rather than utterances and stratifying by class where specified. Utterance-level McNemar values are reported only as descriptive sensitivity checks because repeated utterances from a speaker are not independent.

For a common 300-ms horizon, consumed fraction is $\bar{t}/300$, timeliness is $1-\bar{t}/300$, and the joint harmonic score for a predictive score $s$ is

$$
\operatorname{HM}(s,\text{timeliness})=\frac{2s\,\text{timeliness}}{s+\text{timeliness}}.
$$

We report accuracy- and macro-F1-based versions alongside their constituent scores and latency; the harmonic summaries do not replace those primary quantities.

Each classification-result table within a system is generated from that system's canonical decision or fixed-prefix ledger. Onset diagnostics, profiling, robustness, pre-roll retraining, and error-stratification tables use their own separately hashed artifacts. This wording is deliberately narrower than claiming that every table in the paper comes from one file.

### 4.5. Prespecified robustness and error stratification

Clean-validation temperatures and policy parameters are held fixed during test-time white-noise and imposed-onset-shift analyses. Error strata are computed for speaker, prompt type, utterance duration, detector status, decision entropy, and decision time. These analyses are descriptive: the artifact manifest explicitly records that causal inference was not performed. A deterministic 20-false-positive/20-false-negative blinded review set was generated, but no human categories or reference-onset fields have been completed; no qualitative cause, SNR-proxy finding, or manually measured onset error is reported.

## 5. Results and Discussion

### 5.1. Label-free onset-detector diagnostics

A 600-utterance diagnostic sample (40 per speaker) compares the three detector modes using waveform statistics only. No class label, model output, or test metric enters the comparison.

**Table 11.** Label-free onset-detector behaviour on 600 utterances. “Room-tone prefix” is an operational indicator: the RMS of the resulting 300-ms crop is below 0.01. Neither this indicator nor a sample-zero anchor is a manually measured onset error.

| Mode | Anchor at sample 0 | Median anchor | Median crop RMS | Room-tone prefix | Post/pre energy |
|---|---:|---:|---:|---:|---:|
| `causal_fixed` | 71.7% | 0 ms | 0.0119 | 43.3% | +19.5 dB |
| **`causal_noise_floor`** | **4.7%** | 1,060 ms | **0.1713** | **3.5%** | **+25.3 dB** |
| `recovered_offline_peak` | 45.0% | 40 ms | 0.0149 | 27.5% | +15.2 dB |

![](artifacts/figures/figure_onset_detection.png){width=78%}

**Figure 2.** Rates of sample-zero anchors and low-energy 300-ms crops under each rule. These waveform indicators reveal threshold/recording interactions but do not replace human reference-onset annotation.

The fixed floor 0.003 corresponds to approximately −50.5 dBFS, below the −22 to −41 dBFS opening levels documented in the inspected examples. Initial frames can therefore exceed the fixed floor. In one MC04 example, the level remains near −41 dBFS for about 1.4 s before a much higher-energy segment, and the first ten frames exceed the floor. Some M05 recordings have high initial energy from sample zero. Without reference annotations, neither pattern is assigned a localization-error label.

The earlier peak-relative comparator is additionally non-causal because it uses the maximum RMS of the complete utterance. Its 45.0% sample-zero and 27.5% room-tone-prefix rates show that the earlier onset-alignment claim was unreliable, but they do not quantify localization error. The level-adaptive rule was selected from waveform-only diagnostics already specified in the frozen configuration and is used below.

Condition-matched retraining under the two comparator rules is now complete. Each comparator was retrained on all five folds with its own temperature and its own validation-selected policy, then evaluated on the identical 8,131 out-of-fold utterances. The result contradicts the waveform diagnostic.

**Table 12.** Condition-matched onset-rule retraining, pooled out of fold, complete calibrated policy. Every condition uses the same folds, the same 30-ms pre-roll, seed 20260819, and its own validation-only calibration and policy selection.

| Onset rule | Accuracy | Macro-F1 | Balanced accuracy | Dysarthric recall | Specificity | Mean time | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| `causal_noise_floor` (**primary**) | 41.95% | 41.95% | 45.74% | 59.37% | 32.11% | 269.39 ms | 64.25% |
| `causal_fixed` | 59.28% | 58.09% | 59.17% | 58.76% | 59.57% | 248.16 ms | 41.85% |
| `recovered_offline_peak` | 67.37% | 66.52% | 68.21% | 71.20% | 65.21% | 253.46 ms | 50.81% |

**Table 13.** Comparator minus primary on identical utterances, paired speaker-clustered 95% intervals. Intervals use 10,000 paired replicates over the 15 speaker clusters, stratified by class.

| Contrast | Δ accuracy | Δ macro-F1 | Δ balanced accuracy | Δ dysarthric recall | Δ specificity |
|---|---|---|---|---|---|
| `causal_fixed` − primary | +17.33 [+3.85, +28.97] | +16.14 [+4.15, +27.75] | +13.42 [+0.90, +25.44] | −0.61 [−18.11, +15.84] | +27.46 [+9.08, +44.87] |
| `recovered_offline_peak` − primary | +25.42 [+12.42, +37.14] | +24.57 [+12.54, +36.65] | +22.46 [+11.02, +33.74] | +11.83 [−1.53, +23.86] | +33.10 [+13.95, +52.37] |

Both comparators classify better than the primary rule, and both accuracy intervals exclude zero. Taken alone this would suggest that the rule discarded on waveform evidence was the better rule. It is not, and the reason is visible once the difference is stratified by what the detector actually did.

**Table 14.** Where each rule begins its crop. A crop begins at the recording start when the rule anchors at sample zero or registers a failure and falls back there.

| Rule | Crops from recording start | Control | Dysarthric | Class gap |
|---|---:|---:|---:|---:|
| `causal_noise_floor` (**primary**) | 4.86% | 6.39% | 2.15% | 4.24 points |
| `causal_fixed` | 78.42% | 88.94% | 59.78% | 29.15 points |
| `recovered_offline_peak` | 55.54% | 72.73% | 25.09% | 47.65 points |

**Table 15.** Where the accuracy difference lives. Differences are comparator minus primary within the stratum, in percentage points, on the same utterances.

| Contrast | Stratum | Utterances | Δ accuracy | Δ control | Δ dysarthric |
|---|---|---:|---:|---:|---:|
| `causal_fixed` − primary | Crop from recording start | 6,376 | +26.00 | +33.02 | +7.49 |
| `causal_fixed` − primary | Anchor after sample 0 | 1,755 | −14.13 | −17.22 | −12.63 |
| `recovered_offline_peak` − primary | Crop from recording start | 4,516 | +39.22 | +47.08 | −1.23 |
| `recovered_offline_peak` − primary | Anchor after sample 0 | 3,615 | +8.24 | −4.16 | +16.24 |

Three facts follow. First, the comparator advantage is confined to crops that begin at the recording start; where `causal_fixed` behaves as an onset detector and anchors after sample zero, it is 14.13 points *worse* than the primary rule. Second, within the recording-start stratum the gain falls almost entirely on control utterances — +33.02 against +7.49 for `causal_fixed`, and +47.08 against −1.23 for `recovered_offline_peak`. Third, and decisively, the comparators reach the recording start at sharply different rates for the two classes: a 29.15-point class gap for `causal_fixed` and a 47.65-point gap for `recovered_offline_peak`, against 4.24 points for the primary rule.

A rule that fails differentially by class hands the classifier a shortcut. "This input looks like the beginning of a recording" becomes a usable proxy for "control", and the measured accuracy gain reflects that channel rather than better speech-onset localization. TORGO's control and dysarthric material was collected in separate sessions, so recording-start context plausibly carries session and channel structure that is correlated with class; a confound audit of this corpus has previously reported classifier behaviour driven by recording condition rather than pathology [16]. The stratification establishes where the difference lives, not what inside those crops carries it, and this analysis does not identify the mechanism.

We therefore retain `causal_noise_floor` as the primary rule. It was fixed from waveform diagnostics recorded in the frozen configuration before any test evaluation, and re-designating a primary rule on the strength of these test results would be selection on the test set — the practice this reconstruction was undertaken to correct. The comparators are reported as the prespecified ablation they are.

The broader methodological point is that a label-free detector diagnostic and downstream classification accuracy can point in opposite directions. Sample-zero and room-tone rates correctly identify that a rule is not locating speech; they do not predict, and here are inversely related to, what that rule does to a classifier trained on its crops. Detector quality and task accuracy must be reported separately.

Manual reference-onset annotation remains incomplete, so Section 5.5 reports imposed-shift sensitivity rather than an empirical distribution of natural onset error.

### 5.2. Fixed-duration classification

**Table 16.** Fixed-prefix pooled out-of-fold results over all 8,131 utterances, all metrics. All four fixed prefixes are valid evaluations; 150 ms is not eligible under the adaptive rule.

| Model | Prefix | Raw accuracy | Macro-F1 | Dysarthric recall | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Small CNN | 150 ms | 44.12% | 44.06% | 65.34% | 48.74% |
| Small CNN | 200 ms | 43.18% | 43.14% | 63.50% | 47.60% |
| Small CNN | 250 ms | 44.45% | 44.42% | 58.59% | 47.53% |
| Small CNN | 300 ms | 44.32% | 44.30% | 58.79% | 47.47% |
| CNN–BiLSTM | 150 ms | 41.43% | 41.37% | 52.73% | 43.89% |
| CNN–BiLSTM | 200 ms | 42.66% | 42.42% | 50.07% | 44.28% |
| CNN–BiLSTM | 250 ms | 44.73% | 44.37% | 50.82% | 46.06% |
| CNN–BiLSTM | 300 ms | 46.61% | 46.09% | 50.95% | 47.56% |
| Transformer | 150 ms | 43.94% | 43.89% | 56.78% | 46.74% |
| Transformer | 200 ms | 43.07% | 43.04% | 56.65% | 46.03% |
| Transformer | 250 ms | 42.11% | 42.11% | 57.91% | 45.55% |
| Transformer | 300 ms | 41.94% | 41.93% | 60.05% | 45.88% |
| **WavLM Base+ (frozen)** | 150 ms | **52.50%** | **52.45%** | **67.96%** | **55.87%** |
| **WavLM Base+ (frozen)** | 200 ms | **55.36%** | **54.99%** | **64.25%** | **57.29%** |
| **WavLM Base+ (frozen)** | 250 ms | **56.95%** | **56.40%** | **63.22%** | **58.32%** |
| **WavLM Base+ (frozen)** | 300 ms | **59.29%** | **58.10%** | **58.79%** | **59.18%** |
| Constant control predictor | — | 63.92% | 38.99% | 0.00% | 50.00% |

No learned model beats the constant-control predictor in raw accuracy because all models are trained with class-balanced sampling rather than optimized for the natural 64/36 prior. That constant predictor has zero dysarthric recall and exactly 50% balanced accuracy. Balanced accuracy and macro-F1 are therefore necessary alongside raw accuracy.

At 300 ms, all three compact models remain at or below chance balanced accuracy. WavLM reaches 59.18% balanced accuracy and 58.10% macro-F1. Its raw accuracy rises monotonically from 55.36% at 200 ms to 59.29% at 300 ms; the Transformer declines over the same eligible range.

**Table 17.** Unweighted fold stability at fixed 300 ms.

| Model | Fold-mean raw accuracy | Fold SD |
|---|---:|---:|
| WavLM Base+ | 58.69% | 9.98 points |
| CNN–BiLSTM | 47.45% | 12.96 points |
| Transformer | 43.88% | 19.59 points |
| Small CNN | 46.39% | 21.35 points |

WavLM's fold SD is about half the Transformer's and small CNN's. This is descriptive evidence of greater fold stability, not a separate independent experiment.

![](artifacts/figures/figure_accuracy_earliness.png){width=78%}

**Figure 3.** Balanced accuracy at the fixed 200-, 250-, and 300-ms points matching the adaptive candidate times. The valid fixed-150 result is reported in Table 14 and in the standardized Transformer comparison in Table 22; it is omitted from this figure only because it is not an adaptive candidate. The adaptive-policy point is reported in Section 5.4.

**Table 18.** Per-fold raw accuracy at fixed 300 ms.

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Range |
|---|---:|---:|---:|---:|---:|---:|
| Small CNN | 46.8% | 21.4% | 55.0% | 76.7% | 32.0% | 55.3 points |
| CNN–BiLSTM | 60.9% | 28.0% | 53.3% | 53.9% | 41.2% | 32.8 points |
| Transformer | 48.0% | 24.4% | 39.3% | 75.3% | 32.4% | 50.9 points |
| WavLM Base+ | 72.3% | 60.4% | 62.7% | 51.2% | 46.9% | 25.3 points |

![](artifacts/figures/figure_fold_variance.png){width=78%}

**Figure 4.** Per-fold balanced accuracy at 300 ms. Within-model fold spread exceeds the separation among compact-model means.

Compact-model fold accuracy varies by up to 55.3 percentage points within one architecture, while the pooled differences between compact models are small. Architecture ranking is therefore unstable under this five-fold assignment.

The speaker-clustered 95% interval for WavLM fixed-300 balanced accuracy is 47.9–70.7%; it includes chance. The corresponding Transformer interval is 33.1–59.0%. WavLM is above 50% balanced accuracy on four of five folds and is directionally stronger on the pooled result, but the design cannot establish the advantage at conventional confidence.

### 5.3. Calibration

**Table 19.** Transformer calibration at the 300-ms prefix, pooled out of fold with 15 ECE bins.

| Posterior state | Negative log likelihood | Brier score | Expected calibration error |
|---|---:|---:|---:|
| Raw softmax | 1.393 | 0.446 | **0.429** |
| Temperature-scaled | 0.837 | 0.303 | 0.212 |

The fitted temperature is 11.23 ± 9.21 across folds and reaches the optimizer's upper bound of 20 in two folds. The raw probabilities are therefore unsuitable for interpretation as calibrated confidence.

**Table 20.** ECE and fitted temperature for the compact Transformer and pretrained reference.

| Model | Raw ECE | Temperature-scaled ECE | Fitted temperature |
|---|---:|---:|---:|
| Transformer | 0.4291 | 0.2117 | 11.23 |
| **WavLM Base+** | **0.0445** | 0.0593 | 5.45 |

![](artifacts/figures/figure_reliability.png){width=78%}

**Figure 5.** Pooled reliability diagrams at 300 ms. The Transformer is systematically overconfident; the frozen WavLM head is close to the diagonal before scaling.

WavLM's substantially smaller raw ECE shows that severe overconfidence is not inevitable in this task. The result is associated with better held-out-speaker performance here, but the analysis does not establish why the compact model is overconfident or prove a speaker-memorization mechanism.

### 5.4. Policy ablation, sensitivity, full fold results, and pre-roll

**Table 21.** Transformer policy ablation, pooled out of fold, all reported columns. Raw/scaled status is explicit. Consumed fraction and timeliness use a 300-ms horizon.

| Policy | Accuracy | Macro-F1 | Mean time | Fallback | Consumed fraction | Timeliness | H(acc,time) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed 200 ms | **43.07%** | **43.04%** | 200.0 ms | 0.0% | 0.667 | 0.333 | **0.376** |
| Fixed 250 ms | 42.11% | 42.11% | 250.0 ms | 0.0% | 0.833 | 0.167 | 0.239 |
| Fixed 300 ms | 41.94% | 41.93% | 300.0 ms | 0.0% | 1.000 | 0.000 | 0.000 |
| Entropy only, raw | 42.33% | 42.33% | 209.1 ms | 4.2% | 0.697 | 0.303 | 0.353 |
| Maximum-softmax confidence, raw | 42.33% | 42.33% | 209.1 ms | 4.2% | 0.697 | 0.303 | 0.353 |
| Entropy + class stability, raw | 42.28% | 42.28% | 212.9 ms | 5.3% | 0.710 | 0.290 | 0.344 |
| Entropy + entropy stability, raw | 41.99% | 41.99% | 221.4 ms | 7.3% | 0.738 | 0.262 | 0.323 |
| Complete policy, raw | 41.99% | 41.99% | 224.4 ms | 8.9% | 0.748 | 0.252 | 0.315 |
| Complete policy, scaled | 41.95% | 41.95% | 269.4 ms | 64.2% | 0.898 | 0.102 | 0.164 |

Entropy-only and maximum-softmax arms match to the reported precision, confirming their analytical equivalence. No policy component improves accuracy or macro-F1 over the fixed 200-ms operating point. When the raw and temperature-scaled arms are each selected fold-wise under the same validation-only protocol, forced fallback is 8.9% and 64.2%, respectively; this is not a same-threshold causal comparison.

**Table 22.** Standardized Transformer accuracy–earliness comparison on a common 300-ms horizon. The 150-ms row is a valid fixed-prefix evaluation but is not eligible for adaptive emission. The adaptive row combines independently validation-selected fold policies; it is not one global $(\theta,\epsilon)$ pair. “Consumed” is the fraction of the 300-ms horizon used; “HM acc.” and “HM F1” are the harmonic means of timeliness with accuracy and with macro-F1.

| Policy | Mean observation | Consumed | Timeliness | Accuracy | Macro-F1 | HM acc. | HM F1 | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 150 ms | 150.0 ms | 0.500 | 0.500 | 43.94% | 43.89% | 0.468 | 0.467 | 0.0% |
| Fixed 200 ms | 200.0 ms | 0.667 | 0.333 | 43.07% | 43.04% | 0.376 | 0.376 | 0.0% |
| Fixed 250 ms | 250.0 ms | 0.833 | 0.167 | 42.11% | 42.11% | 0.239 | 0.239 | 0.0% |
| Fixed 300 ms | 300.0 ms | 1.000 | 0.000 | 41.94% | 41.93% | 0.000 | 0.000 | 0.0% |
| Complete calibrated adaptive | 269.4 ms | 0.898 | 0.102 | 41.95% | 41.95% | 0.164 | 0.164 | 64.2% |

![](artifacts/sensitivity/transformer_complete_calibrated/figure_accuracy_earliness_transformer.png){width=78%}

**Figure 6.** Transformer accuracy–earliness comparison on the common 300-ms horizon. The fixed-prefix curve and adaptive point use the same 8,131 pooled out-of-fold utterances. The adaptive point is plotted at 269.39 ms and combines fold-specific validation selections rather than one global threshold pair.

Fixed 150 ms has the highest accuracy and harmonic score on this prespecified scale, although it is not an adaptive candidate because the stability rule needs it as the predecessor to 200 ms. The adaptive policy does not improve the fixed-duration accuracy–earliness frontier. Exact values are supplied in `artifacts/sensitivity/transformer_complete_calibrated/accuracy_earliness_exact.csv`.

**Table 23.** Complete validation-only sensitivity summary and selected cell for each fold.

| Fold | Fixed-300 macro-F1 | Macro-F1 range, 154 cells | Mean-time range | Selected theta | Selected epsilon | Selected macro-F1 | Selected accuracy | Selected mean time | Selected fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.787122 | 0.784278–0.789126 | 212.626–300.000 ms | 0.95 | 0.20 | 0.786278 | 0.822222 | 212.626 ms | 0.068687 |
| 1 | 0.200000 | 0.200000–0.200000 | 300.000–300.000 ms | 0.30 | 0.00 | 0.200000 | 0.250000 | 300.000 ms | 1.000000 |
| 2 | 0.655207 | 0.655207–0.664027 | 228.451–300.000 ms | 0.95 | 0.20 | 0.664027 | 0.696970 | 228.451 ms | 0.168350 |
| 3 | 0.465923 | 0.465923–0.465923 | 300.000–300.000 ms | 0.30 | 0.00 | 0.465923 | 0.478372 | 300.000 ms | 1.000000 |
| 4 | 0.538156 | 0.538156–0.538156 | 300.000–300.000 ms | 0.30 | 0.00 | 0.538156 | 0.563725 | 300.000 ms | 1.000000 |

![](artifacts/sensitivity/transformer_complete_calibrated/figure_theta_epsilon_validation_surface.png){width=78%}

**Figure 7.** Complete calibrated Transformer validation response surfaces. Colour is the change in validation macro-F1 relative to the fold-specific fixed-300-ms value, black contours are mean observation time, and the gold star is the validation-selected cell. Every one of the 154 cells per fold satisfies the within-0.01 macro-F1 constraint. Folds 1, 3, and 4 remain at 300 ms with 100% fallback throughout the grid; only folds 0 and 2 show an earliness response. Test data were not used to construct or select from these surfaces. The exact 770-row surface is supplied as `artifacts/sensitivity/transformer_complete_calibrated/theta_epsilon_surface_exact.csv`.

The grid therefore does not identify one universal threshold. The pooled adaptive result combines five fold-specific validation selections; it is not a test evaluation at one global `(theta,epsilon)` pair.

**Table 24.** Complete calibrated Transformer policy by test fold.

| Fold | n | Accuracy | Macro-F1 | Dysarthric recall | Balanced accuracy | Mean time | Forced fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,689 | 47.90% | 43.78% | 72.26% | 46.95% | 226.11 ms | 10.95% |
| 1 | 1,940 | 24.43% | 22.26% | 61.38% | 33.60% | 300.00 ms | 100.00% |
| 2 | 1,657 | 39.41% | 39.36% | 55.01% | 43.34% | 225.11 ms | 15.33% |
| 3 | 1,288 | 75.31% | 71.52% | 85.03% | 78.73% | 300.00 ms | 100.00% |
| 4 | 1,557 | 32.43% | 31.42% | 27.96% | 31.47% | 300.00 ms | 100.00% |
| **Unweighted mean ± SD** | — | **43.90 ± 19.58%** | **41.67 ± 18.59%** | **60.33 ± 21.39%** | **46.82 ± 18.98%** | **270.24 ± 40.75 ms** | **65.26 ± 47.60%** |

**Table 25.** Pooled complete calibrated Transformer estimates with 95% speaker-clustered bootstrap intervals.

| Metric | Pooled estimate | 95% interval |
|---|---:|---:|
| Accuracy | 41.95% | 30.42–55.19% |
| Macro-F1 | 41.95% | 29.49–54.73% |
| Dysarthric recall | 59.37% | 42.13–79.10% |
| Specificity | 32.11% | 14.47–50.52% |
| Dysarthric precision | 33.06% | 23.75–44.72% |
| Balanced accuracy | 45.74% | 33.21–58.91% |
| Mean decision time | 269.39 ms | 247.57–288.77 ms |
| Forced fallback | 64.25% | 39.20–86.65% |

The wide latency and fallback intervals show that early acceptance varies materially across speakers and folds.

**Table 26.** Complete calibrated policy minus fixed baselines on identical utterances. Difference intervals are speaker-clustered bootstrap intervals. McNemar $p$ is an unclustered, utterance-level descriptive sensitivity value and is not the primary inference.

| Fixed baseline | $\Delta$ accuracy, points [95% interval] | $\Delta$ dysarthric recall, points [95% interval] | $\Delta$ mean time [95% interval] | Descriptive McNemar $p$ |
|---|---|---|---|---:|
| 200 ms | −1.12 [−3.67, +0.95] | +2.73 [+0.59, +4.69] | +69.39 ms [+47.75, +88.82] | 0.0009 |
| 250 ms | −0.16 [−1.93, +1.17] | +1.47 [−0.20, +2.96] | +19.39 ms [−2.29, +39.10] | 0.536 |
| 300 ms | +0.01 [−0.41, +0.44] | −0.68 [−1.54, −0.03] | −30.61 ms [−51.94, −10.91] | 1.000 |

Relative to fixed 300 ms, the policy saves observation time but slightly reduces dysarthric recall, with accuracy differences spanning zero. Relative to fixed 200 ms, it is slower and its accuracy interval again spans zero. Non-significant comparisons are not interpreted as equivalence.

**Table 27.** Condition-matched complete calibrated Transformer policy with 0-ms versus 30-ms pre-roll, seed 20260819. Differences are 0 ms minus 30 ms with 95% paired speaker-clustered bootstrap intervals.

| Metric | 0-ms pre-roll | 30-ms pre-roll | Difference, 95% interval |
|---|---:|---:|---:|
| Accuracy | 50.98% | 41.95% | +9.03 points [0.09, 19.79] |
| Macro-F1 | 50.29% | 41.95% | +8.35 points [−0.37, 19.95] |
| Dysarthric recall | 54.40% | 59.37% | −4.98 points [−11.90, 0.14] |
| Specificity | 49.05% | 32.11% | +16.93 points [3.42, 34.40] |
| Balanced accuracy | 51.72% | 45.74% | +5.98 points [−1.62, 15.23] |
| Mean time | 269.25 ms | 269.39 ms | −0.14 ms [−1.86, 1.06] |
| Forced fallback | 63.24% | 64.25% | −1.01 points [−2.73, 0.51] |

Zero pre-roll improves raw accuracy and specificity in this single-seed comparison, but the balanced-accuracy and macro-F1 intervals include zero and dysarthric recall moves in the opposite direction. The experiment does not support the original claim that 30 ms preserves discriminative onset information or improves classification; it instead reveals a class trade-off that requires replication.

The Transformer was also retrained end to end under two further seeds on the identical folds, each with its own calibration and validation-only policy selection.

**Table 28.** Complete calibrated Transformer policy under three training seeds, pooled out of fold. Each seed is a separate five-fold training run over the same 8,131 utterances. The interval is a descriptive three-seed *t* interval over seed means; it is not a speaker-generalization interval and is not comparable with the speaker-clustered intervals elsewhere in this paper.

| Seed | Accuracy | Macro-F1 | Balanced accuracy | Dysarthric recall | Specificity | Mean time | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260819 (primary) | 41.95% | 41.95% | 45.74% | 59.37% | 32.11% | 269.39 ms | 64.25% |
| 20260820 | 48.41% | 48.03% | 49.89% | 55.21% | 44.56% | 269.79 ms | 62.55% |
| 20260821 | 46.76% | 46.34% | 48.03% | 52.59% | 43.47% | 270.72 ms | 64.11% |
| Mean ± SD | 45.71 ± 3.35% | 45.44 ± 3.14% | 47.89 ± 2.08% | 55.73 ± 3.42% | 40.05 ± 6.89% | 269.97 ± 0.68 ms | 63.64 ± 0.94% |
| Descriptive 95% interval | 37.37–54.04% | 37.64–53.24% | 42.73–53.05% | 47.23–64.22% | 22.93–57.17% | 268.27–271.67 ms | 61.29–65.98% |

Seed choice moves accuracy by 6.46 points and specificity by 12.45 points, which is large relative to the differences between compact architectures in Table 12. It does not change the conclusion: the three-seed mean balanced accuracy is 47.89% and its interval spans 50%, so the calibrated adaptive Transformer remains at or below chance under every seed. Mean decision time and forced-fallback rate are stable across seeds, so the policy's timing behaviour — unlike its accuracy — is not seed-sensitive. The primary seed 20260819 is the least favourable of the three on accuracy; reporting it as the headline is therefore conservative rather than selective.

### 5.5. Test-time perturbation sensitivity

The clean-validation temperature and policy parameters are applied unchanged to every test perturbation. These are test-time sensitivity analyses, not independently trained robustness systems.

**Table 29.** Transformer test-time perturbations, pooled out of fold, complete calibrated policy.

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
| Anchor shift +10 ms | 43.44% | 43.39% | 56.00% | +1.49 points |
| Anchor shift +30 ms | 41.63% | 41.60% | 54.74% | −0.32 points |
| Anchor shift +50 ms | 40.97% | 40.96% | 55.11% | −0.98 points |

Additive white noise at 20 dB SNR *raises* pooled accuracy from 41.95% to 47.86%, and anchor shifts are asymmetric: −50 ms costs 4.30 accuracy points while +50 ms costs 1.09. A system reading speech should not improve when noise is added, so this indicates class-bias sensitivity rather than robustness.

![](artifacts/figures/figure_robustness.png){width=78%}

**Figure 8.** Change in pooled accuracy under each prespecified perturbation relative to the clean reference. Negative anchor shifts are more damaging than positive shifts in this experiment. Moderate added noise increases raw accuracy.

The zero-shift condition reproduces the clean ledger. Shifting the anchor 30–50 ms early reduces accuracy by 6.1–6.7 points, whereas shifts of the same size later have a smaller effect. These imposed offsets establish sensitivity and asymmetry; because natural reference-onset error has not been annotated, they do not estimate real detector-error prevalence.

The 2.6-point improvement at 10 dB white-noise SNR is counterintuitive and signals instability or class-bias sensitivity. It cannot be interpreted as robustness, and the perturbation experiment does not identify a causal mechanism. Synthetic white noise is not evidence of cross-corpus or real far-field performance.

### 5.6. False positives, error strata, and speaker behaviour

**Table 30.** Pooled Transformer confusion matrix, complete calibrated policy.

| | Predicted control | Predicted dysarthric | Row total |
|---|---:|---:|---:|
| True control | 1,669 | **3,528** | 5,197 |
| True dysarthric | 1,192 | 1,742 | 2,934 |
| **Total** | 2,861 | 5,270 | **8,131** |

The matrix yields `(1669+1742)/8131 = 41.95%` accuracy. The same decision rows yield 269.39-ms mean observation, preventing a mismatch between metrics and decision counts.

![](artifacts/figures/figure_confusion.png){width=78%}

**Figure 9.** Pooled out-of-fold confusion matrices for the complete calibrated policy, shaded by row proportion. The compact Transformer's largest error cell is control predicted dysarthric; the pretrained representation has more diagonal mass.

The Transformer false-positive rate is 3,528/5,197 = 67.89%. Table 31 stratifies that numerator without assigning causes.

**Table 31.** Descriptive false-positive strata for the complete calibrated Transformer. Each denominator contains actual-control rows in that stratum. Strata are associations and overlap across stratifiers.

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
| Duration | 0.5–<1 s | 6 / 12 | 50.00% |
| Duration | 1–<2 s | 1,243 / 1,895 | 65.59% |
| Duration | 2–<5 s | 2,050 / 2,981 | 68.77% |
| Duration | ≥5 s | 229 / 309 | 74.11% |
| Decision entropy | 0.2–<0.4 | 170 / 174 | 97.70% |
| Decision entropy | 0.4–<0.6 | 214 / 248 | 86.29% |
| Decision entropy | 0.6–<0.8 | 435 / 584 | 74.49% |
| Decision entropy | ≥0.8 | 2,709 / 4,191 | 64.64% |

The strongest associations are detector-failure rows at 91.87% against 66.25% where an anchor was detected, and sentence prompts at 76.92% against 64.69% for isolated tokens. These are associations; the audit cannot show that any of them causes the error.

One actual-control row had missing prompt text and was retained in the overall denominator but is not shown as a one-row prompt stratum. Rates are especially high among registered detector failures and sentence prompts, but those associations may reflect speaker, channel, or other correlated structure. They neither prove that these factors cause an error nor show that removing one would correct it. The generated blinded 40-example sheet remains unlabelled, so no human qualitative taxonomy or representative-cause claim is reported.

**Table 32.** Per-speaker Transformer behaviour under the complete calibrated policy. Each TORGO speaker belongs to one class; accuracy on that class is reported with the predicted-dysarthric rate.

| Speaker | Class | Utterances | Share | Accuracy | Predicted dysarthric |
|---|---|---:|---:|---:|---:|
| MC02 | control | 464 | 5.7% | 1.7% | 98.3% |
| MC03 | control | 826 | 10.2% | 8.1% | 91.9% |
| MC04 | control | 650 | 8.0% | 20.9% | 79.1% |
| F04 | dysarthric | 246 | 3.0% | 24.0% | 24.0% |
| FC01 | control | 163 | 2.0% | 24.5% | 75.5% |
| F03 | dysarthric | 565 | 6.9% | 28.0% | 28.0% |
| MC01 | control | 1,108 | 13.6% | 31.7% | 68.3% |
| FC03 | control | 992 | 12.2% | 35.0% | 65.0% |
| M03 | dysarthric | 417 | 5.1% | 43.6% | 43.6% |
| M05 | dysarthric | 490 | 6.0% | 58.2% | 58.2% |
| FC02 | control | 994 | 12.2% | 72.4% | 27.6% |
| M02 | dysarthric | 404 | 5.0% | 84.2% | 84.2% |
| M04 | dysarthric | 294 | 3.6% | 85.0% | 85.0% |
| M01 | dysarthric | 386 | 4.7% | 90.2% | 90.2% |
| F01 | dysarthric | 132 | 1.6% | 90.9% | 90.9% |

Six of the seven control speakers are predicted dysarthric for more than half their utterances, ranging from 65.0% to 98.3%, with FC02 the single exception at 27.6%.

**Table 33.** Dysarthric recall by published speaker-level severity, complete calibrated Transformer. Severity categories are the Frenchay-based speaker mapping of Joy et al. [32]; every dysarthric speaker carries one category for all of their utterances.

| Published severity | Speakers | Selected dysarthric utterances | Correctly predicted dysarthric | Dysarthric recall |
|---|---|---:|---:|---:|
| Severe | F01, M01, M02, M04 | 1,216 | 1,058 | 87.01% |
| Severe–moderate | M05 | 490 | 285 | 58.16% |
| Moderate | F03 | 565 | 158 | 27.96% |
| Mild | F04, M03 | 663 | 241 | 36.35% |
| All | Eight dysarthric speakers | 2,934 | 1,742 | 59.37% |

Recall is highest for the severe category and lowest for the single moderate speaker, but the ordering is not monotone in severity and each category contains one or two speakers. Because severity is constant within a speaker, this table cannot separate a severity effect from a speaker effect; it is a descriptive stratification, not a severity-prediction result.

![](artifacts/figures/figure_speaker_bias.png){width=78%}

**Figure 10.** Fraction of each speaker's utterances predicted dysarthric by the Transformer, grouped by true class. Six of seven control speakers lie above 50%.

False positives therefore extend across most control speakers rather than being confined to one speaker. The rates remain highly heterogeneous—27.6% for FC02 and 65.0–98.3% for the other six controls. The supported conclusion is broad dysarthric-class prediction bias with several descriptive correlates, not a unique causal explanation.

### 5.7. Model complexity and latency accounting

**Table 34.** Model complexity and measured pipeline compute. Parameter, serialized-weight, MAC, and FLOP columns use a maximum 300-ms post-anchor prefix. The CPU total-compute and peak-process-RSS columns come from the batch-one 200-ms decision path, whose input tensor spans 230 ms including pre-roll. THOP treats one multiply–accumulate as two FLOPs. Peak RSS is a process-lifetime high-water-mark increment, not isolated model memory. WavLM was not profiled for runtime, memory, or operations in this revision.

| Model | Total parameters | Trainable parameters | Serialized weights | MACs | FLOPs | CPU total compute, 200-ms decision | Peak RSS increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small CNN | 23,538 | 23,538 | 100 KiB | 18.26 M | 36.52 M | 2.14 ms | 5.2 MB |
| CNN–BiLSTM | 69,354 | 69,354 | 279 KiB | 44.70 M | 89.41 M | 3.69 ms | 14.9 MB |
| Transformer | 196,594 | 196,594 | 783 KiB | 46.24 M | 92.49 M | 3.63 ms | 14.5 MB |
| WavLM Base+ + head | 94,383,474 | 1,538 | Not profiled | Not profiled | Not profiled | Not profiled | Not profiled |

**Table 35.** Measured component compute and post-anchor observation for a Transformer decision at 200 ms, 100 repetitions. The first row is CPU/device computation for the RMS algorithm, not acoustic onset-confirmation delay.

| Component | CPU | MPS |
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
| **Observation + measured compute** | **203.629 ms** | **207.732 ms** |

Total warm batch-one compute is far below the acoustic observation time, so end-to-end latency in this system is dominated by waiting for audio, not by computation.

For the adaptive initial-level rule, threshold availability is $a_{\tau}(x)\le110$ ms from recording start (110 ms for 8,127 recordings; 50 or 70 ms for the four short recordings), and a two-frame run is unavailable before 30 ms after its returned anchor. The 200-ms candidate occurs later than both bounds for any nonnegative anchor, so those conditions do not add beyond the 200-ms post-anchor wait; nevertheless, “200 ms” is a post-anchor observation metric, not elapsed time from the beginning of a recording session. At a 200-ms decision, the model input tensor spans exactly 230 ms, including the 30-ms pre-roll slot and any boundary padding. The pre-roll adds no future wait.

For this compact Transformer, measured CPU total compute is 3.629 ms. MPS is slower at batch size 1 because launch/transfer overhead exceeds the small forward-pass cost. The sub-1-MB and sub-4-ms statements apply only to the three compact models; no equivalent WavLM deployment profile was measured.

## 6. Limitations and Interpretation

The findings support a narrow reconstruction, not a deployable detector.

**The pretrained-model direction is unresolved.** WavLM's pooled fixed-300 balanced accuracy is 59.18%, but its speaker-clustered 95% interval is 47.9–70.7% and includes 50%. It falls below chance on one fold. Pooled performance, fold stability, calibration, and prefix response are converging descriptive indicators, not independent replications or proof of an advantage.

**Seed variability is large and only three seeds were run.** Accuracy moves 6.46 points across seeds (Table 25) and the three-seed interval is descriptive, computed over three means rather than over speakers. Only the Transformer was replicated; the other systems use seed 20260819 alone. Apple MPS also prevents a claim of bit-exact replay.

**Onset localization has not been manually validated.** The sample-zero, low-crop-energy, and post/pre-energy measures are label-free diagnostics. The primary implementation ID, `causal_noise_floor`, should not be read as proof that its initial reference frames contain only noise: they can contain speech. Prose therefore describes it as an adaptive initial-level rule with at most 110-ms bounded lookahead. Reference-onset annotation remains incomplete, so the comparator retraining in Section 5.1 shows where an accuracy difference lives without establishing which rule locates speech more accurately.

**The pre-roll comparison is single-seed and class-dependent.** Removing 30 ms increased specificity and raw accuracy while slightly reducing dysarthric recall; the balanced-accuracy interval spanned zero. The study does not establish that either 0 or 30 ms is universally preferable.

**The corpus and channel are narrow.** The evaluation covers 15 TORGO speakers, one head-mounted-microphone stream, and predominantly isolated-token prompts. Severity is available only through an external published speaker mapping and is inseparable from speaker identity in this eight-speaker dysarthric cohort. Etiology is reported only at cohort level because the verified sources do not map it to speaker IDs. The small, uneven validation speaker sets make checkpoint, calibration, and policy selection unstable.

**Cross-corpus, conversational, mobile, and far-field validation were not performed.** Synthetic white noise is not a substitute. Claims are restricted to the evaluated TORGO head-mounted channel and speaker partitions.

**The error audit is descriptive and incomplete qualitatively.** Speaker, prompt, duration, detector status, entropy, and time strata show associations. They do not identify causes. The blinded example sheet contains no completed human labels, and no SNR-proxy or representative-spectrogram conclusion is reported.

**Absolute performance is not clinically useful.** WavLM's raw accuracy remains below the constant-control raw-accuracy baseline, and the compact Transformer has a 67.9% false-positive rate. The task is corpus-defined utterance classification, not neurological diagnosis or severity estimation. Clinical use would require prospective validation not attempted here.

**Only frozen WavLM was evaluated.** Fine-tuning, larger pretrained encoders, and WavLM runtime/FLOPs were not studied.

## 7. Conclusion

Peer review exposed an inconsistency between the earlier confusion matrix and reported metrics. Because the original code and prediction records no longer existed, we rebuilt the experiment and replaced those results rather than attempting to explain numbers that could not be regenerated.

The reconstruction changes the conclusion. Label-free waveform diagnostics show that the earlier peak-relative onset rule often anchors at sample zero and can yield low-energy 300-ms crops, but no manual reference-onset error rate is claimed. Retraining under those rejected rules then raised accuracy rather than lowering it, and the entire gain proved to sit in recording-start crops that the rules produce far more often for control than for dysarthric speakers. Detector failure was itself the signal. A detector diagnostic and a downstream accuracy number therefore answer different questions, and neither substitutes for the other. Under the adaptive bounded-lookahead rule, all three compact spectral models remain at or below chance balanced accuracy and vary sharply across speaker folds. Normalized entropy is equivalent to maximum-softmax confidence for two classes; policy ablations show no accuracy benefit, and the separately validation-selected temperature-scaled arm has substantially more fallback than the raw-posterior arm. Imposed early anchor shifts reduce performance, while moderate noise produces a counterintuitive accuracy increase that indicates instability rather than robustness.

Frozen WavLM Base+ performs directionally better, reaches 59.18% pooled balanced accuracy at 300 ms, is better calibrated, and improves as the fixed prefix lengthens. Its speaker-clustered interval includes chance, so the evidence suggests—without establishing—that pretrained representations use the short-prefix signal more effectively than the compact models studied here.

We therefore do not claim a working minimal-time dysarthria detector. The supported contribution is a reproducible negative result, an explicit accounting of onset and calibration pitfalls, and a protocol for testing accuracy and earliness without mixing pooled and fold-level estimands. The next necessary steps are completed multi-seed replication, manual reference-onset annotation, condition-matched onset-rule retraining, leave-one-speaker-out or larger-speaker evaluation, and external-corpus/channel validation.

## Data and code availability

The reconstruction code, configurations, fold manifests, per-utterance ledgers, policy grids, and analysis artifacts are held in the version-controlled project repository used for this revision. They are not yet publicly hosted. On acceptance the authors will deposit a privacy-screened archive—source, configurations, fold and cohort manifests, per-utterance ledgers, policy grids, robustness and sensitivity tables, error strata, runtime profiles, and a SHA-256 manifest of every file—in a public archive with a citable DOI, and will report that identifier in the published version. The archive excludes TORGO audio, model checkpoints, and the incomplete human-annotation worksheets; TORGO itself remains available from its distributors under their existing licence [1].

Classification tables within each system are generated from its canonical prediction ledger; onset diagnostics, profiling, robustness, pre-roll retraining, sensitivity, and error-stratification results use separately hashed source artifacts. Each artifact records its producer commit and available configuration/dependency hashes.

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

[3] Janbakhshi, P., Kodrasi, I., and Bourlard, H. (2021). Automatic dysarthric speech detection exploiting pairwise distance-based convolutional neural networks. In *ICASSP 2021—2021 IEEE International Conference on Acoustics, Speech and Signal Processing* (pp. 7328–7332). IEEE. https://doi.org/10.1109/ICASSP39728.2021.9413922

[4] Kodrasi, I. (2021). Temporal envelope and fine structure cues for dysarthric speech detection using CNNs. *IEEE Signal Processing Letters*, 28, 1853–1857. https://doi.org/10.1109/LSP.2021.3108509

[5] Janbakhshi, P., and Kodrasi, I. (2022). Experimental investigation on STFT phase representations for deep learning-based dysarthric speech detection. In *ICASSP 2022—2022 IEEE International Conference on Acoustics, Speech and Signal Processing* (pp. 6477–6481). IEEE. https://doi.org/10.1109/ICASSP43922.2022.9747205

[6] Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

[7] Chen, T.-H., Chen, B., and Wang, H.-M. (2006). On using entropy information to improve posterior probability-based confidence measures. In Q. Huo, B. Ma, E.-S. Chng, and H. Li (Eds.), *Chinese Spoken Language Processing: 5th International Symposium, ISCSLP 2006, Proceedings* (Lecture Notes in Computer Science, Vol. 4274, pp. 454–463). Springer. https://doi.org/10.1007/11939993_48

[8] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. In D. Precup and Y. W. Teh (Eds.), *Proceedings of the 34th International Conference on Machine Learning* (Proceedings of Machine Learning Research, Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

[9] Xing, Z., Pei, J., and Yu, P. S. (2012). Early classification on time series. *Knowledge and Information Systems*, 31(1), 105–127. https://doi.org/10.1007/s10115-011-0400-x

[10] Mori, U., Mendiburu, A., Dasgupta, S., and Lozano, J. A. (2018). Early classification of time series by simultaneously optimizing the accuracy and earliness. *IEEE Transactions on Neural Networks and Learning Systems*, 29(10), 4569–4578. https://doi.org/10.1109/TNNLS.2017.2764939

[11] Schäfer, P., and Leser, U. (2020). TEASER: Early and accurate time series classification. *Data Mining and Knowledge Discovery*, 34(5), 1336–1362. https://doi.org/10.1007/s10618-020-00690-z

[12] Tan, M., and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In K. Chaudhuri and R. Salakhutdinov (Eds.), *Proceedings of the 36th International Conference on Machine Learning* (Proceedings of Machine Learning Research, Vol. 97, pp. 6105–6114). PMLR. https://proceedings.mlr.press/v97/tan19a.html

[13] Graves, A., Mohamed, A.-R., and Hinton, G. (2013). Speech recognition with deep recurrent neural networks. In *2013 IEEE International Conference on Acoustics, Speech and Signal Processing* (pp. 6645–6649). IEEE. https://doi.org/10.1109/ICASSP.2013.6638947

[14] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998–6008. https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

[15] Kim, H., Hasegawa-Johnson, M., Perlman, A., Gunderson, J., Huang, T. S., Watkin, K., and Frame, S. (2008). Dysarthric speech database for universal access research. In *Interspeech 2008* (pp. 1741–1744). ISCA. https://doi.org/10.21437/Interspeech.2008-480

[16] Schu, G., Janbakhshi, P., and Kodrasi, I. (2023). On using the UA-Speech and TORGO databases to validate automatic dysarthric speech classification approaches. In *ICASSP 2023—2023 IEEE International Conference on Acoustics, Speech and Signal Processing* (pp. 1–5). IEEE. https://doi.org/10.1109/ICASSP49357.2023.10095981

[17] Yılmaz, E., Mitra, V., Sivaraman, G., and Franco, H. (2019). Articulatory and bottleneck features for speaker-independent ASR of dysarthric speech. *Computer Speech & Language*, 58, 319–334. https://doi.org/10.1016/j.csl.2019.05.002

[18] Janbakhshi, P., and Kodrasi, I. (2022). Adversarial-free speaker identity-invariant representation learning for automatic dysarthric speech classification. In *Interspeech 2022* (pp. 2138–2142). ISCA. https://doi.org/10.21437/Interspeech.2022-402

[19] Xu, L., Liss, J., and Berisha, V. (2023). Dysarthria detection based on a deep learning model with a clinically-interpretable layer. *JASA Express Letters*, 3(1), Article 015201. https://doi.org/10.1121/10.0016833

[20] Javanmardi, F., Kadiri, S. R., and Alku, P. (2024). Pre-trained models for detection and severity level classification of dysarthria from speech. *Speech Communication*, 158, Article 103047. https://doi.org/10.1016/j.specom.2024.103047

[21] Bhat, C., and Strik, H. (2020). Automatic assessment of sentence-level dysarthria intelligibility using BLSTM. *IEEE Journal of Selected Topics in Signal Processing*, 14(2), 322–330. https://doi.org/10.1109/JSTSP.2020.2967652

[22] Gurugubelli, K., and Vuppala, A. K. (2020). Analytic phase features for dysarthric speech detection and intelligibility assessment. *Speech Communication*, 121, 1–15. https://doi.org/10.1016/j.specom.2020.04.006

[23] Bhat, C., and Strik, H. (2025). Speech technology for automatic recognition and assessment of dysarthric speech: An overview. *Journal of Speech, Language, and Hearing Research*, 68(2), 547–577. https://doi.org/10.1044/2024_JSLHR-23-00740

[24] Kadirvelu, B., Stumpf, L., Waibel, S., and Faisal, A. A. (2025). Speaker-independent dysarthria severity classification using self-supervised transformers and multi-task learning. *PLOS Digital Health*, 4(11), e0001076. https://doi.org/10.1371/journal.pdig.0001076

[25] Sohn, J., Kim, N. S., and Sung, W. (1999). A statistical model-based voice activity detection. *IEEE Signal Processing Letters*, 6(1), 1–3. https://doi.org/10.1109/97.736233

[26] Millet, J., and Zeghidour, N. (2019). Learning to detect dysarthria from raw speech. In *ICASSP 2019—2019 IEEE International Conference on Acoustics, Speech and Signal Processing* (pp. 5831–5835). IEEE. https://doi.org/10.1109/ICASSP.2019.8682324

[27] Hernandez, A., Yeo, E. J., Kim, S., and Chung, M. (2020). Dysarthria detection and severity assessment using rhythm-based metrics. In *Interspeech 2020* (pp. 2897–2901). ISCA. https://doi.org/10.21437/Interspeech.2020-2354

[28] Wang, D., Deng, L., Yeung, Y. T., Chen, X., Liu, X., and Meng, H. (2021). Unsupervised domain adaptation for dysarthric speech detection via domain adversarial training and mutual information minimization. In *Interspeech 2021* (pp. 2956–2960). ISCA. https://doi.org/10.21437/Interspeech.2021-2139

[29] Xiong, Y., Berisha, V., Liss, J., and Chakrabarti, C. (2025). Mitigating overfitting during speech foundation model fine-tuning: Applications to dysarthric speech detection. In *Interspeech 2025* (pp. 2133–2137). ISCA. https://doi.org/10.21437/Interspeech.2025-567

[30] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Qian, Y., Wu, J., Zeng, M., Yu, X., and Wei, F. (2022). WavLM: Large-scale self-supervised pre-training for full stack speech processing. *IEEE Journal of Selected Topics in Signal Processing*, 16(6), 1505–1518. https://doi.org/10.1109/JSTSP.2022.3188113

[31] Microsoft. (2021). *WavLM Base+ model card*, revision `4c66d4806a428f2e922ccfa1a962776e232d487b`. Hugging Face. https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b (accessed August 24, 2026).

[32] Joy, N. M., Umesh, S., and Abraham, B. (2017). On improving acoustic models for TORGO dysarthric speech database. In *Interspeech 2017* (pp. 2695–2699). ISCA. https://doi.org/10.21437/Interspeech.2017-878
