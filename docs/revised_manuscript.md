# Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy: A Reproducibility Study

Suresh Kumar Chelliah¹, Arivazhagan N²*

¹,²Department of Computational Intelligence, School of Computing, SRM Institute of Science and Technology, Kattankulathur-603203, Chennai, India.

E-mail: ¹sureshkc@srmist.edu.in, ²arivazhn@srmist.edu.in

\* Corresponding author: arivazhn@srmist.edu.in

ORCID: Suresh Kumar Chelliah — 0009-0006-5641-8344; Arivazhagan N — 0000-0001-9515-4237

## Abstract

Automatic dysarthric speech detection could support rapid screening and assistive communication, and a natural question is how early after speech onset a decision can be made. This paper asks whether dysarthric-versus-control speech can be detected from 200–300 ms speech-onset prefixes of TORGO head-mounted-microphone recordings, and whether posterior entropy can reduce decision time within that window. In responding to reviewer scrutiny of an earlier version of this work, we found that its reported confusion matrix did not reconcile with its reported accuracy, and that the original code and predictions no longer existed. We therefore rebuilt the entire experiment as a frozen, version-controlled pipeline and report what it yields.

Three results follow. First, the speech-onset detector used in the earlier version fails: because its threshold falls below the recordings' room tone, it returns an onset at sample zero for approximately 45% of utterances, so a nominal "onset-aligned" prefix frequently contains no speech. Second, with a corrected level-adaptive causal detector and strictly speaker-independent five-fold evaluation, none of three compact architectures exceeds the majority-class baseline: pooled out-of-fold accuracy is 44.32% (small CNN), 46.61% (CNN–BiLSTM) and 41.94% (Transformer) against a 63.92% constant-class baseline, with per-fold accuracy ranging from 24.4% to 75.3%. Third, the entropy rule does not improve on fixed-duration decisions. For two classes, normalized posterior entropy is a monotone transform of maximum softmax probability, and an explicit ablation confirms the two are numerically identical; posteriors are severely miscalibrated (expected calibration error 0.429, fitted temperature 11.2), and correcting that calibration raises the forced-fallback rate from 8.9% to 64.2%, removing the early decisions rather than improving them.

We conclude that the 200–300 ms speech-onset window does not support reliable speaker-independent dysarthria detection on TORGO with compact spectral models, and that between-fold variance on a 15-speaker corpus exceeds every effect the earlier version reported. We publish the complete pipeline, the per-utterance prediction ledger, and the analyses so the result can be checked and extended.

**Keywords:** dysarthric speech detection, early decision, speaker-independent evaluation, reproducibility, calibration, speech onset, time entropy, TORGO.

## 1. Introduction

Dysarthria is a motor speech disorder in which impaired neuromuscular control of the articulators affects articulation, phonation, resonance, prosody, and overall speech intelligibility. It arises from a wide range of neurological conditions, including cerebral palsy, amyotrophic lateral sclerosis, Parkinson's disease, and stroke, and it varies substantially in type and severity between speakers. Because the disorder degrades the same acoustic cues that listeners and recognizers rely on, people with dysarthria are frequently underserved by mainstream speech interfaces. Automatic detection systems are valuable when they provide objective, repeatable measurements, but to be useful they must cope with high inter-speaker variability and with the limited size of publicly available dysarthric-speech corpora.

The TORGO database contains time-aligned acoustic and articulatory recordings from speakers with dysarthria and from matched control speakers [1], and has become a standard benchmark for pathological-speech research. The present work uses its head-mounted-microphone recordings under a speaker-independent protocol, so that reported behaviour reflects generalization to unseen speakers rather than memorization of speaker-specific characteristics.

Most systems optimize accuracy after receiving a complete utterance or a relatively long segment [2–5]. That is appropriate offline but less suited to interactive or assistive settings where low latency matters. The motivating question is therefore: how early after speech onset can a model make a useful dysarthria decision, and how should uncertainty be handled when the acoustic context is very short?

### 1.1. Why this paper is a reproducibility study

An earlier version of this work reported 76.09% accuracy for a Transformer at 300 ms and 75.89% for an entropy-gated early-decision system at 264.6 ms mean latency. During peer review, a reviewer recomputed the metrics implied by our published confusion matrix and found that they did not agree with the accompanying table: the matrix implies 74.96% accuracy rather than 75.89%, 73.21% dysarthric recall rather than 73.38%, and 73.67% macro-F1 rather than 72.82%. We are grateful for that check.

On investigating, we established that the original training code, model checkpoints, and per-sample prediction files no longer existed, so the discrepancy could not be traced and the reported numbers could not be regenerated. We were unwilling to defend values we could not reproduce. We therefore rebuilt the experiment from raw audio as a frozen, version-controlled pipeline, froze every design decision before inspecting test results, and replaced all previously reported values. Two model families named in the earlier version — CNN-GRU and an "EfficientNet-lite" network — are withdrawn, because no layer definition, variant identifier, checkpoint, or prediction file for either survives; naming a variant after the fact would be reconstruction presented as record.

The rebuild produced a further finding that bears directly on the earlier numbers. The speech-onset detector, which anchors every prefix, thresholds short-time energy at a fixed absolute level that lies *below* the room tone of most TORGO head-mounted recordings. It therefore returns an onset at the first analysis frame for a large fraction of utterances, and the nominal onset-aligned prefix contains room tone rather than speech. Under the earlier version's own peak-relative rule this affects approximately 45% of utterances.

### 1.2. Contributions

1. A quantitative validation of RMS speech-onset detection on TORGO showing that two commonly used thresholding rules fail to localize onset on 45–72% of utterances, with the mechanism identified and a corrected level-adaptive causal rule specified (Section 5.1).
2. A complete, frozen, speaker-independent reproduction pipeline for short-prefix dysarthria detection, with per-utterance prediction ledgers from which every reported table and figure is derived (Sections 3–4).
3. Evidence that compact spectral models do not exceed the majority-class baseline on this task under correct speaker-independent evaluation, and that between-fold variance substantially exceeds between-model variance (Section 5.2).
4. A demonstration that, for binary classification, normalized posterior entropy is exactly equivalent to maximum-softmax confidence, together with a calibration analysis showing that the apparent confidence driving early decisions is miscalibration: correcting it removes the early decisions (Sections 5.3–5.4).
5. A prespecified robustness analysis quantifying the asymmetric cost of onset-localization error and showing that moderate additive noise *improves* measured accuracy, which we interpret as evidence against the robustness of what the models learn (Section 5.5).

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 describes the methodology. Section 4 details the experimental setup. Section 5 presents results. Section 6 discusses limitations and Section 7 concludes.

## 2. Related Work

### 2.1. Dysarthric speech corpora and speaker-independent evaluation

TORGO [1] and UA-Speech are the corpora most often used for dysarthric-speech classification. Both are small in speaker count, which makes the choice of evaluation protocol decisive: with fifteen speakers, a single speaker can dominate a fold, and results obtained without strict speaker separation are not comparable with results obtained under it. We adopt speaker-independent folds in which every speaker appears in exactly one test partition, and we report per-speaker results so that a reader can see how much of any aggregate is attributable to individual speakers.

### 2.2. Deep spectral models for pathological speech

Compact convolutional, recurrent, and Transformer models applied to time–frequency representations are the dominant approach for pathological-speech classification [2–5,12–14]. Recent work increasingly uses self-supervised pretrained representations, which report stronger speaker-independent behaviour than compact spectral models trained from scratch [24]. We specify such a reference (Section 3.6) but did not execute it within this revision, and we state that plainly rather than reporting it as a result.

### 2.3. Early classification and uncertainty-aware decisions

The wait-or-decide problem is formalized in early time-series classification, where a classifier is judged not only by its final label but by how early that label becomes reliable [9–11]. Confidence-based rejection and deferral use posterior probability or entropy as the deferral signal [6,7], and modern calibration work shows that neural posteriors are typically overconfident and require correction before they can be read as confidence [8].

This literature bears on our formulation in a way the earlier version did not acknowledge. For a two-class problem, normalized posterior entropy and maximum softmax probability are related by a strictly monotone transformation, so thresholding one is equivalent to thresholding the other. Normalized entropy is therefore not an independent uncertainty statistic in the binary case. What can be claimed is a *sequential stopping policy* — a calibrated confidence gate combined with stability conditions across consecutive prefixes, with parameters chosen under an explicit latency-versus-accuracy constraint. Section 5.4 tests whether that combination contributes anything measurable.

### 2.4. Protocol-comparable comparison

Table 1 compares representative prior studies along the axes that determine whether numbers can be compared at all. We do not rank across rows: differences in corpus, microphone, segment duration, speaker separation, and latency definition make cross-study accuracy differences uninterpretable as evidence of method superiority.

**Table 1.** Comparison axes for prior dysarthric-speech classification work. "Latency definition" records whether a study defines decision time and, if so, from what reference point.

| Study axis | What must be stated | Why it prevents comparison if omitted |
|---|---|---|
| Corpus | TORGO, UA-Speech, or other | Speaker counts and severity distributions differ |
| Microphone | Head-mounted, array, far-field | Channel characteristics dominate short-prefix energy features |
| Task | Binary detection, severity grading, ASR | Different label spaces and chance levels |
| Segment duration | Full utterance, fixed prefix, windowed | Determines how much evidence the model receives |
| Speaker separation | Speaker-independent or not | Non-separated splits inflate accuracy substantially |
| Evaluation protocol | Folds, repetitions, estimand | Pooled and fold-averaged metrics differ |
| Metric definition | Accuracy, macro-F1, recall, weighting | Class imbalance changes the meaning of accuracy |
| Latency definition | Reference point and whether compute is included | "Latency" may mean observation, compute, or both |

## 3. Materials and Methods

### 3.1. Speech-onset detection

Every waveform is converted to mono at 16 kHz. Let `x[n]` denote the waveform. Short-time root-mean-square energy is computed on left-aligned frames with no centred padding and no future reflection, so that frame *m* depends only on samples at or before its own right edge:

- frame length *N* = 20 ms (320 samples), hop *H* = 10 ms (160 samples);
- frame energy `r[m] = sqrt( (1/N) · Σ_{n=0}^{N−1} x[mH + n]² )`;
- frame *m* is **active** if and only if `r[m] ≥ τ`;
- **onset criterion:** `m* = min{ m : r[m] ≥ τ and r[m+1] ≥ τ }`, the first run of two consecutive active frames. The onset anchor is sample `m*·H`, and the rule is confirmed at sample `(m*+1)·H + N`, the earliest instant at which the detector could have made the decision. Both are recorded, and their difference is reported as detection delay in Section 5.7.

No temporal smoothing is applied to `r[m]`. The earlier version implied a smoothing stage that did not exist; robustness to isolated spurious frames comes from the two-frame confirmation instead.

The threshold `τ` distinguishes three modes:

| Mode | Threshold | Role |
|---|---|---|
| `causal_noise_floor` (**primary**) | `max(0.003, median{r[0..9]} · 10^(15/20))` | Causal, level-adaptive; used for all headline results |
| `causal_fixed` | `0.003` | Causal but level-blind; retained as a comparator |
| `recovered_offline_peak` | `max(0.003, max_m r[m] · 10^(−35/20))` | **Non-causal**; the earlier version's rule, retained for comparison only |

The earlier version used the peak-relative rule, which depends on the maximum energy of the *whole* utterance and therefore reads future audio. It is not a causal real-time detector, which contradicts a minimal-latency claim. Section 5.1 shows that both it and the fixed-threshold rule additionally fail to localize onset on a large fraction of the corpus, and quantifies the consequence.

Utterances in which no two-frame run reaches the threshold are recorded as explicit detector failures with a registered fallback anchor at sample zero. They are counted and reported, never silently dropped.

### 3.2. Speech-onset prefix representation

A prefix of nominal duration *t* contains exactly *t* milliseconds of **post-onset** signal, preceded by up to 30 ms of **already-observed** past context. Its total input length is therefore at most *t* + 30 ms, and the pre-roll never reduces the post-onset evidence available at time *t*. The earlier version's wording was ambiguous on this point: it could be read as placing the pre-roll inside the nominal prefix, which would make a "200 ms" input contain only about 170 ms of post-onset speech while simultaneously claiming the full nominal duration followed onset.

The pre-roll is retained because an energy-threshold detector necessarily anchors *after* low-energy onsets such as weak plosive releases and breathy or aspirated starts, and because it costs no additional waiting: at the moment onset is confirmed, that audio is already buffered. It is history, not latency. Section 5.4 reports a condition-matched ablation showing that it makes no material difference to accuracy; we therefore do not claim, as the earlier version did, that it captures discriminative articulatory information.

From the onset-aligned waveform, prefixes are converted to log-magnitude short-time Fourier transform features using a 512-point FFT, a 25 ms periodic Hann window, a 10 ms hop, no centred padding, and per-utterance standardization. Prefixes of 150, 200, 250, and 300 ms are evaluated; the resulting feature maps have 257 frequency bins and 16, 21, 26, and 31 frames respectively.

### 3.3. Posterior entropy

Let `p(c | X_t)` denote the posterior probability of class *c* after observing prefix `X_t`, and let *C* denote the number of classes. Here **C = 2** (control, dysarthric). Normalized time entropy is

`H_t = − (1 / log C) · Σ_{c=1}^{C} p(c | X_t) · log p(c | X_t)`.

Division by `log C = log 2` maps the entropy onto `[0, 1]` exactly, where 0 is a fully committed posterior and 1 is maximum uncertainty. The purpose is interpretive: it places the threshold on a fixed, dimensionless scale comparable across models, folds, and prefix durations.

We state explicitly what the earlier version did not. For *C* = 2, `H_t` is a strictly monotone decreasing function of the maximum posterior `max_c p(c | X_t)`, so thresholding entropy at θ is equivalent to thresholding confidence at a corresponding value. Normalized entropy is not an independent uncertainty statistic in this setting, and we make no such claim. The normalized form is retained because it keeps θ interpretable and generalizes unchanged to *C* > 2. Section 5.4 includes a maximum-softmax-confidence arm that verifies this equivalence numerically.

The predicted class at time *t* is `ŷ_t = argmax_c p(c | X_t)`.

### 3.4. Early-decision rule

Evaluated prefixes are {150, 200, 250, 300} ms with spacing **Δ = 50 ms**. The candidate decision set is **{200, 250, 300} ms**. The 150 ms prefix is always computed and stored but is **never eligible to emit a decision**: it exists to initialize the stability comparison, so that the earliest emittable decision already has a predecessor.

A decision is accepted at the earliest candidate time *t* satisfying all three conditions:

1. `H_t ≤ θ` — the calibrated posterior is confident;
2. `ŷ_t = ŷ_{t−Δ}` — the predicted class is unchanged from the previous prefix;
3. `|H_t − H_{t−Δ}| ≤ ε` — the entropy is stable between those prefixes.

If no candidate satisfies all three, the system emits the decision at the 300 ms endpoint, bounding worst-case latency.

This corrects an inconsistency in the earlier version, where the stated equation compared *t* against *t* − Δ but the implemented algorithm applied the stability conditions only "if a previous prefix exists", silently permitting a confidence-only decision at 200 ms. Text, pseudocode, and figures now describe one identical rule with no first-candidate exemption.

θ and ε are selected on validation speakers only, by exhaustive search over θ ∈ [0.30, 0.95] in steps of 0.05 and ε ∈ [0.00, 0.20] in steps of 0.02 (154 combinations). The criterion minimizes mean observation time subject to validation macro-F1 remaining within 0.01 absolute of the same fold's fixed-300 ms validation macro-F1, with tie-breaks on higher macro-F1, lower fallback rate, lower θ, then lower ε.

### 3.5. Model architectures

Three compact architectures are evaluated. All receive log-STFT tensors of shape `B × 1 × 257 × T` and emit two logits ordered (control, dysarthric). Convolutions are bias-free where followed by batch normalization, and dropout is 0.30 throughout. Parameter counts are verified programmatically rather than quoted.

**Table 2.** Small CNN — 23,538 trainable parameters.

| Stage | Operation | Output channels | Parameters |
|---|---|---:|---:|
| 1 | Conv2d 1→16, 3×3, stride 1, pad 1; BatchNorm2d; SiLU; MaxPool 2×2 | 16 | 176 |
| 2 | Conv2d 16→32, 3×3, stride 1, pad 1; BatchNorm2d; SiLU; MaxPool 2×2 | 32 | 4,672 |
| 3 | Conv2d 32→64, 3×3, stride 1, pad 1; BatchNorm2d; SiLU; MaxPool 2×2 | 64 | 18,560 |
| Pool | Length-aware mean over valid frequency/time positions | 64 | 0 |
| Head | Dropout 0.30; Linear 64→2 | 2 | 130 |

**Table 3.** CNN–BiLSTM — 69,354 trainable parameters.

| Stage | Operation | Output shape | Parameters |
|---|---|---|---:|
| 1 | Conv2d 1→24, 3×3, pad 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×24×128×T | 264 |
| 2 | Conv2d 24→48, 3×3, pad 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×48×64×T | 10,464 |
| Tokens | Mean over remaining 64 frequency bins, transpose | B×T×48 | 0 |
| Sequence | One-layer bidirectional LSTM, input 48, hidden 64 per direction | B×T×128 | 58,368 |
| Head | Masked temporal mean; Dropout 0.30; Linear 128→2 | 2 | 258 |

**Table 4.** Short-prefix Transformer — 196,594 trainable parameters.

| Stage | Operation | Output shape | Parameters |
|---|---|---|---:|
| 1 | Conv2d 1→24, 3×3, pad 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×24×128×T | 264 |
| 2 | Conv2d 24→48, 3×3, pad 1; BatchNorm2d; SiLU; MaxPool 2×1 | B×48×64×T | 10,464 |
| Tokens | Adaptive mean-pool frequency to 2 bins; flatten 48×2; bias-free Linear 96→96; sinusoidal positional encoding | B×T×96 | 9,216 |
| Encoder | 2 encoder layers, model dimension 96, 4 heads, feed-forward 256, GELU, dropout 0.30, post-norm | B×T×96 | 174,272 |
| Head | Masked temporal mean; Dropout; Linear 96→24; SiLU; Dropout; Linear 24→2 | 2 | 2,378 |

In both sequence models the convolutional front end pools **frequency only**, never time, so exactly one token is retained per input STFT frame and prefix duration maps one-to-one onto sequence length. The Transformer receives a key-padding mask, padded encoded states are cleared, and only valid tokens enter the final masked mean.

### 3.6. Modern pretrained reference

A WavLM Base+ reference is specified on identical folds and input durations: checkpoint `microsoft/wavlm-base-plus` pinned to immutable revision `4c66d4806a428f2e922ccfa1a962776e232d487b`, raw 16 kHz onset-aligned waveform input with a sample-level attention mask, masked-mean pooling, and a frozen backbone with Dropout 0.30 plus Linear 768→2 (1,538 trainable parameters). The loader is implemented and unit-tested and records the resolved commit in every run summary. **This reference was not executed within the present revision, and no result is reported for it.** Given the findings in Section 5.2 it is the most informative outstanding experiment.

### 3.7. Calibration

Posterior calibration is fitted before any policy threshold is applied. A single scalar temperature per fold is fitted by minimizing negative log likelihood on a held-out calibration partition of the validation speakers, and applied to all prefixes. Calibration quality is reported as negative log likelihood, Brier score, and 15-bin expected calibration error, with reliability diagrams, both before and after scaling. Policy results are reported in both states so that the effect of calibration on the decision rule is visible rather than assumed.

## 4. Experimental Setup

### 4.1. Dataset

The head-mounted-microphone subset of TORGO is selected by `usable_prompt = True` and duration ≥ 0.05 s, yielding **8,131 utterances from 15 speakers**: 5,197 control (63.92%) and 2,934 dysarthric (36.08%), an imbalance ratio of 1.771:1.

**Table 5.** Speaker demographics and utterance counts.

| Sex | Dysarthric speakers | Control speakers | Dysarthric utt. | Control utt. | Total |
|---|---:|---:|---:|---:|---:|
| Female | 3 (F01, F03, F04) | 3 (FC01, FC02, FC03) | 943 | 2,149 | 3,092 |
| Male | 5 (M01–M05) | 4 (MC01–MC04) | 1,991 | 3,048 | 5,039 |
| **Total** | **8** | **7** | **2,934** | **5,197** | **8,131** |

**Table 6.** Utterance type, derived from prompt text.

| Prompt type | Control | Dysarthric | Total |
|---|---:|---:|---:|
| Single word | 3,764 | 2,145 | 5,909 (72.7%) |
| Multi-word / sentence | 1,337 | 714 | 2,051 (25.2%) |
| Non-word / instructed task | 96 | 75 | 171 (2.1%) |

The corpus is dominated by isolated single words, so results should not be read as evidence about continuous or conversational speech. Per-speaker utterance counts range from 132 (F01) to 1,108 (MC01), an 8.4-fold spread. **Dysarthria severity and neurological etiology are not present in the corpus index we use**; rather than impute them we label them unavailable and report per-speaker results so that speaker-level heterogeneity remains visible.

### 4.2. Speaker-independent folds

Each of the 15 speakers appears in exactly one test partition, and no speaker appears in more than one role within a fold.

**Table 7.** Fold composition by speaker.

| Fold | Train | Validation | Test | Test utt. |
|---|---|---|---|---:|
| 0 | F01, F03, FC02, FC03, M03, M04, MC01 | F04, M02, MC02, MC03 | FC01, M01, M05, MC04 | 1,689 |
| 1 | F03, FC01, FC02, FC03, M01, M04, M05, MC04 | F01, M03, MC01 | F04, M02, MC02, MC03 | 1,940 |
| 2 | F03, F04, FC01, FC03, M01, M02, M05, MC02, MC03, MC04 | FC02, M04 | F01, M03, MC01 | 1,657 |
| 3 | F01, F04, FC01, M01, M02, M03, M05, MC01, MC02, MC03, MC04 | F03, FC03 | FC02, M04 | 1,288 |
| 4 | F01, F04, FC02, M02, M03, M04, MC01, MC02, MC03 | FC01, M01, M05, MC04 | F03, FC03 | 1,557 |

Test partitions sum to exactly 8,131. Fold test sizes are **unequal** (1,288–1,940), which is why pooled and fold-averaged metrics differ and must be reported separately. Validation partitions are dominated by single speakers — 77% FC02 in fold 2, 67% MC01 in fold 1 — and folds 2 and 3 contain only two validation speakers each. This has a direct methodological consequence discussed in Section 4.3.

### 4.3. Implementation details

**Table 8.** Training configuration, frozen before any test evaluation.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate / schedule | 1.5 × 10⁻⁴, constant |
| Weight decay | 5 × 10⁻⁴ |
| Batch size | 64 |
| Maximum epochs / early-stopping patience | 35 / 8 |
| Gradient clipping | 5.0 global norm |
| Dropout / label smoothing | 0.30 / 0.05 |
| Class balancing | Inverse-frequency weighted sampler, with replacement, training partition only |
| Augmentation (training only) | Gain −3…+3 dB; additive noise 20–35 dB SNR (p = 0.5); frequency masks ≤ 16 bins, time masks ≤ 4 frames (p = 0.5) |
| Checkpoint selection metric | Speaker-balanced macro-F1 |
| Primary seed / replication seeds | 20260819 / 20260820, 20260821 |

Class balancing uses inverse-frequency weights `w_i = 1 / count(class(i))` with replacement, drawing per epoch a number of samples equal to the training partition size. Every class therefore receives equal total weight mass and the expected per-epoch draw distribution is 50/50, achieved by oversampling the dysarthric class within an epoch rather than discarding data. Balancing and augmentation are applied to the **training partition only**; validation and test partitions retain their natural distribution, so no reported metric is computed on a resampled set.

Validation speakers are deterministically partitioned by hashed sample identifier into checkpoint-selection (50%), calibration (25%), and policy-tuning (25%) subsets, so the checkpoint is not chosen on the rows used to calibrate or to select θ and ε.

**Checkpoint selection uses speaker-balanced macro-F1 rather than pooled validation negative log likelihood.** Because validation partitions are 38–77% a single speaker (Section 4.2), pooled NLL largely measures that speaker: in preliminary runs it swung by a factor of two between consecutive epochs and in one fold never improved on the epoch-0 value, so early stopping preserved an effectively untrained checkpoint while training loss fell steadily. Each TORGO speaker holds a single class, so a per-speaker F1 does not exist; instead every utterance is weighted by `1 / n(speaker)` before the macro-F1 is computed. This decision was made from training and validation curves alone, with no test data involved.

**Reproducibility caveat.** The primary runs execute on an Apple Silicon MPS backend, where at least one backward operation used by these models has no deterministic implementation. Runs are seeded, configuration-hashed, and the complete prediction ledger is published, but we do not claim bit-exact reproducibility on this backend.

### 4.4. Evaluation protocol and estimands

Two estimands are reported separately and never mixed:

- **Pooled out-of-fold:** one metric computed over all 8,131 utterances, each appearing exactly once across the five test folds.
- **Unweighted fold mean ± sample SD:** the five per-fold metrics averaged with equal weight per fold.

These differ whenever folds have unequal size, and ours do. Conflating them is what produced the inconsistency identified in the earlier version. All metrics — accuracy, precision, recall, specificity, macro-F1, confusion matrix, mean decision time, forced-fallback rate — are computed from the same decision rows of a single canonical ledger, and a validation step fails the pipeline if any two disagree. Uncertainty is quantified by speaker-clustered bootstrap with 10,000 replicates, resampling *speakers* rather than utterances, because utterances within a speaker are not independent.

## 5. Results and Discussion

### 5.1. Speech-onset detection fails under both previously used thresholds

We report this first because every downstream result depends on it. A label-free diagnostic over 600 utterances (40 per speaker) compares detector modes using only the waveform: no label, no model output, and no test metric enters it, so it can be run before any training.

**Table 9.** Onset-detector behaviour, 600 utterances. "Prefix is room tone" is the fraction of 300 ms onset-aligned prefixes whose RMS falls below 0.01.

| Mode | Fires at sample 0 | Median onset | Median post-onset RMS | Prefix is room tone | Post/pre energy |
|---|---:|---:|---:|---:|---:|
| `causal_fixed` | 71.7% | 0 ms | 0.0119 | 43.3% | +19.5 dB |
| **`causal_noise_floor`** | **4.7%** | 1060 ms | **0.1713** | **3.5%** | **+25.3 dB** |
| `recovered_offline_peak` (earlier version) | **45.0%** | 40 ms | 0.0149 | **27.5%** | +15.2 dB |

The mechanism is a mismatch of scale. An absolute RMS floor of 0.003 corresponds to −50 dBFS, whereas measured room tone in these head-mounted recordings lies between −22 and −41 dBFS — above the threshold. The first analysis frame therefore already exceeds it, the two-frame criterion is satisfied immediately, and the detector returns sample zero. For speaker MC04, room tone holds near −41 dBFS for roughly 1.4 s before speech reaches −9 dBFS, and all ten of the first ten frames clear the threshold. The failure is speaker-dependent because it depends on recording level: speaker M05's recordings open directly on speech, where an anchor at sample zero is approximately correct.

The consequence is not subtle. A fold trained under the fixed-threshold detector scored 77.4–82.2% across its three validation partitions and 52.9% on test against a 51.9% majority baseline; control speaker MC04 — 650 utterances, 38% of that fold's test set, every onset at sample zero — was classified dysarthric on 100% of its utterances. **This affects the earlier version of this work**, whose peak-relative rule fires at sample zero on approximately 45% of utterances, leaving about 27.5% of nominal onset-aligned prefixes containing room tone rather than speech. We cannot quantify the precise effect on the previously reported numbers, because the original predictions no longer exist, but a classifier trained and evaluated on prefixes that are substantially silence is not measuring what was claimed.

The level-adaptive causal rule is therefore primary for all results below. The choice was made on waveform statistics alone, among modes already specified in the frozen configuration, and is a validation outcome rather than tuning on test performance.

### 5.2. Fixed-duration classification does not exceed the majority-class baseline

**Table 10.** Fixed-duration pooled out-of-fold performance over all 8,131 utterances.

| Model | Prefix | Accuracy | Macro-F1 | Dysarthric recall |
|---|---:|---:|---:|---:|
| Small CNN | 200 ms | 43.18% | 43.14% | 63.50% |
| Small CNN | 250 ms | 44.45% | 44.42% | 58.59% |
| Small CNN | 300 ms | 44.32% | 44.30% | 58.79% |
| CNN–BiLSTM | 200 ms | 42.66% | 42.42% | 50.07% |
| CNN–BiLSTM | 250 ms | 44.73% | 44.37% | 50.82% |
| CNN–BiLSTM | 300 ms | **46.61%** | 46.09% | 50.95% |
| Transformer | 200 ms | 43.07% | 43.04% | 56.65% |
| Transformer | 250 ms | 42.11% | 42.11% | 57.91% |
| Transformer | 300 ms | 41.94% | 41.93% | 60.05% |
| **Constant "control" predictor** | — | **63.92%** | — | 0% |

No model at any prefix duration approaches the majority-class baseline of 63.92%. The best pooled result, 46.61%, is 17.3 points below it. Balanced accuracy for the Transformer's complete policy is 45.6% with a speaker-clustered 95% interval of 32.5–58.8%, which **includes 50%**: the system is not statistically distinguishable from chance on this design.

Longer prefixes do not reliably help. The CNN–BiLSTM improves from 42.66% to 46.61% between 200 and 300 ms, the small CNN is flat, and the Transformer *declines* from 43.07% to 41.94%. The earlier version's observation that additional context improves macro-F1 for most models does not reproduce.

**Table 11.** Per-fold accuracy at fixed 300 ms, showing the dominant source of variance.

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Range |
|---|---:|---:|---:|---:|---:|---:|
| Small CNN | 46.8% | 21.4% | 55.0% | 76.7% | 32.0% | 55.3 pts |
| CNN–BiLSTM | 60.9% | 28.0% | 53.3% | 53.9% | 41.2% | 32.8 pts |
| Transformer | 48.0% | 24.4% | 39.3% | 75.3% | 32.4% | 50.9 pts |

Per-fold accuracy varies by up to 55 percentage points within a single architecture. By contrast, on fold 1 the three architectures — spanning 23,538 to 196,594 parameters, an eightfold range — produce validation balanced macro-F1 of 0.3795, 0.3814 and 0.3669, a spread of 0.015. **Between-fold variance exceeds between-model variance by more than an order of magnitude.** Any ranking of these architectures on this corpus is measuring fold assignment, not model capability. This is the strongest methodological finding of the study and it applies retrospectively to the earlier version's comparison table.

Validation performance also fails to predict test performance: the small CNN achieved higher validation balanced macro-F1 than the CNN–BiLSTM on fold 0 (0.7367 versus 0.6894) but lower test accuracy (46.8% versus 60.9%).

### 5.3. Posteriors are severely miscalibrated

**Table 12.** Calibration of the Transformer, pooled out-of-fold at the 300 ms prefix, 15 bins.

| State | Negative log likelihood | Brier score | Expected calibration error |
|---|---:|---:|---:|
| Raw softmax | 1.393 | 0.446 | **0.429** |
| Temperature-scaled | 0.837 | 0.303 | 0.212 |

The fitted temperature is **11.23 ± 9.21** across folds, reaching the optimizer's upper bound of 20 in two folds. A well-calibrated model requires a temperature near 1. An expected calibration error of 0.429 means the raw posteriors are close to meaningless as confidence values, which is precisely the condition under which an entropy threshold cannot be trusted.

### 5.4. The entropy rule contributes nothing measurable

**Table 13.** Policy ablation for the Transformer, pooled out-of-fold. Calibrated unless marked raw.

| Policy | Accuracy | Macro-F1 | Mean decision time | Forced fallback |
|---|---:|---:|---:|---:|
| Fixed 200 ms | **43.07%** | **43.04%** | 200.0 ms | 0% |
| Fixed 250 ms | 42.11% | 42.11% | 250.0 ms | 0% |
| Fixed 300 ms | 41.94% | 41.93% | 300.0 ms | 0% |
| Entropy only (raw) | 42.33% | 42.33% | 209.1 ms | 4.2% |
| **Max-softmax confidence (raw)** | **42.33%** | **42.33%** | **209.1 ms** | **4.2%** |
| Entropy + class stability (raw) | 42.28% | 42.28% | 212.9 ms | 5.3% |
| Entropy + entropy stability (raw) | 41.99% | 41.99% | 221.4 ms | 7.3% |
| Complete policy (raw) | 41.99% | 41.99% | 224.4 ms | 8.9% |
| Complete policy (calibrated) | 41.95% | 41.95% | 269.4 ms | 64.2% |

Three findings follow.

**Entropy and maximum-softmax confidence are numerically identical**, reproducing to every reported digit, as Section 3.3 predicts analytically. The entropy formulation contributes no information beyond thresholding confidence.

**No component of the decision rule improves performance.** All policy variants lie within 0.4 points of one another and at or below the fixed 200 ms baseline. Adding stability conditions increases latency without improving accuracy.

**Calibration removes the early decisions rather than improving them.** Correcting the overconfidence documented in Section 5.3 raises the forced-fallback rate from 8.9% to 64.2%, and to 100% in three of five folds, where the selected parameters degenerate to the lowest admissible θ with ε = 0. The apparent confidence that produced early decisions *was* the miscalibration.

**Table 14.** Selected θ and ε per fold, chosen on validation policy-tuning rows only.

| Fold | θ | ε | Validation macro-F1 | Fixed-300 validation macro-F1 | Mean decision time | Forced fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.95 | 0.20 | 0.786 | 0.787 | 212.6 ms | 6.9% |
| 1 | 0.30 | 0.00 | 0.200 | 0.200 | 300.0 ms | 100% |
| 2 | 0.95 | 0.20 | 0.664 | 0.655 | 228.5 ms | 16.8% |
| 3 | 0.30 | 0.00 | 0.466 | 0.466 | 300.0 ms | 100% |
| 4 | 0.30 | 0.00 | 0.538 | 0.538 | 300.0 ms | 100% |

The selection is bimodal rather than smoothly sensitive: folds either decide early on most utterances or never decide early at all.

**Table 15.** Paired comparison of the complete calibrated policy against each fixed baseline on identical utterances. Speaker-clustered bootstrap 95% intervals on the difference, with exact McNemar p-values.

| Baseline | Δ accuracy | Δ dysarthric recall | Δ mean latency | McNemar *p* |
|---|---|---|---|---:|
| Fixed 200 ms | −0.037 … +0.009 | **+0.006 … +0.047** | +47.8 … +88.8 ms | 0.0009 |
| Fixed 250 ms | −0.019 … +0.012 | −0.002 … +0.030 | −2.3 … +39.1 ms | 0.536 |
| Fixed 300 ms | −0.004 … +0.004 | **−0.015 … −0.0003** | **−51.9 … −10.9 ms** | 1.000 |

Against fixed 300 ms the policy saves 11–52 ms of mean latency but *reduces* dysarthric recall by a small yet statistically detectable amount, with accuracy indistinguishable. Against fixed 200 ms it is 48–89 ms slower and McNemar rejects equality in the baseline's favour. **We therefore withdraw the earlier version's claim that the entropy criterion improves sensitivity.** We do not read the non-significant comparisons as equivalence: with 15 speaker clusters these intervals are wide and the study is underpowered for small effects.

**Pre-roll ablation.** Five condition-matched retrainings with zero pre-roll give a mean validation balanced macro-F1 of 0.5691 ± 0.1383 against 0.5797 ± 0.1462 with 30 ms, a difference of −0.0106 whose sign varies across folds and whose magnitude is an order below the between-fold standard deviation. The pre-roll makes no material difference, and the earlier version's articulatory-information justification is withdrawn.

### 5.5. Robustness to noise and onset error

Temperature and the policy parameters are fitted once on the clean validation partitions and applied unchanged to every perturbed condition; no test row participates in any fit.

**Table 16.** Test-time robustness, Transformer, pooled out-of-fold, complete calibrated policy.

| Condition | Accuracy | Macro-F1 | Dysarthric recall | Δ vs clean |
|---|---:|---:|---:|---:|
| Clean reference | 41.95% | 41.95% | 59.37% | — |
| White noise, 20 dB SNR | 43.09% | 43.09% | 60.26% | +1.14 |
| White noise, 10 dB SNR | **44.57%** | 44.24% | 72.49% | **+2.62** |
| White noise, 0 dB SNR | 38.05% | 35.82% | 78.56% | −3.90 |
| Onset shift −50 ms | 35.84% | 33.60% | 75.12% | −6.11 |
| Onset shift −30 ms | **35.27%** | 33.55% | 71.17% | **−6.68** |
| Onset shift −10 ms | 38.89% | 38.61% | 63.26% | −3.06 |
| Onset shift 0 ms | 41.95% | 41.95% | 59.37% | +0.00 |
| Onset shift +10 ms | 43.44% | 43.39% | 56.00% | +1.49 |
| Onset shift +30 ms | 41.63% | 41.60% | 54.74% | −0.32 |
| Onset shift +50 ms | 40.97% | 40.96% | 55.11% | −0.98 |

The zero-shift condition reproduces the clean reference exactly, confirming that the perturbation machinery is inert at offset zero.

**Onset error is asymmetric and costly.** Anchoring 30–50 ms *early* costs 6.1–6.7 accuracy points, roughly three times the effect of anchoring the same distance late, because an early anchor fills the prefix with pre-onset room tone. This quantifies the sensitivity that makes the Section 5.1 detector failure consequential.

**Moderate additive noise improves measured accuracy**, by 2.6 points at 10 dB SNR. A system extracting genuine speech structure should not improve when noise is added. We report this as evidence against the robustness of what the models have learned, not as a robustness result. Synthetic white noise is in any case a channel stress test and is not evidence of cross-corpus or real far-field robustness.

### 5.6. Error structure and per-speaker behaviour

**Table 17.** Pooled out-of-fold confusion matrix, Transformer, complete calibrated policy.

| | Predicted control | Predicted dysarthric | Row total |
|---|---:|---:|---:|
| True control | 1,669 | **3,528** | 5,197 |
| True dysarthric | 1,192 | 1,742 | 2,934 |
| **Total** | 2,861 | 5,270 | **8,131** |

Accuracy computed from this matrix is (1,669 + 1,742) / 8,131 = **41.95%**, reproducing the reported value exactly, and the decision counts reproduce the reported mean decision time of 269.4 ms exactly. This mutual consistency is enforced by construction, which was the specific failure identified in the earlier version.

The false-positive rate is **67.9%**: two-thirds of control utterances are labelled dysarthric.

**Table 18.** Per-speaker behaviour, Transformer, complete calibrated policy. Each TORGO speaker holds a single class, so accuracy on that class is reported alongside the rate at which the speaker is predicted dysarthric.

| Speaker | Class | Utterances | Share | Accuracy | Predicted dysarthric |
|---|---|---:|---:|---:|---:|
| MC02 | control | 464 | 5.7% | **1.7%** | 98.3% |
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

The table shows what a pooled number cannot. The system is **not** dominated by a few outlier speakers: it carries a global bias toward predicting dysarthric. Six of seven control speakers are labelled dysarthric on 65–98% of their utterances, and the apparently strong scores on dysarthric speakers (84–91%) are a consequence of that same bias rather than independent evidence of detection ability. We therefore do not attribute the false positives to speaker-specific noise, prompt type, or onset error; the honest explanation is that the classifier does not separate the classes on held-out speakers at this prefix length.

### 5.7. Complexity and practical latency

**Table 19.** Complexity, Apple M2 CPU, batch size 1, 300 ms prefix. THOP counts multiply–accumulates; one MAC is reported as two FLOPs.

| Model | Parameters | Serialized weights | MACs | FLOPs | Compute | Peak RSS increment |
|---|---:|---:|---:|---:|---:|---:|
| Small CNN | 23,538 | 100 KiB | 18.26 M | 36.52 M | 2.14 ms | 5.2 MB |
| CNN–BiLSTM | 69,354 | 279 KiB | 44.70 M | 89.41 M | 3.69 ms | 14.9 MB |
| Transformer | 196,594 | 783 KiB | 46.24 M | 92.49 M | 3.63 ms | 14.5 MB |

**Table 20.** Component-level latency, Transformer, decision at 200 ms, 100 repetitions.

| Component | CPU | MPS |
|---|---:|---:|
| Onset detection | 0.079 ms | 0.089 ms |
| Prefix crop | 0.008 ms | 0.011 ms |
| STFT and standardization | 0.342 ms | 0.360 ms |
| Input tensor preparation | 0.015 ms | 0.580 ms |
| Model forward | 3.047 ms | 5.921 ms |
| Logit to probability | 0.037 ms | 0.584 ms |
| Policy computation | 0.120 ms | 0.140 ms |
| **Total compute** | **3.68 ms** | **7.73 ms** |
| Acoustic observation | 200 ms | 200 ms |
| **Observation plus compute** | **203.68 ms** | **207.73 ms** |

Mean decision time is not system latency, and the two are reported as distinct quantities. Compute is 3.68 ms, **1.8%** of end-to-end latency; acoustic observation dominates and is irreducible by any modelling choice. The accelerator is *slower* than the CPU at batch size 1, because kernel-launch and transfer overheads exceed the cost of a 196k-parameter forward pass, so single-utterance deployment should target the CPU. The input span at the 200 ms decision is 230 ms: 200 ms post-onset plus the 30 ms pre-roll.

The "lightweight" claim is supported — all three models are under 1 MB serialized and decide in under 4 ms on a laptop CPU. The "minimal-time" claim is supported only in the sense that compute is negligible; it is not supported in the sense of useful decisions at short latency.

## 6. Limitations and Interpretation

The central finding is negative: on speaker-independent TORGO head-mounted-microphone folds, 200–300 ms of onset-aligned speech does not support reliable dysarthric-versus-control classification with the compact spectral models evaluated here, and the entropy-based early-decision rule provides no measurable benefit over fixed-duration decisions.

Several limitations bound what may be concluded.

**This is a negative result about a specific configuration, not a proof of impossibility.** It covers three compact architectures trained from scratch on log-STFT features, on one corpus, with one microphone type, under one fold structure. A modern self-supervised representation is specified in Section 3.6 but was not run, and remains the most informative outstanding experiment: it would distinguish "the task is very hard at this latency" from "these models are inadequate for it".

**The design is underpowered.** With 15 speakers and 2–4 validation speakers per fold, checkpoint selection is noisy, validation does not predict test, and speaker-clustered intervals span roughly 25 accuracy points. We do not claim precision the design cannot support, and we do not interpret non-significant comparisons as equivalence.

**The corpus is narrow.** 72.7% of utterances are isolated single words, so nothing here speaks to continuous or conversational speech. Severity and etiology are unavailable in the index used, so we cannot report severity-stratified behaviour.

**Cross-corpus and far-field validation were not performed.** They are new-data experiments rather than analyses, requiring separate access and a compatible task mapping. Our generalization claim is therefore narrowed to head-mounted-microphone TORGO speech under the evaluated speaker-independent folds. Additive-noise testing is not a substitute for either, and we do not present it as one.

**This is not a clinical instrument.** The task is utterance-level acoustic classification. It does not establish a neurological diagnosis, does not estimate severity, and TORGO labels reflect corpus speaker grouping rather than per-utterance clinical determination. Any deployment would require clinical validation not attempted here.

## 7. Conclusion

We set out to determine how early after speech onset dysarthria can be detected, and whether posterior entropy can shorten that decision. Reviewer scrutiny of an inconsistency between our previously reported confusion matrix and our previously reported accuracy led us to rebuild the experiment from raw audio, after establishing that the original code and predictions no longer existed.

The rebuild yields four results. The speech-onset detector underlying the earlier work fails on roughly 45% of utterances because its threshold lies below the recordings' room tone, so nominal onset-aligned prefixes frequently contain no speech. With a corrected level-adaptive causal detector, no compact model exceeds the majority-class baseline, and per-fold accuracy varies by up to 55 points, so between-fold variance exceeds between-model variance by more than an order of magnitude. Normalized posterior entropy is exactly equivalent to maximum-softmax confidence for two classes, and the posteriors are so poorly calibrated that correcting them removes the early decisions the method depends on. Onset error costs up to 6.7 accuracy points asymmetrically, while moderate additive noise *improves* measured accuracy.

We therefore do not claim a working minimal-time dysarthria detector. We report instead that the 200–300 ms speech-onset window, as approached here, does not support one, and we document the specific ways in which an earlier and more optimistic conclusion arose. The complete pipeline, per-utterance prediction ledger, frozen configurations, and analysis code are published so the result can be checked, contradicted, or extended.

Future work should evaluate a pretrained self-supervised representation on the identical folds, adopt leave-one-speaker-out evaluation given the demonstrated fold sensitivity, and validate onset detection quantitatively before trusting any short-prefix result — the check in Section 5.1 requires no labels, no training, and no test data, and would have prevented the original error.

## Data and code availability

The reconstruction pipeline, frozen configurations, fold manifests, per-utterance prediction ledgers, calibration and policy-selection records, and all analysis code are available in the project repository. Every reported table is derived from a single canonical out-of-fold ledger, and every artifact records its configuration content hash, model-architecture hash, manifest hashes, and the commit that produced it.
