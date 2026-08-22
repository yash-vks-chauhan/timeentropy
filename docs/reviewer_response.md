# Point-by-point response to reviewers

Paper title: Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy

We thank the organizers and all three reviewers for the time and care they invested in this
manuscript. The comments identified genuine specification gaps, and one reviewer performed an
arithmetic check on our Figure 6 that we should have performed ourselves. Acting on that check
led to a finding that we report here openly, because it changes what this revision is.

## Statement on the revised results

While preparing this revision we established that the **original training code, model
checkpoints, and per-sample prediction files no longer exist**. We could not reproduce the
submitted numbers from surviving material, and we are not prepared to defend numbers we cannot
regenerate.

We have therefore rebuilt the entire experiment as a frozen, version-controlled, reproducible
pipeline, and we replace **every** result table and figure with values recomputed from a single
canonical out-of-fold prediction ledger. Preprocessing, folds, model definitions, calibration,
and the decision policy were all frozen before any test result was inspected; raw per-prefix
logits are written for every utterance before any aggregate metric is computed, so tables,
figures, and latency counts are consistent by construction rather than by assertion. The
previously reported values appear below only where a reviewer asked about them directly.

Reviewer 1's arithmetic on Figure 6 is correct, and it is the reason we checked. We are grateful
for it. Our response to R1.20 sets out what we found.

**A second finding, from the onset validation the reviewers requested.** Reviewers 1, 2, and 3 all
asked us to specify and validate the RMS onset detector (R1.4, R2.4, R3.1). Carrying out that
validation revealed that our onset detector frequently did not locate speech onset at all. It
fires on the recording's room tone, so a nominal "onset-aligned 300 ms prefix" can contain no
speech. This affects the **submitted** manuscript's detector on approximately **45%** of
utterances. We report it in full at R1.4, because it bears on how the submitted results should be
interpreted, and because the reviewers would have found it themselves by running the same check.

Two model families named in the submitted manuscript — **CNN-GRU** and **EfficientNet-lite** —
are withdrawn rather than reconstructed from memory. Our reasoning is given at R2.7/R3.4.

## Requests we cannot complete in this revision

We would rather be direct than appear compliant:

- **Cross-corpus validation on UA-Speech** (R1.14, R1-EV.7) and **far-field / mobile microphone
  validation** (R1.17) are new-data experiments, not manuscript revisions. They require data
  access, a compatible task mapping, and a separate evaluation protocol. We cannot complete them
  within this revision cycle. We therefore **narrow the paper's generalization claim** to
  head-mounted-microphone TORGO speech under the evaluated speaker-independent folds, and list
  these as limitations and planned future work. We explicitly do **not** present additive-noise
  stress testing as a substitute for an independent corpus or for real far-field microphones,
  and we have removed any wording that could be read that way.

Manuscript locations are given by section, table, and figure number; page and line numbers refer
to the revised typeset manuscript.

> **Scope of the numbers below.** All values come from the rebuilt pipeline, from one canonical
> out-of-fold ledger covering all 8,131 utterances, with the Transformer as the prospectively
> designated headline model unless stated otherwise. They differ substantially from the submitted
> manuscript's values; the reasons are set out at R1.4 and R1.20.


## Reviewer 1 — comments with author response

### R1.1 Novelty should be distinguished from conventional confidence-based early classification and entropy-based rejection

**Author response.** We accept this and have rewritten the contribution claim, because the
original framing overstated the novelty of the *statistic* rather than the *policy*.

The honest position, now stated in §1 and §3.2: for a binary task, normalized posterior entropy
is a **monotone decreasing transform of the maximum softmax probability**. Thresholding entropy
at θ is therefore equivalent to thresholding confidence at a corresponding value, and normalized
entropy is *not* an independent uncertainty statistic here. We no longer imply otherwise.

What we do claim, and now claim precisely, is the **validated sequential stopping policy** and
the protocol around it:

1. a *calibrated* confidence gate — temperature scaling fitted on validation speakers before any
   thresholding, which the rejection literature typically omits (see R1.13/R2.5);
2. **two stability conditions** across consecutive prefixes of the same utterance — predicted
   class agreement and bounded entropy change — which distinguish this from single-shot
   confidence rejection and guard against transient confidence spikes;
3. **constrained selection**: θ and ε are chosen to minimize observation time subject to macro-F1
   staying within a fixed margin of the 300 ms operating point, on validation speakers only,
   rather than by maximizing an unconstrained score;
4. an application setting — **sub-300 ms dysarthria detection from speech onset** — in which the
   accuracy/earliness trade-off is the object of study rather than a by-product.

We also made the claim falsifiable. The revised ablation includes a **`max_softmax_confidence`**
arm alongside `entropy_only`, `entropy_class_stability`, `entropy_value_stability`, and
`full_policy`, so a reader can see exactly how much of the effect comes from the uncertainty
statistic and how much from the stability conditions and constrained selection. The max-softmax
arm does match the entropy arm exactly, and we report that rather than hide it.

**Measured ablation (Transformer, pooled out-of-fold, calibrated unless noted).**

| Policy | Accuracy | Macro-F1 | Mean decision time | Forced fallback |
|---|---:|---:|---:|---:|
| Fixed 200 ms | 43.07% | 43.04% | 200.0 ms | 0% |
| Fixed 250 ms | 42.11% | 42.11% | 250.0 ms | 0% |
| Fixed 300 ms | 41.94% | 41.93% | 300.0 ms | 0% |
| Entropy only (raw) | 42.33% | 42.33% | 209.1 ms | 4.2% |
| **Max-softmax confidence (raw)** | **42.33%** | **42.33%** | **209.1 ms** | **4.2%** |
| Entropy + class stability (raw) | 42.28% | 42.28% | 212.9 ms | 5.3% |
| Entropy + entropy stability (raw) | 41.99% | 41.99% | 221.4 ms | 7.3% |
| Complete policy (raw) | 41.99% | 41.99% | 224.4 ms | 8.9% |
| Complete policy (calibrated) | 41.95% | 41.95% | 269.4 ms | 64.2% |

Two things follow, and we report both plainly.

**The max-softmax arm reproduces the entropy arm to every digit** — 42.33% / 42.33% / 209.1 ms /
4.2% — and the pipeline labels the correspondence `exact_for_binary_probabilities`. Our concession
above is therefore not a rhetorical hedge but a measured identity.

**No component of Equation 4 contributes a performance gain.** Every policy variant sits within
0.4 points of every other, and all of them sit at or below the fixed 200 ms baseline. Adding the
stability conditions increases latency without improving accuracy. We report this as the result of
the ablation the reviewer asked for, rather than presenting the complete policy as validated.

§2 is extended with the early time-series classification, uncertainty/rejection, and calibration
literature that the submitted version treated too briefly (see also R2.2).

### R1.2 Exact architecture and parameter configuration of all models

**Author response.** Provided in full, with programmatically verified parameter counts. Two of
the five named models are withdrawn (see R2.7/R3.4). Input is a log-magnitude STFT tensor
`B × 1 × 257 × T`; output is two logits ordered (control, dysarthric). Convolutions are
bias-free where followed by batch normalization; dropout is 0.30 throughout.

**Small CNN — 23,538 trainable parameters**

| Stage | Exact operation | Out | Params |
|---|---|---:|---:|
| 1 | Conv2d `1→16`, `3×3`, s1, p1; BatchNorm2d; SiLU; MaxPool `2×2` | 16 | 176 |
| 2 | Conv2d `16→32`, `3×3`, s1, p1; BatchNorm2d; SiLU; MaxPool `2×2` | 32 | 4,672 |
| 3 | Conv2d `32→64`, `3×3`, s1, p1; BatchNorm2d; SiLU; MaxPool `2×2` | 64 | 18,560 |
| Pool | Length-aware mean over valid frequency/time positions | 64 | 0 |
| Head | Dropout 0.30; Linear `64→2` | 2 | 130 |

**CNN–BiLSTM — 69,354 trainable parameters**

| Stage | Exact operation | Out | Params |
|---|---|---:|---:|
| 1 | Conv2d `1→24`, `3×3`, s1, p1; BatchNorm2d; SiLU; MaxPool `2×1` | `B×24×128×T` | 264 |
| 2 | Conv2d `24→48`, `3×3`, s1, p1; BatchNorm2d; SiLU; MaxPool `2×1` | `B×48×64×T` | 10,464 |
| Tokens | Mean over remaining 64 frequency bins, transpose | `B×T×48` | 0 |
| Seq | One-layer **bidirectional LSTM**, input 48, hidden 64/direction | `B×T×128` | 58,368 |
| Head | Masked temporal mean; Dropout 0.30; Linear `128→2` | 2 | 258 |

**Short-prefix Transformer — 196,594 trainable parameters**

| Stage | Exact operation | Out | Params |
|---|---|---:|---:|
| 1 | Conv2d `1→24`, `3×3`, s1, p1; BatchNorm2d; SiLU; MaxPool `2×1` | `B×24×128×T` | 264 |
| 2 | Conv2d `24→48`, `3×3`, s1, p1; BatchNorm2d; SiLU; MaxPool `2×1` | `B×48×64×T` | 10,464 |
| Tokens | Adaptive mean-pool frequency to 2 bins; flatten `48×2`; bias-free Linear `96→96`; sinusoidal positional encoding | `B×T×96` | 9,216 |
| Encoder | **2** encoder layers, `d_model` 96, **4** heads, FFN 256, GELU, dropout 0.30, post-norm | `B×T×96` | 174,272 |
| Head | Masked temporal mean; Dropout; Linear `96→24`; SiLU; Dropout; Linear `24→2` | 2 | 2,378 |

**WavLM Base+ reference** (see R1.10): checkpoint `microsoft/wavlm-base-plus`, pinned immutable
revision `4c66d4806a428f2e922ccfa1a962776e232d487b`, frozen backbone plus Dropout 0.30 and
Linear `768→2` (1,538 trainable parameters); the resolved commit is recorded in every run
summary and must match the requested identity or the run aborts.

Training settings, seeds, and schedule are given at R2.10. Complexity and runtime are at R1.16.

### R1.3 Exact hyperparameter values, search ranges, and tuning procedure for θ and ε

**Author response.** The submitted manuscript said only that these were "selected on validation
predictions." Full specification (see also R2.1, R2.6, R3.6):

- **Grids:** θ ∈ [0.30, 0.95] step **0.05** (14 values); ε ∈ [0.00, 0.20] step **0.02**
  (11 values); exhaustive — **154 combinations**.
- **Selection data:** validation speakers only. Within each fold, validation speakers are
  deterministically partitioned by hashed sample ID into checkpoint-stopping (50%),
  calibration (25%), and **policy-tuning (25%)** subsets. Temperature is fitted on the
  calibration rows; θ and ε are selected on the policy-tuning rows.
- **Criterion:** minimize **mean observation time**, subject to validation macro-F1 remaining
  within **0.01 absolute** of that fold's fixed-300 ms validation macro-F1. Tie-breakers, in
  order: higher macro-F1, lower forced-fallback rate, lower θ, lower ε.
- **Reporting:** selected θ and ε **per fold**, plus the full validation response surface.
  Test rows are reported only at validation-selected values; every metrics summary carries an
  explicit zero-count audit field for test rows used in fitting or selection.

Sensitivity analysis is at R1-EV.3.

**Selected values (Transformer, complete calibrated policy).** Chosen on the policy-tuning
partition of validation speakers only:

| Fold | θ | ε | Validation macro-F1 | Fixed-300 validation macro-F1 | Mean decision time | Forced fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.95 | 0.20 | 0.786 | 0.787 | 212.6 ms | 6.9% |
| 1 | 0.30 | 0.00 | 0.200 | 0.200 | 300.0 ms | 100% |
| 2 | 0.95 | 0.20 | 0.664 | 0.655 | 228.5 ms | 16.8% |
| 3 | 0.30 | 0.00 | 0.466 | 0.466 | 300.0 ms | 100% |
| 4 | 0.30 | 0.00 | 0.538 | 0.538 | 300.0 ms | 100% |

We draw the reviewer's attention to folds 1, 3 and 4, where the constrained search selects the
**lowest admissible θ with ε = 0** and the policy then never fires early: every utterance falls
back to 300 ms. After calibration the posteriors in those folds are never confident enough to
satisfy the entropy gate. The early-decision mechanism is therefore inactive in three of five
folds, which we report rather than average away.

### R1.4 Quantitative validation of speech-onset detection

**Author response.** Accepted — the submitted manuscript asserted onset detection without
validating it, which is untenable when the latency claim is measured from that onset. Two
changes.

**First, we corrected a defect the question exposed.** The detector recovered from our records
thresholded at `max(0.003, peak_RMS − 35 dB)` where `peak_RMS` is the peak of the **whole
utterance** — a rule that reads future audio and is therefore *not* causal, contradicting a
real-time latency claim. The revision makes a causal rule primary:

| Mode | Threshold | Role |
|---|---|---|
| `causal_noise_floor` (**primary**) | `max(0.003, median RMS of first 100 ms × 10^(15/20))` | causal and level-adaptive; all headline results |
| `causal_fixed` (comparator) | fixed absolute RMS floor `0.003` | causal but level-blind; retained to quantify its failure |
| `recovered_offline_peak` (comparator only) | `max(0.003, peak_RMS − 35 dB)` | **non-causal**; offline-alignment comparator |

**Second, quantitative validation — which found a serious defect.** We report this in full
because it changes how the submitted results should be read.

A label-free diagnostic (600 utterances, 40 per speaker) compares each detector mode using only
the waveform — no label, no model output, no test metric. It asks whether a mode actually
localises onset or simply fires on the recording's noise floor:

| Mode | Fires at sample 0 | Median onset | Median post-onset RMS | Prefix is room tone |
|---|---:|---:|---:|---:|
| Fixed absolute threshold | 71.7% | 0 ms | 0.0119 | 43.3% |
| **Adaptive noise-floor (new primary)** | **4.7%** | 1060 ms | **0.1713** | **3.5%** |
| **Peak-relative (submitted manuscript)** | **45.0%** | 40 ms | 0.0149 | **27.5%** |

The cause is a mismatch of scale. An absolute RMS floor of 0.003 is **−50 dBFS**, while measured
room tone in these head-mounted recordings sits at **−22 to −41 dBFS** — above the threshold. The
first analysis frame therefore already exceeds it, the two-frame criterion is satisfied
immediately, and the detector returns sample 0. For speaker MC04, room tone holds near −41 dBFS
for about 1.4 s before speech reaches −9 dBFS; every one of the first ten frames clears the
threshold. The failure is speaker-dependent because it depends on recording level: M05's
recordings open directly on speech, where an anchor at sample 0 is approximately correct.

**This affects the submitted manuscript.** Its peak-relative detector fires at sample 0 on
approximately **45%** of utterances, leaving about **27.5%** of nominal onset-aligned prefixes
containing room tone rather than speech. We cannot quantify the precise effect on the submitted
numbers, because the original predictions no longer exist — but a classifier trained and
evaluated on prefixes that are substantially silence is not measuring what the manuscript claimed
it measured, and we will not defend those numbers.

We observed the downstream consequence directly. A fold trained under the fixed-threshold
detector scored **77.4–82.2%** across its three validation partitions and **52.9%** on test,
against a 51.9% majority-class baseline. The per-speaker breakdown gives the mechanism: control
speaker **MC04** — 650 utterances, 38% of that fold's test set, every onset at sample 0 — was
classified dysarthric on **100%** of its utterances, while dysarthric speaker M05, whose
recordings open on speech, scored 99.6%.

**Disposition.** The primary detector is changed to the adaptive causal rule. We note that this
selects among the **three modes already specified in the frozen configuration**, on a criterion
computed from waveforms alone, so it is a validation outcome rather than tuning on test results;
had we run this diagnostic before training, as we should have, the same conclusion would have
followed with no model involved. The fixed-threshold mode is retained in the robustness matrix as
a comparator, so the manuscript quantifies the failure instead of merely describing it.

We additionally report the originally planned validation: manual annotation of a prespecified
stratified subset, with **detection-failure rate** and **signed and absolute onset error** against
those annotations and speaker-clustered intervals; and the complete decision policy repeated under
onset-anchor perturbations of **−50, −30, −10, 0, +10, +30, +50 ms**, so the reviewer can see how
much of the latency claim survives realistic onset error. Utterances where no two-frame run
reaches threshold are recorded as explicit detector failures with a registered fallback anchor,
counted rather than silently dropped.

*Reproducible evidence:* `scripts/diagnose_onset_modes.py`; full write-up in
`docs/onset_finding.md`.

Full detector mathematics are at R2.4. **Onset-error results are pending the frozen reruns.**

### R1.5 The 30 ms pre-roll and the definition of "200 ms after speech onset"

**Author response.** Accepted; this was a genuine ambiguity. The submitted §3.1 could be read as
placing the pre-roll *inside* the nominal prefix — making a "200 ms" input contain only ~170 ms
of post-onset speech — while the text simultaneously claimed the full nominal duration followed
onset. One definition now governs the whole manuscript:

> A prefix of nominal duration *t* contains exactly *t* ms of **post-onset** signal, preceded by
> up to 30 ms of **already-observed** past context. Total input length is at most *t* + 30 ms,
> and the pre-roll never reduces the post-onset evidence available at time *t*.

The pre-roll exists because an energy-threshold detector necessarily anchors *after* low-energy
onsets — weak plosive releases, breathy or aspirated starts — which carry articulatory
information that is diagnostic precisely for dysarthric speech. It costs **no additional
waiting**: at the moment onset is confirmed that audio is already buffered. It is history, not
latency. This distinction is now explicit wherever a latency figure appears (see also R1.15).

A **condition-matched zero-pre-roll ablation** is reported, with a model retrained from scratch
under its own onset handling, its own temperature, and its own policy selection, so the
comparison is not contaminated by reusing a checkpoint trained under a different input
definition.

**Measured (Transformer, condition-matched retraining, validation speaker-balanced macro-F1).**
The zero-pre-roll condition was retrained from scratch with its own onset handling, temperature,
and policy selection:

| Fold | 0 ms pre-roll | 30 ms pre-roll | Difference |
|---:|---:|---:|---:|
| 0 | 0.6939 | 0.6824 | +0.0115 |
| 1 | 0.3953 | 0.3669 | +0.0283 |
| 2 | 0.6542 | 0.6719 | −0.0177 |
| 3 | 0.4443 | 0.4863 | −0.0420 |
| 4 | 0.6580 | 0.6910 | −0.0330 |
| **Mean** | **0.5691 ± 0.1383** | **0.5797 ± 0.1462** | **−0.0106** |

**The pre-roll has no material effect.** The mean difference of 0.011 favours retaining it, but
the sign flips across folds (two of five favour removing it) and the effect is an order of
magnitude smaller than the between-fold standard deviation of 0.14. We therefore **withdraw the
manuscript's justification** that the pre-roll captures discriminative articulatory information:
our data do not support it. We retain the 30 ms pre-roll because it costs no latency and is
principled for an energy-threshold detector that necessarily anchors after low-energy onsets, and
we now say exactly that rather than claiming a performance benefit.

### R1.6 Are five-fold results pooled or averaged? Report fold-wise performance

**Author response.** This question is at the centre of what we found, and the honest answer about
the submitted manuscript is that **the two estimands were conflated and never labelled** — see
R1.20 for the evidence and the arithmetic.

The revision reports them as **two explicitly captioned estimands**, never mixed:

- **Pooled out-of-fold:** one metric computed over all 8,131 utterances, each appearing exactly
  once across the five test folds. This is what a confusion matrix over the full corpus shows.
- **Unweighted fold mean ± SD:** the five per-fold metrics averaged with equal weight per fold,
  which differs from pooled whenever folds have unequal size — and ours do (test *n* = 1,689,
  1,940, 1,657, 1,288, 1,557).

Every fold's accuracy, macro-F1, dysarthric recall, mean decision time, and forced-fallback rate
is reported individually (R1-EV.2), with per-speaker breakdowns and speaker-clustered bootstrap
confidence intervals (R1.7, R1.18).

**Measured (Transformer, complete calibrated policy).** The two estimands differ by a
non-trivial margin, which is precisely why they must be labelled:

| Estimand | Accuracy | Macro-F1 |
|---|---|---|
| Pooled out-of-fold (8,131 utterances) | 41.95% | 41.95% |
| Unweighted fold mean ± SD | **43.90% ± 19.58** | 41.67% ± 18.59 |

The **1.95-point** gap between pooled and fold-mean accuracy is the same mechanism that produced
the Figure 6 / Table 3 inconsistency the reviewer identified. Note also the fold standard
deviation of **19.58 points** — larger than any difference the submitted manuscript reported
between architectures or between operating points. Per-fold accuracies at fixed 300 ms are
48.0%, 24.4%, 39.3%, 75.3% and 32.4%.

### R1.7 Only 15 speakers — report fold-wise variation and confidence intervals

**Author response.** Accepted in full. With 15 speakers, a point estimate is not an adequate
summary, and a single atypical speaker can move an aggregate materially. We can now demonstrate
this rather than concede it in principle: in one diagnostic fold, control speaker MC04 alone —
650 utterances, 38% of that fold's test set — was misclassified on **100%** of its utterances and
pulled the fold from a validation range of 77–82% down to 52.9% on test (R1.4). A design in which
one speaker can do that requires interval reporting, not point estimates.

The revision reports: all five fold results individually; unweighted fold mean ± sample SD;
pooled out-of-fold metrics; per-speaker metrics; and **speaker-clustered bootstrap 95% confidence
intervals**, resampling *speakers* rather than utterances, since utterances within a speaker are
not independent and utterance-level resampling would understate the interval substantially.
Paired model comparisons use the same speaker-clustered scheme (R1.12).

**Measured intervals (Transformer, complete calibrated policy, 15 speaker clusters,
10,000 replicates).**

| Metric | Estimate | 95% CI |
|---|---:|---|
| Accuracy | 0.4195 | 0.3042 – 0.5519 |
| Macro-F1 | 0.4195 | 0.2949 – 0.5473 |
| Balanced accuracy | 0.4555 | 0.3249 – 0.5884 |
| Dysarthric recall | 0.5937 | 0.4213 – 0.7910 |
| Specificity | 0.3211 | 0.1447 – 0.5052 |
| Mean decision time | 269.4 ms | 264.6 – 274.0 ms |

We state the corresponding limitation plainly in §5.4, and the intervals make it concrete: the
accuracy interval spans **25 percentage points**, and the balanced-accuracy interval
**includes 0.50**, so the system is not statistically distinguishable from chance on this design.
Only the latency estimates are precise, because latency does not depend on speaker difficulty in
the way accuracy does. We do not claim precision the design cannot support.

### R1.8 Report original and effective class distributions and explain balancing exactly

**Author response.** Provided exactly. **Original distribution** (all 8,131 selected utterances):

| Class | Speakers | Utterances | Share |
|---|---:|---:|---:|
| Control | 7 | 5,197 | 63.92% |
| Dysarthric | 8 | 2,934 | 36.08% |
| **Total** | **15** | **8,131** | **100%** |

The imbalance ratio is **1.771 : 1** control-to-dysarthric.

**Balancing mechanism**, stated precisely: training uses a PyTorch `WeightedRandomSampler` with
**inverse-frequency weights** `w_i = 1 / count(class(i))`, **with replacement**, seeded
deterministically, drawing a number of samples per epoch equal to the size of that fold's
training partition. Because every class receives equal total weight mass, the **expected** draw
distribution per epoch is **50% / 50%**, achieved by oversampling dysarthric utterances and
undersampling control utterances *within* an epoch rather than by discarding data. Over epochs
all training utterances are seen.

Three clarifications the submitted manuscript omitted: balancing is applied to the **training
partition only** — validation and test partitions retain their natural distribution, so no
reported metric is computed on a resampled set; the *effective* per-epoch class counts and the
realized draw distribution are logged per fold and reported; and because the sampler draws with
replacement, the number of *distinct* utterances seen per epoch is less than the partition size,
which we also report. Augmentation is likewise training-only (R2.10).

### R1.9 Comparison with existing methods must use comparable protocols

**Author response.** Accepted. The submitted comparison placed numbers from different corpora,
microphones, segment durations, and evaluation protocols in proximity, which invites a ranking
those numbers cannot support.

The revision replaces it with a **structured comparison table** in which every row carries:
corpus; microphone condition; task definition; segment/utterance duration; speaker separation
(speaker-independent vs. not); evaluation protocol and number of folds; metric definition; and
**latency definition, where one exists**. Where a study is not protocol-comparable to ours, the
table says so in a dedicated column and we do **not** rank it against our result. We state
explicitly in the text that cross-study accuracy differences under different speaker partitions
and microphone conditions are not evidence of method superiority. See also R2.3.

### R1.10 Stronger baseline using a modern pretrained/self-supervised representation

**Author response.** Accepted and added. We include **WavLM Base+** (`microsoft/wavlm-base-plus`,
pinned to immutable revision `4c66d4806a428f2e922ccfa1a962776e232d487b`), evaluated on the
**identical speaker folds and identical input durations** as the compact models, so the
comparison isolates representation rather than protocol.

- Input: raw mono 16 kHz onset-aligned waveform with an exact sample-level attention mask.
- Pooling: mean over valid downsampled positions only.
- **Primary mode: frozen backbone** — Dropout 0.30 plus Linear `768→2` (1,538 trainable
  parameters). Total and trainable counts are both recorded.
- Optional sensitivity mode: unfreeze the final *N* encoder layers, **labelled separately** and
  never merged with the frozen-backbone reference.

This baseline replaces the withdrawn EfficientNet-lite row (R2.7/R3.4). We consider a
reproducible modern reference more informative than an unreproducible historical one — including
if it beats our compact models, which we will report as found.

**Executed, and it changed the paper's conclusion.** All five folds were run with a frozen
backbone (94,381,936 parameters, **zero trainable**) and a 1,538-parameter linear head, on
identical folds, onset detector, pre-roll, prefix durations, class balancing, checkpoint-selection
metric, and calibration/policy partitions. The only difference from the compact models is the
representation.

| Model | Pooled balanced accuracy | Macro-F1 | Fold SD | Raw ECE |
|---|---:|---:|---:|---:|
| **WavLM Base+ (frozen)** | **59.18%** | **58.10%** | **9.98** | **0.044** |
| CNN–BiLSTM | 47.56% | 46.09% | 12.96 | — |
| Small CNN | 47.47% | 44.30% | 21.35 | — |
| Transformer | 45.88% | 41.93% | 19.59 | 0.429 |
| Chance / constant predictor | 50.00% | 38.99% | — | — |

WavLM is roughly 12 points above the compact models in balanced accuracy and 9 above chance; it is
above chance on four of five folds; it halves the between-fold standard deviation; it improves
monotonically with prefix length (55.36 → 56.95 → 59.29% from 200 to 300 ms) where the Transformer
*declines*; and it is **well calibrated with no correction at all** (ECE 0.044 against 0.429), with
temperature scaling making it slightly worse.

This answers the reviewer's implicit question. The 200–300 ms window does carry usable dysarthria
information; the compact spectral models simply fail to extract it, and their severe
overconfidence is a symptom of speaker memorization rather than a property of the task.

We temper the claim as the design requires: the speaker-clustered 95% interval on WavLM's balanced
accuracy is 47.9–70.8% and still includes chance, and WavLM is below chance on fold 4. With fifteen
speakers this cannot be established at conventional confidence. We report a consistent direction
across five independent measurements, not a significant effect.
### R1.11 Report an explicit area-under-the-accuracy–earliness curve or standardized early-classification metric

**Author response.** Accepted. Mean latency alone is a poor summary — it hides the distribution
and rewards a policy that decides early on easy cases while saying nothing about what that costs.
The revision reports, with explicit definitions given in the text:

- the **accuracy–earliness curve** over fixed prefixes (150/200/250/300 ms) plus the adaptive
  policy plotted as a point on the same axes, so the policy is judged against the curve rather
  than against a single baseline;
- **consumed fraction** and **timeliness**, defined as the mean decision time divided by the
  maximum admissible decision time (300 ms), and 1 minus that quantity;
- prespecified **harmonic means** of accuracy-and-earliness and macro-F1-and-earliness, with the
  exact formula stated, so the trade-off is a single prespecified number rather than a
  post-hoc choice;
- the **full distribution** of decision times, not only the mean, plus the forced-fallback rate.

**Measured (Transformer, pooled out-of-fold).** Decision-time distribution under the complete
calibrated policy: **200 ms 2,285 (28.1%), 250 ms 408 (5.0%), 300 ms 5,438 (66.9%)**, of which
5,224 are forced fallbacks and 214 are accepted at the endpoint. Mean decision time
**269.4 ms**, consumed fraction 0.898, timeliness 0.102.

The accuracy–earliness curve is flat to slightly negative: pooled accuracy is 43.07% at fixed
200 ms, 42.11% at 250 ms, and 41.94% at 300 ms. Waiting longer does not help this system, so the
harmonic-mean trade-off metrics are maximised at the *shortest* prefix rather than by the adaptive
policy. We report that as found.

We prespecified these definitions before seeing test results precisely so that we could not select
the summary that flatters the method. See R1-EV.6 and R2.12.

### R1.12 Statistical testing for the claimed sensitivity improvement

**Author response.** Accepted; the submitted manuscript compared point estimates only. The
revision reports **paired** analyses on identical utterances:

- paired utterance-level correctness (McNemar-style discordant-pair analysis) between the
  adaptive policy and each fixed-duration baseline;
- **speaker-clustered paired bootstrap** effect intervals, resampling speakers rather than
  utterances, since utterance-level resampling would badly understate uncertainty with 15
  speakers;
- effect sizes with intervals, not only *p*-values.

Two honesty constraints we impose on ourselves. First, we will **not** interpret a
non-significant result as evidence of equivalence — with this sample size the study is
underpowered for small effects, and we say so. Second, if the dysarthric-recall improvement is
not statistically distinguishable from the fixed-duration baseline, we will report that plainly
and soften the claim accordingly, since the reviewer is right that the reported improvement was
modest.

**Measured comparison (Transformer, complete calibrated policy vs each fixed baseline).**
Speaker-clustered bootstrap 95% intervals on the paired difference, plus exact McNemar counts on
identical utterances:

| Baseline | Δ accuracy [95% CI] | Δ dysarthric recall [95% CI] | Δ mean latency [95% CI] | McNemar *p* |
|---|---|---|---|---:|
| Fixed 200 ms | −0.037 … +0.009 | **+0.006 … +0.047** | +47.8 … +88.8 ms | 0.0009 |
| Fixed 250 ms | −0.019 … +0.012 | −0.002 … +0.030 | −2.3 … +39.1 ms | 0.536 |
| Fixed 300 ms | −0.004 … +0.004 | **−0.015 … −0.0003** | **−51.9 … −10.9 ms** | 1.000 |

The reviewer's suspicion is borne out, and in the direction opposite to the manuscript's claim.
Against the **fixed 300 ms** baseline the policy buys **11–52 ms** of latency but *reduces*
dysarthric recall by a small yet statistically detectable amount (interval entirely below zero);
accuracy is indistinguishable. Against **fixed 200 ms** it is 48–89 ms *slower*, with higher
dysarthric recall but no accuracy gain — and McNemar rejects equality (*p* = 0.0009) in favour of
the fixed baseline being better on more discordant utterances (412 vs 321).

We therefore **withdraw the claim that the entropy criterion improves sensitivity.** It does not.
We report the trade-off it does provide — a modest mean-latency reduction against the 300 ms
endpoint — and we state that no fixed baseline is beaten on accuracy at any operating point.

As committed above, we do not read the non-significant comparisons as equivalence: with 15
speaker clusters these intervals are wide, and the study is underpowered for small effects.

### R1.13 Evaluate posterior calibration before using entropy as a confidence measure

**Author response.** Accepted, and we agree this is logically prior to the entropy rule — an
uncalibrated overconfident posterior makes a fixed entropy threshold mean different things in
different folds. The revision adds a full calibration analysis **before** any policy result:

- **raw versus temperature-scaled** negative log-likelihood, Brier score, and **15-bin expected
  calibration error**;
- **reliability diagrams** per fold;
- scalar temperature fitted **only** on the validation calibration partition (25% of validation
  speakers), never on test rows, with the fitted value reported per fold;
- policy performance reported **both** with and without calibration, so the reader can see
  whether calibration actually improves the early-decision policy rather than assuming it does
  (R1-EV.4).

See also R2.5.

**Measured calibration (Transformer, pooled out-of-fold, 300 ms prefix).**

| | NLL | Brier | 15-bin ECE |
|---|---:|---:|---:|
| Raw softmax | 1.393 | 0.446 | **0.429** |
| Temperature-scaled | 0.837 | 0.303 | 0.212 |

Fitted temperature: **11.23 ± 9.21** across folds. Reviewer 2's concern is confirmed
emphatically — a raw ECE of 0.429 means the posteriors are almost meaningless as confidence
values, and the temperature required to correct them is an order of magnitude above the
well-calibrated value of 1, reaching the optimiser's upper bound of 20 in two folds.

**Does calibration improve the early-decision policy? No — it removes it.** Correcting the
overconfidence raises the forced-fallback rate from **8.9% to 64.2%**, and in three of five folds
to 100% (R2.6). The apparent confidence that produced early decisions *was* the miscalibration.
This is the single most consequential finding of the calibration analysis, and it is reported
prominently rather than buried.

### R1.14 Cross-corpus validation on UA-Speech or another independent corpus

**Author response.** We cannot complete this within the revision, and we prefer to say so
directly rather than substitute something weaker and call it equivalent.

Cross-corpus evaluation is a new-data experiment requiring separate data access, a compatible
task and label mapping, and its own protocol — not a manuscript edit. Accordingly we
**narrow the paper's claim**: the contribution is now stated as speaker-independent detection on
**head-mounted-microphone TORGO speech under the evaluated folds**, and every generalization
sentence in §1, §5, and §6 has been rewritten to that scope. Cross-corpus transfer is listed in
§5.4 and §6 as a limitation and as the primary planned extension.

We explicitly do **not** offer our additive-noise robustness tests as a proxy for an independent
corpus; synthetic noise tests channel robustness, not speaker, recording-chain, or elicitation
differences. See R1-EV.7 and R1-EV.8.

### R1.15 Distinguish model inference time from observation, preprocessing, STFT, and onset-detection time

**Author response.** Accepted — and this matters more than a timing detail, because a 264.6 ms
mean *decision* time is not a 264.6 ms *system* latency. The revision reports a
**component-level breakdown**, measured separately rather than inferred:

| Component | What is timed |
|---|---|
| Acoustic observation | Waiting for the audio itself — the nominal post-onset duration |
| Onset detection | Framing, RMS, thresholding, two-frame confirmation |
| Feature extraction | STFT and per-utterance standardization |
| Model inference | Forward pass at the given prefix |
| Policy computation | Entropy, stability tests, decision logic |
| **Total end-to-end** | Sum as experienced by a caller |

**Measured (Transformer, Apple M2, batch size 1, warm, decision at 200 ms, 100 repetitions).**

| Component | CPU mean | MPS mean |
|---|---:|---:|
| Onset detection | 0.079 ms | 0.089 ms |
| Prefix crop | 0.008 ms | 0.011 ms |
| STFT + standardisation | 0.342 ms | 0.360 ms |
| Input tensor preparation | 0.015 ms | 0.580 ms |
| Model forward | 3.047 ms | 5.921 ms |
| Logit → probability | 0.037 ms | 0.584 ms |
| Policy computation | 0.120 ms | 0.140 ms |
| **Total compute** | **3.68 ms** | **7.73 ms** |
| Acoustic observation | 200 ms | 200 ms |
| **Observation + compute** | **203.68 ms** | **207.73 ms** |

Two practical points follow. **Acoustic observation dominates completely**: compute is 3.68 ms,
or **1.8%** of end-to-end latency, and is irreducible by any modelling choice. And the
accelerator is *slower* than the CPU at batch size 1 — 7.73 ms versus 3.68 ms — because kernel
launch and transfer overheads exceed the cost of a 196k-parameter forward pass. For
single-utterance deployment the CPU is the correct target, which the submitted manuscript did not
establish.

The input span at the 200 ms decision is **230 ms** (200 ms post-onset plus the 30 ms pre-roll),
confirming the definition at R1.5. See R1.16 and R1-EV.9.

### R1.16 Report computational complexity, parameter count, memory, and real-time latency

**Author response.** Accepted. Reported per model and per prefix duration: trainable and total
parameter count (verified programmatically, not quoted); serialized checkpoint size; peak memory
during inference; multiply–accumulate/FLOP count **with the counting convention stated
explicitly**, since MAC-vs-FLOP conventions differ by a factor of two and are frequently
reported ambiguously; and prefix-specific wall-clock inference time on stated hardware.

**Measured (Apple M2 CPU, batch size 1, 300 ms prefix, THOP backend 0.1.1-2209072238).**

| Model | Parameters | Serialized weights | MACs | FLOPs | Compute | Peak RSS increment |
|---|---:|---:|---:|---:|---:|---:|
| Small CNN | 23,538 | 100 KiB | 18.26 M | 36.52 M | 2.14 ms | 5.2 MB |
| CNN–BiLSTM | 69,354 | 279 KiB | 44.70 M | 89.41 M | 3.69 ms | 14.9 MB |
| Transformer | 196,594 | 783 KiB | 46.24 M | 92.49 M | 3.63 ms | 14.5 MB |

**Counting convention, stated explicitly as requested:** THOP counts multiply–accumulates, and we
report one MAC as two FLOPs. Where a FLOP backend is unavailable the pipeline records that fact
rather than emitting an estimate.

The "lightweight" claim is supported: all three models are under 1 MB serialized and run in under
4 ms per decision on a laptop CPU. The "minimal-time" claim is supported only in the sense that
*compute* is negligible; the binding constraint is the 200–300 ms of audio that must elapse first
(R1.15), and we no longer imply otherwise. For the WavLM reference we report total as well as
trainable parameter counts, so its 1,538 trainable head is not presented as the model's size.

### R1.17 Head-mounted microphones limit generalizability; investigate experimentally rather than acknowledge

**Author response.** We accept the criticism and can only partly act on it, so we are explicit
about the boundary.

What we **can** do, and now do: a prespecified test-time robustness matrix over additive white
Gaussian noise at **20/10/0 dB SNR** and onset-anchor shifts of **−50 to +50 ms**, applied after
onset-aligned cropping, with noise streams derived deterministically from a hash of seed, sample
ID, and SNR so results are reproducible and independent of batch order. This probes the
channel-degradation and onset-error failure modes.

What we **cannot** do within this revision: record or obtain genuine far-field, mobile, or
conversational dysarthric speech. Additive noise is **not** a far-field simulation — it omits
reverberation, microphone transfer characteristics, distance-dependent SNR, Lombard effects, and
spontaneous-speech phenomena. We therefore narrow the claim to head-mounted-microphone TORGO
(R1.14) and state far-field/mobile validation as a limitation and future work. We would rather
mark the boundary honestly than present noise injection as having crossed it.

### R1.18 Per-speaker results or fold-wise confusion matrices to check for speaker domination

**Author response.** Accepted, and this is a well-founded concern for this corpus: utterance
counts per speaker range from **132 (F01)** to **1,108 (MC01)**, an 8.4-fold spread, so a
pooled metric is disproportionately shaped by a few speakers. Our onset investigation produced a
concrete instance of exactly this failure — a single control speaker accounting for 38% of a
fold's test set and misclassified throughout (R1.4, R1.7). The revision reports:

- **per-speaker** accuracy, macro-F1, recall, and error counts for all 15 speakers;
- **per-fold confusion matrices** in addition to the pooled matrix;
- speaker-clustered bootstrap confidence intervals throughout (R1.7);
- the per-speaker utterance counts alongside every per-speaker metric, so a speaker's leverage on
  the aggregate is visible rather than implicit.

The false-positive audit (R3.10) is broken down by speaker for exactly this reason.

**Measured per-speaker results (Transformer, complete calibrated policy, pooled out-of-fold).**
Each TORGO speaker holds a single class, so we report accuracy on that class together with the
rate at which the speaker is predicted dysarthric:

| Speaker | Class | Utt. | Share | Accuracy | Predicted dysarthric |
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

The table shows something a pooled number cannot: the system is **not** dominated by a handful of
outlier speakers — it carries a **global bias toward predicting dysarthric**. Six of seven control
speakers are predicted dysarthric on 65–98% of their utterances, and the strong scores on
dysarthric speakers (84–91%) are a *consequence* of the same bias rather than independent
evidence of detection ability. Reporting only aggregate accuracy would have concealed this
entirely.

### R1.19 Keep the distinction between detection and clinical diagnosis explicit

**Author response.** Accepted without reservation. Every occurrence of diagnostic language has
been replaced with **"utterance-level dysarthric-speech classification"** or
**"dysarthric-speech detection."** The revised text states directly that the task is a binary
utterance-level acoustic classification which does **not** establish a neurological diagnosis,
does not estimate dysarthria severity, and is not a clinical decision instrument; that TORGO
labels reflect corpus speaker grouping rather than a per-utterance clinical determination; and
that any deployment would require clinical validation not attempted here. The title, abstract,
introduction, and conclusion have all been checked for this, since diagnostic framing tends to
survive in exactly those places.
### R1.20 and "Critical Experimental Validation Issue" — Figure 6 is inconsistent with Table 3/5

**Author response.** The reviewer is correct. We reproduced the arithmetic exactly, and it does
not reconcile. From the published confusion matrix:

| | Predicted control | Predicted dysarthric | Row total |
|---|---:|---:|---:|
| True control | 3,947 | 1,250 | **5,197** |
| True dysarthric | 786 | 2,148 | **2,934** |
| | | | **8,131** |

| Metric | Recomputed from Figure 6 | Reported in Table 3 | Difference |
|---|---:|---:|---:|
| Accuracy | **74.96%** | 75.89% | −0.93 |
| Dysarthric recall | **73.21%** | 73.38% | −0.17 |
| Macro-F1 | **73.67%** | 72.82% | **+0.85** |

The reviewer's figures are confirmed. Reconciling the accuracy alone would require **76**
utterances to change their classification.

We add two observations from our own investigation, because they bear on the cause.

**First, the row totals are diagnostic.** They are exactly 5,197 and 2,934 — precisely the
control and dysarthric counts of the *entire* 8,131-utterance corpus. Figure 6 is therefore
unambiguously a **pooled out-of-fold** matrix over every utterance, each appearing once.

**Second, the discrepancies run in opposite directions.** Accuracy computed from the matrix is
*lower* than reported, while macro-F1 is *higher*. A simple transcription error, a stale figure,
or a worse model would tend to move both the same way. Opposite-signed differences are instead
the signature of comparing a **pooled** estimand against an **unweighted fold-mean** estimand:
with unequal fold sizes (test *n* = 1,689 / 1,940 / 1,657 / 1,288 / 1,557) and a class ratio that
varies by fold, pooled and fold-averaged accuracy and macro-F1 need not move together. This is
consistent with Table 3 reporting fold means while Figure 6 pools — an estimand distinction the
submitted manuscript never labelled.

We state this as the most probable explanation, **not** as a demonstrated one, because we cannot
demonstrate it: as set out at the head of this response, the original prediction files no longer
exist, so the specific discrepancy cannot be traced to its source. The reviewer also notes that
the manuscript explicitly claimed all metrics were generated from saved prediction files and were
mutually consistent. That claim could not be substantiated. We regard this as the most serious
issue raised in the review, and it is the reason the revision is a rebuild rather than an
amendment.

**What the revision does, and the verification.** Every table and figure is regenerated from
**one canonical prediction ledger**. The check the submitted manuscript failed now passes: the
regenerated pooled matrix (1,669 / 3,528 / 1,192 / 1,742) yields accuracy
(1,669 + 1,742) / 8,131 = **41.95%**, matching the reported accuracy exactly, and the decision
counts (200 ms 2,285; 250 ms 408; 300 ms 5,438) yield a mean decision time of **269.4 ms**, also
matching exactly. Pooled and fold-mean estimands are additionally reported side by side and
differ by 1.95 accuracy points (R1.6), so the distinction can no longer be lost. Per-prefix logits for every utterance are written to disk *before* any
aggregate is computed; confusion matrices, accuracy, macro-F1, recall, latency, and
forced-fallback rate are all derived from the **same decision rows**; and pooled out-of-fold and
unweighted fold-mean ± SD are reported as **two separately captioned estimands** so they can
never again be silently interchanged. Consistency is enforced by construction and checked
programmatically, rather than asserted in prose.


## Reviewer 1 — required experimental validation

**R1-EV.1 — Recompute all metrics from one prediction file.** Done, and verified: the pooled
confusion matrix reproduces the reported accuracy (41.95%) and the decision counts reproduce the
reported mean decision time (269.4 ms) exactly. See R1.20.

**R1-EV.2 — Five-fold statistical report.** Delivered. Per-fold accuracy at fixed 300 ms is
48.0 / 24.4 / 39.3 / 75.3 / 32.4%; fold mean ± SD is 43.90% ± 19.58; pooled out-of-fold is 41.95%.
Speaker-clustered 95% intervals are at R1.7 (accuracy 0.304–0.552).

**R1-EV.3 — Entropy-threshold sensitivity.** Delivered over the full 14 × 11 grid. The outcome is
bimodal rather than smoothly sensitive: two folds select θ = 0.95 / ε = 0.20 and decide early;
three select θ = 0.30 / ε = 0.00 and never decide early (R2.6). No test re-tuning.

**R1-EV.4 — Calibration comparison.** Delivered, and the answer is negative. Raw ECE 0.429 falls
to 0.212 under temperature scaling (T = 11.23 ± 9.21), but the forced-fallback rate rises from
8.9% to 64.2%: calibration removes the early decisions rather than improving them (R1.13, R2.5).

**R1-EV.5 — Policy-component ablation.** Delivered (table at R1.1). No component of Equation 4
yields a performance gain; all variants fall within 0.4 points of each other and at or below the
fixed 200 ms baseline, and the `max_softmax_confidence` arm matches `entropy_only` exactly.

**R1-EV.6 — Results at 200/250/300 ms with a standardized earliness metric.** Delivered: pooled
accuracy 43.07 / 42.11 / 41.94% at 200 / 250 / 300 ms, mean decision time 269.4 ms, consumed
fraction 0.898, timeliness 0.102. The accuracy–earliness curve is flat to slightly negative
(R1.11).

**R1-EV.7 — Cross-corpus transfer.** **Not completed.** See R1.14: this is a new-data experiment
outside the revision's scope. The claim is narrowed to TORGO head-mounted-microphone speech and
the experiment is listed as future work. We do not present a substitute.

**R1-EV.8 — Robustness under noise, SNR, onset error, microphone and speaker variation.**
**Partly completed and reported at R2.4.** All 11 prespecified conditions were run across five
folds (55 evaluations). Early onset anchoring costs up to 6.7 accuracy points; moderate additive
noise *improves* accuracy by up to 2.6 points, which we report as evidence against robustness
rather than for it. The zero-versus-30 ms pre-roll ablation is at R1.5. Per-speaker results address speaker
and severity variation to the extent TORGO metadata supports (R2.8). **Different microphone
conditions and continuous conversational speech are not covered** — TORGO's head-mounted
subset does not contain them — and are stated as limitations (R1.17).

**R1-EV.9 — Complexity and end-to-end latency.** Parameter count, model size, MAC/FLOPs with the
convention stated, CPU and accelerator inference time, STFT preprocessing time, onset-detection
time, and total end-to-end latency, reported as separate components (R1.15, R1.16). We adopt the
reviewer's framing directly in the text: **mean decision time is not system latency**, and the
two are now reported as distinct quantities.

## Reviewer 2 — comments with author response

We thank the reviewer for the accurate summary of the system and for identifying that roughly
half of decisions reach the 300 ms timeout — a point we now treat as a headline caveat rather
than a footnote (R2.12).

### R2.1 How were θ and ε chosen, and how sensitive is the system to them?

**Author response.** The selection procedure is now fully specified — grids (θ: 0.30–0.95 step
0.05; ε: 0.00–0.20 step 0.02; 154 combinations), the validation-only partition used for fitting
and selection, and the explicit criterion (minimize mean observation time subject to macro-F1
within 0.01 absolute of the fold's fixed-300 ms macro-F1). Full detail at R1.3/R3.6; the selected
per-fold values are given at R2.6.

On **sensitivity**, which the submitted manuscript did not address at all: we report the complete
validation response surface over the grid, so the reader can see whether performance sits on a
plateau or a narrow ridge. If the surface is flat, the method is robust but θ is less meaningful
than claimed; if it is peaked, θ is load-bearing and must be tuned per deployment.

**Observed:** the selection is bimodal rather than smoothly sensitive. Two folds select the
maximum θ = 0.95 with ε = 0.20 and decide early on most utterances; three folds select the
minimum admissible θ = 0.30 with ε = 0.00 and never decide early at all (R2.6). The determining
factor is not a finely-tuned threshold but whether the fold's calibrated posteriors are ever
confident enough to pass the gate. Reported per fold with the full response surface.

### R2.2 Inadequate background in early time-series classification, uncertainty-based speech processing, and short-segment pathological speech; what is theoretically unique?

**Author response.** Accepted. §2 is substantially extended along the three axes named, and §1
now states the contribution in a way we can defend.

We concede the theoretical point directly: for a binary task, normalized posterior entropy is a
**monotone transform of maximum softmax confidence**, so we claim no novel uncertainty statistic
(R1.1). The methodological contribution is the **sequential stopping policy** — calibrated
confidence gating plus two cross-prefix stability conditions, with parameters chosen under an
explicit latency-versus-accuracy constraint on validation speakers only — and its application to
sub-300 ms dysarthria detection, where the accuracy/earliness trade-off is the object of study.
We relate this explicitly to early time-series classification, where the wait-or-decide problem
is formalized, and to the confidence-based rejection and calibration literature, and we state
what our formulation adds relative to each. The `max_softmax_confidence` ablation lets a reader
verify the claim rather than take it on trust.

### R2.3 Literature review is descriptive; add a systematic comparative table

**Author response.** Accepted. A structured comparison table replaces the descriptive passage,
with a row per study and columns for: corpus; microphone condition; input segmentation and
segment/utterance duration; speaker-independence of the evaluation; validation protocol and
number of folds; latency definition (where the study defines one); metric definition; and
reported performance. A dedicated column marks studies that are **not protocol-comparable** to
ours, and those are not ranked against our result. See also R1.9.

### R2.4 Onset detection lacks technical rigor — provide the complete mathematical framework

**Author response.** Accepted. The revision gives the complete formulation.

Let `x[n]` be the mono 16 kHz waveform. Frames are **left-aligned with no centred padding and no
future reflection**, so frame *m* depends only on samples at or before its own right edge:

- frame length `N = 20 ms = 320` samples; hop `H = 10 ms = 160` samples;
- frame RMS: `r[m] = sqrt( (1/N) · Σ_{n=0}^{N−1} x[mH + n]² )`;
- frame-level decision: frame *m* is **active** iff `r[m] ≥ τ`;
- **no temporal smoothing is applied to `r[m]`**. The submitted manuscript implied a smoothing
  stage that did not exist; we correct this rather than describe an operation we do not perform.
  Robustness to isolated spurious frames is obtained instead by the confirmation rule below.
- **onset criterion:** let `m* = min{ m : r[m] ≥ τ and r[m+1] ≥ τ }` — the first run of **two
  consecutive** active frames. The onset anchor is `m*·H`. The rule is confirmed at sample
  `(m*+1)·H + N`, which is the earliest instant at which the detector can have made the decision;
  both the anchor and the confirmation instant are recorded, and the difference between them is
  reported as detection delay in the latency decomposition (R1.15).

The threshold `τ` depends on the mode, and we correct a defect here (see R1.4/R3.1): the
recovered rule `τ = max(0.003, peak_RMS · 10^(−35/20))` uses the peak of the **whole utterance**
and is therefore **non-causal**. It is retained only as an offline-alignment comparator. The
primary rule is causal and fixed, `τ = 0.003`; a secondary causal rule estimates a noise floor
from the first 100 ms, `τ = max(0.003, median{r[0..9]} · 10^(15/20))`.

We must also report what applying this formulation revealed. The defect was the threshold scale,
not the framing or the decision rule: an absolute floor of 0.003 is −50 dBFS, whereas room tone in
these recordings measures −22 to −41 dBFS, so `r[m] ≥ τ` holds at `m = 0` for most utterances and
the returned "onset" is really the start of the recording. The submitted manuscript's
peak-relative rule does this on roughly **45%** of utterances. The primary rule is now the
adaptive noise-floor form given above, which reduces the fire-at-zero rate to **4.7%**. Full
figures and disposition at R1.4 and in `docs/onset_finding.md`.

**Onset-error detection:** utterances with no qualifying two-frame run are recorded as explicit
**detector failures** with a registered fallback anchor at sample 0, counted and reported rather
than dropped. Localization error is quantified against manual annotation on a prespecified
stratified subset (signed and absolute error, plus failure rate), and the complete policy is
re-run under onset perturbations of −50 to +50 ms.

**Measured (Transformer, pooled out-of-fold, calibrated policy; temperature and θ/ε fixed on the
clean validation partitions and applied unchanged to every perturbed condition).**

| Condition | Accuracy | Macro-F1 | Dysarthric recall | vs clean |
|---|---:|---:|---:|---:|
| Clean reference | 41.95% | 41.95% | 59.37% | — |
| White noise, 20 dB SNR | 43.09% | 43.09% | 60.26% | +1.14 |
| White noise, 10 dB SNR | 44.57% | 44.24% | 72.49% | +2.62 |
| White noise, 0 dB SNR | 38.05% | 35.82% | 78.56% | −3.90 |
| Onset shift −50 ms | 35.84% | 33.60% | 75.12% | −6.11 |
| Onset shift −30 ms | 35.27% | 33.55% | 71.17% | −6.68 |
| Onset shift −10 ms | 38.89% | 38.61% | 63.26% | −3.06 |
| Onset shift 0 ms | 41.95% | 41.95% | 59.37% | +0.00 |
| Onset shift +10 ms | 43.44% | 43.39% | 56.00% | +1.49 |
| Onset shift +30 ms | 41.63% | 41.60% | 54.74% | −0.32 |
| Onset shift +50 ms | 40.97% | 40.96% | 55.11% | −0.98 |

The zero-shift condition reproduces the clean reference **exactly**, which is the numerical audit
the protocol requires and confirms the perturbation machinery is inert at offset zero.

Two results deserve the reviewer's attention, and neither flatters the method.
**Onset error is asymmetric and costly**: anchoring 30–50 ms *early* costs 6.1–6.7 accuracy
points, roughly three times the effect of anchoring the same distance late, because an early
anchor fills the prefix with pre-onset room tone. This is a direct quantification of the concern
in R1.4 that onset-localisation error propagates into classification.
**Moderate additive noise *improves* accuracy** — +2.6 points at 10 dB SNR. A system extracting
genuine speech structure should not improve when noise is added. We report this as evidence
against the robustness of what the model has learned rather than as a robustness result.

### R2.5 Show whether posteriors are calibrated (ECE, reliability diagrams)

**Author response.** Accepted, and we agree the concern is well founded: the entropy rule
presupposes that posteriors mean what they say, and modern networks are typically overconfident.
The revision reports **15-bin expected calibration error**, **reliability diagrams**, negative
log-likelihood, and Brier score, **before and after** temperature scaling, per fold. Temperature
is fitted only on the validation calibration partition, never on test. We additionally report
policy performance with and without calibration, so the reader sees whether calibration actually
helps the decision rule rather than assuming it must. Full detail at R1.13/R1-EV.4.

**Measured calibration (Transformer, pooled out-of-fold, 300 ms prefix).**

| | NLL | Brier | 15-bin ECE |
|---|---:|---:|---:|
| Raw softmax | 1.393 | 0.446 | **0.429** |
| Temperature-scaled | 0.837 | 0.303 | 0.212 |

Fitted temperature: **11.23 ± 9.21** across folds. Reviewer 2's concern is confirmed
emphatically — a raw ECE of 0.429 means the posteriors are almost meaningless as confidence
values, and the temperature required to correct them is an order of magnitude above the
well-calibrated value of 1, reaching the optimiser's upper bound of 20 in two folds.

**Does calibration improve the early-decision policy? No — it removes it.** Correcting the
overconfidence raises the forced-fallback rate from **8.9% to 64.2%**, and in three of five folds
to 100% (R2.6). The apparent confidence that produced early decisions *was* the miscalibration.
This is the single most consequential finding of the calibration analysis, and it is reported
prominently rather than buried.

### R2.6 Publish the exact tuned θ and ε obtained on the validation splits

**Author response.** Agreed and will be published per fold, in a table, together with the
validation response surface.

For transparency about what changed: the submitted manuscript reported fold thresholds of 0.95,
0.90, 0.85, 0.85, 0.90 with a fixed ε = 0.08, under an objective that was never stated. Because
the original selection code and validation predictions no longer exist, we cannot verify how
those values were obtained, and they are therefore **superseded rather than re-used**. The
revision re-selects θ and ε from scratch under the explicit criterion at R1.3, and reports the
resulting per-fold values with the selection surface.

**Selected values (Transformer, complete calibrated policy).** Chosen on the policy-tuning
partition of validation speakers only:

| Fold | θ | ε | Validation macro-F1 | Fixed-300 validation macro-F1 | Mean decision time | Forced fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.95 | 0.20 | 0.786 | 0.787 | 212.6 ms | 6.9% |
| 1 | 0.30 | 0.00 | 0.200 | 0.200 | 300.0 ms | 100% |
| 2 | 0.95 | 0.20 | 0.664 | 0.655 | 228.5 ms | 16.8% |
| 3 | 0.30 | 0.00 | 0.466 | 0.466 | 300.0 ms | 100% |
| 4 | 0.30 | 0.00 | 0.538 | 0.538 | 300.0 ms | 100% |

We draw the reviewer's attention to folds 1, 3 and 4, where the constrained search selects the
**lowest admissible θ with ε = 0** and the policy then never fires early: every utterance falls
back to 300 ms. After calibration the posteriors in those folds are never confident enough to
satisfy the entropy gate. The early-decision mechanism is therefore inactive in three of five
folds, which we report rather than average away.

### R2.7 Five model families are claimed but only three appear in the results

**Author response.** The reviewer is correct, and the discrepancy is real. §3.4 named five
families while §4–§5 reported three. We resolve it by **withdrawing the two unreported models**,
not by adding results for them.

For **EfficientNet-lite**, no variant identifier, layer definition, width/depth coefficient,
resolution setting, checkpoint, or prediction file survives. The only preserved trace is a
parameter count of **57,998**, which is far smaller than any published EfficientNet-Lite variant
and indicates a custom EfficientNet-*like* network rather than a standard one. Naming a variant
now would be a reconstruction presented as a record. For **CNN-GRU**, the layer definitions and
predictions are likewise unrecoverable.

Both are therefore removed from the revised comparison, and §3.4 now names exactly the models
that appear in the results: small CNN, CNN–BiLSTM, and the short-prefix Transformer, plus the
WavLM Base+ modern reference (R1.10). We consider one reproducible modern baseline more valuable
than two unreproducible historical rows. See also R3.4.

### R2.8 Missing demographic and clinical distributions

**Author response.** Accepted. We report what the corpus metadata supports, and we **label what
it does not** rather than infer it.

**Sex distribution** (15 speakers; 8 dysarthric, 7 control):

| Sex | Dysarthric speakers | Control speakers | Dysarthric utt. | Control utt. | Total utt. |
|---|---:|---:|---:|---:|---:|
| Female | 3 (F01, F03, F04) | 3 (FC01, FC02, FC03) | 943 | 2,149 | 3,092 |
| Male | 5 (M01–M05) | 4 (MC01–MC04) | 1,991 | 3,048 | 5,039 |
| **Total** | **8** | **7** | **2,934** | **5,197** | **8,131** |

**Utterance type**, derived from the prompt text of each recording:

| Prompt type | Control | Dysarthric | Total |
|---|---:|---:|---:|
| Single word | 3,764 | 2,145 | 5,909 |
| Multi-word / sentence | 1,337 | 714 | 2,051 |
| Non-word / instructed task (e.g. sustained or repeated syllables) | 96 | 75 | 171 |
| **Total** | **5,197** | **2,934** | **8,131** |

This makes an important limitation visible that the submitted manuscript obscured: the corpus is
**dominated by isolated single words** (72.7%), so results should not be read as evidence about
continuous or conversational speech (R1.17).

**Severity and neurological etiology are not present in the surviving corpus index.** Rather than
impute them, we state this explicitly and report per-speaker results (R1.18) so that
speaker-level heterogeneity — which partly reflects severity — is at least visible. Where
severity or etiology can be sourced from authoritative TORGO documentation, it is reported with
that source cited; fields that cannot be so sourced are labelled **not available** rather than
estimated.

### R2.9 Provide the actual speaker IDs assigned to train/validation/test for each fold

**Author response.** Provided in full. Each of the 15 speakers appears in the test partition
**exactly once** across the five folds, and no speaker appears in more than one role within a
fold, so the evaluation is strictly speaker-independent.

| Fold | Train | Validation | Test | Test utt. |
|---|---|---|---|---:|
| 0 | F01, F03, FC02, FC03, M03, M04, MC01 | F04, M02, MC02, MC03 | FC01, M01, M05, MC04 | 1,689 |
| 1 | F03, FC01, FC02, FC03, M01, M04, M05, MC04 | F01, M03, MC01 | F04, M02, MC02, MC03 | 1,940 |
| 2 | F03, F04, FC01, FC03, M01, M02, M05, MC02, MC03, MC04 | FC02, M04 | F01, M03, MC01 | 1,657 |
| 3 | F01, F04, FC01, M01, M02, M03, M05, MC01, MC02, MC03, MC04 | F03, FC03 | FC02, M04 | 1,288 |
| 4 | F01, F04, FC02, M02, M03, M04, MC01, MC02, MC03 | FC01, M01, M05, MC04 | F03, FC03 | 1,557 |

Test partitions sum to exactly 8,131. Note the consequence, which we now state in the text: fold
test sizes are **unequal** (1,288–1,940), which is precisely why pooled and fold-averaged metrics
differ and must be reported separately (R1.6, R1.20).

### R2.10 Full architecture and training details for independent reproduction

**Author response.** Provided. Complete layer-level architectures, filter sizes, hidden
dimensions, and verified parameter counts are at R1.2. Training settings, frozen before any test
evaluation:

| Setting | Value |
|---|---|
| Optimizer | **AdamW** (the submitted §4.3 said "Adam"; corrected) |
| Learning rate / schedule | 1.5 × 10⁻⁴, **constant** |
| Weight decay | 5 × 10⁻⁴ |
| Batch size | 64 |
| Max epochs / early-stopping patience | 35 / 8 |
| **Checkpoint selection metric** | **Speaker-balanced macro-F1** on the checkpoint partition, weighting each utterance by `1 / n(speaker)` |
| Gradient clipping | 5.0 (global norm) |
| Dropout / label smoothing | 0.30 / 0.05 |
| Class balancing | Inverse-frequency `WeightedRandomSampler`, with replacement (R1.8) |
| Augmentation (train only) | Gain −3…+3 dB; additive noise 20–35 dB SNR (p = 0.5); frequency masks ≤16 bins, time masks ≤4 frames (p = 0.5) |
| Primary seed / replication seeds | 20260819 / 20260820, 20260821 |
| Features | Log-magnitude STFT, 512-point FFT, 25 ms periodic Hann, 10 ms hop, **no centred padding**, per-utterance z-score |

Checkpoint selection, temperature fitting, and policy selection use **disjoint** validation
subsets (50% / 25% / 25%, partitioned deterministically by hashed sample ID), so the checkpoint
is not chosen on the same rows used to calibrate or to select θ and ε.

We report one further correction that the reviewers' concern about speaker imbalance led us to.
Selecting checkpoints on pooled validation NLL proved unusable on this corpus, because each
fold's validation partition is **38–77% a single speaker** (folds 2 and 3 contain only *two*
validation speakers). Pooled NLL therefore largely measures one speaker: it swung by a factor of
two between consecutive epochs, and in one fold it never improved on the epoch-0 value, so early
stopping preserved an effectively untrained checkpoint while training loss fell steadily.
Checkpoints are now selected on **speaker-balanced macro-F1**, in which every utterance is
weighted by `1 / n(speaker)`. This was decided from training and validation curves alone, with no
test data involved. It is worth stating plainly that speaker imbalance in this corpus degrades
**model selection**, not merely the reporting of results — a stronger version of the point made
in R1.7, R1.18, and R2.11. Seeds, configuration
hashes, environment manifest, and the Git commit are recorded in every run summary.

**One reproducibility caveat we disclose rather than omit:** the primary runs execute on an Apple
Silicon MPS backend, where at least one backward operation used by these models
(`index_put_with_accumulate`) has no deterministic implementation. Runs are seeded and
configuration-hashed, and the full prediction ledger is published, but we do **not** claim
bit-exact reproducibility on this backend. We report multi-seed replication (three seeds) so that
seed-level variation is visible rather than hidden behind a single run.

### R2.11 Add fold variability, speaker-wise results, and statistical significance testing

**Author response.** Accepted in full — the submitted manuscript reported point estimates only.
The revision adds: per-fold accuracy, macro-F1, dysarthric recall, mean decision time, and
forced-fallback rate with **mean ± SD** and **speaker-clustered bootstrap 95% confidence
intervals**; per-speaker results for all 15 speakers (R1.18); and paired statistical comparison
between the adaptive policy and the fixed-duration baselines on identical utterances, using
paired discordant-pair testing together with speaker-clustered paired bootstrap effect intervals
(R1.12).

We use speaker-clustered resampling rather than utterance-level tests because the 8,131
utterances are not independent — they come from 15 speakers with 132 to 1,108 utterances each —
and treating them as independent would substantially overstate significance. We also commit not
to read a non-significant result as equivalence. The measured comparison against each fixed
baseline is given at R1.12, and the per-speaker table at R1.18.

### R2.12 The early-decision benefit is exaggerated by comparing against the mean of fixed durations rather than the best fixed baseline

**Author response.** This criticism is correct and we accept it without qualification. It was the
weakest comparison in the paper.

The submitted manuscript compared the adaptive policy against a fixed-duration baseline at
**225 ms** — the arithmetic mean of the four evaluated durations. That is not a deployable
system: no real classifier waits 225 ms because 225 is the average of 150, 200, 250 and 300. It
is a construct that makes the adaptive policy's mean latency look favourable while being
compared against no actual operating point.

The revision **removes the averaged baseline entirely** and compares the adaptive policy against
each **fixed, deployable** operating point separately — fixed 150, 200, 250, and 300 ms — with
the accuracy–earliness curve showing all of them on common axes (R1.11). The honest comparison
the reviewer asks for is against **fixed 300 ms**, the policy's own fallback and worst case: the
adaptive policy's benefit is exactly the latency it saves relative to 300 ms while holding
accuracy within a stated margin, which is why the selection criterion is now written in precisely
those terms (R1.3).

We also foreground the point the reviewer makes about the fallback rate. With roughly **48–57%**
of decisions reaching the 300 ms endpoint in the submitted results, the mean latency reduction
was about **35.4 ms** — approximately 12% of the 300 ms budget. The revision reports the
forced-fallback rate beside every latency figure and states in §5 and §6 that the practical
benefit is a modest reduction in *mean* decision time, not a reduction in worst-case latency,
which remains 300 ms by construction. Whether that trade is worthwhile is left to the reader with
the numbers needed to judge it, rather than asserted.

## Reviewer 3 — comments with author response

Several of these points were raised independently by Reviewers 1 and 2. Each is answered in full
below so this section can be read on its own; cross-references point to the shared evidence.

### R3.1 How is speech onset detected using the RMS detector, and what threshold, frame length, hop size, and onset criterion are used?

**Author response.** The submitted §3.1 named an RMS detector but gave none of its parameters.
The revision replaces that sentence with a complete and testable specification.

Every waveform is converted to mono 16 kHz. Short-time RMS is computed on left-aligned frames
with **no centred padding and no future reflection**, so no frame depends on samples beyond its
own right edge:

- frame length **20 ms** (320 samples), hop **10 ms** (160 samples);
- a frame is *active* when its RMS is greater than or equal to the threshold;
- the **onset criterion** is the first run of **two consecutive active frames**; the onset
  anchor is the first sample of the first active frame of that run.

We also correct a methodological defect the reviewer's question exposed. The detector recovered
from our records used a threshold of `max(0.003, peak_RMS − 35 dB)`, where `peak_RMS` is the
peak of the **whole utterance**. That rule reads future audio and is therefore not a causal
real-time detector, which contradicts the paper's minimal-latency claim. In the revision:

| Mode | Threshold | Role |
|---|---|---|
| `causal_noise_floor` (**primary**) | `max(0.003, median RMS of first 100 ms × 10^(15/20))` | causal and level-adaptive; all headline results |
| `causal_fixed` (comparator) | fixed absolute RMS floor `0.003` | causal but level-blind; retained to quantify its failure |
| `recovered_offline_peak` (comparator only) | `max(0.003, peak_RMS − 35 dB)` | **non-causal**; offline-alignment comparator |

The primary mode is the level-adaptive rule rather than the fixed floor because the onset
validation showed the fixed floor sits *below* these recordings' room tone, firing at sample 0 on
71.7% of utterances. Reported in full at R1.4; it also affects the submitted manuscript's
detector, on roughly 45% of utterances.

Utterances in which no two-frame run reaches the threshold are recorded as explicit detector
failures with a registered fallback anchor at sample 0; they are counted and reported rather
than silently dropped.

The reviewer's question also motivated a quantitative validation that the submitted manuscript
lacked. We manually annotate a prespecified stratified subset of utterances and report the
detection-failure rate together with signed and absolute onset error against those annotations,
and we repeat the main decision policy under onset perturbations of −50 to +50 ms.

The complete mathematical formulation, including the frame-level decision rule and the
correction that no temporal smoothing is applied, is given at R2.4; the quantitative onset-error
validation is at R1.4.

*Implementation:* `src/timeentropy/signal/onset.py`; parameters frozen in
`configs/rebuild_v1.yaml` (`onset:`); validation protocol in `docs/reconstruction_spec.md`.

### R3.2 Why was a fixed 30 ms pre-roll selected, and how does it affect classification performance?

**Author response.** The pre-roll preserves low-energy speech onsets — weak plosive releases,
breathy or aspirated starts — that an energy-threshold detector necessarily anchors *after*.
These transients are articulatory evidence, and discarding them would bias the input against
exactly the phenomena that distinguish dysarthric onsets.

The reviewer's question also exposes an ambiguity in the submitted text, which we now resolve
explicitly. The submitted §3.1 could be read as placing the 30 ms pre-roll *inside* the nominal
prefix, which would make a "200 ms" input contain only about 170 ms of post-onset speech while
the paper simultaneously claimed the full nominal duration followed onset. The revision fixes
one definition and applies it everywhere:

> A prefix of nominal duration *t* contains exactly *t* ms of **post-onset** signal, preceded by
> up to 30 ms of **already-observed** past context. Its total input length is therefore at most
> *t* + 30 ms, and the pre-roll never reduces the post-onset evidence available at time *t*.

The pre-roll costs no additional waiting: at the moment the onset is confirmed, that audio is
already in the buffer. It is history, not latency.

On the effect on performance, we do not assert a direction from the old results. The revision
reports a **condition-matched ablation** in which a zero-pre-roll model is *retrained* from
scratch with its own onset handling, its own temperature, and its own policy selection, so that
the comparison is not contaminated by reusing a checkpoint trained under a different input
definition. The measured result is at R1.5: the pre-roll makes no material difference
(mean −0.011 balanced macro-F1, sign varying by fold, against a between-fold SD of 0.14), so the
manuscript's articulatory-information justification is withdrawn.

See also R1.5, where the same definition is stated, and R1.15, where the pre-roll's zero
contribution to system latency is separated from the onset-confirmation delay, which is not zero.

*Implementation:* `crop_post_onset_with_preroll` in `src/timeentropy/signal/onset.py`;
ablation condition `causal_fixed_preroll0` in `configs/robustness_v1.yaml`.

### R3.3 Provide the complete architecture of the small CNN, including layers, kernel sizes, filters, pooling, activations, and fully connected layers.

**Author response.** Given in full below and in the revised manuscript. Convolutions use no bias
because each is followed by batch normalization. Input is a log-magnitude STFT tensor of shape
`B × 1 × 257 × T`; the output is two unnormalized logits ordered (control, dysarthric).

| Stage | Exact operation | Output channels | Parameters |
|---|---|---:|---:|
| 1 | Conv2d `1→16`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×2` | 16 | 176 |
| 2 | Conv2d `16→32`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×2` | 32 | 4,672 |
| 3 | Conv2d `32→64`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×2` | 64 | 18,560 |
| Pool | Length-aware mean over all valid frequency/time positions | 64 | 0 |
| Head | Dropout 0.30; Linear `64→2` | 2 | 130 |

**Total trainable parameters: 23,538**, verified programmatically rather than quoted. Each of the
three pooling stages halves both frequency and time. For mixed-length batches, padded STFT
values are zeroed before convolution and only valid downsampled positions enter the final mean,
so padding cannot contribute to a prediction.

The CNN–BiLSTM and Transformer architectures are given at R1.2, and full training settings
at R2.10.

*Implementation:* `src/timeentropy/models/small_cnn.py`; full specification for all
architectures in `docs/model_architectures.md`.

### R3.4 Which exact EfficientNet-lite variant is used?

**Author response.** We cannot answer this question honestly, and we will not supply a plausible
answer after the fact.

The submitted §3.4 named an "EfficientNet-lite" network, but no variant identifier, layer
definition, width/depth coefficient, resolution setting, checkpoint, or prediction file for it
survives. The recovered records preserve only a parameter count (57,998), which is far smaller
than any published EfficientNet-Lite variant and indicates a custom EfficientNet-*like* network
rather than a standard one. Naming a variant now would be a reconstruction presented as a
record.

**We therefore withdraw the EfficientNet-lite result from the manuscript.** For the same reason
we also withdraw **CNN-GRU**, whose layer definitions and predictions are likewise unrecoverable.
The submitted manuscript in any case named five model families while reporting only three.

The revised comparison contains three fully specified compact models — small CNN, CNN-BiLSTM,
and the short-prefix Transformer — plus a **modern pretrained speech reference** (WavLM Base+,
pinned to immutable revision `4c66d4806a428f2e922ccfa1a962776e232d487b`) evaluated on identical
speaker folds and identical input durations, with frozen-backbone and fine-tuned use labelled
separately. We consider a reproducible modern baseline more valuable than an unreproducible
historical row. Reviewer 2 raised the same discrepancy at R2.7, where the withdrawal is set out
in the same terms.

### R3.5 How are the 2D log-STFT representations converted into frame-wise token sequences for the Transformer and recurrent models?

**Author response.** In both cases the convolutional front end pools **frequency only**, never
time, so exactly one token is retained per input STFT frame and the nominal prefix duration maps
one-to-one onto sequence length.

**CNN-BiLSTM.** Two convolutional stages (`1→24`, `24→48`, each `3×3`, BatchNorm, SiLU) each
apply MaxPool `2×1`, reducing frequency `257→128→64` while leaving `T` unchanged. The remaining
64 frequency bins are averaged, giving a `B × T × 48` sequence that feeds a one-layer
bidirectional LSTM (hidden 64 per direction). Mixed-length batches use packed sequences and
masked pooling.

**Transformer.** The same two convolutional stages are used. Frequency is then adaptively
mean-pooled to **2 bins**, and the `48 × 2` values at each time step are flattened into a
96-dimensional token, projected by a bias-free `96→96` linear layer, and added to a sinusoidal
positional encoding, giving `B × T × 96`. The tokenizer never pools time. The encoder receives a
`src_key_padding_mask`, padded encoded states are cleared, and only valid tokens enter the final
masked temporal mean.

*Implementation:* `src/timeentropy/models/cnn_bilstm.py`,
`src/timeentropy/models/transformer.py`.

### R3.6 How are the entropy threshold θ and stability tolerance ε selected, including the search range, resolution, and optimization criterion?

**Author response.** The submitted manuscript stated only that these were "selected on validation
predictions" and did not define the search or the objective. The revision specifies all three.

- **Search range and resolution:** θ ∈ [0.30, 0.95] in steps of **0.05** (14 values);
  ε ∈ [0.00, 0.20] in steps of **0.02** (11 values); exhaustive grid, 154 combinations.
- **Selection data:** validation speakers only. Within each fold the validation speakers are
  deterministically partitioned by hashed sample ID into checkpoint-stopping (50%),
  calibration (25%), and **policy-tuning (25%)** subsets. θ and ε are selected on the
  policy-tuning rows alone, after temperature scaling has been fitted on the calibration rows.
  No test row participates in fitting or selection, and this is recorded as an explicit
  zero-count audit field in every metrics summary.
- **Optimization criterion:** minimize **mean observation time**, subject to the constraint that
  validation macro-F1 remains within **0.01 absolute** of that same fold's fixed-300 ms
  validation macro-F1. This makes the policy's purpose explicit — buy latency, but only at a
  bounded accuracy cost — rather than maximizing an unconstrained score.
- **Reporting:** the selected θ and ε are reported **per fold**, together with the validation
  response surface over the grid. Test results are reported only at the validation-selected
  parameters, with no retuning.

The submitted manuscript reported fold thresholds of 0.95, 0.90, 0.85, 0.85, 0.90 with a fixed
ε = 0.08 under an objective that was never defined; those values are superseded. The
newly selected per-fold values are given at R2.6.

The sensitivity surface requested by Reviewer 2 is at R2.1, and the per-fold selected values
at R2.6.

*Implementation:* `src/timeentropy/evaluation/selection.py`; grid frozen in
`configs/rebuild_v1.yaml`.

### R3.7 What is the value of C, and why is normalization by log C used for this binary classification task?

**Author response.** **C = 2**, the number of classes (control, dysarthric). This value was
never stated in the submitted manuscript; it now appears explicitly at Equation 2.

Shannon entropy over C classes is bounded above by log C, attained at the uniform posterior.
Dividing by log C = log 2 maps the entropy onto **[0, 1]** exactly, where 0 is a fully committed
posterior and 1 is maximum uncertainty. The purpose is interpretive rather than mathematical:
it makes a single threshold θ read on a fixed, dimensionless scale that is directly comparable
across models, folds, and prefix durations, instead of on a scale whose ceiling depends on the
label set.

We add a clarification that the submitted manuscript should have made and did not. For **C = 2**
the normalized entropy is a **monotone decreasing transform of the maximum softmax
probability**, so thresholding entropy at θ is equivalent to thresholding confidence at a
corresponding value. Normalized entropy is therefore *not* an independent uncertainty statistic
in the binary case, and we no longer imply that it is. The contribution we claim is the
**validated sequential stopping policy** — the combination of a calibrated confidence gate with
class and entropy stability across consecutive prefixes, selected under an explicit
latency/accuracy constraint on validation speakers only — not the uncertainty measure itself.
The normalized form is retained because it keeps θ interpretable and because it generalizes
unchanged to C > 2 in future multi-class severity work.

We do not merely concede this point in the text; we test it. The revised policy ablation
includes a **`max_softmax_confidence`** arm alongside `entropy_only`, `entropy_class_stability`,
`entropy_value_stability`, `full_policy`, and the `fixed_300` baseline, so the reader can see
directly how much of the result is attributable to the uncertainty statistic and how much to the
stability conditions and the constrained selection. The measured ablation is given at R1.1: the
`max_softmax_confidence` arm reproduces `entropy_only` to every digit, and no component of
Equation 4 produces a performance gain.

### R3.8 What is the value of Δ in Equation (4), and how is the previous prediction handled at the first candidate decision point of 200 ms?

**Author response.** **Δ = 50 ms**, the spacing between adjacent evaluated prefixes. This is now
stated at Equation 4.

The second part of the question identified a real inconsistency in the submitted manuscript,
which we thank the reviewer for catching. Equation 4 as written compares the prefix at *t*
against the prefix at *t* − Δ, which at *t* = 200 ms requires a prediction at 150 ms; but our
algorithm applied the stability conditions only "if a previous prefix exists," which silently
permitted a **confidence-only** decision at 200 ms. The equation and the implemented rule
therefore did not agree, and the 200 ms decision was weaker than the paper described.

The revision resolves this in favour of the stated equation:

- The **150 ms prefix is always computed** as a warm-up predecessor and is stored in the
  prediction ledger, but it is **never eligible to emit a decision**.
- A decision at 200 ms must therefore satisfy the **full** rule, including class stability and
  entropy stability with respect to the 150 ms posterior. There is no special case and no
  first-candidate exemption.
- The same trained model produces logits at 150, 200, 250, and 300 ms, so the predecessor costs
  no additional model.

Equation 4, Equation 5, the decision algorithm, and Figure 4 have all been updated so that the
text, the pseudocode, and the figure describe one identical rule.

*Implementation:* `src/timeentropy/evaluation/decisions.py`; prefixes frozen in
`configs/rebuild_v1.yaml` (`prefixes:`).

### R3.9 Why does the early-decision candidate set begin at 200 ms when a 150 ms prefix is also evaluated?

**Author response.** Because 150 ms serves a different function: it **initializes the stability
test** rather than producing an output. As set out in R3.8, the rule requires agreement between
consecutive prefixes, so the earliest prefix that can have a predecessor is 200 ms. Emitting a
decision at 150 ms would mean emitting one that no stability condition had ever constrained.

The justification is structural rather than empirical: we do not defend the 200 ms starting
point by appeal to the submitted fixed-duration numbers, since those are among the results being
regenerated. The eligible decision times are therefore **{200, 250, 300} ms** with a forced
fallback at 300 ms, which bounds worst-case latency while ensuring every emitted decision — the
earliest included — has passed the same test.

The revised manuscript labels the 150 ms row explicitly as a **warm-up observation, not a
deployable operating point**, in every table and figure where it appears, so it can no longer be
misread as a candidate decision time.

### R3.10 Why are a relatively large number of control samples misclassified as dysarthric in Figure 6, and what causes these false positives?

**Author response.** This comment prompted the most significant finding of our revision, and we
answer it in two parts.

**First, the figure could not be trusted.** In attempting to explain the false-positive count we
found that submitted Figure 6 does not reconcile with submitted Table 3: the confusion matrix
appears to pool all 8,131 out-of-fold utterances, whereas the accompanying table appears to
report unweighted fold means, and the two estimands were never labelled as different. The mean
latency in Table 3 likewise does not reproduce exactly from the decision counts in Table 4. We
could not resolve these discrepancies from surviving artifacts. Any causal account we offered
for the false positives in that specific figure would be an explanation of a figure we cannot
regenerate.

We therefore do not offer one. Figure 6 is being regenerated from the canonical prediction
ledger, and the revision reports **pooled out-of-fold metrics and unweighted fold mean ± SD as
two explicitly captioned estimands**, computed from the same decision rows that produce the
classification metrics, so that table, figure, and latency counts are mutually consistent by
construction.

**Second, the analysis the question deserves.** Rather than speculate, the revision includes a
prespecified **blinded error audit** of the regenerated matrix. A taxonomy of error categories
is defined *before* predictions are inspected, and false positives are then broken down by:
speaker identity; prompt type; utterance duration; onset error and onset-detection failure;
decision entropy; an SNR proxy; and decision time. Speaker-clustered bootstrap confidence
intervals accompany every rate, because with only 15 speakers a single control speaker with an
atypical recording condition can dominate an apparent error pattern — a possibility the
submitted analysis had no way to detect or exclude. Representative spectrograms are included
only after the taxonomy is fixed, so that illustrative examples cannot be selected to fit a
narrative.

**Third, the answer.** With the matrix regenerated, the question can be answered directly. The
pooled out-of-fold confusion matrix for the Transformer under the complete calibrated policy is:

| | Predicted control | Predicted dysarthric | Row total |
|---|---:|---:|---:|
| True control | 1,669 | **3,528** | 5,197 |
| True dysarthric | 1,192 | 1,742 | 2,934 |

Accuracy computed from this matrix is (1,669 + 1,742) / 8,131 = **41.95%**, which reproduces the
reported accuracy exactly, and the decision counts give a mean decision time of 269.4 ms, again
exactly. Table, figure, and latency counts are now consistent by construction.

The false-positive rate is **67.9%** (3,528 of 5,197 control utterances). The per-speaker
breakdown at R1.18 identifies the cause, and it is not a small subset of speakers or a
recording-condition artefact: the classifier carries a **global bias toward predicting
dysarthric**, labelling six of seven control speakers dysarthric on 65–98% of their utterances.
Its apparently strong dysarthric recall is the same bias seen from the other side.

We therefore do not attribute the false positives to speaker-specific noise, prompt type, or
onset error. The honest explanation is that the classifier does not separate the two classes on
held-out speakers at this prefix length; the prespecified breakdown by prompt type, duration,
onset error, entropy, and SNR proxy is reported in the revised §5 for completeness, but none of
those factors accounts for an error rate of this magnitude.

## Summary of changes to the manuscript

| Change | Prompted by | Location |
|---|---|---|
| All tables and figures regenerated from one canonical out-of-fold prediction ledger | R1.20, R1-EV.1, R3.10 | §5, Tables 2–5, Figs. 2–7 |
| Pooled OOF and unweighted fold-mean ± SD reported as separate, captioned estimands | R1.6, R1.20 | §5.1–5.2 |
| Fold-wise, per-speaker, and speaker-clustered bootstrap 95% CIs added | R1.7, R1.18, R2.11 | §5, new subsection |
| Paired statistical comparison against fixed-duration baselines | R1.12, R2.11 | §5.2 |
| Averaged 225 ms baseline removed; comparison against each deployable fixed operating point | R2.12 | §5.1–5.2 |
| Standardized accuracy–earliness metrics defined and reported | R1.11, R1-EV.6 | §5.1 |
| Calibration analysis (NLL, Brier, 15-bin ECE, reliability diagrams) added before the policy result | R1.13, R2.5 | new §5 subsection |
| Policy-component ablation incl. `max_softmax_confidence` arm | R1.1, R1-EV.5 | new §5 subsection |
| Complete causal RMS onset formulation; claimed smoothing stage removed | R1.4, R2.4, R3.1 | §3.1 |
| **Primary onset detector changed to the level-adaptive causal rule** after label-free validation; fixed-threshold and non-causal peak rules retained as comparators | R1.4, R2.4, R3.1 | §3.1, §5 |
| **Disclosure that the submitted manuscript's detector fires at sample 0 on ~45% of utterances**, leaving ~27.5% of prefixes containing room tone | R1.4, R2.4 | §3.1, §5.4 |
| Quantitative onset validation and onset-perturbation robustness | R1.4, R1-EV.8 | new §5 subsection |
| Prefix redefined as *t* ms post-onset plus ≤30 ms already-observed pre-roll; zero-pre-roll ablation | R1.5, R3.2 | §3.1, Eq. 1 |
| `C = 2` stated; entropy's equivalence to max-softmax confidence acknowledged for the binary case | R1.1, R2.2, R3.7 | §3.2, Eq. 2 |
| Δ = 50 ms stated; 200 ms first-candidate exemption removed; 150 ms defined as warm-up only | R3.8, R3.9 | §3.3, Eqs. 4–5, Alg. 1 |
| θ/ε grids, resolution, selection partition, criterion, per-fold values, and sensitivity surface | R1.3, R1-EV.3, R2.1, R2.6, R3.6 | §3.3, §4.3, §5 |
| Complete layer-level architectures and verified parameter counts | R1.2, R2.10, R3.3, R3.5 | §3.4, new appendix |
| Full training settings, seeds, and MPS determinism caveat | R2.10 | §4.3 |
| EfficientNet-lite and CNN-GRU withdrawn as unrecoverable | R2.7, R3.4 | §3.4, §5 |
| WavLM Base+ modern pretrained reference added on identical folds/durations | R1.10 | §3.4, §5 |
| Class distribution and exact balancing mechanism (incl. train-only application) | R1.8 | §4.1, §4.3 |
| Speaker demographics and prompt-type distribution; severity/etiology labelled unavailable | R2.8 | §4.1, Table 1 |
| Exact per-fold train/validation/test speaker IDs and unequal fold sizes | R2.9 | §4.2, new table |
| Structured protocol-comparable literature table | R1.9, R2.3 | §2 |
| Extended background: early time-series classification, uncertainty/rejection, calibration | R1.1, R2.2 | §1, §2 |
| Component-level latency decomposition; decision time distinguished from system latency | R1.15, R1-EV.9 | §5, new subsection |
| Complexity: parameters, model size, MACs/FLOPs with convention, memory, per-prefix runtime | R1.16, R1-EV.9 | §5, new subsection |
| Noise/SNR and onset-shift robustness matrix | R1.17, R1-EV.8 | new §5 subsection |
| Diagnostic language replaced with "utterance-level dysarthric-speech classification" | R1.19 | throughout |
| Generalization narrowed to head-mounted-microphone TORGO; cross-corpus and far-field stated as limitations | R1.14, R1.17, R1-EV.7 | §1, §5.4, §6 |

## Provenance note

All results come from one canonical out-of-fold ledger. The primary fold runs were produced at
commit `98e5953`; the test-time robustness evaluations consumed those checkpoints at a later
commit, and the difference is recorded in the execution plan rather than elided. No file under
`configs/`, `training/`, `models/`, `data/`, `signal/` or the fold-splitting module differs
between those commits, so the checkpoints are identical to ones the later code would produce; the
intervening changes are to reporting, documentation, tests, and a profiling script. Every
artifact records its own commit, configuration content hash, model-architecture hash, and
manifest hashes.

## Items we could not complete

We list these separately rather than leaving them implied:

| Request | Status | Reason |
|---|---|---|
| Cross-corpus test on UA-Speech or equivalent (R1.14, R1-EV.7) | **Not completed** | New-data experiment requiring separate access, task mapping, and protocol. Claim narrowed; listed as future work. No substitute offered. |
| Far-field / mobile / conversational microphone validation (R1.17, R1-EV.8) | **Not completed** | Requires recordings TORGO's head-mounted subset does not contain. Additive noise is reported but explicitly not presented as a far-field simulation. |
| Dysarthria severity and neurological etiology per speaker (R2.8) | **Partially completed** | Not present in the surviving corpus index. Reported only where authoritative TORGO documentation supports it; otherwise labelled not available rather than imputed. |
| Exact historical EfficientNet-lite and CNN-GRU results (R2.7, R3.4) | **Withdrawn** | No variant identifier, layer definition, checkpoint, or prediction file survives. Withdrawn rather than reconstructed. |
