# Point-by-point response to reviewers

Paper title: Minimal-Time Dysarthric Speech Detection Using Speech-Onset Time Entropy

We thank the organizers and the reviewers for the time and care they invested in this
manuscript. The comments identified genuine specification gaps and, in one case, an
inconsistency between a table and a figure that we were unable to explain from the submitted
artifacts. Acting on that comment led us to a finding we report here openly.

**Statement on the revised results.** While preparing this revision we established that the
original training code, model checkpoints, and per-sample prediction files no longer exist. We
could not reproduce the submitted numbers from surviving material, and we are not willing to
defend numbers we cannot regenerate. We have therefore rebuilt the entire experiment as a
frozen, version-controlled, reproducible pipeline and we replace **every** result table and
figure with values recomputed from a single canonical out-of-fold prediction ledger. The
previously reported values are referred to below only where a reviewer asked about them
directly. Two model families named in the submitted manuscript are withdrawn rather than
reconstructed from memory, for the reasons given in R3.4.

We also state plainly which requests we could not complete. Cross-corpus testing and
far-field/mobile microphone validation are new-data experiments rather than manuscript
revisions; they are now scoped as limitations and future work rather than answered with
synthetic substitutes.

Manuscript locations are given by section, table, and figure number. Page and line numbers
refer to the revised typeset manuscript.

---

## Reviewer 3 — comments with author response

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
| `causal_fixed` (**primary**) | fixed absolute RMS floor `0.003` | causal; used for all headline results |
| `causal_noise_floor` (secondary) | `max(0.003, median RMS of first 100 ms × 10^(15/20))` | causal; adaptive-noise sensitivity check |
| `recovered_offline_peak` (comparator only) | `max(0.003, peak_RMS − 35 dB)` | **non-causal**; retained only as an offline-alignment comparator |

Utterances in which no two-frame run reaches the threshold are recorded as explicit detector
failures with a registered fallback anchor at sample 0; they are counted and reported rather
than silently dropped.

The reviewer's question also motivated a quantitative validation that the submitted manuscript
lacked. We manually annotate a prespecified stratified subset of utterances and report the
detection-failure rate together with signed and absolute onset error against those annotations,
and we repeat the main decision policy under onset perturbations of −50 to +50 ms.

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
definition. **This value is pending completion of the frozen reruns and will be reported in the
revised manuscript.**

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
historical row.

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
ε = 0.08 under an objective that was never defined; those values are superseded. **The new
fold-specific selected values are pending completion of the frozen reruns.**

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
stability conditions and the constrained selection. **These ablation results are pending
completion of the frozen reruns.**

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

**The audit results are pending completion of the frozen reruns** and will appear in the revised
§5. We would rather give the reviewer a defensible answer late than a plausible one now.

---

## Reviewer 1 — comments with author response

*To be completed when Reviewer 1's comments are supplied.*

## Reviewer 2 — comments with author response

*To be completed when Reviewer 2's comments are supplied.*

---

## Summary of changes to the manuscript

| Change | Location |
|---|---|
| Complete causal RMS onset specification; non-causal peak rule demoted to offline comparator | §3.1 |
| Prefix redefined as *t* ms post-onset plus ≤30 ms already-observed pre-roll | §3.1, Eq. 1 |
| `C = 2` stated; normalized entropy's equivalence to max-softmax confidence acknowledged for binary case | §3.2, Eq. 2 |
| Δ = 50 ms stated; 200 ms first-candidate exemption removed; 150 ms defined as warm-up only | §3.3, Eq. 4–5, Alg. 1 |
| θ/ε search range, resolution, selection partition, and optimization criterion specified | §3.3, §4.3 |
| Complete layer-level architectures and verified parameter counts | §3.4, new appendix table |
| EfficientNet-lite and CNN-GRU withdrawn; WavLM Base+ modern reference added | §3.4, §5 |
| AdamW (not Adam) and full optimizer settings stated | §4.3 |
| All tables and figures regenerated from one canonical out-of-fold prediction ledger | §5, Tables 2–5, Figs. 2–7 |
| Pooled OOF and fold-mean ± SD reported as separate, captioned estimands | §5.1–5.2 |
| Onset-accuracy validation, calibration, ablations, robustness, and blinded error audit added | §5, new subsections |
| "Diagnosis" language replaced with "utterance-level dysarthric-speech classification"; generalization narrowed to head-mounted-microphone TORGO | throughout |
| Cross-corpus and far-field validation stated as limitations and future work | §5.4, §6 |
