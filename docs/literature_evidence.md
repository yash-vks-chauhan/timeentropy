# Primary-source literature evidence for the revision

Evidence cutoff: 19 August 2026. This is a targeted evidence review, not a
systematic review. It uses original peer-reviewed papers, official proceedings,
publisher pages, and official dataset/model documentation. A 2026 preprint is
included only where no final publication was found and is labelled accordingly.

`NR` means that the field was not reported in the primary source or was not
verified from the accessible primary text. It must not be filled by inference
from another paper that used the same corpus.

## Bottom line for the revision

1. The work should be framed as **early classification within an utterance**, not
   “early diagnosis” in the longitudinal clinical sense. Early time-series
   classification has two distinct objectives: predictive quality and earliness.
   Both should be reported rather than collapsing the result to accuracy alone.
2. Entropy-threshold exits have clear precedent, but most neural early-exit work
   saves network computation. Here, the resource being saved is **future acoustic
   observation time**. That distinction should be explicit.
3. For two classes, normalized Shannon entropy and maximum softmax probability
   rank examples identically. Entropy is therefore not a separate uncertainty
   signal from maximum softmax confidence in this binary task. The defensible
   contribution is the calibrated temporal stopping policy and its stability
   requirement.
4. Temperature scaling is a suitable post-hoc baseline, but its temperature must
   be fit on validation logits and its benefit must be measured. The stopping
   policy must operate on the calibrated probabilities and be selected without
   test data.
5. The closest verified short-input TORGO study uses 500 ms segments and
   speaker-level majority voting. No primary study found in this review matches
   onset-aligned 150--300 ms, speaker-independent TORGO detection with an
   adaptive observation-time decision. This supports a qualified evidence-gap
   statement, not an unqualified “first” claim.
6. TORGO results are highly protocol-dependent. Split unit, microphone channel,
   prompt subset, segment duration, prediction unit, and aggregation must precede
   any numerical comparison. Raw headline accuracies from incompatible protocols
   must not be ranked as if they measured the same estimand.
7. A frozen WavLM Base+ **backbone with a trained linear head** is a reasonable
   modern reference. It is not a
   reproduction of prior TORGO SSL studies: the strongest directly relevant
   peer-reviewed study used frozen wav2vec 2.0/HuBERT utterance embeddings, and
   selected the TORGO array recording rather than `headMic`.
8. Speaker-level dysarthria severity is available from Joy, Umesh, and Abraham
   (2017), who report speech-language-pathologist/Frenchay categories for all eight
   TORGO dysarthric speakers. Severity can support a descriptive error/recall
   stratification, but is confounded with speaker and is not an utterance-level severity
   label. Verified speaker-level neurological etiology remains unavailable.

## 1. Early time-series classification and earliness

### What the primary literature supports

- [Xing, Pei, and Yu (2012)](https://doi.org/10.1007/s10115-011-0400-x)
  formulated early time-series classification and introduced minimum prediction
  length with an early 1-nearest-neighbour classifier. It supports describing the
  task as prediction from an incomplete prefix while trying to retain full-series
  performance.
- [Mori et al. (2018)](https://doi.org/10.1109/TNNLS.2017.2764939) combined
  prefix-specific probabilistic classifiers with a stopping rule optimized for
  accuracy and earliness. It supports treating the stopping rule as a learned
  component and making its trade-off explicit.
- [Schäfer and Leser (2020), TEASER](https://doi.org/10.1007/s10618-020-00690-z)
  defines accuracy and mean normalized earliness separately and also reports a
  harmonic mean. TEASER periodically obtains probabilities, assesses reliability,
  and requires repeated predictions before exiting. It supports a temporal
  stability condition, but not this paper's particular entropy rule.

For test series `i`, TEASER's normalized earliness is

```text
earliness = mean_i(decision_prefix_i / full_length_i), lower is better.
```

Its equal-weight harmonic summary is

```text
HM = 2 * (1 - earliness) * accuracy
     / ((1 - earliness) + accuracy).
```

The harmonic mean is optional. It encodes an equal weighting between accuracy
and non-earliness and can hide clinically important differences. The revision
should therefore report the underlying quantities even if it adds a scalar
summary.

### Reporting implications for this study

The implemented reporting layer provides accuracy, macro-F1, dysarthric recall,
specificity, dysarthric precision, balanced accuracy, confusion counts, mean/SD,
median and p95 decision time, decision counts, fallback rate, fold, pooled OOF,
per-speaker, calibration, and speaker-clustered interval tables. These quantities
should be reported for fixed prefixes and the adaptive policy as space permits.

AUROC, latency IQR, and an empirical latency CDF were not implemented. They are
reasonable extensions, but the revision must not list them as delivered analyses or
silently reconstruct them from rounded manuscript tables. The underlying ledgers make
future computation possible.

The TEASER denominator is the complete time-series length. This study instead
has a prospective 300 ms decision horizon. If a normalized number is useful,
name it **fraction of the 300 ms policy horizon**, not TEASER earliness. Absolute
milliseconds after detected onset should remain the primary latency measure.
The 30 ms historical pre-roll is input context already observed at onset; state
clearly whether plots show post-onset decision time or total buffered audio.

## 2. Entropy, confidence, early exit, and rejection

### Evidence and its limits

- [Chow (1970)](https://doi.org/10.1109/TIT.1970.1054406) established the
  recognition-error versus reject trade-off under posterior risk. It supports the
  general idea that low-confidence predictions can be withheld. A terminal reject
  option is not the same operation as waiting for more temporal evidence.
- [Teerapittayanon, McDanel, and Kung (2016),
  BranchyNet](https://doi.org/10.1109/ICPR.2016.7900006) exits a neural network at
  an intermediate branch when softmax entropy is below a threshold. It is direct
  precedent for an entropy gate. Its exit axis is network depth and computation,
  not elapsed input time.
- TEASER is closer on the time axis: it evaluates successive prefixes and uses a
  reliability decision plus repeated labels. It does not establish that entropy
  is uniquely preferable to other confidence scores.

### Exact binary entropy--maximum-softmax relationship

For posterior probabilities `(p, 1-p)`, let

```text
m = max(p, 1-p), where 1/2 <= m <= 1.
```

With `C = 2`, normalized Shannon entropy is

```text
H_norm(p) = [-p ln(p) - (1-p) ln(1-p)] / ln(2)
          = h(m).
```

On `m in [1/2, 1]`,

```text
dh/dm = ln((1-m)/m) / ln(2) <= 0,
```

with strict inequality away from `m = 1/2`. Consequently,

```text
H_norm <= theta  if and only if  max_softmax >= q_theta,
```

where `q_theta` is the unique value in `[1/2, 1]` satisfying
`h(q_theta) = theta`. Thus, in a binary problem, entropy thresholding and
maximum-softmax thresholding select exactly the same examples after a one-to-one
threshold conversion.

Normalization by `ln(C)` places entropy in `[0, 1]`; here `C = 2`. It changes
only the numerical scale, not ordering or decisions after the threshold is
converted. For a binary-only paper, defining entropy with logarithm base 2 is
equivalent and may be easier to explain. The equivalence does not extend to
`C > 2`, because distributions with the same maximum probability can distribute
the remaining mass differently.

A temporal condition can still add information. Label stability compares the
current and preceding argmax. Entropy-value stability compares two time points,
but because `h(m)` is nonlinear, a fixed entropy-difference tolerance is not
equivalent to a fixed probability-difference tolerance everywhere. The revision
should describe this as temporal stabilization, not as a second independent
single-prefix uncertainty estimator.

### Revision-safe interpretation

Safe wording:

> Normalized binary entropy is used as a bounded confidence parameterization.
> For two classes it is a monotonic transform of maximum softmax probability;
> the additional temporal criterion tests whether confidence has stabilized
> across successive prefixes.

Avoid claiming that entropy is intrinsically richer than maximum softmax
confidence for this binary task. Keep the maximum-softmax policy as a sanity
check; equivalent performance after threshold conversion is expected, not a
surprising negative result.

## 3. Neural posterior calibration

[Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) showed that
modern neural networks can be miscalibrated and found scalar temperature scaling
effective across the image and document classification experiments they tested.
For logits `z` and `T > 0`, temperature scaling uses `softmax(z/T)`, with `T`
chosen on held-out validation data by minimizing negative log-likelihood.

Safe consequences for this study:

- fit `T` only on a calibration subset of the validation speakers;
- apply temperature scaling before computing entropy and before choosing
  `theta` and `epsilon`;
- select policy parameters on a validation subset distinct from the one used to
  fit `T` when feasible;
- never fit or select calibration/policy parameters on test speakers;
- report pre- and post-calibration NLL, Brier score, ECE with its binning scheme,
  and reliability plots, both overall and by prefix;
- note that positive scalar temperature preserves the argmax but changes
  confidence, entropy, and therefore decision time.

Guo et al. do not establish that temperature scaling will calibrate dysarthric
speech under speaker or recording shift. Improvement must be shown empirically,
and ECE alone is not enough because it depends on the binning scheme. The frozen
three-way validation partition in
[`configs/rebuild_v1.yaml`](../configs/rebuild_v1.yaml) is a prospective design
choice for this rebuild, not a procedure taken from Guo et al.

## 4. TORGO facts that must anchor comparisons

The [original TORGO paper](https://doi.org/10.1007/s10579-011-9145-0) describes
simultaneous recording with a head-mounted electret microphone at 16 kHz and an
eight-element Acoustic Magic Voice Tracker array at 44.1 kHz, followed by
cross-correlation alignment. It reports seven analyzed dysarthric participants
and seven matched controls. The later [official LDC release
LDC2012S02](https://doi.org/10.35111/cp5y-9z47) contains approximately 23 hours
from eight dysarthric speakers and seven controls and distributes mono 16 kHz
waveform files, including head-mounted and array-derived material.

The seven-versus-eight patient count is a corpus-version/detail distinction, not
a fact to silently harmonize. A paper must state the exact included speaker IDs.
Likewise, “TORGO audio at 16 kHz” does not identify the microphone. For every
experiment, report the selected file/channel convention explicitly.

[Joy, Umesh, and Abraham (2017)](https://doi.org/10.21437/Interspeech.2017-878)
report that a speech-language pathologist assessed TORGO dysarthria severity using
overall clinical intelligibility and articulator motor function under the Frenchay
assessment. Their Table 1 maps severe to F01, M01, M02, and M04;
severe-moderate to M05; moderate to F03; and mild to F04 and M03. This mapping covers
the eight dysarthric speakers in the rebuilt cohort. It supports descriptive
severity-stratified dysarthric recall or false-negative rates, not an independent
severity-prediction task: severity is constant within speaker, and two categories have
one speaker while the mild category has two. The paper does not provide a neurological
etiology mapping that should be imputed into this study.

[Schu, Janbakhshi, and Kodrasi (2023)](https://doi.org/10.1109/ICASSP49357.2023.10095981)
provide a particularly important validity warning. On phonetically matched
TORGO and UA-Speech subsets, several classifiers performed as well or better
using non-speech segments than using speech segments. They also found systematic
SNR differences between control and dysarthric recordings. This supports all of
the following checks:

- audit performance on pre-onset/non-speech context;
- repeat the main analysis without pre-roll;
- report onset failures and pre-onset energy by class and speaker;
- stratify errors by speaker, recording session, prompt type, and signal quality;
- avoid interpreting high accuracy as a speech-pathology biomarker without
  showing that the model does not rely primarily on recording conditions.

It does not prove that every TORGO classifier is invalid, nor that this rebuild
will share the same confound. It makes the proposed audits necessary.

The reconstruction provides label-free onset-mode statistics, a no-active-run/onset
status field, a condition-matched zero-pre-roll comparison, and quantitative error
strata. It does **not** provide human-reference onset error, a separate SNR-proxy error
stratum, or human cause coding. Those omissions must remain explicit limitations rather
than being presented as completed versions of every check above.

## 5. Candidate comparison table: binary dysarthria detection

The table deliberately preserves incompatible units and protocols. “Best” values
are descriptive within each paper, not a cross-paper ranking.

| Year | Study and task | Corpus | Microphone | Segment duration | Split unit | Speaker-independent? | Model / pretraining | Reported metric | Explicit comparability caveat |
|---|---|---|---|---|---|---|---|---|---|
| 2019 | [Millet & Zeghidour](https://doi.org/10.1109/ICASSP.2019.8682324), binary detection | TORGO, 7 dysarthric + 7 control speakers | NR | Recordings padded to 2.5 s | Fixed, disjoint train/validation/test speaker sets | Yes | Learnable time-domain filterbank; LSTM + attention; no external speech pretraining | Best reported test UAR `82.4 +/- 0.4` over 3 initializations | One fixed speaker split, 2.5 s inputs, and UAR; not an onset-aligned early-prefix policy |
| 2020 | [Hernandez et al.](https://doi.org/10.21437/Interspeech.2020-2354), binary detection (plus a separate severity task) | TORGO, 15 speakers; selected sentence set | NR | Whole sentence; exact duration NR | Sentences/prompts separated; all 15 speakers occur in train and test; 10-fold CV for tuning plus separate test | **No** for TORGO | RF, RBF-SVM, MLP; prosodic and rhythm features | Best TORGO detection accuracy `85.7%`, F1 `85.7%` | Prompt-disjoint but speaker-overlapping; cannot benchmark a speaker-independent system against this accuracy |
| 2021 | [Wang et al.](https://doi.org/10.21437/Interspeech.2021-2139), cross-domain binary detection | Labelled UA-Speech source; unlabelled TORGO target | NR | Whole utterance mel spectrogram; duration NR | Corpus-domain adaptation; TORGO audio is used unlabelled during training | N/A as a within-TORGO split | CBRNN attention encoder with dysarthria classification, domain adversarial training, and mutual-information minimization | UDA improved utterance weighted recall by `22.2` points and speaker accuracy by `20.0` points over its cross-domain baseline | Transductive/UDA use of unlabelled target-corpus audio; improvement is not a standard held-out TORGO score |
| 2022 | [Janbakhshi & Kodrasi](https://doi.org/10.1109/ICASSP43922.2022.9747205), binary detection from magnitude/phase representations | PC-GITA, 50 Parkinson's disease speakers + 50 neurotypical speakers; 24 words and one phonetically balanced text | NR | 500 ms representation windows (`50` non-overlapping 10 ms STFT frames), extracted with 50% overlap | Speaker-independent stratified 10-fold CV; development set same size as test set; five seeds and five speaker splits; segment scores soft-voted per speaker | Yes | Single- and dual-input CNNs; no external speech pretraining | Magnitude + instantaneous-frequency CNN: speaker-level ACC `93.68 +/- 5.32`, AUC `0.97 +/- 0.05` | Different disease/corpus/language and 500 ms windows; final score pools every segment available for a speaker, so it is not a single-prefix or online decision |
| 2023 | [Schu et al.](https://doi.org/10.1109/ICASSP49357.2023.10095981), binary detection and confound audit | TORGO subset, 7 dysarthric + 7 controls, 62 matched utterances per speaker | NR | 500 ms with 250 ms shift for Mel methods; utterance/segment aggregation varies by method | Leave-one-speaker-out; 90/10 training-speaker train/validation; majority vote for held-out speaker | Yes | Handcrafted SVMs, CNN/SRL Mel models, frozen and partially fine-tuned wav2vec 2.0 | Speaker accuracy for SRL: speech `71.1 +/- 3.1`, non-speech `100.0 +/- 0.0`, complete `91.1 +/- 3.1` | Primary purpose is confound diagnosis; speaker-level vote over many observations, not one early utterance decision; non-speech performance warns against naive comparison |
| 2024 | [Javanmardi et al.](https://doi.org/10.1016/j.specom.2024.103047), binary detection (plus separate severity task) | TORGO, 8 dysarthric + 7 control speakers; non-words, words, and sentences | **Array microphone**, 16 kHz | Complete variable-length utterance; exact duration NR | Leave-one-speaker-out; CNN reserves 10% of every training speaker's samples for validation | Yes | Frozen wav2vec 2.0 Base/Large and HuBERT layer embeddings, temporally averaged; SVM or CNN | HuBERT-24 + SVM: ACC `76.12%`, SE `.55`, SP `.81`, F1 `.57`, EER `.340`; HuBERT-25 + CNN ACC `75.12%` | Strong modern comparator, but different microphone, whole utterances, LOSO aggregation, prompt mixture, and no observation-time policy |
| 2025 | [Xiong et al.](https://doi.org/10.21437/Interspeech.2025-567), binary detection with ASR multitask tuning | Private 200 h source; TORGO used only as cross-corpus test | NR | Whole raw waveform sample; duration NR | Source train/evaluation/test are speaker-disjoint; no TORGO target tuning | Yes relative to the source corpus; within-TORGO split N/A | Partially fine-tuned wav2vec 2.0; frozen CNN encoder, tuned Transformer; ASR-primary gradient-projection MTL | TORGO sample-level accuracy `74.18%` | Cross-corpus generalization, much larger private source data, and sample-level metric; not a within-TORGO fold comparison |
| 2026 | **This reconstruction**, binary detection | TORGO LDC release subset, 8 dysarthric + 7 controls, `headMic` | **Head-mounted**, 16 kHz | 150/200/250/300 ms post-onset plus up to 30 ms historical pre-roll | Five predefined speaker-disjoint test folds; validation-speaker utterances partitioned without test use | Yes | Shared-prefix STFT CNN/CNN-BiLSTM/Transformer; frozen WavLM Base+ backbone + trained linear head | At fixed 300 ms, WavLM: ACC `59.29%`, macro-F1 `58.10%`, balanced ACC `59.18%`; compact-model balanced ACC `45.88–47.56%` | Newly generated reconstruction, not recovery of submitted results; one corpus/channel, short horizon, 15 speaker clusters, and WavLM balanced-accuracy interval includes chance; severity strata are nested in only 8 dysarthric speakers |

## 6. Short-segment and adjacent pathological-speech evidence

These studies inform design choices but must not be mixed into the binary TORGO
detection comparison.

[Janbakhshi and Kodrasi (2022)](https://doi.org/10.1109/ICASSP43922.2022.9747205),
included in the binary-detection table above, is the clearest exact-duration
non-TORGO reference found: its CNN inputs span 500 ms. Its final decision is a
soft vote over all windows belonging to a speaker, however, so the result does
not demonstrate accurate classification from one isolated 500 ms prefix.

| Study | Task | What it safely supports | Why its score is not comparable |
|---|---|---|---|
| [Gupta et al. (2021)](https://doi.org/10.1016/j.neunet.2021.02.008) | Dysarthria **severity-level classification**, UA-Speech | A ResNet can classify mel representations from what the paper calls short-duration segments; the paper reports overall accuracy `98.90%` and F1 `98.00%` | Not binary detection, not TORGO, exact segment duration and split details are NR in the accessible publisher text, and later work notes subset/protocol differences |
| [Ribas et al. (2023)](https://doi.org/10.1109/ACCESS.2023.3243986) | Healthy versus **general voice pathology**, SVD with AVFAD extension | Frozen/fine-tuned wav2vec 2.0, HuBERT, and WavLM representations can support pathological-voice classification; 5-fold SVD evaluation is speaker-disjoint; extension samples are 4 s | Voice disorders are not dysarthria; recordings, language, pathology mix, segment length, and training extension differ |
| [Javanmardi et al. (2024)](https://doi.org/10.1016/j.specom.2024.103047), severity experiments | Dysarthria severity, UA-Speech and TORGO | Frozen SSL layers can be probed for severity and layer choice matters | Severity excludes controls and has a different label space and split construction from detection |
| [Hernandez et al. (2026 preprint)](https://arxiv.org/abs/2603.22225) | Cross-lingual PD dysarthria detection from oral `/pa-ta-ka/` | Frozen HuBERT-Large, WavLM-Large, and XLS-R speaker embeddings can be used for speaker-independent dysarthria detection; language structure can confound SSL embeddings | Preprint, non-TORGO, Parkinson's disease only, DDK recordings pooled to speaker vectors, cross-lingual adaptation, and up-to-20-s chunks |

ASR, intelligibility regression, speech conversion, and severity classification
should be discussed in a separate paragraph of related work. WER, intelligibility
correlation, and severity accuracy do not estimate healthy-versus-dysarthric
detection performance.

## 7. Modern self-supervised speech representations

### Foundational model claims

- [wav2vec 2.0](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)
  learns contextualized raw-audio representations with masked latent inputs and a
  contrastive objective. Its original results are ASR results, not evidence of
  dysarthria detection.
- [HuBERT](https://doi.org/10.1109/TASLP.2021.3122291) uses masked prediction of
  clustered hidden-unit targets. It supports the model description used by
  Javanmardi et al.; its original paper does not establish clinical validity.
- [WavLM](https://doi.org/10.1109/JSTSP.2022.3188113) combines masked speech
  prediction with denoising and gated relative position bias for broad speech
  tasks. The [official Microsoft WavLM Base+ model
  card](https://huggingface.co/microsoft/wavlm-base-plus) specifies 16 kHz input
  and pretraining on 94,000 hours drawn from Libri-Light, GigaSpeech, and
  VoxPopuli. It does not claim TORGO or dysarthria training.

### What applied dysarthria studies establish

- Javanmardi et al. establish a strong case for a modern frozen representation
  baseline on TORGO. Their best TORGO detector uses a temporally averaged HuBERT
  layer, not WavLM, and the backbone is a feature extractor rather than an
  end-to-end fine-tuned clinical model.
- Schu et al. test frozen and partially fine-tuned wav2vec 2.0, but also show that
  recording conditions can dominate the apparent task. A modern encoder does not
  remove the need for confound audits.
- Xiong et al. show partially fine-tuned wav2vec 2.0 can be evaluated across
  corpora, while their results also illustrate the difficulty of distribution
  shift.
- The 2026 Hernandez et al. preprint applies WavLM-Large directly to
  speaker-independent dysarthria detection, but on multilingual Parkinson DDK
  corpora rather than TORGO. It supports the relevance of WavLM to dysarthria,
  not a numerical benchmark for this paper.

Therefore the revision may call `microsoft/wavlm-base-plus` a **modern
self-supervised reference**. It should not call it a previously established
TORGO state of the art. Report checkpoint ID and immutable revision, the frozen
backbone/trained-head distinction, pooling, trainable and total parameter counts,
and the exact raw-audio sample counts. WavLM runtime/FLOPs were not profiled in the
reconstruction and must not be inferred from compact-model profiles.

## 8. Claim-to-source ledger

| Primary source | Claims it can safely support | Claims it cannot safely support here |
|---|---|---|
| [Chow 1970](https://doi.org/10.1109/TIT.1970.1054406) | Posterior uncertainty can define an error--reject trade-off | A temporal wait policy, entropy specifically, or neural calibration |
| [Xing et al. 2012](https://doi.org/10.1007/s10115-011-0400-x) | Prefix classification and minimum prediction length are established eTSC ideas | This entropy/stability policy or speech-specific conclusions |
| [BranchyNet 2016](https://doi.org/10.1109/ICPR.2016.7900006) | Softmax entropy thresholds have been used for neural early exits | Savings in acoustic observation time; it exits across network depth |
| [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a.html) | Neural nets may be miscalibrated; held-out scalar temperature scaling is a strong simple baseline | Guaranteed calibration for TORGO or under speaker/recording shift |
| [Mori et al. 2018](https://doi.org/10.1109/TNNLS.2017.2764939) | Probabilistic prefix classifiers and an optimized stopping rule can trade accuracy for earliness | The particular entropy threshold, stability tolerance, or metric constraint used here |
| [TEASER 2020](https://doi.org/10.1007/s10618-020-00690-z) | Accuracy and normalized earliness are separate objectives; repeated predictions can support reliable exit | Treating a harmonic mean as the only outcome; this paper's absolute millisecond semantics |
| [Rudzicz et al. 2012](https://doi.org/10.1007/s10579-011-9145-0) | Original TORGO collection, tasks, participants analyzed, microphones, sample rates, and alignment | Which microphone or subset a later study used unless that study says so |
| [LDC2012S02](https://doi.org/10.35111/cp5y-9z47) | Distributed-release scope, 8 dysarthric + 7 control speakers, waveform organization | Exact experimental inclusion or preprocessing in a paper |
| [Joy et al. 2017](https://doi.org/10.21437/Interspeech.2017-878) | Speech-language-pathologist/Frenchay severity category for each of the 8 TORGO dysarthric speakers | Speaker etiology, utterance-level severity variation, or severity effects independent of speaker |
| [Millet & Zeghidour 2019](https://doi.org/10.1109/ICASSP.2019.8682324) | Speaker-disjoint TORGO detection from 2.5 s padded recordings; learnable waveform frontend | Sub-second detection, adaptive latency, or array/headMic comparability |
| [Hernandez et al. 2020](https://doi.org/10.21437/Interspeech.2020-2354) | Rhythm features improved its prompt-disjoint TORGO detector | Speaker-independent TORGO performance; comparison to the current fold protocol |
| [Wang et al. 2021](https://doi.org/10.21437/Interspeech.2021-2139) | Cross-corpus DSD can be posed as unsupervised domain adaptation | Standard target-independent evaluation, because unlabelled TORGO target audio is used in training |
| [Gupta et al. 2021](https://doi.org/10.1016/j.neunet.2021.02.008) | Short-duration inputs have been studied for dysarthria severity on UA-Speech | Binary TORGO detection or 150--300 ms feasibility |
| [Janbakhshi & Kodrasi 2022](https://doi.org/10.1109/ICASSP43922.2022.9747205) | Exact 500 ms PC-GITA inputs, speaker-independent 10-fold evaluation, and complementary STFT magnitude/phase evidence for binary detection | TORGO, a one-window decision, adaptive latency, or a conclusion that phase caused the full performance gain independently of speaker-level aggregation |
| [Schu et al. 2023](https://doi.org/10.1109/ICASSP49357.2023.10095981) | Speaker-independent TORGO evaluation, a verified 500 ms setup for Mel models, and strong recording-confound evidence | A clean disease-only interpretation of high TORGO accuracy; online single-utterance latency |
| [Ribas et al. 2023](https://doi.org/10.1109/ACCESS.2023.3243986) | SSL representations including WavLM can detect general voice pathology under its protocol | Dysarthria-specific or TORGO performance |
| [Javanmardi et al. 2024](https://doi.org/10.1016/j.specom.2024.103047) | LOSO TORGO detection using frozen wav2vec 2.0/HuBERT utterance embeddings and array audio | `headMic`, WavLM, sub-second, or fine-tuned-backbone claims |
| [Xiong et al. 2025](https://doi.org/10.21437/Interspeech.2025-567) | Partially tuned wav2vec 2.0 with ASR multitask learning and no TORGO target tuning can be tested cross-corpus | A within-TORGO comparison or a small-data training recipe |
| [Hernandez et al. 2026 preprint](https://arxiv.org/abs/2603.22225) | WavLM/HuBERT/XLS-R embeddings are relevant to speaker-independent cross-lingual PD dysarthria detection | Peer-reviewed final evidence, TORGO performance, or short-prefix performance |

## 9. Revision-ready claims and claims to avoid

### Defensible claims

- “We study early within-utterance dysarthria classification as an explicit
  accuracy--observation-time trade-off.”
- “The stopping criterion uses normalized binary entropy, which is a monotonic
  reparameterization of maximum softmax confidence, together with temporal
  stability across prefixes.”
- “In the primary sources reviewed through 19 August 2026, we did not identify a
  directly comparable speaker-independent TORGO study using onset-aligned
  150--300 ms prefixes and adaptive stopping.”
- “Prior TORGO results are contextualized rather than ranked because microphone,
  segment duration, speaker split, prompt subset, prediction unit, and aggregation
  differ.”
- “A frozen WavLM Base+ backbone with a trained linear head is evaluated as a
  modern reference alongside models trained from scratch.”

### Claims to avoid

- “Entropy is more informative than confidence” for two classes.
- “First ever” or “state of the art” without a systematic search and a matched
  evaluation protocol.
- “Early diagnosis” when the experiment only shortens audio observation within an
  already labelled utterance.
- “Speaker-independent” for a split that separates utterances or prompts but lets
  the same speaker occur in train and test.
- “TORGO uses the head microphone” for another study unless that study identifies
  it; the corpus contains multiple simultaneous recording streams.
- Any direct ranking of this rebuild against whole-utterance, array-microphone,
  speaker-level-vote, or target-adaptation results.
