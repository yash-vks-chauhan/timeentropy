# Reviewer action matrix

This is a final-state disposition map, not an experiment progress tracker. A row is
supported only when its cited artifact exists and passes its own validation. The
authoritative run-completeness check is `scripts/run_experiments.py status`.

## Governing reconstruction decision

The submitted numerical results cannot be reproduced because the original code,
checkpoints, and predictions were deleted. The revision replaces them with results
from the frozen reconstruction. Historical values are mentioned only to explain the
revision; they are not mixed with regenerated evidence.

Results come from several hash-linked artifact families—primary evaluation ledgers,
reports, robustness ledgers, retrained ablations, sensitivity tables, the onset
diagnostic, error strata, and runtime profiles—not from one physical ledger or one Git
commit. Each claim must name its actual source.

## Experimental integrity and statistics

| Reviewer issue | Final disposition | Auditable evidence |
|---|---|---|
| Submitted Figure 6 did not reconcile with its tables | Replace the figure and every metric with regenerated out-of-fold results; validate labels, counts, latency, and selected-prefix linkage from the same decision rows | `artifacts/evaluation/<system>/`, `artifacts/reports/<system>/`, ledger invariant tests |
| Pooled and fold-averaged estimates were conflated | Report pooled OOF metrics and unweighted five-fold mean ± sample SD as different estimands | `pooled_metrics.*`, `fold_metrics.*`, `fold_summary.*` |
| Mean latency did not reproduce from decision counts | Compute latency and fallback rate from the canonical decision rows | `latency_counts.*`, `latency_summary.*`, `decisions.parquet` |
| A fixed-duration arithmetic mean was treated as a baseline | Compare deployable fixed 150, 200, 250, and 300 ms policies separately; compare the adaptive policy primarily with the best relevant fixed endpoint, not an average duration | `fixed_prefix_metrics.csv`, paired policy tables, accuracy–earliness coordinates |
| Historical threshold selection may have used test data | Split validation-speaker utterances deterministically into disjoint checkpoint (50%), calibration (25%), and policy-tuning (25%) row subsets; never fit on test speakers | `split_role` in prefix ledgers, temperature fits, policy selections |
| Fold/speaker domination was hidden | Publish all five folds, per-speaker metrics, pooled metrics, and speaker-clustered intervals | report fold/speaker/bootstrap tables |
| Statistical significance was unclear | Use speaker-clustered paired bootstrap for primary effect intervals; retain utterance-level McNemar counts only as an explicitly unclustered sensitivity analysis | paired comparison and McNemar tables |
| Seed sensitivity was unreported | Report the primary Transformer seed plus two frozen replication seeds on identical folds; describe three-seed intervals as descriptive, not speaker-generalization intervals | `artifacts/evaluation/transformer-seed*`, `transformer-three-seed/` |

## Method specification

| Reviewer request | Final disposition |
|---|---|
| RMS frame, hop, threshold, onset criterion | Specify 20 ms left-aligned RMS frames, 10 ms hops, two consecutive active frames, all threshold formulae, and separate onset anchor from acoustic availability/confirmation. The adaptive threshold uses ten frame starts at 0–90 ms; the last frame ends at 110 ms. |
| Primary onset mode | Use the level-adaptive causal noise-floor rule. Retain the absolute-floor and whole-utterance peak-relative rules as fixed-threshold and noncausal comparators. Do not call the label-free diagnostic human onset validation. |
| Quantitative human onset validation | Not performed: the deterministic 300-row worksheet is unannotated. Report only label-free anchor diagnostics and imposed ±10/30/50 ms shifts; do not report manual onset error or inter-annotator agreement. |
| 30 ms pre-roll | Define a nominal `t` prefix as `t` ms post-onset plus up to 30 ms already-observed history. Report the condition-matched zero-pre-roll retraining result; do not claim the pre-roll helps when the measured comparison does not show that. |
| Complete small-CNN architecture | Publish every convolution, batch-normalization, ReLU, time/frequency pooling operation, adaptive pooling, dropout, linear layer, tensor shape, and verified 23,538-parameter count. |
| CNN–BiLSTM tokens | Pool frequency only; the remaining `T` positions form `B × T × 48`, followed by one bidirectional LSTM layer and masked temporal pooling. |
| Transformer tokens | Pool frequency only, adapt frequency to two bins, and flatten `48 × 2` values into a 96-dimensional token for each time frame before positional encoding and the Transformer encoder. |
| EfficientNet-Lite and CNN-GRU | Remove both historical results. No exact architecture/checkpoint survives; do not invent a variant. |
| Modern speech reference | Use `microsoft/wavlm-base-plus` at immutable revision `4c66d4806a428f2e922ccfa1a962776e232d487b`, frozen backbone, masked-mean pooling, dropout 0.30, and a trained 768→2 head. |
| Entropy grid and selection | Search `theta=0.30…0.95` by 0.05 and `epsilon=0.00…0.20` by 0.02 on policy-tuning rows. Minimize validation observation time subject to macro-F1 being within 0.01 absolute of that fold's fixed-300-ms validation macro-F1; publish fold selections and the full validation surface. |
| `C` and `log C` | `C=2`; division by `log(2)` maps Shannon entropy to `[0,1]`. State that binary entropy is a monotone reparameterization of maximum softmax confidence. |
| `Delta` and first predecessor | `Delta=50 ms`; the 150 ms prediction is the predecessor for the 200 ms stability checks and is never eligible to emit a decision. |
| Why candidates start at 200 ms | 150 ms initializes temporal stability. Eligible outputs are 200, 250, and 300 ms, with forced fallback at 300 ms. |
| Training details | Publish optimizer, learning rate, weight decay, batch size, epochs/patience, label smoothing, gradient clipping, augmentation, weighted sampling, prefix sampling, seeds, validation partition, checkpoint metric, device, and nondeterminism caveat. |

## Validation and analysis disposition

| Requested analysis | Final evidence and interpretation |
|---|---|
| Calibration | Raw and temperature-scaled NLL, Brier, 15-bin ECE, reliability bins/figures, temperatures, and policy results. Calibration changes confidence/exit behavior; it is not assumed to improve accuracy. |
| Policy components | Fixed policies; entropy only; entropy + class stability; entropy + entropy-change stability; complete policy; exact maximum-softmax equivalence check. |
| Threshold sensitivity | Full validation-grid surface and fold summaries; test reporting only at validation-selected values. |
| Accuracy–earliness | Absolute decision time, decision counts, fallback rate, consumed fraction, timeliness, and the explicitly defined harmonic summaries. Do not hide accuracy and latency behind one scalar. |
| Noise/onset-shift robustness | All 55 prespecified fold-condition evaluations, pooled and fold summaries. Interpret white noise as a synthetic stress test; counterintuitive improvements indicate instability/class-bias shifts, not a robustness mechanism. |
| Input-definition robustness | Condition-matched zero-pre-roll, causal-fixed, and recovered-offline-peak retraining with condition-specific temperature and policy selection. Never substitute test-time recropping for retraining. |
| False positives/negatives | Quantitative rates by speaker, diagnosis, prompt type, decision time, duration, onset status, and calibrated entropy. These are associations, not causes. The current audit has no human-reference onset-error bin and no SNR-proxy stratum. Dysarthric recall by the published speaker-level severity mapping is reported descriptively; severity is constant within speaker, so it cannot be separated from speaker identity. |
| Human qualitative error causes | Not performed: the deterministic review sample has blank taxonomy fields. Do not claim blinded causal review or representative cause-labelled examples. Exclude the unblinded key from public supplements. |
| Practical latency | Separate post-onset observation time from detector acoustic availability, buffered pre-roll, crop/STFT/model/posterior/policy compute, and total warm batch-1 compute. A detector's CPU execution time is not its acoustic look-ahead. |
| Complexity | Report compact-model parameter counts, serialized state bytes, MAC/FLOP convention, and measured runtime. WavLM FLOPs/runtime and isolated model memory were not profiled. |

## Scope and claims

- Use “utterance-level dysarthric-speech classification” or “detection,” not clinical
  diagnosis.
- Limit empirical claims to the selected TORGO head-mounted channel and the evaluated
  speaker-exclusive folds.
- Report class balancing as inverse-frequency sampling with replacement and keep
  original cohort counts distinct from expected sampled draws.
- Report exact fold speaker IDs and only source-supported demographics. Joy, Umesh, and
  Abraham (2017; <https://doi.org/10.21437/Interspeech.2017-878>) publish
  speech-language-pathologist/Frenchay severity categories: severe (F01, M01, M02,
  M04), severe-moderate (M05), moderate (F03), and mild (F04, M03). Neurological
  etiology remains unavailable and must not be inferred.
- Treat severity-stratified recall/error rates as descriptive. Severity is constant
  within speaker, several categories contain only one or two speakers, and the study
  does not estimate severity as a prediction target.
- Contextualize prior studies by corpus, microphone, segment duration, split unit,
  prediction unit, aggregation, metric, and latency definition. Do not rank incompatible
  headline scores.
- Do not describe poor held-out performance as proof of “speaker memorization”; it is
  associated with poor speaker generalization, but the mechanism was not isolated.
- Do not call additive noise a proxy for far-field/mobile microphones.

## Reviewer requests not completed with the available evidence

| Request | Honest response |
|---|---|
| Cross-corpus validation | Not performed. It requires legally accessible data and a compatible task mapping. |
| Real far-field/mobile microphone validation | Not performed. TORGO `headMic` plus synthetic noise does not test a real channel shift. |
| Neurological etiology per dysarthric speaker | Not available from the verified sources; do not infer it from dysarthria or severity. |
| Manual onset accuracy and inter-annotator agreement | Not performed; human reference fields remain blank. |
| Human qualitative cause audit | Not performed; taxonomy fields remain blank. |
| WavLM deployment complexity/runtime | Not profiled. Only parameter identity and classification results are reported. |

These are limitations, not unfinished manuscript promises.

## Final deliverable gate

Before resubmission, require all of the following:

1. `run_experiments.py status` reports 100 complete jobs and no pending/invalid jobs.
2. Seed and retrained-ablation evaluations/aggregates exist and validate.
3. Error-audit outputs are regenerated after the merge fix, add descriptive dysarthric
   recall/false-negative rate by the published Joy et al. speaker-level severity mapping,
   and contain no unsupported human-onset, etiology, or qualitative-cause claims.
4. Every manuscript/response number is linked to its actual artifact family.
5. Clean and marked manuscript plus response DOCX files are rebuilt and every rendered
   page is inspected.
6. Numerical verification, all tests, Ruff, source audits, package privacy checks, and
   archive hashes pass.
7. The public package excludes TORGO audio/archives, checkpoints, private paths,
   incomplete human worksheets, the unblinded key, and superseded evidence.
