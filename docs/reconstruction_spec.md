# Reconstruction specification and evidence register

## Reconstruction status

The original source code, checkpoints, validation logits, test predictions, and plotting
scripts are unavailable. The submitted manuscript's numerical results are therefore
historical claims, not reproduced evidence. The revision is a new, frozen reconstruction
whose outputs replace the deleted results.

The experiment matrix defines 100 independently verifiable units across primary compact
models, Transformer seed replications, WavLM, condition-matched input ablations, and
test-time robustness. `scripts/run_experiments.py status` is the authoritative
completion record; unit tests establish implementation invariants but not corpus-result
completion.

## Directly recovered and independently checked

- The surviving index contains 8,216 head-mounted-microphone rows from 15 speakers.
- Retaining `usable_prompt=True`, `duration_sec>=0.05`, and `headMic` gives exactly
  8,131 rows: 5,197 control and 2,934 dysarthric.
- The selected recordings total approximately 7.07 hours and 0.76 GiB of PCM16 sample
  payload.
- `configs/folds_v1.yaml` reproduces the older manuscript's validation/test speaker
  groups; each speaker is a test speaker exactly once.
- `data/metadata/torgo_folds_v1.csv` contains the exact sample/fold/role expansion used
  by the pipeline.
- Joy, Umesh, and Abraham (2017; <https://doi.org/10.21437/Interspeech.2017-878>)
  publish speech-language-pathologist/Frenchay severity categories for all eight
  dysarthric speakers: severe (F01, M01, M02, M04), severe-moderate (M05), moderate
  (F03), and mild (F04, M03).

The corpus and fold facts above do not authenticate the deleted preprocessing or model
implementation.

## Historical narrative settings, not proof of deleted code

- mono 16 kHz audio;
- a 20 ms RMS frame, 10 ms hop, threshold equal to the larger of 0.003 and
  35 dB below whole-utterance peak RMS, plus 30 ms pre-roll;
- 512-point FFT, 25 ms Hann window, 10 ms hop, no centered padding, log magnitude,
  and per-utterance standardization;
- AdamW at 0.00015, weight decay 0.0005, dropout 0.30, label smoothing 0.05,
  gradient clipping 5.0, at most 35 epochs, and patience 8;
- gain, additive-noise, frequency-mask, and time-mask augmentation;
- historical entropy values/grid whose exact objective and implementation did not
  survive; and
- narrative parameter counts for five model names without complete definitions.

These items motivated the reconstruction but are not cited as evidence that the old
experiments used exactly the rebuilt operations.

## Contradictions resolved by the reconstruction

1. The historical peak-relative onset threshold uses the whole utterance and is
   noncausal. It is retained only as an offline comparator.
2. The fixed absolute RMS floor often anchors at the first frame because it is below
   many recordings' initial level. The label-free diagnostic therefore selected the
   prespecified adaptive causal noise-floor rule as primary.
3. The old prefix wording placed 30 ms pre-roll inside a nominal duration while also
   claiming that the full duration followed onset. Rebuilt prefixes contain the full
   nominal post-onset duration plus up to 30 ms of history.
4. One old algorithm allowed a 200 ms entropy-only decision when no predecessor
   existed. The rebuild always computes 150 ms as the predecessor for 200 ms stability.
5. Old fold means, pooled confusion counts, and latency counts were not labelled as
   different estimands and did not reconcile. The rebuild derives each reported system
   from validated row-level decisions and labels pooled versus fold estimates.
6. An arithmetic average of fixed-duration scores was treated as a deployable baseline.
   The rebuild reports each fixed policy separately.
7. The old manuscript named five model families but did not preserve exact CNN-GRU or
   EfficientNet-Lite definitions. Those results are removed rather than reconstructed by
   guesswork.

## Frozen reconstruction decisions

### Audio, onset, and prefix semantics

- Decode finite audio, average channels before resampling, and use mono 16 kHz.
- Use left-aligned RMS frames of 20 ms (320 samples) with 10 ms hops (160 samples).
- The primary threshold is
  `max(0.003, median(first ten RMS frames) * 10^(15/20))`.
- Ten frame starts at 0, 10, ..., 90 ms require samples through 110 ms. The threshold
  is therefore acoustically available at 110 ms, not 100 ms.
- The onset anchor is the first frame in a run of two active frames. Online
  confirmation cannot precede either the end of that two-frame run or threshold
  availability. Detector execution time and acoustic availability are different
  quantities.
- A no-onset event is recorded. Any sample-zero fallback must not be described as a
  successfully localized onset.
- Nominal prefix time `t` is `t` ms of post-anchor signal plus up to 30 ms already
  observed pre-roll. Boundary padding preserves exact input length.
- Training/evaluation prefixes are 150, 200, 250, and 300 ms. The same model serves all
  lengths.

### Features and models

- Compact models consume log-magnitude STFT features from a 512-point FFT, 25 ms
  periodic Hann window, 10 ms hop, `center=False`, and per-utterance z-normalization.
- Small CNN, CNN–BiLSTM, and short-prefix Transformer architectures are fully defined in
  source and `docs/model_architectures.md`; tests pin parameter counts and padding/time
  semantics.
- WavLM uses raw mono waveform, immutable `microsoft/wavlm-base-plus` revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`, a frozen backbone, masked-mean pooling,
  dropout 0.30, and a trained 768→2 head.
- CNN-GRU and undefined EfficientNet-like results are excluded.

### Training and validation separation

- The primary seed is 20260819; Transformer replication seeds are 20260820 and
  20260821.
- Weighted sampling with replacement balances expected control/dysarthric draws within
  each training epoch; it does not alter the original cohort counts.
- Validation-speaker utterances—not separate speaker sets—are partitioned by SHA-256
  sample ID into checkpoint (50%), calibration (25%), and policy-tuning (25%) rows.
- Checkpoints are selected by speaker-balanced macro-F1 on checkpoint rows. Every
  speaker's utterances receive total weight one before the binary macro-F1 is computed.
- One scalar temperature per fold is fitted by NLL on all prefix logits in calibration
  rows.
- Policy parameters are selected only on policy-tuning rows.
- The primary Transformer architecture and primary seed were designated without using
  test performance.

### Sequential policy

- `C=2`, and normalized entropy divides Shannon entropy by `log(2)`.
- `theta` spans 0.30–0.95 by 0.05; `epsilon` spans 0.00–0.20 by 0.02.
- The policy minimizes validation mean observation time subject to macro-F1 being
  within 0.01 absolute of that fold's fixed-300-ms validation macro-F1.
- Tie-breakers are higher macro-F1, lower forced-fallback rate, lower `theta`, then
  lower `epsilon`.
- The 150 ms prediction initializes stability. Eligible outputs are 200, 250, and
  300 ms (`Delta=50 ms`), with forced fallback at 300 ms.
- Binary entropy gating is exactly equivalent to maximum-softmax gating after threshold
  conversion. Temporal stability is the only additional policy information.

### Reporting and inference

- Report pooled OOF metrics and unweighted fold mean ± sample SD separately.
- Report per-speaker metrics and speaker-clustered bootstrap 95% intervals.
- Use speaker-clustered paired bootstrap for primary comparisons; McNemar counts are an
  unclustered sensitivity analysis.
- Derive every metric, confusion count, and latency statistic for a system from the
  same validated decision rows.
- Runtime reporting separates post-onset observation, detector acoustic availability,
  and measured compute components.
- Severity-stratified dysarthric recall/false-negative rate may be reported descriptively
  from the published speaker-level mapping. Severity is nested within only eight
  dysarthric speakers—two strata have one speaker and one has two—so it cannot be
  disentangled from speaker effects or treated as an independent severity-prediction
  experiment.
- Do not infer a causal mechanism from noise shifts, speaker strata, onset status, or
  other descriptive associations.

## Generated evidence families

The phrase “one canonical ledger” means one validated decision source per reported
system, not one physical file for the entire paper. Evidence is distributed as follows:

- `artifacts/evaluation/`: calibrated prefix, fixed-policy, and adaptive decision ledgers;
- `artifacts/reports/`: pooled/fold/speaker/calibration/bootstrap/comparison tables;
- `artifacts/onset_diagnosis/`: label-free 600-utterance onset-mode diagnostic;
- `artifacts/sensitivity/`: exact theta/epsilon validation surface;
- `artifacts/robustness/test-time/`: 55 imposed noise/onset-shift fold-conditions;
- `artifacts/robustness/retrained/`: condition-matched onset/pre-roll comparisons;
- `artifacts/evaluation/transformer-three-seed/`: replication aggregate;
- `artifacts/error_audit/`: quantitative error strata and an uncompleted human-review
  sample; and
- `artifacts/profiling/`: compact-model timing and complexity.

Each artifact family records its own inputs, hashes, configuration identity,
environment, and producing commit. It is false to state that all results were generated
at one commit.

## Evidence that is not available

- human reference onsets, onset-error statistics, or inter-annotator agreement;
- completed human qualitative error taxonomy or causal false-positive explanations;
- independent-corpus validation;
- real far-field/mobile microphone validation;
- verified neurological etiology for each dysarthric speaker; and
- WavLM FLOPs, runtime, or isolated deployment memory.

These remain limitations. The empty annotation/review templates must not be treated as
completed evidence.

## Reproducibility and privacy notes

- The frozen `configs/rebuild_v1.yaml`, manifest metadata, all 8,131 values in the cohort
  manifest's forensic `recovered_path` column, and one invariant test contain
  machine-local paths. The superseded README example did too; this replacement removes
  that example. Frozen provenance inputs should not be silently edited after results are
  stamped. Public packaging must create portable staged copies and record pre/post
  hashes and transformations.
- TORGO raw audio and archives must never be redistributed in the supplement without
  explicit authorization. The public evidence package excludes them by construction.
- Checkpoints are large and excluded from the compact evidence package; run summaries
  retain their hashes, and published prediction/decision ledgers permit metric audits.
- The unblinded qualitative-review key and incomplete human worksheets are excluded.
- Apple MPS training was seeded but is not claimed bit-exact because a required backward
  operation is nondeterministic. Multi-seed results quantify seed variation; they do not
  replace speaker-clustered generalization uncertainty.
- The config's planning-era `reporting.canonical_*` paths are not the actual final
  per-system output locations. Evaluation/report CLIs use explicit paths under
  `artifacts/evaluation/` and `artifacts/reports/`.
