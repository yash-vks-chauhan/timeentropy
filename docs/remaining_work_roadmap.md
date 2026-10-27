# Roadmap for the three remaining reviewer requests

This covers only the requests that can still be closed with work. It is written to be executed
by someone who has not been following the reconstruction, so each task states what the reviewer
asked, what exists already, what to change, the exact commands, and how to tell it worked.

Two other open requests are deliberately out of scope here because they need a human listener
(manual onset annotation, the 40-example qualitative review), and one is judged unfixable
(speaker-level neurological etiology, which TORGO reports only at cohort level).

| Task | Reviewer items | Blocker | Effort |
|---|---|---|---|
| A. Specify and train EfficientNet-lite0 and CNN-GRU | R3.4, R2.7 | none — compute only | ~1 day |
| B. Array-microphone evaluation | R1.17, R1-EV.8 | full TORGO release must be downloaded | ~1 day after download |
| C. Cross-corpus test on UA-Speech | R1.14, R1-EV.7 | UIUC licence, weeks of lead time | ~2 days after access |

## Rules that apply to all three

1. **Do not commit while a sweep is running.** `scripts/run_experiments.py run` stamps the
   preflight commit into every artifact and refuses any artifact carrying a different commit,
   even for a documentation-only change. Land commits first, then launch.
2. **`configs/rebuild_v1.yaml` is `status: frozen_before_test_evaluation` and must stay frozen.**
   None of these tasks may edit it. Each adds a *new* config file, so the existing 100-job
   matrix and every artifact hash it produced remain valid and reproducible.
3. **Nothing is a result until `scripts/run_experiments.py status` says so.** Unit tests
   establish implementation invariants, not experimental completion.
4. **Every number that reaches a document must be added to `scripts/verify_response_numbers.py`.**
   That script re-derives each quoted value from its artifact *and* asserts the value appears in
   a document, so a stale sentence cannot survive a rerun.

---

## Task A — Specify and train EfficientNet-lite0 and CNN-GRU

### What the reviewers asked

Reviewer 3 comment 4 asks which exact EfficientNet-lite variant was used. Reviewer 2 comment 7
notes that CNN-GRU and EfficientNet-lite are named as evaluated model families but never appear
in the results.

### Why the current answer is weak

Both are withdrawn, because no recoverable definition or checkpoint survives. That is honest but
it is not an answer to "which variant" — it declines the question. The fix is to stop trying to
recover the historical model and instead **specify one now**: train a named, fully documented
variant under the identical frozen protocol and report it as a new baseline, explicitly not a
recovery of the deleted result.

### Steps

**1. Add the dependency.** Neither `torchvision` nor `timm` is installed.

```bash
.venv/bin/pip install "timm>=1.0,<2"
```

Add it to `pyproject.toml` under a new `baselines` optional-dependency group, so the environment
stays declarative:

```toml
baselines = [
  "timm>=1.0,<2",
]
```

**2. Implement the two architectures** in `src/timeentropy/models/`, following the existing
files as the pattern — each takes `dropout`, accepts the `B×1×257×T` log-STFT tensor the other
compact models receive, and emits two logits ordered `(control, dysarthric)`.

- `efficientnet_lite.py` — wrap `timm.create_model("efficientnet_lite0", pretrained=False,
  in_chans=1, num_classes=2)`. Record the resolved `timm` version and the model's exact
  configuration in the run summary; that record *is* the answer to R3.4.
- `cnn_gru.py` — mirror `cnn_bilstm.py` exactly, substituting `nn.GRU` for `nn.LSTM` so the two
  differ in the recurrent cell alone and the comparison stays interpretable.

Register both in `src/timeentropy/models/factory.py`:

```python
_MODEL_TYPES: dict[str, type[nn.Module]] = {
    "small_cnn": SmallCNN,
    "cnn_bilstm": CNNBiLSTM,
    "cnn_gru": CNNGRU,                    # new
    "efficientnet_lite0": EfficientNetLite0,  # new
    "transformer": ShortPrefixTransformer,
}
```

Export them from `src/timeentropy/models/__init__.py`, and add a parameter-count test to
`tests/test_models.py` beside the existing ones — the manuscript quotes exact parameter counts,
so they must be asserted, not assumed.

**3. Create `configs/baselines_v1.yaml`.** Copy `configs/rebuild_v1.yaml` and change only:

- `experiment_name: "reviewer_baselines_v1"`
- `models.primary: [cnn_gru, efficientnet_lite0]`
- delete the `models.remove_unsupported_historical_claims` block — it no longer applies
- keep every other field byte-identical: same folds, same manifest, same onset rule, same
  prefixes, same training hyperparameters, same `primary_seed: 20260819`

Identical settings are the whole point. Any difference makes the new numbers incomparable with
the four systems already reported.

**4. Run.** Ten fold-runs, roughly 20 minutes each.

```bash
.venv/bin/python scripts/run_experiments.py preflight --config configs/baselines_v1.yaml
nohup caffeinate -dimsu .venv/bin/python scripts/run_experiments.py run \
  --config configs/baselines_v1.yaml --device mps --git-policy clean \
  --output-dir artifacts/orchestration/baselines > artifacts/orchestration/baselines.log 2>&1 &
```

Run it plugged in with Low Power Mode off, and leave the lid open — closing it suspends training.

**5. Evaluate and report**, following the pattern in `scripts/evaluate_completion_stages.sh`:

```bash
.venv/bin/python scripts/evaluate_ledgers.py --config configs/baselines_v1.yaml \
  --prefix-ledger artifacts/runs/reviewer_baselines_v1-fold{0..4}-cnn_gru-seed20260819-causal_noise_floor/prefix_predictions.parquet \
  --output-dir artifacts/evaluation/cnn_gru-baseline
.venv/bin/python scripts/generate_reports.py --config configs/baselines_v1.yaml \
  --prefix-ledger artifacts/evaluation/cnn_gru-baseline/calibrated_prefix_predictions.parquet \
  --decision-ledger artifacts/evaluation/cnn_gru-baseline/decisions.parquet \
  --output-dir artifacts/reports/cnn_gru-baseline
```

Repeat for `efficientnet_lite0`.

### Document changes

- **Manuscript Table 16** (fixed-prefix pooled results): add both models' rows at every prefix.
- **Manuscript §3.5**: add layer tables for both, matching Tables 3–5 in form, with verified
  parameter counts. State the exact `timm` model string and version for EfficientNet-lite0.
- **Manuscript §5.2**: extend the prose to five compact systems plus the WavLM reference.
- **Response R3.4**: replace the withdrawal with the specified variant, and say plainly that this
  is a newly specified baseline, not a recovery — the historical variant remains unknown.
- **Response R2.7**: both families now appear in the fixed-duration results.
- **Response "Requests Not Completed"**: remove the EfficientNet-Lite/CNN-GRU bullet.

### Done when

`run_experiments.py status --config configs/baselines_v1.yaml` reports all ten complete; both
models appear in Table 16; `verify_response_numbers.py` passes with the new checks added; and
`build_submission_documents.py` builds without `--allow-placeholders`.

### Honest limits to keep in the text

This does not recover what the earlier version ran. It replaces an unanswerable question with a
specified, reproducible baseline. Say so explicitly, or a reviewer will reasonably read the new
row as the old result restored.

---

## Task B — Array-microphone evaluation

### What the reviewers asked

Reviewer 1 comment 17: using only head-mounted recordings limits generalizability to mobile,
conversational and far-field settings, and "the limitation should be experimentally investigated
rather than only acknowledged." R1-EV.8 asks for evaluation under different microphone
conditions.

### Why this is the highest-value task

TORGO recorded **simultaneously** with a head-mounted electret microphone at 16 kHz and an
eight-element Acoustic Magic Voice Tracker array at 44.1 kHz — the same speakers, the same
utterances, a different microphone at a distance. See `docs/literature_evidence.md`, TORGO
collection section. Javanmardi et al. (2024) [20] used the array channel.

That makes a *matched* cross-channel experiment possible: identical speakers, prompts, folds and
protocol, with only the microphone changed. It is a genuine answer, not a synthetic proxy, and it
is far stronger evidence than the additive-noise stress test currently reported.

### Blocker

The local copy is a head-mic-only Kaggle subset, and `data/archives/kaggle/torgo-audio.zip` is a
truncated 1 MB placeholder, not a usable archive. Only `wav_headMic` exists under
`data/raw/torgo/`. Obtain the full TORGO release from the University of Toronto distribution and
confirm its licence terms before use.

### Steps

**1. Verify the array channel is present** after download:

```bash
find /path/to/torgo -type d -name "wav_arrayMic" | wc -l   # expect one per speaker-session
```

**2. Make the channel selectable.** `src/timeentropy/data/manifest.py` currently hard-codes the
channel in two places even though `configs/rebuild_v1.yaml` already declares
`data.selection.mic: "headMic"`:

- the row filter, `source["mic"].eq("headMic")`
- the recorded provenance dict, `"selection": {"mic": "headMic", ...}`

Change both to read `data.selection.mic` from the config. This is a real latent bug: the config
field is currently decorative. Add a test asserting a non-default channel selects a different
row set.

**3. Build the array manifest** into new files, leaving the frozen head-mic manifest untouched:

```bash
.venv/bin/python scripts/build_manifests.py --source-index /path/to/torgo_index.csv \
  --mic arrayMic --manifest data/metadata/torgo_manifest_array_v1.csv
```

Expect a different utterance count — array and head-mic sessions do not correspond one-to-one.
**Do not force it to 8,131.** Record the true count and, for the paired comparison, use the
intersection on `sample_id`, reporting how many utterances that intersection contains.

**4. Create `configs/array_mic_v1.yaml`.** Copy `configs/rebuild_v1.yaml` and change only the
manifest paths, `data.selection.mic: "arrayMic"`, and `experiment_name`. Leave `audio.sample_rate:
16000` alone — `src/timeentropy/data/audio.py` already resamples with `resample_poly`, so the
44.1 kHz source is handled. Verify on a few files that resampled duration matches the manifest.

**5. Run both arms.** The reviewer question has two halves, and they need different runs:

- **Transfer**: apply the existing head-mic checkpoints to array-mic audio. No training. This
  answers "does the deployed system survive a microphone change?" Add it as a new condition in
  `configs/robustness_v1.yaml` under the evaluation-only family, then use
  `scripts/run_robustness.py test-time`.
- **Condition-matched retraining**: train on array-mic audio under the identical protocol, five
  folds, via `run_experiments.py run --config configs/array_mic_v1.yaml`. This answers "is the
  method itself viable on a far-field channel?"

Report both. Transfer alone understates the method; retraining alone dodges the deployment
question.

**6. Compare** with the tracked paired analysis, which already handles this shape:

```bash
.venv/bin/python scripts/compare_retrained_condition.py \
  --condition-decisions artifacts/evaluation/transformer-arraymic/decisions.parquet \
  --condition-label array_mic --baseline-label head_mic \
  --output-dir artifacts/robustness/retrained/array_mic/comparison_vs_primary
```

### Document changes

- **Manuscript §5.5**: new subsection for the cross-channel result, with the paired
  speaker-clustered intervals, alongside the existing synthetic perturbations.
- **Manuscript §4.1**: state both channels, their native sample rates, and the resampling.
- **Manuscript §6**: narrow the channel limitation to what remains untested — mobile handsets,
  reverberant rooms, spontaneous conversational speech. The array microphone is not a phone.
- **Response R1.17 and R1-EV.8**: replace "partly completed" with the measured result.
- **Abstract and §1.2**: add the finding only if it changes a conclusion.

### Done when

Both arms complete and verify; the paired comparison exists with speaker-clustered intervals; the
response no longer claims synthetic noise as a substitute; and the numbers are in
`verify_response_numbers.py`.

### Honest limits to keep in the text

The array microphone is a fixed desktop array in the same recording room, not a mobile handset
and not a reverberant far-field environment. It answers the microphone half of R1.17 and leaves
the conversational and environmental halves open. Say that plainly.

---

## Task C — Cross-corpus evaluation on UA-Speech

### What the reviewers asked

Reviewer 1 comment 14 requires cross-corpus validation to support broader applicability. R1-EV.7
calls train-on-TORGO, test-on-UA-Speech "the strongest validation."

### Blocker

UA-Speech requires a signed licence from the University of Illinois Urbana-Champaign. Expect
weeks. Start the request before anything else in this document, because the lead time dominates.

### What to do before access arrives

The evaluation path can be built and tested against a held-out TORGO slice standing in for the
foreign corpus, so that the day access lands the run is mechanical.

**1. Write `scripts/evaluate_cross_corpus.py`.** It must load a trained TORGO checkpoint and a
foreign-corpus manifest, and apply the frozen policy **without refitting anything** — no
recalibration, no threshold reselection. Refitting on the target corpus would make it a domain
adaptation experiment, which is not what was asked.

**2. Decide the label mapping first, and write it down.** UA-Speech has different speaker
counts, severity strata, and prompt types. Specify before looking at any result:

- which UA-Speech speakers count as control versus dysarthric
- which prompt types are admissible, given TORGO training was mostly isolated tokens
- whether the microphone channel is comparable
- whether the sample rate needs resampling

**3. Prespecify the metrics** — the same estimands used throughout: pooled out-of-fold accuracy,
macro-F1, balanced accuracy, dysarthric recall, with speaker-clustered bootstrap intervals. Fix
this before the data arrives, so the analysis cannot drift toward a flattering framing.

**4. Dry-run the whole path** by treating one TORGO fold as a "foreign" corpus with an
independently built manifest. That exercises every code path and catches schema problems while
they are cheap.

### After access arrives

```bash
.venv/bin/python scripts/build_manifests.py --source-index /path/to/uaspeech_index.csv \
  --manifest data/metadata/uaspeech_manifest_v1.csv
.venv/bin/python scripts/evaluate_cross_corpus.py \
  --checkpoint artifacts/runs/<primary transformer fold>/best_checkpoint.pt \
  --manifest data/metadata/uaspeech_manifest_v1.csv \
  --output-dir artifacts/cross_corpus/uaspeech
```

Run every fold's checkpoint and report the spread across folds, not a single favourable one.

### Document changes

- **Manuscript**: new results subsection, and remove cross-corpus from §6 limitations.
- **Response R1.14 and R1-EV.7**: replace "Not completed" with the result.
- **Response "Requests Not Completed"**: remove the cross-corpus bullet.

### Done when

Every fold checkpoint is evaluated on UA-Speech with no refitting, intervals are
speaker-clustered, and the prespecified mapping is stated in the manuscript.

### Honest limits to keep in the text

Expect the numbers to be poor. The compact models are at or below chance on held-out TORGO
speakers already, so a cross-corpus test will most likely confirm the negative result rather than
rescue it. That is a legitimate and publishable outcome for a reproducibility study — but decide
now that it will be reported either way, so the analysis is not quietly abandoned if it is
unflattering.

---

## Suggested order

1. **Start the UA-Speech licence request today.** It is the long pole and costs nothing to begin.
2. **Task A**, which needs no external input and can run overnight.
3. **Task B** as soon as the full TORGO download completes — the highest evidential value of the
   three.
4. **Task C** when access arrives.

After each task: rerun `pytest`, `ruff check .`, `verify_response_numbers.py`, then
`build_submission_documents.py` with no `--allow-placeholders`, and re-check the rendered output
in Microsoft Word rather than LibreOffice — LibreOffice silently hides compatibility-mode and
caption-style differences that Word shows.
