#!/usr/bin/env python3
"""Build an original-anchored, strictly minimal reviewer revision."""

from __future__ import annotations

import csv
import re
import tempfile
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "Time_Entropy_Dysarthric_Speech_Classification_IRO_Formatted.docx"
CLEAN = ROOT / "Time_Entropy_Dysarthric_Speech_Classification_STRICT_MINIMAL_REVISED.docx"
MARKED = ROOT / "Time_Entropy_Dysarthric_Speech_Classification_STRICT_MINIMAL_REVISED_MARKED.docx"
RESPONSE_SOURCE = ROOT / "Response_to_Reviewers_MINIMAL_REVISION.docx"
RESPONSE_OUT = ROOT / "Response_to_Reviewers_STRICT_MINIMAL_REVISION.docx"
FIGURE_DIR = ROOT / "tmp/strict_minimal_revision/figures"
SUPPLEMENT_NAME = "Supplementary_Material_MINIMAL_REVISION.docx"

EXPECTED_ORIGINAL_SHA256 = "c48a04e4ae143d06ad31d277a1417fcbdda464c8fb153cd95b49b14a59670190"


ABSTRACT = (
    "Automatic dysarthric speech detection can support rapid screening and assistive communication, "
    "but many speech-processing systems wait for relatively long utterance segments before producing "
    "a decision. This paper evaluates whether dysarthric-versus-control speech can be detected from "
    "short speech-onset prefixes and whether posterior entropy can reduce decision time inside a "
    "200-300 ms operating window. Experiments are conducted on the TORGO corpus with speaker-independent "
    "folds and head-mounted microphone recordings. Log-STFT prefixes are evaluated with compact "
    "convolutional, recurrent, and Transformer baselines. Regenerated out-of-fold results show that the "
    "compact models do not exceed chance balanced accuracy; the fixed 300 ms Transformer reaches 41.94% "
    "accuracy and 45.88% balanced accuracy. Its calibrated entropy-and-stability policy reaches 41.95% "
    "accuracy at 269.4 ms mean post-onset observation, with a 64.25% forced fallback. A frozen WavLM "
    "Base+ reference reaches 59.18% balanced accuracy at 300 ms, although its speaker-clustered interval "
    "includes chance. The results therefore support a reproducible latency audit, not a deployable detector."
)


PARAGRAPH_REPLACEMENTS: dict[int, str] = {
    7: ABSTRACT,
    17: (
        "A comparison of compact CNN, CNN-BiLSTM, and Transformer baselines over short spectrogram "
        "prefixes, together with a frozen WavLM Base+ reference [26]. Historical CNN-GRU and "
        "EfficientNet-lite entries are not retained because their exact implementations could not be recovered."
    ),
    48: (
        "The experiment trains three compact spectral classifiers on the same speaker-independent split "
        "structure: a small CNN, a CNN-BiLSTM, and a lightweight Transformer encoder. Small CNN: three "
        "Conv2d-BatchNorm-SiLU-MaxPool blocks (16/32/64 filters; 3 x 3 kernels, stride 1, padding 1; 2 x 2 "
        "pooling), valid-position mean, dropout 0.30, and Linear 64 to 2 (23,538 parameters). Sequence front ends "
        "use two 3 x 3 Conv2d blocks (24/48 filters) and 2 x 1 frequency-only pooling: CNN-BiLSTM averages "
        "frequency into 48-dimensional frame tokens; Transformer pools to two bins, forms and projects "
        "96-dimensional tokens, adds sinusoidal positions, and masks padding. Full shapes are in Supplementary "
        "Tables S1-S4. Unrecoverable CNN-GRU and EfficientNet-lite results are omitted. All use the same log-STFT "
        "input."
    ),
    60: (
        "Table 2 shows that the regenerated compact-model results do not exceed chance balanced accuracy. "
        "At 300 ms, the small CNN, CNN-BiLSTM, and Transformer obtain 47.47%, 47.56%, and 45.88% balanced "
        "accuracy, respectively; the Transformer's corresponding accuracy and macro-F1 are 41.94% and "
        "41.93%. A frozen WavLM Base+ reference reaches 59.18% balanced accuracy and 58.10% macro-F1 at "
        "300 ms, but its speaker-clustered 95% balanced-accuracy interval (47.84-71.30%) includes chance. "
        "Figures 2 and 3 show the corrected accuracy-latency and fixed-duration trends."
    ),
    62: "Figure 2. Corrected accuracy-latency curve for fixed-duration and entropy-based decisions.",
    64: "Figure 3. Corrected fixed-duration accuracy across speech-onset prefixes.",
    66: (
        "Table 3 summarizes the regenerated complete temperature-scaled early decisions together with fixed "
        "300 ms references. The Transformer early-decision system reaches 41.95% accuracy, 41.95% macro-F1, "
        "and 269.4 ms mean observation. This saves 30.6 ms without improving accuracy over its fixed 300 ms "
        "result (41.94%). Its 64.25% forced-fallback rate means that most samples still require the endpoint. "
        "Table 4 reports the corresponding decision counts."
    ),
    67: "Table 3. Corrected early-decision and fixed 300 ms aggregate test performance.",
    68: "Table 4. Corrected test-set latency counts for temperature-scaled early decisions.",
    69: (
        "The corrected latency distribution in Table 4 shows that 28.1% of Transformer decisions occur at "
        "200 ms, 5.0% at 250 ms, and 66.9% at 300 ms. Dysarthric recall is 59.37%, slightly below the fixed "
        "300 ms value of 60.05%. The raw Transformer's 300 ms expected calibration error is 0.429 and falls "
        "to 0.212 after temperature scaling. Figures 4 and 5 show the corrected latency counts and entropy trends."
    ),
    71: "Figure 4. Corrected distribution of temperature-scaled early-decision latencies.",
    73: "Figure 5. Corrected normalized entropy trajectories across prefix duration.",
    74: "5.3. Corrected Baseline Comparison",
    75: (
        "The regenerated experiment does not support the earlier focused-tuning improvement. Table 5 therefore "
        "compares corrected fixed 300 ms results on the same pooled out-of-fold utterances, including the frozen "
        "WavLM Base+ reference."
    ),
    76: "Table 5. Corrected fixed 300 ms comparison on pooled out-of-fold test utterances.",
    77: (
        "WavLM Base+ is the strongest evaluated reference at 300 ms, but its speaker-clustered interval includes "
        "chance. The entropy rule does not improve the compact-model accuracy-latency frontier. Figure 6 shows "
        "the corrected Transformer confusion matrix, and Figure 7 retains the example TORGO spectrogram prefixes."
    ),
    79: "Figure 6. Corrected confusion matrix for the temperature-scaled Transformer early-decision system.",
    83: (
        "The main finding is that the regenerated compact-model balanced accuracies are at or below chance and "
        "the entropy rule reduces mean observation without improving accuracy. It should not, however, be "
        "interpreted as a clinical diagnosis system. The corpus is small, the inter-speaker variation is high, "
        "and binary TORGO detection does not address severity grading or differential diagnosis. The current "
        "results also depend on reliable speech-onset detection, so performance may change in continuous speech, "
        "conversation, or noisy far-field recordings. The corrected Transformer has 3,528 false positives among "
        "5,197 controls; rates are higher when onset detection fails (91.87% versus 66.25%) and for sentence "
        "prompts (76.92% versus 64.69% for isolated tokens), but these overlapping strata do not prove causation. "
        "Removing the 30 ms pre-roll changes balanced accuracy by +5.98 points, but its speaker-clustered 95% "
        "interval (-1.62 to +15.23) crosses zero; the buffer is therefore a boundary safeguard, not a demonstrated "
        "performance advantage."
    ),
    84: (
        "Future work should add UA-Speech after access approval, report normal/mild/moderate/severe severity "
        "classification, compare the head-mounted and array-microphone channels, evaluate cross-corpus transfer, "
        "and test calibration methods that reduce the number of forced 300 ms decisions without increasing the "
        "false-positive rate. The WavLM result suggests that pretrained representations merit further study, "
        "although its present interval includes chance and does not establish clinical utility [26]."
    ),
    86: (
        "This paper presented a speech-onset time-entropy framework for minimal-time dysarthric speech detection. "
        "On speaker-independent TORGO folds, regenerated compact models remain at or below chance balanced "
        "accuracy; the complete temperature-scaled Transformer obtains 41.95% accuracy at 269.4 ms with 64.25% "
        "forced fallback. Frozen WavLM Base+ reaches 59.18% balanced accuracy at 300 ms, but its speaker-clustered "
        "interval includes chance. Taken together, these results do not establish a deployable detector; they "
        "support a reproducible evaluation of the early-decision protocol and its limitations. The framework is "
        "deliberately simple and model-agnostic, and it can be reused with stronger backbones or larger corpora as "
        "they become available."
    ),
}


RESPONSE_TEXT_REPLACEMENTS: dict[int, str] = {
    3: (
        "To keep the revision focused and traceable, this second revision is built directly from the original "
        "IRO-formatted manuscript. The title, six-section order, clinical background, five table positions, seven "
        "figure positions, declarations, and original references are retained. Only short reviewer-requested method "
        "clauses and unsupported results are changed in the main paper. Architectures, fold assignments, threshold "
        "surfaces, robustness checks, error strata, and runtime details remain in the Supplementary Material; minor "
        "requests that do not justify another main-text paragraph are answered here."
    ),
    9: (
        "Section 3.4 now gives a brief retained-model and frame-token description, while Supplementary Tables S1-S4 "
        "give every layer, output shape, and parameter count. The small CNN has 23,538 trainable parameters, "
        "CNN-BiLSTM 69,354, and Transformer 196,594. Historical CNN-GRU and EfficientNet-lite claims were removed "
        "because no exact definition or prediction record survives."
    ),
    12: (
        "We report exhaustive validation-only search over theta = 0.30-0.95 by 0.05 and epsilon = 0.00-0.20 by "
        "0.02 (154 pairs per fold). Selection minimizes validation observation time subject to macro-F1 remaining "
        "within 0.01 of that fold's fixed-300-ms macro-F1, with stated tie-breakers. The full 770-row surface "
        "accompanies the supplement."
    ),
    18: (
        "A t-ms prefix contains 30 ms of already observed history plus the full t ms after the returned anchor; "
        "pre-roll adds no future waiting. The condition-matched comparison did not establish a balanced-accuracy "
        "benefit for 30 ms, so the earlier performance-preservation claim was withdrawn."
    ),
    21: (
        "The main manuscript identifies pooled out-of-fold estimates explicitly. Complete fold means, sample SD, "
        "speaker-clustered intervals, and per-speaker results are supplied in the supplement to avoid enlarging the "
        "main paper."
    ),
    30: (
        "Section 2.2 retains a concise comparison with prior dysarthria work. We did not add a large main-text ranking "
        "table because corpora, input durations, speaker protocols, endpoints, and clinical targets differ; reporting "
        "those scores as directly ranked would be misleading."
    ),
    36: (
        "The revision reports accuracy and observation time separately because a single composite changes with the "
        "chosen normalization. Supplementary Table S20 reports consumed fraction, timeliness, and the stated "
        "accuracy-timeliness harmonic score; the adaptive policy does not improve the fixed-duration frontier."
    ),
    51: (
        "Supplementary Tables S2-S4 and S18 report compact-model parameter counts, serialized weights, MACs, FLOPs, "
        "CPU compute, and peak-process RSS increments. These details were kept out of the main manuscript to avoid "
        "adding another large table. WavLM deployment complexity was not profiled and is marked unavailable."
    ),
    57: (
        "Complete fold metrics and per-speaker accuracy/prediction rates are reported in Supplementary Tables S10 "
        "and S16. They show substantial heterogeneity that is not attributable to one speaker."
    ),
    63: (
        "The submitted Figure 6 and its metrics were withdrawn. The corrected matrix is generated from the same "
        "8,131 decision rows as all pooled metrics and gives (1,669 + 1,742) / 8,131 = 41.95% accuracy and "
        "269.39-ms mean observation."
    ),
    70: (
        "Completed. Every fold, mean and sample SD, speaker-clustered intervals, and per-speaker results are reported "
        "in Supplementary Tables S10, S13, S16, and S19 rather than adding another main-text table."
    ),
    82: (
        "Completed, with fixed 150 ms retained as a valid baseline and mandatory predecessor. Main Tables 2-4 report "
        "accuracy, macro-F1, observation time, and fallback; Supplementary Table S20 reports the standardized "
        "accuracy-timeliness comparison and states its normalization."
    ),
    91: (
        "Completed for compact models in Supplementary Table S18: parameters, serialized size, MACs, FLOPs, CPU "
        "inference path, RMS, STFT, policy, total warm compute, and acoustic observation are separated. WavLM "
        "deployment profiling remains unavailable."
    ),
    95: (
        "Exact theta and epsilon ranges, resolution, objective, tie-breakers, fold selections, and the full "
        "validation surface are reported. No universal threshold was supported."
    ),
    101: (
        "Section 2.2 compares the operating objective with prior work in concise prose. We did not add a large score "
        "table because input unit, corpus, speaker protocol, endpoint, and clinical target are incompatible across "
        "studies; a direct numerical ranking would imply comparability that is not present."
    ),
    104: (
        "Section 3.1 specifies the 20-ms left-aligned RMS frames, 10-ms hops, adaptive threshold, "
        "two-consecutive-frame rule, lack of smoothing, and failure handling. Label-free diagnostics and imposed "
        "shifts are reported separately, with timing details retained in the supplementary material."
    ),
    110: (
        "The selected (theta, epsilon) pairs are reported for all folds in Supplementary Table S12. Folds 0 and 2 "
        "selected (0.95, 0.20); folds 1, 3, and 4 selected (0.30, 0.00)."
    ),
    132: (
        "The detector uses 20-ms left-aligned frames, a 10-ms hop, no smoothing, an adaptive threshold equal to the "
        "larger of 0.003 and 5.62 times the first-ten-frame median RMS, and the first run of two active frames. "
        "Failure handling and signal-availability timing are also stated."
    ),
    138: (
        "We agree that reproducibility requires the complete small-CNN specification in the main paper. Section "
        "3.4 now gives all filters, kernels, pooling, activations, mean/dropout/head operations, and the parameter "
        "total; Supplementary Table S2 retains the layer-wise shapes."
    ),
    144: (
        "We agree that the 2D-to-sequence conversion must be explicit. Section 3.4 now gives frequency-only "
        "pooling, the exact CNN-BiLSTM and Transformer token dimensions, projection, positional encoding, and "
        "padding mask; Supplementary Tables S3-S4 retain the full shapes."
    ),
    147: (
        "The complete theta and epsilon ranges, resolution, 154-pair grid, validation partition, constrained "
        "objective, and tie-breakers are reported. Exact fold selections and the full surface accompany the supplement."
    ),
    150: (
        "C = 2. Division by log 2 maps Shannon entropy to [0, 1]; for a binary posterior it is a monotone "
        "reparameterization of maximum-softmax confidence."
    ),
    153: (
        "Delta = 50 ms. The stored 150-ms prediction supplies the predecessor for both class and entropy stability "
        "at 200 ms; there is no entropy-only exemption at the first candidate."
    ),
    162: (
        "Uncompleted requests are stated as limitations: human onset annotation, cross-corpus evaluation, real "
        "mobile, conversational, far-field, or array-microphone tests, historical model recovery, human error "
        "labels, and WavLM profiling. None is represented by an invented result."
    ),
}


RESPONSE_LOCATIONS: dict[int, str] = {
    7: "Introduction; Sections 3.2-3.3.",
    10: "Section 3.4; Supplementary Tables S1-S4.",
    13: "Sections 3.3 and 4.3; Supplementary Table S12 and accompanying CSV.",
    16: "Section 3.1; Supplementary Tables S7-S9 and S14.",
    19: "Sections 3.1 and 5.4; Supplementary Table S11.",
    22: "Section 4.2; Tables 2-5; Supplementary Tables S10 and S19.",
    25: "Sections 4.2 and 5.4; Supplementary Tables S10, S13, S16, and S19.",
    28: "Sections 4.1 and 4.3; Supplementary Table S5.",
    31: "Section 2.2; this response.",
    34: "Sections 3.4, 5.1, and 5.3; Table 5.",
    37: "Section 5.2; Supplementary Table S20.",
    40: "Section 5.2.",
    43: "Sections 4.3 and 5.2.",
    46: "Sections 2.1 and 5.4; Supplementary limitations.",
    49: "Section 3.1; Supplementary Table S18.",
    52: "Supplementary Tables S2-S4 and S18.",
    55: "Section 5.4; Supplementary limitations.",
    58: "Tables 2-5; Supplementary Tables S10 and S16.",
    61: "Abstract; Sections 5.4 and 6.",
    64: "Section 5.3; Figure 6.",
    68: "Sections 4.2 and 5.1-5.3; Tables 2-5; Figure 6.",
    71: "Supplementary Tables S10, S13, S16, and S19.",
    74: "Sections 4.3 and 5.2; Supplementary Table S12 and accompanying CSV.",
    77: "Section 5.2.",
    80: "Section 5.2; Supplementary Table S20.",
    83: "Sections 5.1-5.2; Tables 2-4; Supplementary Table S20.",
    86: "Section 5.4; Supplementary limitations.",
    89: "Section 5.4; Supplementary Tables S14-S17.",
    92: "Supplementary Table S18.",
    96: "Sections 3.3 and 4.3; Supplementary Table S12.",
    99: "Introduction; Sections 2.2 and 3.2.",
    102: "Section 2.2; this response.",
    105: "Section 3.1; Supplementary Tables S7-S9 and S14.",
    108: "Sections 4.3 and 5.2.",
    111: "Section 4.3; Supplementary Table S12.",
    114: "Introduction; Section 3.4.",
    117: "Section 4.1; Supplementary Table S17.",
    120: "Supplementary Table S6.",
    123: "Sections 3.4 and 4.3; Supplementary Tables S1-S5.",
    126: "Sections 4.2 and 5.4; Supplementary Tables S10, S13, S16, and S19.",
    129: "Section 5.2; Tables 2-3; Supplementary Table S20.",
    133: "Section 3.1.",
    136: "Sections 3.1 and 5.4; Supplementary Table S11.",
    139: "Section 3.4; Supplementary Table S2.",
    142: "Introduction; Section 3.4.",
    145: "Section 3.4; Supplementary Tables S3-S4.",
    148: "Section 4.3; Supplementary Table S12.",
    151: "Section 3.2.",
    154: "Section 3.3.",
    157: "Figure 1; Section 3.3.",
    160: "Section 5.4; Figure 6; Supplementary Tables S15-S16.",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fixed_rows(model: str) -> list[dict[str, str]]:
    return load_rows(ROOT / f"artifacts/evaluation/{model}-primary/fixed_prefix_metrics.csv")


def policy_row(model: str, policy_id: str) -> dict[str, str]:
    rows = load_rows(ROOT / f"artifacts/reports/{model}-primary/pooled_metrics.csv")
    return next(row for row in rows if row["policy_id"] == policy_id)


def latency_rows(model: str) -> list[dict[str, str]]:
    rows = load_rows(ROOT / f"artifacts/reports/{model}-primary/latency_counts.csv")
    return [row for row in rows if row["policy_id"] == "complete-temperature_scaled"]


def pct(value: str) -> str:
    return f"{100 * float(value):.2f}"


def make_figures() -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"cnn_bilstm": "#1f77b4", "small_cnn": "#ff7f0e", "transformer": "#2ca02c"}
    labels = {"cnn_bilstm": "CNN-BiLSTM", "small_cnn": "Small CNN", "transformer": "Transformer"}
    models = ["cnn_bilstm", "small_cnn", "transformer"]

    fig, ax = plt.subplots(figsize=(16, 10))
    for model in models:
        rows = fixed_rows(model)
        ax.scatter(
            [float(row["prefix_ms"]) for row in rows],
            [float(row["accuracy"]) for row in rows],
            s=120,
            color=colors[model],
            label=f"{labels[model]} fixed",
        )
        early = policy_row(model, "complete-temperature_scaled")
        ax.scatter(
            float(early["mean_latency_ms"]),
            float(early["accuracy"]),
            s=150,
            marker="X",
            color=colors[model],
            label=f"{labels[model]} entropy",
        )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.5, label="50% reference")
    ax.set(xlabel="Average decision latency (ms)", ylabel="Pooled test accuracy", ylim=(0.35, 0.62))
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    fig2 = FIGURE_DIR / "figure2_accuracy_latency.png"
    fig.savefig(fig2, dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 10))
    for model in models:
        rows = fixed_rows(model)
        ax.plot(
            [int(row["prefix_ms"]) for row in rows],
            [float(row["accuracy"]) for row in rows],
            marker="o",
            linewidth=2.5,
            color=colors[model],
            label=labels[model],
        )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.5, label="50% reference")
    ax.set(
        xlabel="Speech prefix duration (ms)",
        ylabel="Pooled test accuracy",
        xticks=[150, 200, 250, 300],
        ylim=(0.35, 0.62),
    )
    ax.grid(alpha=0.18)
    ax.legend(loc="best")
    fig.tight_layout()
    fig3 = FIGURE_DIR / "figure3_fixed_accuracy.png"
    fig.savefig(fig3, dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(3)
    width = 0.24
    for index, model in enumerate(models):
        rows = latency_rows(model)
        by_time = {int(row["decision_time_ms"]): int(row["count"]) for row in rows}
        ax.bar(
            x + (index - 1) * width,
            [by_time[t] for t in (200, 250, 300)],
            width,
            color=colors[model],
            label=labels[model],
        )
    ax.set(xlabel="Early-decision latency (ms)", ylabel="Utterance count", xticks=x, xticklabels=[200, 250, 300])
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="best")
    fig.tight_layout()
    fig4 = FIGURE_DIR / "figure4_latency_counts.png"
    fig.savefig(fig4, dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 10))
    linestyles = {"cnn_bilstm": "--", "small_cnn": "-", "transformer": ":"}
    for model in models:
        frame = pd.read_parquet(ROOT / f"artifacts/evaluation/{model}-primary/calibrated_prefix_predictions.parquet")
        frame = frame[frame["split_role"] == "test"].copy()
        for label_id, label_name, color in [(0, "control", "#1f77b4"), (1, "dysarthric", "#ff7f0e")]:
            values = (
                frame[frame["label_id"] == label_id]
                .groupby("prefix_ms", sort=True)["calibrated_normalized_entropy"]
                .mean()
            )
            ax.plot(
                values.index,
                values.values,
                color=color,
                linestyle=linestyles[model],
                linewidth=2.5,
                label=f"{label_name}, {labels[model]}",
            )
    ax.set(
        xlabel="Speech prefix duration (ms)",
        ylabel="Mean normalized entropy",
        xticks=[150, 200, 250, 300],
        ylim=(0.75, 1.0),
    )
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, loc="lower left")
    fig.tight_layout()
    fig5 = FIGURE_DIR / "figure5_entropy.png"
    fig.savefig(fig5, dpi=100)
    plt.close(fig)

    transformer = policy_row("transformer", "complete-temperature_scaled")
    matrix = np.array(
        [
            [int(transformer["true_control"]), int(transformer["false_dysarthric"])],
            [int(transformer["false_control"]), int(transformer["true_dysarthric"])],
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", fontsize=18)
    ax.set(
        title="Temperature-scaled Transformer early-decision matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["control", "dysarthric"],
        yticklabels=["control", "dysarthric"],
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig6 = FIGURE_DIR / "figure6_confusion.png"
    fig.savefig(fig6, dpi=100)
    plt.close(fig)

    return {
        "word/media/image2.png": ROOT / "artifacts/figures/figure_architecture.png",
        "word/media/image8.png": fig2,
        "word/media/image9.png": fig3,
        "word/media/image10.png": fig4,
        "word/media/image11.png": fig5,
        "word/media/image12.png": fig6,
    }


def token_chunks(old: str, new: str) -> list[tuple[str, bool]]:
    old_tokens = re.findall(r"\s+|[^\s]+", old)
    new_tokens = re.findall(r"\s+|[^\s]+", new)
    matcher = SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    chunks: list[tuple[str, bool]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            continue
        text = "".join(new_tokens[j1:j2])
        if text:
            chunks.append((text, tag != "equal"))
    return chunks


def reset_paragraph(paragraph: Paragraph, old: str, new: str, marked: bool) -> None:
    base_rpr = None
    for run in paragraph.runs:
        if run._r.rPr is not None:
            base_rpr = deepcopy(run._r.rPr)
            break
    for child in list(paragraph._p):
        if child.tag in {qn("w:r"), qn("w:hyperlink"), qn("m:oMath"), qn("m:oMathPara")}:
            paragraph._p.remove(child)
    chunks = token_chunks(old, new) if marked else [(new, False)]
    for text, highlight in chunks:
        run = paragraph.add_run(text)
        if base_rpr is not None:
            if run._r.rPr is not None:
                run._r.remove(run._r.rPr)
            run._r.insert(0, deepcopy(base_rpr))
        if highlight:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def reset_labeled_paragraph(paragraph: Paragraph, label: str, body: str) -> None:
    for child in list(paragraph._p):
        if child.tag in {qn("w:r"), qn("w:hyperlink"), qn("m:oMath"), qn("m:oMathPara")}:
            paragraph._p.remove(child)
    label_run = paragraph.add_run(label)
    label_run.bold = True
    paragraph.add_run(f" {body}")


def replace_substring(paragraph: Paragraph, old_substring: str, new_substring: str, marked: bool) -> None:
    old = paragraph.text
    if old_substring not in old:
        raise ValueError(f"Expected substring not found in paragraph: {old_substring}")
    reset_paragraph(paragraph, old, old.replace(old_substring, new_substring), marked)


def set_cell_text(cell, new: str, marked: bool) -> None:
    paragraph = cell.paragraphs[0]
    old = paragraph.text
    reset_paragraph(paragraph, old, new, marked)


def update_tables(document: Document, marked: bool) -> None:
    table2 = document.tables[1]
    models = [("cnn_bilstm", "cnn-bilstm"), ("small_cnn", "small-cnn"), ("transformer", "transformer")]
    row_index = 1
    for model, label in models:
        for row in fixed_rows(model):
            values = [label, row["prefix_ms"], pct(row["accuracy"]), pct(row["macro_f1"])]
            for col, value in enumerate(values):
                set_cell_text(table2.cell(row_index, col), value, marked)
            row_index += 1

    table3 = document.tables[2]
    table3_rows: list[list[str]] = []
    for model, label in models:
        row = policy_row(model, "complete-temperature_scaled")
        table3_rows.append(
            [
                "Entropy early decision",
                label,
                pct(row["accuracy"]),
                pct(row["macro_f1"]),
                pct(row["dysarthric_recall"]),
                f"{float(row['mean_latency_ms']):.1f}",
                pct(row["forced_fallback_rate"]),
            ]
        )
    for model, label in models:
        row = policy_row(model, "fixed_300")
        table3_rows.append(
            [
                "Fixed 300 ms",
                label,
                pct(row["accuracy"]),
                pct(row["macro_f1"]),
                pct(row["dysarthric_recall"]),
                "300.0",
                "0.00",
            ]
        )
    for ridx, values in enumerate(table3_rows, start=1):
        for cidx, value in enumerate(values):
            set_cell_text(table3.cell(ridx, cidx), value, marked)

    table4 = document.tables[3]
    for ridx, (model, label) in enumerate(models, start=1):
        rows = latency_rows(model)
        by_time = {int(row["decision_time_ms"]): row for row in rows}
        distribution = "; ".join(
            f"{time} ms: {int(by_time[time]['count'])} ({100 * float(by_time[time]['proportion']):.1f}%)"
            for time in (200, 250, 300)
        )
        for cidx, value in enumerate([label, "8131", distribution]):
            set_cell_text(table4.cell(ridx, cidx), value, marked)

    table5 = document.tables[4]
    header = ["System", "Decision", "Model", "Time", "Acc. (%)", "Macro-F1 (%)", "Mean latency (ms)", "Forced 300 ms (%)"]
    for cidx, value in enumerate(header):
        set_cell_text(table5.cell(0, cidx), value, marked)
    table5_rows = []
    for model, label in models:
        row = policy_row(model, "fixed_300")
        table5_rows.append(["Compact", "fixed", label, "300 ms", pct(row["accuracy"]), pct(row["macro_f1"]), "300.0", "0.00"])
    wavlm = fixed_rows("wavlm")[3]
    table5_rows.append(
        ["Frozen reference", "fixed", "WavLM Base+", "300 ms", pct(wavlm["accuracy"]), pct(wavlm["macro_f1"]), "300.0", "0.00"]
    )
    for ridx, values in enumerate(table5_rows, start=1):
        for cidx, value in enumerate(values):
            set_cell_text(table5.cell(ridx, cidx), value, marked)


def append_reference(document: Document, marked: bool) -> None:
    reference = (
        "[26]\tChen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., "
        "Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Wu, J., Zeng, M., Yu, X., "
        "& Wei, F. (2022). WavLM: Large-scale self-supervised pre-training for full stack speech "
        "processing. IEEE Journal of Selected Topics in Signal Processing, 16(6), 1505-1518. "
        "https://doi.org/10.1109/JSTSP.2022.3188113"
    )
    source = document.paragraphs[-1]
    clone = deepcopy(source._p)
    source._p.addnext(clone)
    paragraph = Paragraph(clone, document._body)
    reset_paragraph(paragraph, source.text, reference, marked)


def replace_media(docx_path: Path, replacements: dict[str, Path]) -> None:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False, dir=docx_path.parent) as handle:
        temp_path = Path(handle.name)
    try:
        with ZipFile(docx_path, "r") as source, ZipFile(temp_path, "w", ZIP_DEFLATED) as target:
            for item in source.infolist():
                data = replacements[item.filename].read_bytes() if item.filename in replacements else source.read(item.filename)
                target.writestr(item, data)
        temp_path.replace(docx_path)
    finally:
        temp_path.unlink(missing_ok=True)


def build(output: Path, marked: bool, media: dict[str, Path]) -> None:
    document = Document(ORIGINAL)
    for index, new_text in PARAGRAPH_REPLACEMENTS.items():
        paragraph = document.paragraphs[index]
        reset_paragraph(paragraph, paragraph.text, new_text, marked)

    replace_substring(
        document.paragraphs[26],
        "EfficientNet-style compact convolutional networks [12], recurrent networks [13], and Transformer encoders [14]",
        "Compact convolutional, recurrent [13], and Transformer encoders [14]",
        marked,
    )
    replace_substring(
        document.paragraphs[28],
        "a small CNN, a CNN-LSTM, a CNN-GRU, an EfficientNet-lite network, and a lightweight Transformer encoder",
        "a small CNN, a CNN-BiLSTM, and a lightweight Transformer encoder",
        marked,
    )
    replace_substring(
        document.paragraphs[32],
        "A 30 ms pre-roll is retained before the detected onset so that plosive or breathy starts are not removed, which is important because such transients can carry discriminative articulatory information.",
        (
            "The detector uses 20 ms left-aligned frames, a 10 ms hop, no smoothing, and the first two "
            "consecutive frames above max(0.003, 5.62 times the median RMS of the first 100 ms); failures are "
            "recorded and cropped from sample zero. A 30 ms pre-roll is retained as a boundary buffer; its "
            "ablation is reported in Supplementary Table S11."
        ),
        marked,
    )
    replace_substring(
        document.paragraphs[38],
        "This normalization maps the entropy into the interval [0, 1] for the binary task",
        "Here C = 2, so normalization by log 2 maps entropy into [0, 1] for the binary task",
        marked,
    )
    replace_substring(
        document.paragraphs[40],
        "Using the normalized entropy as the confidence signal keeps the decision logic independent of the particular classifier, since every model under comparison produces a posterior distribution over the same two classes.",
        (
            "Using normalized entropy keeps the decision logic independent of the classifier. For C = 2, "
            "entropy is a monotone transformation of maximum-softmax confidence, so the evaluated contribution "
            "is the sequential onset-prefix and stability protocol rather than a new uncertainty statistic."
        ),
        marked,
    )
    replace_substring(
        document.paragraphs[42],
        "The early-decision rule is constrained to the candidate set  = {200, 250, 300} ms.",
        (
            "The early-decision rule is constrained to candidate times {200, 250, 300} ms; the 150 ms "
            "prefix is a fixed baseline and supplies the previous prediction for the first 200 ms test."
        ),
        marked,
    )
    replace_substring(
        document.paragraphs[44],
        "where θ is the entropy threshold, ε is the entropy-stability tolerance, and Δ is the step between adjacent prefixes.",
        "where θ is the entropy threshold, ε is the entropy-stability tolerance, and Δ = 50 ms is the step between adjacent prefixes.",
        marked,
    )
    replace_substring(
        document.paragraphs[54],
        "Table 2 reports the fixed-duration test performance of the three retained baseline families across the four prefix durations.",
        (
            "Table 2 reports pooled out-of-fold fixed-duration performance over all 8,131 test utterances; "
            "fold variation, speaker-clustered 95% intervals, and per-speaker results are separated in "
            "Supplementary Tables S10, S13, and S16."
        ),
        marked,
    )
    implementation = document.paragraphs[57]
    implementation_old = implementation.text
    implementation_new = implementation_old.replace(
        "Optimization uses mini-batch gradient descent with the Adam optimizer",
        "Optimization uses AdamW (learning rate 1.5 x 10^-4, weight decay 5 x 10^-4, batch size 64)",
    ).replace(
        "Model checkpoints are selected by monitoring the validation loss",
        "Model checkpoints are selected by validation speaker-balanced macro-F1",
    ).replace(
        (
            "The entropy threshold, the stability tolerance, and the checkpoint are all chosen on the validation "
            "fold of each split, and the test fold is evaluated once with those frozen choices."
        ),
        (
            "Disjoint validation subsets fit temperature and search theta = 0.30-0.95 by 0.05 and epsilon = "
            "0.00-0.20 by 0.02, minimizing latency subject to macro-F1 within 0.01 of fixed 300 ms; the test fold "
            "is evaluated once with frozen choices (Supplementary Tables S5 and S12)."
        ),
    )
    reset_paragraph(implementation, implementation_old, implementation_new, marked)

    update_tables(document, marked)
    append_reference(document, marked)
    document.core_properties.comments = (
        "Strictly minimal reviewer revision built directly from the IRO-formatted source. "
        f"Detailed evidence is in {SUPPLEMENT_NAME}."
    )
    document.save(output)
    replace_media(output, media)


def build_response(source: Path = RESPONSE_SOURCE, output: Path = RESPONSE_OUT) -> None:
    document = Document(source)
    for index, body in RESPONSE_TEXT_REPLACEMENTS.items():
        paragraph = document.paragraphs[index]
        if index in {3, 162}:
            reset_paragraph(paragraph, paragraph.text, body, marked=False)
        else:
            reset_labeled_paragraph(paragraph, "Author Response:", body)
    for index, body in RESPONSE_LOCATIONS.items():
        reset_labeled_paragraph(document.paragraphs[index], "Manuscript location:", body)
    document.core_properties.comments = (
        "Response aligned to the original-anchored strict minimal revision; detailed evidence remains in the supplement."
    )
    document.save(output)


def main() -> None:
    import hashlib

    actual = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    if actual != EXPECTED_ORIGINAL_SHA256:
        raise SystemExit(f"Original checksum mismatch: {actual}")
    media = make_figures()
    build(CLEAN, marked=False, media=media)
    build(MARKED, marked=True, media=media)
    build_response()
    print(CLEAN)
    print(MARKED)
    print(RESPONSE_OUT)


if __name__ == "__main__":
    main()
