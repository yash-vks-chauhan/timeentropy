#!/usr/bin/env python3
"""Render the manuscript figures from the generated report tables.

Every figure reads a report CSV; none recomputes a metric, so a figure cannot disagree
with the table beside it. The categorical palette is fixed in one place and was validated
for colour-vision deficiency separation and surface contrast before use.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette, assigned in this fixed order and never cycled.
MODELS = ("small_cnn", "cnn_bilstm", "transformer", "wavlm")
LABELS = {
    "small_cnn": "Small CNN",
    "cnn_bilstm": "CNN–BiLSTM",
    "transformer": "Transformer",
    "wavlm": "WavLM Base+",
}
COLOURS = {
    "small_cnn": "#0072B2",
    "cnn_bilstm": "#D55E00",
    "transformer": "#009E73",
    "wavlm": "#8B5FA8",
}
MARKERS = {"small_cnn": "o", "cnn_bilstm": "s", "transformer": "^", "wavlm": "D"}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
    }
)


def _recessive(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)


def _pooled(reports: Path, model: str) -> pd.DataFrame:
    return pd.read_csv(reports / f"{model}-primary" / "pooled_metrics.csv").set_index("policy_id")


def figure_accuracy_earliness(reports: Path, out: Path) -> None:
    """Balanced accuracy against observation time — the accuracy-earliness view (R1.11)."""

    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    prefixes = [200, 250, 300]
    endpoints = []
    for model in MODELS:
        pooled = _pooled(reports, model)
        values = [pooled.loc[f"fixed_{p}", "balanced_accuracy"] * 100 for p in prefixes]
        axis.plot(
            prefixes, values, color=COLOURS[model], marker=MARKERS[model], markersize=6,
            linewidth=2, label=LABELS[model], zorder=3,
        )
        endpoints.append([values[-1], model])

    # Direct labels collide when two models finish within a hair of each other, as the
    # small CNN and CNN-BiLSTM do. Push overlapping labels apart along y before drawing.
    endpoints.sort()
    minimum_gap = 0.9
    for lower, upper in pairwise(endpoints):
        if upper[0] - lower[0] < minimum_gap:
            upper[0] = lower[0] + minimum_gap
    for label_y, model in endpoints:
        axis.annotate(
            LABELS[model], (prefixes[-1], label_y), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=8, color=INK,
        )
    axis.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axis.annotate("chance", (200, 50), xytext=(0, 4), textcoords="offset points",
                  fontsize=8, color=MUTED)
    axis.set_xlabel("Post-onset observation time (ms)")
    axis.set_ylabel("Balanced accuracy (%)")
    axis.set_xticks(prefixes)
    axis.set_xlim(190, 372)
    axis.grid(axis="y")
    # Small CNN and CNN-BiLSTM converge at 300 ms, so a direct label alone is ambiguous
    # about which line it belongs to; the legend carries the marker shape that resolves it.
    axis.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.62))
    _recessive(axis)
    figure.tight_layout()
    figure.savefig(out / "figure_accuracy_earliness.png", dpi=300)
    plt.close(figure)


def figure_fold_variance(reports: Path, out: Path) -> None:
    """Per-fold balanced accuracy — fold variance against model variance (R1.7, R2.11)."""

    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    for index, model in enumerate(MODELS):
        folds = pd.read_csv(reports / f"{model}-primary" / "fold_metrics.csv")
        folds = folds[folds["policy_id"] == "fixed_300"].sort_values("fold")
        values = folds["balanced_accuracy"].to_numpy() * 100
        axis.scatter(
            [index] * len(values), values, s=46, color=COLOURS[model],
            edgecolor=SURFACE, linewidth=1.4, zorder=3,
        )
        axis.plot([index - 0.18, index + 0.18], [values.mean()] * 2,
                  color=INK, linewidth=2, zorder=4)
    axis.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axis.annotate("chance", (-0.42, 50), xytext=(0, 4), textcoords="offset points",
                  fontsize=8, color=MUTED)
    axis.set_xticks(range(len(MODELS)))
    axis.set_xticklabels([LABELS[m] for m in MODELS], fontsize=8)
    axis.set_ylabel("Balanced accuracy per fold (%)")
    axis.set_xlim(-0.5, len(MODELS) - 0.5)
    axis.grid(axis="y")
    _recessive(axis)
    figure.tight_layout()
    figure.savefig(out / "figure_fold_variance.png", dpi=300)
    plt.close(figure)


def figure_reliability(reports: Path, out: Path) -> None:
    """Reliability diagrams: the compact model against the pretrained one (R1.13, R2.5)."""

    panels = [("transformer", "raw"), ("transformer", "temperature_scaled"), ("wavlm", "raw")]
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True)
    for axis, (model, state) in zip(axes, panels, strict=True):
        bins = pd.read_csv(reports / f"{model}-primary" / "calibration_reliability_bins.csv")
        bins = bins[
            bins["fold"].isna() & (bins["prefix_ms"] == 300) & (bins["calibration_state"] == state)
        ].dropna(subset=["mean_confidence", "accuracy"])
        axis.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        axis.plot(
            bins["mean_confidence"], bins["accuracy"], color=COLOURS[model],
            marker=MARKERS[model], markersize=5, linewidth=2, zorder=3,
        )
        metrics = pd.read_csv(reports / f"{model}-primary" / "calibration_metrics.csv")
        metrics = metrics[metrics["fold"].isna() & (metrics["prefix_ms"] == 300)]
        ece = float(metrics[metrics["calibration_state"] == state].iloc[0][
            "expected_calibration_error"])
        label = "uncalibrated" if state == "raw" else "temperature-scaled"
        axis.set_title(f"{LABELS[model]}, {label}\nECE {ece:.3f}", fontsize=9)
        axis.set_xlabel("Mean predicted confidence")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True)
        _recessive(axis)
    axes[0].set_ylabel("Observed accuracy")
    figure.tight_layout()
    figure.savefig(out / "figure_reliability.png", dpi=300)
    plt.close(figure)


def figure_confusion(reports: Path, out: Path) -> None:
    """Regenerated confusion matrices, sequential single hue (R1.20, R3.10)."""

    figure, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for axis, model in zip(axes, ("transformer", "wavlm"), strict=True):
        frame = pd.read_csv(reports / f"{model}-primary" / "confusion_matrices.csv")
        frame = frame[frame["policy_id"] == "complete-temperature_scaled"]
        counts = frame.set_index(["actual_class", "predicted_class"])["count"]
        grid = [
            [int(counts.loc[("control", "control")]), int(counts.loc[("control", "dysarthric")])],
            [int(counts.loc[("dysarthric", "control")]),
             int(counts.loc[("dysarthric", "dysarthric")])],
        ]
        total = sum(sum(row) for row in grid)
        shares = [[cell / sum(row) for cell in row] for row in grid]
        axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                axis.text(
                    j, i, f"{grid[i][j]:,}\n{shares[i][j]*100:.1f}%", ha="center", va="center",
                    fontsize=9, color=SURFACE if shares[i][j] > 0.55 else INK,
                )
        correct = (grid[0][0] + grid[1][1]) / total * 100
        axis.set_title(f"{LABELS[model]} — {correct:.2f}% accuracy", fontsize=9)
        axis.set_xticks([0, 1], ["control", "dysarthric"], fontsize=8)
        axis.set_yticks([0, 1], ["control", "dysarthric"], fontsize=8)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(out / "figure_confusion.png", dpi=300)
    plt.close(figure)


def figure_speaker_bias(reports: Path, out: Path) -> None:
    """Predicted-dysarthric rate per speaker — the global bias (R1.18, R3.10)."""

    frame = pd.read_csv(reports / "transformer-primary" / "speaker_metrics.csv")
    frame = frame[frame["policy_id"] == "complete-temperature_scaled"].sort_values(
        ["speaker_class", "predicted_dysarthric_rate"]
    )
    figure, axis = plt.subplots(figsize=(6.2, 3.4))
    positions = range(len(frame))
    colours = [COLOURS["transformer"] if c == "dysarthric" else COLOURS["cnn_bilstm"]
               for c in frame["speaker_class"]]
    axis.barh(list(positions), frame["predicted_dysarthric_rate"] * 100, color=colours, height=0.62)
    axis.axvline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axis.set_yticks(list(positions), frame["speaker_id"], fontsize=8)
    axis.set_xlabel("Utterances predicted dysarthric (%)")
    axis.set_xlim(0, 100)
    handles = [
        plt.Line2D([0], [0], color=COLOURS["cnn_bilstm"], linewidth=6, label="true class: control"),
        plt.Line2D([0], [0], color=COLOURS["transformer"], linewidth=6,
                   label="true class: dysarthric"),
    ]
    axis.legend(handles=handles, fontsize=8, loc="lower right")
    axis.grid(axis="x")
    _recessive(axis)
    figure.tight_layout()
    figure.savefig(out / "figure_speaker_bias.png", dpi=300)
    plt.close(figure)


def figure_robustness(robustness: Path, out: Path) -> None:
    """Accuracy change under each prespecified perturbation (R2.4, R1-EV.8)."""

    frame = pd.read_csv(robustness / "pooled_oof_metrics.csv")
    frame = frame[frame["policy_id"] == "complete-temperature_scaled"]
    order = [
        ("white_noise_snr_p20_db", "noise 20 dB"), ("white_noise_snr_p10_db", "noise 10 dB"),
        ("white_noise_snr_p0_db", "noise 0 dB"), ("onset_shift_m50_ms", "onset −50 ms"),
        ("onset_shift_m30_ms", "onset −30 ms"), ("onset_shift_m10_ms", "onset −10 ms"),
        ("onset_shift_p10_ms", "onset +10 ms"), ("onset_shift_p30_ms", "onset +30 ms"),
        ("onset_shift_p50_ms", "onset +50 ms"),
    ]
    clean = float(frame[frame["robustness_analysis_id"] == "clean_reference"].iloc[0]["accuracy"])
    labels, deltas = [], []
    for analysis_id, label in order:
        row = frame[frame["robustness_analysis_id"] == analysis_id]
        if row.empty:
            continue
        labels.append(label)
        deltas.append((float(row.iloc[0]["accuracy"]) - clean) * 100)

    figure, axis = plt.subplots(figsize=(6.0, 3.4))
    # Diverging encoding: two hues around a neutral zero, never a rainbow.
    colours = [COLOURS["small_cnn"] if d >= 0 else COLOURS["cnn_bilstm"] for d in deltas]
    axis.barh(range(len(labels)), deltas, color=colours, height=0.62)
    axis.axvline(0, color=MUTED, linewidth=1)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Change in pooled accuracy vs clean reference (points)")
    # Leave room for the value labels at both ends: without it the largest negative
    # label runs into the y-axis tick text and the largest positive one is clipped.
    span = max(abs(min(deltas)), abs(max(deltas)))
    axis.set_xlim(min(deltas) - span * 0.28, max(deltas) + span * 0.28)
    for index, delta in enumerate(deltas):
        axis.annotate(
            # Unicode minus, to match the axis tick labels rather than an ASCII hyphen.
            f"{delta:+.1f}".replace("-", "\u2212"), (delta, index),
            xytext=(4 if delta >= 0 else -4, 0),
            textcoords="offset points", va="center",
            ha="left" if delta >= 0 else "right", fontsize=8, color=INK,
        )
    axis.grid(axis="x")
    _recessive(axis)
    figure.tight_layout()
    figure.savefig(out / "figure_robustness.png", dpi=300)
    plt.close(figure)


def figure_onset(diagnostic: Path, out: Path) -> None:
    """Onset-detector failure rates by mode (R1.4, R2.4)."""

    import json

    summary = json.loads((diagnostic / "onset_mode_summary.json").read_text())
    modes = [
        ("causal_noise_floor", "Level-adaptive\n(revised primary)"),
        ("causal_fixed", "Fixed threshold"),
        ("recovered_offline_peak", "Peak-relative\n(earlier version)"),
    ]
    zero = [summary[m]["fires_at_zero_rate"] * 100 for m, _ in modes]
    quiet = [summary[m]["quiet_prefix_rate"] * 100 for m, _ in modes]
    positions = range(len(modes))
    figure, axis = plt.subplots(figsize=(5.8, 3.4))
    axis.bar([p - 0.19 for p in positions], zero, width=0.36,
             color=COLOURS["cnn_bilstm"], label="onset returned at sample 0")
    axis.bar([p + 0.19 for p in positions], quiet, width=0.36,
             color=COLOURS["small_cnn"], label="prefix contains room tone")
    for p, (z, q) in enumerate(zip(zero, quiet, strict=True)):
        axis.annotate(f"{z:.1f}%", (p - 0.19, z), xytext=(0, 3), textcoords="offset points",
                      ha="center", fontsize=8, color=INK)
        axis.annotate(f"{q:.1f}%", (p + 0.19, q), xytext=(0, 3), textcoords="offset points",
                      ha="center", fontsize=8, color=INK)
    axis.set_xticks(list(positions), [label for _, label in modes], fontsize=8)
    axis.set_ylabel("Utterances affected (%)")
    axis.set_ylim(0, 100)
    axis.legend(fontsize=8)
    axis.grid(axis="y")
    _recessive(axis)
    figure.tight_layout()
    figure.savefig(out / "figure_onset_detection.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("artifacts/reports"))
    parser.add_argument(
        "--robustness", type=Path, default=Path("artifacts/robustness/test-time/combined")
    )
    parser.add_argument("--onset", type=Path, default=Path("artifacts/onset_diagnosis"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    figure_onset(arguments.onset, arguments.output_dir)
    figure_accuracy_earliness(arguments.reports, arguments.output_dir)
    figure_reliability(arguments.reports, arguments.output_dir)
    figure_fold_variance(arguments.reports, arguments.output_dir)
    figure_confusion(arguments.reports, arguments.output_dir)
    figure_speaker_bias(arguments.reports, arguments.output_dir)
    figure_robustness(arguments.robustness, arguments.output_dir)

    for path in sorted(arguments.output_dir.glob("*.png")):
        print(f"  {path}  ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
