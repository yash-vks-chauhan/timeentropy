#!/usr/bin/env python3
"""Draw the pipeline overview figure for the revised manuscript.

The submitted version's overview diagram is not reusable: it lists CNN-GRU and
EfficientNet-lite, both withdrawn in this revision because no recoverable definition exists,
it omits the frozen WavLM reference added for Reviewer 1 comment 10, and it presents 150 ms
as a decision window when the sequential rule uses it only as the mandatory predecessor.
This redraws the same eight stages to match the system actually evaluated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.switch_backend("Agg")

ACCENT = "#1F4D78"
FILL_PLAIN = "#F7F7F7"
FILL_HIGHLIGHT = "#EAF1F8"
EDGE = "#333333"

# (number, title, detail, chips, highlighted)
STAGES = [
    (1, "Speech recordings", "8,131 TORGO head-mounted utterances; 15 speakers", (), False),
    (2, "Speaker-wise split", "Five folds, speaker-exclusive; every speaker tested once", (), False),
    (
        3,
        "Find speech onset and crop early windows",
        "Level-adaptive RMS detector with bounded lookahead; 30-ms pre-roll",
        ("150 ms warm-up", "200 ms", "250 ms", "300 ms fallback"),
        True,
    ),
    (4, "Convert audio windows to log-STFT features", "Each crop becomes a time-frequency input", (), False),
    (
        5,
        "Train and compare short-window classifiers",
        "Same folds and prefixes for every system",
        ("Small CNN", "CNN–BiLSTM", "Transformer", "Frozen WavLM Base+"),
        True,
    ),
    (
        6,
        "Calibrate and estimate confidence",
        "Validation-only temperature scaling; normalized posterior entropy",
        (),
        False,
    ),
]
PATHS = [
    (7, "Reference path:\nfixed-duration decision",
     "Evaluate every fixed prefix\non the same utterances"),
    (8, "Compared path:\nentropy-gated early decision",
     "Accept the earliest stable low-entropy\nprediction, otherwise wait for 300 ms"),
]
OUTPUT_LINE = (
    "Reported output: control/dysarthric label, decision time, forced-fallback rate,\n"
    "accuracy, macro-F1, balanced accuracy, confusion matrix"
)


def box(ax, x, y, width, height, *, highlight=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="square,pad=0",
            linewidth=1.0, edgecolor=EDGE,
            facecolor=FILL_HIGHLIGHT if highlight else FILL_PLAIN,
        )
    )
    ax.add_patch(Rectangle((x, y + height - 0.012), width, 0.012, color=ACCENT, linewidth=0))


def numbered(ax, number, x, y):
    ax.add_patch(plt.Circle((x, y), 0.018, color=ACCENT, zorder=3))
    ax.text(x, y, str(number), color="white", fontsize=8.5, fontweight="bold",
            ha="center", va="center", zorder=4)


def chips(ax, labels, x, y, width):
    if not labels:
        return
    slot = width / len(labels)
    for index, label in enumerate(labels):
        left = x + index * slot + 0.008
        ax.add_patch(
            Rectangle((left, y - 0.019), slot - 0.016, 0.038,
                      facecolor="white", edgecolor=EDGE, linewidth=0.8)
        )
        ax.text(left + (slot - 0.016) / 2, y, label, fontsize=7.1, ha="center", va="center")


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                        linewidth=1.0, color=EDGE, shrinkA=0, shrinkB=0)
    )


def draw(output: Path) -> Path:
    figure, ax = plt.subplots(figsize=(7.2, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top, left, width = 0.975, 0.03, 0.94
    for number, title, detail, labels, highlight in STAGES:
        height = 0.108 if labels else 0.078
        bottom = top - height
        box(ax, left, bottom, width, height, highlight=highlight)
        numbered(ax, number, left + 0.035, top - 0.026)
        ax.text(left + 0.065, top - 0.026, title, fontsize=10.5, fontweight="bold", va="center")
        ax.text(left + 0.065, top - 0.052, detail, fontsize=8.4, va="center")
        chips(ax, labels, left + 0.06, bottom + 0.026, width - 0.09)
        if number < 6:
            arrow(ax, (0.5, bottom), (0.5, bottom - 0.026))
        top = bottom - 0.026

    split = top + 0.026
    half = (width - 0.03) / 2
    height = 0.140
    for index, (number, title, detail) in enumerate(PATHS):
        x = left + index * (half + 0.03)
        bottom = split - 0.026 - height
        box(ax, x, bottom, half, height)
        numbered(ax, number, x + 0.032, split - 0.052)
        ax.text(x + 0.058, split - 0.048, title, fontsize=9.0, fontweight="bold",
                va="top", linespacing=1.35)
        ax.text(x + 0.058, bottom + 0.052, detail, fontsize=8.0, va="top", linespacing=1.35)
        arrow(ax, (0.5, split), (x + half / 2, split - 0.026))
        arrow(ax, (x + half / 2, bottom), (0.5, bottom - 0.026))

    final = bottom - 0.026
    output_height = 0.070
    box(ax, left, final - output_height, width, output_height, highlight=True)
    ax.text(0.5, final - output_height / 2, OUTPUT_LINE, fontsize=8.4,
            ha="center", va="center", linespacing=1.4)
    ax.set_ylim(final - output_height - 0.015, 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/figures/figure_architecture.png",
    )
    print(draw(parser.parse_args().output))


if __name__ == "__main__":
    main()
