#!/usr/bin/env python3
"""Build validation-surface and accuracy-earliness artifacts from frozen outputs.

This presents the already frozen validation candidate grid and out-of-fold test summaries.
It does not train, refit, calibrate, or select a policy, and it writes only to `artifacts/`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "artifacts/evaluation/transformer-primary"
REPORTS = ROOT / "artifacts/reports/transformer-primary"
OUTPUT = ROOT / "artifacts/sensitivity/transformer_complete_calibrated"

CANDIDATES = EVALUATION / "policy_candidate_grid.csv"
SELECTIONS = EVALUATION / "policy_selections.csv"
FIXED = EVALUATION / "fixed_prefix_metrics.csv"
POOLED = REPORTS / "pooled_metrics.csv"

THETA = tuple(round(0.30 + 0.05 * index, 2) for index in range(14))
EPSILON = tuple(round(0.02 * index, 2) for index in range(11))
FOLDS = tuple(range(5))
MARGIN = 0.01
HORIZON_MS = 300.0

INK = "#202124"
MUTED = "#666666"
GRID = "#d8d8d8"
SURFACE = "#ffffff"
BLUE = "#2166ac"
ORANGE = "#d95f02"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks required columns: {missing}")


def close(left: Any, right: Any, *, atol: float = 1.0e-12) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))


def harmonic_mean_with_timeliness(score: float, consumed_fraction: float) -> float:
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1]")
    if not 0.0 <= consumed_fraction <= 1.0:
        raise ValueError("consumed fraction must be in [0, 1]")
    timeliness = 1.0 - consumed_fraction
    denominator = score + timeliness
    return 0.0 if denominator == 0.0 else 2.0 * score * timeliness / denominator


def load_surface() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(CANDIDATES)
    selections = pd.read_csv(SELECTIONS)
    require_columns(
        candidates,
        {
            "run_id",
            "fold",
            "policy_id",
            "policy_variant",
            "calibration_state",
            "selected",
            "theta",
            "epsilon",
            "macro_f1",
            "accuracy",
            "mean_latency_ms",
            "forced_fallback_rate",
            "feasible",
        },
        "candidate grid",
    )
    require_columns(
        selections,
        {
            "run_id",
            "fold",
            "policy_variant",
            "calibration_state",
            "macro_f1_margin",
            "fixed_300_policy_macro_f1",
            "theta",
            "epsilon",
            "macro_f1",
            "accuracy",
            "mean_latency_ms",
            "forced_fallback_rate",
        },
        "policy selections",
    )

    query = (
        candidates["policy_variant"].eq("complete")
        & candidates["calibration_state"].eq("temperature_scaled")
    )
    surface = candidates.loc[query].copy()
    selection = selections.loc[
        selections["policy_variant"].eq("complete")
        & selections["calibration_state"].eq("temperature_scaled")
    ].copy()

    expected_cells = len(FOLDS) * len(THETA) * len(EPSILON)
    if len(surface) != expected_cells:
        raise ValueError(f"Expected {expected_cells} surface cells, found {len(surface)}")
    if tuple(sorted(surface["fold"].astype(int).unique())) != FOLDS:
        raise ValueError("Surface does not contain exactly folds 0--4")
    if len(selection) != len(FOLDS) or selection["fold"].nunique() != len(FOLDS):
        raise ValueError("Expected exactly one selection row for each of five folds")

    expected_pairs = {(theta, epsilon) for theta in THETA for epsilon in EPSILON}
    for fold, group in surface.groupby("fold", sort=True):
        observed = {
            (round(float(row.theta), 2), round(float(row.epsilon), 2))
            for row in group.itertuples(index=False)
        }
        if observed != expected_pairs or group.duplicated(["theta", "epsilon"]).any():
            raise ValueError(f"Fold {fold} does not contain the exact 14 x 11 grid")
        selected = group.loc[group["selected"].astype(bool)]
        if len(selected) != 1:
            raise ValueError(f"Fold {fold} has {len(selected)} selected surface cells")
        canonical = selection.loc[selection["fold"].eq(fold)]
        if len(canonical) != 1:
            raise ValueError(f"Fold {fold} lacks one canonical selection")
        left, right = selected.iloc[0], canonical.iloc[0]
        for column in (
            "theta",
            "epsilon",
            "macro_f1",
            "accuracy",
            "mean_latency_ms",
            "forced_fallback_rate",
        ):
            if not close(left[column], right[column]):
                raise ValueError(f"Fold {fold} selected cell disagrees on {column}")
        if not close(right["macro_f1_margin"], MARGIN):
            raise ValueError("The recorded macro-F1 margin differs from the frozen 0.01")

    baseline = selection[
        ["fold", "fixed_300_policy_macro_f1", "macro_f1_margin"]
    ].copy()
    baseline["macro_f1_constraint_floor"] = (
        baseline["fixed_300_policy_macro_f1"] - baseline["macro_f1_margin"]
    )
    surface = surface.merge(baseline, on="fold", how="left", validate="many_to_one")
    surface["macro_f1_delta_from_fixed_300"] = (
        surface["macro_f1"] - surface["fixed_300_policy_macro_f1"]
    )
    derived_feasible = (
        surface["macro_f1"] + 1.0e-12 >= surface["macro_f1_constraint_floor"]
    )
    if not derived_feasible.eq(surface["feasible"].astype(bool)).all():
        raise ValueError("Stored feasibility flags disagree with the frozen selection rule")

    exact = surface[
        [
            "run_id",
            "fold",
            "theta",
            "epsilon",
            "macro_f1",
            "accuracy",
            "mean_latency_ms",
            "forced_fallback_rate",
            "fixed_300_policy_macro_f1",
            "macro_f1_margin",
            "macro_f1_constraint_floor",
            "macro_f1_delta_from_fixed_300",
            "feasible",
            "selected",
        ]
    ].rename(
        columns={
            "macro_f1": "validation_macro_f1",
            "accuracy": "validation_accuracy",
            "mean_latency_ms": "validation_mean_observation_ms",
            "forced_fallback_rate": "validation_forced_fallback_rate",
            "feasible": "within_macro_f1_constraint",
            "selected": "selected_for_fold_test_evaluation",
        }
    )
    exact = exact.sort_values(["fold", "theta", "epsilon"], ignore_index=True)

    summaries: list[dict[str, Any]] = []
    for fold, group in exact.groupby("fold", sort=True):
        chosen = group.loc[group["selected_for_fold_test_evaluation"].astype(bool)].iloc[0]
        summaries.append(
            {
                "fold": int(fold),
                "grid_cells": len(group),
                "all_cells_within_macro_f1_constraint": bool(
                    group["within_macro_f1_constraint"].astype(bool).all()
                ),
                "fixed_300_validation_macro_f1": float(
                    chosen["fixed_300_policy_macro_f1"]
                ),
                "surface_macro_f1_min": float(group["validation_macro_f1"].min()),
                "surface_macro_f1_max": float(group["validation_macro_f1"].max()),
                "surface_macro_f1_distinct_values": int(
                    group["validation_macro_f1"].nunique()
                ),
                "surface_mean_observation_ms_min": float(
                    group["validation_mean_observation_ms"].min()
                ),
                "surface_mean_observation_ms_max": float(
                    group["validation_mean_observation_ms"].max()
                ),
                "selected_theta": float(chosen["theta"]),
                "selected_epsilon": float(chosen["epsilon"]),
                "selected_validation_macro_f1": float(chosen["validation_macro_f1"]),
                "selected_validation_accuracy": float(chosen["validation_accuracy"]),
                "selected_validation_mean_observation_ms": float(
                    chosen["validation_mean_observation_ms"]
                ),
                "selected_validation_forced_fallback_rate": float(
                    chosen["validation_forced_fallback_rate"]
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    return exact, summary


def build_accuracy_earliness() -> pd.DataFrame:
    fixed = pd.read_csv(FIXED)
    pooled = pd.read_csv(POOLED)
    require_columns(
        fixed,
        {"prefix_ms", "samples", "accuracy", "macro_f1", "balanced_accuracy"},
        "fixed-prefix metrics",
    )
    require_columns(
        pooled,
        {
            "policy_id",
            "samples",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "mean_latency_ms",
            "latency_sd_ms",
            "median_latency_ms",
            "p95_latency_ms",
            "forced_fallback_rate",
            "consumed_fraction",
            "timeliness",
            "accuracy_earliness_harmonic_mean",
            "macro_f1_earliness_harmonic_mean",
        },
        "pooled report",
    )
    expected_prefixes = (150, 200, 250, 300)
    if tuple(fixed.sort_values("prefix_ms")["prefix_ms"].astype(int)) != expected_prefixes:
        raise ValueError("Fixed metrics must contain exactly 150, 200, 250, and 300 ms")

    rows: list[dict[str, Any]] = []
    for record in fixed.sort_values("prefix_ms").itertuples(index=False):
        consumed = float(record.prefix_ms) / HORIZON_MS
        rows.append(
            {
                "policy_id": f"fixed_{int(record.prefix_ms)}",
                "policy_type": "fixed_duration",
                "curve_role": "fixed_reference_curve",
                "samples": int(record.samples),
                "mean_observation_ms": float(record.prefix_ms),
                "observation_sd_ms": 0.0,
                "median_observation_ms": float(record.prefix_ms),
                "p95_observation_ms": float(record.prefix_ms),
                "maximum_observation_ms": HORIZON_MS,
                "consumed_fraction_of_300ms": consumed,
                "timeliness_one_minus_consumed_fraction": 1.0 - consumed,
                "accuracy": float(record.accuracy),
                "macro_f1": float(record.macro_f1),
                "balanced_accuracy": float(record.balanced_accuracy),
                "accuracy_timeliness_harmonic_mean": harmonic_mean_with_timeliness(
                    float(record.accuracy), consumed
                ),
                "macro_f1_timeliness_harmonic_mean": harmonic_mean_with_timeliness(
                    float(record.macro_f1), consumed
                ),
                "forced_fallback_rate": 0.0,
                "theta": np.nan,
                "epsilon": np.nan,
                "theta_epsilon_scope": "not_applicable_fixed_duration",
                "single_global_theta_epsilon": False,
            }
        )

    adaptive = pooled.loc[pooled["policy_id"].eq("complete-temperature_scaled")]
    if len(adaptive) != 1:
        raise ValueError("Expected one pooled complete-temperature_scaled report row")
    adaptive = adaptive.iloc[0]
    consumed = float(adaptive["mean_latency_ms"]) / HORIZON_MS
    if not close(consumed, adaptive["consumed_fraction"]):
        raise ValueError("Adaptive consumed fraction disagrees with mean observation time")
    if not close(1.0 - consumed, adaptive["timeliness"]):
        raise ValueError("Adaptive timeliness is internally inconsistent")
    if not close(
        harmonic_mean_with_timeliness(float(adaptive["accuracy"]), consumed),
        adaptive["accuracy_earliness_harmonic_mean"],
    ):
        raise ValueError("Adaptive accuracy-timeliness harmonic mean is inconsistent")
    if not close(
        harmonic_mean_with_timeliness(float(adaptive["macro_f1"]), consumed),
        adaptive["macro_f1_earliness_harmonic_mean"],
    ):
        raise ValueError("Adaptive macro-F1-timeliness harmonic mean is inconsistent")

    rows.append(
        {
            "policy_id": "complete-temperature_scaled",
            "policy_type": "adaptive_complete",
            "curve_role": "pooled_oof_point_with_fold_specific_validation_selected_parameters",
            "samples": int(adaptive["samples"]),
            "mean_observation_ms": float(adaptive["mean_latency_ms"]),
            "observation_sd_ms": float(adaptive["latency_sd_ms"]),
            "median_observation_ms": float(adaptive["median_latency_ms"]),
            "p95_observation_ms": float(adaptive["p95_latency_ms"]),
            "maximum_observation_ms": HORIZON_MS,
            "consumed_fraction_of_300ms": consumed,
            "timeliness_one_minus_consumed_fraction": 1.0 - consumed,
            "accuracy": float(adaptive["accuracy"]),
            "macro_f1": float(adaptive["macro_f1"]),
            "balanced_accuracy": float(adaptive["balanced_accuracy"]),
            "accuracy_timeliness_harmonic_mean": float(
                adaptive["accuracy_earliness_harmonic_mean"]
            ),
            "macro_f1_timeliness_harmonic_mean": float(
                adaptive["macro_f1_earliness_harmonic_mean"]
            ),
            "forced_fallback_rate": float(adaptive["forced_fallback_rate"]),
            "theta": np.nan,
            "epsilon": np.nan,
            "theta_epsilon_scope": (
                "fold_specific_validation_selected; see theta_epsilon_fold_summary.csv"
            ),
            "single_global_theta_epsilon": False,
        }
    )
    result = pd.DataFrame(rows)
    if result["samples"].nunique() != 1:
        raise ValueError("Fixed and adaptive test estimands do not contain the same sample count")
    return result


def set_plot_style() -> None:
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
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "savefig.facecolor": SURFACE,
        }
    )


def figure_theta_epsilon(exact: pd.DataFrame, summary: pd.DataFrame) -> None:
    set_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.55), sharex=True, sharey=True)
    flat_axes = list(axes.flat)
    delta_limit = max(abs(float(exact["macro_f1_delta_from_fixed_300"].min())), abs(float(exact["macro_f1_delta_from_fixed_300"].max())))
    delta_limit = math.ceil(delta_limit * 1000.0) / 1000.0
    normalization = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)
    image = None

    for fold, axis in zip(FOLDS, flat_axes[:5], strict=True):
        fold_data = exact.loc[exact["fold"].eq(fold)]
        delta = fold_data.pivot(index="epsilon", columns="theta", values="macro_f1_delta_from_fixed_300").reindex(index=EPSILON, columns=THETA)
        latency = fold_data.pivot(index="epsilon", columns="theta", values="validation_mean_observation_ms").reindex(index=EPSILON, columns=THETA)
        image = axis.imshow(
            delta.to_numpy(dtype=float),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu",
            norm=normalization,
            extent=(-0.5, len(THETA) - 0.5, -0.5, len(EPSILON) - 0.5),
        )

        selected = fold_data.loc[
            fold_data["selected_for_fold_test_evaluation"].astype(bool)
        ].iloc[0]
        x_selected = THETA.index(round(float(selected["theta"]), 2))
        y_selected = EPSILON.index(round(float(selected["epsilon"]), 2))
        axis.scatter(
            [x_selected],
            [y_selected],
            marker="*",
            s=145,
            color="#ffcc00",
            edgecolor=INK,
            linewidth=0.9,
            zorder=5,
            clip_on=False,
        )

        minimum = float(latency.to_numpy().min())
        maximum = float(latency.to_numpy().max())
        if maximum - minimum > 1.0e-9:
            levels = [value for value in (225.0, 250.0, 275.0) if minimum < value < maximum]
            if levels:
                contours = axis.contour(
                    np.arange(len(THETA)),
                    np.arange(len(EPSILON)),
                    latency.to_numpy(dtype=float),
                    levels=levels,
                    colors=INK,
                    linewidths=0.75,
                    alpha=0.9,
                )
                axis.clabel(contours, fmt=lambda value: f"{value:.0f} ms", fontsize=7)
        else:
            axis.text(
                0.5,
                0.50,
                "All cells: 300 ms\n(100% fallback)",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color=INK,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.83, "edgecolor": GRID},
            )

        row = summary.loc[summary["fold"].eq(fold)].iloc[0]
        axis.set_title(
            f"Fold {fold}  |  fixed-300 macro-F1 = {row['fixed_300_validation_macro_f1']:.3f}"
        )
        axis.set_xticks(range(0, len(THETA), 2), [f"{THETA[i]:.2f}" for i in range(0, len(THETA), 2)])
        axis.set_yticks(range(0, len(EPSILON), 2), [f"{EPSILON[i]:.2f}" for i in range(0, len(EPSILON), 2)])
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    legend_axis = flat_axes[5]
    legend_axis.axis("off")
    legend_axis.scatter([0.08], [0.76], marker="*", s=160, color="#ffcc00", edgecolor=INK, transform=legend_axis.transAxes)
    legend_axis.text(0.16, 0.76, "validation-selected cell", va="center", transform=legend_axis.transAxes)
    legend_axis.plot([0.03, 0.13], [0.61, 0.61], color=INK, lw=0.8, transform=legend_axis.transAxes)
    legend_axis.text(0.16, 0.61, "mean-observation contour", va="center", transform=legend_axis.transAxes)
    legend_axis.text(
        0.03,
        0.43,
        "Selection rule",
        weight="bold",
        transform=legend_axis.transAxes,
    )
    legend_axis.text(
        0.03,
        0.36,
        "Minimize mean observation time\nsubject to macro-F1 ≥\nfixed-300 macro-F1 − 0.01.",
        va="top",
        linespacing=1.35,
        fontsize=8.5,
        transform=legend_axis.transAxes,
    )
    legend_axis.text(
        0.03,
        0.055,
        "All 154 cells were feasible in every fold.",
        weight="bold",
        fontsize=8.5,
        transform=legend_axis.transAxes,
    )

    for axis in axes[:, 0]:
        axis.set_ylabel("Stability tolerance, ε")
    for axis in axes[1, :2]:
        axis.set_xlabel("Normalized-entropy threshold, θ")

    if image is None:
        raise RuntimeError("No heatmap was drawn")
    color_axis = figure.add_axes([0.89, 0.18, 0.018, 0.62])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("Δ validation macro-F1 from fold fixed-300")
    figure.suptitle(
        "Complete calibrated Transformer: validation-only θ–ε response surface",
        fontsize=12,
        y=0.985,
    )
    figure.subplots_adjust(left=0.075, right=0.85, top=0.91, bottom=0.09, wspace=0.22, hspace=0.28)
    figure.savefig(OUTPUT / "figure_theta_epsilon_validation_surface.png", dpi=320)
    figure.savefig(
        OUTPUT / "figure_theta_epsilon_validation_surface.pdf",
        metadata={"Creator": "timeentropy sensitivity artifact generator", "CreationDate": None},
    )
    plt.close(figure)


def figure_accuracy_earliness(table: pd.DataFrame) -> None:
    set_plot_style()
    fixed = table.loc[table["policy_type"].eq("fixed_duration")].sort_values(
        "mean_observation_ms"
    )
    adaptive = table.loc[table["policy_type"].eq("adaptive_complete")].iloc[0]

    figure, axis = plt.subplots(figsize=(6.35, 4.05))
    axis.plot(
        fixed["consumed_fraction_of_300ms"],
        fixed["accuracy"] * 100.0,
        color=BLUE,
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="Fixed-duration reference curve",
        zorder=3,
    )
    for row in fixed.itertuples(index=False):
        axis.annotate(
            f"{int(row.mean_observation_ms)} ms",
            (row.consumed_fraction_of_300ms, row.accuracy * 100.0),
            xytext=(0, 7 if row.mean_observation_ms != 300 else -13),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=BLUE,
        )
    axis.scatter(
        [adaptive["consumed_fraction_of_300ms"]],
        [adaptive["accuracy"] * 100.0],
        marker="*",
        s=175,
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.8,
        label="Adaptive complete policy (pooled OOF)†",
        zorder=5,
    )
    axis.annotate(
        f"Adaptive\n{adaptive['mean_observation_ms']:.1f} ms",
        (
            adaptive["consumed_fraction_of_300ms"],
            adaptive["accuracy"] * 100.0,
        ),
        xytext=(-8, -27),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=ORANGE,
    )

    axis.set_xlabel("Mean fraction of the 300 ms decision horizon consumed")
    axis.set_ylabel("Pooled out-of-fold accuracy (%)")
    axis.set_xlim(0.47, 1.025)
    values = np.r_[fixed["accuracy"].to_numpy(), float(adaptive["accuracy"])] * 100.0
    axis.set_ylim(float(values.min()) - 0.75, float(values.max()) + 0.9)
    axis.grid(axis="both")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper right", fontsize=8, frameon=False)

    top = axis.secondary_xaxis(
        "top",
        functions=(lambda fraction: fraction * HORIZON_MS, lambda milliseconds: milliseconds / HORIZON_MS),
    )
    top.set_xlabel("Equivalent mean post-onset observation (ms)")
    top.set_xticks([150, 200, 250, 300])

    figure.text(
        0.5,
        0.025,
        "† Pooled OOF predictions combine fold-specific validation-selected pairs; there is no "
        "single pooled θ–ε pair.\nFolds 0 and 2 use (.95, .20); folds 1, 3, and 4 use (.30, .00).",
        fontsize=7.5,
        color=MUTED,
        ha="center",
        linespacing=1.35,
    )
    figure.subplots_adjust(left=0.13, right=0.97, top=0.83, bottom=0.27)
    figure.savefig(OUTPUT / "figure_accuracy_earliness_transformer.png", dpi=320)
    figure.savefig(
        OUTPUT / "figure_accuracy_earliness_transformer.pdf",
        metadata={"Creator": "timeentropy sensitivity artifact generator", "CreationDate": None},
    )
    plt.close(figure)


def write_manifest(outputs: list[Path], exact: pd.DataFrame, summary: pd.DataFrame) -> None:
    manifest = {
        "artifact_family": "transformer_complete_calibrated_sensitivity_and_earliness",
        "derivation": (
            "Presentation-only derivation from frozen validation candidate grid, policy "
            "selections, fixed-prefix metrics, and pooled OOF report; no refitting or test tuning"
        ),
        "policy_variant": "complete",
        "calibration_state": "temperature_scaled",
        "grid": {
            "theta": {"start": 0.30, "stop": 0.95, "step": 0.05, "values": list(THETA)},
            "epsilon": {"start": 0.00, "stop": 0.20, "step": 0.02, "values": list(EPSILON)},
            "cells_per_fold": len(THETA) * len(EPSILON),
            "folds": list(FOLDS),
        },
        "selection": {
            "partition": "policy_tuning validation rows only",
            "primary": "minimum_validation_mean_observation_time",
            "constraint": "macro_f1_at_least_fixed_300_minus_0.01_absolute",
            "margin_absolute": MARGIN,
            "all_cells_feasible_all_folds": bool(
                exact["within_macro_f1_constraint"].astype(bool).all()
            ),
        },
        "accuracy_earliness": {
            "maximum_observation_ms": HORIZON_MS,
            "consumed_fraction": "mean_observation_ms / 300",
            "timeliness": "1 - consumed_fraction",
            "harmonic_mean": "2 * score * timeliness / (score + timeliness)",
            "adaptive_parameter_scope": (
                "fold-specific validation-selected pairs; no single pooled theta-epsilon pair"
            ),
        },
        "validation": {
            "surface_rows": len(exact),
            "fold_summary_rows": len(summary),
            "complete_grid_verified": True,
            "selected_cells_match_policy_selections": True,
            "stored_feasibility_recomputed_and_verified": True,
            "fixed_and_adaptive_oof_sample_counts_match": True,
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (CANDIDATES, SELECTIONS, FIXED, POOLED)
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in outputs
        },
    }
    path = OUTPUT / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    exact, summary = load_surface()
    accuracy_earliness = build_accuracy_earliness()

    exact_path = OUTPUT / "theta_epsilon_surface_exact.csv"
    summary_path = OUTPUT / "theta_epsilon_fold_summary.csv"
    earliness_path = OUTPUT / "accuracy_earliness_exact.csv"
    exact.to_csv(exact_path, index=False, float_format="%.15g")
    summary.to_csv(summary_path, index=False, float_format="%.15g")
    accuracy_earliness.to_csv(earliness_path, index=False, float_format="%.15g")

    figure_theta_epsilon(exact, summary)
    figure_accuracy_earliness(accuracy_earliness)
    outputs = [
        exact_path,
        summary_path,
        earliness_path,
        OUTPUT / "figure_theta_epsilon_validation_surface.png",
        OUTPUT / "figure_theta_epsilon_validation_surface.pdf",
        OUTPUT / "figure_accuracy_earliness_transformer.png",
        OUTPUT / "figure_accuracy_earliness_transformer.pdf",
    ]
    write_manifest(outputs, exact, summary)

    print(f"Wrote {len(exact)} exact validation cells and {len(accuracy_earliness)} earliness rows")
    for path in [*outputs, OUTPUT / "artifact_manifest.json"]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
