#!/usr/bin/env python3
"""Plot the frozen fixed-weight versus MGDA development experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARM_COLORS = {"fixed": "#626A73", "mgda": "#007C83"}
PAIR_COLORS = {"FS": "#C2410C", "FK": "#2563EB", "SK": "#15803D"}
COST_COLORS = {"F": "#C2410C", "S": "#2563EB", "K": "#15803D"}
ARM_LABELS = {"fixed": "Fixed weight", "mgda": "MGDA"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows(rows: list[dict[str, str]], arm: str) -> list[dict[str, str]]:
    return sorted(
        (row for row in rows if row["arm"] == arm),
        key=lambda row: int(row["update"] if "update" in row else row["global_update"]),
    )


def _x(rows: list[dict[str, str]]) -> np.ndarray:
    key = "update" if "update" in rows[0] else "global_update"
    return np.asarray([int(row[key]) for row in rows], dtype=float)


def _y(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _style_axis(axis: plt.Axes, xlabel: str = "update") -> None:
    axis.set_xlabel(xlabel)
    axis.grid(True, alpha=0.24, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_overview(output_dir: Path, geometry: list[dict[str, str]], evaluation: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)

    axis = axes[0, 0]
    for arm in ("fixed", "mgda"):
        rows = _rows(evaluation, arm)
        x = _x(rows)
        color = ARM_COLORS[arm]
        axis.plot(x, _y(rows, "strict_valid_rate"), color=color, linewidth=2.0, label=f"{ARM_LABELS[arm]} strict")
        axis.plot(x, _y(rows, "safe_at_k"), color=color, linewidth=1.5, linestyle="--", label=f"{ARM_LABELS[arm]} Safe@K")
    axis.set_title("Development validation validity")
    axis.set_ylabel("rate")
    axis.set_ylim(bottom=0)
    _style_axis(axis)
    axis.legend(fontsize=8, frameon=False)

    axis = axes[0, 1]
    for arm in ("fixed", "mgda"):
        rows = _rows(geometry, arm)
        axis.plot(_x(rows), _y(rows, "mgda_ratio_median"), color=ARM_COLORS[arm], linewidth=2.0, label=ARM_LABELS[arm])
    axis.set_title("Median MGDA norm ratio")
    axis.set_ylabel("||g_MGDA|| / mean(||g_i||)")
    _style_axis(axis)
    axis.legend(frameon=False)

    axis = axes[0, 2]
    for arm in ("fixed", "mgda"):
        rows = _rows(geometry, arm)
        x = _x(rows)
        color = ARM_COLORS[arm]
        axis.plot(x, _y(rows, "fixed_common_descent_rate"), color=color, linewidth=2.0, label=f"{ARM_LABELS[arm]} fixed common")
        axis.plot(x, _y(rows, "fixed_worsens_any_rate"), color=color, linewidth=1.5, linestyle="--", label=f"{ARM_LABELS[arm]} fixed worsens")
    axis.set_title("Fixed scalarization direction")
    axis.set_ylabel("rate")
    axis.set_ylim(-0.02, 1.05)
    _style_axis(axis)
    axis.legend(fontsize=8, frameon=False)

    axis = axes[1, 0]
    for arm in ("fixed", "mgda"):
        rows = _rows(geometry, arm)
        x = _x(rows)
        for pair in ("FS", "FK", "SK"):
            axis.plot(
                x,
                _y(rows, f"cos_{pair}_mean"),
                color=PAIR_COLORS[pair],
                linewidth=1.9 if arm == "mgda" else 1.2,
                linestyle="-" if arm == "mgda" else "--",
                alpha=0.95 if arm == "mgda" else 0.7,
                label=f"{pair} ({ARM_LABELS[arm]})",
            )
    axis.axhline(0.0, color="#202124", linewidth=0.8)
    axis.set_title("Mean pairwise gradient cosine")
    axis.set_ylabel("cosine")
    _style_axis(axis)
    axis.legend(fontsize=7.5, ncol=2, frameon=False)

    axis = axes[1, 1]
    for arm in ("fixed", "mgda"):
        rows = _rows(geometry, arm)
        x = _x(rows)
        for name in ("F", "S", "K"):
            axis.plot(
                x,
                _y(rows, f"alpha_{name}_median"),
                color=COST_COLORS[name],
                linewidth=1.9 if arm == "mgda" else 1.2,
                linestyle="-" if arm == "mgda" else "--",
                alpha=0.95 if arm == "mgda" else 0.7,
                label=f"alpha_{name} ({ARM_LABELS[arm]})",
            )
    axis.set_title("MGDA simplex coefficients")
    axis.set_ylabel("median alpha")
    axis.set_ylim(-0.02, 1.02)
    _style_axis(axis)
    axis.legend(fontsize=7.5, ncol=2, frameon=False)

    axis = axes[1, 2]
    for arm in ("fixed", "mgda"):
        rows = _rows(evaluation, arm)
        axis.plot(_x(rows), _y(rows, "task_cost"), color=ARM_COLORS[arm], linewidth=2.0, label=ARM_LABELS[arm])
    axis.set_title("Development validation task cost")
    axis.set_ylabel("task cost")
    _style_axis(axis)
    axis.legend(frameon=False)

    figure.suptitle("Fixed-weight vs MGDA direct-cost training", fontsize=17, fontweight="bold")
    _save(figure, output_dir / "comparison_overview.png")


def _plot_geometry_pairs(output_dir: Path, geometry: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True)
    pair_titles = {"FS": "F-S", "FK": "F-kappa", "SK": "S-kappa"}
    for row_index, arm in enumerate(("fixed", "mgda")):
        rows = _rows(geometry, arm)
        x = _x(rows)
        for column_index, pair in enumerate(("FS", "FK", "SK")):
            axis = axes[row_index, column_index]
            axis.plot(x, _y(rows, f"cos_{pair}_mean"), color=PAIR_COLORS[pair], linewidth=2.0)
            axis.fill_between(x, 0.0, _y(rows, f"cos_{pair}_mean"), color=PAIR_COLORS[pair], alpha=0.10)
            axis.axhline(0.0, color="#202124", linewidth=0.9)
            axis.set_title(f"{ARM_LABELS[arm]} arm: {pair_titles[pair]}")
            axis.set_ylabel("mean cosine" if column_index == 0 else "")
            _style_axis(axis)
    figure.suptitle("Pairwise gradient conflict trajectories", fontsize=17, fontweight="bold")
    _save(figure, output_dir / "comparison_geometry_pairs.png")


def _plot_validation(output_dir: Path, evaluation: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    axis = axes[0, 0]
    for arm in ("fixed", "mgda"):
        rows = _rows(evaluation, arm)
        x = _x(rows)
        color = ARM_COLORS[arm]
        for key, label, linestyle in (("strict_valid_rate", "strict", "-"), ("safe_at_1", "Safe@1", ":"), ("safe_at_k", "Safe@K", "--")):
            axis.plot(x, _y(rows, key), color=color, linestyle=linestyle, linewidth=2.0, label=f"{ARM_LABELS[arm]} {label}")
    axis.set_title("Validity metrics")
    axis.set_ylabel("rate")
    axis.set_ylim(bottom=0)
    _style_axis(axis)
    axis.legend(fontsize=8, ncol=2, frameon=False)

    axis = axes[0, 1]
    for arm in ("fixed", "mgda"):
        rows = _rows(evaluation, arm)
        x = _x(rows)
        for name, key in (("F", "forbidden_cost"), ("S", "stability_cost"), ("K", "curvature_cost")):
            axis.plot(
                x,
                _y(rows, key),
                color=COST_COLORS[name],
                linestyle="-" if arm == "mgda" else "--",
                linewidth=2.0,
                label=f"{name} ({ARM_LABELS[arm]})",
            )
    axis.set_title("Validation privileged cost components")
    axis.set_ylabel("cost")
    _style_axis(axis)
    axis.legend(fontsize=8, ncol=2, frameon=False)

    axis = axes[1, 0]
    for arm in ("fixed", "mgda"):
        rows = _rows(evaluation, arm)
        x = _x(rows)
        color = ARM_COLORS[arm]
        axis.plot(x, _y(rows, "stability_regression_rate"), color=color, linewidth=2.0, label=f"{ARM_LABELS[arm]} stability")
        axis.plot(x, _y(rows, "curvature_regression_rate"), color=color, linewidth=1.7, linestyle="--", label=f"{ARM_LABELS[arm]} curvature")
    axis.set_title("Regression rates against Stage 1")
    axis.set_ylabel("regression rate")
    axis.set_ylim(bottom=0)
    _style_axis(axis)
    axis.legend(fontsize=8, ncol=2, frameon=False)

    axis = axes[1, 1]
    for arm in ("fixed", "mgda"):
        rows = _rows(evaluation, arm)
        x = _x(rows)
        color = ARM_COLORS[arm]
        axis.plot(x, _y(rows, "forbidden_ok_rate"), color=color, linewidth=1.8, label=f"{ARM_LABELS[arm]} F ok")
        axis.plot(x, _y(rows, "stability_ok_rate"), color=color, linewidth=1.8, linestyle="--", label=f"{ARM_LABELS[arm]} S ok")
        axis.plot(x, _y(rows, "curvature_ok_rate"), color=color, linewidth=1.8, linestyle=":", label=f"{ARM_LABELS[arm]} K ok")
    axis.set_title("Per-component feasibility rates")
    axis.set_ylabel("rate")
    axis.set_ylim(0, 1.05)
    _style_axis(axis)
    axis.legend(fontsize=8, ncol=2, frameon=False)

    figure.suptitle("Development validation comparison", fontsize=17, fontweight="bold")
    _save(figure, output_dir / "comparison_validation.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("diagnostics/direct_cost_mgda_vs_fixed_sgd_20260807"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = _read_csv(input_dir / "geometry_trace.csv")
    evaluation = _read_csv(input_dir / "evaluation_trace.csv")
    _plot_overview(output_dir, geometry, evaluation)
    _plot_geometry_pairs(output_dir, geometry)
    _plot_validation(output_dir, evaluation)
    print(output_dir / "comparison_overview.png")
    print(output_dir / "comparison_geometry_pairs.png")
    print(output_dir / "comparison_validation.png")


if __name__ == "__main__":
    main()
