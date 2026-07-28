"""Aggregate several Stage-1 initial-noise optimization diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="diagnostics/stage1_noise")
    parser.add_argument("--env", default="env000010")
    parser.add_argument("--path-ids", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-name", default="sweep_summary")
    return parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = []
    all_random = []
    all_optimized = []
    all_gt = []
    all_initial_energy = []
    all_optimized_energy = []
    all_displacement = []

    for path_id in args.path_ids:
        result_dir = root / f"{args.env}_path{path_id}_seed{args.seed}"
        with (result_dir / "summary.json").open() as handle:
            summary = json.load(handle)

        samples = summary["per_restart"]
        random_cost = np.asarray([x["initial_cost"] for x in samples])
        optimized_cost = np.asarray([x["best_cost"] for x in samples])
        initial_energy = np.asarray([x["initial_z_energy_per_dim"] for x in samples])
        optimized_energy = np.asarray([x["optimized_z_energy_per_dim"] for x in samples])
        displacement = np.asarray([x["z_displacement_norm"] for x in samples])
        gt_cost = float(summary["ground_truth_reference_cost"])

        condition_rows.append(
            {
                "path_id": path_id,
                "num_restarts": len(samples),
                "gt_reference_cost": gt_cost,
                "random_cost_mean": float(random_cost.mean()),
                "optimized_cost_mean": float(optimized_cost.mean()),
                "random_to_gt_ratio_mean": float(np.mean(random_cost / gt_cost)),
                "optimized_to_gt_ratio_mean": float(np.mean(optimized_cost / gt_cost)),
                "mean_relative_cost_drop": float(np.mean((random_cost - optimized_cost) / random_cost)),
                "random_success_count": int(np.sum(random_cost <= gt_cost)),
                "random_success_rate": float(np.mean(random_cost <= gt_cost)),
                "optimized_success_count": int(np.sum(optimized_cost <= gt_cost)),
                "optimized_success_rate": float(np.mean(optimized_cost <= gt_cost)),
                "initial_energy_per_dim_mean": float(initial_energy.mean()),
                "optimized_energy_per_dim_mean": float(optimized_energy.mean()),
                "noise_displacement_norm_mean": float(displacement.mean()),
            }
        )
        all_random.append(random_cost)
        all_optimized.append(optimized_cost)
        all_gt.append(np.full_like(random_cost, gt_cost))
        all_initial_energy.append(initial_energy)
        all_optimized_energy.append(optimized_energy)
        all_displacement.append(displacement)

    random_cost = np.concatenate(all_random)
    optimized_cost = np.concatenate(all_optimized)
    gt_cost = np.concatenate(all_gt)
    initial_energy = np.concatenate(all_initial_energy)
    optimized_energy = np.concatenate(all_optimized_energy)
    displacement = np.concatenate(all_displacement)

    aggregate = {
        "environment": args.env,
        "path_ids": args.path_ids,
        "num_conditions": len(condition_rows),
        "restarts_per_condition": sorted(set(row["num_restarts"] for row in condition_rows)),
        "num_condition_noise_evaluations": int(random_cost.size),
        "common_random_seed_across_conditions": args.seed,
        "random_success_count": int(np.sum(random_cost <= gt_cost)),
        "random_success_rate": float(np.mean(random_cost <= gt_cost)),
        "optimized_success_count": int(np.sum(optimized_cost <= gt_cost)),
        "optimized_success_rate": float(np.mean(optimized_cost <= gt_cost)),
        "mean_relative_cost_drop": float(np.mean((random_cost - optimized_cost) / random_cost)),
        "random_to_gt_ratio": stats(random_cost / gt_cost),
        "optimized_to_gt_ratio": stats(optimized_cost / gt_cost),
        "initial_noise_energy_per_dim": stats(initial_energy),
        "optimized_noise_energy_per_dim": stats(optimized_energy),
        "noise_displacement_norm": stats(displacement),
        "per_condition": condition_rows,
    }

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)

    with (output_dir / "per_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=condition_rows[0].keys())
        writer.writeheader()
        writer.writerows(condition_rows)

    labels = [str(row["path_id"]) for row in condition_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x - width / 2, [row["random_success_rate"] for row in condition_rows], width, label="random z")
    axes[0].bar(x + width / 2, [row["optimized_success_rate"] for row in condition_rows], width, label="optimized z")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("path_id")
    axes[0].set_ylabel("fraction below GT reference cost")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, [row["initial_energy_per_dim_mean"] for row in condition_rows], width, label="initial z")
    axes[1].bar(x + width / 2, [row["optimized_energy_per_dim_mean"] for row in condition_rows], width, label="optimized z")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="N(0,I) expectation")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("path_id")
    axes[1].set_ylabel(r"mean $||z||^2/48$")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "sweep.png", dpi=180)
    plt.close(fig)

    print(json.dumps({key: value for key, value in aggregate.items() if key != "per_condition"}, indent=2))
    print(f"Saved aggregate artifacts to {output_dir}")


if __name__ == "__main__":
    main()
