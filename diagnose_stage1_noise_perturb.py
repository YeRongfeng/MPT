"""Measure the local width around optimized Stage-1 initial noises.

This script reuses saved optimized noises from ``diagnose_stage1_latent.py``.
For each optimized noise z*, it evaluates z* + sigma * epsilon without changing
the frozen Stage-1 model or running noise optimization again.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bspline_utils import DifferentiableBSpline
from diagnose_stage1_latent import (
    generate_from_z,
    load_frozen_model,
    load_scene,
    per_sample_cost,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate z* + sigma * epsilon around saved optimized noises."
    )
    parser.add_argument("--model-dir", default="data/sim")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="data/sim_dataset/val")
    parser.add_argument("--env", default="env000010")
    parser.add_argument("--path-ids", type=int, nargs="+", required=True)
    parser.add_argument("--optimization-seed", type=int, default=0)
    parser.add_argument("--perturb-seed", type=int, default=2026)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--samples-per-center", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-traj-points", type=int, default=100)
    parser.add_argument("--noise-root", default="diagnostics/stage1_noise")
    parser.add_argument("--output-name", default="perturbation_summary")
    return parser.parse_args()


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p90": float(torch.quantile(values, 0.9)),
        "max": float(values.max()),
    }


@torch.inference_mode()
def evaluate_noises(
    model,
    noises: torch.Tensor,
    scene: dict,
    bspline: DifferentiableBSpline,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    costs = []
    for start in range(0, noises.shape[0], batch_size):
        z_batch = noises[start : start + batch_size].to(device)
        trajectories = generate_from_z(
            model,
            z_batch,
            scene["map_input"],
            scene["start_normalized"],
            scene["goal_normalized"],
            bspline,
        )
        costs.append(per_sample_cost(trajectories, scene, device).cpu())
    return torch.cat(costs)


def main() -> None:
    args = parse_args()
    if args.samples_per_center < 1 or args.batch_size < 1:
        raise ValueError("--samples-per-center and --batch-size must be positive")
    if any(sigma <= 0 for sigma in args.sigmas):
        raise ValueError("All perturbation sigmas must be positive")

    seed_everything(args.perturb_seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This experiment is intentionally GPU-only; use --device cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process")

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Loading frozen Stage-1 model on {device} ...")
    model, checkpoint = load_frozen_model(args.model_dir, args.checkpoint, device)
    bspline = DifferentiableBSpline(
        num_control_points=model.num_control_points,
        num_output_points=args.num_traj_points,
        degree=3,
    ).to(device)

    output_dir = Path(args.noise_root) / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    per_condition_rows: list[dict] = []
    per_center_rows: list[dict] = []

    for condition_index, path_id in enumerate(args.path_ids):
        result_dir = (
            Path(args.noise_root)
            / f"{args.env}_path{path_id}_seed{args.optimization_seed}"
        )
        with (result_dir / "summary.json").open() as handle:
            optimization_summary = json.load(handle)
        saved = np.load(result_dir / "trajectories.npz")
        optimized_z = torch.from_numpy(saved["optimized_z"]).float()
        center_cost = torch.tensor(
            [row["best_cost"] for row in optimization_summary["per_restart"]],
            dtype=torch.float32,
        )
        if optimized_z.shape[0] != center_cost.numel():
            raise ValueError(f"Mismatched saved data in {result_dir}")

        scene = load_scene(args.dataset, args.env, path_id, device)
        gt_cost = float(optimization_summary["ground_truth_reference_cost"])
        num_centers = optimized_z.shape[0]
        print(
            f"path_{path_id}: {num_centers} centers, GT reference={gt_cost:.6g}"
        )

        # Use the same epsilon bank for every condition and sigma. This makes
        # sigma comparisons paired rather than confounded by Monte Carlo noise.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.perturb_seed)
        epsilon = torch.randn(
            num_centers,
            args.samples_per_center,
            *optimized_z.shape[1:],
            generator=generator,
        )

        for sigma in args.sigmas:
            perturbed_z = optimized_z[:, None] + float(sigma) * epsilon
            flat_z = perturbed_z.reshape(-1, *optimized_z.shape[1:])
            try:
                flat_cost = evaluate_noises(
                    model, flat_z, scene, bspline, args.batch_size, device
                )
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM with --batch-size {args.batch_size}; rerun with a smaller value"
                ) from error

            costs = flat_cost.reshape(num_centers, args.samples_per_center)
            success = costs <= gt_cost
            center_success_rate = success.float().mean(dim=1)
            cost_multiplier = costs / center_cost[:, None].clamp_min(1e-12)
            robust_80 = center_success_rate >= 0.8

            per_condition_rows.append(
                {
                    "path_id": path_id,
                    "sigma": float(sigma),
                    "num_centers": num_centers,
                    "samples_per_center": args.samples_per_center,
                    "num_trajectories": int(costs.numel()),
                    "success_count": int(success.sum()),
                    "success_rate": float(success.float().mean()),
                    "center_success_rate_mean": float(center_success_rate.mean()),
                    "center_success_rate_median": float(center_success_rate.median()),
                    "centers_with_at_least_80pct_success": int(robust_80.sum()),
                    "centers_with_at_least_80pct_success_rate": float(robust_80.float().mean()),
                    "cost_mean": float(costs.mean()),
                    "cost_median": float(costs.median()),
                    "cost_multiplier_mean": float(cost_multiplier.mean()),
                    "cost_multiplier_median": float(cost_multiplier.median()),
                    "noise_energy_per_dim_mean": float(flat_z.square().mean(dim=(1, 2)).mean()),
                }
            )
            for center_index in range(num_centers):
                per_center_rows.append(
                    {
                        "path_id": path_id,
                        "center": center_index,
                        "sigma": float(sigma),
                        "center_cost": float(center_cost[center_index]),
                        "success_rate": float(center_success_rate[center_index]),
                        "perturbed_cost_mean": float(costs[center_index].mean()),
                        "cost_multiplier_mean": float(cost_multiplier[center_index].mean()),
                    }
                )
            print(
                f"  sigma={sigma:g}: success={success.float().mean().item():.2%}, "
                f"robust centers={int(robust_80.sum())}/{num_centers}, "
                f"cost x{cost_multiplier.mean().item():.3f}"
            )

        del scene
        torch.cuda.empty_cache()

    aggregate_rows = []
    for sigma in args.sigmas:
        rows = [row for row in per_center_rows if row["sigma"] == float(sigma)]
        success_rates = torch.tensor([row["success_rate"] for row in rows])
        multipliers = torch.tensor([row["cost_multiplier_mean"] for row in rows])
        aggregate_rows.append(
            {
                "sigma": float(sigma),
                "num_centers": len(rows),
                "num_trajectories": len(rows) * args.samples_per_center,
                "success_rate": float(success_rates.mean()),
                "center_success_rate": tensor_stats(success_rates),
                "center_cost_multiplier": tensor_stats(multipliers),
                "centers_with_at_least_80pct_success": int((success_rates >= 0.8).sum()),
                "centers_with_at_least_80pct_success_rate": float((success_rates >= 0.8).float().mean()),
            }
        )

    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "environment": args.env,
        "path_ids": args.path_ids,
        "optimization_seed": args.optimization_seed,
        "perturb_seed": args.perturb_seed,
        "sigmas": args.sigmas,
        "samples_per_center": args.samples_per_center,
        "num_conditions": len(args.path_ids),
        "num_centers": len(per_center_rows) // len(args.sigmas),
        "total_perturbed_trajectories": sum(row["num_trajectories"] for row in aggregate_rows),
        "aggregate_by_sigma": aggregate_rows,
        "per_condition": per_condition_rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    for filename, rows in (
        ("per_condition.csv", per_condition_rows),
        ("per_center.csv", per_center_rows),
    ):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    labels = [str(path_id) for path_id in args.path_ids]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for sigma in args.sigmas:
        rows = [
            row for row in per_condition_rows if row["sigma"] == float(sigma)
        ]
        axes[0].plot(
            x,
            [row["success_rate"] for row in rows],
            marker="o",
            label=fr"$\sigma={sigma:g}$",
        )
        axes[1].plot(
            x,
            [row["cost_multiplier_mean"] for row in rows],
            marker="o",
            label=fr"$\sigma={sigma:g}$",
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("path_id")
    axes[0].set_ylabel("perturbed success rate")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xticks(x, labels)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("path_id")
    axes[1].set_ylabel("mean perturbed cost / center cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "perturbation.png", dpi=180)
    plt.close(fig)

    print(f"Saved perturbation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
