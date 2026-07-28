"""Diagnose whether a frozen Stage-1 model contains low-physics-cost paths.

For one fixed map/start/goal condition, this script compares trajectories from
Gaussian initial noise with trajectories obtained by directly optimizing that
noise.  Model parameters are never updated.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bspline_utils import DifferentiableBSpline
from dataLoader_dit import compute_map_yaw_bins, generate_sdf_from_yaw_stability
from dit.Models import PathDiffusionTransformer
from grad_optimizer import cost_on_dense_trajectory


MAP_INFO = {
    "resolution": 0.4,
    "origin": (-20.0, -20.0, -np.pi),
    "size": (100, 100, 36),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze Stage 1 and optimize only its initial input noise z."
    )
    parser.add_argument("--model-dir", default="data/sim")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="data/sim_dataset/val")
    parser.add_argument("--env", default="env000010")
    parser.add_argument("--path-index", type=int, default=40)
    parser.add_argument("--num-restarts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-traj-points", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Optional physical-cost threshold used only to report success rate.",
    )
    parser.add_argument("--output-dir", default="diagnostics/stage1_noise")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_pose(pose: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((pose.shape[0], 4), device=pose.device, dtype=pose.dtype)
    result[:, :2] = torch.clamp(pose[:, :2] / 20.0, -1.0, 1.0)
    result[:, 2] = torch.cos(pose[:, 2])
    result[:, 3] = torch.sin(pose[:, 2])
    return result


def load_scene(
    dataset: str, env_name: str, path_index: int, device: torch.device
) -> dict[str, torch.Tensor | np.ndarray]:
    env_dir = Path(dataset) / env_name
    with (env_dir / "map.p").open("rb") as handle:
        map_data = pickle.load(handle)
    with (env_dir / f"path_{path_index}.p").open("rb") as handle:
        path_data = pickle.load(handle)

    map_tensor = np.asarray(map_data["tensor"], dtype=np.float32)
    trajectory = np.asarray(path_data["path"], dtype=np.float32)
    normals = map_tensor[:, :, 1:4]
    map_input = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0).to(device)
    start_pose = torch.from_numpy(trajectory[0]).unsqueeze(0).to(device)
    goal_pose = torch.from_numpy(trajectory[-1]).unsqueeze(0).to(device)

    # This reproduces the Stage-2 cost-map construction in dataLoader_dit.py.
    yaw_stability = compute_map_yaw_bins(
        map_input[0, 0], map_input[0, 1], map_input[0, 2], yaw_bins=36
    )
    cost_map = generate_sdf_from_yaw_stability(
        yaw_stability, voxel_size_xy=0.1, yaw_weight=1.4
    ).to(device)

    return {
        "map_input": map_input,
        "start_pose": start_pose,
        "goal_pose": goal_pose,
        "start_normalized": normalize_pose(start_pose),
        "goal_normalized": normalize_pose(goal_pose),
        "cost_map": cost_map,
        "ground_truth": trajectory,
        "elevation": map_tensor[:, :, 0],
    }


def load_frozen_model(
    model_dir: str, checkpoint_path: str | None, device: torch.device
) -> tuple[PathDiffusionTransformer, str]:
    model_dir_path = Path(model_dir)
    with (model_dir_path / "model_params.json").open() as handle:
        model_params = json.load(handle)

    model = PathDiffusionTransformer(**model_params["model_args"]).to(device)
    checkpoint = checkpoint_path or str(model_dir_path / "stage1_best_model.pth")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def expand_condition(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] != 1:
        raise ValueError(f"Cannot expand condition with shape {tuple(value.shape)}")
    return value.expand(batch_size, *value.shape[1:])


def generate_from_z(
    model: PathDiffusionTransformer,
    z: torch.Tensor,
    map_input: torch.Tensor,
    start_normalized: torch.Tensor,
    goal_normalized: torch.Tensor,
    bspline: DifferentiableBSpline,
) -> torch.Tensor:
    """The differentiable equivalent of vis_dit.py's pmf_onestep sampler."""
    batch_size = z.shape[0]
    map_batch = expand_condition(map_input, batch_size)
    start_batch = expand_condition(start_normalized, batch_size)
    goal_batch = expand_condition(goal_normalized, batch_size)
    t = torch.ones(batch_size, device=z.device, dtype=z.dtype)
    r = torch.zeros_like(t)

    control_points = model(map_batch, z, t, r, start_batch, goal_batch)
    # 与 vis_dit.py 的推理语义一致：网络处理完整26点噪声，生成后才覆盖真实端点。
    control_points = control_points.clone()
    control_points[:, 0, :] = start_batch[:, :2]
    control_points[:, -1, :] = goal_batch[:, :2]
    control_points = torch.clamp(control_points, -1.0, 1.0) * 20.0
    return bspline(control_points)


def per_sample_cost(
    trajectory: torch.Tensor,
    scene: dict[str, torch.Tensor | np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    batch_size = trajectory.shape[0]
    return cost_on_dense_trajectory(
        trajectory,
        expand_condition(scene["start_pose"], batch_size),
        expand_condition(scene["goal_pose"], batch_size),
        scene["cost_map"],
        MAP_INFO,
        device=device,
        return_per_sample=True,
    )


def summarize(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().cpu().float()
    return {
        "min": float(array.min()),
        "median": float(array.median()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def save_figure(
    output_path: Path,
    elevation: np.ndarray,
    ground_truth: np.ndarray,
    initial_trajectory: np.ndarray,
    optimized_trajectory: np.ndarray,
    history: np.ndarray,
    initial_cost: float,
    optimized_cost: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.imshow(
        np.ma.masked_invalid(elevation),
        extent=[-20, 20, -20, 20],
        origin="lower",
        cmap="terrain",
        aspect="equal",
    )
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], "w--", lw=2, label="ground truth")
    ax.plot(
        initial_trajectory[:, 0],
        initial_trajectory[:, 1],
        color="tab:orange",
        lw=2,
        label=f"initial z ({initial_cost:.4g})",
    )
    ax.plot(
        optimized_trajectory[:, 0],
        optimized_trajectory[:, 1],
        color="tab:blue",
        lw=2.5,
        label=f"optimized z ({optimized_cost:.4g})",
    )
    ax.scatter(ground_truth[0, 0], ground_truth[0, 1], c="purple", s=55, zorder=5)
    ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c="red", s=55, zorder=5)
    ax.set_title("Best initial-noise restart")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.legend(fontsize=8)

    best_so_far = np.minimum.accumulate(history, axis=0)
    axes[1].plot(best_so_far, alpha=0.7)
    axes[1].plot(
        np.min(best_so_far, axis=1),
        color="black",
        lw=2.5,
        label="best-so-far",
    )
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel("physical cost")
    axes[1].set_yscale("log")
    axes[1].set_title("Optimize z only; model frozen")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_restarts < 1 or args.steps < 0:
        raise ValueError("--num-restarts must be >= 1 and --steps must be >= 0")

    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir) / f"{args.env}_path{args.path_index}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading scene {args.env}/path_{args.path_index}.p ...")
    scene = load_scene(args.dataset, args.env, args.path_index, device)
    model, checkpoint = load_frozen_model(args.model_dir, args.checkpoint, device)
    print(f"Frozen checkpoint: {checkpoint}")

    bspline = DifferentiableBSpline(
        num_control_points=model.num_control_points,
        num_output_points=args.num_traj_points,
        degree=3,
    ).to(device)
    z = torch.nn.Parameter(
        torch.randn(args.num_restarts, model.num_control_points, 2, device=device)
    )
    optimizer = torch.optim.Adam([z], lr=args.lr)

    with torch.no_grad():
        initial_z = z.detach().clone()
        initial_trajectories = generate_from_z(
            model,
            initial_z,
            scene["map_input"],
            scene["start_normalized"],
            scene["goal_normalized"],
            bspline,
        )
        initial_costs = per_sample_cost(initial_trajectories, scene, device)
        best_costs = initial_costs.clone()
        best_z = initial_z.clone()
        best_trajectories = initial_trajectories.clone()

        gt_xy = torch.from_numpy(scene["ground_truth"][:, :2]).to(device).unsqueeze(0)
        gt_cost = float(per_sample_cost(gt_xy, scene, device)[0])

    history = [initial_costs.detach().cpu().numpy()]
    print(f"Initial costs: {summarize(initial_costs)}")
    print(f"Ground-truth reference cost: {gt_cost:.6g}")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        trajectories = generate_from_z(
            model,
            z,
            scene["map_input"],
            scene["start_normalized"],
            scene["goal_normalized"],
            bspline,
        )
        costs = per_sample_cost(trajectories, scene, device)
        loss = costs.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite noise objective at step {step}: {loss.item()}")
        loss.backward()
        if z.grad is None or not torch.isfinite(z.grad).all():
            raise FloatingPointError(f"Invalid z gradient at step {step}")
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            # Evaluate after the update so that the last optimizer step is included.
            trajectories = generate_from_z(
                model,
                z,
                scene["map_input"],
                scene["start_normalized"],
                scene["goal_normalized"],
                bspline,
            )
            costs = per_sample_cost(trajectories, scene, device)
            improved = costs < best_costs
            best_costs[improved] = costs[improved]
            best_z[improved] = z.detach()[improved]
            best_trajectories[improved] = trajectories[improved]
            history.append(costs.detach().cpu().numpy())

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step {step:4d}/{args.steps}: mean={costs.mean().item():.6g}, "
                f"best={best_costs.min().item():.6g}"
            )

    best_restart = int(torch.argmin(best_costs))
    history_array = np.stack(history, axis=0)
    initial_summary = summarize(initial_costs)
    optimized_summary = summarize(best_costs)
    relative_reduction = (initial_costs - best_costs) / initial_costs.clamp_min(1e-12)

    summary = {
        "checkpoint": os.path.abspath(checkpoint),
        "dataset": os.path.abspath(args.dataset),
        "environment": args.env,
        "path_index": args.path_index,
        "device": str(device),
        "seed": args.seed,
        "num_restarts": args.num_restarts,
        "steps": args.steps,
        "learning_rate": args.lr,
        "sampler": "pmf_onestep (t=1, r=0)",
        "model_parameters_trainable": sum(p.requires_grad for p in model.parameters()),
        "ground_truth_reference_cost": gt_cost,
        "initial_cost": initial_summary,
        "optimized_best_cost": optimized_summary,
        "best_restart": best_restart,
        "best_cost": float(best_costs[best_restart]),
        "mean_relative_reduction": float(relative_reduction.mean()),
        "initial_fraction_below_gt_reference": float(
            (initial_costs <= gt_cost).float().mean()
        ),
        "optimized_fraction_below_gt_reference": float(
            (best_costs <= gt_cost).float().mean()
        ),
        "per_restart": [
            {
                "restart": index,
                "initial_cost": float(initial_costs[index]),
                "best_cost": float(best_costs[index]),
                "relative_reduction": float(relative_reduction[index]),
                "initial_z_norm": float(initial_z[index].norm()),
                "optimized_z_norm": float(best_z[index].norm()),
                "z_displacement_norm": float((best_z[index] - initial_z[index]).norm()),
                "initial_z_energy_per_dim": float(initial_z[index].square().mean()),
                "optimized_z_energy_per_dim": float(best_z[index].square().mean()),
                "initial_z_rms": float(initial_z[index].square().mean().sqrt()),
                "best_z_rms": float(best_z[index].square().mean().sqrt()),
            }
            for index in range(args.num_restarts)
        ],
    }
    if args.success_threshold is not None:
        summary["success_threshold"] = args.success_threshold
        summary["initial_success_rate"] = float(
            (initial_costs <= args.success_threshold).float().mean()
        )
        summary["optimized_success_rate"] = float(
            (best_costs <= args.success_threshold).float().mean()
        )

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output_dir / "trajectories.npz",
        ground_truth=scene["ground_truth"],
        initial=initial_trajectories.detach().cpu().numpy(),
        optimized=best_trajectories.detach().cpu().numpy(),
        initial_z=initial_z.detach().cpu().numpy(),
        optimized_z=best_z.detach().cpu().numpy(),
        cost_history=history_array,
    )
    save_figure(
        output_dir / "diagnostic.png",
        scene["elevation"],
        scene["ground_truth"],
        initial_trajectories[best_restart].detach().cpu().numpy(),
        best_trajectories[best_restart].detach().cpu().numpy(),
        history_array,
        float(initial_costs[best_restart]),
        float(best_costs[best_restart]),
    )

    print("\nResult")
    print(f"  Best random-z cost:    {initial_summary['min']:.6g}")
    print(f"  Best optimized-z cost: {optimized_summary['min']:.6g}")
    print(f"  Mean relative drop:    {float(relative_reduction.mean()):.2%}")
    print(
        "  Below GT reference:    "
        f"{int((initial_costs <= gt_cost).sum())}/{args.num_restarts} random, "
        f"{int((best_costs <= gt_cost).sum())}/{args.num_restarts} optimized"
    )
    print(f"  Artifacts: {output_dir}")
    if args.success_threshold is None:
        print(
            "  No automatic verdict was assigned. Compare optimized cost/trajectory "
            "with the ground-truth reference or rerun with --success-threshold."
        )


if __name__ == "__main__":
    main()
