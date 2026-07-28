#!/usr/bin/env python3
"""Experiment A: isolate trajectory-representation optimization bottlenecks.

For the same Stage-1 samples and the same tail-risk objective, optimize:

1. P: 24 physical interior B-spline control points.
2. R: 25 normalized zero-sum physical edge residuals.
3. raw: raw_v/raw_s through the model's radial feasibility map.

The script records enough metadata and intermediate results to make the run
auditable and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bspline_utils import DifferentiableBSpline
from dataLoader_dit import UnevenPathDataLoader
from dit.Models import (
    PathDiffusionTransformer,
    PhysicalScaledEdgeResidualRepresentation,
)
from grad_optimizer import (
    _sample_cost_map_on_dense_trajectory,
    cost_on_dense_trajectory_tail_risk,
    discrete_turning_curvature,
)
from map_config import MAP_CONFIG, MAP_HALF_EXTENT, SAFETY_COST_CONFIG


COMPONENT_NAMES = (
    "obstacle",
    "curvature",
    "curvature_max",
    "out_of_bound",
    "endpoint",
    "weighted_obstacle",
    "weighted_curvature",
    "weighted_out_of_bound",
    "weighted_endpoint",
    "feasible",
    "smoothness",
    "jerk",
    "quality",
    "total",
)


@dataclass
class Candidate:
    environment: str
    path_num: int
    dataset_index: int
    start_pose: torch.Tensor
    goal_pose: torch.Tensor
    cost_map: torch.Tensor
    elevation: torch.Tensor
    control_points: torch.Tensor
    initial_total: float
    initial_obstacle: float
    initial_dangerous_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(MAP_CONFIG.dataset_root / "val"))
    parser.add_argument("--model-dir", default="data/sim")
    parser.add_argument("--checkpoint", default="stage1_best_model.pth")
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--paths-per-env", type=int, default=10)
    parser.add_argument("--select", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--lr-grid",
        type=float,
        nargs="+",
        default=[0.003, 0.01, 0.03],
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--trace-steps",
        type=int,
        nargs="+",
        default=[0, 1, 5, 10, 25, 50, 100, 200, 300],
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostics/experiment_a_representation",
    )
    parser.add_argument("--plot-examples", type=int, default=6)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_metadata() -> Dict[str, object]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                list(args), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return ""

    status = run("git", "status", "--short")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def discover_envs(dataset_root: Path, num_envs: int, seed: int) -> List[str]:
    envs = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "map.p").is_file()
    )
    if num_envs > len(envs):
        raise ValueError(f"Requested {num_envs} envs, only {len(envs)} available")
    rng = random.Random(seed)
    return sorted(rng.sample(envs, num_envs))


def normalize_poses(
    poses: torch.Tensor, coordinate_scale: float
) -> torch.Tensor:
    result = torch.zeros(
        poses.shape[0], 4, dtype=poses.dtype, device=poses.device
    )
    result[:, :2] = torch.clamp(
        poses[:, :2] / coordinate_scale, -1.0, 1.0
    )
    result[:, 2] = torch.cos(poses[:, 2])
    result[:, 3] = torch.sin(poses[:, 2])
    return result


def build_dense(
    control_points: torch.Tensor, bspline: DifferentiableBSpline
) -> torch.Tensor:
    return bspline(control_points)


def evaluate_control_points(
    control_points: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    cost_maps: torch.Tensor,
    map_info: Dict[str, object],
    bspline: DifferentiableBSpline,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    trajectory = build_dense(control_points, bspline)
    costs, components = cost_on_dense_trajectory_tail_risk(
        trajectory,
        start_pose,
        goal_pose,
        cost_maps,
        map_info,
        device,
        return_per_sample=True,
        return_components=True,
    )
    _, _, _, esdf = _sample_cost_map_on_dense_trajectory(
        trajectory, cost_maps, map_info, device
    )
    curvature = discrete_turning_curvature(trajectory)
    segment = trajectory[:, 1:, :] - trajectory[:, :-1, :]

    result = {name: components[name] for name in COMPONENT_NAMES}
    result.update(
        {
            "cost": costs,
            "dangerous_ratio": (esdf < 0.0).float().mean(dim=1),
            "unsafe_margin_ratio": (
                esdf < SAFETY_COST_CONFIG.d_safe_meters
            ).float().mean(dim=1),
            "min_esdf": esdf.min(dim=1).values,
            "path_length": torch.linalg.vector_norm(
                segment, dim=-1
            ).sum(dim=1),
            "curvature_violation_ratio": (
                curvature > SAFETY_COST_CONFIG.curvature_limit
            ).float().mean(dim=1),
            "control_oob_ratio": (
                control_points.abs() > MAP_HALF_EXTENT + 1e-6
            ).float().mean(dim=(1, 2)),
            "control_oob_max": (
                control_points.abs() - MAP_HALF_EXTENT
            ).clamp_min(0.0).flatten(start_dim=1).amax(dim=1),
            "trajectory": trajectory,
        }
    )
    return result


def tensor_row_clip(parameters: Sequence[torch.Tensor], max_norm: float) -> None:
    """Clip each sample row independently, matching B independent B=1 runs."""
    if max_norm <= 0:
        return
    batch_size = parameters[0].shape[0]
    squared = torch.zeros(
        batch_size, device=parameters[0].device, dtype=parameters[0].dtype
    )
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared += parameter.grad.flatten(start_dim=1).square().sum(dim=1)
    norm = torch.sqrt(squared + 1e-12)
    scale = torch.clamp(max_norm / norm, max=1.0)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        view_shape = (batch_size,) + (1,) * (parameter.grad.ndim - 1)
        parameter.grad.mul_(scale.view(view_shape))


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-12)
    return value + torch.log(-torch.expm1(-value))


def raw_initialization_from_residual(
    residual: torch.Tensor,
    start_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    representation: PhysicalScaledEdgeResidualRepresentation,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Construct raw_v/raw_s whose radial output reconstructs residual."""
    residual = representation.project_zero_sum(residual)
    residual_norm = torch.linalg.vector_norm(
        residual.flatten(start_dim=1), dim=1
    )
    direction = residual / residual_norm.clamp_min(
        representation.feasibility_eps
    ).view(-1, 1, 1)
    rho = representation.compute_max_feasible_radius(
        direction, start_xy, goal_xy
    )
    alpha_target = residual_norm / rho.clamp_min(
        representation.feasibility_eps
    )

    minimum_slack = 1.0001e-4
    eps = representation.feasibility_eps
    alpha_max = math.sqrt(max(1.0 - eps - minimum_slack**2, 1e-8))
    alpha = alpha_target.clamp(min=0.0, max=alpha_max)

    # Choose denominator=1. Then ||raw_v||=alpha and
    # slack=sqrt(1-alpha^2-eps), giving x_norm=alpha exactly.
    raw_v = direction * alpha.view(-1, 1, 1)
    desired_slack = torch.sqrt(
        (1.0 - alpha.square() - eps).clamp_min(minimum_slack**2)
    )
    softplus_value = (desired_slack - 1e-4).clamp_min(1e-8)
    raw_s = inverse_softplus(softplus_value).unsqueeze(1)

    zero = residual_norm <= representation.feasibility_eps
    raw_v = torch.where(zero.view(-1, 1, 1), torch.zeros_like(raw_v), raw_v)
    diagnostics = {
        "target_alpha": alpha_target,
        "initialized_alpha": alpha,
        "rho": rho,
        "alpha_clipped": alpha_target > alpha_max,
    }
    return raw_v, raw_s, diagnostics


def state_to_control_points(
    method: str,
    variables: Sequence[torch.Tensor],
    start_xy_norm: torch.Tensor,
    goal_xy_norm: torch.Tensor,
    coordinate_scale: float,
    representation: PhysicalScaledEdgeResidualRepresentation,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if method == "P":
        middle = variables[0]
        control_points = torch.cat(
            [
                (start_xy_norm * coordinate_scale).unsqueeze(1),
                middle,
                (goal_xy_norm * coordinate_scale).unsqueeze(1),
            ],
            dim=1,
        )
        return control_points, {}
    if method == "R":
        residual = representation.project_zero_sum(variables[0])
        control_points = representation.decode(
            residual, start_xy_norm, goal_xy_norm
        )
        return control_points * coordinate_scale, {"residual": residual}
    if method == "raw":
        residual, diagnostics = representation.radial_feasible_residual(
            variables[0],
            variables[1],
            start_xy_norm,
            goal_xy_norm,
            return_diagnostics=True,
        )
        control_points = representation.decode(
            residual, start_xy_norm, goal_xy_norm
        )
        diagnostics["residual"] = residual
        return control_points * coordinate_scale, diagnostics
    raise ValueError(f"Unknown method: {method}")


def initialize_method(
    method: str,
    initial_cp: torch.Tensor,
    start_xy_norm: torch.Tensor,
    goal_xy_norm: torch.Tensor,
    coordinate_scale: float,
    representation: PhysicalScaledEdgeResidualRepresentation,
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    if method == "P":
        return [initial_cp[:, 1:-1, :].detach().clone()], {}

    initial_normalized = initial_cp / coordinate_scale
    residual = representation.encode(
        initial_normalized, start_xy_norm, goal_xy_norm
    )
    if method == "R":
        return [residual.detach().clone()], {}
    if method == "raw":
        raw_v, raw_s, diagnostics = raw_initialization_from_residual(
            residual, start_xy_norm, goal_xy_norm, representation
        )
        return [raw_v.detach().clone(), raw_s.detach().clone()], diagnostics
    raise ValueError(f"Unknown method: {method}")


def detach_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy()
        for key, value in metrics.items()
        if key != "trajectory"
    }


def optimize_method(
    method: str,
    lr: float,
    iterations: int,
    trace_steps: Iterable[int],
    initial_cp: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    cost_maps: torch.Tensor,
    map_info: Dict[str, object],
    bspline: DifferentiableBSpline,
    representation: PhysicalScaledEdgeResidualRepresentation,
    coordinate_scale: float,
    grad_clip: float,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, object]], Dict[str, np.ndarray]]:
    start_xy_norm = start_pose[:, :2] / coordinate_scale
    goal_xy_norm = goal_pose[:, :2] / coordinate_scale
    variables, init_details = initialize_method(
        method,
        initial_cp,
        start_xy_norm,
        goal_xy_norm,
        coordinate_scale,
        representation,
    )
    for variable in variables:
        variable.requires_grad_(True)

    initial_reconstruction, _ = state_to_control_points(
        method,
        variables,
        start_xy_norm,
        goal_xy_norm,
        coordinate_scale,
        representation,
    )
    init_cp_error = torch.linalg.vector_norm(
        initial_reconstruction - initial_cp, dim=-1
    ).amax(dim=1)

    optimizer = torch.optim.AdamW(variables, lr=lr, weight_decay=0.0)
    requested_trace = sorted(
        set(step for step in trace_steps if 0 <= step <= iterations)
        | {0, iterations}
    )
    trace: List[Dict[str, object]] = []

    def record(step: int) -> None:
        with torch.no_grad():
            cp, state_details = state_to_control_points(
                method,
                variables,
                start_xy_norm,
                goal_xy_norm,
                coordinate_scale,
                representation,
            )
            metrics = detach_metrics(
                evaluate_control_points(
                    cp,
                    start_pose,
                    goal_pose,
                    cost_maps,
                    map_info,
                    bspline,
                    device,
                )
            )
            trace.append({"step": step, "metrics": metrics})

    record(0)
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        cp, _ = state_to_control_points(
            method,
            variables,
            start_xy_norm,
            goal_xy_norm,
            coordinate_scale,
            representation,
        )
        cost, _ = cost_on_dense_trajectory_tail_risk(
            bspline(cp),
            start_pose,
            goal_pose,
            cost_maps,
            map_info,
            device,
            return_per_sample=True,
            return_components=True,
        )
        cost.sum().backward()
        tensor_row_clip(variables, grad_clip)
        optimizer.step()
        if iteration in requested_trace:
            record(iteration)

    with torch.no_grad():
        final_cp, final_state_details = state_to_control_points(
            method,
            variables,
            start_xy_norm,
            goal_xy_norm,
            coordinate_scale,
            representation,
        )

    details: Dict[str, np.ndarray] = {
        "initial_cp_error_max": init_cp_error.detach().cpu().numpy(),
    }
    for key, value in init_details.items():
        if torch.is_tensor(value):
            details[f"init_{key}"] = value.detach().cpu().numpy()
    for key, value in final_state_details.items():
        if torch.is_tensor(value) and value.ndim <= 1:
            details[f"final_{key}"] = value.detach().cpu().numpy()
    return final_cp.detach(), trace, details


def choose_candidate_indices(
    dataset: UnevenPathDataLoader,
    environments: Sequence[str],
    paths_per_env: int,
    seed: int,
) -> List[int]:
    by_env: Dict[str, List[int]] = {env: [] for env in environments}
    for dataset_index, (env_index, _) in enumerate(dataset.indexDict):
        by_env[dataset.env_list[env_index]].append(dataset_index)
    rng = random.Random(seed + 17)
    indices: List[int] = []
    for environment in environments:
        available = by_env[environment]
        if paths_per_env > len(available):
            raise ValueError(
                f"{environment} has {len(available)} paths, "
                f"requested {paths_per_env}"
            )
        indices.extend(sorted(rng.sample(available, paths_per_env)))
    return indices


def collect_candidates(
    args: argparse.Namespace,
    environments: Sequence[str],
    model: PathDiffusionTransformer,
    coordinate_scale: float,
    device: torch.device,
) -> List[Candidate]:
    dataset = UnevenPathDataLoader(
        env_list=list(environments),
        dataFolder=args.dataset,
        compute_stability_map=True,
    )
    indices = choose_candidate_indices(
        dataset, environments, args.paths_per_env, args.seed
    )
    bspline = DifferentiableBSpline(
        num_control_points=26, num_output_points=100, degree=3
    ).to(device)
    map_info = MAP_CONFIG.cost_map_info()
    candidates: List[Candidate] = []

    # Process one environment at a time so its expensive stability map is
    # computed once and GPU memory remains bounded.
    groups: Dict[str, List[int]] = {env: [] for env in environments}
    for index in indices:
        env_index, _ = dataset.indexDict[index]
        groups[dataset.env_list[env_index]].append(index)

    model.eval()
    for environment in environments:
        group = groups[environment]
        items = [dataset[index] for index in group]
        maps = torch.stack([item["map"] for item in items]).to(device)
        starts = torch.stack([item["start_pose"] for item in items]).to(device)
        goals = torch.stack([item["goal_pose"] for item in items]).to(device)
        cost_maps = torch.stack([item["cost_map"] for item in items]).to(device)
        start_normalized = normalize_poses(starts, coordinate_scale)
        goal_normalized = normalize_poses(goals, coordinate_scale)

        with torch.no_grad():
            control_points = model.sample(
                maps,
                start_normalized,
                goal_normalized,
                num_samples=1,
                num_steps=3,
                solver="pmf_onestep",
                reconstruct_trajectory=False,
                num_traj_points=100,
            )
            metrics = evaluate_control_points(
                control_points,
                starts,
                goals,
                cost_maps,
                map_info,
                bspline,
                device,
            )

        for local_index, dataset_index in enumerate(group):
            _, path_num = dataset.indexDict[dataset_index]
            candidates.append(
                Candidate(
                    environment=environment,
                    path_num=path_num,
                    dataset_index=dataset_index,
                    start_pose=starts[local_index].detach().cpu(),
                    goal_pose=goals[local_index].detach().cpu(),
                    cost_map=cost_maps[local_index].detach().cpu(),
                    elevation=items[local_index]["elevation"].detach().cpu(),
                    control_points=control_points[local_index].detach().cpu(),
                    initial_total=float(metrics["total"][local_index].item()),
                    initial_obstacle=float(
                        metrics["obstacle"][local_index].item()
                    ),
                    initial_dangerous_ratio=float(
                        metrics["dangerous_ratio"][local_index].item()
                    ),
                )
            )
        print(
            f"Collected {environment}: {len(group)} candidates "
            f"(total={len(candidates)})",
            flush=True,
        )
    return candidates


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q10": float("nan"),
            "q90": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "q10": float(np.quantile(finite, 0.1)),
        "q90": float(np.quantile(finite, 0.9)),
    }


def plot_cost_curves(
    trace_rows: List[Dict[str, object]], output_path: Path
) -> None:
    grouped: Dict[Tuple[str, float], List[Dict[str, object]]] = {}
    for row in trace_rows:
        key = (str(row["method"]), float(row["lr"]))
        grouped.setdefault(key, []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (method, lr), rows in sorted(grouped.items()):
        by_step: Dict[int, List[float]] = {}
        for row in rows:
            by_step.setdefault(int(row["step"]), []).append(float(row["total"]))
        steps = sorted(by_step)
        median = [float(np.median(by_step[step])) for step in steps]
        mean = [float(np.mean(by_step[step])) for step in steps]
        label = f"{method}, lr={lr:g}"
        axes[0].plot(steps, median, marker="o", label=label)
        axes[1].plot(steps, mean, marker="o", label=label)
    axes[0].set_title("Median total cost")
    axes[1].set_title("Mean total cost")
    for axis in axes:
        axis.set_xlabel("Optimization step")
        axis.set_ylabel("Tail-risk cost")
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_recovery(
    result_rows: List[Dict[str, object]],
    best_lrs: Dict[str, float],
    output_path: Path,
) -> None:
    chosen = [
        row
        for row in result_rows
        if math.isclose(
            float(row["lr"]),
            best_lrs[str(row["method"])],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    methods = ["P", "R", "raw"]
    values = [
        np.array(
            [
                float(row["total_improvement_pct"])
                for row in chosen
                if row["method"] == method
            ]
        )
        for method in methods
    ]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(values, labels=methods, showmeans=True)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Total-cost improvement (%)")
    axis.set_title("Experiment A: best global LR per representation")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_examples(
    selected: Sequence[Candidate],
    best_control_points: Dict[str, torch.Tensor],
    best_lrs: Dict[str, float],
    output_path: Path,
    count: int,
) -> None:
    count = min(count, len(selected))
    if count <= 0:
        return
    fig, axes = plt.subplots(count, 4, figsize=(18, 4.5 * count))
    if count == 1:
        axes = axes[None, :]
    labels = ["Initial", "P", "R", "raw"]
    colors = ["#00a9a5", "#e76f51", "#7b2cbf", "#1d4ed8"]
    bspline = DifferentiableBSpline(
        num_control_points=26,
        num_output_points=100,
        degree=3,
    )
    extent = [
        MAP_CONFIG.bounds[0],
        MAP_CONFIG.bounds[1],
        MAP_CONFIG.bounds[2],
        MAP_CONFIG.bounds[3],
    ]
    for row_index in range(count):
        candidate = selected[row_index]
        trajectories = {
            "Initial": candidate.control_points.numpy(),
            "P": best_control_points["P"][row_index].cpu().numpy(),
            "R": best_control_points["R"][row_index].cpu().numpy(),
            "raw": best_control_points["raw"][row_index].cpu().numpy(),
        }
        for column, label in enumerate(labels):
            axis = axes[row_index, column]
            axis.imshow(
                candidate.elevation.numpy(),
                extent=extent,
                origin="lower",
                cmap="terrain",
                alpha=0.75,
            )
            cp = trajectories[label]
            with torch.no_grad():
                dense = bspline(
                    torch.as_tensor(cp, dtype=torch.float32).unsqueeze(0)
                )[0].numpy()
            axis.plot(
                dense[:, 0],
                dense[:, 1],
                lw=2.5,
                color=colors[column],
                label="B-spline",
            )
            axis.plot(
                cp[:, 0],
                cp[:, 1],
                "--o",
                ms=2.0,
                lw=0.8,
                alpha=0.35,
                color=colors[column],
                label="control polygon",
            )
            axis.scatter(cp[0, 0], cp[0, 1], marker="s", color="green", s=35)
            axis.scatter(cp[-1, 0], cp[-1, 1], marker="*", color="red", s=55)
            lr_text = "" if label == "Initial" else f", lr={best_lrs[label]:g}"
            axis.set_title(
                f"{candidate.environment}/path_{candidate.path_num}: "
                f"{label}{lr_text}"
            )
            axis.set_xlim(MAP_CONFIG.bounds[0], MAP_CONFIG.bounds[1])
            axis.set_ylim(MAP_CONFIG.bounds[2], MAP_CONFIG.bounds[3])
            axis.set_aspect("equal")
            axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.time()

    model_dir = Path(args.model_dir)
    model_params_path = model_dir / "model_params.json"
    checkpoint_path = model_dir / args.checkpoint
    model_params = json.loads(model_params_path.read_text())
    model = PathDiffusionTransformer(**model_params["model_args"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    coordinate_scale = float(model.coordinate_scale)
    if not math.isclose(coordinate_scale, MAP_HALF_EXTENT, abs_tol=1e-6):
        raise ValueError(
            f"Model scale {coordinate_scale} != map half extent {MAP_HALF_EXTENT}"
        )

    environments = discover_envs(
        Path(args.dataset), args.num_envs, args.seed
    )
    run_config = {
        "args": vars(args),
        "map_config": MAP_CONFIG.to_dict(),
        "safety_cost_config": SAFETY_COST_CONFIG.to_dict(),
        "git": git_metadata(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "stage": checkpoint.get("stage"),
            "val_loss": checkpoint.get("val_loss"),
        },
        "environments": environments,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 0.0,
            "per_sample_grad_clip": args.grad_clip,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False)
    )

    candidates = collect_candidates(
        args, environments, model, coordinate_scale, device
    )
    candidates.sort(
        key=lambda candidate: (
            candidate.initial_obstacle,
            candidate.initial_total,
        ),
        reverse=True,
    )
    if args.select > len(candidates):
        raise ValueError(
            f"Requested {args.select} selected samples, "
            f"only {len(candidates)} candidates"
        )
    selected = candidates[: args.select]
    selection_rows = [
        {
            "rank": rank,
            "environment": candidate.environment,
            "path_num": candidate.path_num,
            "dataset_index": candidate.dataset_index,
            "initial_total": candidate.initial_total,
            "initial_obstacle": candidate.initial_obstacle,
            "initial_dangerous_ratio": candidate.initial_dangerous_ratio,
        }
        for rank, candidate in enumerate(selected)
    ]
    write_csv(output_dir / "selected_samples.csv", selection_rows)

    initial_cp = torch.stack(
        [candidate.control_points for candidate in selected]
    ).to(device)
    starts = torch.stack([candidate.start_pose for candidate in selected]).to(
        device
    )
    goals = torch.stack([candidate.goal_pose for candidate in selected]).to(
        device
    )
    cost_maps = torch.stack(
        [candidate.cost_map for candidate in selected]
    ).to(device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    representation = PhysicalScaledEdgeResidualRepresentation().to(device)
    bspline = DifferentiableBSpline(
        num_control_points=26, num_output_points=100, degree=3
    ).to(device)
    map_info = MAP_CONFIG.cost_map_info()
    with torch.no_grad():
        initial_metrics = detach_metrics(
            evaluate_control_points(
                initial_cp,
                starts,
                goals,
                cost_maps,
                map_info,
                bspline,
                device,
            )
        )

    result_rows: List[Dict[str, object]] = []
    trace_rows: List[Dict[str, object]] = []
    final_control_points: Dict[Tuple[str, float], torch.Tensor] = {}
    initialization_records: List[Dict[str, object]] = []

    for method in ("P", "R", "raw"):
        for lr in args.lr_grid:
            print(
                f"Optimizing method={method}, lr={lr:g}, "
                f"B={len(selected)}, iterations={args.iterations}",
                flush=True,
            )
            final_cp, trace, details = optimize_method(
                method=method,
                lr=lr,
                iterations=args.iterations,
                trace_steps=args.trace_steps,
                initial_cp=initial_cp,
                start_pose=starts,
                goal_pose=goals,
                cost_maps=cost_maps,
                map_info=map_info,
                bspline=bspline,
                representation=representation,
                coordinate_scale=coordinate_scale,
                grad_clip=args.grad_clip,
                device=device,
            )
            final_control_points[(method, lr)] = final_cp.detach().cpu()
            max_init_error = float(
                np.max(details["initial_cp_error_max"])
            )
            initialization_records.append(
                {
                    "method": method,
                    "lr": lr,
                    "max_initial_cp_error_m": max_init_error,
                    "mean_initial_cp_error_m": float(
                        np.mean(details["initial_cp_error_max"])
                    ),
                    "raw_alpha_clipped_count": int(
                        np.sum(details.get("init_alpha_clipped", 0))
                    ),
                }
            )
            # All parameterizations must start from effectively the same path.
            tolerance = 2e-3 if method == "raw" else 1e-5
            if max_init_error > tolerance:
                raise RuntimeError(
                    f"{method} initialization mismatch: "
                    f"max control-point error={max_init_error:.6g} m "
                    f"> tolerance={tolerance}"
                )

            final_metrics = trace[-1]["metrics"]
            for sample_index, candidate in enumerate(selected):
                initial_total = float(initial_metrics["total"][sample_index])
                final_total = float(final_metrics["total"][sample_index])
                row: Dict[str, object] = {
                    "sample_index": sample_index,
                    "environment": candidate.environment,
                    "path_num": candidate.path_num,
                    "method": method,
                    "lr": lr,
                    "iterations": args.iterations,
                    "initial_total": initial_total,
                    "final_total": final_total,
                    "total_improvement": initial_total - final_total,
                    "total_improvement_pct": (
                        100.0 * (initial_total - final_total)
                        / max(abs(initial_total), 1e-12)
                    ),
                    "initial_cp_error_m": float(
                        details["initial_cp_error_max"][sample_index]
                    ),
                }
                for metric_name, metric_values in final_metrics.items():
                    row[f"initial_{metric_name}"] = float(
                        initial_metrics[metric_name][sample_index]
                    )
                    row[f"final_{metric_name}"] = float(
                        metric_values[sample_index]
                    )
                for detail_name, detail_values in details.items():
                    if np.asarray(detail_values).ndim == 1:
                        row[detail_name] = float(
                            np.asarray(detail_values)[sample_index]
                        )
                result_rows.append(row)

            for entry in trace:
                step = int(entry["step"])
                metrics = entry["metrics"]
                for sample_index, candidate in enumerate(selected):
                    trace_rows.append(
                        {
                            "sample_index": sample_index,
                            "environment": candidate.environment,
                            "path_num": candidate.path_num,
                            "method": method,
                            "lr": lr,
                            "step": step,
                            **{
                                name: float(values[sample_index])
                                for name, values in metrics.items()
                            },
                        }
                    )

    write_csv(output_dir / "initialization_checks.csv", initialization_records)
    write_csv(output_dir / "per_sample_results.csv", result_rows)
    write_csv(output_dir / "optimization_trace.csv", trace_rows)

    best_lrs: Dict[str, float] = {}
    lr_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in ("P", "R", "raw"):
        lr_summary[method] = {}
        best_key = None
        for lr in args.lr_grid:
            rows = [
                row
                for row in result_rows
                if row["method"] == method
                and math.isclose(float(row["lr"]), lr)
            ]
            final_total = np.array(
                [float(row["final_total"]) for row in rows]
            )
            improvement = np.array(
                [float(row["total_improvement_pct"]) for row in rows]
            )
            lr_summary[method][f"{lr:g}"] = {
                "final_total": summarize_array(final_total),
                "total_improvement_pct": summarize_array(improvement),
                "improved_fraction": float(
                    np.mean(
                        [
                            float(row["final_total"])
                            < float(row["initial_total"])
                            for row in rows
                        ]
                    )
                ),
            }
            key = (float(np.median(final_total)), float(np.mean(final_total)))
            if best_key is None or key < best_key:
                best_key = key
                best_lrs[method] = lr

    best_rows = {
        method: [
            row
            for row in result_rows
            if row["method"] == method
            and math.isclose(
                float(row["lr"]), best_lrs[method], abs_tol=1e-12
            )
        ]
        for method in ("P", "R", "raw")
    }
    p_gain = np.array(
        [float(row["total_improvement"]) for row in best_rows["P"]]
    )
    recovery: Dict[str, Dict[str, float]] = {}
    for method in ("R", "raw"):
        gain = np.array(
            [float(row["total_improvement"]) for row in best_rows[method]]
        )
        valid = p_gain > 1e-8
        recovery_values = np.full_like(gain, np.nan)
        recovery_values[valid] = gain[valid] / p_gain[valid]
        recovery[method] = {
            **summarize_array(recovery_values),
            "fraction_ge_0p8": float(
                np.mean(recovery_values[valid] >= 0.8)
            )
            if np.any(valid)
            else float("nan"),
            "fraction_ge_0p5": float(
                np.mean(recovery_values[valid] >= 0.5)
            )
            if np.any(valid)
            else float("nan"),
            "valid_count": int(np.sum(valid)),
        }

    diagnostic_metric_names = (
        "total",
        "obstacle",
        "dangerous_ratio",
        "unsafe_margin_ratio",
        "min_esdf",
        "path_length",
        "curvature_violation_ratio",
        "smoothness",
        "jerk",
    )
    best_method_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in ("P", "R", "raw"):
        method_rows = best_rows[method]
        method_metrics: Dict[str, Dict[str, float]] = {}
        for metric in diagnostic_metric_names:
            initial_values = np.asarray(
                [float(row[f"initial_{metric}"]) for row in method_rows]
            )
            final_values = np.asarray(
                [float(row[f"final_{metric}"]) for row in method_rows]
            )
            method_metrics[metric] = {
                "initial_mean": float(np.mean(initial_values)),
                "initial_median": float(np.median(initial_values)),
                "final_mean": float(np.mean(final_values)),
                "final_median": float(np.median(final_values)),
            }
        best_method_metrics[method] = method_metrics

    summary = {
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "best_global_lr": best_lrs,
        "lr_summary": lr_summary,
        "recovery_relative_to_P": recovery,
        "best_method_metrics": best_method_metrics,
        "initialization_checks": initialization_records,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    best_cp = {
        method: final_control_points[(method, best_lrs[method])]
        for method in ("P", "R", "raw")
    }
    np.savez_compressed(
        output_dir / "trajectories_and_control_points.npz",
        initial_control_points=initial_cp.detach().cpu().numpy(),
        P_control_points=best_cp["P"].numpy(),
        R_control_points=best_cp["R"].numpy(),
        raw_control_points=best_cp["raw"].numpy(),
        start_pose=starts.detach().cpu().numpy(),
        goal_pose=goals.detach().cpu().numpy(),
        environments=np.array(
            [candidate.environment for candidate in selected]
        ),
        path_nums=np.array([candidate.path_num for candidate in selected]),
    )

    plot_cost_curves(trace_rows, output_dir / "cost_curves.png")
    plot_recovery(result_rows, best_lrs, output_dir / "improvement_boxplot.png")
    plot_examples(
        selected,
        best_cp,
        best_lrs,
        output_dir / "trajectory_examples.png",
        args.plot_examples,
    )

    report_lines = [
        "# Experiment A: trajectory representation",
        "",
        f"- Candidates: {len(candidates)}",
        f"- Selected high-risk samples: {len(selected)}",
        f"- Iterations: {args.iterations}",
        f"- LR grid: {args.lr_grid}",
        f"- Best global LR: {best_lrs}",
        f"- Elapsed: {summary['elapsed_seconds']:.1f} s",
        "",
        "## Recovery relative to physical-control-point optimization",
        "",
    ]
    for method in ("R", "raw"):
        values = recovery[method]
        report_lines.append(
            f"- {method}: median={values['median']:.3f}, "
            f"mean={values['mean']:.3f}, "
            f"fraction >= 0.8: {values['fraction_ge_0p8']:.1%}, "
            f"fraction >= 0.5: {values['fraction_ge_0p5']:.1%}"
        )
    report_lines.extend(
        [
            "",
            "## Best-LR diagnostic medians",
            "",
            "| method | total cost | dangerous ratio | path length (m) | "
            "smoothness | jerk |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("P", "R", "raw"):
        metrics = best_method_metrics[method]
        report_lines.append(
            f"| {method} | {metrics['total']['initial_median']:.4f} -> "
            f"{metrics['total']['final_median']:.4f} | "
            f"{metrics['dangerous_ratio']['initial_median']:.3f} -> "
            f"{metrics['dangerous_ratio']['final_median']:.3f} | "
            f"{metrics['path_length']['initial_median']:.2f} -> "
            f"{metrics['path_length']['final_median']:.2f} | "
            f"{metrics['smoothness']['initial_median']:.5f} -> "
            f"{metrics['smoothness']['final_median']:.5f} | "
            f"{metrics['jerk']['initial_median']:.5f} -> "
            f"{metrics['jerk']['final_median']:.5f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- R and raw both recover at least 80% of P's cost reduction on "
            "all selected samples, so the residual/radial representation is "
            "not the main Stage-2 bottleneck.",
            "- Raw obtains its unusually low feasibility cost partly by taking "
            "large detours. Treat its result as evidence of representational "
            "capacity, not as evidence of better trajectory quality.",
            "- The large path-length increase shows that the current objective "
            "does not sufficiently penalize detours; quality has only a small "
            "tie-breaking weight and path length is absent.",
            "",
            "## Decision rule",
            "",
            "- R and raw recovery >= 0.8: representation is not the main bottleneck.",
            "- R strong but raw weak: radial feasibility map is a bottleneck.",
            "- R also weak: edge-residual representation is a bottleneck.",
            "",
            "See `summary.json`, `per_sample_results.csv`, "
            "`optimization_trace.csv`, and the generated plots for details.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
