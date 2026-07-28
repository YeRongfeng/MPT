#!/usr/bin/env python3
"""A/B/C/D test for optimizer-induced Stage-2 source/trajectory coupling.

The experiment keeps the Stage-1 architecture and trajectory representation
fixed.  For every condition it draws K explicit source noises, evaluates the
frozen Stage-1 proposal R0, and constructs a locally constrained privileged
teacher R*.  Three Stage-2 variants start from the same Stage-1 checkpoint:

  A: frozen Stage 1 (no training);
  B: cost-only Stage 2;
  C: teacher targets randomly re-paired among noises of the same condition;
  D: optimizer-induced (noise_i, R*_i) pairs.

Train and held-out conditions use disjoint maps.  The saved pair file contains
the explicit source noise, Stage-1 and teacher residuals/costs, topology mode,
and path length required to audit the coupling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from bspline_utils import DifferentiableBSpline
from dataLoader_dit import UnevenPathDataLoader
from dit.Models import (
    PathDiffusionTransformer,
    PhysicalScaledEdgeResidualRepresentation,
)
from experiment_a_representation import (
    evaluate_control_points,
    git_metadata,
    normalize_poses,
    set_deterministic,
    summarize_array,
)
from experiment_cplus_overfit import (
    baseline_dense_and_length,
    corrected_objective,
    load_model,
    model_output,
)
from map_config import MAP_CONFIG, MAP_HALF_EXTENT, SAFETY_COST_CONFIG


VARIANTS = ("B_cost_only", "C_random_coupling", "D_induced_coupling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dataset", default=str(MAP_CONFIG.dataset_root / "train")
    )
    parser.add_argument(
        "--heldout-dataset", default=str(MAP_CONFIG.dataset_root / "val")
    )
    parser.add_argument("--model-dir", default="data/sim")
    parser.add_argument("--checkpoint", default="stage1_best_model.pth")
    parser.add_argument("--train-envs", type=int, default=3)
    parser.add_argument("--heldout-envs", type=int, default=2)
    parser.add_argument(
        "--train-environment-list",
        nargs="+",
        default=None,
        help="Explicit nested Stage-2 train maps; overrides --train-envs.",
    )
    parser.add_argument(
        "--heldout-environment-list",
        nargs="+",
        default=None,
        help="Explicit fixed Stage-2 held-out maps; overrides --heldout-envs.",
    )
    parser.add_argument("--conditions-per-env", type=int, default=3)
    parser.add_argument("--pool-paths-per-env", type=int, default=10)
    parser.add_argument("--noises-per-condition", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--teacher-steps", type=int, default=300)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--teacher-grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-dev", type=float, default=0.2)
    parser.add_argument("--dev-scale-m", type=float, default=1.0)
    parser.add_argument("--lambda-len", type=float, default=200.0)
    parser.add_argument("--length-ratio-limit", type=float, default=1.15)
    parser.add_argument("--max-control-displacement", type=float, default=2.0)

    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="If set, derive train steps from samples/batch and override --train-steps.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Stage-2 variants to run; useful for splitting long comparisons.",
    )
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--train-lr", type=float, default=2e-5)
    parser.add_argument("--train-grad-clip", type=float, default=1.0)
    parser.add_argument("--teacher-huber-beta", type=float, default=0.1)
    parser.add_argument("--lambda-teacher-safe", type=float, default=0.01)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--mode-threshold-m", type=float, default=0.05)
    parser.add_argument("--safe-length-ratio", type=float, default=1.17)
    parser.add_argument(
        "--use-precomputed-stability",
        action="store_true",
        help="Load stability_map.npz instead of rebuilding yaw ESDF maps.",
    )
    parser.add_argument(
        "--stability-map-filename",
        default="stability_map.npz",
    )
    parser.add_argument("--output-dir", default="diagnostics/stage2_coupling_abcd")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def choose_disjoint_environments(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    # Optimized copies are derived data, not independent maps.
    train_available = sorted(
        path.name
        for path in Path(args.train_dataset).iterdir()
        if path.is_dir()
        and (path / "map.p").is_file()
        and not path.name.endswith("_optimized")
    )
    heldout_available = sorted(
        path.name
        for path in Path(args.heldout_dataset).iterdir()
        if path.is_dir()
        and (path / "map.p").is_file()
        and not path.name.endswith("_optimized")
    )
    rng = random.Random(args.seed)
    if args.train_environment_list is None:
        train_envs = sorted(rng.sample(train_available, args.train_envs))
    else:
        train_envs = list(args.train_environment_list)
        missing = sorted(set(train_envs) - set(train_available))
        if missing:
            raise ValueError(f"Unknown train environments: {missing}")
    heldout_candidates = [
        environment for environment in heldout_available
        if environment not in set(train_envs)
    ]
    if args.heldout_environment_list is None:
        heldout_envs = sorted(rng.sample(heldout_candidates, args.heldout_envs))
    else:
        heldout_envs = list(args.heldout_environment_list)
        missing = sorted(set(heldout_envs) - set(heldout_available))
        if missing:
            raise ValueError(f"Unknown held-out environments: {missing}")
        overlap = sorted(set(train_envs) & set(heldout_envs))
        if overlap:
            raise ValueError(f"Train/held-out environments overlap: {overlap}")
    return train_envs, heldout_envs


def normalize_selected_poses(
    starts: torch.Tensor, goals: torch.Tensor, coordinate_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    return normalize_poses(starts, coordinate_scale), normalize_poses(
        goals, coordinate_scale
    )


@torch.no_grad()
def stage1_from_noise(
    model: PathDiffusionTransformer,
    maps: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    start_n, goal_n = normalize_selected_poses(
        starts, goals, float(model.coordinate_scale)
    )
    batch_size = maps.shape[0]
    residual = model(
        maps,
        noise,
        torch.ones(batch_size, device=maps.device),
        torch.zeros(batch_size, device=maps.device),
        start_n,
        goal_n,
    )
    control_points = (
        model.trajectory_representation.decode(
            residual, start_n[:, :2], goal_n[:, :2]
        )
        * float(model.coordinate_scale)
    )
    return residual, control_points


def candidate_score(
    model: PathDiffusionTransformer,
    items: Sequence[Dict[str, torch.Tensor]],
    seed: int,
    device: torch.device,
) -> np.ndarray:
    maps = torch.stack([item["map"] for item in items]).to(device)
    starts = torch.stack([item["start_pose"] for item in items]).to(device)
    goals = torch.stack([item["goal_pose"] for item in items]).to(device)
    cost_maps = torch.stack([item["cost_map"] for item in items]).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = model.project_zero_sum(
        torch.randn(
            len(items), model.num_edges, 2, generator=generator, device=device
        )
    )
    _, control_points = stage1_from_noise(model, maps, starts, goals, noise)
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    metrics = evaluate_control_points(
        control_points,
        starts,
        goals,
        cost_maps,
        MAP_CONFIG.cost_map_info(),
        bspline,
        device,
    )
    # Prefer conditions where Stage 1 has actual terrain risk.
    return (
        1000.0 * metrics["dangerous_ratio"] + metrics["total"]
    ).detach().cpu().numpy()


def collect_conditions(
    split: str,
    dataset_root: str,
    environments: Sequence[str],
    args: argparse.Namespace,
    model: PathDiffusionTransformer,
    device: torch.device,
) -> List[Dict[str, object]]:
    dataset = UnevenPathDataLoader(
        env_list=list(environments),
        dataFolder=dataset_root,
        compute_stability_map=True,
        use_precomputed_stability=args.use_precomputed_stability,
        stability_map_filename=args.stability_map_filename,
        compute_stability_if_missing=False,
    )
    records: List[Dict[str, object]] = []
    for env_offset, environment in enumerate(environments):
        indices = [
            index
            for index, (env_index, _) in enumerate(dataset.indexDict)
            if dataset.env_list[env_index] == environment
        ][: args.pool_paths_per_env]
        if len(indices) < args.conditions_per_env:
            raise ValueError(f"{environment} does not have enough path conditions")
        items = [dataset[index] for index in indices]
        scores = candidate_score(
            model, items, args.seed + 1009 * (env_offset + 1), device
        )
        chosen_local = np.argsort(scores)[::-1][: args.conditions_per_env]
        for local_index in chosen_local:
            dataset_index = indices[int(local_index)]
            _, path_num = dataset.indexDict[dataset_index]
            item = items[int(local_index)]
            records.append(
                {
                    "split": split,
                    "environment": environment,
                    "path_num": int(path_num),
                    "map": item["map"].cpu(),
                    "cost_map": item["cost_map"].cpu(),
                    "elevation": item["elevation"].cpu(),
                    "start": item["start_pose"].cpu(),
                    "goal": item["goal_pose"].cpu(),
                    "screening_score": float(scores[int(local_index)]),
                }
            )
    return records


def expand_source_noises(
    conditions: Sequence[Dict[str, object]],
    model: PathDiffusionTransformer,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    maps: List[torch.Tensor] = []
    cost_maps: List[torch.Tensor] = []
    elevations: List[torch.Tensor] = []
    starts: List[torch.Tensor] = []
    goals: List[torch.Tensor] = []
    split: List[str] = []
    environments: List[str] = []
    path_nums: List[int] = []
    condition_ids: List[int] = []
    source_ids: List[int] = []
    noise_rows: List[torch.Tensor] = []
    for condition_id, condition in enumerate(conditions):
        condition_key = (
            f"{args.seed}|{condition['split']}|{condition['environment']}|"
            f"{condition['path_num']}"
        )
        condition_seed = int.from_bytes(
            hashlib.sha256(condition_key.encode("utf-8")).digest()[:8],
            byteorder="little",
        ) % (2**63 - 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(condition_seed)
        condition_noise = model.project_zero_sum(
            torch.randn(
                args.noises_per_condition,
                model.num_edges,
                2,
                device=device,
                generator=generator,
            )
        ).cpu()
        for source_id in range(args.noises_per_condition):
            maps.append(condition["map"])
            cost_maps.append(condition["cost_map"])
            elevations.append(condition["elevation"])
            starts.append(condition["start"])
            goals.append(condition["goal"])
            split.append(str(condition["split"]))
            environments.append(str(condition["environment"]))
            path_nums.append(int(condition["path_num"]))
            condition_ids.append(condition_id)
            source_ids.append(source_id)
            noise_rows.append(condition_noise[source_id])

    tensor_data: Dict[str, torch.Tensor] = {
        "map": torch.stack(maps),
        "cost_map": torch.stack(cost_maps),
        "elevation": torch.stack(elevations),
        "start": torch.stack(starts),
        "goal": torch.stack(goals),
        "condition_id": torch.tensor(condition_ids, dtype=torch.long),
        "source_id": torch.tensor(source_ids, dtype=torch.long),
    }
    noise = torch.stack(noise_rows).to(device)
    outputs_r: List[torch.Tensor] = []
    outputs_cp: List[torch.Tensor] = []
    # Evaluate one condition (its fixed K noises) at a time.  This keeps the
    # CUDA batch shape and numerical path identical when N_map changes.
    for condition_id in range(len(conditions)):
        begin = condition_id * args.noises_per_condition
        end = begin + args.noises_per_condition
        r0, cp0 = stage1_from_noise(
            model,
            tensor_data["map"][begin:end].to(device),
            tensor_data["start"][begin:end].to(device),
            tensor_data["goal"][begin:end].to(device),
            noise[begin:end],
        )
        outputs_r.append(r0.cpu())
        outputs_cp.append(cp0.cpu())
    tensor_data["noise"] = noise.cpu()
    tensor_data["r0"] = torch.cat(outputs_r)
    tensor_data["cp0"] = torch.cat(outputs_cp)
    return {
        **tensor_data,
        "split_name": split,
        "environment": environments,
        "path_num": path_nums,
    }


def tensor_subset(data: Dict[str, object], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    count = len(data["split_name"])
    return {
        key: value.index_select(0, indices)
        for key, value in data.items()
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count
    }


def side_score(control_points: torch.Tensor) -> torch.Tensor:
    start = control_points[:, :1, :]
    chord = control_points[:, -1:, :] - start
    relative = control_points - start
    cross = chord[..., 0] * relative[..., 1] - chord[..., 1] * relative[..., 0]
    chord_norm = torch.linalg.vector_norm(chord, dim=-1).clamp_min(1e-6)
    return (cross / chord_norm).mean(dim=1)


def mode_label(control_points: torch.Tensor, threshold: float) -> torch.Tensor:
    score = side_score(control_points)
    return torch.where(
        score > threshold,
        torch.ones_like(score, dtype=torch.long),
        torch.where(
            score < -threshold,
            -torch.ones_like(score, dtype=torch.long),
            torch.zeros_like(score, dtype=torch.long),
        ),
    )


def self_intersection(control_points: torch.Tensor) -> torch.Tensor:
    # Use the control polygon as a conservative and inexpensive abnormality
    # check.  Adjacent segments and the first/last open-path pair are skipped.
    p = control_points.detach().cpu().numpy()
    result = np.zeros(p.shape[0], dtype=bool)

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(np.cross(b - a, c - a))

    for batch in range(p.shape[0]):
        for i in range(p.shape[1] - 1):
            for j in range(i + 2, p.shape[1] - 1):
                a, b = p[batch, i], p[batch, i + 1]
                c, d = p[batch, j], p[batch, j + 1]
                if orient(a, b, c) * orient(a, b, d) < 0.0 and orient(
                    c, d, a
                ) * orient(c, d, b) < 0.0:
                    result[batch] = True
                    break
            if result[batch]:
                break
    return torch.from_numpy(result)


def path_length(control_points: torch.Tensor) -> torch.Tensor:
    bspline = DifferentiableBSpline(26, 100, 3)
    with torch.no_grad():
        dense = bspline(control_points.cpu())
    return torch.linalg.vector_norm(
        dense[:, 1:] - dense[:, :-1], dim=-1
    ).sum(dim=1)


def generate_trust_region_teacher(
    data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimize physical control points with hard local trust projections."""
    model_representation = PhysicalScaledEdgeResidualRepresentation().to(device)
    cp0 = data["cp0"].to(device)
    starts = data["start"].to(device)
    goals = data["goal"].to(device)
    cost_maps = data["cost_map"].to(device)
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    dense0, length0 = baseline_dense_and_length(cp0, bspline)
    mode0 = mode_label(cp0, args.mode_threshold_m)
    side0 = side_score(cp0)
    middle = cp0[:, 1:-1].detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [middle], lr=args.teacher_lr, weight_decay=0.0
    )

    def full_control_points() -> torch.Tensor:
        return torch.cat(
            [cp0[:, :1], middle, cp0[:, -1:]], dim=1
        )

    def project_trust_region() -> None:
        with torch.no_grad():
            delta = middle - cp0[:, 1:-1]
            displacement = torch.linalg.vector_norm(
                delta, dim=-1, keepdim=True
            )
            delta.mul_(
                torch.clamp(
                    args.max_control_displacement
                    / displacement.clamp_min(1e-8),
                    max=1.0,
                )
            )
            candidate_middle = torch.clamp(
                cp0[:, 1:-1] + delta,
                min=-MAP_HALF_EXTENT + 1e-4,
                max=MAP_HALF_EXTENT - 1e-4,
            )
            delta = candidate_middle - cp0[:, 1:-1]

            # Binary-search the largest interpolation factor satisfying the
            # hard path-length ratio.  alpha=0 is exactly R0 and is feasible.
            low = torch.zeros(cp0.shape[0], device=device)
            high = torch.ones_like(low)
            for _ in range(10):
                alpha = (low + high) * 0.5
                candidate = torch.cat(
                    [
                        cp0[:, :1],
                        cp0[:, 1:-1] + alpha[:, None, None] * delta,
                        cp0[:, -1:],
                    ],
                    dim=1,
                )
                candidate_dense = bspline(candidate)
                candidate_length = torch.linalg.vector_norm(
                    candidate_dense[:, 1:] - candidate_dense[:, :-1], dim=-1
                ).sum(dim=1)
                feasible = (
                    candidate_length / length0.clamp_min(1e-6)
                    <= args.safe_length_ratio
                )
                low = torch.where(feasible, alpha, low)
                high = torch.where(feasible, high, alpha)
            alpha = low

            # The side score is affine under interpolation.  If a candidate
            # crosses the original left/right side, stop just before zero.
            candidate = torch.cat(
                [
                    cp0[:, :1],
                    cp0[:, 1:-1] + alpha[:, None, None] * delta,
                    cp0[:, -1:],
                ],
                dim=1,
            )
            side1 = side_score(candidate)
            crosses = ((mode0 > 0) & (side1 <= 0.0)) | (
                (mode0 < 0) & (side1 >= 0.0)
            )
            crossing_alpha = (
                side0.abs()
                / (side0.abs() + side1.abs()).clamp_min(1e-8)
                * alpha
                * 0.99
            )
            alpha = torch.where(crosses, crossing_alpha, alpha)
            middle.copy_(
                cp0[:, 1:-1] + alpha[:, None, None] * delta
            )

    for step in range(1, args.teacher_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        control_points = full_control_points()
        objective, _ = corrected_objective(
            control_points,
            dense0,
            length0,
            starts,
            goals,
            cost_maps,
            bspline,
            args,
            device,
        )
        objective.sum().backward()
        with torch.no_grad():
            grad = middle.grad
            if grad is not None:
                grad_norm = torch.linalg.vector_norm(
                    grad.flatten(start_dim=1), dim=1
                )
                grad_scale = torch.clamp(
                    args.teacher_grad_clip / grad_norm.clamp_min(1e-12),
                    max=1.0,
                )
                grad.mul_(grad_scale[:, None, None])
        optimizer.step()
        project_trust_region()
        if step == 1 or step % 100 == 0 or step == args.teacher_steps:
            print(
                f"teacher: step {step}/{args.teacher_steps}, "
                f"objective={objective.median().item():.6g}"
            )

    with torch.no_grad():
        cp_star = full_control_points()
        start_n, goal_n = normalize_selected_poses(
            starts, goals, coordinate_scale
        )
        r_star = model_representation.encode(
            cp_star / coordinate_scale,
            start_n[:, :2],
            goal_n[:, :2],
        )
        r_star = model_representation.radial_project_residual(
            r_star, start_n[:, :2], goal_n[:, :2]
        )
        cp_star = (
            model_representation.decode(
                r_star, start_n[:, :2], goal_n[:, :2]
            )
            * coordinate_scale
        )
    del optimizer, model_representation
    return r_star.cpu(), cp_star.cpu()


def enforce_teacher_trust_region(
    data: Dict[str, object],
    r_star: torch.Tensor,
    cp_star: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    cp0 = data["cp0"]
    length0 = path_length(cp0)
    length_star = path_length(cp_star)
    max_displacement = torch.linalg.vector_norm(
        cp_star - cp0, dim=-1
    ).amax(dim=1)
    mode0 = mode_label(cp0, args.mode_threshold_m)
    side_star = side_score(cp_star)
    # A straight/ambiguous proposal (mode 0) does not define a topology side.
    # For left/right proposals, "same topology" means not crossing zero; it
    # does not require preserving an arbitrary classification margin.
    topology_ok = (
        (mode0 == 0)
        | ((mode0 > 0) & (side_star > 0.0))
        | ((mode0 < 0) & (side_star < 0.0))
    )
    length_ok = length_star / length0.clamp_min(1e-6) <= args.safe_length_ratio
    displacement_ok = max_displacement <= args.max_control_displacement
    finite = torch.isfinite(cp_star).flatten(start_dim=1).all(dim=1)
    valid = topology_ok & length_ok & displacement_ok & finite
    safe_r = torch.where(valid[:, None, None], r_star, data["r0"])
    safe_cp = torch.where(valid[:, None, None], cp_star, cp0)
    return safe_r, safe_cp, {
        "teacher_valid": valid,
        "teacher_topology_ok": topology_ok,
        "teacher_length_ok": length_ok,
        "teacher_displacement_ok": displacement_ok,
        "teacher_max_control_displacement": max_displacement,
    }


def component_metrics(
    control_points: torch.Tensor,
    data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    chunks: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for begin in range(0, control_points.shape[0], args.eval_batch_size):
            end = min(begin + args.eval_batch_size, control_points.shape[0])
            cp0 = data["cp0"][begin:end].to(device)
            dense0, length0 = baseline_dense_and_length(cp0, bspline)
            _, components = corrected_objective(
                control_points[begin:end].to(device),
                dense0,
                length0,
                data["start"][begin:end].to(device),
                data["goal"][begin:end].to(device),
                data["cost_map"][begin:end].to(device),
                bspline,
                args,
                device,
            )
            physical = evaluate_control_points(
                control_points[begin:end].to(device),
                data["start"][begin:end].to(device),
                data["goal"][begin:end].to(device),
                data["cost_map"][begin:end].to(device),
                MAP_CONFIG.cost_map_info(),
                bspline,
                device,
            )
            for key in (
                "dangerous_ratio",
                "unsafe_margin_ratio",
                "min_esdf",
                "curvature_violation_ratio",
                "control_oob_ratio",
                "path_length",
            ):
                components[key] = physical[key]
            for key, value in components.items():
                chunks.setdefault(key, []).append(value.detach().cpu())
    result = {key: torch.cat(value).numpy() for key, value in chunks.items()}
    result["mode"] = mode_label(
        control_points, args.mode_threshold_m
    ).numpy()
    result["self_intersection"] = self_intersection(control_points).numpy()
    return result


def fixed_random_repair_indices(
    condition_ids: torch.Tensor, source_ids: torch.Tensor
) -> torch.Tensor:
    result = torch.empty_like(condition_ids)
    for condition_id in condition_ids.unique(sorted=True):
        group = torch.nonzero(condition_ids == condition_id).flatten()
        ordered = group[torch.argsort(source_ids[group])]
        if len(ordered) < 2:
            raise ValueError("Random coupling needs at least two noises per condition")
        result[ordered] = torch.roll(ordered, shifts=-1)
    if torch.any(result == torch.arange(len(result))):
        raise AssertionError("Random coupling unexpectedly retained an identity pair")
    return result


def training_batches(count: int, args: argparse.Namespace) -> List[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 303)
    if args.train_epochs is not None:
        batches: List[torch.Tensor] = []
        for _ in range(args.train_epochs):
            order = torch.randperm(count, generator=generator)
            for begin in range(0, count, args.train_batch_size):
                batches.append(order[begin : begin + args.train_batch_size])
        return batches

    order = torch.randperm(count, generator=generator)
    cursor = 0
    batches: List[torch.Tensor] = []
    for _ in range(args.train_steps):
        if cursor + args.train_batch_size > count:
            order = torch.randperm(count, generator=generator)
            cursor = 0
        batches.append(order[cursor : cursor + args.train_batch_size].clone())
        cursor += args.train_batch_size
    return batches


def evaluate_model(
    model: PathDiffusionTransformer,
    data: Dict[str, torch.Tensor],
    r_star: torch.Tensor,
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
    output_r: List[torch.Tensor] = []
    output_cp: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, data["map"].shape[0], args.eval_batch_size):
            end = min(begin + args.eval_batch_size, data["map"].shape[0])
            r_hat, cp_hat = model_output(
                model,
                data["map"][begin:end].to(device),
                data["noise"][begin:end].to(device),
                data["start"][begin:end].to(device),
                data["goal"][begin:end].to(device),
                coordinate_scale,
            )
            output_r.append(r_hat.cpu())
            output_cp.append(cp_hat.cpu())
    r_hat = torch.cat(output_r)
    cp_hat = torch.cat(output_cp)
    metrics = component_metrics(cp_hat, data, args, device)
    teacher_error = torch.linalg.vector_norm(
        (r_hat - r_star).flatten(start_dim=1), dim=1
    )
    teacher_delta = torch.linalg.vector_norm(
        (data["r0"] - r_star).flatten(start_dim=1), dim=1
    )
    relative_error = teacher_error / teacher_delta.clamp_min(1e-8)
    relative_error = torch.where(
        teacher_delta > 1e-5,
        relative_error,
        torch.full_like(relative_error, float("nan")),
    )
    metrics["relative_teacher_error"] = relative_error.numpy()
    return r_hat, cp_hat, metrics


def measure_inference_ms(
    model: PathDiffusionTransformer,
    data: Dict[str, torch.Tensor],
    coordinate_scale: float,
    device: torch.device,
) -> float:
    count = data["map"].shape[0]
    maps = data["map"].to(device)
    noise = data["noise"].to(device)
    starts = data["start"].to(device)
    goals = data["goal"].to(device)
    with torch.no_grad():
        for _ in range(2):
            model_output(model, maps, noise, starts, goals, coordinate_scale)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(5):
            model_output(model, maps, noise, starts, goals, coordinate_scale)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / (5 * count)


def train_variant(
    variant: str,
    model_params: Dict[str, object],
    checkpoint: Dict[str, object],
    train_data: Dict[str, torch.Tensor],
    heldout_data: Dict[str, torch.Tensor],
    train_r_star: torch.Tensor,
    heldout_r_star: torch.Tensor,
    random_targets: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    coordinate_scale = float(model_params["model_args"]["coordinate_scale"])
    model = load_model(model_params, checkpoint, device)
    model.eval()  # deterministic Stage-2 comparison: dropout is disabled
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.train_lr, weight_decay=0.0
    )
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    batches = training_batches(train_data["map"].shape[0], args)
    target = (
        random_targets
        if variant == "C_random_coupling"
        else train_r_star
    )
    losses: List[float] = []
    for step, indices in enumerate(batches, 1):
        maps = train_data["map"][indices].to(device)
        starts = train_data["start"][indices].to(device)
        goals = train_data["goal"][indices].to(device)
        noise = train_data["noise"][indices].to(device)
        cp0 = train_data["cp0"][indices].to(device)
        cost_maps = train_data["cost_map"][indices].to(device)
        dense0, length0 = baseline_dense_and_length(cp0, bspline)
        optimizer.zero_grad(set_to_none=True)
        r_hat, cp_hat = model_output(
            model, maps, noise, starts, goals, coordinate_scale
        )
        corrected, _ = corrected_objective(
            cp_hat,
            dense0,
            length0,
            starts,
            goals,
            cost_maps,
            bspline,
            args,
            device,
        )
        if variant == "B_cost_only":
            loss = corrected.mean()
        else:
            teacher_loss = F.smooth_l1_loss(
                r_hat,
                target[indices].to(device),
                beta=args.teacher_huber_beta,
            )
            loss = teacher_loss + args.lambda_teacher_safe * corrected.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.train_steps:
            losses.append(float(loss.detach()))
            print(f"{variant}: step {step}/{args.train_steps}, loss={losses[-1]:.6g}")

    train_r, train_cp, train_metrics = evaluate_model(
        model,
        train_data,
        train_r_star,
        args,
        coordinate_scale,
        device,
    )
    heldout_r, heldout_cp, heldout_metrics = evaluate_model(
        model,
        heldout_data,
        heldout_r_star,
        args,
        coordinate_scale,
        device,
    )
    inference_ms = measure_inference_ms(
        model, heldout_data, coordinate_scale, device
    )
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "train_r": train_r,
        "train_cp": train_cp,
        "train_metrics": train_metrics,
        "heldout_r": heldout_r,
        "heldout_cp": heldout_cp,
        "heldout_metrics": heldout_metrics,
        "loss_trace": losses,
        "inference_ms_per_sample": inference_ms,
    }


def recovery(
    initial: np.ndarray, output: np.ndarray, teacher: np.ndarray
) -> np.ndarray:
    denominator = initial - teacher
    result = np.full_like(initial, np.nan, dtype=np.float64)
    valid = denominator > 1e-6
    result[valid] = (initial[valid] - output[valid]) / denominator[valid]
    return result


def aggregate_metrics(
    metrics: Dict[str, np.ndarray],
    initial: Dict[str, np.ndarray],
    teacher: Dict[str, np.ndarray],
    condition_ids: torch.Tensor,
    cp: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, object]:
    safe_recovery = recovery(initial["safe"], metrics["safe"], teacher["safe"])
    corrected_recovery = recovery(
        initial["corrected"], metrics["corrected"], teacher["corrected"]
    )
    mode_preserved = metrics["mode"] == initial["mode"]
    abnormal = (
        metrics["self_intersection"].astype(bool)
        | (metrics["length_ratio"] > args.safe_length_ratio)
        | (metrics["control_oob_ratio"] > 0.0)
        | ~np.isfinite(metrics["safe"])
    )
    bspline = DifferentiableBSpline(26, 100, 3)
    with torch.no_grad():
        dense = bspline(cp.cpu())
    condition_rows: List[Dict[str, object]] = []
    diversity_values: List[float] = []
    for condition_id in condition_ids.unique(sorted=True):
        indices = torch.nonzero(condition_ids == condition_id).flatten()
        local_dense = dense[indices]
        pairwise = torch.cdist(
            local_dense.flatten(start_dim=1),
            local_dense.flatten(start_dim=1),
        ) / math.sqrt(local_dense.shape[1])
        upper = pairwise[
            torch.triu(
                torch.ones_like(pairwise, dtype=torch.bool), diagonal=1
            )
        ]
        diversity = float(upper.mean()) if upper.numel() else 0.0
        diversity_values.append(diversity)
        local = indices.numpy()
        safe_mask = (
            metrics["dangerous_ratio"][local] <= 0.0
        ) & (metrics["length_ratio"][local] <= args.safe_length_ratio)
        condition_rows.append(
            {
                "condition_id": int(condition_id),
                "safe_at_k": bool(np.any(safe_mask)),
                "best_of_k_safe_cost": float(np.min(metrics["safe"][local])),
                "mode_count": int(np.unique(metrics["mode"][local]).size),
                "diversity_rms_m": diversity,
            }
        )
    return {
        "safe_cost": summarize_array(metrics["safe"]),
        "corrected_cost": summarize_array(metrics["corrected"]),
        "recovery_safe": summarize_array(safe_recovery[np.isfinite(safe_recovery)]),
        "recovery_corrected": summarize_array(
            corrected_recovery[np.isfinite(corrected_recovery)]
        ),
        "valid_safe_recovery_count": int(np.isfinite(safe_recovery).sum()),
        "mode_preservation_rate": float(np.mean(mode_preserved)),
        "path_length_ratio": summarize_array(metrics["length_ratio"]),
        "self_intersection_rate": float(np.mean(metrics["self_intersection"])),
        "abnormal_rate": float(np.mean(abnormal)),
        "relative_teacher_error": (
            summarize_array(
                metrics["relative_teacher_error"][
                    np.isfinite(metrics["relative_teacher_error"])
                ]
            )
            if "relative_teacher_error" in metrics
            and np.isfinite(metrics["relative_teacher_error"]).any()
            else None
        ),
        "safe_at_k": float(np.mean([row["safe_at_k"] for row in condition_rows])),
        "best_of_k_safe_cost": summarize_array(
            np.asarray([row["best_of_k_safe_cost"] for row in condition_rows])
        ),
        "diversity_rms_m": summarize_array(np.asarray(diversity_values)),
        "condition_metrics": condition_rows,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.noises_per_condition < 2:
        raise ValueError("--noises-per-condition must be at least 2")
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this controlled experiment")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    model_dir = Path(args.model_dir)
    model_params = json.loads((model_dir / "model_params.json").read_text())
    checkpoint_path = model_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    coordinate_scale = float(model_params["model_args"]["coordinate_scale"])
    if not math.isclose(coordinate_scale, MAP_HALF_EXTENT, abs_tol=1e-6):
        raise ValueError("Model and map coordinate scales differ")

    train_envs, heldout_envs = choose_disjoint_environments(args)
    collection_model = load_model(model_params, checkpoint, device)
    train_conditions = collect_conditions(
        "train",
        args.train_dataset,
        train_envs,
        args,
        collection_model,
        device,
    )
    heldout_conditions = collect_conditions(
        "heldout",
        args.heldout_dataset,
        heldout_envs,
        args,
        collection_model,
        device,
    )
    all_data = expand_source_noises(
        train_conditions + heldout_conditions,
        collection_model,
        args,
        device,
    )
    del collection_model
    torch.cuda.empty_cache()

    train_indices = torch.tensor(
        [
            index
            for index, split in enumerate(all_data["split_name"])
            if split == "train"
        ],
        dtype=torch.long,
    )
    heldout_indices = torch.tensor(
        [
            index
            for index, split in enumerate(all_data["split_name"])
            if split == "heldout"
        ],
        dtype=torch.long,
    )
    train_data = tensor_subset(all_data, train_indices)
    heldout_data = tensor_subset(all_data, heldout_indices)

    print("Generating locally constrained privileged teachers ...")
    train_r_star, train_cp_star = generate_trust_region_teacher(
        train_data,
        args,
        coordinate_scale,
        device,
    )
    train_r_star, train_cp_star, train_trust = enforce_teacher_trust_region(
        train_data, train_r_star, train_cp_star, args
    )
    heldout_r_star, heldout_cp_star = generate_trust_region_teacher(
        heldout_data,
        args,
        coordinate_scale,
        device,
    )
    heldout_r_star, heldout_cp_star, heldout_trust = enforce_teacher_trust_region(
        heldout_data, heldout_r_star, heldout_cp_star, args
    )

    r_star = torch.empty_like(all_data["r0"])
    cp_star = torch.empty_like(all_data["cp0"])
    r_star[train_indices] = train_r_star
    r_star[heldout_indices] = heldout_r_star
    cp_star[train_indices] = train_cp_star
    cp_star[heldout_indices] = heldout_cp_star
    trust: Dict[str, torch.Tensor] = {}
    for key in train_trust:
        combined = torch.empty(
            len(all_data["split_name"]),
            *train_trust[key].shape[1:],
            dtype=train_trust[key].dtype,
        )
        combined[train_indices] = train_trust[key]
        combined[heldout_indices] = heldout_trust[key]
        trust[key] = combined

    print(
        f"Teacher trust-region acceptance: "
        f"{trust['teacher_valid'].float().mean().item():.1%}"
    )
    print(
        "  topology/length/displacement acceptance: "
        f"{trust['teacher_topology_ok'].float().mean().item():.1%}/"
        f"{trust['teacher_length_ok'].float().mean().item():.1%}/"
        f"{trust['teacher_displacement_ok'].float().mean().item():.1%}"
    )

    if args.train_epochs is not None:
        if args.train_epochs <= 0:
            raise ValueError("--train-epochs must be positive")
        steps_per_epoch = math.ceil(
            train_data["map"].shape[0] / args.train_batch_size
        )
        args.train_steps = args.train_epochs * steps_per_epoch
        print(
            f"Stage-2 schedule: {args.train_epochs} epochs x "
            f"{steps_per_epoch} steps = {args.train_steps} steps"
        )
    re_pair = fixed_random_repair_indices(
        train_data["condition_id"], train_data["source_id"]
    )
    random_targets = train_r_star[re_pair]

    initial_metrics = {
        "train": component_metrics(train_data["cp0"], train_data, args, device),
        "heldout": component_metrics(
            heldout_data["cp0"], heldout_data, args, device
        ),
    }
    teacher_metrics = {
        "train": component_metrics(train_cp_star, train_data, args, device),
        "heldout": component_metrics(
            heldout_cp_star, heldout_data, args, device
        ),
    }
    selected_variants = tuple(args.variants)
    results: Dict[str, Dict[str, object]] = {}
    for variant in selected_variants:
        print(f"\nTraining {variant} ...")
        results[variant] = train_variant(
            variant,
            model_params,
            checkpoint,
            train_data,
            heldout_data,
            train_r_star,
            heldout_r_star,
            random_targets,
            args,
            device,
        )

    summaries: Dict[str, object] = {
        "A_stage1": {},
        "teacher": {},
        **{variant: {} for variant in selected_variants},
    }
    for split_name, split_data, split_cp_star in (
        ("train", train_data, train_cp_star),
        ("heldout", heldout_data, heldout_cp_star),
    ):
        summaries["A_stage1"][split_name] = aggregate_metrics(
            initial_metrics[split_name],
            initial_metrics[split_name],
            teacher_metrics[split_name],
            split_data["condition_id"],
            split_data["cp0"],
            args,
        )
        summaries["teacher"][split_name] = aggregate_metrics(
            teacher_metrics[split_name],
            initial_metrics[split_name],
            teacher_metrics[split_name],
            split_data["condition_id"],
            split_cp_star,
            args,
        )
        for variant in selected_variants:
            summaries[variant][split_name] = aggregate_metrics(
                results[variant][f"{split_name}_metrics"],
                initial_metrics[split_name],
                teacher_metrics[split_name],
                split_data["condition_id"],
                results[variant][f"{split_name}_cp"],
                args,
            )
            summaries[variant]["inference_ms_per_sample"] = results[variant][
                "inference_ms_per_sample"
            ]

    config = {
        "experiment": "Stage2 A/B/C/D optimizer-induced coupling",
        "parameters": vars(args),
        "train_environments": train_envs,
        "heldout_environments": heldout_envs,
        "map_disjoint": not bool(set(train_envs) & set(heldout_envs)),
        "random_coupling": "fixed cyclic derangement within each condition",
        "training_control": {
            "same_architecture": True,
            "same_parameter_count": True,
            "same_initial_checkpoint": True,
            "same_training_batches": True,
            "FIM": False,
            "dropout": False,
            "weight_decay": 0.0,
        },
        "teacher_trust_region": {
            "max_control_displacement_m": args.max_control_displacement,
            "max_length_ratio": args.safe_length_ratio,
            "topology_side_preserved": True,
            "endpoint_constraint": "structural zero-sum residual decode",
            "curvature_jerk_endpoint_pose": "included in privileged cost",
            "accepted_fraction": float(trust["teacher_valid"].float().mean()),
        },
        "map_config": MAP_CONFIG.to_dict(),
        "safety_cost_config": SAFETY_COST_CONFIG.to_dict(),
        "checkpoint": str(checkpoint_path.resolve()),
        "code_version": git_metadata(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    summary = {
        "elapsed_seconds": time.time() - started,
        "train_samples": len(train_indices),
        "heldout_samples": len(heldout_indices),
        "noises_per_condition": args.noises_per_condition,
        "results": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # Canonical pair data: map/condition/noise/R0/R*/cost/mode/path length.
    all_initial = component_metrics(
        all_data["cp0"],
        tensor_subset(all_data, torch.arange(len(all_data["split_name"]))),
        args,
        device,
    )
    all_teacher = component_metrics(
        cp_star,
        tensor_subset(all_data, torch.arange(len(all_data["split_name"]))),
        args,
        device,
    )
    np.savez_compressed(
        output_dir / "optimizer_pairs.npz",
        maps=all_data["map"].numpy(),
        privileged_cost_maps=all_data["cost_map"].numpy(),
        starts=all_data["start"].numpy(),
        goals=all_data["goal"].numpy(),
        source_noise=all_data["noise"].numpy(),
        stage1_residual=all_data["r0"].numpy(),
        optimized_residual=r_star.numpy(),
        stage1_cost=all_initial["safe"],
        optimized_cost=all_teacher["safe"],
        stage1_mode=all_initial["mode"],
        optimized_mode=all_teacher["mode"],
        stage1_path_length=all_initial["path_length"],
        optimized_path_length=all_teacher["path_length"],
        teacher_valid=trust["teacher_valid"].numpy(),
        teacher_topology_ok=trust["teacher_topology_ok"].numpy(),
        teacher_length_ok=trust["teacher_length_ok"].numpy(),
        teacher_displacement_ok=trust["teacher_displacement_ok"].numpy(),
        teacher_max_control_displacement=trust[
            "teacher_max_control_displacement"
        ].numpy(),
        split=np.asarray(all_data["split_name"]),
        environment=np.asarray(all_data["environment"]),
        path_num=np.asarray(all_data["path_num"]),
        condition_id=all_data["condition_id"].numpy(),
        source_id=all_data["source_id"].numpy(),
    )

    rows: List[Dict[str, object]] = []
    for split_name, indices, split_data in (
        ("train", train_indices, train_data),
        ("heldout", heldout_indices, heldout_data),
    ):
        for local, global_index in enumerate(indices.tolist()):
            row: Dict[str, object] = {
                "split": split_name,
                "environment": all_data["environment"][global_index],
                "path_num": all_data["path_num"][global_index],
                "condition_id": int(all_data["condition_id"][global_index]),
                "source_id": int(all_data["source_id"][global_index]),
                "stage1_safe_cost": float(initial_metrics[split_name]["safe"][local]),
                "teacher_safe_cost": float(teacher_metrics[split_name]["safe"][local]),
                "stage1_mode": int(initial_metrics[split_name]["mode"][local]),
                "teacher_mode": int(teacher_metrics[split_name]["mode"][local]),
                "teacher_valid": bool(trust["teacher_valid"][global_index]),
            }
            for variant in selected_variants:
                metrics = results[variant][f"{split_name}_metrics"]
                row[f"{variant}_safe_cost"] = float(metrics["safe"][local])
                row[f"{variant}_mode"] = int(metrics["mode"][local])
                row[f"{variant}_recovery"] = float(
                    recovery(
                        initial_metrics[split_name]["safe"][local : local + 1],
                        metrics["safe"][local : local + 1],
                        teacher_metrics[split_name]["safe"][local : local + 1],
                    )[0]
                )
            rows.append(row)
    write_csv(output_dir / "per_sample.csv", rows)

    np.savez_compressed(
        output_dir / "model_outputs.npz",
        train_indices=train_indices.numpy(),
        heldout_indices=heldout_indices.numpy(),
        **{
            f"{variant}_{split}_{kind}": results[variant][f"{split}_{kind}"].numpy()
            for variant in selected_variants
            for split in ("train", "heldout")
            for kind in ("r", "cp")
        },
    )

    report_lines = [
        "# Stage-2 coupling A/B/C/D",
        "",
        f"- Train maps: {', '.join(train_envs)}",
        f"- Held-out maps: {', '.join(heldout_envs)}",
        f"- K={args.noises_per_condition}; train/held-out samples: "
        f"{len(train_indices)}/{len(heldout_indices)}",
        f"- Teacher trust-region acceptance: "
        f"{float(trust['teacher_valid'].float().mean()):.1%}",
        "",
        "| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | "
        "Abnormal | Diversity RMS (m) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("A_stage1", "teacher", *selected_variants):
        for split_name in ("train", "heldout"):
            values = summaries[name][split_name]
            report_lines.append(
                f"| {name} | {split_name} | "
                f"{values['recovery_safe']['median']:.3f} | "
                f"{values['mode_preservation_rate']:.1%} | "
                f"{values['safe_at_k']:.1%} | "
                f"{values['abnormal_rate']:.1%} | "
                f"{values['diversity_rms_m']['median']:.3f} |"
            )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    print(f"\nCompleted. Results: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
