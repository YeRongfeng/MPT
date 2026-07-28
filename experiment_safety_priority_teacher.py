#!/usr/bin/env python3
"""Generate a safety-first, topology-preserving privileged trajectory teacher.

The optimizer uses a smooth maximum of per-point yaw-ESDF violations as its
primary objective.  Local deviation, length, curvature, endpoint yaw and jerk
remain secondary.  Its trust region expands only for candidates that have not
yet reached strict safety, and each proposal uses a small number of local
starts.  Candidate selection is lexicographic: strict safety first, then
minimum physical quality cost.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from bspline_utils import DifferentiableBSpline
from dit.Models import PhysicalScaledEdgeResidualRepresentation
from experiment_a_representation import (
    evaluate_control_points,
    git_metadata,
    normalize_poses,
    set_deterministic,
)
from experiment_cplus_overfit import (
    baseline_dense_and_length,
    corrected_objective,
)
from experiment_stage2_coupling import mode_label, side_score
from grad_optimizer import _sample_cost_map_on_dense_trajectory
from map_config import MAP_CONFIG, MAP_HALF_EXTENT, SAFETY_COST_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        default="diagnostics/stage2_scale/N060_v6/optimizer_pairs.npz",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--multi-starts", type=int, default=2)
    parser.add_argument("--steps-per-stage", type=int, default=100)
    parser.add_argument(
        "--trust-displacements",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
    )
    parser.add_argument(
        "--trust-length-ratios",
        type=float,
        nargs="+",
        default=[1.10, 1.17, 1.25],
    )
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.03)
    parser.add_argument("--violation-softplus-beta", type=float, default=30.0)
    parser.add_argument("--violation-lse-beta", type=float, default=20.0)
    parser.add_argument("--safety-weight", type=float, default=10.0)
    parser.add_argument("--quality-weight", type=float, default=0.02)
    parser.add_argument("--perturbation-m", type=float, default=0.15)
    parser.add_argument("--lambda-dev", type=float, default=0.2)
    parser.add_argument("--dev-scale-m", type=float, default=1.0)
    parser.add_argument("--lambda-len", type=float, default=200.0)
    parser.add_argument("--length-ratio-limit", type=float, default=1.15)
    parser.add_argument("--safe-length-ratio", type=float, default=1.17)
    parser.add_argument("--mode-threshold-m", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default="diagnostics/safety_priority_teacher",
    )
    return parser.parse_args()


def load_pairs(path: Path) -> Dict[str, object]:
    with np.load(path) as pairs:
        data: Dict[str, object] = {
            "map": torch.from_numpy(pairs["maps"].copy()),
            "cost_map": torch.from_numpy(
                pairs["privileged_cost_maps"].copy()
            ),
            "start": torch.from_numpy(pairs["starts"].copy()),
            "goal": torch.from_numpy(pairs["goals"].copy()),
            "noise": torch.from_numpy(pairs["source_noise"].copy()),
            "r0": torch.from_numpy(pairs["stage1_residual"].copy()),
            "baseline_r_star": torch.from_numpy(
                pairs["optimized_residual"].copy()
            ),
            "split_name": pairs["split"].astype(str),
            "environment": pairs["environment"].astype(str),
            "path_num": pairs["path_num"].copy(),
            "condition_id": pairs["condition_id"].copy(),
            "source_id": pairs["source_id"].copy(),
        }
    representation = PhysicalScaledEdgeResidualRepresentation()
    start_n = data["start"][:, :2] / MAP_HALF_EXTENT
    goal_n = data["goal"][:, :2] / MAP_HALF_EXTENT
    with torch.no_grad():
        data["cp0"] = (
            representation.decode(data["r0"], start_n, goal_n)
            * MAP_HALF_EXTENT
        )
        data["baseline_cp_star"] = (
            representation.decode(
                data["baseline_r_star"], start_n, goal_n
            )
            * MAP_HALF_EXTENT
        )
    return data


def repeat_starts(value: torch.Tensor, starts: int) -> torch.Tensor:
    return value.repeat_interleave(starts, dim=0)


def initial_middle(
    cp0: torch.Tensor,
    starts: int,
    perturbation_m: float,
    generator: torch.Generator,
) -> torch.Tensor:
    middle = cp0[:, 1:-1].clone()
    if starts <= 1 or perturbation_m <= 0.0:
        return middle
    chord = cp0[:, -1] - cp0[:, 0]
    normal = torch.stack([-chord[:, 1], chord[:, 0]], dim=-1)
    normal = normal / torch.linalg.vector_norm(
        normal, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    progress = torch.linspace(
        0.0, 1.0, middle.shape[1] + 2, device=middle.device
    )[1:-1]
    envelope = torch.sin(math.pi * progress)[None, :, None]
    for start_index in range(1, starts):
        rows = torch.arange(start_index, len(middle), starts, device=middle.device)
        noise = torch.randn(
            len(rows),
            middle.shape[1],
            1,
            device=middle.device,
            generator=generator,
        )
        middle[rows] += (
            perturbation_m
            * envelope
            * torch.tanh(noise)
            * normal[rows, None, :]
        )
    return middle


def strict_metrics(
    control_points: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    cost_maps: torch.Tensor,
    reference_length: torch.Tensor,
    bspline: DifferentiableBSpline,
    safe_length_ratio: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        dense = bspline(control_points)
        _, _, _, esdf = _sample_cost_map_on_dense_trajectory(
            dense, cost_maps, MAP_CONFIG.cost_map_info(), device
        )
        length = torch.linalg.vector_norm(
            dense[:, 1:] - dense[:, :-1], dim=-1
        ).sum(dim=1)
        length_ratio = length / reference_length.clamp_min(1e-6)
        return {
            "strict_safe": (
                (esdf >= 0.0).all(dim=1)
                & (length_ratio <= safe_length_ratio)
            ),
            "min_esdf": esdf.min(dim=1).values,
            "dangerous_count": (esdf < 0.0).sum(dim=1),
            "max_violation": (-esdf).clamp_min(0.0).amax(dim=1),
            "length_ratio": length_ratio,
        }


def safety_objective(
    control_points: torch.Tensor,
    cost_maps: torch.Tensor,
    bspline: DifferentiableBSpline,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dense = bspline(control_points)
    _, _, _, esdf = _sample_cost_map_on_dense_trajectory(
        dense, cost_maps, MAP_CONFIG.cost_map_info(), device
    )
    point_violation = torch.nn.functional.softplus(
        args.violation_softplus_beta * (args.safety_margin - esdf)
    ) / (
        args.violation_softplus_beta
        * SAFETY_COST_CONFIG.d_safe_meters
    )
    beta = args.violation_lse_beta
    smooth_max = (
        torch.logsumexp(beta * point_violation, dim=1)
        - math.log(point_violation.shape[1])
    ) / beta
    return smooth_max, esdf


def project_trust_region(
    middle: torch.Tensor,
    cp0: torch.Tensor,
    reference_length: torch.Tensor,
    original_mode: torch.Tensor,
    original_side: torch.Tensor,
    displacement_limit: torch.Tensor,
    length_limit: torch.Tensor,
    bspline: DifferentiableBSpline,
) -> None:
    with torch.no_grad():
        delta = middle - cp0[:, 1:-1]
        displacement = torch.linalg.vector_norm(
            delta, dim=-1, keepdim=True
        )
        delta *= torch.clamp(
            displacement_limit[:, None, None]
            / displacement.clamp_min(1e-8),
            max=1.0,
        )
        candidate_middle = torch.clamp(
            cp0[:, 1:-1] + delta,
            min=-MAP_HALF_EXTENT + 1e-4,
            max=MAP_HALF_EXTENT - 1e-4,
        )
        delta = candidate_middle - cp0[:, 1:-1]

        low = torch.zeros(len(middle), device=middle.device)
        high = torch.ones_like(low)
        for _ in range(10):
            alpha = 0.5 * (low + high)
            candidate = torch.cat(
                [
                    cp0[:, :1],
                    cp0[:, 1:-1] + alpha[:, None, None] * delta,
                    cp0[:, -1:],
                ],
                dim=1,
            )
            dense = bspline(candidate)
            length = torch.linalg.vector_norm(
                dense[:, 1:] - dense[:, :-1], dim=-1
            ).sum(dim=1)
            feasible = (
                length / reference_length.clamp_min(1e-6)
                <= length_limit
            )
            low = torch.where(feasible, alpha, low)
            high = torch.where(feasible, high, alpha)
        alpha = low

        candidate = torch.cat(
            [
                cp0[:, :1],
                cp0[:, 1:-1] + alpha[:, None, None] * delta,
                cp0[:, -1:],
            ],
            dim=1,
        )
        candidate_side = side_score(candidate)
        crosses = (
            ((original_mode > 0) & (candidate_side <= 0.0))
            | ((original_mode < 0) & (candidate_side >= 0.0))
        )
        crossing_alpha = (
            original_side.abs()
            / (
                original_side.abs() + candidate_side.abs()
            ).clamp_min(1e-8)
            * alpha
            * 0.99
        )
        alpha = torch.where(crosses, crossing_alpha, alpha)
        middle.copy_(
            cp0[:, 1:-1] + alpha[:, None, None] * delta
        )


def optimize_batch(
    data: Dict[str, object],
    indices: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    starts_count = args.multi_starts
    cp0_original = data["cp0"][indices].to(device)
    starts_original = data["start"][indices].to(device)
    goals_original = data["goal"][indices].to(device)
    maps_original = data["cost_map"][indices].to(device)
    cp0 = repeat_starts(cp0_original, starts_count)
    starts = repeat_starts(starts_original, starts_count)
    goals = repeat_starts(goals_original, starts_count)
    cost_maps = repeat_starts(maps_original, starts_count)
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    dense0, reference_length = baseline_dense_and_length(cp0, bspline)
    original_mode = mode_label(cp0, args.mode_threshold_m)
    original_side = side_score(cp0)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + int(indices[0]) * 7919)
    middle = initial_middle(
        cp0, starts_count, args.perturbation_m, generator
    ).requires_grad_(True)
    optimizer = torch.optim.AdamW([middle], lr=args.lr, weight_decay=0.0)
    representation = PhysicalScaledEdgeResidualRepresentation().to(device)

    def encode_control_points(
        control_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_n = normalize_poses(starts, MAP_HALF_EXTENT)
        goal_n = normalize_poses(goals, MAP_HALF_EXTENT)
        residual = representation.encode(
            control_points / MAP_HALF_EXTENT,
            start_n[:, :2],
            goal_n[:, :2],
        )
        residual = representation.radial_project_residual(
            residual, start_n[:, :2], goal_n[:, :2]
        )
        decoded = (
            representation.decode(
                residual, start_n[:, :2], goal_n[:, :2]
            )
            * MAP_HALF_EXTENT
        )
        return residual, decoded

    best_score = torch.full(
        (len(cp0),), float("inf"), device=device
    )
    best_r = torch.empty(
        len(cp0), 25, 2, device=device
    )
    best_cp = torch.empty_like(cp0)

    locked = torch.zeros(len(cp0), dtype=torch.bool, device=device)
    locked_displacement = torch.zeros(len(cp0), device=device)
    locked_length = torch.ones(len(cp0), device=device)
    for stage, (stage_displacement, stage_length) in enumerate(zip(
        args.trust_displacements, args.trust_length_ratios
    )):
        displacement_limit = torch.where(
            locked,
            locked_displacement,
            torch.full_like(locked_displacement, stage_displacement),
        )
        length_limit = torch.where(
            locked,
            locked_length,
            torch.full_like(locked_length, stage_length),
        )
        for _ in range(args.steps_per_stage):
            optimizer.zero_grad(set_to_none=True)
            control_points = torch.cat(
                [cp0[:, :1], middle, cp0[:, -1:]], dim=1
            )
            safety, _ = safety_objective(
                control_points, cost_maps, bspline, args, device
            )
            quality, _ = corrected_objective(
                control_points,
                dense0,
                reference_length,
                starts,
                goals,
                cost_maps,
                bspline,
                args,
                device,
            )
            objective = (
                args.safety_weight * safety
                + args.quality_weight * quality
            )
            objective.sum().backward()
            with torch.no_grad():
                grad_norm = torch.linalg.vector_norm(
                    middle.grad.flatten(start_dim=1), dim=1
                )
                middle.grad *= torch.clamp(
                    args.grad_clip / grad_norm.clamp_min(1e-12),
                    max=1.0,
                )[:, None, None]
            optimizer.step()
            project_trust_region(
                middle,
                cp0,
                reference_length,
                original_mode,
                original_side,
                displacement_limit,
                length_limit,
                bspline,
            )

        with torch.no_grad():
            raw_control_points = torch.cat(
                [cp0[:, :1], middle, cp0[:, -1:]], dim=1
            )
            stage_r, stage_cp = encode_control_points(raw_control_points)
            stage_metrics = strict_metrics(
                stage_cp,
                starts,
                goals,
                cost_maps,
                reference_length,
                bspline,
                args.safe_length_ratio,
                device,
            )
            stage_quality = evaluate_control_points(
                stage_cp,
                starts,
                goals,
                cost_maps,
                MAP_CONFIG.cost_map_info(),
                bspline,
                device,
            )["cost"]
            stage_score = torch.where(
                stage_metrics["strict_safe"],
                stage_quality,
                1e6
                + 1e3 * stage_metrics["max_violation"]
                + stage_quality,
            )
            improved = stage_score < best_score
            best_score[improved] = stage_score[improved]
            best_r[improved] = stage_r[improved]
            best_cp[improved] = stage_cp[improved]
        newly_safe = stage_metrics["strict_safe"] & ~locked
        if newly_safe.any():
            actual_displacement = torch.linalg.vector_norm(
                raw_control_points - cp0, dim=-1
            ).amax(dim=1)
            locked_displacement[newly_safe] = (
                actual_displacement[newly_safe] + 1e-3
            )
            locked_length[newly_safe] = torch.minimum(
                stage_metrics["length_ratio"][newly_safe] + 1e-3,
                torch.full_like(
                    stage_metrics["length_ratio"][newly_safe],
                    args.safe_length_ratio,
                ),
            )
            locked |= newly_safe
        print(
            f"  batch {int(indices[0])}:{int(indices[-1]) + 1}, "
            f"stage {stage + 1}, safe starts="
            f"{int(stage_metrics['strict_safe'].sum())}/{len(cp0)}"
        )

    with torch.no_grad():
        residual = best_r
        control_points = best_cp
        metrics = evaluate_control_points(
            control_points,
            starts,
            goals,
            cost_maps,
            MAP_CONFIG.cost_map_info(),
            bspline,
            device,
        )
        _, _, _, esdf = _sample_cost_map_on_dense_trajectory(
            metrics["trajectory"],
            cost_maps,
            MAP_CONFIG.cost_map_info(),
            device,
        )
        length_ratio = metrics["path_length"] / reference_length
        strict_safe = (
            (esdf >= 0.0).all(dim=1)
            & (length_ratio <= args.safe_length_ratio)
        )
        max_violation = (-esdf).clamp_min(0.0).amax(dim=1)
        # Strict safety dominates selection.  Among safe starts choose the
        # lowest physical cost; otherwise choose minimum max violation.
        score = torch.where(
            strict_safe,
            metrics["cost"],
            1e6 + 1e3 * max_violation + metrics["cost"],
        )
        score = score.view(len(indices), starts_count)
        chosen = score.argmin(dim=1)
        flat = (
            torch.arange(len(indices), device=device) * starts_count
            + chosen
        )
        selected_cp = control_points[flat]
        selected_r = residual[flat]
        selected_esdf = esdf[flat]
        selected_length_ratio = length_ratio[flat]
        return {
            "r_star": selected_r.cpu(),
            "cp_star": selected_cp.cpu(),
            "safe_cost": metrics["cost"][flat].cpu(),
            "strict_safe": strict_safe[flat].cpu(),
            "min_esdf": selected_esdf.min(dim=1).values.cpu(),
            "dangerous_count": (selected_esdf < 0.0).sum(dim=1).cpu(),
            "max_violation": (
                -selected_esdf
            ).clamp_min(0.0).amax(dim=1).cpu(),
            "length_ratio": selected_length_ratio.cpu(),
            "mode": mode_label(
                selected_cp, args.mode_threshold_m
            ).cpu(),
            "max_displacement": torch.linalg.vector_norm(
                selected_cp - cp0_original, dim=-1
            ).amax(dim=1).cpu(),
            "chosen_start": chosen.cpu(),
        }


def safe_at_k(
    safe: np.ndarray,
    condition_id: np.ndarray,
    split: np.ndarray,
    split_name: str,
) -> float:
    selected = split == split_name
    conditions = np.unique(condition_id[selected])
    return float(np.mean([
        np.any(safe[(condition_id == condition) & selected])
        for condition in conditions
    ]))


def main() -> None:
    args = parse_args()
    if len(args.trust_displacements) != len(args.trust_length_ratios):
        raise ValueError("Trust displacement/length schedules must match")
    if args.multi_starts < 1:
        raise ValueError("--multi-starts must be positive")
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_pairs(Path(args.pairs))
    started = time.time()

    chunks: Dict[str, List[torch.Tensor]] = {}
    for begin in range(0, len(data["split_name"]), args.batch_size):
        indices = torch.arange(
            begin, min(begin + args.batch_size, len(data["split_name"]))
        )
        result = optimize_batch(data, indices, args, device)
        for key, value in result.items():
            chunks.setdefault(key, []).append(value)
    result = {key: torch.cat(value) for key, value in chunks.items()}

    split = data["split_name"]
    safe = result["strict_safe"].numpy()
    summary = {
        "experiment": "safety-priority topology-preserving teacher",
        "elapsed_seconds": time.time() - started,
        "train_safe_at_k": safe_at_k(
            safe, data["condition_id"], split, "train"
        ),
        "heldout_safe_at_k": safe_at_k(
            safe, data["condition_id"], split, "heldout"
        ),
        "train_safe_candidate_rate": float(np.mean(safe[split == "train"])),
        "heldout_safe_candidate_rate": float(
            np.mean(safe[split == "heldout"])
        ),
        "heldout_min_esdf_median": float(
            result["min_esdf"][split == "heldout"].median()
        ),
        "heldout_dangerous_count_median": float(
            result["dangerous_count"][split == "heldout"].float().median()
        ),
        "parameters": vars(args),
        "pairs": str(Path(args.pairs).resolve()),
        "map_config": MAP_CONFIG.to_dict(),
        "safety_cost_config": SAFETY_COST_CONFIG.to_dict(),
        "code_version": git_metadata(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    np.savez_compressed(
        output_dir / "safety_priority_teacher.npz",
        **{key: value.numpy() for key, value in result.items()},
        split=split,
        environment=data["environment"],
        path_num=data["path_num"],
        condition_id=data["condition_id"],
        source_id=data["source_id"],
    )
    report = [
        "# Safety-priority privileged teacher",
        "",
        f"- Train safe@K: {summary['train_safe_at_k']:.1%}",
        f"- Held-out safe@K: {summary['heldout_safe_at_k']:.1%}",
        f"- Train safe candidate rate: "
        f"{summary['train_safe_candidate_rate']:.1%}",
        f"- Held-out safe candidate rate: "
        f"{summary['heldout_safe_candidate_rate']:.1%}",
        f"- Held-out min ESDF median: "
        f"{summary['heldout_min_esdf_median']:.3f} m",
        f"- Held-out dangerous point count median: "
        f"{summary['heldout_dangerous_count_median']:.1f}",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"Completed: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
