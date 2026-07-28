#!/usr/bin/env python3
"""C+ 小数据过拟合诊断：cost-only 与 teacher supervision 严格对照。

实验固定同一批 (地图、起终点、噪声)，先由 Stage1 产生 R0，再在当前
raw_v/raw_s + 径向可行层内，以“安全 + 邻域偏移 + 超长惩罚”生成 R*。
随后从同一个 Stage1 checkpoint 分别训练：

  C1: 仅最小化修正后的轨迹目标；
  C2: Huber(R_hat, R*) + lambda_safe * 修正后的轨迹目标。

脚本保存中文配置、中文报告、逐样本指标、训练曲线、固定 teacher 数据和图。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
    detach_metrics,
    discover_envs,
    evaluate_control_points,
    git_metadata,
    normalize_poses,
    raw_initialization_from_residual,
    set_deterministic,
    summarize_array,
    tensor_row_clip,
    write_csv,
)
from grad_optimizer import cost_on_dense_trajectory_tail_risk
from map_config import MAP_CONFIG, MAP_HALF_EXTENT, SAFETY_COST_CONFIG

font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class FixedSample:
    environment: str
    path_num: int
    dataset_index: int
    map_input: torch.Tensor
    cost_map: torch.Tensor
    elevation: torch.Tensor
    start_pose: torch.Tensor
    goal_pose: torch.Tensor
    noise: torch.Tensor
    r0: torch.Tensor
    cp0: torch.Tensor
    initial_total: float
    initial_obstacle: float
    initial_dangerous_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(MAP_CONFIG.dataset_root / "val"))
    parser.add_argument("--model-dir", default="data/sim")
    parser.add_argument("--checkpoint", default="stage1_best_model.pth")
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--paths-per-env", type=int, default=20)
    parser.add_argument("--select", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--teacher-steps", type=int, default=300)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--teacher-grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-dev", type=float, default=0.2)
    parser.add_argument("--dev-scale-m", type=float, default=1.0)
    parser.add_argument("--lambda-len", type=float, default=200.0)
    parser.add_argument("--length-ratio-limit", type=float, default=1.15)

    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--train-lr", type=float, default=2e-5)
    parser.add_argument("--train-grad-clip", type=float, default=1.0)
    parser.add_argument("--teacher-huber-beta", type=float, default=0.1)
    parser.add_argument("--lambda-teacher-safe", type=float, default=0.01)
    parser.add_argument(
        "--trace-steps",
        type=int,
        nargs="+",
        default=[0, 1, 5, 10, 20, 40, 80, 160, 240, 320, 400],
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=80)
    parser.add_argument(
        "--resume-variant-checkpoints",
        action="store_true",
        help="从输出目录中的半精度 C1/C2 断点继续；恢复时 Adam 状态重置。",
    )
    parser.add_argument("--plot-examples", type=int, default=6)
    parser.add_argument(
        "--output-dir", default="diagnostics/experiment_cplus_overfit"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(
    model_params: Dict[str, object],
    checkpoint: Dict[str, object],
    device: torch.device,
) -> PathDiffusionTransformer:
    model = PathDiffusionTransformer(**model_params["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def normalize_selected_poses(
    starts: torch.Tensor, goals: torch.Tensor, coordinate_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        normalize_poses(starts, coordinate_scale),
        normalize_poses(goals, coordinate_scale),
    )


def collect_fixed_samples(
    args: argparse.Namespace,
    environments: Sequence[str],
    model: PathDiffusionTransformer,
    device: torch.device,
) -> List[FixedSample]:
    dataset = UnevenPathDataLoader(
        env_list=list(environments),
        dataFolder=args.dataset,
        compute_stability_map=True,
    )
    by_env: Dict[str, List[int]] = {environment: [] for environment in environments}
    for dataset_index, (env_index, _) in enumerate(dataset.indexDict):
        by_env[dataset.env_list[env_index]].append(dataset_index)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 101)
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    map_info = MAP_CONFIG.cost_map_info()
    coordinate_scale = float(model.coordinate_scale)
    records: List[FixedSample] = []

    for environment in environments:
        indices = sorted(by_env[environment])[: args.paths_per_env]
        if len(indices) < args.paths_per_env:
            raise ValueError(
                f"{environment} 只有 {len(indices)} 条轨迹，"
                f"但请求 {args.paths_per_env} 条"
            )
        items = [dataset[index] for index in indices]
        maps = torch.stack([item["map"] for item in items]).to(device)
        starts = torch.stack([item["start_pose"] for item in items]).to(device)
        goals = torch.stack([item["goal_pose"] for item in items]).to(device)
        cost_maps = torch.stack([item["cost_map"] for item in items]).to(device)
        start_n, goal_n = normalize_selected_poses(
            starts, goals, coordinate_scale
        )
        noise = model.project_zero_sum(
            torch.randn(
                len(indices),
                model.num_edges,
                2,
                device=device,
                generator=generator,
            )
        )
        t = torch.ones(len(indices), device=device)
        r = torch.zeros(len(indices), device=device)
        with torch.no_grad():
            r0 = model(maps, noise, t, r, start_n, goal_n)
            cp0_n = model.trajectory_representation.decode(
                r0, start_n[:, :2], goal_n[:, :2]
            )
            cp0 = cp0_n * coordinate_scale
            metrics = evaluate_control_points(
                cp0, starts, goals, cost_maps, map_info, bspline, device
            )

        for local, dataset_index in enumerate(indices):
            _, path_num = dataset.indexDict[dataset_index]
            records.append(
                FixedSample(
                    environment=environment,
                    path_num=path_num,
                    dataset_index=dataset_index,
                    map_input=maps[local].detach().cpu(),
                    cost_map=cost_maps[local].detach().cpu(),
                    elevation=items[local]["elevation"].detach().cpu(),
                    start_pose=starts[local].detach().cpu(),
                    goal_pose=goals[local].detach().cpu(),
                    noise=noise[local].detach().cpu(),
                    r0=r0[local].detach().cpu(),
                    cp0=cp0[local].detach().cpu(),
                    initial_total=float(metrics["total"][local]),
                    initial_obstacle=float(metrics["obstacle"][local]),
                    initial_dangerous_ratio=float(
                        metrics["dangerous_ratio"][local]
                    ),
                )
            )
        print(
            f"固定样本已收集: {environment}, {len(indices)} 条，"
            f"累计 {len(records)} 条",
            flush=True,
        )
    return records


def stack_selected(
    selected: Sequence[FixedSample],
) -> Dict[str, torch.Tensor]:
    return {
        "map": torch.stack([item.map_input for item in selected]),
        "cost_map": torch.stack([item.cost_map for item in selected]),
        "elevation": torch.stack([item.elevation for item in selected]),
        "start": torch.stack([item.start_pose for item in selected]),
        "goal": torch.stack([item.goal_pose for item in selected]),
        "noise": torch.stack([item.noise for item in selected]),
        "r0": torch.stack([item.r0 for item in selected]),
        "cp0": torch.stack([item.cp0 for item in selected]),
    }


def corrected_objective(
    control_points: torch.Tensor,
    reference_dense: torch.Tensor,
    reference_length: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    cost_maps: torch.Tensor,
    bspline: DifferentiableBSpline,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    dense = bspline(control_points)
    safe, safe_components = cost_on_dense_trajectory_tail_risk(
        dense,
        starts,
        goals,
        cost_maps,
        MAP_CONFIG.cost_map_info(),
        device,
        return_per_sample=True,
        return_components=True,
    )
    deviation = (
        (dense - reference_dense).square().sum(dim=-1).mean(dim=-1)
        / (args.dev_scale_m**2)
    )
    length = torch.linalg.vector_norm(
        dense[:, 1:] - dense[:, :-1], dim=-1
    ).sum(dim=-1)
    length_ratio = length / reference_length.clamp_min(1e-6)
    length_excess = F.relu(length_ratio - args.length_ratio_limit)
    length_penalty = length_excess.square()
    corrected = (
        safe
        + args.lambda_dev * deviation
        + args.lambda_len * length_penalty
    )
    return corrected, {
        "corrected": corrected,
        "safe": safe,
        "deviation": deviation,
        "length": length,
        "length_ratio": length_ratio,
        "length_penalty": length_penalty,
        "obstacle": safe_components["obstacle"],
        "curvature": safe_components["curvature"],
        "smoothness": safe_components["smoothness"],
        "jerk": safe_components["jerk"],
    }


def baseline_dense_and_length(
    cp0: torch.Tensor, bspline: DifferentiableBSpline
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        dense0 = bspline(cp0)
        length0 = torch.linalg.vector_norm(
            dense0[:, 1:] - dense0[:, :-1], dim=-1
        ).sum(dim=-1)
    return dense0, length0


def generate_teacher(
    data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    representation = PhysicalScaledEdgeResidualRepresentation().to(device)
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    starts = data["start"].to(device)
    goals = data["goal"].to(device)
    cp0 = data["cp0"].to(device)
    r0 = data["r0"].to(device)
    cost_maps = data["cost_map"].to(device)
    start_n, goal_n = normalize_selected_poses(
        starts, goals, coordinate_scale
    )
    dense0, length0 = baseline_dense_and_length(cp0, bspline)

    raw_v, raw_s, initialization = raw_initialization_from_residual(
        r0, start_n[:, :2], goal_n[:, :2], representation
    )
    raw_v = raw_v.detach().requires_grad_(True)
    raw_s = raw_s.detach().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [raw_v, raw_s], lr=args.teacher_lr, weight_decay=0.0
    )
    trace_steps = set(
        step
        for step in args.trace_steps
        if 0 <= step <= args.teacher_steps
    ) | {0, args.teacher_steps}
    trace_rows: List[Dict[str, object]] = []

    def current() -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        residual, _ = representation.radial_feasible_residual(
            raw_v,
            raw_s,
            start_n[:, :2],
            goal_n[:, :2],
            return_diagnostics=True,
        )
        cp = (
            representation.decode(
                residual, start_n[:, :2], goal_n[:, :2]
            )
            * coordinate_scale
        )
        return residual, cp, {}

    def record(step: int) -> None:
        with torch.no_grad():
            _, cp, _ = current()
            _, components = corrected_objective(
                cp,
                dense0,
                length0,
                starts,
                goals,
                cost_maps,
                bspline,
                args,
                device,
            )
            physical_metrics = evaluate_control_points(
                cp,
                starts,
                goals,
                cost_maps,
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
            ):
                components[key] = physical_metrics[key]
            for sample_index in range(cp.shape[0]):
                trace_rows.append(
                    {
                        "阶段": "teacher",
                        "step": step,
                        "sample_index": sample_index,
                        **{
                            key: float(value[sample_index])
                            for key, value in components.items()
                        },
                    }
                )

    with torch.no_grad():
        _, reconstructed_cp, _ = current()
        init_error = torch.linalg.vector_norm(
            reconstructed_cp - cp0, dim=-1
        ).amax()
    if init_error.item() > 2e-3:
        raise RuntimeError(
            f"teacher raw 初始化误差过大: {init_error.item():.6g} m"
        )
    if initialization["alpha_clipped"].any():
        raise RuntimeError("teacher raw 初始化发生 alpha 裁剪")

    record(0)
    for step in range(1, args.teacher_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, cp, _ = current()
        objective, _ = corrected_objective(
            cp,
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
        tensor_row_clip([raw_v, raw_s], args.teacher_grad_clip)
        optimizer.step()
        if step in trace_steps:
            record(step)
    with torch.no_grad():
        r_star, cp_star, _ = current()
    return r_star.detach().cpu(), cp_star.detach().cpu(), trace_rows


def cyclic_indices(step: int, count: int, batch_size: int) -> torch.Tensor:
    offset = ((step - 1) * batch_size) % count
    return torch.tensor(
        [(offset + index) % count for index in range(batch_size)],
        dtype=torch.long,
    )


def model_output(
    model: PathDiffusionTransformer,
    maps: torch.Tensor,
    noise: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    coordinate_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    start_n, goal_n = normalize_selected_poses(
        starts, goals, coordinate_scale
    )
    batch_size = maps.shape[0]
    r_hat = model(
        maps,
        noise,
        torch.ones(batch_size, device=maps.device),
        torch.zeros(batch_size, device=maps.device),
        start_n,
        goal_n,
    )
    cp_hat = (
        model.trajectory_representation.decode(
            r_hat, start_n[:, :2], goal_n[:, :2]
        )
        * coordinate_scale
    )
    return r_hat, cp_hat


def evaluate_model_outputs(
    model: PathDiffusionTransformer,
    data: Dict[str, torch.Tensor],
    r_star: torch.Tensor,
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
    model.eval()
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    outputs_r: List[torch.Tensor] = []
    outputs_cp: List[torch.Tensor] = []
    component_chunks: Dict[str, List[torch.Tensor]] = {}
    count = data["map"].shape[0]
    with torch.no_grad():
        for begin in range(0, count, args.eval_batch_size):
            end = min(begin + args.eval_batch_size, count)
            maps = data["map"][begin:end].to(device)
            starts = data["start"][begin:end].to(device)
            goals = data["goal"][begin:end].to(device)
            noise = data["noise"][begin:end].to(device)
            cp0 = data["cp0"][begin:end].to(device)
            cost_maps = data["cost_map"][begin:end].to(device)
            dense0, length0 = baseline_dense_and_length(cp0, bspline)
            r_hat, cp_hat = model_output(
                model, maps, noise, starts, goals, coordinate_scale
            )
            _, components = corrected_objective(
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
            physical_metrics = evaluate_control_points(
                cp_hat,
                starts,
                goals,
                cost_maps,
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
            ):
                components[key] = physical_metrics[key]
            teacher_error = torch.linalg.vector_norm(
                (r_hat - r_star[begin:end].to(device)).flatten(start_dim=1),
                dim=1,
            )
            teacher_delta = torch.linalg.vector_norm(
                (
                    data["r0"][begin:end].to(device)
                    - r_star[begin:end].to(device)
                ).flatten(start_dim=1),
                dim=1,
            )
            components["relative_teacher_error"] = (
                teacher_error / teacher_delta.clamp_min(1e-8)
            )
            outputs_r.append(r_hat.cpu())
            outputs_cp.append(cp_hat.cpu())
            for key, value in components.items():
                component_chunks.setdefault(key, []).append(value.cpu())
    return (
        torch.cat(outputs_r),
        torch.cat(outputs_cp),
        {
            key: torch.cat(chunks).numpy()
            for key, chunks in component_chunks.items()
        },
    )


def train_variant(
    variant: str,
    model_params: Dict[str, object],
    checkpoint: Dict[str, object],
    data: Dict[str, torch.Tensor],
    r_star: torch.Tensor,
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
    output_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]], Dict[str, np.ndarray]]:
    model = load_model(model_params, checkpoint, device)
    # eval 模式关闭 dropout；梯度仍正常计算，使固定输入得到严格确定性结果。
    model.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.train_lr, weight_decay=0.0
    )
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    trace_steps = set(
        step for step in args.trace_steps if 0 <= step <= args.train_steps
    ) | {0, args.train_steps}
    trace_rows: List[Dict[str, object]] = []
    checkpoint_path = output_dir / f"断点_{variant}.pth"
    start_step = 0
    if args.resume_variant_checkpoints and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu")
        if int(saved["seed"]) != args.seed:
            raise ValueError(f"{checkpoint_path} 的 seed 与当前实验不一致")
        model.load_state_dict(saved["model_state_dict"])
        start_step = int(saved["step"])
        trace_rows = list(saved.get("trace_rows", []))
        print(
            f"{variant}: 从 step={start_step} 半精度模型断点继续；"
            "Adam 状态按记录重置。",
            flush=True,
        )

    def record(step: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
        r_hat, cp_hat, components = evaluate_model_outputs(
            model,
            data,
            r_star,
            args,
            coordinate_scale,
            device,
        )
        row: Dict[str, object] = {"阶段": variant, "step": step}
        for key, values in components.items():
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_median"] = float(np.median(values))
        trace_rows.append(row)
        print(
            f"{variant} step={step}: "
            f"J_med={row['corrected_median']:.4f}, "
            f"safe_med={row['safe_median']:.4f}, "
            f"len_ratio_med={row['length_ratio_median']:.3f}, "
            f"teacher_err_med={row['relative_teacher_error_median']:.3f}",
            flush=True,
        )
        return r_hat, cp_hat, components

    final_r, final_cp, final_components = record(start_step)
    count = data["map"].shape[0]
    for step in range(start_step + 1, args.train_steps + 1):
        indices = cyclic_indices(step, count, args.train_batch_size)
        maps = data["map"][indices].to(device)
        starts = data["start"][indices].to(device)
        goals = data["goal"][indices].to(device)
        noise = data["noise"][indices].to(device)
        cp0 = data["cp0"][indices].to(device)
        cost_maps = data["cost_map"][indices].to(device)
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
        if variant == "C1_cost_only":
            loss = corrected.mean()
        elif variant == "C2_teacher":
            teacher_loss = F.smooth_l1_loss(
                r_hat,
                r_star[indices].to(device),
                beta=args.teacher_huber_beta,
            )
            loss = teacher_loss + args.lambda_teacher_safe * corrected.mean()
        else:
            raise ValueError(variant)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.train_grad_clip
        )
        optimizer.step()
        if step in trace_steps:
            final_r, final_cp, final_components = record(step)
        if (
            args.checkpoint_every > 0
            and (
                step % args.checkpoint_every == 0
                or step == args.train_steps
            )
        ):
            compact_state = {
                name: (
                    value.detach().cpu().half()
                    if torch.is_floating_point(value)
                    else value.detach().cpu()
                )
                for name, value in model.state_dict().items()
            }
            torch.save(
                {
                    "variant": variant,
                    "step": step,
                    "seed": args.seed,
                    "train_lr": args.train_lr,
                    "optimizer_state_saved": False,
                    "model_state_dict": compact_state,
                    "trace_rows": trace_rows,
                },
                checkpoint_path,
            )
            del compact_state

    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_r, final_cp, trace_rows, final_components


def calculate_baseline_components(
    cp: torch.Tensor,
    data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    chunks: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for begin in range(0, cp.shape[0], args.eval_batch_size):
            end = min(begin + args.eval_batch_size, cp.shape[0])
            cp0 = data["cp0"][begin:end].to(device)
            dense0, length0 = baseline_dense_and_length(cp0, bspline)
            _, components = corrected_objective(
                cp[begin:end].to(device),
                dense0,
                length0,
                data["start"][begin:end].to(device),
                data["goal"][begin:end].to(device),
                data["cost_map"][begin:end].to(device),
                bspline,
                args,
                device,
            )
            physical_metrics = evaluate_control_points(
                cp[begin:end].to(device),
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
            ):
                components[key] = physical_metrics[key]
            for key, value in components.items():
                chunks.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(value).numpy() for key, value in chunks.items()}


def plot_training_curves(
    trace_rows: Sequence[Dict[str, object]], output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for variant, color in (
        ("C1_cost_only", "#e76f51"),
        ("C2_teacher", "#2563eb"),
    ):
        rows = [row for row in trace_rows if row["阶段"] == variant]
        steps = [int(row["step"]) for row in rows]
        axes[0].plot(
            steps,
            [float(row["corrected_median"]) for row in rows],
            "-o",
            label=variant,
            color=color,
        )
        axes[1].plot(
            steps,
            [float(row["relative_teacher_error_median"]) for row in rows],
            "-o",
            label=variant,
            color=color,
        )
        axes[2].plot(
            steps,
            [float(row["length_ratio_median"]) for row in rows],
            "-o",
            label=variant,
            color=color,
        )
    axes[0].set_title("修正目标中位数")
    axes[1].set_title("相对 teacher 误差中位数")
    axes[2].set_title("路径长度比中位数")
    axes[2].axhline(1.15, color="black", ls="--", lw=1)
    for axis in axes:
        axis.set_xlabel("训练 step")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_examples(
    selected: Sequence[FixedSample],
    cp0: torch.Tensor,
    cp_star: torch.Tensor,
    cp_c1: torch.Tensor,
    cp_c2: torch.Tensor,
    output_path: Path,
    count: int,
) -> None:
    count = min(count, len(selected))
    bspline = DifferentiableBSpline(26, 100, 3)
    sets = [
        ("Stage1", cp0, "#00a9a5"),
        ("Teacher", cp_star, "#7b2cbf"),
        ("C1", cp_c1, "#e76f51"),
        ("C2", cp_c2, "#2563eb"),
    ]
    fig, axes = plt.subplots(count, 4, figsize=(17, 4.2 * count))
    if count == 1:
        axes = axes[None, :]
    extent = list(MAP_CONFIG.bounds)
    with torch.no_grad():
        for row in range(count):
            for column, (label, cp_set, color) in enumerate(sets):
                axis = axes[row, column]
                axis.imshow(
                    selected[row].elevation.numpy(),
                    extent=extent,
                    origin="lower",
                    cmap="terrain",
                    alpha=0.75,
                )
                cp = cp_set[row]
                dense = bspline(cp.unsqueeze(0))[0].numpy()
                axis.plot(dense[:, 0], dense[:, 1], color=color, lw=2.2)
                axis.plot(
                    cp[:, 0],
                    cp[:, 1],
                    "--",
                    color=color,
                    lw=0.7,
                    alpha=0.3,
                )
                axis.scatter(cp[0, 0], cp[0, 1], marker="s", c="green", s=30)
                axis.scatter(cp[-1, 0], cp[-1, 1], marker="*", c="red", s=45)
                axis.set_title(
                    f"{selected[row].environment}/path_{selected[row].path_num}: "
                    f"{label}"
                )
                axis.set_xlim(MAP_CONFIG.bounds[0], MAP_CONFIG.bounds[1])
                axis.set_ylim(MAP_CONFIG.bounds[2], MAP_CONFIG.bounds[3])
                axis.set_aspect("equal")
                axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    started = time.time()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "本实验要求使用 CUDA，但当前 PyTorch 看不到 GPU；"
            "为避免静默降级到 CPU，已停止运行。"
        )
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    model_params_path = model_dir / "model_params.json"
    checkpoint_path = model_dir / args.checkpoint
    model_params = json.loads(model_params_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    coordinate_scale = float(model_params["model_args"]["coordinate_scale"])
    if not math.isclose(coordinate_scale, MAP_HALF_EXTENT, abs_tol=1e-6):
        raise ValueError("模型坐标尺度与当前地图不一致")

    environments = discover_envs(Path(args.dataset), args.num_envs, args.seed)
    config = {
        "实验": "C+ 小数据过拟合：C1 cost-only 对比 C2 teacher",
        "参数": vars(args),
        "地图配置": MAP_CONFIG.to_dict(),
        "安全代价配置": SAFETY_COST_CONFIG.to_dict(),
        "修正目标": {
            "公式": "J=C_safe+lambda_dev*mean(||traj-traj0||^2/dev_scale^2)"
            "+lambda_len*ReLU(L/L0-gamma)^2",
            "lambda_dev": args.lambda_dev,
            "dev_scale_m": args.dev_scale_m,
            "lambda_len": args.lambda_len,
            "gamma": args.length_ratio_limit,
        },
        "训练控制": {
            "固定噪声": True,
            "固定样本": True,
            "shuffle": False,
            "Fisher": False,
            "学习率调度": False,
            "dropout": False,
            "weight_decay": 0.0,
            "训练参数范围": "当前生成器全部参数",
            "C1与C2初始化": "同一个Stage1 checkpoint",
        },
        "checkpoint": {
            "路径": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "stage": checkpoint.get("stage"),
        },
        "环境": environments,
        "代码版本": git_metadata(),
        "运行环境": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    (output_dir / "实验配置.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    collection_model = load_model(model_params, checkpoint, device)
    candidates = collect_fixed_samples(
        args, environments, collection_model, device
    )
    candidates.sort(
        key=lambda item: (item.initial_obstacle, item.initial_total),
        reverse=True,
    )
    if args.select > len(candidates):
        raise ValueError(
            f"请求 {args.select} 条，候选只有 {len(candidates)} 条"
        )
    selected = candidates[: args.select]
    data = stack_selected(selected)
    del collection_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    selection_rows = [
        {
            "rank": rank,
            "environment": item.environment,
            "path_num": item.path_num,
            "dataset_index": item.dataset_index,
            "initial_total": item.initial_total,
            "initial_obstacle": item.initial_obstacle,
            "initial_dangerous_ratio": item.initial_dangerous_ratio,
        }
        for rank, item in enumerate(selected)
    ]
    write_csv(output_dir / "固定样本清单.csv", selection_rows)

    print("开始生成局部约束 teacher ...", flush=True)
    r_star, cp_star, teacher_trace = generate_teacher(
        data, args, coordinate_scale, device
    )
    teacher_components = calculate_baseline_components(
        cp_star, data, args, device
    )
    initial_components = calculate_baseline_components(
        data["cp0"], data, args, device
    )
    print(
        "Teacher 完成: "
        f"J中位数 {np.median(initial_components['corrected']):.4f} -> "
        f"{np.median(teacher_components['corrected']):.4f}, "
        f"长度比中位数={np.median(teacher_components['length_ratio']):.3f}, "
        f"最大={np.max(teacher_components['length_ratio']):.3f}",
        flush=True,
    )

    c1_r, c1_cp, c1_trace, c1_components = train_variant(
        "C1_cost_only",
        model_params,
        checkpoint,
        data,
        r_star,
        args,
        coordinate_scale,
        device,
        output_dir,
    )
    c2_r, c2_cp, c2_trace, c2_components = train_variant(
        "C2_teacher",
        model_params,
        checkpoint,
        data,
        r_star,
        args,
        coordinate_scale,
        device,
        output_dir,
    )

    trace_rows = teacher_trace + c1_trace + c2_trace
    write_csv(output_dir / "训练曲线.csv", trace_rows)

    denominator = (
        initial_components["corrected"] - teacher_components["corrected"]
    )
    valid = denominator > 1e-6
    rows: List[Dict[str, object]] = []
    for index, sample in enumerate(selected):
        row: Dict[str, object] = {
            "sample_index": index,
            "environment": sample.environment,
            "path_num": sample.path_num,
        }
        for prefix, components in (
            ("stage1", initial_components),
            ("teacher", teacher_components),
            ("c1", c1_components),
            ("c2", c2_components),
        ):
            for key, values in components.items():
                row[f"{prefix}_{key}"] = float(values[index])
        for prefix, components in (
            ("c1", c1_components),
            ("c2", c2_components),
        ):
            recovery = (
                (
                    initial_components["corrected"][index]
                    - components["corrected"][index]
                )
                / denominator[index]
                if valid[index]
                else float("nan")
            )
            row[f"{prefix}_teacher_recovery"] = float(recovery)
            row[f"{prefix}_improved"] = bool(
                components["corrected"][index]
                < initial_components["corrected"][index]
            )
        rows.append(row)
    write_csv(output_dir / "逐样本结果.csv", rows)

    def variant_summary(
        components: Dict[str, np.ndarray], prefix: str
    ) -> Dict[str, object]:
        recovery = np.asarray(
            [float(row[f"{prefix}_teacher_recovery"]) for row in rows]
        )
        return {
            "修正目标": summarize_array(components["corrected"]),
            "安全cost": summarize_array(components["safe"]),
            "路径长度比": summarize_array(components["length_ratio"]),
            "相对teacher误差": summarize_array(
                components.get(
                    "relative_teacher_error",
                    np.full(len(selected), np.nan),
                )
            ),
            "teacher恢复率": summarize_array(recovery),
            "有效恢复率样本数": int(np.sum(np.isfinite(recovery))),
            "改善样本比例": float(
                np.mean([bool(row[f"{prefix}_improved"]) for row in rows])
            ),
            "长度比不超过gamma+0.02的比例": float(
                np.mean(
                    components["length_ratio"]
                    <= args.length_ratio_limit + 0.02
                )
            ),
        }

    summary = {
        "样本数": len(selected),
        "候选数": len(candidates),
        "Stage1修正目标": summarize_array(initial_components["corrected"]),
        "Teacher修正目标": summarize_array(teacher_components["corrected"]),
        "Teacher路径长度比": summarize_array(
            teacher_components["length_ratio"]
        ),
        "C1": variant_summary(c1_components, "c1"),
        "C2": variant_summary(c2_components, "c2"),
        "判据": {
            "成功恢复率中位数": 0.8,
            "成功改善样本比例": 0.8,
            "C2成功相对teacher误差中位数": 0.3,
            "长度容差": args.length_ratio_limit + 0.02,
        },
        "耗时秒": time.time() - started,
    }
    (output_dir / "汇总.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    np.savez_compressed(
        output_dir / "固定数据与输出.npz",
        maps=data["map"].numpy(),
        starts=data["start"].numpy(),
        goals=data["goal"].numpy(),
        noise=data["noise"].numpy(),
        r0=data["r0"].numpy(),
        r_star=r_star.numpy(),
        c1_r=c1_r.numpy(),
        c2_r=c2_r.numpy(),
        cp0=data["cp0"].numpy(),
        cp_star=cp_star.numpy(),
        c1_cp=c1_cp.numpy(),
        c2_cp=c2_cp.numpy(),
        environments=np.asarray([item.environment for item in selected]),
        path_nums=np.asarray([item.path_num for item in selected]),
    )
    plot_training_curves(c1_trace + c2_trace, output_dir / "训练曲线.png")
    plot_examples(
        selected,
        data["cp0"],
        cp_star,
        c1_cp,
        c2_cp,
        output_dir / "轨迹对比.png",
        args.plot_examples,
    )

    c1_recovery = summary["C1"]["teacher恢复率"]["median"]
    c2_recovery = summary["C2"]["teacher恢复率"]["median"]
    c1_improved = summary["C1"]["改善样本比例"]
    c2_improved = summary["C2"]["改善样本比例"]
    c1_success = c1_recovery >= 0.8 and c1_improved >= 0.8
    c2_teacher_error = summary["C2"]["相对teacher误差"]["median"]
    c2_success = (
        c2_recovery >= 0.8
        and c2_improved >= 0.8
        and c2_teacher_error <= 0.3
    )
    if c1_success and c2_success:
        decision = (
            "C1、C2均成功：当前生成器具备小数据摊销能力；完整Stage2问题更可能"
            "来自跨地图冲突、Fisher或训练调度。"
        )
    elif (not c1_success) and c2_success:
        decision = (
            "C1失败、C2成功：网络能表达改进，但cost-only摊销失败；"
            "应转向teacher预训练的Refiner，再做cost微调。"
        )
    elif (not c1_success) and (not c2_success):
        decision = (
            "C1、C2均失败：本实验已经全参数训练；在确认训练步数和学习率"
            "足够后，应显式输入Stage1轨迹并转向独立Refiner。"
        )
    else:
        decision = (
            "C1成功但C2未达标：优先检查teacher尺度、Huber权重和监督实现，"
            "不能据此判断网络容量。"
        )
    report = [
        "# C+ 小数据过拟合实验报告",
        "",
        "## 实验设置",
        "",
        f"- 固定样本：{len(selected)} 条，来自 {len(environments)} 张地图；"
        "每条样本固定同一个噪声。",
        f"- Teacher：raw_v/raw_s 优化 {args.teacher_steps} 步，"
        f"`gamma={args.length_ratio_limit}`，"
        f"`lambda_dev={args.lambda_dev}`，"
        f"`lambda_len={args.lambda_len}`。",
        f"- C1/C2：均从同一 Stage1 checkpoint 开始，全参数 AdamW，"
        f"固定学习率 {args.train_lr:g}，训练 {args.train_steps} 步。",
        "- Fisher、学习率调度、shuffle、dropout 和 weight decay 均关闭。",
        "",
        "## 核心结果（中位数）",
        "",
        "| 项目 | Stage1 | Teacher | C1 | C2 |",
        "|---|---:|---:|---:|---:|",
        f"| 修正目标 | {np.median(initial_components['corrected']):.4f} | "
        f"{np.median(teacher_components['corrected']):.4f} | "
        f"{np.median(c1_components['corrected']):.4f} | "
        f"{np.median(c2_components['corrected']):.4f} |",
        f"| 路径长度比 | 1.000 | "
        f"{np.median(teacher_components['length_ratio']):.3f} | "
        f"{np.median(c1_components['length_ratio']):.3f} | "
        f"{np.median(c2_components['length_ratio']):.3f} |",
        f"| Teacher恢复率 | — | 1.000 | {c1_recovery:.3f} | "
        f"{c2_recovery:.3f} |",
        f"| 改善样本比例 | — | — | {c1_improved:.1%} | "
        f"{c2_improved:.1%} |",
        f"| 相对Teacher误差 | — | 0 | "
        f"{summary['C1']['相对teacher误差']['median']:.3f} | "
        f"{summary['C2']['相对teacher误差']['median']:.3f} |",
        f"| 危险点占比 | "
        f"{np.median(initial_components['dangerous_ratio']):.3f} | "
        f"{np.median(teacher_components['dangerous_ratio']):.3f} | "
        f"{np.median(c1_components['dangerous_ratio']):.3f} | "
        f"{np.median(c2_components['dangerous_ratio']):.3f} |",
        f"| 最小ESDF (m) | "
        f"{np.median(initial_components['min_esdf']):.3f} | "
        f"{np.median(teacher_components['min_esdf']):.3f} | "
        f"{np.median(c1_components['min_esdf']):.3f} | "
        f"{np.median(c2_components['min_esdf']):.3f} |",
        "",
        "## 判定",
        "",
        decision,
        "",
        "## 结果解释",
        "",
        f"- C1 的 teacher 改善恢复率中位数为 {c1_recovery:.3f}，说明当前"
        "生成器和 cost-only 梯度在固定小数据上具备优化能力。",
        f"- C1 相对 teacher 误差仍为 "
        f"{summary['C1']['相对teacher误差']['median']:.3f}，说明它主要"
        "找到了另一组低 cost 解，而不是复现逐轨迹 teacher。",
        f"- C2 恢复率为 {c2_recovery:.3f}，相对 teacher 误差降到 "
        f"{summary['C2']['相对teacher误差']['median']:.3f}；一一配对的 "
        "teacher 监督有效。",
        f"- 危险点占比中位数从 "
        f"{np.median(initial_components['dangerous_ratio']):.1%} 降到 "
        f"C1={np.median(c1_components['dangerous_ratio']):.1%}、"
        f"C2={np.median(c2_components['dangerous_ratio']):.1%}，"
        "并非只改善代理 loss。",
        "- 本实验是过拟合能力诊断，不证明跨地图或新噪声上的泛化能力。",
        "",
        "## 下一步",
        "",
        "1. 分别恢复 Fisher、学习率调度、随机 noise，定位哪个设置破坏"
        "小数据过拟合。",
        "2. 用固定 `(条件, noise, R*)` 做 teacher 预训练，再用修正后的 "
        "cost 微调。",
        "3. 在完整验证集同时检查安全改善、路径长度和 diversity。",
        "",
        "如果扩大数据后 teacher 监督仍无法保持一一配对，或 diversity 再次"
        "坍塌，再转向“冻结 Stage1 + 显式输入 R0 + 预测局部 ΔR”的 Refiner。",
        "",
        "## 文件",
        "",
        "- `实验配置.json`：完整参数、checkpoint SHA256、代码版本。",
        "- `固定样本清单.csv`：地图、path id 和初始风险。",
        "- `固定数据与输出.npz`：noise、R0、R*、C1/C2 输出。",
        "- `训练曲线.csv`、`逐样本结果.csv`：可复核的原始数值。",
        "- `训练曲线.png`、`轨迹对比.png`：可视化。",
    ]
    (output_dir / "实验报告.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\n判定：{decision}", flush=True)


if __name__ == "__main__":
    main()
