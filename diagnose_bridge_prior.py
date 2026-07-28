#!/usr/bin/env python3
"""Verify the projected edge-Gaussian discrete bridge prior without training."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from bspline_utils import DifferentiableBSpline, fit_bspline_least_squares
from direct_safe_distribution.splits import (
    load_manifest,
    verify_selected_split_files,
)
from dit.Models import PhysicalScaledEdgeResidualRepresentation
from experiment_stage2_coupling import self_intersection
from map_config import MAP_HALF_EXTENT


def theoretical_covariance(
    num_edges: int, sigma: float, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    indices = torch.arange(num_edges + 1, dtype=dtype)
    k = indices[:, None]
    ell = indices[None, :]
    return (sigma**2 / num_edges**2) * (
        torch.minimum(k, ell) - k * ell / num_edges
    )


def edge_bridge_samples(
    count: int,
    num_edges: int,
    sigma: float,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    increments = sigma * torch.randn(
        count, num_edges, 2, generator=generator, dtype=dtype
    )
    residual = increments - increments.mean(dim=1, keepdim=True)
    offsets = torch.cat(
        (
            torch.zeros(count, 1, 2, dtype=dtype),
            torch.cumsum(residual, dim=1) / num_edges,
        ),
        dim=1,
    )
    return residual, offsets


def waypoint_bridge_samples(
    count: int,
    num_edges: int,
    sigma: float,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Independent interior waypoint noise with bridge-matched marginals."""

    variance = theoretical_covariance(num_edges, sigma, dtype=dtype).diagonal()
    noise = torch.randn(
        count, num_edges + 1, 2, generator=generator, dtype=dtype
    )
    offsets = noise * torch.sqrt(variance)[None, :, None]
    offsets[:, 0] = 0.0
    offsets[:, -1] = 0.0
    return offsets


def load_target_control_points(
    manifest_path: Path, max_targets: int
) -> torch.Tensor:
    manifest = load_manifest(manifest_path, verify_files=False)
    verify_selected_split_files(manifest, ("train",))
    root = Path(manifest["dataset_folder"])
    rows = []
    for environment in manifest["splits"]["train"]:
        for path_file in sorted(
            (root / environment).glob("path_*.p"),
            key=lambda path: int(path.stem.split("_")[-1]),
        ):
            with path_file.open("rb") as handle:
                trajectory = np.asarray(pickle.load(handle)["path"])
            control, _, _ = fit_bspline_least_squares(
                trajectory[:, :2], num_control_points=26, degree=3
            )
            control[0] = trajectory[0, :2]
            control[-1] = trajectory[-1, :2]
            rows.append(torch.from_numpy(control).to(torch.float64))
            if len(rows) >= max_targets:
                return torch.stack(rows)
    if not rows:
        raise ValueError("No target trajectories found")
    return torch.stack(rows)


def source_quality(control: torch.Tensor) -> Dict[str, float]:
    bspline = DifferentiableBSpline(26, 100, 3).to(dtype=control.dtype)
    with torch.no_grad():
        dense = bspline(control)
    segment = dense[:, 1:] - dense[:, :-1]
    length = torch.linalg.vector_norm(segment, dim=-1).sum(dim=1)
    oob = (dense.abs() > MAP_HALF_EXTENT).flatten(start_dim=1).any(dim=1)
    intersections = self_intersection(control).to(torch.bool)
    abnormal = oob | intersections | ~torch.isfinite(length)
    return {
        "path_length_mean": float(length.mean()),
        "path_length_median": float(length.median()),
        "oob_rate": float(oob.float().mean()),
        "self_intersection_rate": float(intersections.float().mean()),
        "abnormal_rate": float(abnormal.float().mean()),
    }


def compare_sources(
    targets: torch.Tensor,
    sigma: float,
    seed: int,
    coordinate_scale: float,
) -> Dict[str, Dict[str, object]]:
    count = len(targets)
    n = targets.shape[1] - 1
    generator = torch.Generator().manual_seed(seed)
    start = targets[:, :1]
    goal = targets[:, -1:]
    progress = torch.linspace(0.0, 1.0, n + 1, dtype=targets.dtype)
    chord = (
        (1.0 - progress)[None, :, None] * start
        + progress[None, :, None] * goal
    )

    # Model residuals and standard Gaussian sources live in normalized map
    # coordinates.  Convert every source to physical meters before comparing
    # against physical target control points.
    absolute = coordinate_scale * sigma * torch.randn(
        count, n + 1, 2, generator=generator, dtype=targets.dtype
    )
    waypoint = chord + coordinate_scale * waypoint_bridge_samples(
        count, n, sigma, generator=generator, dtype=targets.dtype
    )
    edge_residual, edge_offsets = edge_bridge_samples(
        count, n, sigma, generator=generator, dtype=targets.dtype
    )
    edge = chord + coordinate_scale * edge_offsets
    representation = PhysicalScaledEdgeResidualRepresentation().to(
        dtype=targets.dtype
    )
    targets_normalized = targets / coordinate_scale
    target_residual = representation.encode(
        targets_normalized,
        targets_normalized[:, 0],
        targets_normalized[:, -1],
    )
    native_velocities = {
        "absolute_control_point_gaussian": (
            absolute / coordinate_scale - targets_normalized
        ),
        "independent_waypoint_bridge_marginals": (
            waypoint / coordinate_scale - targets_normalized
        ),
        "projected_edge_increment_bridge": (
            edge_residual - target_residual
        ),
    }
    sources = {
        "absolute_control_point_gaussian": absolute,
        "independent_waypoint_bridge_marginals": waypoint,
        "projected_edge_increment_bridge": edge,
    }
    result: Dict[str, Dict[str, object]] = {}
    for name, source in sources.items():
        physical_velocity = source - targets
        native_velocity = native_velocities[name]
        endpoint_error = 0.5 * (
            torch.linalg.vector_norm(source[:, 0] - targets[:, 0], dim=-1)
            + torch.linalg.vector_norm(source[:, -1] - targets[:, -1], dim=-1)
        )
        metrics: Dict[str, object] = {
            "physical_control_source_target_squared_distance_mean": float(
                physical_velocity.square().sum(dim=(1, 2)).mean()
            ),
            "native_meanflow_target_squared_distance_mean": float(
                native_velocity.square().sum(dim=(1, 2)).mean()
            ),
            "native_meanflow_target_vector_norm_mean": float(
                torch.linalg.vector_norm(
                    native_velocity.flatten(start_dim=1), dim=1
                ).mean()
            ),
            "per_control_point_physical_velocity_rms": torch.sqrt(
                physical_velocity.square().mean(dim=(0, 2))
            ).tolist(),
            "endpoint_error_mean": float(endpoint_error.mean()),
            **source_quality(source),
        }
        result[name] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--targets", type=int, default=500)
    parser.add_argument("--num-edges", type=int, default=25)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "diagnostics/direct_safe_distribution/split_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/direct_safe_distribution"),
    )
    args = parser.parse_args()
    if args.samples < 2 or args.targets < 1:
        raise ValueError("--samples must be >=2 and --targets must be positive")
    generator = torch.Generator().manual_seed(args.seed)
    residual, offsets = edge_bridge_samples(
        args.samples,
        args.num_edges,
        args.sigma,
        generator=generator,
    )
    one_axis = offsets[:, :, 0]
    empirical_mean = one_axis.mean(dim=0)
    empirical_cov = torch.cov(one_axis.T)
    theory = theoretical_covariance(args.num_edges, args.sigma)
    covariance_error = empirical_cov - theory
    targets = load_target_control_points(args.manifest, args.targets)
    source_comparison = compare_sources(
        targets, args.sigma, args.seed + 1, MAP_HALF_EXTENT
    )

    variance_relative_error = (
        (empirical_cov.diagonal() - theory.diagonal()).abs()
        / theory.diagonal().clamp_min(1e-12)
    )
    interior = slice(1, -1)
    summary = {
        "num_samples": args.samples,
        "num_edges": args.num_edges,
        "sigma": args.sigma,
        "mean_absolute_max": float(empirical_mean.abs().max()),
        "zero_sum_max": float(residual.sum(dim=1).abs().max()),
        "covariance_rmse": float(torch.sqrt(covariance_error.square().mean())),
        "covariance_max_abs_error": float(covariance_error.abs().max()),
        "interior_variance_relative_error_mean": float(
            variance_relative_error[interior].mean()
        ),
        "interior_variance_relative_error_max": float(
            variance_relative_error[interior].max()
        ),
        "endpoint_variance_empirical": [
            float(empirical_cov[0, 0]),
            float(empirical_cov[-1, -1]),
        ],
        "midpoint_variance_empirical": float(
            empirical_cov[args.num_edges // 2, args.num_edges // 2]
        ),
        "midpoint_variance_theory": float(
            theory[args.num_edges // 2, args.num_edges // 2]
        ),
        "target_count": len(targets),
        "physical_coordinate_scale": MAP_HALF_EXTENT,
        "source_comparison": source_comparison,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bridge_prior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "bridge_prior_statistics.npz",
        empirical_mean=empirical_mean.numpy(),
        empirical_covariance=empirical_cov.numpy(),
        theoretical_covariance=theory.numpy(),
        covariance_error=covariance_error.numpy(),
    )
    rows = [
        "# 零和边增量桥先验诊断",
        "",
        f"- Monte Carlo 样本：{args.samples:,}",
        f"- `max |E[Y_k]|`：{summary['mean_absolute_max']:.3e}",
        f"- 协方差 RMSE：{summary['covariance_rmse']:.3e}",
        f"- 协方差最大绝对误差：{summary['covariance_max_abs_error']:.3e}",
        "- 内部点方差平均/最大相对误差："
        f"{summary['interior_variance_relative_error_mean']:.2%} / "
        f"{summary['interior_variance_relative_error_max']:.2%}",
        "- 中点经验/理论方差："
        f"{summary['midpoint_variance_empirical']:.6f} / "
        f"{summary['midpoint_variance_theory']:.6f}",
        "",
        "理论与实证均支持：当前 projected Gaussian edge source 在控制点偏移"
        "空间中就是离散 Brownian bridge；端点方差为零、中间方差最大。",
        "",
        "## 同一目标数据上的 source 对照",
        "",
        f"下表先按模型 `coordinate_scale={MAP_HALF_EXTENT:g} m` 将三种"
        " normalized source 统一换算到物理控制点空间。",
        "",
        "| Source | physical CP squared distance | native vector norm | endpoint error | length | abnormal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in source_comparison.items():
        rows.append(
            f"| {name} | "
            f"{metrics['physical_control_source_target_squared_distance_mean']:.3f} | "
            f"{metrics['native_meanflow_target_vector_norm_mean']:.3f} | "
            f"{metrics['endpoint_error_mean']:.3f} | "
            f"{metrics['path_length_median']:.3f} | "
            f"{metrics['abnormal_rate']:.1%} |"
        )
    rows.extend(
        [
            "",
            "绝对控制点高斯不固定端点；另外两种桥源固定端点。独立 waypoint"
            " 对照只匹配每个位置的边际方差，不匹配跨控制点协方差，因此不能据此"
            " 替换当前 source。表中 distance 是物理控制点距离；vector norm "
            "是各表示真正训练时的 native MeanFlow 状态范数（绝对/waypoint "
            "为 normalized control state，当前方法为 25× scaled residual），"
            "两者不能混作同一单位。逐控制点物理速度 RMS 保存在 JSON 中。",
        ]
    )
    (args.output_dir / "bridge_prior_report.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
