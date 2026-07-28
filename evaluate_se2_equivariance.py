#!/usr/bin/env python3
"""Measure control-point and dense-trajectory SE(2) equivariance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from bspline_utils import DifferentiableBSpline
from direct_safe_distribution.splits import (
    load_manifest,
    verify_selected_split_files,
)
from direct_safe_distribution.inference import decode_global_feasible
from dit.Models import PathDiffusionTransformer
from evaluate_direct_safe_distribution import CostMapCache
from experiment_a_representation import evaluate_control_points
from experiment_stage2_coupling import mode_label
from geometry.canonicalization import (
    canonicalize_map,
    canonicalize_pose,
    inverse_transform_trajectory_se2,
    transform_map_se2,
    transform_pose_se2,
    transform_residual_se2,
)
from map_config import MAP_CONFIG
from train_direct_safe_distribution import DirectTargetDataset, VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-params",
        type=Path,
        default=None,
        help="Required only for legacy checkpoints without embedded model_args.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "diagnostics/direct_safe_distribution/split_manifest.json"
        ),
    )
    parser.add_argument(
        "--angles-deg",
        type=float,
        nargs="+",
        default=[-180, -135, -90, -45, 45, 90, 135, 180],
    )
    parser.add_argument("--conditions", type=int, default=16)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--max-translation-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def preprocess_map(
    normal_map: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    *,
    canonical: bool,
    coordinate_scale: float,
) -> torch.Tensor:
    if not canonical and abs(coordinate_scale - MAP_CONFIG.half_extent) < 1e-6:
        return normal_map
    output_bounds = (
        -coordinate_scale,
        coordinate_scale,
        -coordinate_scale,
        coordinate_scale,
    )
    if canonical:
        return canonicalize_map(
            normal_map,
            start,
            goal,
            source_bounds=(-10.0, 9.8, -10.0, 9.8),
            output_bounds=output_bounds,
        ).values
    identity_start = torch.tensor([0.0, 0.0, 0.0])
    identity_goal = torch.tensor([1.0, 0.0, 0.0])
    return canonicalize_map(
        normal_map,
        identity_start,
        identity_goal,
        source_bounds=(-10.0, 9.8, -10.0, 9.8),
        output_bounds=output_bounds,
    ).values


def model_poses(
    start: torch.Tensor,
    goal: torch.Tensor,
    canonical: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if canonical:
        return (
            canonicalize_pose(start, start, goal),
            canonicalize_pose(goal, start, goal),
        )
    return start, goal


@torch.no_grad()
def generate_control(
    model: PathDiffusionTransformer,
    map_input: torch.Tensor,
    start_global: torch.Tensor,
    goal_global: torch.Tensor,
    source: torch.Tensor,
    *,
    canonical: bool,
    device: torch.device,
) -> torch.Tensor:
    start, goal = model_poses(start_global, goal_global, canonical)
    start_n = torch.zeros(1, 4, device=device)
    goal_n = torch.zeros(1, 4, device=device)
    start_n[:, :2] = start[:2].to(device) / float(model.coordinate_scale)
    goal_n[:, :2] = goal[:2].to(device) / float(model.coordinate_scale)
    start_n[:, 2] = torch.cos(start[2])
    start_n[:, 3] = torch.sin(start[2])
    goal_n[:, 2] = torch.cos(goal[2])
    goal_n[:, 3] = torch.sin(goal[2])
    residual = model(
        map_input.unsqueeze(0).to(device),
        source.unsqueeze(0).to(device),
        torch.ones(1, device=device),
        torch.zeros(1, device=device),
        start_n,
        goal_n,
    )
    _, control = decode_global_feasible(
        residual,
        start_global.unsqueeze(0).to(device),
        goal_global.unsqueeze(0).to(device),
        model_coordinate_scale=float(model.coordinate_scale),
        canonical=canonical,
    )
    return control[0]


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("--allow-test is required before opening strict test maps")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    manifest = load_manifest(args.manifest, verify_files=False)
    verify_selected_split_files(manifest, (args.split,))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    variant = str(checkpoint.get("variant", "legacy_global"))
    if variant in VARIANTS:
        _, canonical = VARIANTS[variant]
    elif variant == "legacy_global":
        canonical = False
    else:
        raise ValueError(f"Unknown checkpoint variant: {variant}")
    strict_checkpoint = False
    if variant in VARIANTS:
        protocol = checkpoint.get("data_protocol", {})
        strict_checkpoint = (
            protocol.get("split_manifest_sha256")
            == manifest["manifest_sha256"]
            and sorted(protocol.get("training_map_ids", []))
            == sorted(manifest["splits"]["train"])
            and set(protocol.get("allowed_splits", [])) == {"train"}
            and not bool(protocol.get("test_maps_opened", True))
        )
        if not strict_checkpoint:
            raise ValueError(
                "A/B/C/D checkpoint lacks strict split provenance: "
                f"{protocol}"
            )
    if "model_args" in checkpoint:
        model_args = checkpoint["model_args"]
    else:
        params_path = args.model_params or (
            args.checkpoint.parent / "model_params.json"
        )
        model_args = json.loads(
            params_path.read_text(encoding="utf-8")
        )["model_args"]
    model = PathDiffusionTransformer(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset_manifest = manifest
    dataset_split = args.split
    if args.split == "test":
        dataset_manifest = dict(manifest)
        dataset_manifest["splits"] = dict(manifest["splits"])
        dataset_manifest["splits"]["validation"] = manifest["splits"]["test"]
        dataset_split = "validation"
    dataset = DirectTargetDataset(
        manifest=dataset_manifest,
        split=dataset_split,
        target_kind="raw",
        canonical=canonical,
        canvas_half_extent=float(model.coordinate_scale),
        max_paths_per_map=1,
    )
    count = min(args.conditions, len(dataset))
    cost_cache = CostMapCache(Path(manifest["dataset_folder"]))
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows: List[Dict[str, object]] = []
    for condition_id in range(count):
        map_id, path_num, start, goal, _ = dataset._target(condition_id)
        normal = dataset.map_cache.get(map_id)
        base_map = preprocess_map(
            normal,
            start,
            goal,
            canonical=canonical,
            coordinate_scale=float(model.coordinate_scale),
        )
        source = model.project_zero_sum(
            torch.randn(
                1,
                model.num_edges,
                2,
                generator=generator,
                device=device,
            )
        )[0]
        base_control = generate_control(
            model,
            base_map,
            start,
            goal,
            source,
            canonical=canonical,
            device=device,
        )
        base_dense = bspline(base_control.unsqueeze(0))[0]
        cost_map = cost_cache.get(map_id).unsqueeze(0).to(device)
        base_metrics = evaluate_control_points(
            base_control.unsqueeze(0),
            start.unsqueeze(0).to(device),
            goal.unsqueeze(0).to(device),
            cost_map,
            MAP_CONFIG.cost_map_info(),
            bspline,
            device,
        )
        base_safe = bool(base_metrics["min_esdf"][0] >= 0.0)
        base_mode = int(mode_label(base_control.unsqueeze(0), 0.05)[0])
        for angle_deg in args.angles_deg:
            angle = torch.tensor(
                math.radians(angle_deg), dtype=start.dtype
            )
            if args.max_translation_m > 0.0:
                translation = (
                    torch.rand(2, generator=generator) * 2.0 - 1.0
                ) * args.max_translation_m
            else:
                translation = torch.zeros(2)
            transformed_start = transform_pose_se2(
                start, angle, translation
            )
            transformed_goal = transform_pose_se2(
                goal, angle, translation
            )
            transformed_raster = transform_map_se2(
                normal,
                angle,
                translation,
                source_bounds=(-10.0, 9.8, -10.0, 9.8),
                output_bounds=(-10.0, 9.8, -10.0, 9.8),
            )
            transformed_map = preprocess_map(
                transformed_raster.values,
                transformed_start,
                transformed_goal,
                canonical=canonical,
                coordinate_scale=float(model.coordinate_scale),
            )
            transformed_source = (
                source
                if canonical
                else transform_residual_se2(source, angle)
            )
            transformed_control = generate_control(
                model,
                transformed_map,
                transformed_start,
                transformed_goal,
                transformed_source,
                canonical=canonical,
                device=device,
            )
            aligned_control = inverse_transform_trajectory_se2(
                transformed_control, angle.to(device), translation.to(device)
            )
            aligned_dense = bspline(aligned_control.unsqueeze(0))[0]
            control_error = torch.sqrt(
                (aligned_control - base_control).square().mean()
            )
            dense_error = torch.sqrt(
                (aligned_dense - base_dense).square().mean()
            )
            control_denominator = torch.sqrt(
                (base_control - start[:2].to(device)).square().mean()
            ).clamp_min(1e-6)
            dense_denominator = torch.sqrt(
                (base_dense - start[:2].to(device)).square().mean()
            ).clamp_min(1e-6)
            aligned_metrics = evaluate_control_points(
                aligned_control.unsqueeze(0),
                start.unsqueeze(0).to(device),
                goal.unsqueeze(0).to(device),
                cost_map,
                MAP_CONFIG.cost_map_info(),
                bspline,
                device,
            )
            aligned_safe = bool(aligned_metrics["min_esdf"][0] >= 0.0)
            aligned_mode = int(
                mode_label(aligned_control.unsqueeze(0), 0.05)[0]
            )
            base_cost = float(base_metrics["cost"][0])
            aligned_cost = float(aligned_metrics["cost"][0])
            rows.append(
                {
                    "condition_id": condition_id,
                    "map_id": map_id,
                    "path_num": path_num,
                    "angle_deg": angle_deg,
                    "translation_x": float(translation[0]),
                    "translation_y": float(translation[1]),
                    "control_equivariance_error_m": float(control_error),
                    "control_equivariance_error_relative": float(
                        control_error / control_denominator
                    ),
                    "dense_equivariance_error_m": float(dense_error),
                    "dense_equivariance_error_relative": float(
                        dense_error / dense_denominator
                    ),
                    "mode_changed": base_mode != aligned_mode,
                    "strict_safe_changed": base_safe != aligned_safe,
                    "cost_absolute_difference": abs(
                        base_cost - aligned_cost
                    ),
                    "cost_relative_difference": abs(
                        base_cost - aligned_cost
                    )
                    / max(abs(base_cost), 1e-6),
                    "transformed_raster_valid_fraction": float(
                        transformed_raster.valid_mask.mean()
                    ),
                }
            )
        print(f"{variant} equivariance {condition_id + 1}/{count}")

    angle_summary = {}
    for angle in args.angles_deg:
        selected = [row for row in rows if row["angle_deg"] == angle]
        angle_summary[str(angle)] = {
            "control_error_relative_mean": float(
                np.mean(
                    [
                        row["control_equivariance_error_relative"]
                        for row in selected
                    ]
                )
            ),
            "dense_error_relative_mean": float(
                np.mean(
                    [
                        row["dense_equivariance_error_relative"]
                        for row in selected
                    ]
                )
            ),
            "mode_change_rate": float(
                np.mean([row["mode_changed"] for row in selected])
            ),
            "strict_safe_change_rate": float(
                np.mean([row["strict_safe_changed"] for row in selected])
            ),
            "cost_relative_difference_mean": float(
                np.mean(
                    [row["cost_relative_difference"] for row in selected]
                )
            ),
        }
    summary = {
        "variant": variant,
        "canonical": canonical,
        "strict_checkpoint_provenance": strict_checkpoint,
        "strict_ood": strict_checkpoint and args.split == "test",
        "split": args.split,
        "condition_count": count,
        "continuous_formula": "strict SE(2)",
        "raster_statement": (
            "approximate because two bilinear resampling operations and finite "
            "raster support introduce interpolation/padding error"
        ),
        "by_angle": angle_summary,
    }
    output_dir = args.output_dir or (
        args.checkpoint.parent / f"equivariance_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_transform.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        f"# SE(2) equivariance: {variant} / {args.split}",
        "",
    ]
    if variant == "legacy_global":
        report.extend(
            [
                "> 诊断限定：旧 checkpoint 训练过 dataset0 全部地图；这里仅测"
                "几何变换敏感性，不是 unseen-map 泛化。",
                "",
            ]
        )
    report.extend(
        [
        "| angle | control rel. error | dense rel. error | mode change | safe change | cost rel. diff |",
        "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for angle in args.angles_deg:
        value = angle_summary[str(angle)]
        report.append(
            f"| {angle:g}° | "
            f"{value['control_error_relative_mean']:.4f} | "
            f"{value['dense_error_relative_mean']:.4f} | "
            f"{value['mode_change_rate']:.1%} | "
            f"{value['strict_safe_change_rate']:.1%} | "
            f"{value['cost_relative_difference_mean']:.2%} |"
        )
    report.extend(
        [
            "",
            "连续点/向量变换解析上严格；表中误差包含有限栅格双线性重采样与"
            " padding，不能表述为数值严格等变。",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Completed {output_dir}")


if __name__ == "__main__":
    main()
