#!/usr/bin/env python3
"""Evaluate one direct-safe model with nested K and privileged offline oracle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch

from bspline_utils import DifferentiableBSpline
from dataLoader_dit import compute_map_yaw_bins, generate_sdf_from_yaw_stability
from direct_safe_distribution.splits import (
    load_manifest,
    verify_selected_split_files,
)
from direct_safe_distribution.inference import decode_global_feasible
from dit.Models import (
    PathDiffusionTransformer,
    PhysicalScaledEdgeResidualRepresentation,
)
from experiment_a_representation import evaluate_control_points, normalize_poses
from experiment_stage2_coupling import mode_label, self_intersection
from grad_optimizer import discrete_turning_curvature
from map_config import MAP_CONFIG, MAP_HALF_EXTENT, SAFETY_COST_CONFIG
from train_direct_safe_distribution import DirectTargetDataset, VARIANTS


K_VALUES = (1, 4, 8, 16, 32)


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
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required acknowledgement before opening strict test maps.",
    )
    parser.add_argument("--conditions-per-map", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode-threshold-m", type=float, default=0.05)
    parser.add_argument("--dedup-rms-m", type=float, default=0.10)
    parser.add_argument("--abnormal-length-chord-ratio", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


class CostMapCache:
    def __init__(self, root: Path, max_items: int = 2):
        self.root = root
        self.max_items = max_items
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, map_id: str) -> torch.Tensor:
        if map_id in self.cache:
            value = self.cache.pop(map_id)
            self.cache[map_id] = value
            return value
        import pickle

        with (self.root / map_id / "map.p").open("rb") as handle:
            tensor = np.asarray(pickle.load(handle)["tensor"], dtype=np.float32)
        stability = compute_map_yaw_bins(
            tensor[..., 1],
            tensor[..., 2],
            tensor[..., 3],
            yaw_bins=MAP_CONFIG.yaw_bins,
        )
        cost_map = generate_sdf_from_yaw_stability(
            stability,
            voxel_size_xy=MAP_CONFIG.resolution,
            yaw_weight=SAFETY_COST_CONFIG.yaw_esdf_weight,
        )
        value = torch.as_tensor(cost_map, dtype=torch.float32)
        self.cache[map_id] = value
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


def endpoint_yaw_error(
    dense: torch.Tensor, start: torch.Tensor, goal: torch.Tensor
) -> torch.Tensor:
    first = dense[:, 1] - dense[:, 0]
    last = dense[:, -1] - dense[:, -2]
    first_yaw = torch.atan2(first[:, 1], first[:, 0])
    last_yaw = torch.atan2(last[:, 1], last[:, 0])
    first_error = torch.atan2(
        torch.sin(first_yaw - start[:, 2]),
        torch.cos(first_yaw - start[:, 2]),
    ).abs()
    last_error = torch.atan2(
        torch.sin(last_yaw - goal[:, 2]),
        torch.cos(last_yaw - goal[:, 2]),
    ).abs()
    return 0.5 * (first_error + last_error)


def greedy_unique_count(
    control: np.ndarray, valid: np.ndarray, threshold: float
) -> int:
    representatives: List[np.ndarray] = []
    for row in control[valid]:
        if all(
            float(np.sqrt(np.mean((row - representative) ** 2))) >= threshold
            for representative in representatives
        ):
            representatives.append(row)
    return len(representatives)


def aggregate_prefix(
    rows: Mapping[str, np.ndarray],
    condition_ids: np.ndarray,
    source_ids: np.ndarray,
    k: int,
    dedup_rms_m: float,
) -> Dict[str, float]:
    selected = source_ids < k
    safe = rows["strict_safe"].astype(bool)
    conditions = np.unique(condition_ids)
    safe_at_k = []
    best_cost = []
    safe_count = []
    mode_coverage = []
    unique_valid = []
    for condition in conditions:
        mask = selected & (condition_ids == condition)
        safe_mask = mask & safe
        safe_at_k.append(bool(np.any(safe_mask)))
        if np.any(safe_mask):
            best_cost.append(float(np.min(rows["safe_cost"][safe_mask])))
        else:
            best_cost.append(float(np.min(rows["safe_cost"][mask])))
        safe_count.append(int(np.sum(safe_mask)))
        mode_coverage.append(
            len(set(rows["mode_label"][safe_mask].astype(int).tolist()))
        )
        local_control = rows["control_points"][mask]
        local_valid = safe[mask] & ~rows["abnormal"][mask].astype(bool)
        unique_valid.append(
            greedy_unique_count(local_control, local_valid, dedup_rms_m)
        )
    return {
        "safe_at_k": float(np.mean(safe_at_k)),
        "valid_rate": float(np.mean(safe[selected])),
        "best_cost_at_k_mean": float(np.mean(best_cost)),
        "best_cost_at_k_median": float(np.median(best_cost)),
        "safe_candidate_count_mean": float(np.mean(safe_count)),
        "safe_mode_coverage_mean": float(np.mean(mode_coverage)),
        "deduplicated_valid_count_mean": float(np.mean(unique_valid)),
    }


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
                "A/B/C/D checkpoint does not prove strict train-map-only "
                f"provenance: {protocol}"
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = DirectTargetDataset(
        manifest=manifest,
        split="validation" if args.split == "validation" else "train",
        target_kind="raw",
        canonical=canonical,
        canvas_half_extent=float(model.coordinate_scale),
        max_paths_per_map=args.conditions_per_map,
        legacy_global_input=(
            float(model.coordinate_scale) == MAP_CONFIG.half_extent
            and not canonical
        ),
    )
    if args.split == "test":
        # Construct a test-only view only after the explicit acknowledgement.
        test_manifest = dict(manifest)
        test_manifest["splits"] = dict(manifest["splits"])
        test_manifest["splits"]["validation"] = manifest["splits"]["test"]
        dataset = DirectTargetDataset(
            manifest=test_manifest,
            split="validation",
            target_kind="raw",
            canonical=canonical,
            canvas_half_extent=float(model.coordinate_scale),
            max_paths_per_map=args.conditions_per_map,
        )
    elif args.split == "train":
        dataset = DirectTargetDataset(
            manifest=manifest,
            split="train",
            target_kind="raw",
            canonical=canonical,
            canvas_half_extent=float(model.coordinate_scale),
            max_paths_per_map=args.conditions_per_map,
        )
    cost_cache = CostMapCache(Path(manifest["dataset_folder"]))
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    global_representation = PhysicalScaledEdgeResidualRepresentation().to(device)
    stored: Dict[str, List[np.ndarray]] = {}
    latency_rows = []
    peak_memory = 0

    def append(key: str, value: object) -> None:
        array = (
            value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value)
        )
        stored.setdefault(key, []).append(array)

    for condition_id in range(len(dataset)):
        item = dataset[condition_id]
        map_id, path_num, original_start, original_goal, _ = dataset._target(
            condition_id
        )
        map_batch = item["map"].unsqueeze(0).repeat(K_VALUES[-1], 1, 1, 1)
        start_pose_model = item["start_pose"].unsqueeze(0).repeat(
            K_VALUES[-1], 1
        )
        goal_pose_model = item["goal_pose"].unsqueeze(0).repeat(
            K_VALUES[-1], 1
        )
        start_model = normalize_poses(
            start_pose_model, float(model.coordinate_scale)
        )
        goal_model = normalize_poses(
            goal_pose_model, float(model.coordinate_scale)
        )
        generator = torch.Generator(device=device).manual_seed(
            args.seed + condition_id * 7919
        )
        source = model.project_zero_sum(
            torch.randn(
                K_VALUES[-1],
                model.num_edges,
                2,
                generator=generator,
                device=device,
            )
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            residual = model(
                map_batch.to(device),
                source,
                torch.ones(K_VALUES[-1], device=device),
                torch.zeros(K_VALUES[-1], device=device),
                start_model.to(device),
                goal_model.to(device),
            )
            original_starts = original_start.unsqueeze(0).to(device).repeat(
                K_VALUES[-1], 1
            )
            original_goals = original_goal.unsqueeze(0).to(device).repeat(
                K_VALUES[-1], 1
            )
            _, control_global = decode_global_feasible(
                residual,
                original_starts,
                original_goals,
                model_coordinate_scale=float(model.coordinate_scale),
                canonical=canonical,
                representation=global_representation,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = max(
                peak_memory, torch.cuda.max_memory_allocated(device)
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_rows.append(elapsed_ms)
        starts = original_start.unsqueeze(0).repeat(K_VALUES[-1], 1).to(device)
        goals = original_goal.unsqueeze(0).repeat(K_VALUES[-1], 1).to(device)
        cost_maps = cost_cache.get(map_id).unsqueeze(0).repeat(
            K_VALUES[-1], 1, 1, 1
        )
        with torch.no_grad():
            metrics = evaluate_control_points(
                control_global,
                starts,
                goals,
                cost_maps.to(device),
                MAP_CONFIG.cost_map_info(),
                bspline,
                device,
            )
            dense = metrics["trajectory"]
            intersections = self_intersection(control_global.cpu()).to(device)
            chord = torch.linalg.vector_norm(
                goals[:, :2] - starts[:, :2], dim=1
            )
            length_ratio = metrics["path_length"] / chord.clamp_min(1e-6)
            dense_oob = (
                dense.abs() > MAP_HALF_EXTENT + 1e-6
            ).flatten(start_dim=1).any(dim=1)
            strict_safe = (
                (metrics["min_esdf"] >= 0.0)
                & ~dense_oob
                & torch.isfinite(metrics["cost"])
            )
            abnormal = (
                dense_oob
                | intersections
                | ~torch.isfinite(metrics["cost"])
                | (length_ratio > args.abnormal_length_chord_ratio)
            )
            yaw_error = endpoint_yaw_error(dense, starts, goals)
            mode = mode_label(control_global, args.mode_threshold_m)
        append("condition_id", np.repeat(condition_id, K_VALUES[-1]))
        append("source_id", np.arange(K_VALUES[-1]))
        append("map_id", np.repeat(map_id, K_VALUES[-1]))
        append("path_num", np.repeat(path_num, K_VALUES[-1]))
        append("source_noise", source)
        append("output_residual", residual)
        append("control_points", control_global)
        append("strict_safe", strict_safe)
        append("max_esdf_violation", (-metrics["min_esdf"]).clamp_min(0.0))
        append("safe_cost", metrics["cost"])
        append("path_length", metrics["path_length"])
        append("curvature_max", metrics["curvature_max"])
        append(
            "curvature_exceedance",
            (discrete_turning_curvature(dense) > SAFETY_COST_CONFIG.curvature_limit)
            .float()
            .mean(dim=1),
        )
        append("jerk", metrics["jerk"])
        append("endpoint_yaw_error", yaw_error)
        append("self_intersection", intersections)
        append("abnormal", abnormal)
        append("mode_label", mode)
        append("length_chord_ratio", length_ratio)
        print(
            f"{variant} {args.split} condition {condition_id + 1}/{len(dataset)}: "
            f"valid={int(strict_safe.sum())}/{K_VALUES[-1]}"
        )

    arrays = {key: np.concatenate(value, axis=0) for key, value in stored.items()}
    prefixes = {
        str(k): aggregate_prefix(
            arrays,
            arrays["condition_id"],
            arrays["source_id"],
            k,
            args.dedup_rms_m,
        )
        for k in K_VALUES
    }
    summary = {
        "variant": variant,
        "split": args.split,
        "strict_ood": strict_checkpoint and args.split == "test",
        "strict_ood_note": (
            "Legacy checkpoint trained on all dataset0 maps; this is a "
            "pipeline diagnostic, not unseen-map evidence."
            if variant == "legacy_global"
            else (
                "Strict checkpoint provenance verified; this is strict OOD "
                "only when split=test."
            )
        ),
        "condition_count": len(dataset),
        "candidate_count": len(arrays["strict_safe"]),
        "strict_safe_definition": (
            "all dense yaw-ESDF samples >=0, dense trajectory inside original "
            "map bounds, and finite privileged safe cost"
        ),
        "prefix_metrics": prefixes,
        "single_candidate": {
            "max_esdf_violation_median": float(
                np.median(arrays["max_esdf_violation"])
            ),
            "safe_cost_median": float(np.median(arrays["safe_cost"])),
            "path_length_median": float(np.median(arrays["path_length"])),
            "curvature_exceedance_mean": float(
                np.mean(arrays["curvature_exceedance"])
            ),
            "jerk_median": float(np.median(arrays["jerk"])),
            "endpoint_yaw_error_median_rad": float(
                np.median(arrays["endpoint_yaw_error"])
            ),
            "self_intersection_rate": float(
                np.mean(arrays["self_intersection"])
            ),
            "abnormal_rate": float(np.mean(arrays["abnormal"])),
        },
        "batch_inference": {
            "k": K_VALUES[-1],
            "latency_ms_mean": float(np.mean(latency_rows)),
            "latency_ms_median": float(np.median(latency_rows)),
            "peak_memory_bytes": int(peak_memory),
        },
        "oracle_note": (
            "Best-Cost@K uses privileged yaw-aware ESDF offline only; it is "
            "not an online candidate selector."
        ),
    }
    output_dir = args.output_dir or (
        args.checkpoint.parent / f"evaluation_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "per_candidate.npz", **arrays)
    scalar_keys = [
        key
        for key, value in arrays.items()
        if value.ndim == 1 and key not in ("map_id",)
    ]
    with (output_dir / "per_candidate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["map_id", *scalar_keys]
        )
        writer.writeheader()
        for index in range(len(arrays["map_id"])):
            writer.writerow(
                {
                    "map_id": arrays["map_id"][index],
                    **{
                        key: arrays[key][index].item()
                        for key in scalar_keys
                    },
                }
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        f"# Direct-safe evaluation: {variant} / {args.split}",
        "",
    ]
    if variant == "legacy_global":
        report.extend(
            [
                "> 诊断限定：该旧 checkpoint 训练过 dataset0 全部地图，本结果"
                "不是 strict OOD。",
                "",
            ]
        )
    report.extend(
        [
        "| K | Safe@K | Valid Rate | Best-Cost@K | Safe count | Modes | Dedup valid |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in K_VALUES:
        value = prefixes[str(k)]
        report.append(
            f"| {k} | {value['safe_at_k']:.1%} | "
            f"{value['valid_rate']:.1%} | "
            f"{value['best_cost_at_k_median']:.3f} | "
            f"{value['safe_candidate_count_mean']:.2f} | "
            f"{value['safe_mode_coverage_mean']:.2f} | "
            f"{value['deduplicated_valid_count_mean']:.2f} |"
        )
    report.extend(
        [
            "",
            "Best-Cost@K 是 privileged ESDF 离线 Oracle 上限，不属于在线方法。",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Completed {output_dir}")


if __name__ == "__main__":
    main()
