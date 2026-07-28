#!/usr/bin/env python3
"""Generate train-map-only, topology-preserving safety-teacher trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from bspline_utils import DifferentiableBSpline
from dataLoader_dit import UnevenPathDataLoader
from direct_safe_distribution.splits import (
    load_manifest,
    sha256_file,
    verify_selected_split_files,
)
from dit.Models import PathDiffusionTransformer
from experiment_a_representation import evaluate_control_points, git_metadata
from experiment_cplus_overfit import load_model
from experiment_safety_priority_teacher import optimize_batch
from experiment_stage2_coupling import mode_label, stage1_from_noise
from map_config import MAP_CONFIG


REQUIRED_CANDIDATE_FIELDS = (
    "map_id",
    "condition_id",
    "start_pose",
    "goal_pose",
    "source_noise",
    "stage1_residual",
    "optimized_residual",
    "strict_safe",
    "max_esdf_violation",
    "safe_cost",
    "path_length",
    "curvature",
    "jerk",
    "mode_label",
    "max_control_point_displacement",
    "length_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "diagnostics/direct_safe_distribution/split_manifest.json"
        ),
    )
    parser.add_argument("--model-params", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditions-per-map", type=int, default=16)
    parser.add_argument(
        "--max-maps",
        type=int,
        default=None,
        help="Optional train-map subset for smoke/scaling studies.",
    )
    parser.add_argument("--condition-batch-size", type=int, default=2)
    parser.add_argument("--k-teacher", type=int, choices=(8, 16), default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--multi-starts", type=int, default=2)
    parser.add_argument("--steps-per-stage", type=int, default=100)
    parser.add_argument(
        "--trust-displacements", type=float, nargs="+", default=[1.0, 2.0, 3.0]
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
    parser.add_argument(
        "--minimum-violation-improvement-m", type=float, default=0.02
    )
    parser.add_argument("--use-precomputed-stability", action="store_true")
    parser.add_argument(
        "--stability-map-filename", default="stability_map.npz"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "diagnostics/direct_safe_distribution/safety_teacher_dataset"
        ),
    )
    parser.add_argument(
        "--allow-incompatible-checkpoint",
        action="store_true",
        help="Diagnostic only; marks output non-compliant.",
    )
    return parser.parse_args()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def validate_checkpoint_protocol(
    checkpoint: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    allow_incompatible: bool,
) -> bool:
    protocol = checkpoint.get("data_protocol", {})
    expected_maps = sorted(manifest["splits"]["train"])
    observed_hash = protocol.get("split_manifest_sha256")
    observed_maps = sorted(protocol.get("training_map_ids", []))
    compatible = (
        observed_hash == manifest["manifest_sha256"]
        and observed_maps == expected_maps
        and set(protocol.get("allowed_splits", [])) == {"train"}
    )
    if not compatible and not allow_incompatible:
        raise ValueError(
            "Stage-1 checkpoint is not proven train-map-only. Expected "
            f"manifest={manifest['manifest_sha256']} and exactly "
            f"{len(expected_maps)} train map IDs; checkpoint protocol={protocol}"
        )
    return compatible


def _tensor_chunks(
    control: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    cost_maps: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    chunks: Dict[str, List[torch.Tensor]] = {}
    for begin in range(0, len(control), batch_size):
        end = min(begin + batch_size, len(control))
        metrics = evaluate_control_points(
            control[begin:end].to(device),
            starts[begin:end].to(device),
            goals[begin:end].to(device),
            cost_maps[begin:end].to(device),
            MAP_CONFIG.cost_map_info(),
            bspline,
            device,
        )
        for key, value in metrics.items():
            if key == "trajectory":
                continue
            chunks.setdefault(key, []).append(value.detach().cpu())
    return {key: torch.cat(rows) for key, rows in chunks.items()}


def _append(store: Dict[str, List[np.ndarray]], **values: object) -> None:
    for key, value in values.items():
        if torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        store.setdefault(key, []).append(array)


def _concatenate(store: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        key: np.concatenate(values, axis=0)
        for key, values in store.items()
    }


def write_condition_summary(
    path: Path,
    candidates: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for condition in np.unique(candidates["condition_id"]):
        selected = candidates["condition_id"] == condition
        safe = selected & candidates["strict_safe"].astype(bool)
        modes = candidates["mode_label"][safe].astype(int)
        counts = {
            str(label): int(np.sum(modes == label)) for label in (-1, 0, 1)
        }
        rows.append(
            {
                "condition_id": int(condition),
                "map_id": str(candidates["map_id"][selected][0]),
                "path_num": int(candidates["path_num"][selected][0]),
                "teacher_count": int(np.sum(selected)),
                "strict_safe_teacher_count": int(np.sum(safe)),
                "safe_left_count": counts["1"],
                "safe_right_count": counts["-1"],
                "safe_near_straight_count": counts["0"],
                "has_strict_safe": bool(np.any(safe)),
                "has_two_safe_modes": len(set(modes.tolist())) >= 2,
            }
        )
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    args = parse_args()
    if len(args.trust_displacements) != len(args.trust_length_ratios):
        raise ValueError("Trust displacement and length schedules must match")
    if args.conditions_per_map < 1 or args.condition_batch_size < 1:
        raise ValueError("Condition counts must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    manifest = load_manifest(args.manifest, verify_files=False)
    verify_selected_split_files(manifest, ("train",))
    model_params = json.loads(args.model_params.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_compatible = validate_checkpoint_protocol(
        checkpoint,
        manifest,
        allow_incompatible=args.allow_incompatible_checkpoint,
    )
    model: PathDiffusionTransformer = load_model(
        model_params, checkpoint, device
    )
    if abs(float(model.coordinate_scale) - MAP_CONFIG.half_extent) > 1e-6:
        raise ValueError(
            "Teacher proposal checkpoint must use the legacy physical "
            f"coordinate scale {MAP_CONFIG.half_extent}; got "
            f"{model.coordinate_scale}"
        )
    model.eval()

    train_maps = list(manifest["splits"]["train"])
    if args.max_maps is not None:
        if args.max_maps < 1:
            raise ValueError("--max-maps must be positive")
        train_maps = train_maps[: args.max_maps]
    dataset = UnevenPathDataLoader(
        env_list=train_maps,
        dataFolder=manifest["dataset_folder"],
        compute_stability_map=True,
        use_precomputed_stability=args.use_precomputed_stability,
        stability_map_filename=args.stability_map_filename,
        compute_stability_if_missing=not args.use_precomputed_stability,
    )
    indices_by_map: Dict[str, List[int]] = {name: [] for name in train_maps}
    for dataset_index, (env_index, _) in enumerate(dataset.indexDict):
        indices_by_map[dataset.env_list[env_index]].append(dataset_index)

    teacher_args = SimpleNamespace(**vars(args))
    started = time.time()
    stored: Dict[str, List[np.ndarray]] = {}
    global_condition_id = 0
    for map_offset, map_id in enumerate(train_maps):
        indices = indices_by_map[map_id][: args.conditions_per_map]
        if len(indices) < args.conditions_per_map:
            raise ValueError(
                f"{map_id} has only {len(indices)} path conditions"
            )
        for condition_begin in range(0, len(indices), args.condition_batch_size):
            selected_indices = indices[
                condition_begin : condition_begin + args.condition_batch_size
            ]
            items = [dataset[index] for index in selected_indices]
            condition_count = len(items)
            maps = torch.stack([item["map"] for item in items])
            cost_maps_by_condition = torch.stack(
                [item["cost_map"] for item in items]
            )
            starts_by_condition = torch.stack(
                [item["start_pose"] for item in items]
            )
            goals_by_condition = torch.stack(
                [item["goal_pose"] for item in items]
            )
            path_nums = np.asarray(
                [dataset.indexDict[index][1] for index in selected_indices],
                dtype=np.int64,
            )
            condition_ids = np.arange(
                global_condition_id,
                global_condition_id + condition_count,
                dtype=np.int64,
            )
            global_condition_id += condition_count

            repeated_maps = maps.repeat_interleave(args.k_teacher, dim=0)
            repeated_cost_maps = cost_maps_by_condition.repeat_interleave(
                args.k_teacher, dim=0
            )
            starts = starts_by_condition.repeat_interleave(
                args.k_teacher, dim=0
            )
            goals = goals_by_condition.repeat_interleave(
                args.k_teacher, dim=0
            )
            noise_rows = []
            condition_seed_rows = []
            for local, path_num in enumerate(path_nums):
                condition_seed = _stable_seed(
                    args.seed,
                    manifest["manifest_sha256"],
                    map_id,
                    int(path_num),
                )
                generator = torch.Generator(device=device).manual_seed(
                    condition_seed
                )
                noise = model.project_zero_sum(
                    torch.randn(
                        args.k_teacher,
                        model.num_edges,
                        2,
                        generator=generator,
                        device=device,
                    )
                )
                noise_rows.append(noise.cpu())
                condition_seed_rows.extend([condition_seed] * args.k_teacher)
            noise = torch.cat(noise_rows)
            with torch.no_grad():
                r0, cp0 = stage1_from_noise(
                    model,
                    repeated_maps.to(device),
                    starts.to(device),
                    goals.to(device),
                    noise.to(device),
                )
            local_data = {
                "cp0": cp0.cpu(),
                "start": starts,
                "goal": goals,
                "cost_map": repeated_cost_maps,
            }
            local_indices = torch.arange(len(cp0))
            teacher_args.seed = _stable_seed(
                args.seed, map_id, condition_begin, "teacher"
            )
            optimized = optimize_batch(
                local_data, local_indices, teacher_args, device
            )
            initial_metrics = _tensor_chunks(
                cp0.cpu(),
                starts,
                goals,
                repeated_cost_maps,
                device=device,
            )
            final_metrics = _tensor_chunks(
                optimized["cp_star"],
                starts,
                goals,
                repeated_cost_maps,
                device=device,
            )
            source_ids = np.tile(
                np.arange(args.k_teacher, dtype=np.int64), condition_count
            )
            repeated_condition_ids = np.repeat(
                condition_ids, args.k_teacher
            )
            repeated_path_nums = np.repeat(path_nums, args.k_teacher)
            initial_violation = (-initial_metrics["min_esdf"]).clamp_min(0.0)
            final_violation = optimized["max_violation"]
            improved = (
                final_violation
                <= initial_violation - args.minimum_violation_improvement_m
            ) | optimized["strict_safe"]
            _append(
                stored,
                map_id=np.repeat(map_id, len(cp0)),
                condition_id=repeated_condition_ids,
                path_num=repeated_path_nums,
                source_id=source_ids,
                condition_seed=np.asarray(condition_seed_rows, dtype=np.int64),
                start_pose=starts,
                goal_pose=goals,
                source_noise=noise,
                stage1_residual=r0.cpu(),
                optimized_residual=optimized["r_star"],
                strict_safe=optimized["strict_safe"],
                improved=improved,
                training_eligible_strict=optimized["strict_safe"],
                training_eligible_improved=improved,
                max_esdf_violation=optimized["max_violation"],
                initial_max_esdf_violation=initial_violation,
                safe_cost=final_metrics["cost"],
                initial_safe_cost=initial_metrics["cost"],
                path_length=final_metrics["path_length"],
                curvature=final_metrics["curvature_max"],
                jerk=final_metrics["jerk"],
                mode_label=optimized["mode"],
                stage1_mode_label=mode_label(
                    cp0.cpu(), args.mode_threshold_m
                ),
                max_control_point_displacement=optimized["max_displacement"],
                length_ratio=optimized["length_ratio"],
                min_esdf=optimized["min_esdf"],
                dangerous_count=optimized["dangerous_count"],
                chosen_local_start=optimized["chosen_start"],
            )
            print(
                f"{map_id} conditions {condition_begin}:"
                f"{condition_begin + condition_count}, "
                f"strict={int(optimized['strict_safe'].sum())}/{len(cp0)}"
            )

    candidates = _concatenate(stored)
    missing = set(REQUIRED_CANDIDATE_FIELDS) - set(candidates)
    if missing:
        raise AssertionError(f"Missing required fields: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "candidates.npz", **candidates)
    condition_rows = write_condition_summary(
        args.output_dir / "conditions.csv", candidates
    )
    safe = candidates["strict_safe"].astype(bool)
    eligible_improved = candidates["training_eligible_improved"].astype(bool)
    summary = {
        "schema_version": 1,
        "compliant_train_only_checkpoint": checkpoint_compatible,
        "candidate_count": len(safe),
        "condition_count": len(condition_rows),
        "map_count": len(train_maps),
        "k_teacher": args.k_teacher,
        "strict_safe_candidate_count": int(safe.sum()),
        "strict_safe_candidate_rate": float(safe.mean()),
        "improved_candidate_count": int(eligible_improved.sum()),
        "conditions_with_strict_safe": int(
            sum(row["has_strict_safe"] for row in condition_rows)
        ),
        "conditions_with_two_safe_modes": int(
            sum(row["has_two_safe_modes"] for row in condition_rows)
        ),
        "elapsed_seconds": time.time() - started,
        "manifest": str(args.manifest.resolve()),
        "split_manifest_sha256": manifest["manifest_sha256"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_params": str(args.model_params.resolve()),
        "teacher_parameters": vars(args),
        "map_config": MAP_CONFIG.to_dict(),
        "code_version": git_metadata(),
        "training_policy": {
            "strict_default_mask": "training_eligible_strict",
            "unsafe_improved_optional_mask": "training_eligible_improved",
            "unsafe targets are never relabeled strict_safe": True,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Completed {args.output_dir}; safe rate={safe.mean():.1%}")


if __name__ == "__main__":
    main()
