#!/usr/bin/env python3
"""Diagnostic upper bound for proposal-local privileged terrain information.

This experiment reuses canonical source/proposal/teacher pairs from
``experiment_stage2_coupling.py``.  Along every fixed Stage-1 proposal it
samples the true yaw-aware ESDF at 100 B-spline points.  Two paired-teacher
Stage-2 variants receive either:

  E_value:          E_yaw(x, y, psi)
  E_value_gradient: E_yaw(x, y, psi), dE/dx, dE/dy

The feature branch is zero initialized, so both variants begin at exactly the
same Stage-1 function.  This is an oracle diagnostic, not a deployable model:
the privileged ESDF is supplied at inference on held-out maps.
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
import torch.nn.functional as F

from bspline_utils import DifferentiableBSpline
from dit.Models import (
    PathDiffusionTransformer,
    PhysicalScaledEdgeResidualRepresentation,
)
from experiment_a_representation import (
    git_metadata,
    normalize_poses,
    set_deterministic,
)
from experiment_cplus_overfit import (
    baseline_dense_and_length,
    corrected_objective,
)
from experiment_stage2_coupling import (
    aggregate_metrics,
    component_metrics,
    training_batches,
)
from grad_optimizer import compute_theta_from_xy
from map_config import MAP_CONFIG, MAP_HALF_EXTENT


VARIANTS = {
    "E_value": 1,
    "E_value_gradient": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        default="diagnostics/stage2_scale/N060_v6/optimizer_pairs.npz",
    )
    parser.add_argument(
        "--baseline-summary",
        default="diagnostics/stage2_scale/N060_v6/summary.json",
    )
    parser.add_argument("--model-dir", default="data/sim")
    parser.add_argument("--checkpoint", default="stage1_best_model.pth")
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--train-lr", type=float, default=2e-5)
    parser.add_argument("--train-grad-clip", type=float, default=1.0)
    parser.add_argument("--teacher-huber-beta", type=float, default=0.1)
    parser.add_argument("--lambda-teacher-safe", type=float, default=0.01)
    parser.add_argument("--lambda-dev", type=float, default=0.2)
    parser.add_argument("--dev-scale-m", type=float, default=1.0)
    parser.add_argument("--lambda-len", type=float, default=200.0)
    parser.add_argument("--length-ratio-limit", type=float, default=1.15)
    parser.add_argument("--safe-length-ratio", type=float, default=1.17)
    parser.add_argument("--mode-threshold-m", type=float, default=0.05)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default="diagnostics/stage2_privileged_local_upper_bound",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def decode_residuals(
    residual: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    coordinate_scale: float,
) -> torch.Tensor:
    representation = PhysicalScaledEdgeResidualRepresentation()
    start_xy = starts[:, :2] / coordinate_scale
    goal_xy = goals[:, :2] / coordinate_scale
    with torch.no_grad():
        return (
            representation.decode(residual, start_xy, goal_xy)
            * coordinate_scale
        )


def load_pairs(path: Path, coordinate_scale: float) -> Dict[str, object]:
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
            "r_star": torch.from_numpy(
                pairs["optimized_residual"].copy()
            ),
            "condition_id": torch.from_numpy(
                pairs["condition_id"].copy()
            ),
            "source_id": torch.from_numpy(pairs["source_id"].copy()),
            "split_name": pairs["split"].astype(str).tolist(),
            "environment": pairs["environment"].astype(str).tolist(),
            "path_num": pairs["path_num"].copy().tolist(),
        }
    data["cp0"] = decode_residuals(
        data["r0"], data["start"], data["goal"], coordinate_scale
    )
    data["cp_star"] = decode_residuals(
        data["r_star"], data["start"], data["goal"], coordinate_scale
    )
    return data


def subset(data: Dict[str, object], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    count = len(data["split_name"])
    return {
        key: value.index_select(0, indices)
        for key, value in data.items()
        if torch.is_tensor(value)
        and value.ndim > 0
        and value.shape[0] == count
    }


def sample_volume(
    volume: torch.Tensor,
    xy: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Trilinearly sample (B,H,W,D) volumes at physical (x,y,yaw)."""
    map_info = MAP_CONFIG.cost_map_info()
    origin = map_info["origin"]
    resolution = float(map_info["resolution"])
    width, height, yaw_bins = map_info["size"]
    x_index = torch.clamp(
        (xy[..., 0] - origin[0]) / resolution, 0.0, float(width - 1)
    )
    y_index = torch.clamp(
        (xy[..., 1] - origin[1]) / resolution, 0.0, float(height - 1)
    )
    yaw_index = torch.clamp(
        torch.remainder(yaw - origin[2], 2.0 * math.pi)
        / (2.0 * math.pi / yaw_bins),
        0.0,
        float(yaw_bins - 1),
    )
    grid = torch.stack(
        [
            2.0 * x_index / (width - 1) - 1.0,
            2.0 * y_index / (height - 1) - 1.0,
            2.0 * yaw_index / (yaw_bins - 1) - 1.0,
        ],
        dim=-1,
    ).view(volume.shape[0], xy.shape[1], 1, 1, 3)
    volume_5d = volume.permute(0, 3, 1, 2).unsqueeze(1)
    return F.grid_sample(
        volume_5d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(1).squeeze(-1).squeeze(-1)


def privileged_features(
    data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Return [ESDF, dESDF/dx, dESDF/dy] along the fixed R0 trajectory."""
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    rows: List[torch.Tensor] = []
    resolution = float(MAP_CONFIG.resolution)
    with torch.no_grad():
        for begin in range(0, data["map"].shape[0], args.feature_batch_size):
            end = min(begin + args.feature_batch_size, data["map"].shape[0])
            dense = bspline(data["cp0"][begin:end].to(device))
            yaw = compute_theta_from_xy(dense)
            esdf = data["cost_map"][begin:end].to(device)
            # Map tensors use (row=y, column=x, yaw).
            grad_y, grad_x = torch.gradient(
                esdf,
                spacing=(resolution, resolution),
                dim=(1, 2),
                edge_order=1,
            )
            rows.append(
                torch.stack(
                    [
                        sample_volume(esdf, dense, yaw),
                        sample_volume(grad_x, dense, yaw),
                        sample_volume(grad_y, dense, yaw),
                    ],
                    dim=-1,
                ).cpu()
            )
    return torch.cat(rows)


def standardize_features(
    train_features: torch.Tensor,
    heldout_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=(0, 1), keepdim=True)
    std = train_features.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    return (
        (train_features - mean) / std,
        (heldout_features - mean) / std,
        mean.flatten(),
        std.flatten(),
    )


def load_oracle_model(
    model_params: Dict[str, object],
    checkpoint: Dict[str, object],
    local_dim: int,
    device: torch.device,
) -> PathDiffusionTransformer:
    model_args = dict(model_params["model_args"])
    model_args["privileged_local_dim"] = local_dim
    model = PathDiffusionTransformer(**model_args).to(device)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    expected_missing = {
        f"privileged_local_embed.{index}.{suffix}"
        for index in (0, 2)
        for suffix in ("weight", "bias")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            f"Unexpected missing checkpoint keys: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def model_output(
    model: PathDiffusionTransformer,
    data: Dict[str, torch.Tensor],
    features: torch.Tensor,
    indices: torch.Tensor,
    coordinate_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    maps = data["map"][indices].to(device)
    starts = data["start"][indices].to(device)
    goals = data["goal"][indices].to(device)
    start_n = normalize_poses(starts, coordinate_scale)
    goal_n = normalize_poses(goals, coordinate_scale)
    count = len(indices)
    residual = model(
        maps,
        data["noise"][indices].to(device),
        torch.ones(count, device=device),
        torch.zeros(count, device=device),
        start_n,
        goal_n,
        privileged_local_features=features[indices].to(device),
    )
    control_points = (
        model.trajectory_representation.decode(
            residual, start_n[:, :2], goal_n[:, :2]
        )
        * coordinate_scale
    )
    return residual, control_points


def evaluate(
    model: PathDiffusionTransformer,
    data: Dict[str, torch.Tensor],
    features: torch.Tensor,
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
    residuals: List[torch.Tensor] = []
    control_points: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, data["map"].shape[0], args.eval_batch_size):
            indices = torch.arange(begin, min(
                begin + args.eval_batch_size, data["map"].shape[0]
            ))
            residual, control = model_output(
                model, data, features, indices, coordinate_scale, device
            )
            residuals.append(residual.cpu())
            control_points.append(control.cpu())
    residual = torch.cat(residuals)
    control = torch.cat(control_points)
    metrics = component_metrics(control, data, args, device)
    teacher_error = torch.linalg.vector_norm(
        (residual - data["r_star"]).flatten(start_dim=1), dim=1
    )
    teacher_delta = torch.linalg.vector_norm(
        (data["r0"] - data["r_star"]).flatten(start_dim=1), dim=1
    )
    relative_error = teacher_error / teacher_delta.clamp_min(1e-8)
    relative_error[teacher_delta <= 1e-5] = float("nan")
    metrics["relative_teacher_error"] = relative_error.numpy()
    return residual, control, metrics


def train_variant(
    name: str,
    local_dim: int,
    model_params: Dict[str, object],
    checkpoint: Dict[str, object],
    train_data: Dict[str, torch.Tensor],
    heldout_data: Dict[str, torch.Tensor],
    train_features: torch.Tensor,
    heldout_features: torch.Tensor,
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
) -> Dict[str, object]:
    model = load_oracle_model(
        model_params, checkpoint, local_dim, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.train_lr, weight_decay=0.0
    )
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    batches = training_batches(train_data["map"].shape[0], args)
    losses: List[float] = []
    for step, indices in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        residual, control = model_output(
            model,
            train_data,
            train_features[..., :local_dim],
            indices,
            coordinate_scale,
            device,
        )
        cp0 = train_data["cp0"][indices].to(device)
        dense0, length0 = baseline_dense_and_length(cp0, bspline)
        corrected, _ = corrected_objective(
            control,
            dense0,
            length0,
            train_data["start"][indices].to(device),
            train_data["goal"][indices].to(device),
            train_data["cost_map"][indices].to(device),
            bspline,
            args,
            device,
        )
        teacher_loss = F.smooth_l1_loss(
            residual,
            train_data["r_star"][indices].to(device),
            beta=args.teacher_huber_beta,
        )
        loss = teacher_loss + args.lambda_teacher_safe * corrected.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == len(batches):
            losses.append(float(loss.detach()))
            print(
                f"{name}: step {step}/{len(batches)}, "
                f"loss={losses[-1]:.6g}"
            )

    train_r, train_cp, train_metrics = evaluate(
        model,
        train_data,
        train_features[..., :local_dim],
        args,
        coordinate_scale,
        device,
    )
    heldout_r, heldout_cp, heldout_metrics = evaluate(
        model,
        heldout_data,
        heldout_features[..., :local_dim],
        args,
        coordinate_scale,
        device,
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
    }


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
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

    all_data = load_pairs(Path(args.pairs), coordinate_scale)
    train_indices = torch.tensor([
        index for index, split in enumerate(all_data["split_name"])
        if split == "train"
    ])
    heldout_indices = torch.tensor([
        index for index, split in enumerate(all_data["split_name"])
        if split == "heldout"
    ])
    train_data = subset(all_data, train_indices)
    heldout_data = subset(all_data, heldout_indices)
    print(
        f"Loaded canonical pairs: train={len(train_indices)}, "
        f"heldout={len(heldout_indices)}"
    )

    print("Sampling privileged ESDF values and spatial gradients along R0 ...")
    train_features = privileged_features(train_data, args, device)
    heldout_features = privileged_features(heldout_data, args, device)
    train_features, heldout_features, feature_mean, feature_std = (
        standardize_features(train_features, heldout_features)
    )
    torch.save(
        {
            "train": train_features,
            "heldout": heldout_features,
            "mean": feature_mean,
            "std": feature_std,
            "channels": ["esdf", "gradient_x", "gradient_y"],
        },
        output_dir / "privileged_local_features.pt",
    )

    initial_metrics = {
        "train": component_metrics(
            train_data["cp0"], train_data, args, device
        ),
        "heldout": component_metrics(
            heldout_data["cp0"], heldout_data, args, device
        ),
    }
    teacher_metrics = {
        "train": component_metrics(
            train_data["cp_star"], train_data, args, device
        ),
        "heldout": component_metrics(
            heldout_data["cp_star"], heldout_data, args, device
        ),
    }
    args.train_steps = args.train_epochs * math.ceil(
        train_data["map"].shape[0] / args.train_batch_size
    )
    print(
        f"Stage-2 schedule: {args.train_epochs} epochs, "
        f"{args.train_steps} total steps"
    )

    results: Dict[str, object] = {}
    summaries: Dict[str, object] = {}
    for name in args.variants:
        print(f"\nTraining {name} ...")
        local_dim = VARIANTS[name]
        result = train_variant(
            name,
            local_dim,
            model_params,
            checkpoint,
            train_data,
            heldout_data,
            train_features,
            heldout_features,
            args,
            coordinate_scale,
            device,
        )
        results[name] = result
        summaries[name] = {}
        for split_name, split_data in (
            ("train", train_data),
            ("heldout", heldout_data),
        ):
            summaries[name][split_name] = aggregate_metrics(
                result[f"{split_name}_metrics"],
                initial_metrics[split_name],
                teacher_metrics[split_name],
                split_data["condition_id"],
                result[f"{split_name}_cp"],
                args,
            )

    baseline = json.loads(Path(args.baseline_summary).read_text())
    summary = {
        "experiment": "privileged proposal-local ESDF upper bound",
        "elapsed_seconds": time.time() - started,
        "train_samples": len(train_indices),
        "heldout_samples": len(heldout_indices),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "baseline_D_induced_coupling": baseline["results"][
            "D_induced_coupling"
        ],
        "results": summaries,
        "config": vars(args),
        "checkpoint": str(checkpoint_path.resolve()),
        "pairs": str(Path(args.pairs).resolve()),
        "code_version": git_metadata(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    np.savez_compressed(
        output_dir / "model_outputs.npz",
        **{
            f"{name}_{split}_{kind}": result[f"{split}_{kind}"].numpy()
            for name, result in results.items()
            for split in ("train", "heldout")
            for kind in ("r", "cp")
        },
    )

    report = [
        "# Privileged proposal-local information upper bound",
        "",
        f"- Canonical pairs: `{Path(args.pairs)}`",
        f"- Train/held-out samples: {len(train_indices)}/{len(heldout_indices)}",
        "- Local samples: 100 points along fixed Stage-1 R0",
        "- Oracle features are used in both training and held-out inference.",
        "",
        "| Variant | Split | Recovery median | Teacher error median | "
        "Mode keep | safe@K | Abnormal |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    display = {
        "D_baseline": baseline["results"]["D_induced_coupling"],
        **summaries,
    }
    for name, values in display.items():
        for split_name in ("train", "heldout"):
            row = values[split_name]
            report.append(
                f"| {name} | {split_name} | "
                f"{row['recovery_safe']['median']:.3f} | "
                f"{row['relative_teacher_error']['median']:.3f} | "
                f"{row['mode_preservation_rate']:.1%} | "
                f"{row['safe_at_k']:.1%} | "
                f"{row['abnormal_rate']:.1%} |"
            )
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"\nCompleted. Results: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
