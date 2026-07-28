#!/usr/bin/env python3
"""Compare z-to-teacher generation with explicit R0-to-delta correction.

Both variants use the same canonical N=60 data, safety-priority teacher,
pretrained map/DiT weights, batches, epochs and loss:

  D_z_to_Rstar:       (map, source noise) -> R*
  P_R0_to_delta:      (map, fixed R0) -> R0 + delta_R

The second interface zero-initializes only its delta output layer, so its
initial prediction is exactly the frozen Stage-1 proposal.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
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
    load_model,
)
from experiment_stage2_coupling import (
    aggregate_metrics,
    component_metrics,
    mode_label,
    training_batches,
)
from map_config import MAP_CONFIG, MAP_HALF_EXTENT, SAFETY_COST_CONFIG


VARIANTS = ("D_z_to_Rstar", "P_R0_to_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        default="diagnostics/stage2_scale/N060_v6/optimizer_pairs.npz",
    )
    parser.add_argument(
        "--teacher",
        default=(
            "diagnostics/safety_priority_teacher/"
            "safety_priority_teacher.npz"
        ),
    )
    parser.add_argument("--model-dir", default="data/sim")
    parser.add_argument("--checkpoint", default="stage1_best_model.pth")
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default="diagnostics/proposal_conditioned_correction",
    )
    return parser.parse_args()


def decode(
    residual: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
) -> torch.Tensor:
    representation = PhysicalScaledEdgeResidualRepresentation()
    with torch.no_grad():
        return (
            representation.decode(
                residual,
                starts[:, :2] / MAP_HALF_EXTENT,
                goals[:, :2] / MAP_HALF_EXTENT,
            )
            * MAP_HALF_EXTENT
        )


def load_data(
    pairs_path: Path,
    teacher_path: Path,
) -> Dict[str, object]:
    with np.load(pairs_path) as pairs:
        data: Dict[str, object] = {
            "map": torch.from_numpy(pairs["maps"].copy()),
            "cost_map": torch.from_numpy(
                pairs["privileged_cost_maps"].copy()
            ),
            "start": torch.from_numpy(pairs["starts"].copy()),
            "goal": torch.from_numpy(pairs["goals"].copy()),
            "noise": torch.from_numpy(pairs["source_noise"].copy()),
            "r0": torch.from_numpy(pairs["stage1_residual"].copy()),
            "split_name": pairs["split"].astype(str),
            "environment": pairs["environment"].astype(str),
            "path_num": pairs["path_num"].copy(),
            "condition_id": torch.from_numpy(
                pairs["condition_id"].copy()
            ),
            "source_id": torch.from_numpy(pairs["source_id"].copy()),
        }
    with np.load(teacher_path) as teacher:
        for key in (
            "split", "environment", "path_num", "condition_id", "source_id"
        ):
            expected = data[
                "split_name" if key == "split" else key
            ]
            if torch.is_tensor(expected):
                expected = expected.numpy()
            actual = teacher[key].astype(str) if key in {
                "split", "environment"
            } else teacher[key]
            if not np.array_equal(expected, actual):
                raise ValueError(f"Teacher metadata mismatch: {key}")
        data["r_star"] = torch.from_numpy(teacher["r_star"].copy())
        data["cp_star"] = torch.from_numpy(teacher["cp_star"].copy())
        data["teacher_strict_safe"] = torch.from_numpy(
            teacher["strict_safe"].copy()
        )
    data["cp0"] = decode(
        data["r0"], data["start"], data["goal"]
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


class ProposalCorrectionTransformer(nn.Module):
    """Pretrained trajectory DiT with an explicit zero-initialized delta head."""

    def __init__(
        self,
        model_params: Dict[str, object],
        checkpoint: Dict[str, object],
    ):
        super().__init__()
        self.backbone = PathDiffusionTransformer(
            **model_params["model_args"]
        )
        self.backbone.load_state_dict(checkpoint["model_state_dict"])
        # Reuse the pretrained two-layer head but make correction zero at
        # initialization.  The remainder of the pretrained map/DiT stack is
        # unchanged.
        nn.init.zeros_(self.backbone.main_pred[-1].weight)
        nn.init.zeros_(self.backbone.main_pred[-1].bias)

    def forward(
        self,
        map_input: torch.Tensor,
        proposal_r0: torch.Tensor,
        start_pose: torch.Tensor,
        goal_pose: torch.Tensor,
    ) -> torch.Tensor:
        model = self.backbone
        feat_50 = model.map_fe_block1(map_input)
        feat_25 = model.map_fe_block2(feat_50)
        feat_12 = model.map_fe_block3(feat_25)
        feat_12 = model.map_fe_block4(feat_12)
        map_tokens = model.reorder_dims(feat_12)
        map_tokens = model.map_position_enc(
            map_tokens, conv_shape=feat_12.shape[-2:]
        )

        path_tokens = model.path_patchify(proposal_r0)
        x = model.layer_norm(path_tokens + model.path_pos_embed)
        x = model.dropout(x)
        batch_size = map_input.shape[0]
        timestep = torch.zeros(
            batch_size, device=map_input.device, dtype=map_input.dtype
        )
        t_emb = model.time_embedder(timestep)
        h_emb = model.time_embedder(timestep)
        start_emb = model.pose_embedder(start_pose)
        goal_emb = model.pose_embedder(goal_pose)
        condition = model.cond_mlp(torch.cat(
            [t_emb, h_emb, start_emb, goal_emb], dim=-1
        ))
        guidance = model.guidance_encoder(
            map_tokens=map_tokens,
            start_pose=start_pose,
            goal_pose=goal_pose,
            t_emb=t_emb,
            h_emb=h_emb,
        )
        for block in model.dit_blocks:
            x = block(x, guidance, condition)
        delta = model.main_pred(model.layer_norm(x))
        delta = model.project_zero_sum(delta)
        corrected = model.project_zero_sum(proposal_r0 + delta)
        return model.trajectory_representation.radial_project_residual(
            corrected, start_pose[:, :2], goal_pose[:, :2]
        )


def model_output(
    model: nn.Module,
    variant: str,
    data: Dict[str, torch.Tensor],
    indices: torch.Tensor,
    coordinate_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    maps = data["map"][indices].to(device)
    starts = data["start"][indices].to(device)
    goals = data["goal"][indices].to(device)
    start_n = normalize_poses(starts, coordinate_scale)
    goal_n = normalize_poses(goals, coordinate_scale)
    if variant == "D_z_to_Rstar":
        count = len(indices)
        residual = model(
            maps,
            data["noise"][indices].to(device),
            torch.ones(count, device=device),
            torch.zeros(count, device=device),
            start_n,
            goal_n,
        )
        representation = model.trajectory_representation
    elif variant == "P_R0_to_delta":
        residual = model(
            maps,
            data["r0"][indices].to(device),
            start_n,
            goal_n,
        )
        representation = model.backbone.trajectory_representation
    else:
        raise ValueError(variant)
    control_points = (
        representation.decode(
            residual, start_n[:, :2], goal_n[:, :2]
        )
        * coordinate_scale
    )
    return residual, control_points


def build_model(
    variant: str,
    model_params: Dict[str, object],
    checkpoint: Dict[str, object],
    device: torch.device,
) -> nn.Module:
    if variant == "D_z_to_Rstar":
        model = load_model(model_params, checkpoint, device)
    elif variant == "P_R0_to_delta":
        model = ProposalCorrectionTransformer(
            model_params, checkpoint
        ).to(device)
    else:
        raise ValueError(variant)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def evaluate(
    model: nn.Module,
    variant: str,
    data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
    residual_rows: List[torch.Tensor] = []
    control_rows: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, data["map"].shape[0], args.eval_batch_size):
            indices = torch.arange(
                begin,
                min(begin + args.eval_batch_size, data["map"].shape[0]),
            )
            residual, control = model_output(
                model,
                variant,
                data,
                indices,
                coordinate_scale,
                device,
            )
            residual_rows.append(residual.cpu())
            control_rows.append(control.cpu())
    residual = torch.cat(residual_rows)
    control = torch.cat(control_rows)
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
    variant: str,
    model_params: Dict[str, object],
    checkpoint: Dict[str, object],
    train_data: Dict[str, torch.Tensor],
    heldout_data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    coordinate_scale: float,
    device: torch.device,
) -> Dict[str, object]:
    model = build_model(variant, model_params, checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.train_lr, weight_decay=0.0
    )
    batches = training_batches(train_data["map"].shape[0], args)
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    losses: List[float] = []
    for step, indices in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        residual, control = model_output(
            model,
            variant,
            train_data,
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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.train_grad_clip
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == len(batches):
            losses.append(float(loss.detach()))
            print(
                f"{variant}: step {step}/{len(batches)}, "
                f"loss={losses[-1]:.6g}"
            )
    train_r, train_cp, train_metrics = evaluate(
        model,
        variant,
        train_data,
        args,
        coordinate_scale,
        device,
    )
    heldout_r, heldout_cp, heldout_metrics = evaluate(
        model,
        variant,
        heldout_data,
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


def teacher_safe_condition_ids(
    data: Dict[str, torch.Tensor],
    teacher_metrics: Dict[str, np.ndarray],
    safe_length_ratio: float,
) -> List[int]:
    candidate_safe = (
        (teacher_metrics["dangerous_ratio"] <= 0.0)
        & (teacher_metrics["length_ratio"] <= safe_length_ratio)
    )
    condition_ids = data["condition_id"].numpy()
    return [
        int(condition)
        for condition in np.unique(condition_ids)
        if np.any(candidate_safe[condition_ids == condition])
    ]


def subset_summary(
    metrics: Dict[str, np.ndarray],
    initial: Dict[str, np.ndarray],
    teacher: Dict[str, np.ndarray],
    control: torch.Tensor,
    data: Dict[str, torch.Tensor],
    conditions: Iterable[int],
    args: argparse.Namespace,
) -> Dict[str, object]:
    conditions = list(conditions)
    mask = torch.zeros(len(data["condition_id"]), dtype=torch.bool)
    for condition in conditions:
        mask |= data["condition_id"] == condition
    indices = torch.nonzero(mask).flatten()
    local_data = {
        key: value.index_select(0, indices)
        for key, value in data.items()
        if torch.is_tensor(value)
        and value.ndim > 0
        and value.shape[0] == len(mask)
    }
    remap = {
        int(condition): local
        for local, condition in enumerate(conditions)
    }
    local_data["condition_id"] = torch.tensor([
        remap[int(value)] for value in local_data["condition_id"]
    ])
    return aggregate_metrics(
        {key: value[indices.numpy()] for key, value in metrics.items()},
        {key: value[indices.numpy()] for key, value in initial.items()},
        {key: value[indices.numpy()] for key, value in teacher.items()},
        local_data["condition_id"],
        control.index_select(0, indices),
        args,
    )


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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

    data = load_data(Path(args.pairs), Path(args.teacher))
    train_indices = torch.from_numpy(np.flatnonzero(
        data["split_name"] == "train"
    ))
    heldout_indices = torch.from_numpy(np.flatnonzero(
        data["split_name"] == "heldout"
    ))
    train_data = subset(data, train_indices)
    heldout_data = subset(data, heldout_indices)
    args.train_steps = args.train_epochs * math.ceil(
        len(train_indices) / args.train_batch_size
    )
    print(
        f"Loaded train/heldout={len(train_indices)}/{len(heldout_indices)}; "
        f"schedule={args.train_steps} steps"
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
    results: Dict[str, object] = {}
    summaries: Dict[str, object] = {}
    for variant in args.variants:
        print(f"\nTraining {variant} ...")
        result = train_variant(
            variant,
            model_params,
            checkpoint,
            train_data,
            heldout_data,
            args,
            coordinate_scale,
            device,
        )
        results[variant] = result
        summaries[variant] = {}
        for split_name, split_data in (
            ("train", train_data),
            ("heldout", heldout_data),
        ):
            safe_conditions = teacher_safe_condition_ids(
                split_data,
                teacher_metrics[split_name],
                args.safe_length_ratio,
            )
            summaries[variant][split_name] = aggregate_metrics(
                result[f"{split_name}_metrics"],
                initial_metrics[split_name],
                teacher_metrics[split_name],
                split_data["condition_id"],
                result[f"{split_name}_cp"],
                args,
            )
            summaries[variant][
                f"{split_name}_teacher_safe"
            ] = subset_summary(
                result[f"{split_name}_metrics"],
                initial_metrics[split_name],
                teacher_metrics[split_name],
                result[f"{split_name}_cp"],
                split_data,
                safe_conditions,
                args,
            )
            summaries[variant][
                f"{split_name}_teacher_safe_condition_count"
            ] = len(safe_conditions)

    summary = {
        "experiment": "proposal-conditioned correction diagnostic",
        "elapsed_seconds": time.time() - started,
        "train_samples": len(train_indices),
        "heldout_samples": len(heldout_indices),
        "results": summaries,
        "parameters": vars(args),
        "pairs": str(Path(args.pairs).resolve()),
        "teacher": str(Path(args.teacher).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "map_config": MAP_CONFIG.to_dict(),
        "safety_cost_config": SAFETY_COST_CONFIG.to_dict(),
        "code_version": git_metadata(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    np.savez_compressed(
        output_dir / "model_outputs.npz",
        **{
            f"{variant}_{split}_{kind}": result[f"{split}_{kind}"].numpy()
            for variant, result in results.items()
            for split in ("train", "heldout")
            for kind in ("r", "cp")
        },
    )

    report = [
        "# Proposal-conditioned correction diagnostic",
        "",
        f"- Train/held-out samples: {len(train_indices)}/{len(heldout_indices)}",
        f"- Epochs/batch: {args.train_epochs}/{args.train_batch_size}",
        "",
        "| Variant | Split | Recovery median | Teacher error median | "
        "Mode keep | safe@4 | Teacher-safe subset safe@4 | Abnormal |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in args.variants:
        for split_name in ("train", "heldout"):
            values = summaries[variant][split_name]
            safe_values = summaries[variant][
                f"{split_name}_teacher_safe"
            ]
            report.append(
                f"| {variant} | {split_name} | "
                f"{values['recovery_safe']['median']:.3f} | "
                f"{values['relative_teacher_error']['median']:.3f} | "
                f"{values['mode_preservation_rate']:.1%} | "
                f"{values['safe_at_k']:.1%} | "
                f"{safe_values['safe_at_k']:.1%} | "
                f"{values['abnormal_rate']:.1%} |"
            )
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"\nCompleted: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
