"""Inference-only Q1 geometry diagnosis.

Compares the two Q1 checkpoints on:

1. hidden-state component norms (path MLP / index PE / coordinate PE;
   map CNN / grid PE / canonical coordinate PE);
2. source-paired map-swap and map-CA-off trajectory sensitivity;
3. canonical control-point second differences and decoded curvature;
4. CNN 12x12 feature-center-of-mass versus the assumed linspace centers.

No weights are modified and no training is started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_config import SAFETY_COST_CONFIG
from posterior_pipeline import make_partial_dataset
from train_compact_stage1 import build_model


class ZeroMapCrossAttention(nn.Module):
    def forward(self, query, key, value):
        return torch.zeros_like(query)


def _norm_stats(tensor):
    tensor = tensor.detach().float()
    norms = torch.linalg.vector_norm(tensor, dim=-1)
    return {
        "mean": float(norms.mean()),
        "median": float(norms.median()),
        "p90": float(torch.quantile(norms, 0.9)),
        "max": float(norms.max()),
    }


def _controls_second_difference(model, state, start, goal):
    start_yaw = torch.atan2(start[:, 3], start[:, 2])
    goal_yaw = torch.atan2(goal[:, 3], goal[:, 2])
    delta = goal[:, :2] - start[:, :2]
    distance = torch.linalg.vector_norm(delta, dim=1)
    theta = torch.atan2(delta[:, 1], delta[:, 0])
    start_direction = torch.stack(
        [torch.cos(start_yaw - theta), torch.sin(start_yaw - theta)], dim=1
    )
    goal_direction = torch.stack(
        [torch.cos(goal_yaw - theta), torch.sin(goal_yaw - theta)], dim=1
    )
    controls = model.trajectory_representation.canonical_control_points(
        state, start_direction, goal_direction
    )
    d2 = controls[:, 2:] - 2.0 * controls[:, 1:-1] + controls[:, :-2]
    return controls, d2


def _map_components(model, map_input, start, goal, state):
    """Return component tensors for the architecture-specific forward."""
    batch = map_input.shape[0]
    if hasattr(model, "_task_frame") and hasattr(model, "coord_pe"):
        distance, theta, start_direction, goal_direction = model._task_frame(
            start, goal
        )
        rotated_map = model._rotate_horizontal_normals(map_input, theta)
        levels = model._encode_map_levels(rotated_map)
        features = levels["12"]
        cnn = model.reorder_dims(features)
        grid = model.map_position_enc(
            torch.zeros_like(cnn), conv_shape=features.shape[-2:]
        )
        centers = model.map_feature_centers_norm.to(
            device=cnn.device, dtype=cnn.dtype
        )
        canonical_centers = model._canonical_xy(
            centers.unsqueeze(0).expand(batch, -1, -1),
            start,
            distance,
            theta,
        )
        map_coord_pe = model.coord_pe(canonical_centers)
        map_tokens = cnn + grid + map_coord_pe
        free_controls = (
            model.trajectory_representation.canonical_control_points(
                state, start_direction, goal_direction
            )[:, 2:24]
        )
        path_coord_pe = model.coord_pe(free_controls)
        path_mlp = model.path_patchify(state)
        path_index = model.path_pos_embed.expand(batch, -1, -1)
        return {
            "map_cnn": cnn,
            "map_grid_pe": grid,
            "map_coord_pe": map_coord_pe,
            "map_tokens": map_tokens,
            "path_mlp": path_mlp,
            "path_index_pe": path_index,
            "path_coord_pe": path_coord_pe,
            "path_tokens": path_mlp + path_index + path_coord_pe,
        }
    levels = model._encode_map_levels(map_input)
    features = levels["12"]
    cnn = model.reorder_dims(features)
    grid = model.map_position_enc(
        torch.zeros_like(cnn), conv_shape=features.shape[-2:]
    )
    path_mlp = model.path_patchify(state)
    path_index = model.path_pos_embed.expand(batch, -1, -1)
    return {
        "map_cnn": cnn,
        "map_grid_pe": grid,
        "map_coord_pe": None,
        "map_tokens": cnn + grid,
        "path_mlp": path_mlp,
        "path_index_pe": path_index,
        "path_coord_pe": None,
        "path_tokens": path_mlp + path_index,
    }


def _trajectory_delta(model, state_a, state_b, start, goal):
    geometry_a = model.evaluate_trajectory_state(state_a, start, goal)
    geometry_b = model.evaluate_trajectory_state(state_b, start, goal)
    delta = geometry_a["position"] - geometry_b["position"]
    return torch.sqrt(delta.square().mean(dim=(1, 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint",
                        default="diagnostics/q1_baseline/stage1_best.pth")
    parser.add_argument("--alignment-checkpoint",
                        default="diagnostics/q1_alignment_only/stage1_best.pth")
    parser.add_argument("--baseline-summary",
                        default="diagnostics/q1_baseline/summary.json")
    parser.add_argument("--alignment-summary",
                        default="diagnostics/q1_alignment_only/summary.json")
    parser.add_argument("--contexts", type=int, default=48)
    parser.add_argument("--sources", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="diagnostics/q1_geometry_diagnosis.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnosis")
    device = torch.device(args.device)

    baseline_summary = json.loads(Path(args.baseline_summary).read_text())
    alignment_summary = json.loads(Path(args.alignment_summary).read_text())
    if baseline_summary["environment_split"] != alignment_summary["environment_split"]:
        raise RuntimeError("environment splits differ between arms")
    split = baseline_summary["environment_split"]

    val_set, _ = make_partial_dataset(
        "data/dataset1",
        "val",
        compute_stability_map=False,
        mask_seed=args.seed,
        p_mask=0.5,
        mask_mode="stage1_demo_valid",
        environment_names=split["validation"],
        dynamic_mask_noise=False,
    )
    contexts = min(args.contexts, len(val_set))
    val_set = Subset(
        val_set,
        np.linspace(0, len(val_set) - 1, contexts, dtype=int).tolist(),
    )
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    baseline_model, _ = build_model("spatial_map", device, map_layout="single_12")
    alignment_model, _ = build_model("alignment_only", device, map_layout="single_12")
    baseline_model.load_state_dict(
        torch.load(args.baseline_checkpoint, map_location=device)["model_state_dict"],
        strict=True,
    )
    alignment_model.load_state_dict(
        torch.load(args.alignment_checkpoint, map_location=device)["model_state_dict"],
        strict=True,
    )
    baseline_model.eval()
    alignment_model.eval()
    baseline_model.use_gradient_checkpoint = False
    alignment_model.use_gradient_checkpoint = False

    source_generator = torch.Generator(device=device).manual_seed(args.seed + 5000)
    source_bank = torch.randn(
        contexts, args.sources, 22, 2, generator=source_generator, device=device
    )

    component_rows = []
    sensitivity_rows = []
    geometry_rows = []
    map_rows = {}
    original_map_cross_attns = {}

    with torch.no_grad():
        for context_offset, batch in enumerate(loader):
            contexts_in_batch = int(batch["map"].shape[0])
            indices = [int(v) for v in batch["dataset_index"]]
            maps = batch["map"].float().to(device)
            physical_start = batch["start_pose"].float().to(device)
            physical_goal = batch["goal_pose"].float().to(device)
            # normalize poses as training does
            start_n = torch.zeros(contexts_in_batch, 4, device=device)
            goal_n = torch.zeros(contexts_in_batch, 4, device=device)
            start_n[:, :2] = physical_start[:, :2] / 10.0
            goal_n[:, :2] = physical_goal[:, :2] / 10.0
            start_n[:, 2] = torch.cos(physical_start[:, 2])
            start_n[:, 3] = torch.sin(physical_start[:, 2])
            goal_n[:, 2] = torch.cos(physical_goal[:, 2])
            goal_n[:, 3] = torch.sin(physical_goal[:, 2])
            source = source_bank[
                [context_offset * args.batch_size + local for local in range(contexts_in_batch)],
                0,
            ]
            one = torch.ones(contexts_in_batch, device=device)
            zero = torch.zeros(contexts_in_batch, device=device)

            # Component norm audit on the first source of each context.
            for name, model, checkpoint_summary in (
                ("baseline", baseline_model, baseline_summary),
                ("alignment_only", alignment_model, alignment_summary),
            ):
                components = _map_components(model, maps, start_n, goal_n, source)
                for key, tensor in components.items():
                    if tensor is None:
                        continue
                    component_rows.append({
                        "arm": name,
                        "component": key,
                        **_norm_stats(tensor),
                    })
                state = model(maps, source, one, zero, start_n, goal_n)
                controls, d2 = _controls_second_difference(
                    model, state, start_n, goal_n
                )
                geometry = model.evaluate_trajectory_state(state, start_n, goal_n)
                max_curvature = geometry["curvature"].amax(dim=1)
                geometry_rows.extend([
                    {
                        "arm": name,
                        "context": index,
                        "control_d2_rms": float(
                            torch.sqrt(d2.square().mean(dim=(1, 2)))[local]
                        ),
                        "control_d2_max": float(d2[local].abs().max()),
                        "max_curvature": float(max_curvature[local]),
                        "curvature_pass": float(
                            (max_curvature[local]
                             <= SAFETY_COST_CONFIG.curvature_limit
                             + SAFETY_COST_CONFIG.hard_constraint_epsilon)
                        ),
                    }
                    for local, index in enumerate(indices)
                ])

            # Map sensitivity: donor map from the next context block.
            donor_indices = [
                (i + 1) % len(val_set) for i in range(contexts_in_batch)
            ]
            donor_maps = torch.stack([val_set[i]["map"] for i in donor_indices]).float().to(device)
            map_rows["batch"] = context_offset
            for name, model in (
                ("baseline", baseline_model),
                ("alignment_only", alignment_model),
            ):
                state = model(maps, source, one, zero, start_n, goal_n)
                state_swap = model(donor_maps, source, one, zero, start_n, goal_n)
                delta_swap = _trajectory_delta(
                    model, state, state_swap, start_n, goal_n
                )
                # CA-off via temporary zero modules.
                for block in model.dit_blocks:
                    original_map_cross_attns[(name, id(block))] = block.map_cross_attn
                    block.map_cross_attn = ZeroMapCrossAttention()
                state_ca_off = model(maps, source, one, zero, start_n, goal_n)
                for block in model.dit_blocks:
                    block.map_cross_attn = original_map_cross_attns[(name, id(block))]
                delta_ca_off = _trajectory_delta(
                    model, state, state_ca_off, start_n, goal_n
                )
                for local, index in enumerate(indices):
                    sensitivity_rows.append({
                        "arm": name,
                        "context": index,
                        "map_swap_trajectory_rms_m": float(delta_swap[local]),
                        "ca_off_trajectory_rms_m": float(delta_ca_off[local]),
                    })

    # Feature-center probe on the baseline CNN.
    probe_model = baseline_model
    probe_input = torch.randn(1, 4, 100, 100, device=device, requires_grad=True)
    assumed = alignment_model.map_feature_centers_norm.cpu()
    # Reconstruct linspace centers for comparison; feature row/col center of
    # mass is estimated from gradient energy.
    centers = []
    for r in range(12):
        for c in range(12):
            probe_model.zero_grad()
            if probe_input.grad is not None:
                probe_input.grad.zero_()
            levels = probe_model._encode_map_levels(probe_input)
            loss = levels["12"][0, :, r, c].sum()
            loss.backward()
            grad = probe_input.grad[0].abs().sum(dim=0)
            weight = grad + 1e-9
            total = weight.sum()
            row_c = (weight.sum(dim=1) * torch.arange(100, device=device, dtype=torch.float32)).sum() / total
            col_c = (weight.sum(dim=0) * torch.arange(100, device=device, dtype=torch.float32)).sum() / total
            x_norm = -1.0 + col_c * (1.98 / 99.0)
            y_norm = -1.0 + row_c * (1.98 / 99.0)
            centers.append((float(x_norm), float(y_norm), float(assumed[12*r+c,0]), float(assumed[12*r+c,1])))
    centers = np.asarray(centers)
    center_error = np.sqrt(
        (centers[:, 0] - centers[:, 2]) ** 2
        + (centers[:, 1] - centers[:, 3]) ** 2
    )
    feature_centers = {
        "probe_center_x_mean": float(centers[:, 0].mean()),
        "probe_center_y_mean": float(centers[:, 1].mean()),
        "assumed_center_x_mean": float(centers[:, 2].mean()),
        "assumed_center_y_mean": float(centers[:, 3].mean()),
        "center_error_mean": float(center_error.mean()),
        "center_error_max": float(center_error.max()),
    }

    component_summary = {
        "baseline": {
            key: _norm_stats(torch.stack([
                torch.as_tensor(0.0)  # placeholder, replaced below
            ]))
            for key in ("unused",)
        },
        "alignment_only": {},
    }
    # Aggregate simple numeric summaries instead of stacked tensors.
    component_aggregate = {}
    for row in component_rows:
        component_aggregate.setdefault(row["arm"], {}).setdefault(row["component"], []).append(row)

    payload = {
        "status": "inference_only_q1_geometry_diagnosis",
        "contexts": contexts,
        "sources": args.sources,
        "component_norm_rows": component_rows,
        "map_sensitivity_rows": sensitivity_rows,
        "geometry_rows": geometry_rows,
        "feature_center_probe": feature_centers,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "feature_center_probe": feature_centers,
        "component_aggregate": {
            arm: {
                comp: {
                    "mean": float(np.mean([x["mean"] for x in rows])),
                    "median": float(np.mean([x["median"] for x in rows])),
                }
                for comp, rows in comps.items()
            }
            for arm, comps in component_aggregate.items()
        },
        "sensitivity_means": {
            arm: {
                "map_swap": float(np.mean([x["map_swap_trajectory_rms_m"] for x in rows])),
                "ca_off": float(np.mean([x["ca_off_trajectory_rms_m"] for x in rows])),
            }
            for arm, rows in (
                ("baseline", [x for x in sensitivity_rows if x["arm"] == "baseline"]),
                ("alignment_only", [x for x in sensitivity_rows if x["arm"] == "alignment_only"]),
            )
        },
        "geometry_means": {
            arm: {
                "control_d2_rms": float(np.mean([x["control_d2_rms"] for x in rows])),
                "control_d2_max": float(np.mean([x["control_d2_max"] for x in rows])),
                "max_curvature": float(np.mean([x["max_curvature"] for x in rows])),
                "curvature_pass": float(np.mean([x["curvature_pass"] for x in rows])),
            }
            for arm, rows in (
                ("baseline", [x for x in geometry_rows if x["arm"] == "baseline"]),
                ("alignment_only", [x for x in geometry_rows if x["arm"] == "alignment_only"]),
            )
        },
    }, indent=2))


if __name__ == "__main__":
    main()
