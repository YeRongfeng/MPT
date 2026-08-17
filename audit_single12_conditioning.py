"""Inference-only audit of the current single_12 S/G conditioning path.

The audit keeps the map, source, time, and true decoder boundary conditions
fixed.  It changes only the S/G values supplied to AdaLN, so the measured
output change is a condition-injection effect rather than a decoder effect.
No model parameters or checkpoints are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from audit_single12_causal import audit_forward
from posterior_pipeline import make_partial_dataset, normalize_poses
from train_compact_stage1 import build_model


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pose_variant(start, goal, name):
    if name == "baseline":
        return start, goal
    if name == "position_zero":
        start = start.clone()
        goal = goal.clone()
        start[:, :2] = 0.0
        goal[:, :2] = 0.0
        return start, goal
    if name == "yaw_zero":
        start = start.clone()
        goal = goal.clone()
        start[:, 2:] = torch.tensor([1.0, 0.0], device=start.device)
        goal[:, 2:] = torch.tensor([1.0, 0.0], device=goal.device)
        return start, goal
    if name == "sg_zero":
        start = torch.zeros_like(start)
        goal = torch.zeros_like(goal)
        start[:, 2] = 1.0
        goal[:, 2] = 1.0
        return start, goal
    raise ValueError(f"unknown pose variant: {name}")


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("diagnostics/spatial_map_resolution/single_12"))
    parser.add_argument("--data-folder", default="data/dataset1")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/spatial_map_resolution/single_12/conditioning_audit"))
    parser.add_argument("--sources", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mask-seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    causal_summary = json.loads(
        (args.run_dir / "causal_audit" / "map_causal_summary.json").read_text(encoding="utf-8")
    )
    safe_arrays = np.load(args.run_dir / "causal_audit" / "safe_basin_arrays.npz")
    validation_indices = list(range(int(safe_arrays["p_safe"].shape[0])))
    selected_indices = [int(v) for v in causal_summary["selected_contexts"]]
    donor_indices = [int(v) for v in causal_summary["map_swap_donor_indices"]]
    donor_by_context = dict(zip(selected_indices, donor_indices))
    p_safe = safe_arrays["p_safe"]
    position_by_index = {index: position for position, index in enumerate(validation_indices)}
    selected_position = {index: position for position, index in enumerate(selected_indices)}
    high_count = len(selected_indices) // 2

    checkpoint = torch.load(args.run_dir / "stage1_best.pth", map_location=device)
    model, _ = build_model("spatial_map", device, map_layout="single_12")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.use_gradient_checkpoint = False

    split = run_summary["environment_split"]
    dataset, _ = make_partial_dataset(
        args.data_folder,
        "val",
        compute_stability_map=False,
        mask_seed=args.mask_seed,
        p_mask=0.5,
        mask_mode="stage1_demo_valid",
        environment_names=split["validation"],
        dynamic_mask_noise=False,
    )
    donor_poses = {}
    for index in selected_indices:
        item = dataset[donor_by_context[index]]
        donor_poses[index] = (item["start_pose"], item["goal_pose"])

    source_bank = safe_arrays["source_bank"]
    variants = ["baseline", "position_zero", "yaw_zero", "sg_zero", "sg_swapped"]
    sample_rows = []
    memory_rows = []
    loader = DataLoader(Subset(dataset, selected_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    with torch.inference_mode():
        for batch in loader:
            contexts = int(batch["map"].shape[0])
            batch_indices = [int(v) for v in batch["dataset_index"]]
            maps = batch["map"].float().to(device)
            physical_start = batch["start_pose"].float().to(device)
            physical_goal = batch["goal_pose"].float().to(device)
            true_start, true_goal = normalize_poses(physical_start, physical_goal, model.coordinate_scale, device)
            source = torch.as_tensor(
                source_bank[[position_by_index[index] for index in batch_indices], : args.sources],
                dtype=torch.float32,
                device=device,
            ).reshape(-1, model.num_edges, 2)
            repeats = int(args.sources)
            map_k = maps.repeat_interleave(repeats, dim=0)
            start_k = true_start.repeat_interleave(repeats, dim=0)
            goal_k = true_goal.repeat_interleave(repeats, dim=0)
            one = torch.ones(source.shape[0], device=device)
            zero = torch.zeros_like(one)

            baseline, _ = audit_forward(model, map_k, source, one, zero, start_k, goal_k)
            baseline_geometry = model.evaluate_trajectory_state(baseline, start_k, goal_k)
            memory = model._build_map_memory(maps)
            donor_maps = torch.stack([dataset[donor_by_context[index]]["map"] for index in batch_indices]).float().to(device)
            donor_memory = model._build_map_memory(donor_maps)
            memory_rows.append({
                "contexts": contexts,
                "map_memory_rms_vs_donor": float(torch.sqrt((memory - donor_memory).square().mean())),
                "map_memory_norm": float(torch.linalg.vector_norm(memory, dim=-1).mean()),
            })

            for name in variants:
                if name == "sg_swapped":
                    donor_start = torch.stack([donor_poses[index][0] for index in batch_indices]).float().to(device)
                    donor_goal = torch.stack([donor_poses[index][1] for index in batch_indices]).float().to(device)
                    cond_start, cond_goal = normalize_poses(donor_start, donor_goal, model.coordinate_scale, device)
                else:
                    cond_start, cond_goal = _pose_variant(true_start, true_goal, name)
                cond_start_k = cond_start.repeat_interleave(repeats, dim=0)
                cond_goal_k = cond_goal.repeat_interleave(repeats, dim=0)
                states, _ = audit_forward(model, map_k, source, one, zero, cond_start_k, cond_goal_k)
                geometry = model.evaluate_trajectory_state(states, start_k, goal_k)
                delta_state = states - baseline
                delta_position = geometry["position"] - baseline_geometry["position"]
                state_rms = torch.sqrt(delta_state.square().mean(dim=(1, 2))).cpu().numpy()
                traj_rms = torch.sqrt(delta_position.square().mean(dim=(1, 2))).cpu().numpy()
                mean_disp = torch.linalg.vector_norm(delta_position, dim=-1).mean(dim=1).cpu().numpy()
                group = "high" if selected_position[batch_indices[0]] < high_count else "low"
                for local, index in enumerate(batch_indices):
                    sl = slice(local * repeats, (local + 1) * repeats)
                    sample_rows.append({
                        "context": index,
                        "p_safe": float(p_safe[position_by_index[index]]),
                        "group": group,
                        "variant": name,
                        "state_rms": float(state_rms[sl].mean()),
                        "trajectory_rms_m": float(traj_rms[sl].mean()),
                        "mean_displacement_m": float(mean_disp[sl].mean()),
                    })

    aggregate_rows = []
    for group in ("all", "high", "low"):
        for name in variants:
            rows = [row for row in sample_rows if row["variant"] == name and (group == "all" or row["group"] == group)]
            aggregate_rows.append({
                "group": group,
                "variant": name,
                "contexts": len(rows),
                "state_rms_mean": float(np.mean([row["state_rms"] for row in rows])),
                "trajectory_rms_m_mean": float(np.mean([row["trajectory_rms_m"] for row in rows])),
                "mean_displacement_m_mean": float(np.mean([row["mean_displacement_m"] for row in rows])),
            })

    _write_csv(args.output_dir / "conditioning_audit_samples.csv", sample_rows)
    _write_csv(args.output_dir / "conditioning_audit_aggregate.csv", aggregate_rows)
    _write_json(args.output_dir / "conditioning_audit_summary.json", {
        "checkpoint": str(args.run_dir / "stage1_best.pth"),
        "checkpoint_update": checkpoint.get("update"),
        "contexts": len(selected_indices),
        "sources_per_context": args.sources,
        "variants": variants,
        "selected_contexts": selected_indices,
        "donor_contexts": donor_indices,
        "map_memory": {
            "rms_vs_donor": _summary([row["map_memory_rms_vs_donor"] for row in memory_rows]),
            "norm": _summary([row["map_memory_norm"] for row in memory_rows]),
        },
        "aggregate": aggregate_rows,
        "decoder_boundary": "all variant states decoded with the true context S/G",
        "no_training": True,
    })
    print(json.dumps({"output_dir": str(args.output_dir), "aggregate": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
