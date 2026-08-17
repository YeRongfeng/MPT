"""Inference-only map CA-off diagnosis for the path_xy screen.

Compares completed baseline and path_xy checkpoints on source-paired
map-swap and map-CA-off trajectory RMS. No weights are modified.
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

from posterior_pipeline import make_partial_dataset
from train_compact_stage1 import build_model


class ZeroMapCrossAttention(nn.Module):
    def forward(self, query, key, value):
        return torch.zeros_like(query)


def _norm_stats(tensor):
    norms = torch.linalg.vector_norm(tensor.detach().float(), dim=-1)
    return {
        "mean": float(norms.mean()),
        "median": float(norms.median()),
        "p90": float(torch.quantile(norms, 0.9)),
        "max": float(norms.max()),
    }


def _trajectory_delta(model, state_a, state_b, start, goal):
    geometry_a = model.evaluate_trajectory_state(state_a, start, goal)
    geometry_b = model.evaluate_trajectory_state(state_b, start, goal)
    delta = geometry_a["position"] - geometry_b["position"]
    return torch.sqrt(delta.square().mean(dim=(1, 2)))


def _path_components(model, state, start, goal, map_input=None):
    if hasattr(model, "path_token_components"):
        components = model.path_token_components(state, start, goal)
        result = {
            "path_mlp": components["path_mlp"],
            "path_index_pe": components["path_index_pe"],
            "path_xy_hint": components["path_xy_hint"],
            "path_tokens": components["path_tokens"],
        }
        if map_input is not None and hasattr(model, "map_token_components"):
            map_components = model.map_token_components(map_input, start, goal)
            result["map_xy_hint"] = map_components["map_xy_hint"]
        return result
    path_mlp = model.path_patchify(state)
    path_index = model.path_pos_embed.expand_as(path_mlp)
    return {
        "path_mlp": path_mlp,
        "path_index_pe": path_index,
        "path_xy_hint": None,
        "path_tokens": path_mlp + path_index,
    }


def _load_arm(run_dir, device):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    architecture = summary["architecture"]
    build_kwargs = {}
    if architecture in ("path_xy", "shared_xy"):
        build_kwargs["path_xy_alpha"] = summary.get("path_xy_alpha", 0.1)
    model, _ = build_model(
        architecture,
        device,
        map_layout=summary.get("map_layout", "single_12"),
        **build_kwargs,
    )
    checkpoint_path = run_dir / "stage1_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.use_gradient_checkpoint = False
    return {
        "name": architecture,
        "run_dir": str(run_dir),
        "summary": summary,
        "model": model,
        "update": checkpoint.get("update"),
        "checkpoint": str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[
            "diagnostics/path_xy_screen/baseline",
            "diagnostics/path_xy_screen/path_xy",
            "diagnostics/path_xy_screen/shared_xy",
        ],
    )
    parser.add_argument("--contexts", type=int, default=32)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default="diagnostics/path_xy_screen/ca_off_diagnosis.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnosis")
    device = torch.device(args.device)

    arms = [_load_arm(path, device) for path in args.run_dirs]
    split = arms[0]["summary"]["environment_split"]
    if any(arm["summary"]["environment_split"] != split for arm in arms):
        raise RuntimeError("environment splits differ between arms")

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
    loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    source_generator = torch.Generator(device=device).manual_seed(args.seed + 5000)
    source_bank = torch.randn(
        contexts, args.sources, 22, 2, generator=source_generator, device=device
    )

    component_rows = []
    sensitivity_rows = []
    with torch.no_grad():
        for context_offset, batch in enumerate(loader):
            contexts_in_batch = int(batch["map"].shape[0])
            indices = [int(v) for v in batch["dataset_index"]]
            maps = batch["map"].float().to(device)
            physical_start = batch["start_pose"].float().to(device)
            physical_goal = batch["goal_pose"].float().to(device)
            start_n = torch.zeros(contexts_in_batch, 4, device=device)
            goal_n = torch.zeros(contexts_in_batch, 4, device=device)
            start_n[:, :2] = physical_start[:, :2] / 10.0
            goal_n[:, :2] = physical_goal[:, :2] / 10.0
            start_n[:, 2] = torch.cos(physical_start[:, 2])
            start_n[:, 3] = torch.sin(physical_start[:, 2])
            goal_n[:, 2] = torch.cos(physical_goal[:, 2])
            goal_n[:, 3] = torch.sin(physical_goal[:, 2])
            source = source_bank[
                [
                    context_offset * args.batch_size + local
                    for local in range(contexts_in_batch)
                ],
                0,
            ]
            one = torch.ones(contexts_in_batch, device=device)
            zero = torch.zeros(contexts_in_batch, device=device)
            donor_indices = [
                (i + 1) % len(val_set) for i in range(contexts_in_batch)
            ]
            donor_maps = torch.stack(
                [val_set[i]["map"] for i in donor_indices]
            ).float().to(device)

            for arm in arms:
                model = arm["model"]
                name = arm["name"]
                components = _path_components(
                    model, source, start_n, goal_n, map_input=maps
                )
                for key, tensor in components.items():
                    if tensor is None:
                        continue
                    component_rows.append(
                        {"arm": name, "component": key, **_norm_stats(tensor)}
                    )
                state = model(maps, source, one, zero, start_n, goal_n)
                state_swap = model(
                    donor_maps, source, one, zero, start_n, goal_n
                )
                originals = []
                for block in model.dit_blocks:
                    originals.append(block.map_cross_attn)
                    block.map_cross_attn = ZeroMapCrossAttention()
                try:
                    state_ca_off = model(
                        maps, source, one, zero, start_n, goal_n
                    )
                finally:
                    for block, original in zip(model.dit_blocks, originals):
                        block.map_cross_attn = original
                delta_swap = _trajectory_delta(
                    model, state, state_swap, start_n, goal_n
                )
                delta_ca_off = _trajectory_delta(
                    model, state, state_ca_off, start_n, goal_n
                )
                for local, index in enumerate(indices):
                    sensitivity_rows.append(
                        {
                            "arm": name,
                            "context": index,
                            "map_swap_trajectory_rms_m": float(
                                delta_swap[local]
                            ),
                            "ca_off_trajectory_rms_m": float(
                                delta_ca_off[local]
                            ),
                        }
                    )

    component_aggregate = {}
    for row in component_rows:
        component_aggregate.setdefault(row["arm"], {}).setdefault(
            row["component"], []
        ).append(row)
    sensitivity_means = {}
    for arm in arms:
        name = arm["name"]
        rows = [row for row in sensitivity_rows if row["arm"] == name]
        sensitivity_means[name] = {
            "map_swap": float(
                np.mean([row["map_swap_trajectory_rms_m"] for row in rows])
            ),
            "ca_off": float(
                np.mean([row["ca_off_trajectory_rms_m"] for row in rows])
            ),
        }

    payload = {
        "status": "inference_only_path_xy_screen_diagnosis",
        "contexts": contexts,
        "sources": args.sources,
        "arms": [
            {
                "name": arm["name"],
                "run_dir": arm["run_dir"],
                "checkpoint": arm["checkpoint"],
                "update": arm["update"],
            }
            for arm in arms
        ],
        "component_norm_rows": component_rows,
        "map_sensitivity_rows": sensitivity_rows,
        "component_aggregate": {
            arm: {
                comp: {
                    "mean": float(np.mean([item["mean"] for item in rows])),
                    "median": float(np.mean([item["median"] for item in rows])),
                }
                for comp, rows in comps.items()
            }
            for arm, comps in component_aggregate.items()
        },
        "sensitivity_means": sensitivity_means,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "wrote": str(output),
        "sensitivity_means": sensitivity_means,
        "component_aggregate": payload["component_aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
