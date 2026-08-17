"""Evaluate Q1 Stage-1 checkpoints after both arms finish.

For each run directory this reports, separately for ``stage1_best.pth`` and
``stage1_last.pth``:

* fixed-source validation MeanFlow loss;
* Safe@1 and Safe@8;
* forbidden / stability / curvature pass rates.

No training is started and no protected test split is opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_stage2_dev_selection import _evaluate_model
from map_config import MAP_CONFIG
from posterior_pipeline import (
    make_partial_dataset,
    prior_transport_loss,
)
from train_compact_stage1 import build_model


def evaluate_meanflow(model, loader, device, seed):
    model.eval()
    losses = []
    flows = []
    endpoints = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            generator = torch.Generator(device=device).manual_seed(seed + index)
            loss, terms = prior_transport_loss(
                model, batch, device, generator=generator
            )
            losses.append(float(loss.detach().cpu()))
            flows.append(float(terms["flow"].cpu()))
            endpoints.append(float(terms["endpoint"].cpu()))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "flow": float(np.mean(flows)) if flows else float("nan"),
        "endpoint": float(np.mean(endpoints)) if endpoints else float("nan"),
        "batches": len(losses),
    }


def evaluate_checkpoint(model, checkpoint_path, loaders, device, seed):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    update = checkpoint.get("update")
    meanflow = evaluate_meanflow(model, loaders["validation"], device, seed + 2000)
    endpoint = {}
    for k in (1, 8):
        metrics, _ = _evaluate_model(
            model,
            loaders["validation"],
            device,
            sources_per_context=k,
            source_seed=seed + 3000 + k,
        )
        endpoint[str(k)] = {
            "safe_at_k": metrics["safe_at_k"],
            "strict_valid_rate": metrics["strict_valid_rate"],
            "forbidden_ok_rate": metrics["forbidden_ok_rate"],
            "stability_ok_rate": metrics["stability_ok_rate"],
            "curvature_ok_rate": metrics["curvature_ok_rate"],
            "yaw_ok_rate": metrics["yaw_ok_rate"],
            "task_cost": metrics["task_cost"],
        }
    return {
        "checkpoint": str(checkpoint_path),
        "update": int(update) if update is not None else None,
        "validation_meanflow": meanflow,
        "endpoint_metrics": endpoint,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="data/dataset1")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-contexts", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="diagnostics/q1_evaluation.json")
    parser.add_argument(
        "--include-updates",
        action="store_true",
        help="Also evaluate stage1_update_*.pth snapshots.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Q1 evaluation")
    device = torch.device(args.device)
    output = {}
    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text)
        summary = json.loads((run_dir / "summary.json").read_text())
        architecture = summary["architecture"]
        map_layout = summary.get("map_layout", "single_12")
        split = summary["environment_split"]
        validation_set, _ = make_partial_dataset(
            args.data_folder,
            "val",
            compute_stability_map=True,
            mask_seed=args.seed,
            p_mask=0.5,
            mask_mode="stage1_demo_valid",
            environment_names=split["validation"],
            dynamic_mask_noise=False,
        )
        validation_set = Subset(
            validation_set,
            np.linspace(
                0,
                len(validation_set) - 1,
                min(args.validation_contexts, len(validation_set)),
                dtype=int,
            ).tolist(),
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        build_kwargs = {}
        if architecture in ("path_xy", "shared_xy"):
            build_kwargs["path_xy_alpha"] = summary.get("path_xy_alpha", 0.1)
        model, _ = build_model(
            architecture, device, map_layout=map_layout, **build_kwargs
        )
        checkpoints = [run_dir / name for name in ("stage1_best.pth", "stage1_last.pth")]
        if args.include_updates:
            checkpoints.extend(sorted(run_dir.glob("stage1_update_*.pth")))
        arm = {
            "run_dir": str(run_dir),
            "architecture": architecture,
            "map_layout": map_layout,
            "validation_contexts": len(validation_set),
            "checkpoints": [
                evaluate_checkpoint(
                    model,
                    path,
                    {"validation": validation_loader},
                    device,
                    args.seed,
                )
                for path in checkpoints
                if path.exists()
            ],
        }
        output[str(run_dir)] = arm
        print(json.dumps(arm, indent=2))
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
