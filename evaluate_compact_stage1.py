"""Independent fixed-source evaluation for compact Stage-1 checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dit.Models import CompactPathMeanFlowTransformer, SpatialMapPathMeanFlowTransformer
from evaluate_stage2_dev_selection import _evaluate_model
from map_config import MAP_CONFIG, discover_environments
from posterior_pipeline import (
    frozen_stage1_environment_split,
    load_model,
    make_partial_dataset,
    prior_transport_loss,
)
from train_compact_stage1 import ARCHITECTURES, build_model


def evaluate_meanflow(model, loader, device, seed):
    model.eval()
    losses, flows, endpoints = [], [], []
    for index, batch in enumerate(loader):
        generator = torch.Generator(device=device).manual_seed(seed + index)
        loss, terms = prior_transport_loss(model, batch, device, generator=generator)
        losses.append(float(loss.detach().cpu()))
        flows.append(float(terms["flow"].cpu()))
        endpoints.append(float(terms["endpoint"].cpu()))
    return {
        "loss": float(np.mean(losses)),
        "flow": float(np.mean(flows)),
        "endpoint": float(np.mean(endpoints)),
        "batches": len(losses),
    }


def evaluate_checkpoint(path, model, train_loader, val_loader, device, seed):
    checkpoint = torch.load(path, map_location=device)
    checkpoint_layout = checkpoint.get("map_layout", "single_12")
    expected_layout = getattr(model, "map_layout", "single_12")
    if checkpoint_layout != expected_layout:
        raise ValueError(
            f"Checkpoint map_layout {checkpoint_layout!r} does not match "
            f"evaluator layout {expected_layout!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return {
        "checkpoint": str(path),
        "checkpoint_update": checkpoint.get("update"),
        "best_update": checkpoint.get("best_update"),
        "train": evaluate_meanflow(model, train_loader, device, seed + 1000),
        "validation": evaluate_meanflow(model, val_loader, device, seed + 2000),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataFolder", default="data/dataset1")
    parser.add_argument("--run-dir", default="diagnostics/compact_stage1_long")
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="compact")
    parser.add_argument(
        "--map-layout",
        choices=SpatialMapPathMeanFlowTransformer.MAP_LAYOUTS,
        default="single_12",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-contexts", type=int, default=400)
    parser.add_argument("--validation-contexts", type=int, default=400)
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint evaluation")
    device = torch.device("cuda")
    run_dir = Path(args.run_dir)
    with open(run_dir / "summary.json") as handle:
        summary = json.load(handle)
    split = summary["environment_split"]

    train_set, _ = make_partial_dataset(
        args.dataFolder,
        "train",
        mask_seed=args.seed,
        p_mask=0.5,
        mask_mode="stage1_demo_valid",
        environment_names=split["train"],
        dynamic_mask_noise=False,
    )
    val_set, _ = make_partial_dataset(
        args.dataFolder,
        "val",
        compute_stability_map=True,
        mask_seed=args.seed,
        p_mask=0.5,
        mask_mode="stage1_demo_valid",
        environment_names=split["validation"],
        dynamic_mask_noise=False,
    )
    train_set = Subset(train_set, np.linspace(0, len(train_set) - 1, min(args.train_contexts, len(train_set)), dtype=int).tolist())
    val_set = Subset(val_set, np.linspace(0, len(val_set) - 1, min(args.validation_contexts, len(val_set)), dtype=int).tolist())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    build_kwargs = {}
    if args.architecture in ("path_xy", "shared_xy"):
        build_kwargs["path_xy_alpha"] = summary.get("path_xy_alpha", 0.1)
    model, _ = build_model(
        args.architecture, device, map_layout=args.map_layout, **build_kwargs
    )
    checkpoint_paths = [
        run_dir / "stage1_post_best.pth",
        run_dir / "stage1_best.pth",
        run_dir / "stage1_last.pth",
    ]
    results = [
        evaluate_checkpoint(path, model, train_loader, val_loader, device, args.seed)
        for path in checkpoint_paths if path.exists()
    ]
    compact_checkpoint = run_dir / "stage1_best.pth"
    compact_state = torch.load(compact_checkpoint, map_location=device)
    if compact_state.get("architecture") != args.architecture:
        raise ValueError("Checkpoint architecture does not match evaluator")
    if compact_state.get("map_layout", "single_12") != args.map_layout:
        raise ValueError("Checkpoint map_layout does not match evaluator")
    model.load_state_dict(compact_state["model_state_dict"], strict=True)
    endpoint_results = {
        args.architecture: {
            str(k): _evaluate_model(
                model,
                val_loader,
                device,
                sources_per_context=k,
                source_seed=args.seed + 3000 + k,
            )[0]
            for k in (1, 8)
        }
    }
    post_best_path = run_dir / "stage1_post_best.pth"
    if post_best_path.exists():
        post_best_state = torch.load(post_best_path, map_location=device)
        model.load_state_dict(post_best_state["model_state_dict"], strict=True)
        endpoint_results[f"{args.architecture}_last"] = {
            "1": _evaluate_model(
                model,
                val_loader,
                device,
                sources_per_context=1,
                source_seed=args.seed + 3001,
            )[0]
        }
    baseline_path = Path("data/boundary_constrained_path_meanflow_v1/stage1_best.pth")
    if baseline_path.exists():
        baseline, _, _ = load_model(baseline_path, device)
        endpoint_results["production_baseline"] = {
            str(k): _evaluate_model(
                baseline,
                val_loader,
                device,
                sources_per_context=k,
                source_seed=args.seed + 3000 + k,
            )[0]
            for k in (1, 8)
        }
        endpoint_results["production_baseline"]["meanflow_validation"] = (
            evaluate_meanflow(baseline, val_loader, device, args.seed + 2000)
        )
    output = {
        "status": "evaluated",
        "device": torch.cuda.get_device_name(0),
        "train_contexts": len(train_set),
        "validation_contexts": len(val_set),
        "fixed_source_seed": args.seed,
        "results": results,
        "endpoint_metrics": endpoint_results,
    }
    (run_dir / "independent_evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
