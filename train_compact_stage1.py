"""Long Stage-1 run for the compact Path MeanFlow generator.

Uses the repository's frozen data construction and Path MeanFlow objective, but
keeps the compact architecture and checkpoints separate from the legacy model.
Validation is evaluated on a fixed, held-out terrain subset every N updates.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dit.Models import (
    AdditiveCapacityMatchedTokenAlignedSpatialPathMeanFlowTransformer,
    AlignmentOnlySpatialPathMeanFlowTransformer,
    AlignmentOnlyV2SpatialPathMeanFlowTransformer,
    CompactPathMeanFlowTransformer,
    PathXYSpatialPathMeanFlowTransformer,
    SharedXYSpatialPathMeanFlowTransformer,
    SpatialMapPathMeanFlowTransformer,
    TaskEncoderSpatialPathMeanFlowTransformer,
    TaskMemorySpatialPathMeanFlowTransformer,
    SplitRoutingTokenAlignedSpatialPathMeanFlowTransformer,
    TokenAlignedSpatialPathMeanFlowTransformer,
)
from posterior_pipeline import (
    frozen_stage1_environment_split,
    make_partial_dataset,
    prior_transport_loss,
)
from map_config import MAP_CONFIG, discover_environments


ARCHITECTURES = {
    "compact": {
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 256,
        "d_inner": 768,
    },
    "medium": {
        "n_layers": 6,
        "n_heads": 6,
        "d_model": 384,
        "d_inner": 1152,
    },
    "spatial_map": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "token_aligned": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "alignment_only": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "alignment_only_v2": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "token_aligned_capacity_matched": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "token_aligned_split_routing": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "path_xy": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "shared_xy": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "task_encoder": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
    "task_memory": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
    },
}


def condition_semantics(architecture):
    if architecture in (
        "token_aligned",
        "token_aligned_capacity_matched",
        "token_aligned_split_routing",
    ):
        return "token_aligned_coordinate_pe_v1"
    if architecture == "alignment_only_v2":
        return "alignment_only_coordinate_pe_v2"
    if architecture == "alignment_only":
        return "alignment_only_coordinate_pe_v1"
    if architecture == "path_xy":
        return "path_xy_linear_hint_v1"
    if architecture == "shared_xy":
        return "shared_xy_linear_hint_v1"
    if architecture == "task_encoder":
        return "task_encoder_feature_fuse_v2"
    if architecture == "task_memory":
        return "task_memory_patch_sg_sa_v1"
    return "single_vector_global_condition_v1"


def build_model(
    architecture,
    device,
    dropout=0.1,
    map_layout="single_12",
    task_cond_mode="direction_scale",
    condition_fusion="additive",
    coord_pe_max_freq=16.0,
    path_xy_alpha=PathXYSpatialPathMeanFlowTransformer.DEFAULT_PATH_XY_ALPHA,
):
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; "
            f"choose from {sorted(ARCHITECTURES)}"
        )
    args = {**ARCHITECTURES[architecture], "dropout": dropout}
    if architecture == "spatial_map":
        model_class = SpatialMapPathMeanFlowTransformer
        model = model_class(map_layout=map_layout, **args).to(device)
    elif architecture == "token_aligned":
        model_class = TokenAlignedSpatialPathMeanFlowTransformer
        model = model_class(
            map_layout=map_layout,
            task_cond_mode=task_cond_mode,
            condition_fusion=condition_fusion,
            coord_pe_max_freq=coord_pe_max_freq,
            **args,
        ).to(device)
    elif architecture in ("alignment_only", "alignment_only_v2"):
        model_class = (
            AlignmentOnlyV2SpatialPathMeanFlowTransformer
            if architecture == "alignment_only_v2"
            else AlignmentOnlySpatialPathMeanFlowTransformer
        )
        model = model_class(
            map_layout=map_layout,
            coord_pe_max_freq=coord_pe_max_freq,
            **args,
        ).to(device)
    elif architecture == "token_aligned_capacity_matched":
        model_class = (
            AdditiveCapacityMatchedTokenAlignedSpatialPathMeanFlowTransformer
        )
        model = model_class(
            map_layout=map_layout,
            task_cond_mode=task_cond_mode,
            coord_pe_max_freq=coord_pe_max_freq,
            **args,
        ).to(device)
    elif architecture == "token_aligned_split_routing":
        model_class = (
            SplitRoutingTokenAlignedSpatialPathMeanFlowTransformer
        )
        model = model_class(
            map_layout=map_layout,
            task_cond_mode=task_cond_mode,
            coord_pe_max_freq=coord_pe_max_freq,
            **args,
        ).to(device)
    elif architecture == "path_xy":
        model_class = PathXYSpatialPathMeanFlowTransformer
        model = model_class(
            map_layout=map_layout,
            path_xy_alpha=path_xy_alpha,
            **args,
        ).to(device)
    elif architecture == "shared_xy":
        model_class = SharedXYSpatialPathMeanFlowTransformer
        model = model_class(
            map_layout=map_layout,
            path_xy_alpha=path_xy_alpha,
            **args,
        ).to(device)
    elif architecture == "task_encoder":
        model_class = TaskEncoderSpatialPathMeanFlowTransformer
        model = model_class(map_layout=map_layout, **args).to(device)
    elif architecture == "task_memory":
        model_class = TaskMemorySpatialPathMeanFlowTransformer
        model = model_class(map_layout=map_layout, **args).to(device)
    else:
        if map_layout != "single_12":
            raise ValueError(
                "map_layout only applies to spatial_map or token_aligned"
            )
        model_class = CompactPathMeanFlowTransformer
        model = model_class(**args).to(device)
    return model, args


def evaluate(model, loader, device, max_batches=None, seed=20260815):
    was_training = model.training
    model.eval()
    losses = []
    flows = []
    endpoints = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            generator = torch.Generator(device=device).manual_seed(
                int(seed) + batch_index
            )
            loss, terms = prior_transport_loss(
                model, batch, device, generator=generator
            )
            losses.append(float(loss.detach().cpu()))
            flows.append(float(terms["flow"].cpu()))
            endpoints.append(float(terms["endpoint"].cpu()))
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "flow": float(np.mean(flows)) if flows else float("nan"),
        "endpoint": float(np.mean(endpoints)) if endpoints else float("nan"),
        "batches": len(losses),
    }
    model.train(was_training)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataFolder", default="data/dataset1")
    parser.add_argument("--output-dir", default="diagnostics/compact_stage1_long")
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="compact")
    parser.add_argument(
        "--map-layout",
        choices=SpatialMapPathMeanFlowTransformer.MAP_LAYOUTS,
        default="single_12",
        help="Static map memory layout for spatial_map / token_aligned.",
    )
    parser.add_argument(
        "--task-cond-mode",
        choices=TokenAlignedSpatialPathMeanFlowTransformer.TASK_COND_MODES,
        default="direction_scale",
        help="Residual task-condition content for token_aligned only.",
    )
    parser.add_argument(
        "--condition-fusion",
        choices=TokenAlignedSpatialPathMeanFlowTransformer.CONDITION_FUSIONS,
        default="additive",
        help="time/task modulation fusion for token_aligned only.",
    )
    parser.add_argument(
        "--coord-pe-max-freq",
        type=float,
        default=16.0,
        help="Maximum Fourier frequency of the coordinate encoding.",
    )
    parser.add_argument(
        "--path-xy-alpha",
        type=float,
        default=PathXYSpatialPathMeanFlowTransformer.DEFAULT_PATH_XY_ALPHA,
        help="Fixed scale of the path_xy Linear(2,d) hint. Ignored otherwise.",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--split-seed", type=int, default=20260814)
    parser.add_argument("--train-environments", type=int, default=80)
    parser.add_argument("--validation-environments", type=int, default=20)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0, also write stage1_update_{N}.pth at those evals.",
    )
    parser.add_argument("--eval-seed", type=int, default=20260815)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-contexts", type=int, default=400)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for long compact Stage-1 training")
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    environments = discover_environments(
        Path(args.dataFolder) / "train",
        expected_count=MAP_CONFIG.expected_environments,
    )
    split = frozen_stage1_environment_split(
        environments,
        seed=args.split_seed,
        train_count=args.train_environments,
        validation_count=args.validation_environments,
    )
    train_set, train_envs = make_partial_dataset(
        args.dataFolder,
        "train",
        mask_seed=args.seed,
        p_mask=0.5,
        mask_mode="stage1_demo_valid",
        environment_names=split["train"],
        dynamic_mask_noise=True,
    )
    validation_set, validation_envs = make_partial_dataset(
        args.dataFolder,
        "val",
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
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed + 1000),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model, architecture_args = build_model(
        args.architecture,
        device,
        map_layout=args.map_layout,
        task_cond_mode=args.task_cond_mode,
        condition_fusion=args.condition_fusion,
        coord_pe_max_freq=args.coord_pe_max_freq,
        path_xy_alpha=args.path_xy_alpha,
    )
    model_args = {
        **architecture_args,
        "coordinate_scale": model.coordinate_scale,
        "map_channels": 4,
        "use_radial_output": False,
        "architecture": args.architecture,
        "map_layout": getattr(model, "map_layout", "none"),
        "output_head": getattr(model, "OUTPUT_HEAD_TYPE", "unknown"),
    }
    if args.architecture in (
        "token_aligned",
        "token_aligned_capacity_matched",
        "token_aligned_split_routing",
        "alignment_only",
        "alignment_only_v2",
    ):
        model_args["coord_pe_max_freq"] = getattr(
            model, "coord_pe_max_freq"
        )
    if args.architecture in (
        "token_aligned",
        "token_aligned_capacity_matched",
        "token_aligned_split_routing",
    ):
        model_args["task_cond_mode"] = getattr(model, "task_cond_mode")
        model_args["condition_fusion"] = getattr(
            model, "condition_fusion"
        )
    if args.architecture in ("path_xy", "shared_xy"):
        model_args["path_xy_alpha"] = getattr(model, "path_xy_alpha")
        model_args["path_token_semantics"] = getattr(
            model, "PATH_TOKEN_SEMANTICS"
        )
    if args.architecture == "task_encoder":
        model_args["task_fusion"] = "concat_linear_map_token_and_sg_vector"
    if args.architecture == "task_memory":
        model_args["patch_size"] = getattr(model, "PATCH_SIZE")
        model_args["task_sa_layers"] = getattr(model, "TASK_SA_LAYERS")
    model_args["parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_meanflow_generator = torch.Generator(device=device).manual_seed(
        args.seed + 2000
    )
    history = []
    best_loss = float("inf")
    best_update = None
    start_update = 0
    evaluations_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("architecture") != args.architecture:
            raise ValueError(
                "Resume checkpoint architecture does not match requested "
                f"architecture {args.architecture!r}"
            )
        if checkpoint.get("environment_split") != split:
            raise ValueError("Resume checkpoint uses a different environment split")
        checkpoint_layout = checkpoint.get("map_layout", "single_12")
        if checkpoint_layout != getattr(model, "map_layout", "single_12"):
            raise ValueError(
                "Resume checkpoint uses a different map_layout: "
                f"{checkpoint_layout!r} vs {getattr(model, 'map_layout', 'single_12')!r}"
            )
        if args.architecture in (
            "token_aligned",
            "token_aligned_capacity_matched",
            "token_aligned_split_routing",
            "alignment_only",
        ):
            for key in (
                "coord_pe_max_freq",
                "task_cond_mode",
                "condition_fusion",
            ):
                checkpoint_value = checkpoint.get("model_args", {}).get(key)
                current_value = getattr(model, key, None)
                if checkpoint_value is not None and checkpoint_value != current_value:
                    raise ValueError(
                        f"Resume checkpoint {key} does not match requested "
                        f"architecture: {checkpoint_value!r} vs {current_value!r}"
                    )
        if args.architecture in ("path_xy", "shared_xy"):
            checkpoint_alpha = checkpoint.get("model_args", {}).get(
                "path_xy_alpha"
            )
            if (
                checkpoint_alpha is not None
                and float(checkpoint_alpha) != float(model.path_xy_alpha)
            ):
                raise ValueError(
                    "Resume checkpoint path_xy_alpha does not match requested "
                    f"architecture: {checkpoint_alpha!r} vs {model.path_xy_alpha!r}"
                )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_update = int(checkpoint["update"])
        history = list(checkpoint.get("history", []))
        resume_validation = evaluate(
            model,
            validation_loader,
            device,
            max_batches=args.validation_batches,
            seed=args.eval_seed,
        )
        best_loss = float(resume_validation["loss"])
        best_update = start_update
        evaluations_without_improvement = 0
        print(
            f"resumed={args.resume} update={start_update} "
            f"best_update={best_update} best_val={best_loss:.7f}",
            flush=True,
        )
    if start_update >= args.updates:
        raise ValueError(
            f"Resume update {start_update} already reached target {args.updates}"
        )
    train_iter = iter(train_loader)
    train_window = []
    flow_window = []
    endpoint_window = []
    last_update = start_update
    stop_reason = "maximum_updates"
    model.train()
    for update in range(start_update + 1, args.updates + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = prior_transport_loss(
            model,
            batch,
            device,
            generator=train_meanflow_generator,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite train loss at update {update}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_update = update
        train_window.append(float(loss.detach().cpu()))
        flow_window.append(float(terms["flow"].cpu()))
        endpoint_window.append(float(terms["endpoint"].cpu()))
        if update % args.eval_every == 0 or update == 1:
            validation = evaluate(
                model,
                validation_loader,
                device,
                max_batches=args.validation_batches,
                seed=args.eval_seed,
            )
            train_record = {
                "update": update,
                "train_loss": float(np.mean(train_window)),
                "train_flow": float(np.mean(flow_window)),
                "train_endpoint": float(np.mean(endpoint_window)),
                "train_batches": len(train_window),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "validation": validation,
            }
            train_window.clear()
            flow_window.clear()
            endpoint_window.clear()
            history.append(train_record)
            print(
                f"update={update} train={train_record['train_loss']:.7f} "
                f"val={validation['loss']:.7f} flow={validation['flow']:.7f} "
                f"grad={train_record['gradient_norm']:.4f}",
                flush=True,
            )
            metadata = {
                "stage": "stage1",
                "architecture": args.architecture,
                "map_layout": getattr(model, "map_layout", "none"),
                "output_head": getattr(model, "OUTPUT_HEAD_TYPE", "unknown"),
                "representation_semantic_version": "gauge_fixed_first_order_projected_bridge_44d_v1",
                "update": update,
                "train_loss": train_record["train_loss"],
                "val_loss": validation["loss"],
                "history": history,
                "environment_split": split,
                "train_environments": train_envs,
                "validation_environments": validation_envs,
                "validation_contexts": len(validation_set),
                "model_args": model_args,
                "optimizer_state_dict": optimizer.state_dict(),
                "mask_seed": args.seed,
                "p_mask": 0.5,
                "input_mask_semantics": "normal_channels_masked_with_gaussian_noise_plus_explicit_support_mask",
                "condition_semantics": condition_semantics(args.architecture),
                "checkpoint_role": "long_stage1_diagnostic",
                "early_stopping": {
                    "selection_metric": "fixed_source_validation_meanflow_loss",
                    "eval_seed": args.eval_seed,
                    "patience_evals": args.patience_evals,
                    "min_delta": args.min_delta,
                    "evaluations_without_improvement": evaluations_without_improvement,
                },
            }
            improved = validation["loss"] < best_loss - args.min_delta
            if improved:
                best_loss = validation["loss"]
                best_update = update
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            metadata["early_stopping"]["evaluations_without_improvement"] = (
                evaluations_without_improvement
            )
            if improved:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        **metadata,
                        "best_update": best_update,
                    },
                    output / "stage1_best.pth",
                )
            torch.save(
                {"model_state_dict": model.state_dict(), **metadata},
                output / "stage1_last.pth",
            )
            if args.save_every > 0 and update % args.save_every == 0:
                snapshot = {
                    key: value
                    for key, value in metadata.items()
                    if key != "optimizer_state_dict"
                }
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        **snapshot,
                        "best_update": best_update,
                    },
                    output / f"stage1_update_{update}.pth",
                )
            if evaluations_without_improvement >= args.patience_evals:
                stop_reason = "validation_patience_exhausted"
                print(
                    f"early_stop update={update} best_update={best_update} "
                    f"best_val={best_loss:.7f}",
                    flush=True,
                )
                break

    summary = {
        "status": "completed",
        "architecture": args.architecture,
        "map_layout": getattr(model, "map_layout", "none"),
        "output_head": getattr(model, "OUTPUT_HEAD_TYPE", "unknown"),
        "condition_semantics": condition_semantics(args.architecture),
        "device": torch.cuda.get_device_name(0),
        "updates": last_update,
        "best_update": best_update,
        "best_validation_loss": best_loss,
        "stop_reason": stop_reason,
        "evaluations_without_improvement": evaluations_without_improvement,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "path_xy_alpha": getattr(model, "path_xy_alpha", None),
        "path_token_semantics": getattr(model, "PATH_TOKEN_SEMANTICS", None),
        "environment_split": split,
        "train_contexts": len(train_set),
        "validation_contexts": len(validation_set),
        "history": history,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
