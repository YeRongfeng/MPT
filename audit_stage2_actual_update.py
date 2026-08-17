#!/usr/bin/env python3
"""Development-only audit of the real Stage-2 MGDA optimizer update.

This script uses the current Stage-2 forward, raw MGDA helper, global gradient
clip, and Adam step.  It reads only the development train split.  It never
writes a training checkpoint and never evaluates validation or test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from dataLoader_dit import MASK_GENERATION_SEMANTICS
from grad_optimizer import PRIVILEGED_COST_SEMANTICS
from map_config import MAP_CONFIG, SAFETY_COST_CONFIG, discover_environments
from posterior_pipeline import (
    DIRECT_COST_CONTEXT_SEMANTICS,
    DIRECT_COST_STAGE2_SEMANTICS,
    DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS,
    DirectCostEpochDataset,
    _direct_cost_forward,
    _direct_cost_mgda_update,
    _direct_cost_optimizer_name,
    _make_direct_cost_optimizer,
    _require_current_mask_semantics,
    _require_demo_target_semantics,
    _require_main_method_model,
    load_model,
    make_partial_dataset,
)


OBJECTIVE_NAMES = ("F", "S", "K")
COMPONENT_KEYS = {
    "F": "forbidden_region",
    "S": "stability",
    "K": "curvature",
}
WEIGHTED_COMPONENT_KEYS = {
    "F": "weighted_forbidden_region",
    "S": "weighted_stability",
    "K": "weighted_curvature",
}
FIXED_WEIGHTS = {
    "F": float(SAFETY_COST_CONFIG.forbidden_weight),
    "S": float(SAFETY_COST_CONFIG.obstacle_weight),
    "K": float(SAFETY_COST_CONFIG.curvature_weight),
}
DEFAULT_CHECKPOINT = Path("data/path_meanflow/stage1_best.pth")
DEFAULT_OUTPUT = Path(
    "diagnostics/direct_cost_actual_update_mgda_20260808"
)


def _jsonable(value: Any):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _jsonable(value.detach().cpu().item())
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but CUDA is unavailable")
    return device


def _source_digest(source: torch.Tensor) -> str:
    return hashlib.sha256(
        source.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _batch_identifiers(batch, environments) -> dict[str, Any]:
    env_indices = [int(value) for value in batch["env_index"].tolist()]
    return {
        "dataset_indices": ",".join(
            str(int(value)) for value in batch["dataset_index"].tolist()
        ),
        "environment_names": ",".join(
            str(environments[index]) for index in env_indices
        ),
        "path_indices": ",".join(
            str(int(value)) for value in batch["path_index"].tolist()
        ),
        "mask_variants": ",".join(
            str(int(value)) for value in batch["mask_variant"].tolist()
        ),
        "mask_noise_seeds": ",".join(
            str(int(value)) for value in batch["mask_noise_seed"].tolist()
        ),
    }


def _mean_components(output) -> tuple[dict[str, float], dict[str, float]]:
    raw = {
        name: float(output["components"][key].mean().detach().cpu())
        for name, key in COMPONENT_KEYS.items()
    }
    weighted = {
        name: float(output["components"][key].mean().detach().cpu())
        for name, key in WEIGHTED_COMPONENT_KEYS.items()
    }
    return raw, weighted


def _forward_costs(model, batch, device, sources_per_context, source):
    with torch.no_grad():
        output = _direct_cost_forward(
            model,
            batch,
            device,
            sources_per_context=sources_per_context,
            source=source,
        )
        raw, weighted = _mean_components(output)
    return raw, weighted


def _parameter_snapshot(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def _actual_step_geometry(
    parameters,
    before_parameters,
    gradient_tuples,
):
    """Compare Adam's actual parameter displacement with raw objective grads."""
    device = parameters[0].device
    delta_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
    clipped_grad_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
    step_dots = [torch.zeros((), dtype=torch.float64, device=device) for _ in range(3)]
    for parameter, before, gradients in zip(
        parameters,
        before_parameters,
        zip(*gradient_tuples),
    ):
        delta = parameter.detach() - before
        delta_float = delta.float()
        delta_norm_sq = delta_norm_sq + delta_float.square().sum().double()
        if parameter.grad is not None:
            clipped_grad = parameter.grad.detach().float()
            clipped_grad_norm_sq = (
                clipped_grad_norm_sq + clipped_grad.square().sum().double()
            )
        for index, gradient in enumerate(gradients):
            if gradient is not None:
                step_dots[index] = step_dots[index] + (
                    delta_float * gradient.detach().float()
                ).sum().double()
    delta_norm = torch.sqrt(delta_norm_sq).detach()
    clipped_grad_norm = torch.sqrt(clipped_grad_norm_sq).detach()
    combined_dot = torch.zeros((), dtype=torch.float64, device=device)
    for parameter, before in zip(parameters, before_parameters):
        if parameter.grad is not None:
            combined_dot = combined_dot + (
                (parameter.detach() - before).float()
                * parameter.grad.detach().float()
            ).sum().double()
    cosine = float("nan")
    denominator = float(delta_norm * clipped_grad_norm)
    if denominator > 0.0:
        cosine = float(combined_dot.detach().cpu()) / denominator
    return {
        "actual_update_norm": float(delta_norm.cpu()),
        "clipped_gradient_norm": float(clipped_grad_norm.cpu()),
        "actual_update_dot_F": float(step_dots[0].detach().cpu()),
        "actual_update_dot_S": float(step_dots[1].detach().cpu()),
        "actual_update_dot_K": float(step_dots[2].detach().cpu()),
        "actual_update_cosine_with_clipped_gMGDA": cosine,
    }


def _gram_metrics(details):
    gram = torch.as_tensor(details["gram"], dtype=torch.float64, device="cpu")
    norms = [float(value) for value in details["component_norms"]]
    cosines = {}
    for first, second, suffix in ((0, 1, "FS"), (0, 2, "FK"), (1, 2, "SK")):
        denominator = norms[first] * norms[second]
        cosines[suffix] = (
            float(gram[first, second]) / denominator
            if denominator > 0.0
            else float("nan")
        )
    return {
        "norm_g_F": norms[0],
        "norm_g_S": norms[1],
        "norm_g_K": norms[2],
        "cos_FS": cosines["FS"],
        "cos_FK": cosines["FK"],
        "cos_SK": cosines["SK"],
        "H_FF": float(gram[0, 0]),
        "H_FS": float(gram[0, 1]),
        "H_FK": float(gram[0, 2]),
        "H_SS": float(gram[1, 1]),
        "H_SK": float(gram[1, 2]),
        "H_KK": float(gram[2, 2]),
    }


def _probe_row(update, before, after):
    row = {"update": int(update)}
    for name in OBJECTIVE_NAMES:
        row[f"probe_J_{name}_before"] = float(before[name])
        row[f"probe_J_{name}_after"] = float(after[name])
        row[f"probe_delta_{name}"] = float(after[name] - before[name])
        row[f"probe_weighted_{name}_before"] = float(
            before["weighted"][name]
        )
        row[f"probe_weighted_{name}_after"] = float(after["weighted"][name])
        row[f"probe_weighted_delta_{name}"] = float(
            after["weighted"][name] - before["weighted"][name]
        )
    row["probe_raw_all_decreased"] = all(
        row[f"probe_delta_{name}"] < 0.0 for name in OBJECTIVE_NAMES
    )
    return row


def _stats(rows, key):
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rate(rows, predicate):
    return float(sum(bool(predicate(row)) for row in rows) / len(rows)) if rows else None


def _build_summary(rows, probe_rows, config, checkpoint_metadata):
    same_batch = {
        "common_descent_rate": _rate(
            rows, lambda row: row["mgda_common_descent"]
        ),
        "actual_raw_all_decreased_rate": _rate(
            rows, lambda row: row["actual_raw_all_decreased"]
        ),
        "actual_first_order_all_decreased_rate": _rate(
            rows, lambda row: row["actual_first_order_all_decreased"]
        ),
        "predicted_common_but_actual_not_rate": _rate(
            rows,
            lambda row: row["mgda_common_descent"]
            and not row["actual_raw_all_decreased"],
        ),
        "predicted_common_but_actual_first_order_not_rate": _rate(
            rows,
            lambda row: row["mgda_common_descent"]
            and not row["actual_first_order_all_decreased"],
        ),
        "dot_F_MGDA": _stats(rows, "mgda_dot_F"),
        "dot_S_MGDA": _stats(rows, "mgda_dot_S"),
        "dot_K_MGDA": _stats(rows, "mgda_dot_K"),
        "actual_update_dot_F": _stats(rows, "actual_update_dot_F"),
        "actual_update_dot_S": _stats(rows, "actual_update_dot_S"),
        "actual_update_dot_K": _stats(rows, "actual_update_dot_K"),
        "actual_update_cosine_with_clipped_gMGDA": _stats(
            rows, "actual_update_cosine_with_clipped_gMGDA"
        ),
    }
    probe = {
        "raw_all_decreased_rate": _rate(
            probe_rows[1:], lambda row: row["probe_raw_all_decreased"]
        ),
        "delta_F": _stats(probe_rows[1:], "probe_delta_F"),
        "delta_S": _stats(probe_rows[1:], "probe_delta_S"),
        "delta_K": _stats(probe_rows[1:], "probe_delta_K"),
        "last_update": int(probe_rows[-1]["update"]) if probe_rows else None,
    }
    return {
        "audit": {
            "development_only": True,
            "network_training_performed": True,
            "training_checkpoint_saved": False,
            "data_split": "train_only",
            "validation_read": False,
            "external_30_read": False,
            "final_test_read": False,
            "updates_executed": int(config["updates"]),
            "audited_updates": len(rows),
            "probe_rows": len(probe_rows),
        },
        "checkpoint": checkpoint_metadata,
        "config": config,
        "cost_contract": {
            "mgda_objectives": "mean(per-trajectory raw forbidden_region, stability, curvature)",
            "point_reduction": {
                "forbidden": "top_tail_mean(forbidden_point_violation, forbidden_tail_ratio=0.05)",
                "stability": "top_tail_mean(stability_point_violation, stability_tail_ratio=0.15)",
                "curvature": "top_tail_mean(curvature_point_violation, curvature_tail_ratio=0.05)",
            },
            "source_reduction": "mean over contexts repeated sources",
            "tensorboard_named_cost_reduction": "mean(weighted component)",
            "tensorboard_named_cost_weights": FIXED_WEIGHTS,
            "tensorboard_names_match_mgda_J": False,
        },
        "same_batch": same_batch,
        "probe": probe,
        "mgda": {
            "alpha_F": _stats(rows, "alpha_F"),
            "alpha_S": _stats(rows, "alpha_S"),
            "alpha_K": _stats(rows, "alpha_K"),
            "mgda_norm": _stats(rows, "mgda_norm"),
            "mgda_ratio": _stats(rows, "mgda_ratio"),
            "active_sets": {
                label: sum(row["mgda_active_set"] == label for row in rows)
                for label in sorted({row["mgda_active_set"] for row in rows})
            },
        },
        "artifacts": {},
    }


def _make_plots(output_dir: Path, rows, probe_rows):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        (output_dir / "plot_error.txt").write_text(
            str(error) + "\n", encoding="utf-8"
        )
        return []
    output_dir.joinpath(".matplotlib").mkdir(parents=True, exist_ok=True)
    paths = []

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, name in zip(axes, OBJECTIVE_NAMES):
        updates = [row["update"] for row in rows]
        before = [row[f"J_{name}_before"] for row in rows]
        after = [row[f"J_{name}_after"] for row in rows]
        axis.plot(updates, before, "o-", label="before")
        axis.plot(updates, after, "s-", label="after")
        axis.set_ylabel(f"J_{name} raw")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("optimizer update")
    figure.suptitle("Actual training batch: raw MGDA objectives")
    figure.tight_layout()
    path = output_dir / "same_batch_before_after.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, name in zip(axes, OBJECTIVE_NAMES):
        updates = [row["update"] for row in probe_rows]
        values = [row[f"probe_J_{name}_after"] for row in probe_rows]
        axis.plot(updates, values, "o-")
        axis.set_ylabel(f"probe J_{name}")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("optimizer update (0 is initial probe)")
    figure.suptitle("Fixed train-only probe: raw objectives")
    figure.tight_layout()
    path = output_dir / "fixed_probe_trace.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    updates = [row["update"] for row in rows]
    axes[0].plot(updates, [row["mgda_dot_F"] for row in rows], label="gF dot gMGDA")
    axes[0].plot(updates, [row["mgda_dot_S"] for row in rows], label="gS dot gMGDA")
    axes[0].plot(updates, [row["mgda_dot_K"] for row in rows], label="gK dot gMGDA")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("raw first-order dots")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        updates,
        [row["actual_update_dot_F"] for row in rows],
        label="gF dot delta_theta",
    )
    axes[1].plot(
        updates,
        [row["actual_update_dot_S"] for row in rows],
        label="gS dot delta_theta",
    )
    axes[1].plot(
        updates,
        [row["actual_update_dot_K"] for row in rows],
        label="gK dot delta_theta",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("actual-step first-order dots")
    axes[1].set_xlabel("optimizer update")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle("Predicted direction versus actual optimizer displacement")
    figure.tight_layout()
    path = output_dir / "predicted_vs_actual_direction.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(updates, [row["actual_update_cosine_with_clipped_gMGDA"] for row in rows], "o-")
    axes[0].set_ylabel("cos(delta_theta, clipped gMGDA)")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(alpha=0.25)
    axes[1].plot(updates, [row["gradient_norm_pre_clip"] for row in rows], label="pre-clip")
    axes[1].plot(updates, [row["clipped_gradient_norm"] for row in rows], label="post-clip")
    axes[1].set_ylabel("gradient norm")
    axes[1].set_xlabel("optimizer update")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle("Optimizer/clip update geometry")
    figure.tight_layout()
    path = output_dir / "adam_update_alignment.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)
    return paths


def _write_report(path: Path, summary):
    same_batch = summary["same_batch"]
    probe = summary["probe"]
    contract = summary["cost_contract"]
    report = f"""# Actual-Update MGDA Mechanism Audit

This is a development-only audit. It starts from the Stage-1 checkpoint and
executes the current Stage-2 MGDA update path for `{summary['audit']['updates_executed']}`
updates on `data/dataset1/train` only. No validation, external-30, or final-test
data was read, and no training checkpoint was saved.

## Protocol

- Batch, mask, and source are held fixed for the same-batch before/after forward.
- The optimizer is the formal Stage-2 `{summary['config']['optimizer']}` with lr `{summary['config']['stage2_lr']}`
  and global clip norm `{summary['config']['grad_clip_norm']}`.
- The probe is one fixed train-only batch with fixed source noise; row 0 is its
  initial value.
- MGDA J values are raw per-trajectory F/S/K means. The current TensorBoard
  tags named `forbidden_cost`, `stability_cost`, and `curvature_cost` are
  weighted component means, with weights `{FIXED_WEIGHTS}`. Therefore those
  three existing tags are not numerically identical to MGDA J values.

## Results

| quantity | rate/value |
|---|---:|
| MGDA first-order common descent | {_fmt_rate(same_batch['common_descent_rate'])} |
| same update: all three raw costs decreased | {_fmt_rate(same_batch['actual_raw_all_decreased_rate'])} |
| actual Adam displacement first-order descending for all three | {_fmt_rate(same_batch['actual_first_order_all_decreased_rate'])} |
| predicted common but same-batch actual not all down | {_fmt_rate(same_batch['predicted_common_but_actual_not_rate'])} |
| predicted common but actual displacement not first-order down | {_fmt_rate(same_batch['predicted_common_but_actual_first_order_not_rate'])} |
| median cos(actual displacement, clipped gMGDA) | {_fmt_num(same_batch['actual_update_cosine_with_clipped_gMGDA']['median'])} |
| fixed probe all-three-down rate | {_fmt_rate(probe['raw_all_decreased_rate'])} |

The three raw MGDA dot products are summarized separately in `summary.json` and
`per_update.csv`; positive `g_i dot g_MGDA` predicts a decrease for a small
plain step, while negative `g_i dot delta_theta` means the actual Adam
parameter displacement predicts a decrease.

## Artifacts

- `per_update.csv`: audited real training batches, raw/weighted before and
  after costs, alpha, MGDA dots, actual optimizer displacement dots, and IDs.
- `probe_trace.csv`: fixed train-only probe before/after trace.
- `summary.json`: rates, contract, protocol, and checkpoint metadata.
- `same_batch_before_after.png`, `fixed_probe_trace.png`,
  `predicted_vs_actual_direction.png`, `adam_update_alignment.png`.
"""
    path.write_text(report, encoding="utf-8")


def _fmt_rate(value):
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _fmt_num(value):
    return "n/a" if value is None else f"{float(value):.6g}"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-folder", type=Path, default=Path("data/dataset1"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--audit-every-updates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sources-per-context", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--p-mask", type=float, default=1.0)
    parser.add_argument(
        "--vehicle-radius-meters",
        type=float,
        default=SAFETY_COST_CONFIG.vehicle_radius_meters,
    )
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser.parse_args()


def run_audit(args):
    if args.updates <= 0 or args.audit_every_updates <= 0:
        raise ValueError("updates and audit interval must be positive")
    if args.batch_size <= 0 or args.sources_per_context <= 0:
        raise ValueError("batch size and sources per context must be positive")
    if not 0.0 < args.p_mask <= 1.0:
        raise ValueError("p-mask must be in (0,1]")
    if args.stage2_lr != 1e-5:
        raise ValueError("This audit must use the current Stage-2 lr=1e-5")
    if args.grad_clip_norm != 1.0:
        raise ValueError("This audit must use the current Stage-2 clip norm=1.0")
    if args.data_folder.name != "dataset1":
        raise ValueError("This audit is restricted to data/dataset1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _resolve_device(args.device)

    model, model_args, checkpoint = load_model(
        args.checkpoint,
        torch.device("cpu"),
    )
    _require_current_mask_semantics(checkpoint)
    _require_demo_target_semantics(checkpoint)
    _require_main_method_model(model)
    if checkpoint.get("stage") != "stage1":
        raise ValueError("Actual-update audit must start from a Stage-1 checkpoint")
    if checkpoint.get("privileged_cost_semantics") not in (None, PRIVILEGED_COST_SEMANTICS):
        raise ValueError("Checkpoint privileged-cost semantics are incompatible")

    environments = discover_environments(
        args.data_folder / "train",
        expected_count=MAP_CONFIG.expected_environments,
    )
    if any("val" in name.lower() or "test" in name.lower() for name in environments):
        raise ValueError("Environment selection violates train-only guard")
    dataset, discovered = make_partial_dataset(
        args.data_folder,
        "train",
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=args.p_mask,
        mask_mode="stage2_independent",
        vehicle_radius_meters=args.vehicle_radius_meters,
        environment_names=environments,
        dynamic_mask_noise=False,
    )
    if sorted(discovered) != sorted(environments):
        raise ValueError("Train environment discovery changed unexpectedly")
    train_epoch_set = DirectCostEpochDataset(dataset)
    train_epoch_set.set_epoch(0)
    data_generator = torch.Generator().manual_seed(int(args.seed) + 60_000)
    loader = DataLoader(
        train_epoch_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=data_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # The probe is a fixed batch from the same train-only dataset and mask
    # variant, with a source tensor frozen for the whole audit.
    probe_batch = default_collate(
        [dataset.get_item(index, mask_variant=1) for index in range(args.batch_size)]
    )
    probe_generator = torch.Generator(device=device).manual_seed(
        int(args.seed) + 90_001
    )
    probe_count = args.batch_size * args.sources_per_context
    probe_source = torch.randn(
        probe_count,
        model.num_edges,
        2,
        dtype=torch.float32,
        device=device,
        generator=probe_generator,
    )

    model = model.to(device)
    model.eval()
    if hasattr(model, "use_gradient_checkpoint"):
        model.use_gradient_checkpoint = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = _make_direct_cost_optimizer(model, args)
    source_generator = torch.Generator(device=device).manual_seed(
        int(args.seed) + 80_000
    )
    probe_raw, probe_weighted = _forward_costs(
        model,
        probe_batch,
        device,
        args.sources_per_context,
        probe_source,
    )
    probe_rows = [
        _probe_row(
            0,
            {**probe_raw, "weighted": probe_weighted},
            {**probe_raw, "weighted": probe_weighted},
        )
    ]
    rows = []
    update_count = 0
    for batch in loader:
        if update_count >= int(args.updates):
            break
        update_count += 1
        audit_now = update_count % int(args.audit_every_updates) == 0
        before_parameters = _parameter_snapshot(parameters) if audit_now else None
        optimizer.zero_grad(set_to_none=True)
        # The formal training loop passes one persistent source generator.  An
        # explicit draw lets this audit reuse exactly the same source after the
        # optimizer step.
        count = int(batch["map"].shape[0]) * int(args.sources_per_context)
        source_draw = torch.randn(
            count,
            model.num_edges,
            2,
            dtype=torch.float32,
            device=device,
            generator=source_generator,
        )
        output_before = _direct_cost_forward(
            model,
            batch,
            device,
            sources_per_context=args.sources_per_context,
            source=source_draw,
        )
        raw_before, weighted_before = _mean_components(output_before)
        loss = output_before["task_cost"].mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at update {update_count}")

        if audit_now:
            details = _direct_cost_mgda_update(
                model,
                output_before,
                device,
                return_geometry=True,
            )
        else:
            details = _direct_cost_mgda_update(model, output_before, device)
        source_for_after = output_before["source"].detach().clone()
        if audit_now:
            raw_gradient_tuples = details["gradient_tuples"]

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(args.grad_clip_norm),
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient at update {update_count}")
        optimizer.step()

        if audit_now:
            raw_after, weighted_after = _forward_costs(
                model,
                batch,
                device,
                args.sources_per_context,
                source_for_after,
            )
            probe_before_raw, probe_before_weighted = _forward_costs(
                model,
                probe_batch,
                device,
                args.sources_per_context,
                probe_source,
            )
            # probe_before_* above is the value after this update; the previous
            # probe row stores the value before this update.
            previous_probe = {
                name: probe_rows[-1][f"probe_J_{name}_after"]
                for name in OBJECTIVE_NAMES
            }
            previous_probe_weighted = {
                name: probe_rows[-1][f"probe_weighted_{name}_after"]
                for name in OBJECTIVE_NAMES
            }
            probe_rows.append(
                _probe_row(
                    update_count,
                    {**previous_probe, "weighted": previous_probe_weighted},
                    {**probe_before_raw, "weighted": probe_before_weighted},
                )
            )
            actual_geometry = _actual_step_geometry(
                parameters,
                before_parameters,
                raw_gradient_tuples,
            )
            gram_metrics = _gram_metrics(details)
            before_values = {**raw_before, **weighted_before}
            after_values = {**raw_after, **weighted_after}
            row = {
                "update": update_count,
                "contexts": int(output_before["contexts"]),
                "trajectories": int(output_before["task_cost"].numel()),
                **_batch_identifiers(batch, discovered),
                "source_sha256": _source_digest(source_for_after),
                "J_F_before": raw_before["F"],
                "J_S_before": raw_before["S"],
                "J_K_before": raw_before["K"],
                "J_F_after": raw_after["F"],
                "J_S_after": raw_after["S"],
                "J_K_after": raw_after["K"],
                "delta_F": raw_after["F"] - raw_before["F"],
                "delta_S": raw_after["S"] - raw_before["S"],
                "delta_K": raw_after["K"] - raw_before["K"],
                "formal_loss_before": float(loss.detach().cpu()),
                "tensorboard_forbidden_cost_before": before_values["F"],
                "tensorboard_stability_cost_before": before_values["S"],
                "tensorboard_curvature_cost_before": before_values["K"],
                "tensorboard_forbidden_cost_after": after_values["F"],
                "tensorboard_stability_cost_after": after_values["S"],
                "tensorboard_curvature_cost_after": after_values["K"],
                "tensorboard_delta_F": after_values["F"] - before_values["F"],
                "tensorboard_delta_S": after_values["S"] - before_values["S"],
                "tensorboard_delta_K": after_values["K"] - before_values["K"],
                "alpha_F": float(details["alpha_F"]),
                "alpha_S": float(details["alpha_S"]),
                "alpha_K": float(details["alpha_K"]),
                "mgda_norm": float(details["mgda_norm"]),
                "mgda_ratio": float(details["mgda_ratio"]),
                "mgda_active_set": str(details["mgda_active_set"]),
                "mgda_dot_F": float(details["mgda_dot_F"]),
                "mgda_dot_S": float(details["mgda_dot_S"]),
                "mgda_dot_K": float(details["mgda_dot_K"]),
                "mgda_common_descent": bool(details["mgda_common_descent"]),
                "actual_raw_all_decreased": all(
                    raw_after[name] < raw_before[name]
                    for name in OBJECTIVE_NAMES
                ),
                "actual_first_order_all_decreased": all(
                    actual_geometry[f"actual_update_dot_{name}"] < 0.0
                    for name in OBJECTIVE_NAMES
                ),
                "gradient_norm_pre_clip": float(gradient_norm.detach().cpu()),
                **gram_metrics,
                **actual_geometry,
            }
            rows.append(row)
            del raw_gradient_tuples, before_parameters
            del details
        del output_before, source_for_after
        if device.type == "cuda" and audit_now:
            torch.cuda.empty_cache()

    if update_count != int(args.updates):
        raise RuntimeError(
            f"Loader ended after {update_count} updates, expected {args.updates}"
        )

    checkpoint_metadata = {
        "path": str(args.checkpoint),
        "stage": checkpoint.get("stage"),
        "representation_semantic_version": checkpoint.get(
            "representation_semantic_version"
        ),
        "input_mask_semantics": checkpoint.get("input_mask_semantics"),
        "demo_target_semantics": checkpoint.get("demo_target_semantics"),
        "model_args": dict(model_args),
    }
    config = {
        "checkpoint": str(args.checkpoint),
        "data_folder": str(args.data_folder),
        "split": "train",
        "train_environments": list(environments),
        "train_environment_count": len(environments),
        "contexts_per_environment": 200,
        "batch_size": int(args.batch_size),
        "sources_per_context": int(args.sources_per_context),
        "updates": int(update_count),
        "audit_every_updates": int(args.audit_every_updates),
        "seed": int(args.seed),
        "mask_seed": int(args.mask_seed),
        "mask_variant": 1,
        "p_mask": float(args.p_mask),
        "vehicle_radius_meters": float(args.vehicle_radius_meters),
        "source_generator_seed": int(args.seed) + 80_000,
        "probe_source_seed": int(args.seed) + 90_001,
        "stage2_lr": float(args.stage2_lr),
        "grad_clip_norm": float(args.grad_clip_norm),
        "optimizer": _direct_cost_optimizer_name(args),
        "use_mgda": True,
        "mask_generation_semantics": MASK_GENERATION_SEMANTICS,
        "privileged_cost_semantics": PRIVILEGED_COST_SEMANTICS,
        "stage2_objective_semantics": DIRECT_COST_STAGE2_SEMANTICS,
        "stage2_context_semantics": DIRECT_COST_CONTEXT_SEMANTICS,
        "stage2_training_protocol_semantics": DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS,
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_tensor_count": len(parameters),
    }
    summary = _build_summary(rows, probe_rows, config, checkpoint_metadata)
    _write_csv(output_dir / "per_update.csv", rows)
    _write_csv(output_dir / "probe_trace.csv", probe_rows)
    _write_json(output_dir / "summary.json", summary)
    plots = _make_plots(output_dir, rows, probe_rows)
    summary["artifacts"] = {
        "report": str(output_dir / "REPORT.md"),
        "summary": str(output_dir / "summary.json"),
        "per_update": str(output_dir / "per_update.csv"),
        "probe_trace": str(output_dir / "probe_trace.csv"),
        "plots": [str(output_dir / name) for name in plots],
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "REPORT.md", summary)
    return summary


def main():
    args = _parse_args()
    summary = run_audit(args)
    print(json.dumps(_jsonable(summary["audit"]), indent=2, ensure_ascii=False))
    print(f"REPORT.md: {Path(args.output_dir) / 'REPORT.md'}")


if __name__ == "__main__":
    main()
