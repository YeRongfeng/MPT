#!/usr/bin/env python3
"""Paired development experiment: fixed direct cost vs exact MGDA plain SGD.

This script is intentionally separate from the formal Stage-2 training entry.
It freezes one train/evaluation manifest, reloads the same Stage-1 checkpoint
for both arms, and applies manual plain-SGD updates.  The MGDA arm uses the
three raw privileged-cost gradients and the exact three-objective active-set
solver from ``audit_direct_cost_mgda_geometry.py``.  No Adam state, momentum,
weight decay, or gradient clipping is used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

from audit_direct_cost_mgda_geometry import solve_mgda_active_set
from dataLoader_dit import MASK_GENERATION_SEMANTICS
from grad_optimizer import PRIVILEGED_COST_SEMANTICS
from map_config import SAFETY_COST_CONFIG
from posterior_pipeline import (
    DIRECT_COST_CONTEXT_SEMANTICS,
    DIRECT_COST_STAGE2_SEMANTICS,
    DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS,
    _direct_cost_forward,
    _require_current_mask_semantics,
    _require_demo_target_semantics,
    _require_main_method_model,
    evaluate_direct_cost_stage2,
    load_model,
    make_partial_dataset,
)


OBJECTIVE_NAMES = ("F", "S", "K")
COMPONENT_KEYS = {
    "F": "forbidden_region",
    "S": "stability",
    "K": "curvature",
}
FIXED_WEIGHTS = {
    "F": float(SAFETY_COST_CONFIG.forbidden_weight),
    "S": float(SAFETY_COST_CONFIG.obstacle_weight),
    "K": float(SAFETY_COST_CONFIG.curvature_weight),
}
PAIR_COLUMNS = {("F", "S"): "FS", ("F", "K"): "FK", ("S", "K"): "SK"}
STEP_INTERVAL = 100
STATIONARY_RATIO_THRESHOLD = 1e-3
STATIONARY_ABSOLUTE_THRESHOLD = 1e-12
LEGACY_ENVIRONMENT_MANIFEST_COST_SEMANTICS = (
    "gauge44_stable_yaw_log1p_curvature_dense200_physical_bounds_v12"
)
LEGACY_ENVIRONMENT_MANIFEST_MASK_SEMANTICS = (
    "informed_ellipse_two_stage_obstacle_curriculum_v4"
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


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but CUDA is unavailable")
    return device


def _model_parameters(model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return parameters


def _cost_losses(output):
    return {
        name: output["components"][COMPONENT_KEYS[name]].mean()
        for name in OBJECTIVE_NAMES
    }


def _gradient_norm(gradients, device) -> float:
    total = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().float().square().sum().double()
    return float(total.sqrt().detach().cpu())


def _gram_from_gradient_tuples(gradient_tuples, device) -> torch.Tensor:
    gram = torch.zeros((3, 3), dtype=torch.float64, device="cpu")
    for i in range(3):
        for j in range(i, 3):
            value = torch.zeros((), dtype=torch.float64, device=device)
            for first, second in zip(gradient_tuples[i], gradient_tuples[j]):
                if first is not None and second is not None:
                    value = value + (first.detach().float() * second.detach().float()).sum().double()
            scalar = float(value.detach().cpu())
            gram[i, j] = scalar
            gram[j, i] = scalar
    return gram


def _raw_gradients(output, parameters, device):
    losses = _cost_losses(output)
    ordered_losses = [losses[name] for name in OBJECTIVE_NAMES]
    gradient_tuples = []
    for index, loss in enumerate(ordered_losses):
        gradient_tuples.append(
            torch.autograd.grad(
                loss,
                parameters,
                retain_graph=index < len(ordered_losses) - 1,
                allow_unused=True,
            )
        )
    gram = _gram_from_gradient_tuples(gradient_tuples, device)
    return losses, gradient_tuples, gram


def _apply_gradient_tuple(parameters, gradients, learning_rate: float) -> None:
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                parameter.add_(gradient, alpha=-float(learning_rate))


def _apply_mgda_gradients(parameters, gradient_tuples, alpha, learning_rate: float) -> None:
    with torch.no_grad():
        for parameter, gradients in zip(parameters, zip(*gradient_tuples)):
            for coefficient, gradient in zip(alpha, gradients):
                if gradient is not None and float(coefficient) != 0.0:
                    parameter.add_(gradient, alpha=-float(learning_rate) * float(coefficient))


def _source_for_seed(model, count: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(
        count,
        model.num_edges,
        2,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def _batch_from_indices(dataset, indices, mask_variant: int):
    samples = [dataset.get_item(int(index), mask_variant=int(mask_variant)) for index in indices]
    return default_collate(samples)


def _finite_values(rows, key):
    return [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]


def _mean(rows, key):
    values = _finite_values(rows, key)
    return float(np.mean(values)) if values else None


def _median(rows, key):
    values = _finite_values(rows, key)
    return float(np.median(values)) if values else None


def _rate(rows, key, predicate):
    values = [row.get(key) for row in rows]
    return float(sum(bool(predicate(value)) for value in values) / len(values)) if values else None


def _cosine(gram, first: int, second: int, norms):
    denominator = norms[first] * norms[second]
    return float(gram[first, second]) / denominator if denominator > 0.0 else float("nan")


def _geometry_row_from_values(
    *,
    arm: str,
    update: int,
    probe_batch: int,
    values,
    gram,
    solution,
):
    norms = [math.sqrt(max(float(gram[index, index]), 0.0)) for index in range(3)]
    alpha = solution["alpha"]
    mgda_dots = gram @ alpha
    fixed_weight_tensor = torch.tensor(
        [FIXED_WEIGHTS[name] for name in OBJECTIVE_NAMES], dtype=torch.float64
    )
    fixed_dots = gram @ fixed_weight_tensor
    mgda_norm = math.sqrt(max(float(solution["objective"]), 0.0))
    fixed_norm = math.sqrt(max(float(fixed_weight_tensor @ gram @ fixed_weight_tensor), 0.0))
    denominator = float(np.mean(norms))
    ratio = mgda_norm / denominator if denominator > 0.0 else float("nan")
    cosines = {
        suffix: _cosine(gram, OBJECTIVE_NAMES.index(pair[0]), OBJECTIVE_NAMES.index(pair[1]), norms)
        for pair, suffix in PAIR_COLUMNS.items()
    }
    mgda_dot_values = [float(value) for value in mgda_dots]
    fixed_dot_values = [float(value) for value in fixed_dots]
    return {
        "arm": arm,
        "update": int(update),
        "probe_batch": int(probe_batch),
        "J_F": float(values["F"].detach().cpu()),
        "J_S": float(values["S"].detach().cpu()),
        "J_K": float(values["K"].detach().cpu()),
        "norm_g_F": norms[0],
        "norm_g_S": norms[1],
        "norm_g_K": norms[2],
        "cos_FS": cosines["FS"],
        "cos_FK": cosines["FK"],
        "cos_SK": cosines["SK"],
        "alpha_F": float(alpha[0]),
        "alpha_S": float(alpha[1]),
        "alpha_K": float(alpha[2]),
        "mgda_norm": mgda_norm,
        "mgda_ratio": ratio,
        "mgda_active_set": solution["label"],
        "dot_F_MGDA": mgda_dot_values[0],
        "dot_S_MGDA": mgda_dot_values[1],
        "dot_K_MGDA": mgda_dot_values[2],
        "common_descent": all(value > 0.0 for value in mgda_dot_values),
        "approximate_pareto_stationary": (
            mgda_norm <= STATIONARY_ABSOLUTE_THRESHOLD
            or (math.isfinite(ratio) and ratio <= STATIONARY_RATIO_THRESHOLD)
        ),
        "fixed_norm": fixed_norm,
        "dot_F_fixed": fixed_dot_values[0],
        "dot_S_fixed": fixed_dot_values[1],
        "dot_K_fixed": fixed_dot_values[2],
        "fixed_common_descent": all(value > 0.0 for value in fixed_dot_values),
        "fixed_worsens_any": any(value < 0.0 for value in fixed_dot_values),
        "fixed_worsens_F": fixed_dot_values[0] < 0.0,
        "fixed_worsens_S": fixed_dot_values[1] < 0.0,
        "fixed_worsens_K": fixed_dot_values[2] < 0.0,
    }


def _aggregate_geometry(rows, *, arm: str, update: int) -> dict[str, Any]:
    result = {"arm": arm, "update": int(update), "probe_batches": len(rows)}
    for key in ("J_F", "J_S", "J_K", "norm_g_F", "norm_g_S", "norm_g_K", "mgda_norm", "mgda_ratio", "alpha_F", "alpha_S", "alpha_K", "cos_FS", "cos_FK", "cos_SK"):
        result[f"{key}_mean"] = _mean(rows, key)
        result[f"{key}_median"] = _median(rows, key)
    result["common_descent_rate"] = _rate(rows, "common_descent", bool)
    result["approximate_pareto_stationary_rate"] = _rate(rows, "approximate_pareto_stationary", bool)
    result["fixed_common_descent_rate"] = _rate(rows, "fixed_common_descent", bool)
    result["fixed_worsens_any_rate"] = _rate(rows, "fixed_worsens_any", bool)
    for name in OBJECTIVE_NAMES:
        result[f"fixed_worsens_{name}_rate"] = _rate(rows, f"fixed_worsens_{name}", bool)
    result["cos_FS_negative_rate"] = _rate(rows, "cos_FS", lambda value: value is not None and math.isfinite(float(value)) and float(value) < 0.0)
    result["cos_FK_negative_rate"] = _rate(rows, "cos_FK", lambda value: value is not None and math.isfinite(float(value)) and float(value) < 0.0)
    result["cos_SK_negative_rate"] = _rate(rows, "cos_SK", lambda value: value is not None and math.isfinite(float(value)) and float(value) < 0.0)
    result["dominant_max_alpha_rate"] = float(
        sum(max(float(row["alpha_F"]), float(row["alpha_S"]), float(row["alpha_K"])) >= 0.9 for row in rows) / len(rows)
    ) if rows else None
    result["dominant_F_rate"] = _rate(rows, "alpha_F", lambda value: value is not None and float(value) >= 0.9)
    result["dominant_S_rate"] = _rate(rows, "alpha_S", lambda value: value is not None and float(value) >= 0.9)
    result["dominant_K_rate"] = _rate(rows, "alpha_K", lambda value: value is not None and float(value) >= 0.9)
    result["active_set_counts"] = dict(sorted(Counter(str(row["mgda_active_set"]) for row in rows).items()))
    return result


def _run_geometry_probe(
    model,
    parameters,
    dataset,
    *,
    arm: str,
    update: int,
    probe_indices: Sequence[Sequence[int]],
    mask_variant: int,
    source_seeds: Sequence[int],
    sources_per_context: int,
    device: torch.device,
):
    rows = []
    for probe_batch, (indices, source_seed) in enumerate(zip(probe_indices, source_seeds)):
        batch = _batch_from_indices(dataset, indices, mask_variant)
        count = len(indices) * int(sources_per_context)
        source = _source_for_seed(model, count, source_seed, device)
        output = _direct_cost_forward(
            model,
            batch,
            device,
            sources_per_context=sources_per_context,
            source=source,
        )
        values, gradient_tuples, gram = _raw_gradients(output, parameters, device)
        solution = solve_mgda_active_set(gram)
        rows.append(
            _geometry_row_from_values(
                arm=arm,
                update=update,
                probe_batch=probe_batch,
                values=values,
                gram=gram,
                solution=solution,
            )
        )
        del output, gradient_tuples, gram, solution, source, batch
    aggregate = _aggregate_geometry(rows, arm=arm, update=update)
    return aggregate, rows


def _build_train_manifest(
    dataset_size: int,
    *,
    epochs: int,
    batch_size: int,
    order_seed: int,
    source_seed: int,
):
    if dataset_size % batch_size != 0:
        raise ValueError("Train dataset size must be divisible by batch size")
    generator = torch.Generator(device="cpu").manual_seed(int(order_seed))
    updates_per_epoch = dataset_size // batch_size
    epochs_manifest = []
    global_update = 0
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size, generator=generator).tolist()
        updates = []
        for batch_index in range(updates_per_epoch):
            indices = permutation[batch_index * batch_size : (batch_index + 1) * batch_size]
            global_update += 1
            updates.append(
                {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_update": global_update,
                    "indices": [int(index) for index in indices],
                    "mask_variant": epoch + 1,
                    "source_seed": _stable_seed(source_seed, "train-source", global_update),
                }
            )
        epochs_manifest.append(updates)
    return epochs_manifest


def _build_geometry_manifest(dataset_size: int, *, batch_size: int, batches: int, seed: int):
    required = int(batch_size) * int(batches)
    if required > dataset_size:
        raise ValueError("Geometry probe exceeds train dataset")
    indices = []
    seeds = []
    for batch_index in range(batches):
        indices.append(list(range(batch_index * batch_size, (batch_index + 1) * batch_size)))
        seeds.append(_stable_seed(seed, "geometry-source", batch_index))
    return indices, seeds


def _validation_key(metrics: Mapping[str, Any]):
    return (
        -float(metrics["task_cost"]),
        float(metrics["safe_at_1"]),
        float(metrics["strict_valid_rate"]),
    )


def _save_model_checkpoint(path: Path, model, model_args, metadata, *, min_free_bytes=1_000_000_000):
    usage = shutil.disk_usage(path.parent)
    if usage.free < min_free_bytes:
        raise OSError(f"Insufficient free disk space before checkpoint save: {usage.free} bytes")
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_args": dict(model_args),
        **metadata,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _run_training_arm(
    *,
    arm: str,
    args,
    model_args,
    train_dataset,
    val_loader,
    baseline_snapshot,
    baseline_metrics,
    train_manifest,
    geometry_indices,
    geometry_seeds,
    protocol_hash: str,
    device: torch.device,
    output_dir: Path,
):
    arm_dir = output_dir / f"arm_{arm}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    model, loaded_args, checkpoint = load_model(args.stage1_checkpoint, torch.device("cpu"))
    if dict(loaded_args) != dict(model_args):
        raise ValueError("Stage-1 model arguments changed between protocol and arm reload")
    del checkpoint
    gc.collect()
    _require_main_method_model(model)
    model = model.to(device)
    model.eval()
    if hasattr(model, "use_gradient_checkpoint"):
        model.use_gradient_checkpoint = False
    parameters = _model_parameters(model)

    arm_trace = []
    geometry_trace = []
    geometry_probe_trace = []
    evaluation_trace = []
    start_time = time.time()
    initial_geometry, initial_probe_rows = _run_geometry_probe(
        model,
        parameters,
        train_dataset,
        arm=arm,
        update=0,
        probe_indices=geometry_indices,
        mask_variant=1,
        source_seeds=geometry_seeds,
        sources_per_context=args.sources_per_context,
        device=device,
    )
    geometry_trace.append(initial_geometry)
    geometry_probe_trace.extend(initial_probe_rows)
    evaluation_trace.append({"arm": arm, "update": 0, **dict(baseline_metrics)})
    best_key = _validation_key(baseline_metrics)
    best_update = 0
    total_updates = sum(len(epoch_updates) for epoch_updates in train_manifest)
    max_updates = args.max_updates_per_arm
    if max_updates is not None:
        total_updates = min(total_updates, int(max_updates))
    completed_updates = 0

    for epoch_updates in train_manifest:
        if completed_updates >= total_updates:
            break
        for item in epoch_updates:
            if completed_updates >= total_updates:
                break
            update = int(item["global_update"])
            indices = item["indices"]
            batch = _batch_from_indices(train_dataset, indices, int(item["mask_variant"]))
            source = _source_for_seed(
                model,
                len(indices) * int(args.sources_per_context),
                int(item["source_seed"]),
                device,
            )
            output = _direct_cost_forward(
                model,
                batch,
                device,
                sources_per_context=args.sources_per_context,
                source=source,
            )
            values = _cost_losses(output)
            if not all(bool(torch.isfinite(value)) for value in values.values()):
                raise FloatingPointError(f"Non-finite raw cost in {arm} arm at update {update}")
            if arm == "mgda":
                losses, gradient_tuples, gram = _raw_gradients(output, parameters, device)
                solution = solve_mgda_active_set(gram)
                alpha = solution["alpha"].tolist()
                _apply_mgda_gradients(parameters, gradient_tuples, alpha, args.learning_rate)
                gradient_norm = math.sqrt(max(float(solution["objective"]), 0.0))
                batch_mgda_ratio = gradient_norm / float(np.mean([math.sqrt(max(float(gram[index, index]), 0.0)) for index in range(3)])) if float(gram.diag().sum()) > 0.0 else float("nan")
                update_details = {
                    "mgda_active_set": solution["label"],
                    "alpha_F": float(alpha[0]),
                    "alpha_S": float(alpha[1]),
                    "alpha_K": float(alpha[2]),
                    "batch_mgda_ratio": batch_mgda_ratio,
                    "batch_common_descent": all(float(value) > 0.0 for value in (gram @ solution["alpha"])),
                    "batch_mgda_norm": gradient_norm,
                }
                del gradient_tuples, gram, solution
            else:
                fixed_loss = sum(FIXED_WEIGHTS[name] * values[name] for name in OBJECTIVE_NAMES)
                gradients = torch.autograd.grad(fixed_loss, parameters, allow_unused=True)
                gradient_norm = _gradient_norm(gradients, device)
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError(f"Non-finite fixed gradient at update {update}")
                _apply_gradient_tuple(parameters, gradients, args.learning_rate)
                update_details = {
                    "mgda_active_set": "not_computed_in_training",
                    "alpha_F": None,
                    "alpha_S": None,
                    "alpha_K": None,
                    "batch_mgda_ratio": None,
                    "batch_common_descent": None,
                    "batch_mgda_norm": None,
                }
                del gradients, fixed_loss
            cost_values = {name: float(values[name].detach().cpu()) for name in OBJECTIVE_NAMES}
            trace_row = {
                "arm": arm,
                "epoch": int(item["epoch"] + 1),
                "batch_index": int(item["batch_index"]),
                "global_update": update,
                "mask_variant": int(item["mask_variant"]),
                "indices": ",".join(str(index) for index in indices),
                "source_seed": int(item["source_seed"]),
                "J_F": cost_values["F"],
                "J_S": cost_values["S"],
                "J_K": cost_values["K"],
                "fixed_weight_loss": sum(FIXED_WEIGHTS[name] * cost_values[name] for name in OBJECTIVE_NAMES),
                "gradient_norm": gradient_norm,
                **update_details,
            }
            arm_trace.append(trace_row)
            del output, values, source, batch
            completed_updates += 1

            if update % STEP_INTERVAL == 0 or completed_updates == total_updates:
                geometry_summary, probe_rows = _run_geometry_probe(
                    model,
                    parameters,
                    train_dataset,
                    arm=arm,
                    update=update,
                    probe_indices=geometry_indices,
                    mask_variant=1,
                    source_seeds=geometry_seeds,
                    sources_per_context=args.sources_per_context,
                    device=device,
                )
                geometry_trace.append(geometry_summary)
                geometry_probe_trace.extend(probe_rows)
                validation = evaluate_direct_cost_stage2(
                    model,
                    val_loader,
                    device,
                    sources_per_context=16,
                    seed=args.evaluation_source_seed,
                    baseline_snapshot=baseline_snapshot,
                )
                evaluation_trace.append({"arm": arm, "update": update, **validation})
                current_key = _validation_key(validation)
                if current_key > best_key:
                    best_key = current_key
                    best_update = update
                    _save_model_checkpoint(
                        arm_dir / "best.pth",
                        model,
                        model_args,
                        {
                            "stage": "development_direct_cost_sgd_pair",
                            "arm": arm,
                            "global_update": update,
                            "epoch": int(item["epoch"] + 1),
                            "protocol_sha256": protocol_hash,
                            "checkpoint_semantics": "weights_only_no_optimizer_plain_sgd",
                        },
                    )
                elapsed = time.time() - start_time
                print(
                    f"[{arm}] update={update}/{total_updates} "
                    f"J=({cost_values['F']:.5g},{cost_values['S']:.5g},{cost_values['K']:.5g}) "
                    f"geom_common={geometry_summary['common_descent_rate']:.1%} "
                    f"ratio={geometry_summary['mgda_ratio_median']:.4g} "
                    f"strict={validation['strict_valid_rate']:.2%} "
                    f"Safe@1={validation['safe_at_1']:.2%} "
                    f"elapsed={elapsed / 60.0:.1f}m",
                    flush=True,
                )

        if completed_updates > 0 and (completed_updates % (len(train_manifest[0]) if train_manifest else 1) == 0 or completed_updates == total_updates):
            _save_model_checkpoint(
                arm_dir / "last.pth",
                model,
                model_args,
                {
                    "stage": "development_direct_cost_sgd_pair",
                    "arm": arm,
                    "global_update": completed_updates,
                    "epoch": int(completed_updates // max(len(train_manifest[0]), 1)),
                    "protocol_sha256": protocol_hash,
                    "checkpoint_semantics": "weights_only_no_optimizer_plain_sgd",
                },
            )

    if not (arm_dir / "last.pth").exists():
        _save_model_checkpoint(
            arm_dir / "last.pth",
            model,
            model_args,
            {
                "stage": "development_direct_cost_sgd_pair",
                "arm": arm,
                "global_update": completed_updates,
                "epoch": int(completed_updates // max(len(train_manifest[0]), 1)),
                "protocol_sha256": protocol_hash,
                "checkpoint_semantics": "weights_only_no_optimizer_plain_sgd",
            },
        )
    if any(parameter.grad is not None for parameter in parameters):
        raise AssertionError(f"{arm} arm left parameter .grad populated")
    final_validation = evaluation_trace[-1]
    result = {
        "arm": arm,
        "requested_epochs": int(args.epochs),
        "executed_updates": int(completed_updates),
        "updates_per_epoch": int(len(train_manifest[0]) if train_manifest else 0),
        "final_epoch": int(math.ceil(completed_updates / max(len(train_manifest[0]), 1))),
        "best_update": int(best_update),
        "best_validation_key": list(best_key),
        "final_validation": final_validation,
        "initial_geometry": geometry_trace[0],
        "final_geometry": geometry_trace[-1],
        "min_geometry_mgda_ratio": min(float(row["mgda_ratio_median"]) for row in geometry_trace if row["mgda_ratio_median"] is not None),
        "max_geometry_pareto_near_zero_rate": max(float(row["approximate_pareto_stationary_rate"]) for row in geometry_trace if row["approximate_pareto_stationary_rate"] is not None),
        "trace_rows": len(arm_trace),
        "geometry_rows": len(geometry_trace),
        "elapsed_seconds": time.time() - start_time,
    }
    del model, parameters
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, arm_trace, geometry_trace, geometry_probe_trace, evaluation_trace


def _load_and_validate_protocol(args, device):
    stage1_model, model_args, stage1_checkpoint = load_model(args.stage1_checkpoint, torch.device("cpu"))
    _require_current_mask_semantics(stage1_checkpoint)
    _require_demo_target_semantics(stage1_checkpoint)
    _require_main_method_model(stage1_model)
    if stage1_checkpoint.get("stage") != "stage1":
        raise ValueError("--stage1-checkpoint must be an original Stage-1 checkpoint")
    stage1_p_mask = stage1_checkpoint.get("p_mask")
    environment_checkpoint = torch.load(
        args.environment_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    _require_current_mask_semantics(environment_checkpoint)
    _require_demo_target_semantics(environment_checkpoint)
    if environment_checkpoint.get("stage") != "stage2_direct_privileged_cost":
        raise ValueError("--environment-checkpoint must be a current direct-cost Stage-2 checkpoint")
    if environment_checkpoint.get("stage2_objective_semantics") != DIRECT_COST_STAGE2_SEMANTICS:
        raise ValueError("Stage-2 objective semantics mismatch")
    if environment_checkpoint.get("stage2_context_semantics") != DIRECT_COST_CONTEXT_SEMANTICS:
        raise ValueError("Stage-2 context semantics mismatch")
    environment_cost_semantics = environment_checkpoint.get(
        "privileged_cost_semantics"
    )
    environment_mask_semantics = environment_checkpoint.get(
        "mask_generation_semantics"
    )
    legacy_environment_manifest = (
        environment_cost_semantics != PRIVILEGED_COST_SEMANTICS
        or environment_mask_semantics != MASK_GENERATION_SEMANTICS
    )
    if legacy_environment_manifest:
        if not args.allow_legacy_environment_manifest:
            raise ValueError(
                "Environment checkpoint is a legacy manifest. Pass "
                "--allow-legacy-environment-manifest only when reusing its "
                "environment split and validation indices with the current "
                "cost/mask implementation."
            )
        if environment_cost_semantics != LEGACY_ENVIRONMENT_MANIFEST_COST_SEMANTICS:
            raise ValueError(
                "Unsupported legacy environment cost semantics: "
                f"{environment_cost_semantics!r}"
            )
        if environment_mask_semantics != LEGACY_ENVIRONMENT_MANIFEST_MASK_SEMANTICS:
            raise ValueError(
                "Unsupported legacy environment mask semantics: "
                f"{environment_mask_semantics!r}"
            )
    if int(environment_checkpoint.get("mask_seed")) != int(args.mask_seed):
        raise ValueError("Stage-2 environment checkpoint mask seed differs from protocol")
    if not np.isclose(float(environment_checkpoint.get("p_mask")), float(args.p_mask)):
        raise ValueError("Stage-2 environment checkpoint p_mask differs from protocol")
    if not np.isclose(float(environment_checkpoint.get("vehicle_radius_meters")), float(args.vehicle_radius_meters)):
        raise ValueError("Stage-2 environment checkpoint vehicle radius differs from protocol")
    environment_split = environment_checkpoint.get("stage2_environment_split") or {}
    train_environments = sorted(str(value) for value in environment_split.get("train", []))
    validation_environments = sorted(str(value) for value in environment_split.get("validation", []))
    if not train_environments or not validation_environments:
        raise ValueError("Stage-2 environment checkpoint has no frozen train/validation split")
    validation_indices = [int(value) for value in environment_checkpoint.get("fixed_validation_indices", [])]
    if len(validation_indices) != len(validation_environments):
        raise ValueError("Stage-2 checkpoint fixed validation manifest is incomplete")
    stage1_hash = _sha256_file(args.stage1_checkpoint)
    environment_hash = _sha256_file(args.environment_checkpoint)
    del stage1_checkpoint, environment_checkpoint
    gc.collect()

    train_dataset, discovered_train = make_partial_dataset(
        args.data_folder,
        "train",
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=args.p_mask,
        mask_mode="stage2_independent",
        vehicle_radius_meters=args.vehicle_radius_meters,
        environment_names=train_environments,
        dynamic_mask_noise=False,
    )
    val_dataset, discovered_val = make_partial_dataset(
        args.data_folder,
        "val",
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=args.p_mask,
        mask_mode="stage2_independent",
        vehicle_radius_meters=args.vehicle_radius_meters,
        environment_names=validation_environments,
        dynamic_mask_noise=False,
    )
    if sorted(discovered_train) != train_environments or sorted(discovered_val) != validation_environments:
        raise ValueError("Discovered environments differ from the frozen Stage-2 manifest")
    if any(index < 0 or index >= len(val_dataset) for index in validation_indices):
        raise ValueError("Frozen validation index is outside the validation dataset")
    val_loader = DataLoader(
        Subset(val_dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    train_manifest = _build_train_manifest(
        len(train_dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        order_seed=args.order_seed,
        source_seed=args.source_seed,
    )
    geometry_indices, geometry_seeds = _build_geometry_manifest(
        len(train_dataset),
        batch_size=args.geometry_batch_size,
        batches=args.geometry_probe_batches,
        seed=args.geometry_source_seed,
    )
    protocol = {
        "development_only": True,
        "network_training_isolated_from_formal_entry": True,
        "optimizer": "manual_plain_sgd",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "gradient_clipping": False,
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "stage1_checkpoint_sha256": stage1_hash,
        "environment_checkpoint": str(args.environment_checkpoint),
        "environment_checkpoint_sha256": environment_hash,
        "data_folder": str(args.data_folder),
        "train_split": "train",
        "validation_split": "val",
        "external_30_read": False,
        "final_test_read": False,
        "model_args": dict(model_args),
        "stage2_objective_semantics": DIRECT_COST_STAGE2_SEMANTICS,
        "stage2_context_semantics": DIRECT_COST_CONTEXT_SEMANTICS,
        "stage2_training_protocol_semantics": DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS,
        "privileged_cost_semantics": PRIVILEGED_COST_SEMANTICS,
        "environment_manifest_legacy": bool(legacy_environment_manifest),
        "environment_manifest_cost_semantics": environment_cost_semantics,
        "environment_manifest_mask_generation_semantics": environment_mask_semantics,
        "environment_manifest_reused_only_for": [
            "stage2_environment_split",
            "fixed_validation_indices",
        ],
        "train_environments": train_environments,
        "validation_environments": validation_environments,
        "batch_size": int(args.batch_size),
        "sources_per_context": int(args.sources_per_context),
        "learning_rate": float(args.learning_rate),
        "epochs": int(args.epochs),
        "updates_per_epoch": len(train_manifest[0]),
        "total_updates_per_arm": sum(len(epoch_updates) for epoch_updates in train_manifest),
        "mask_seed": int(args.mask_seed),
        "p_mask": float(args.p_mask),
        "mask_variant_by_epoch": "epoch_index_plus_one",
        "vehicle_radius_meters": float(args.vehicle_radius_meters),
        "order_seed": int(args.order_seed),
        "source_seed": int(args.source_seed),
        "evaluation_source_seed": int(args.evaluation_source_seed),
        "geometry_source_seed": int(args.geometry_source_seed),
        "geometry_interval_updates": STEP_INTERVAL,
        "geometry_probe_batches": int(args.geometry_probe_batches),
        "geometry_probe_batch_size": int(args.geometry_batch_size),
        "validation_indices": validation_indices,
        "validation_sources_per_context": 16,
        "fixed_weights": FIXED_WEIGHTS,
        "train_manifest": train_manifest,
        "geometry_manifest": {
            "indices": geometry_indices,
            "source_seeds": geometry_seeds,
            "mask_variant": 1,
        },
        "stage1_model_checkpoint_p_mask": (
            None if stage1_p_mask is None else float(stage1_p_mask)
        ),
    }
    return stage1_model, model_args, train_dataset, val_loader, protocol


def _protocol_hash(protocol) -> str:
    encoded = json.dumps(_jsonable(protocol), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_plots(output_dir: Path, geometry_rows, evaluation_rows, training_rows):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        (output_dir / "plot_error.txt").write_text(str(error) + "\n", encoding="utf-8")
        return []
    paths = []
    colors = {"fixed": "tab:gray", "mgda": "tab:purple"}
    for filename, key, ylabel, title in (
        ("geometry_common_descent.png", "common_descent_rate", "common descent rate", "Geometry common-descent trajectory"),
        ("geometry_mgda_ratio.png", "mgda_ratio_median", "median MGDA ratio", "MGDA norm ratio trajectory"),
        ("geometry_fs_cosine.png", "cos_FS_mean", "mean cos(F,S)", "F-S gradient conflict trajectory"),
        ("geometry_pareto_stationary.png", "approximate_pareto_stationary_rate", "near-zero rate", "Approximate Pareto-stationary rate"),
    ):
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for arm in ("fixed", "mgda"):
            values = [row for row in geometry_rows if row["arm"] == arm]
            axis.plot([row["update"] for row in values], [row[key] for row in values], label=arm, color=colors[arm], linewidth=1.5, marker=".")
        axis.set_xlabel("update")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path.name)

    for filename, key, ylabel, title in (
        ("validation_task_cost.png", "task_cost", "task cost", "Development validation task cost"),
        ("validation_strict_safe.png", "strict_valid_rate", "rate", "Development validation validity"),
    ):
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for arm in ("fixed", "mgda"):
            values = [row for row in evaluation_rows if row["arm"] == arm]
            axis.plot([row["update"] for row in values], [row[key] for row in values], label=arm, color=colors[arm], linewidth=1.5, marker=".")
        if key == "strict_valid_rate":
            for arm in ("fixed", "mgda"):
                values = [row for row in evaluation_rows if row["arm"] == arm]
                axis.plot([row["update"] for row in values], [row["safe_at_1"] for row in values], linestyle="--", color=colors[arm], alpha=0.65, label=f"{arm} Safe@1")
        axis.set_xlabel("update")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path.name)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for arm in ("fixed", "mgda"):
        values = [row for row in training_rows if row["arm"] == arm]
        if not values:
            continue
        updates = np.asarray([row["global_update"] for row in values])
        for key, label, color in (("J_F", "J_F", "tab:red"), ("J_S", "J_S", "tab:blue"), ("J_K", "J_K", "tab:green")):
            window = max(1, min(100, len(values)))
            series = np.asarray([row[key] for row in values], dtype=float)
            kernel = np.ones(window) / window
            smoothed = np.convolve(series, kernel, mode="valid")
            axis.plot(updates[window - 1 :], smoothed, label=f"{arm} {label}", color=color, alpha=0.8 if arm == "mgda" else 0.45)
    axis.set_xlabel("update")
    axis.set_ylabel("100-update moving mean")
    axis.set_title("Training raw privileged costs")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    path = output_dir / "training_raw_costs.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)
    return paths


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}g}"


def _pct(value):
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _write_report(path: Path, summary):
    fixed = summary["arms"]["fixed"]
    mgda = summary["arms"]["mgda"]
    fixed_start = fixed["initial_geometry"]
    fixed_end = fixed["final_geometry"]
    mgda_start = mgda["initial_geometry"]
    mgda_end = mgda["final_geometry"]
    lines = [
        "# Fixed-Weight vs MGDA Plain-SGD Direct-Cost Experiment",
        "",
        "## Scope",
        "",
        "This is an isolated development experiment. It does not modify or call the formal training entry.",
        "",
        f"- Stage-1 checkpoint: `{summary['protocol']['stage1_checkpoint']}`.",
        f"- Arms: fixed `(5F+1S+0.5K)` and exact three-objective MGDA.",
        f"- Updates: `{summary['protocol']['epochs']}` epochs x `{summary['protocol']['updates_per_epoch']}` updates per arm.",
        f"- Optimizer: manual plain SGD, learning rate `{summary['protocol']['learning_rate']}`, no Adam, momentum, weight decay, or clipping.",
        f"- Shared train protocol: `{len(summary['protocol']['train_environments'])}` environments, batch size `{summary['protocol']['batch_size']}`, `{summary['protocol']['sources_per_context']}` sources/context, frozen order/mask/source manifest.",
        f"- Geometry probe: `{summary['protocol']['geometry_probe_batches']}` fixed batches at update 0 and every `{summary['protocol']['geometry_interval_updates']}` updates.",
        "- Validation: fixed development `val` manifest only; external 30 and final test were not read.",
        "",
        "## Endpoint Geometry",
        "",
        "| arm | update | common descent | median mgda_ratio | near-zero rate | max alpha >= 0.9 | mean cos(F,S) | fixed common descent | fixed worsens any |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| fixed | 0 | {_pct(fixed_start['common_descent_rate'])} | {_fmt(fixed_start['mgda_ratio_median'])} | {_pct(fixed_start['approximate_pareto_stationary_rate'])} | {_pct(fixed_start['dominant_max_alpha_rate'])} | {_fmt(fixed_start['cos_FS_mean'])} | {_pct(fixed_start['fixed_common_descent_rate'])} | {_pct(fixed_start['fixed_worsens_any_rate'])} |",
        f"| fixed | {fixed_end['update']} | {_pct(fixed_end['common_descent_rate'])} | {_fmt(fixed_end['mgda_ratio_median'])} | {_pct(fixed_end['approximate_pareto_stationary_rate'])} | {_pct(fixed_end['dominant_max_alpha_rate'])} | {_fmt(fixed_end['cos_FS_mean'])} | {_pct(fixed_end['fixed_common_descent_rate'])} | {_pct(fixed_end['fixed_worsens_any_rate'])} |",
        f"| mgda | 0 | {_pct(mgda_start['common_descent_rate'])} | {_fmt(mgda_start['mgda_ratio_median'])} | {_pct(mgda_start['approximate_pareto_stationary_rate'])} | {_pct(mgda_start['dominant_max_alpha_rate'])} | {_fmt(mgda_start['cos_FS_mean'])} | {_pct(mgda_start['fixed_common_descent_rate'])} | {_pct(mgda_start['fixed_worsens_any_rate'])} |",
        f"| mgda | {mgda_end['update']} | {_pct(mgda_end['common_descent_rate'])} | {_fmt(mgda_end['mgda_ratio_median'])} | {_pct(mgda_end['approximate_pareto_stationary_rate'])} | {_pct(mgda_end['dominant_max_alpha_rate'])} | {_fmt(mgda_end['cos_FS_mean'])} | {_pct(mgda_end['fixed_common_descent_rate'])} | {_pct(mgda_end['fixed_worsens_any_rate'])} |",
        "",
        "The geometry rows are computed from raw F/S/K gradients at the frozen probe batches. The fixed arm's training update is the direct fixed scalarization; the geometry columns remain an unweighted MGDA diagnostic.",
        "",
        "## Validation Endpoint",
        "",
        "| arm | final update | task cost | strict valid | Safe@1 | Safe@K | stability regression | curvature regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("fixed", "mgda"):
        final = summary["arms"][arm]["final_validation"]
        lines.append(
            f"| {arm} | {final['update']} | {_fmt(final['task_cost'])} | {_pct(final['strict_valid_rate'])} | {_pct(final['safe_at_1'])} | {_pct(final['safe_at_k'])} | {_pct(final.get('stability_regression_rate'))} | {_pct(final.get('curvature_regression_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- MGDA ratio start -> final: fixed `{_fmt(fixed_start['mgda_ratio_median'])}` -> `{_fmt(fixed_end['mgda_ratio_median'])}`; MGDA `{_fmt(mgda_start['mgda_ratio_median'])}` -> `{_fmt(mgda_end['mgda_ratio_median'])}`.",
            f"- MGDA arm maximum observed near-zero geometry rate: `{_pct(mgda['max_geometry_pareto_near_zero_rate'])}`; minimum observed ratio median: `{_fmt(mgda['min_geometry_mgda_ratio'])}`.",
            "- A difference between arms is attributable to the update direction under this paired manifest, but it is not by itself a causal proof about all possible Stage-2 schedules.",
            "- Validation metrics are development validation evidence and are not final-test evidence.",
            "",
            "## Artifacts",
            "",
            "- `protocol.json` freezes train order, masks, sources, validation indices, and geometry probes.",
            "- `training_trace.csv` records both arms at every update.",
            "- `geometry_trace.csv` records probe aggregates every 100 updates.",
            "- `geometry_probe_per_batch.csv` records the individual probe batches.",
            "- `evaluation_trace.csv` records paired fixed validation metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("data/path_meanflow/stage1_best.pth"))
    parser.add_argument("--environment-checkpoint", type=Path, default=Path("data/pmf_s2_stability_only/stage2_last.pth"))
    parser.add_argument("--data-folder", type=Path, default=Path("data/dataset1"))
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/direct_cost_mgda_vs_fixed_sgd_20260807"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sources-per-context", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--p-mask", type=float, default=1.0)
    parser.add_argument("--vehicle-radius-meters", type=float, default=0.2)
    parser.add_argument("--order-seed", type=int, default=20260807 + 60000)
    parser.add_argument("--source-seed", type=int, default=20260807 + 80000)
    parser.add_argument("--evaluation-source-seed", type=int, default=20260807 + 90000)
    parser.add_argument("--geometry-source-seed", type=int, default=20260807)
    parser.add_argument("--geometry-probe-batches", type=int, default=16)
    parser.add_argument("--geometry-batch-size", type=int, default=16)
    parser.add_argument(
        "--allow-legacy-environment-manifest",
        action="store_true",
        help=(
            "Reuse only the environment split and validation indices from a "
            "known v12 Stage-2 checkpoint; all current training/evaluation "
            "costs and masks use the current implementation."
        ),
    )
    parser.add_argument("--max-updates-per-arm", type=int, default=None, help="Smoke-only cap; full runs should omit this.")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.sources_per_context <= 0:
        raise ValueError("epochs, batch size, and sources per context must be positive")
    if args.epochs < 6 and args.max_updates_per_arm is None:
        raise ValueError("Full paired experiment requires at least 6 epochs")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if args.max_updates_per_arm is not None and args.max_updates_per_arm <= 0:
        raise ValueError("max-updates-per-arm must be positive")
    if args.geometry_probe_batches <= 0 or args.geometry_batch_size <= 0:
        raise ValueError("geometry probe dimensions must be positive")
    if not 0.0 < args.p_mask <= 1.0:
        raise ValueError("p-mask must be in (0,1]")
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260807)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260807)

    stage1_model, model_args, train_dataset, val_loader, protocol = _load_and_validate_protocol(args, device)
    del stage1_model
    protocol["max_updates_per_arm"] = args.max_updates_per_arm
    protocol_hash = _protocol_hash(protocol)
    protocol["protocol_sha256"] = protocol_hash
    _write_json(output_dir / "protocol.json", protocol)

    baseline_model, _, baseline_checkpoint = load_model(args.stage1_checkpoint, torch.device("cpu"))
    del baseline_checkpoint
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    baseline_metrics, baseline_snapshot = evaluate_direct_cost_stage2(
        baseline_model,
        val_loader,
        device,
        sources_per_context=16,
        seed=args.evaluation_source_seed,
        return_snapshot=True,
    )
    _write_json(output_dir / "baseline_validation.json", baseline_metrics)
    del baseline_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_manifest = protocol["train_manifest"]
    geometry_manifest = protocol["geometry_manifest"]
    all_results = {}
    all_training_rows = []
    all_geometry_rows = []
    all_geometry_probe_rows = []
    all_evaluation_rows = []
    for arm in ("fixed", "mgda"):
        result, training_rows, geometry_rows, probe_rows, evaluation_rows = _run_training_arm(
            arm=arm,
            args=args,
            model_args=model_args,
            train_dataset=train_dataset,
            val_loader=val_loader,
            baseline_snapshot=baseline_snapshot,
            baseline_metrics=baseline_metrics,
            train_manifest=train_manifest,
            geometry_indices=geometry_manifest["indices"],
            geometry_seeds=geometry_manifest["source_seeds"],
            protocol_hash=protocol_hash,
            device=device,
            output_dir=output_dir,
        )
        all_results[arm] = result
        all_training_rows.extend(training_rows)
        all_geometry_rows.extend(geometry_rows)
        all_geometry_probe_rows.extend(probe_rows)
        all_evaluation_rows.extend(evaluation_rows)

    fixed_rows = sorted([row for row in all_training_rows if row["arm"] == "fixed"], key=lambda row: row["global_update"])
    mgda_rows = sorted([row for row in all_training_rows if row["arm"] == "mgda"], key=lambda row: row["global_update"])
    paired_training_protocol = len(fixed_rows) == len(mgda_rows) and all(
        tuple(row[key] for key in ("global_update", "epoch", "batch_index", "mask_variant", "indices", "source_seed"))
        == tuple(other[key] for key in ("global_update", "epoch", "batch_index", "mask_variant", "indices", "source_seed"))
        for row, other in zip(fixed_rows, mgda_rows)
    )
    fixed_evaluation = sorted([row for row in all_evaluation_rows if row["arm"] == "fixed"], key=lambda row: row["update"])
    mgda_evaluation = sorted([row for row in all_evaluation_rows if row["arm"] == "mgda"], key=lambda row: row["update"])
    paired_evaluation_protocol = [row["update"] for row in fixed_evaluation] == [row["update"] for row in mgda_evaluation]
    artifacts = {
        "protocol": str(output_dir / "protocol.json"),
        "baseline_validation": str(output_dir / "baseline_validation.json"),
        "training_trace": str(output_dir / "training_trace.csv"),
        "geometry_trace": str(output_dir / "geometry_trace.csv"),
        "geometry_probe_per_batch": str(output_dir / "geometry_probe_per_batch.csv"),
        "evaluation_trace": str(output_dir / "evaluation_trace.csv"),
        "report": str(output_dir / "REPORT.md"),
    }
    _write_csv(output_dir / "training_trace.csv", all_training_rows)
    _write_csv(output_dir / "geometry_trace.csv", all_geometry_rows)
    _write_csv(output_dir / "geometry_probe_per_batch.csv", all_geometry_probe_rows)
    _write_csv(output_dir / "evaluation_trace.csv", all_evaluation_rows)
    summary = {
        "experiment": {
            "development_only": True,
            "network_training_performed": True,
            "formal_training_entry_modified": False,
            "optimizer_created": False,
            "manual_plain_sgd": True,
            "external_30_read": False,
            "final_test_read": False,
        },
        "protocol": {
            key: value
            for key, value in protocol.items()
            if key not in {"train_manifest", "geometry_manifest"}
        }
        | {
            "updates_per_arm": (
                sum(len(epoch_updates) for epoch_updates in train_manifest)
                if args.max_updates_per_arm is None
                else args.max_updates_per_arm
            ),
            "paired_training_manifest": paired_training_protocol,
            "paired_evaluation_manifest": paired_evaluation_protocol,
        },
        "baseline_validation": baseline_metrics,
        "arms": all_results,
        "paired_protocol_checks": {
            "training_rows_same_batch_order_mask_source": paired_training_protocol,
            "evaluation_updates_same": paired_evaluation_protocol,
            "fixed_training_rows": len(fixed_rows),
            "mgda_training_rows": len(mgda_rows),
        },
        "artifacts": artifacts,
    }
    plot_paths = _make_plots(output_dir, all_geometry_rows, all_evaluation_rows, all_training_rows)
    artifacts["plots"] = [str(output_dir / name) for name in plot_paths]
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "REPORT.md", summary)
    print(json.dumps(_jsonable(summary["arms"]), indent=2, ensure_ascii=False))
    print(f"REPORT.md: {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
