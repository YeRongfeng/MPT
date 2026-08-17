#!/usr/bin/env python3
"""Freeze and compare Stage-1/Stage-2 checkpoints on development validation.

This script is intentionally restricted to the validation manifest embedded in
one direct-cost Stage-2 run.  It does not discover or read external/final-test
data, and it does not select an ineligible checkpoint past the declared update
boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from grad_optimizer import trajectory_validity_metrics
from map_config import MAP_CONFIG
from posterior_pipeline import (
    DIRECT_COST_PAIRED_VALIDITY_KEYS,
    _direct_cost_forward,
    _direct_cost_source_distribution_metrics,
    _require_current_mask_semantics,
    _require_demo_target_semantics,
    _require_main_method_model,
    load_model,
    make_partial_dataset,
)


DEFAULT_RUN_DIR = Path("data/path_meanflow_stage2_mgda_sgd")
DEFAULT_OUTPUT = Path(
    "diagnostics/stage2_mgda_sgd_dev_selection_20260808"
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


def _write_csv(path: Path, rows) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    values = torch.as_tensor(values, dtype=torch.float32).flatten().cpu()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "median": float(torch.quantile(values, 0.5)),
        "p10": float(torch.quantile(values, 0.1)),
        "p90": float(torch.quantile(values, 0.9)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


@torch.no_grad()
def _evaluate_model(
    model,
    loader,
    device,
    *,
    sources_per_context: int,
    source_seed: int,
):
    model.eval()
    source_generator = torch.Generator(device=device).manual_seed(
        int(source_seed)
    )
    value_keys = (
        "J_F",
        "J_S",
        "J_K",
        "task_cost",
        "weighted_F",
        "weighted_S",
        "weighted_K",
        "path_length_m",
        "normalized_length",
        "third_difference_regularization",
        "curvature_max",
        "min_stability_margin",
        "composite_clearance_min",
    )
    values = {key: [] for key in value_keys}
    validity_chunks = {key: [] for key in DIRECT_COST_PAIRED_VALIDITY_KEYS}
    endpoint_yaw_chunks = []
    condition_count = 0
    trajectory_count = 0
    safe_at_1_count = 0
    safe_at_k_count = 0
    pairwise_sum = 0.0
    effective_rank_sum = 0.0

    for batch in loader:
        output = _direct_cost_forward(
            model,
            batch,
            device,
            sources_per_context=sources_per_context,
            source_generator=source_generator,
        )
        validity = trajectory_validity_metrics(
            output["geometry"]["position"],
            output["cost_map"],
            MAP_CONFIG.cost_map_info(),
            analytic_yaw=output["geometry"]["yaw"],
            analytic_curvature=output["geometry"]["curvature"],
            analytic_curvature_audit=output["curvature_audit"],
            mask=output["mask"],
            signed_mask_distance_map=output["signed_mask_distance"],
            start_pose=output["start_pose"],
            goal_pose=output["goal_pose"],
            condition_ids=output["condition_ids"],
        )
        components = output["components"]
        contexts = int(output["contexts"])
        repeats = int(output["sources_per_context"])
        count = int(output["task_cost"].numel())
        strict = validity["strict_valid"].reshape(contexts, repeats)
        safe_at_1_count += int(strict[:, 0].sum())
        safe_at_k_count += int(strict.any(dim=1).sum())
        condition_count += contexts
        trajectory_count += count

        positions = output["geometry"]["position"].reshape(
            contexts,
            repeats,
            output["geometry"]["position"].shape[-2],
            2,
        )
        diversity = _direct_cost_source_distribution_metrics(positions)
        pairwise_sum += diversity["pairwise_path_distance_m"] * contexts
        effective_rank_sum += diversity["covariance_effective_rank"] * contexts

        component_mapping = {
            "J_F": "forbidden_region",
            "J_S": "stability",
            "J_K": "curvature",
            "task_cost": "task_cost",
            "weighted_F": "weighted_forbidden_region",
            "weighted_S": "weighted_stability",
            "weighted_K": "weighted_curvature",
            "path_length_m": "path_length",
            "normalized_length": "normalized_length",
            "third_difference_regularization": (
                "third_difference_regularization"
            ),
            "curvature_max": "curvature_max",
            "min_stability_margin": "min_stability_margin",
            "composite_clearance_min": "composite_clearance_min",
        }
        for output_key, component_key in component_mapping.items():
            values[output_key].append(
                components[component_key].detach().float().cpu().reshape(-1)
            )
        for key in DIRECT_COST_PAIRED_VALIDITY_KEYS:
            validity_chunks[key].append(
                validity[key].detach().bool().cpu().reshape(-1)
            )
        endpoint_yaw_chunks.append(
            validity["endpoint_yaw_ok"].detach().bool().cpu().reshape(-1)
        )
        del output, validity

    if condition_count == 0 or trajectory_count == 0:
        raise RuntimeError("Development validation loader produced no samples")
    distributions = {
        key: _distribution(torch.cat(chunks)) for key, chunks in values.items()
    }
    snapshot = {
        key: torch.cat(chunks) for key, chunks in validity_chunks.items()
    }
    endpoint_yaw = torch.cat(endpoint_yaw_chunks)
    metrics = {
        "conditions": condition_count,
        "sources_per_context": int(sources_per_context),
        "trajectories": trajectory_count,
        "J_F": distributions["J_F"]["mean"],
        "J_S": distributions["J_S"]["mean"],
        "J_K": distributions["J_K"]["mean"],
        "task_cost": distributions["task_cost"]["mean"],
        "strict_valid_rate": float(snapshot["strict_valid"].float().mean()),
        "safe_at_1": safe_at_1_count / float(condition_count),
        "safe_at_k": safe_at_k_count / float(condition_count),
        "forbidden_ok_rate": float(
            snapshot["forbidden_region_ok"].float().mean()
        ),
        "stability_ok_rate": float(snapshot["stability_ok"].float().mean()),
        "curvature_ok_rate": float(snapshot["curvature_ok"].float().mean()),
        "yaw_ok_rate": float(endpoint_yaw.float().mean()),
        "pairwise_path_distance_m": pairwise_sum / float(condition_count),
        "covariance_effective_rank": effective_rank_sum / float(condition_count),
        "quality": {
            key: distributions[key]
            for key in (
                "path_length_m",
                "normalized_length",
                "third_difference_regularization",
                "curvature_max",
                "min_stability_margin",
                "composite_clearance_min",
            )
        },
        "weighted_costs": {
            "forbidden": distributions["weighted_F"]["mean"],
            "stability": distributions["weighted_S"]["mean"],
            "curvature": distributions["weighted_K"]["mean"],
        },
    }
    return metrics, snapshot


def _paired_transitions(current, baseline):
    result = {}
    for key in DIRECT_COST_PAIRED_VALIDITY_KEYS:
        base = baseline[key]
        value = current[key]
        result[key] = {
            "improvement_count": int(((~base) & value).sum()),
            "regression_count": int((base & (~value)).sum()),
            "unchanged_count": int((base == value).sum()),
        }
    return result


def _tensorboard_boundary(run_dir: Path, endpoint_updates: int):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    events = sorted(
        (run_dir / "tensorboard" / "stage2_direct_cost").glob(
            "events.out.tfevents.*"
        )
    )
    if len(events) != 1:
        raise ValueError(f"Expected one TensorBoard event file, found {events}")
    accumulator = EventAccumulator(
        str(events[0]), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    validation = accumulator.Scalars(
        "stage2_direct_cost/validation/task_cost"
    )
    train = accumulator.Scalars("stage2_direct_cost/train/task_cost")
    pre = [item for item in validation if item.step <= endpoint_updates]
    post = [item for item in validation if item.step > endpoint_updates]
    pre_best = min(pre, key=lambda item: item.value)
    post_best = min(post, key=lambda item: item.value) if post else None
    return {
        "event_file": str(events[0]),
        "last_train_step": int(train[-1].step),
        "last_validation_step": int(validation[-1].step),
        "pre_endpoint_best_step": int(pre_best.step),
        "pre_endpoint_best_task_cost": float(pre_best.value),
        "post_endpoint_best_step": (
            int(post_best.step) if post_best is not None else None
        ),
        "post_endpoint_best_task_cost": (
            float(post_best.value) if post_best is not None else None
        ),
    }


def _write_report(path: Path, summary) -> None:
    rows = summary["comparison_rows"]
    lines = [
        "# Stage-2 MGDA+SGD Development Selection Audit",
        "",
        "This audit reads only the frozen `dataset1_val/train` manifest embedded "
        "in the Stage-2 checkpoints. External-30 and final-test data were not "
        "opened.",
        "",
        "## Selection Integrity",
        "",
        f"- Intended 10-epoch endpoint supplied to this audit: "
        f"`{summary['selection']['intended_endpoint_updates']}` updates.",
        f"- Stored training configuration: "
        f"`{summary['selection']['configured_epochs']}` epochs, corresponding to "
        f"`{summary['selection']['configured_endpoint_updates']}` updates.",
        f"- Actual last training step: `{summary['selection']['last_train_step']}`.",
        f"- Pre-endpoint best: step `{summary['selection']['pre_endpoint_best_step']}` "
        f"with task cost `{summary['selection']['pre_endpoint_best_task_cost']:.9f}`.",
        f"- Current `stage2_best.pth`: step `{summary['selection']['current_best_step']}`; "
        "it is post-endpoint and ineligible.",
        "- The eligible pre-endpoint best checkpoint file was overwritten and no "
        "historical copy was found.",
        "- Selection is therefore **BLOCKED**. If 10 epochs was the independently "
        "predefined endpoint, its best checkpoint is missing. If the stored "
        "20-epoch configuration governs, the run is incomplete.",
        "- `stage2_last.pth` remains the reproducible 10-epoch endpoint, but choosing "
        "it now would replace the predefined best rule with a last-epoch rule.",
        "",
        "## Frozen Validation Comparison",
        "",
        "| model | within 10 epochs | J_F | J_S | J_K | task cost | strict | Safe@1 | Safe@K | pairwise m | eff. rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['eligible']} | {row['J_F']:.6f} | "
            f"{row['J_S']:.6f} | {row['J_K']:.6f} | "
            f"{row['task_cost']:.6f} | {row['strict_valid_rate']:.3%} | "
            f"{row['safe_at_1']:.3%} | {row['safe_at_k']:.3%} | "
            f"{row['pairwise_path_distance_m']:.6f} | "
            f"{row['covariance_effective_rank']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Detailed path-length, regularization, curvature, margin, paired "
            "transition, checkpoint-hash, and protocol results are in "
            "`summary.json` and `metrics.csv`.",
            "",
            "MGDA geometry acceptance and external evaluation were not run because "
            "there is no protocol-eligible selected checkpoint yet.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint-updates", type=int, default=50_000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.endpoint_updates <= 0:
        raise ValueError("endpoint-updates must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this frozen evaluation")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "stage2_best.pth"
    last_path = run_dir / "stage2_last.pth"

    last_checkpoint = torch.load(last_path, map_location="cpu")
    best_checkpoint = torch.load(best_path, map_location="cpu")
    if last_checkpoint.get("stage2_optimizer") != "SGD":
        raise ValueError("Run is not the frozen MGDA+SGD protocol")
    if not bool(last_checkpoint.get("stage2_use_mgda")):
        raise ValueError("Run did not use MGDA")
    if int(last_checkpoint.get("stage2_global_update", -1)) != args.endpoint_updates:
        raise ValueError("stage2_last.pth is not the declared endpoint")
    for key in (
        "fixed_validation_indices",
        "stage2_environment_split",
        "stage2_validation_data",
        "stage2_validation_split",
    ):
        if best_checkpoint.get(key) != last_checkpoint.get(key):
            raise ValueError(f"best/last validation manifest mismatch for {key}")

    validation_root = Path(last_checkpoint["stage2_validation_data"]).resolve()
    expected_validation_root = Path("data/dataset1_val").resolve()
    if validation_root != expected_validation_root:
        raise ValueError("Development audit is restricted to data/dataset1_val")
    validation_split = str(last_checkpoint["stage2_validation_split"])
    if validation_split != "train":
        raise ValueError("Development audit requires dataset1_val/train")
    environment_split = last_checkpoint["stage2_environment_split"]
    validation_environments = list(environment_split["validation"])
    validation_indices = list(last_checkpoint["fixed_validation_indices"])
    config = last_checkpoint["direct_cost_config"]
    sources = int(config["validation_sources"])
    if len(validation_environments) != 50 or len(validation_indices) != 50:
        raise ValueError("Frozen validation manifest is not 50 environments/contexts")
    if sources != 16:
        raise ValueError("Frozen validation manifest does not use 16 sources")
    completed_epochs = int(last_checkpoint["stage2_epoch"]) + 1
    if args.endpoint_updates % completed_epochs != 0:
        raise ValueError("Cannot infer an integral update count per epoch")
    updates_per_epoch = args.endpoint_updates // completed_epochs
    configured_epochs = int(config["epochs"])
    configured_endpoint_updates = configured_epochs * updates_per_epoch

    dataset, discovered = make_partial_dataset(
        validation_root,
        validation_split,
        compute_stability_map=True,
        mask_seed=int(last_checkpoint["mask_seed"]),
        p_mask=float(last_checkpoint["p_mask"]),
        mask_mode="stage2_independent",
        vehicle_radius_meters=float(last_checkpoint["vehicle_radius_meters"]),
        environment_names=validation_environments,
        dynamic_mask_noise=False,
        expected_environment_count=None,
        compute_stability_if_missing=True,
    )
    if list(discovered) != validation_environments:
        raise ValueError("Frozen validation environment order changed")
    loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    stage1_path = Path(last_checkpoint["source_checkpoint"]).resolve()
    model_specs = (
        ("stage1", stage1_path, True),
        ("stage2_last_10epoch", last_path, True),
        ("stage2_best_post_endpoint", best_path, False),
    )
    metrics_by_model = {}
    snapshots = {}
    checkpoint_manifest = {}
    for name, checkpoint_path, eligible in model_specs:
        model, _, checkpoint = load_model(checkpoint_path, torch.device("cpu"))
        _require_current_mask_semantics(checkpoint)
        _require_demo_target_semantics(checkpoint)
        _require_main_method_model(model)
        model.use_gradient_checkpoint = False
        model = model.to(device)
        metrics, snapshot = _evaluate_model(
            model,
            loader,
            device,
            sources_per_context=sources,
            source_seed=int(last_checkpoint.get("seed", 2026)) + 90_000,
        )
        metrics["eligible"] = bool(eligible)
        metrics["checkpoint"] = str(checkpoint_path)
        metrics["global_update"] = checkpoint.get("stage2_global_update", 0)
        metrics_by_model[name] = metrics
        snapshots[name] = snapshot
        checkpoint_manifest[name] = {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "stage": checkpoint.get("stage"),
            "stage2_global_update": checkpoint.get("stage2_global_update"),
            "stage2_epoch": checkpoint.get("stage2_epoch"),
            "eligible": bool(eligible),
        }
        del model, checkpoint
        torch.cuda.empty_cache()

    baseline = snapshots["stage1"]
    paired = {
        name: _paired_transitions(snapshot, baseline)
        for name, snapshot in snapshots.items()
        if name != "stage1"
    }
    boundary = _tensorboard_boundary(run_dir, args.endpoint_updates)
    comparison_rows = []
    for name, _, eligible in model_specs:
        metrics = metrics_by_model[name]
        comparison_rows.append(
            {
                "model": name,
                "eligible": bool(eligible),
                "J_F": metrics["J_F"],
                "J_S": metrics["J_S"],
                "J_K": metrics["J_K"],
                "task_cost": metrics["task_cost"],
                "strict_valid_rate": metrics["strict_valid_rate"],
                "safe_at_1": metrics["safe_at_1"],
                "safe_at_k": metrics["safe_at_k"],
                "forbidden_ok_rate": metrics["forbidden_ok_rate"],
                "stability_ok_rate": metrics["stability_ok_rate"],
                "curvature_ok_rate": metrics["curvature_ok_rate"],
                "pairwise_path_distance_m": metrics[
                    "pairwise_path_distance_m"
                ],
                "covariance_effective_rank": metrics[
                    "covariance_effective_rank"
                ],
                "path_length_m_mean": metrics["quality"]["path_length_m"][
                    "mean"
                ],
                "third_difference_regularization_mean": metrics["quality"][
                    "third_difference_regularization"
                ]["mean"],
            }
        )

    summary = {
        "audit": {
            "development_only": True,
            "validation_root": str(validation_root),
            "validation_split": validation_split,
            "validation_environment_count": len(validation_environments),
            "validation_context_count": len(validation_indices),
            "sources_per_context": sources,
            "external_30_read": False,
            "final_test_read": False,
        },
        "selection": {
            "status": "blocked_protocol_endpoint_mismatch_and_missing_best",
            "rule": "minimum validation task_cost; Safe@1 then strict only break exact ties",
            "intended_endpoint_updates": int(args.endpoint_updates),
            "updates_per_epoch": int(updates_per_epoch),
            "configured_epochs": configured_epochs,
            "configured_endpoint_updates": configured_endpoint_updates,
            **boundary,
            "current_best_step": int(
                best_checkpoint["stage2_global_update"]
            ),
            "current_best_eligible": False,
            "last_step": int(last_checkpoint["stage2_global_update"]),
            "last_eligible_endpoint": True,
            "selected_checkpoint": None,
            "geometry_acceptance_run": False,
            "external_gate_open": False,
        },
        "checkpoint_manifest": checkpoint_manifest,
        "validation_manifest": {
            "environments": validation_environments,
            "indices": validation_indices,
            "source_seed": int(last_checkpoint.get("seed", 2026)) + 90_000,
        },
        "metrics": metrics_by_model,
        "paired_transitions_vs_stage1": paired,
        "comparison_rows": comparison_rows,
        "artifacts": {
            "report": str(output_dir / "REPORT.md"),
            "summary": str(output_dir / "summary.json"),
            "metrics": str(output_dir / "metrics.csv"),
            "frozen_manifest": str(output_dir / "frozen_manifest.json"),
        },
    }
    frozen_manifest = {
        "checkpoints": checkpoint_manifest,
        "tensorboard": {
            "path": boundary["event_file"],
            "sha256": _sha256(Path(boundary["event_file"])),
            "size_bytes": Path(boundary["event_file"]).stat().st_size,
        },
        "validation_environments": validation_environments,
        "validation_indices": validation_indices,
        "sources_per_context": sources,
        "source_seed": summary["validation_manifest"]["source_seed"],
        "selection_status": summary["selection"]["status"],
    }
    _write_csv(output_dir / "metrics.csv", comparison_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "frozen_manifest.json", frozen_manifest)
    _write_report(output_dir / "REPORT.md", summary)
    print(json.dumps(_jsonable(summary["selection"]), indent=2))
    print(f"REPORT.md: {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
