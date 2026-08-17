"""Distribution-level first-order constrained Stage-2 audit.

This is an independent research audit.  It deliberately does not modify the
formal ``train_flow.py`` Stage-2 entry point and does not use an optimizer
state.  All three arms replay the same frozen condition/mask/source schedule:

    A: J_S only
    B: fixed weighted sum lambda_S J_S + lambda_F R_F
    C: first-order constrained update with a closed-form lambda_F

Only the development ``train`` split is opened.  The calibration and fixed
evaluation sets are deterministic subsets of that same development split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dataLoader_dit import UnevenPathDataLoader
from grad_optimizer import trajectory_validity_metrics
from map_config import MAP_CONFIG, SAFETY_COST_CONFIG, discover_environments
from posterior_pipeline import (
    _direct_cost_forward,
    _require_current_mask_semantics,
    _require_demo_target_semantics,
    _require_main_method_model,
    load_model,
)


AUDIT_SEMANTICS = "distribution_level_first_order_constraint_update_v1"
ARM_NAMES = ("stability_only", "fixed_weighted_sum", "distribution_constrained")
REQUIRED_BATCH_KEYS = (
    "map",
    "start_pose",
    "goal_pose",
    "cost_map",
    "mask",
    "dataset_index",
    "env_index",
    "path_index",
    "mask_variant",
    "mask_noise_seed",
)


@dataclass(frozen=True)
class ContextRecord:
    dataset_index: int
    env_index: int
    env_name: str
    path_index: int


@dataclass
class BatchSpec:
    batch: Dict[str, torch.Tensor]
    source: torch.Tensor
    epoch: int
    update: int
    source_sha256: str


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _seeded_order(values: Sequence, seed: int) -> List:
    values = list(values)
    generator = random.Random(int(seed))
    generator.shuffle(values)
    return values


def _stable_name_seed(name: str) -> int:
    digest = hashlib.sha256(str(name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 1_000_003


def _select_contexts(
    dataset: UnevenPathDataLoader,
    environment_names: Sequence[str],
    count: int,
    seed: int,
) -> List[ContextRecord]:
    """Select a deterministic round-robin context pool from train only."""
    if int(count) <= 0:
        raise ValueError("context count must be positive")
    wanted = {str(name) for name in environment_names}
    by_environment = {name: [] for name in sorted(wanted)}
    for dataset_index, (env_index, path_index) in enumerate(dataset.indexDict):
        env_name = str(dataset.env_list[int(env_index)])
        if env_name in by_environment:
            by_environment[env_name].append((int(dataset_index), int(path_index)))
    for env_name, values in by_environment.items():
        if not values:
            raise ValueError(f"Development environment has no paths: {env_name}")
        values[:] = _seeded_order(values, seed + _stable_name_seed(env_name))

    selected: List[ContextRecord] = []
    depth = 0
    names = sorted(by_environment)
    while len(selected) < int(count):
        added = False
        for env_name in names:
            values = by_environment[env_name]
            if depth < len(values):
                dataset_index, path_index = values[depth]
                env_index = int(dataset.indexDict[dataset_index][0])
                selected.append(
                    ContextRecord(
                        dataset_index=dataset_index,
                        env_index=env_index,
                        env_name=env_name,
                        path_index=path_index,
                    )
                )
                added = True
                if len(selected) >= int(count):
                    break
        if not added:
            break
        depth += 1
    if len(selected) != int(count):
        available = sum(len(values) for values in by_environment.values())
        raise ValueError(
            f"Requested {count} contexts from development train split, "
            f"selected {len(selected)} of {available}"
        )
    return selected


def _context_item(
    dataset: UnevenPathDataLoader,
    record: ContextRecord,
    *,
    mask_variant: int,
    noise_seed: int,
) -> Dict[str, torch.Tensor]:
    item = dataset.get_item(
        int(record.dataset_index),
        mask_variant=int(mask_variant),
        noise_seed=int(noise_seed),
    )
    return {key: item[key].detach().cpu() for key in REQUIRED_BATCH_KEYS}


def _stack_items(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not items:
        raise ValueError("Cannot stack an empty batch")
    return {
        key: torch.stack([item[key] for item in items], dim=0)
        for key in REQUIRED_BATCH_KEYS
    }


def _build_fixed_batches(
    dataset: UnevenPathDataLoader,
    contexts: Sequence[ContextRecord],
    *,
    batch_size: int,
    sources_per_context: int,
    epochs: int,
    seed: int,
    mask_seed: int,
    model_num_edges: int,
    max_updates: Optional[int] = None,
) -> Tuple[List[BatchSpec], List[Dict[str, object]]]:
    """Freeze all inputs needed by the three arms before training begins."""
    if int(batch_size) <= 0 or int(sources_per_context) <= 0:
        raise ValueError("batch_size and sources_per_context must be positive")
    batches: List[BatchSpec] = []
    manifest_rows: List[Dict[str, object]] = []
    global_update = 0
    context_list = list(contexts)
    for epoch in range(int(epochs)):
        order = _seeded_order(
            list(range(len(context_list))),
            int(seed) + 1009 * (epoch + 1),
        )
        for offset in range(0, len(order), int(batch_size)):
            if max_updates is not None and global_update >= int(max_updates):
                return batches, manifest_rows
            local_indices = order[offset : offset + int(batch_size)]
            records = [context_list[index] for index in local_indices]
            mask_variant = int(epoch) + 1
            items = []
            item_rows = []
            for local_position, record in enumerate(records):
                noise_seed = (
                    int(mask_seed)
                    + 1_000_003 * int(record.dataset_index)
                    + 15_485_863 * mask_variant
                    + 7_919
                    + 104_729 * global_update
                    + local_position
                )
                item = _context_item(
                    dataset,
                    record,
                    mask_variant=mask_variant,
                    noise_seed=noise_seed,
                )
                items.append(item)
                item_rows.append(
                    {
                        "dataset_index": record.dataset_index,
                        "env_index": record.env_index,
                        "env_name": record.env_name,
                        "path_index": record.path_index,
                        "mask_variant": mask_variant,
                        "mask_noise_seed": noise_seed,
                        "mask_sha256": _tensor_sha256(item["mask"]),
                    }
                )
            batch = _stack_items(items)
            source_generator = torch.Generator().manual_seed(
                int(seed) + 7_000_001 + global_update
            )
            source = torch.randn(
                len(records) * int(sources_per_context),
                int(model_num_edges),
                2,
                generator=source_generator,
                dtype=torch.float32,
            )
            source_sha256 = _tensor_sha256(source)
            batches.append(
                BatchSpec(
                    batch=batch,
                    source=source,
                    epoch=epoch,
                    update=global_update + 1,
                    source_sha256=source_sha256,
                )
            )
            manifest_rows.append(
                {
                    "epoch": epoch + 1,
                    "update": global_update + 1,
                    "contexts": item_rows,
                    "source_shape": list(source.shape),
                    "source_sha256": source_sha256,
                }
            )
            global_update += 1
    return batches, manifest_rows


def _zero_if_none(
    gradients: Sequence[Optional[torch.Tensor]],
    parameters: Sequence[torch.nn.Parameter],
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients)
    )


def _gradient_metrics(
    g_s: Sequence[torch.Tensor],
    g_f: Sequence[torch.Tensor],
    eps: float,
) -> Dict[str, float]:
    norm_s_sq = torch.zeros((), dtype=torch.float64, device=g_s[0].device)
    norm_f_sq = torch.zeros((), dtype=torch.float64, device=g_s[0].device)
    dot = torch.zeros((), dtype=torch.float64, device=g_s[0].device)
    for stability_gradient, forbidden_gradient in zip(g_s, g_f):
        stability_gradient = stability_gradient.double()
        forbidden_gradient = forbidden_gradient.double()
        norm_s_sq = norm_s_sq + stability_gradient.square().sum()
        norm_f_sq = norm_f_sq + forbidden_gradient.square().sum()
        dot = dot + torch.sum(stability_gradient * forbidden_gradient)
    norm_s = torch.sqrt(norm_s_sq.clamp_min(0.0))
    norm_f = torch.sqrt(norm_f_sq.clamp_min(0.0))
    cosine = dot / (norm_s * norm_f + float(eps))
    return {
        "stability_grad_norm": float(norm_s.detach().cpu()),
        "forbidden_grad_norm": float(norm_f.detach().cpu()),
        "gradient_dot": float(dot.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
    }


def closed_form_constraint_update(
    g_s: Sequence[torch.Tensor],
    g_f: Sequence[torch.Tensor],
    h_f: float,
    eta: float,
    eps: float,
) -> Tuple[float, Tuple[torch.Tensor, ...], float]:
    """Return lambda_F, combined gradient, and linearized R_F increment."""
    dot = torch.zeros((), dtype=torch.float64, device=g_s[0].device)
    norm_f_sq = torch.zeros((), dtype=torch.float64, device=g_s[0].device)
    for stability_gradient, forbidden_gradient in zip(g_s, g_f):
        stability_gradient = stability_gradient.double()
        forbidden_gradient = forbidden_gradient.double()
        dot = dot + torch.sum(forbidden_gradient * stability_gradient)
        norm_f_sq = norm_f_sq + forbidden_gradient.square().sum()
    numerator = float(h_f) / float(eta) - float(dot.detach().cpu())
    denominator = float(norm_f_sq.detach().cpu()) + float(eps)
    lambda_f = max(0.0, numerator / denominator)
    combined = tuple(
        stability_gradient + float(lambda_f) * forbidden_gradient
        for stability_gradient, forbidden_gradient in zip(g_s, g_f)
    )
    linearized_increment = -float(eta) * (
        float(dot.detach().cpu()) + float(lambda_f) * float(norm_f_sq.detach().cpu())
    )
    return lambda_f, combined, linearized_increment


def _apply_parameters(
    parameters: Sequence[torch.nn.Parameter],
    original: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    eta: float,
) -> None:
    with torch.no_grad():
        for parameter, old_value, gradient in zip(parameters, original, direction):
            parameter.copy_(old_value - float(eta) * gradient)


def _restore_parameters(
    parameters: Sequence[torch.nn.Parameter],
    original: Sequence[torch.Tensor],
) -> None:
    with torch.no_grad():
        for parameter, old_value in zip(parameters, original):
            parameter.copy_(old_value)


def _forward_costs(
    model,
    batch: Dict[str, torch.Tensor],
    source: torch.Tensor,
    device: torch.device,
    *,
    sources_per_context: int,
    scale_s: float,
    scale_f: float,
    with_grad: bool,
) -> Dict[str, object]:
    context_count = int(batch["map"].shape[0])
    if with_grad:
        output = _direct_cost_forward(
            model,
            batch,
            device,
            sources_per_context=int(sources_per_context),
            source=source.to(device=device),
        )
    else:
        with torch.no_grad():
            output = _direct_cost_forward(
                model,
                batch,
                device,
                sources_per_context=int(sources_per_context),
                source=source.to(device=device),
            )
    stability_values = output["components"]["stability"]
    forbidden_values = output["components"]["forbidden_region"]
    stability_raw = stability_values.mean()
    forbidden_raw = forbidden_values.mean()
    return {
        "output": output,
        "stability_raw": stability_raw,
        "forbidden_raw": forbidden_raw,
        "stability_scaled": stability_raw / float(scale_s),
        "forbidden_scaled": forbidden_raw / float(scale_f),
        "context_count": context_count,
    }


def _get_gradients(
    model,
    costs: Dict[str, object],
    parameters: Sequence[torch.nn.Parameter],
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    stability_scaled = costs["stability_scaled"]
    forbidden_scaled = costs["forbidden_scaled"]
    gradients_s = torch.autograd.grad(
        stability_scaled,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    gradients_f = torch.autograd.grad(
        forbidden_scaled,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    return _zero_if_none(gradients_s, parameters), _zero_if_none(
        gradients_f, parameters
    )


def _accept_candidate(
    *,
    forbidden_before: float,
    forbidden_after: float,
    tau_f: float,
    stability_before: float,
    stability_after: float,
    constraint_tolerance: float,
    stability_rise_tolerance: float,
    forbidden_decrease_tolerance: float,
) -> Tuple[bool, str]:
    stability_ok = stability_after <= stability_before + float(
        stability_rise_tolerance
    )
    if forbidden_before <= tau_f:
        if forbidden_after > tau_f + float(constraint_tolerance):
            return False, "feasible_constraint_regression"
        if not stability_ok:
            return False, "stability_rise"
        return True, "feasible_preserved"
    if forbidden_after >= forbidden_before - float(forbidden_decrease_tolerance):
        return False, "forbidden_not_reduced"
    if not stability_ok:
        return False, "stability_rise"
    return True, "constraint_repaired"


def _evaluate_schedule(
    model,
    schedule: Sequence[BatchSpec],
    device: torch.device,
    *,
    sources_per_context: int,
    reference_positions: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[Dict[str, float], List[torch.Tensor]]:
    """Evaluate one frozen schedule without opening validation/test data."""
    sums = {
        "stability_soft_cost": 0.0,
        "forbidden_soft_cost": 0.0,
        "stability_ok": 0.0,
        "forbidden_ok": 0.0,
    }
    proposal_count = 0
    curvature_values: List[torch.Tensor] = []
    drift_values: List[torch.Tensor] = []
    diversity_values: List[float] = []
    positions_out: List[torch.Tensor] = []
    for batch_index, spec in enumerate(schedule):
        costs = _forward_costs(
            model,
            spec.batch,
            spec.source,
            device,
            sources_per_context=int(sources_per_context),
            scale_s=1.0,
            scale_f=1.0,
            with_grad=False,
        )
        output = costs["output"]
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
        positions = output["geometry"]["position"].detach().cpu()
        positions_out.append(positions)
        if reference_positions is not None:
            drift_values.append(
                torch.sqrt(
                    (positions - reference_positions[batch_index]).square().mean(
                        dim=(1, 2)
                    )
                )
            )
        contexts = int(output["contexts"])
        repeats = int(output["sources_per_context"])
        grouped = positions.reshape(contexts, repeats, positions.shape[-2], 2)
        for context_paths in grouped:
            if repeats > 1:
                distances = torch.pdist(context_paths.flatten(1), p=2)
                diversity_values.append(
                    float(distances.median() / math.sqrt(context_paths.shape[1]))
                )
            else:
                diversity_values.append(0.0)
        count = int(output["task_cost"].numel())
        proposal_count += count
        sums["stability_soft_cost"] += float(components["stability"].sum())
        sums["forbidden_soft_cost"] += float(
            components["forbidden_region"].sum()
        )
        sums["stability_ok"] += float(validity["stability_ok"].float().sum())
        sums["forbidden_ok"] += float(
            validity["forbidden_region_ok"].float().sum()
        )
        curvature_values.append(validity["max_curvature"].detach().cpu())
    if proposal_count <= 0:
        raise RuntimeError("Evaluation schedule is empty")
    curvature = torch.cat(curvature_values).numpy()
    metrics = {
        "stability_soft_cost": sums["stability_soft_cost"] / proposal_count,
        "forbidden_soft_cost": sums["forbidden_soft_cost"] / proposal_count,
        "hard_stability_rate": sums["stability_ok"] / proposal_count,
        "hard_forbidden_rate": sums["forbidden_ok"] / proposal_count,
        "curvature_q90": float(np.quantile(curvature, 0.90)),
        "curvature_q99": float(np.quantile(curvature, 0.99)),
        "curvature_max": float(np.max(curvature)),
        "path_drift_rms_m": float(
            torch.cat(drift_values).mean() if drift_values else torch.tensor(0.0)
        ),
        "diversity_m": float(np.mean(diversity_values)),
        "proposals": proposal_count,
    }
    return metrics, positions_out


def _record_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite audit value: {value}")
    return value


def _safe_mean(values: Sequence[float]):
    return float(np.mean(values)) if values else None


def _summarize_updates(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    accepted = [row for row in rows if bool(row["accepted"])]
    repair_rows = [
        row
        for row in rows
        if float(row["forbidden_before"]) > float(row["tau_f"])
    ]
    repair_accepted = [row for row in repair_rows if bool(row["accepted"])]
    sacrifice_rows = [
        row
        for row in accepted
        if float(row["forbidden_before"]) > float(row["tau_f"])
        and float(row["stability_after"]) > float(row["stability_before"])
    ]
    return {
        "updates": len(rows),
        "accepted_updates": len(accepted),
        "accepted_rate": len(accepted) / max(len(rows), 1),
        "constraint_repair_candidates": len(repair_rows),
        "constraint_repair_accepted": len(repair_accepted),
        "constraint_repair_rate": len(repair_accepted) / max(len(repair_rows), 1),
        "accepted_with_stability_sacrifice": len(sacrifice_rows),
        "mean_lambda_f": _safe_mean(
            [float(row["lambda_f"]) for row in rows if math.isfinite(float(row["lambda_f"]))]
        ),
        "mean_accepted_eta": _safe_mean(
            [float(row["accepted_eta"]) for row in rows if float(row["accepted_eta"]) > 0]
        ),
        "skip_reasons": {
            reason: sum(1 for row in rows if row["skip_reason"] == reason)
            for reason in sorted({str(row["skip_reason"]) for row in rows})
            if reason
        },
    }


def _plot_results(output_dir: Path, eval_rows, update_rows, tau_f: float) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return []

    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm in ARM_NAMES:
        rows = [row for row in eval_rows if row["arm"] == arm]
        rows.sort(key=lambda row: int(row["update"]))
        x = [int(row["update"]) for row in rows]
        axes[0].plot(x, [float(row["stability_soft_cost"]) for row in rows], marker="o", label=arm)
        axes[1].plot(x, [float(row["forbidden_soft_cost"]) for row in rows], marker="o", label=arm)
    axes[0].set_title("Fixed evaluation stability soft cost")
    axes[1].set_title("Fixed evaluation forbidden soft cost")
    axes[0].set_xlabel("update")
    axes[1].set_xlabel("update")
    axes[0].set_ylabel("cost")
    axes[1].set_ylabel("cost")
    axes[1].axhline(tau_f, color="black", linestyle="--", linewidth=1, label="tau_F")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    path = output_dir / "cost_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    constrained = [row for row in update_rows if row["arm"] == "distribution_constrained"]
    constrained.sort(key=lambda row: int(row["update"]))
    x = [int(row["update"]) for row in constrained]
    axes[0].plot(x, [float(row["lambda_f"]) for row in constrained], marker="o")
    axes[0].set_title("Constraint multiplier lambda_F")
    axes[1].plot(x, [float(row["gradient_cosine"]) for row in constrained], marker="o")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Gradient cosine: g_S vs g_F")
    for axis in axes:
        axis.set_xlabel("update")
        axis.grid(alpha=0.25)
    path = output_dir / "constraint_update_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path.name)
    return figures


def _write_report(
    path: Path,
    *,
    config: Dict[str, object],
    calibration: Dict[str, object],
    eval_rows: Sequence[Dict[str, object]],
    update_summary: Dict[str, object],
    figures: Sequence[str],
) -> None:
    final_rows = {}
    for arm in ARM_NAMES:
        candidates = [row for row in eval_rows if row["arm"] == arm]
        if candidates:
            final_rows[arm] = max(candidates, key=lambda row: int(row["update"]))
    lines = [
        "# Distribution-Level First-Order Stage 2 Audit",
        "",
        f"- Semantics: `{AUDIT_SEMANTICS}`",
        "- Data scope: development `train` split only; validation, external 30, and final test were not opened.",
        f"- Epochs: `{config['epochs']}`; executed updates: `{config['executed_updates']}`.",
        f"- tau_F: `{float(config['tau_f']):.8g}` from `{config['tau_source']}`.",
        "- All arms replayed the same frozen condition, mask, source, and update order.",
        "",
        "## Calibration",
        "",
        f"Stage 1 calibration forbidden soft risk: `{float(calibration['forbidden_soft_cost']):.8g}`.",
        f"Fixed calibration proposals: `{int(calibration['proposals'])}`.",
        "",
        "## Final Fixed-Evaluation Snapshot",
        "",
        "| arm | stability soft | forbidden soft | hard stability | hard forbidden | curvature q90 | curvature q99 | drift (m) | diversity (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_NAMES:
        row = final_rows.get(arm)
        if row is None:
            continue
        lines.append(
            "| {arm} | {s:.6g} | {f:.6g} | {hs:.3%} | {hf:.3%} | {q90:.6g} | {q99:.6g} | {d:.6g} | {div:.6g} |".format(
                arm=arm,
                s=float(row["stability_soft_cost"]),
                f=float(row["forbidden_soft_cost"]),
                hs=float(row["hard_stability_rate"]),
                hf=float(row["hard_forbidden_rate"]),
                q90=float(row["curvature_q90"]),
                q99=float(row["curvature_q99"]),
                d=float(row["path_drift_rms_m"]),
                div=float(row["diversity_m"]),
            )
        )
    lines.extend(
        [
            "",
            "## Constrained-Arm Diagnostics",
            "",
            "The constrained arm uses the Euclidean first-order update and recomputes lambda_F for every backtracked eta. It has no Adam state, momentum, or per-trajectory dual variable.",
            "",
            f"- Update summary: `{json.dumps(update_summary, ensure_ascii=False)}`",
            "- A positive stability change on a repaired update is a local trade-off, not evidence of global superiority.",
            "- The fixed weighted-sum arm is a comparator for a persistent scalarized compromise; the constrained arm is evaluated by whether it reduces forbidden risk while preserving or only slightly increasing stability cost.",
            "",
            "## Interpretation Boundary",
            "",
            "This audit can distinguish the behavior of the three update rules under one frozen development schedule. It does not establish validation or final-test generalization, and it does not turn a successful local repair into a hard safety guarantee.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`",
            "- `updates.csv`",
            "- `evaluation.csv`",
            "- `audit_manifest.json`",
        ]
    )
    for figure in figures:
        lines.append(f"- `{figure}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit distribution-level first-order constrained Stage 2."
    )
    parser.add_argument("--checkpoint", default="data/path_meanflow/stage1_best.pth")
    parser.add_argument("--data-folder", default=str(MAP_CONFIG.dataset_root))
    parser.add_argument(
        "--output-dir",
        default="diagnostics/distribution_constrained_stage2_audit",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--p-mask", type=float, default=1.0)
    parser.add_argument(
        "--vehicle-radius-meters",
        type=float,
        default=SAFETY_COST_CONFIG.vehicle_radius_meters,
    )
    parser.add_argument("--train-environments", type=int, default=16)
    parser.add_argument("--calibration-environments", type=int, default=4)
    parser.add_argument("--eval-environments", type=int, default=4)
    parser.add_argument("--train-contexts", type=int, default=64)
    parser.add_argument("--calibration-contexts", type=int, default=16)
    parser.add_argument("--eval-contexts", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sources-per-context", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Smoke-only cap. Omit for a complete epoch over the frozen train pool.",
    )
    parser.add_argument("--eta", type=float, default=1e-5)
    parser.add_argument("--max-backtracks", type=int, default=5)
    parser.add_argument("--min-eta", type=float, default=1e-12)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-5)
    parser.add_argument("--stability-rise-tolerance", type=float, default=1e-3)
    parser.add_argument("--forbidden-decrease-tolerance", type=float, default=0.0)
    parser.add_argument("--gradient-epsilon", type=float, default=1e-12)
    parser.add_argument("--formula-epsilon", type=float, default=1e-12)
    parser.add_argument("--scale-s", type=float, default=1.0)
    parser.add_argument("--scale-f", type=float, default=1.0)
    parser.add_argument(
        "--scale-manifest",
        default=None,
        help=(
            "Optional frozen Stage-1 calibration JSON containing scale_S and "
            "scale_F. Non-unit scales require this file."
        ),
    )
    parser.add_argument("--tau-f", type=float, default=None)
    parser.add_argument("--tau-f-factor", type=float, default=1.0)
    parser.add_argument(
        "--weighted-stability",
        type=float,
        default=SAFETY_COST_CONFIG.obstacle_weight,
    )
    parser.add_argument(
        "--weighted-forbidden",
        type=float,
        default=SAFETY_COST_CONFIG.forbidden_weight,
    )
    parser.add_argument("--eval-every-updates", type=int, default=4)
    parser.add_argument("--save-arm-checkpoints", action="store_true")
    return parser


def _validate_args(args) -> None:
    positive = (
        "train_environments",
        "calibration_environments",
        "eval_environments",
        "train_contexts",
        "calibration_contexts",
        "eval_contexts",
        "batch_size",
        "sources_per_context",
        "epochs",
        "eta",
        "min_eta",
        "eval_every_updates",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    if args.max_backtracks < 0:
        raise ValueError("--max-backtracks cannot be negative")
    if args.p_mask < 0.0 or args.p_mask > 1.0:
        raise ValueError("--p-mask must be in [0,1]")
    if args.scale_s <= 0.0 or args.scale_f <= 0.0:
        raise ValueError("--scale-s and --scale-f must be positive")
    if args.scale_manifest is None and (
        not math.isclose(float(args.scale_s), 1.0)
        or not math.isclose(float(args.scale_f), 1.0)
    ):
        raise ValueError(
            "Non-unit cost scales require --scale-manifest; the default audit "
            "uses fixed scale_S=scale_F=1.0."
        )
    if args.tau_f is not None and args.tau_f < 0.0:
        raise ValueError("--tau-f cannot be negative")
    if args.tau_f_factor <= 0.0:
        raise ValueError("--tau-f-factor must be positive")
    if args.weighted_stability < 0.0 or args.weighted_forbidden < 0.0:
        raise ValueError("weighted coefficients cannot be negative")


def run_audit(args) -> Path:
    _validate_args(args)
    scale_manifest = None
    if args.scale_manifest is not None:
        scale_manifest_path = Path(args.scale_manifest)
        scale_manifest = json.loads(scale_manifest_path.read_text(encoding="utf-8"))
        if "scale_S" not in scale_manifest or "scale_F" not in scale_manifest:
            raise ValueError("scale manifest must contain scale_S and scale_F")
        args.scale_s = float(scale_manifest["scale_S"])
        args.scale_f = float(scale_manifest["scale_F"])
        if args.scale_s <= 0.0 or args.scale_f <= 0.0:
            raise ValueError("scale manifest scales must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    model, model_args, checkpoint = load_model(checkpoint_path, device)
    _require_main_method_model(model)
    _require_current_mask_semantics(checkpoint)
    _require_demo_target_semantics(checkpoint)
    model.use_gradient_checkpoint = False
    model.eval()
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    initial_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    all_environment_names = discover_environments(
        Path(args.data_folder) / "train",
        expected_count=None,
    )
    split_count = (
        int(args.train_environments)
        + int(args.calibration_environments)
        + int(args.eval_environments)
    )
    if split_count > len(all_environment_names):
        raise ValueError(
            f"Requested {split_count} development environments, "
            f"but train split has {len(all_environment_names)}"
        )
    split_order = _seeded_order(all_environment_names, int(args.seed) + 500)
    train_envs = sorted(split_order[: int(args.train_environments)])
    calibration_envs = sorted(
        split_order[
            int(args.train_environments) : int(args.train_environments)
            + int(args.calibration_environments)
        ]
    )
    eval_envs = sorted(
        split_order[
            int(args.train_environments) + int(args.calibration_environments) : split_count
        ]
    )
    dataset = UnevenPathDataLoader(
        train_envs + calibration_envs + eval_envs,
        str(Path(args.data_folder) / "train"),
        compute_stability_map=True,
        use_precomputed_stability=True,
        compute_stability_if_missing=False,
        partial_observation=True,
        include_mask=True,
        mask_seed=int(args.mask_seed),
        p_mask=float(args.p_mask),
        dynamic_mask_noise=False,
        mask_mode="stage2_independent",
        vehicle_radius_meters=float(args.vehicle_radius_meters),
        encode_path_coordinates=True,
    )
    train_contexts = _select_contexts(
        dataset,
        train_envs,
        int(args.train_contexts),
        int(args.seed) + 1000,
    )
    calibration_contexts = _select_contexts(
        dataset,
        calibration_envs,
        int(args.calibration_contexts),
        int(args.seed) + 2000,
    )
    eval_contexts = _select_contexts(
        dataset,
        eval_envs,
        int(args.eval_contexts),
        int(args.seed) + 3000,
    )

    calibration_schedule, calibration_manifest = _build_fixed_batches(
        dataset,
        calibration_contexts,
        batch_size=int(args.batch_size),
        sources_per_context=int(args.sources_per_context),
        epochs=1,
        seed=int(args.seed) + 20_000,
        mask_seed=int(args.mask_seed),
        model_num_edges=model.num_edges,
    )
    eval_schedule, eval_manifest = _build_fixed_batches(
        dataset,
        eval_contexts,
        batch_size=int(args.batch_size),
        sources_per_context=int(args.sources_per_context),
        epochs=1,
        seed=int(args.seed) + 30_000,
        mask_seed=int(args.mask_seed),
        model_num_edges=model.num_edges,
    )
    train_schedule, train_manifest = _build_fixed_batches(
        dataset,
        train_contexts,
        batch_size=int(args.batch_size),
        sources_per_context=int(args.sources_per_context),
        epochs=int(args.epochs),
        seed=int(args.seed) + 10_000,
        mask_seed=int(args.mask_seed),
        model_num_edges=model.num_edges,
        max_updates=args.max_updates,
    )
    if not train_schedule:
        raise RuntimeError("Training schedule is empty")
    if args.max_updates is None and len(train_schedule) < int(args.epochs):
        raise RuntimeError("Training schedule does not contain a complete epoch")

    calibration_metrics, _ = _evaluate_schedule(
        model,
        calibration_schedule,
        device,
        sources_per_context=int(args.sources_per_context),
    )
    calibration_forbidden_risk = float(calibration_metrics["forbidden_soft_cost"])
    if args.tau_f is None:
        tau_f = calibration_forbidden_risk * float(args.tau_f_factor)
        tau_source = f"stage1_calibration_forbidden_risk*{args.tau_f_factor:g}"
    else:
        tau_f = float(args.tau_f)
        tau_source = "explicit_cli_tau_f"

    manifest = {
        "audit_semantics": AUDIT_SEMANTICS,
        "checkpoint": str(checkpoint_path),
        "data_folder": str(Path(args.data_folder).resolve()),
        "data_split": "train",
        "validation_opened": False,
        "external_30_opened": False,
        "final_test_opened": False,
        "environment_split": {
            "train": train_envs,
            "calibration": calibration_envs,
            "fixed_evaluation": eval_envs,
        },
        "context_pools": {
            "train": [record.__dict__ for record in train_contexts],
            "calibration": [record.__dict__ for record in calibration_contexts],
            "fixed_evaluation": [record.__dict__ for record in eval_contexts],
        },
        "train_schedule": train_manifest,
        "calibration_schedule": calibration_manifest,
        "evaluation_schedule": eval_manifest,
        "cost_contract": {
            "stability_component": "privileged_planning_cost.components.stability",
            "forbidden_component": "privileged_planning_cost.components.forbidden_region",
            "forward": "model(c_obs,z,t=1,r=0) -> evaluate_trajectory_state",
            "scale_s": float(args.scale_s),
            "scale_f": float(args.scale_f),
            "scale_manifest": str(args.scale_manifest)
            if args.scale_manifest is not None
            else None,
        },
    }
    _write_json(output_dir / "audit_manifest.json", manifest)

    initial_eval_metrics, reference_positions = _evaluate_schedule(
        model,
        eval_schedule,
        device,
        sources_per_context=int(args.sources_per_context),
    )
    eval_rows: List[Dict[str, object]] = []
    update_rows: List[Dict[str, object]] = []
    for arm in ARM_NAMES:
        model.load_state_dict(initial_state, strict=True)
        model.eval()
        eval_metrics, _ = _evaluate_schedule(
            model,
            eval_schedule,
            device,
            sources_per_context=int(args.sources_per_context),
            reference_positions=reference_positions,
        )
        eval_rows.append({"arm": arm, "epoch": 0, "update": 0, **eval_metrics})
        last_eval_update = 0
        for spec in train_schedule:
            current = _forward_costs(
                model,
                spec.batch,
                spec.source,
                device,
                sources_per_context=int(args.sources_per_context),
                scale_s=float(args.scale_s),
                scale_f=float(args.scale_f),
                with_grad=True,
            )
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            g_s, g_f = _get_gradients(model, current, parameters)
            gradient_stats = _gradient_metrics(g_s, g_f, float(args.formula_epsilon))
            stability_before = _finite_float(current["stability_raw"].detach().cpu())
            forbidden_before = _finite_float(current["forbidden_raw"].detach().cpu())
            h_scaled = float(current["forbidden_scaled"].detach().cpu()) - float(
                tau_f / float(args.scale_f)
            )
            h_raw = forbidden_before - tau_f
            base_eta = float(args.eta)
            proposed_eta = base_eta
            lambda_proposed = float("nan")
            linearized_proposed_raw = float("nan")
            accepted_eta = 0.0
            accepted_lambda = float("nan")
            actual_new_stability = stability_before
            actual_new_forbidden = forbidden_before
            accepted = False
            skip_reason = ""
            original = [parameter.detach().clone() for parameter in parameters]

            if arm == "distribution_constrained":
                if (
                    h_scaled > 0.0
                    and gradient_stats["forbidden_grad_norm"]
                    <= float(args.gradient_epsilon)
                ):
                    skip_reason = "forbidden_gradient_too_small"
                    _restore_parameters(parameters, original)
                else:
                    lambda_proposed, _, linearized_increment = closed_form_constraint_update(
                        g_s,
                        g_f,
                        h_scaled,
                        base_eta,
                        float(args.formula_epsilon),
                    )
                    linearized_proposed_raw = (
                        forbidden_before + linearized_increment * float(args.scale_f)
                    )
                    for backtrack in range(int(args.max_backtracks) + 1):
                        candidate_eta = base_eta / (2.0**backtrack)
                        if candidate_eta < float(args.min_eta):
                            break
                        lambda_candidate, combined, linearized_increment = (
                            closed_form_constraint_update(
                                g_s,
                                g_f,
                                h_scaled,
                                candidate_eta,
                                float(args.formula_epsilon),
                            )
                        )
                        _restore_parameters(parameters, original)
                        _apply_parameters(parameters, original, combined, candidate_eta)
                        candidate = _forward_costs(
                            model,
                            spec.batch,
                            spec.source,
                            device,
                            sources_per_context=int(args.sources_per_context),
                            scale_s=float(args.scale_s),
                            scale_f=float(args.scale_f),
                            with_grad=False,
                        )
                        try:
                            candidate_stability = _finite_float(
                                candidate["stability_raw"]
                            )
                            candidate_forbidden = _finite_float(
                                candidate["forbidden_raw"]
                            )
                        except FloatingPointError:
                            _restore_parameters(parameters, original)
                            accepted = False
                            skip_reason = "non_finite_candidate"
                            continue
                        accepted, reason = _accept_candidate(
                            forbidden_before=forbidden_before,
                            forbidden_after=candidate_forbidden,
                            tau_f=tau_f,
                            stability_before=stability_before,
                            stability_after=candidate_stability,
                            constraint_tolerance=float(args.constraint_tolerance),
                            stability_rise_tolerance=float(args.stability_rise_tolerance),
                            forbidden_decrease_tolerance=float(
                                args.forbidden_decrease_tolerance
                            ),
                        )
                        if accepted:
                            accepted_eta = candidate_eta
                            accepted_lambda = lambda_candidate
                            actual_new_stability = candidate_stability
                            actual_new_forbidden = candidate_forbidden
                            linearized_proposed_raw = (
                                forbidden_before
                                + linearized_increment * float(args.scale_f)
                            )
                            skip_reason = reason
                            break
                        _restore_parameters(parameters, original)
                        skip_reason = reason
                    if not accepted:
                        _restore_parameters(parameters, original)
                        accepted_eta = 0.0
                        actual_new_stability = stability_before
                        actual_new_forbidden = forbidden_before
            else:
                if arm == "stability_only":
                    combined = g_s
                else:
                    combined = tuple(
                        float(args.weighted_stability) * stability_gradient
                        + float(args.weighted_forbidden) * forbidden_gradient
                        for stability_gradient, forbidden_gradient in zip(g_s, g_f)
                    )
                _apply_parameters(parameters, original, combined, base_eta)
                candidate = _forward_costs(
                    model,
                    spec.batch,
                    spec.source,
                    device,
                    sources_per_context=int(args.sources_per_context),
                    scale_s=float(args.scale_s),
                    scale_f=float(args.scale_f),
                    with_grad=False,
                )
                try:
                    actual_new_stability = _finite_float(candidate["stability_raw"])
                    actual_new_forbidden = _finite_float(candidate["forbidden_raw"])
                    accepted = True
                    accepted_eta = base_eta
                    skip_reason = "fixed_sgd_step"
                    linearized_increment = -base_eta * sum(
                        float(torch.sum(g_f_i * combined_i).detach().cpu())
                        for g_f_i, combined_i in zip(g_f, combined)
                    )
                    linearized_proposed_raw = (
                        forbidden_before + linearized_increment * float(args.scale_f)
                    )
                except FloatingPointError:
                    _restore_parameters(parameters, original)
                    accepted = False
                    accepted_eta = 0.0
                    actual_new_stability = stability_before
                    actual_new_forbidden = forbidden_before
                    skip_reason = "non_finite_candidate"

            if not all(
                math.isfinite(value)
                for value in (
                    actual_new_stability,
                    actual_new_forbidden,
                    gradient_stats["stability_grad_norm"],
                    gradient_stats["forbidden_grad_norm"],
                )
            ):
                _restore_parameters(parameters, original)
                accepted = False
                accepted_eta = 0.0
                actual_new_stability = stability_before
                actual_new_forbidden = forbidden_before
                skip_reason = "non_finite_candidate"

            row = {
                "arm": arm,
                "epoch": int(spec.epoch) + 1,
                "update": int(spec.update),
                "batch_source_sha256": spec.source_sha256,
                "stability_before": stability_before,
                "forbidden_before": forbidden_before,
                "J_S": stability_before,
                "R_F": forbidden_before,
                "h_f": h_raw,
                "h_f_scaled": h_scaled,
                "tau_f": tau_f,
                "scale_s": float(args.scale_s),
                "scale_f": float(args.scale_f),
                "J_S_scaled": stability_before / float(args.scale_s),
                "R_F_scaled": forbidden_before / float(args.scale_f),
                "stability_grad_norm": gradient_stats["stability_grad_norm"],
                "forbidden_grad_norm": gradient_stats["forbidden_grad_norm"],
                "gradient_dot": gradient_stats["gradient_dot"],
                "gradient_cosine": gradient_stats["gradient_cosine"],
                "lambda_f": accepted_lambda
                if arm == "distribution_constrained" and accepted
                else lambda_proposed
                if arm == "distribution_constrained"
                else float("nan"),
                "lambda_F": accepted_lambda
                if arm == "distribution_constrained" and accepted
                else lambda_proposed
                if arm == "distribution_constrained"
                else float("nan"),
                "lambda_f_proposed": lambda_proposed,
                "proposed_eta": proposed_eta,
                "accepted_eta": accepted_eta,
                "eta_proposed": proposed_eta,
                "eta_accepted": accepted_eta,
                "linearized_predicted_forbidden": linearized_proposed_raw,
                "linearized_predicted_R_F": linearized_proposed_raw,
                "stability_after": actual_new_stability,
                "forbidden_after": actual_new_forbidden,
                "J_S_new": actual_new_stability,
                "R_F_new": actual_new_forbidden,
                "accepted": bool(accepted),
                "skip_reason": skip_reason,
            }
            update_rows.append(row)

            if (
                int(spec.update) % int(args.eval_every_updates) == 0
                or int(spec.update) == len(train_schedule)
            ):
                eval_metrics, _ = _evaluate_schedule(
                    model,
                    eval_schedule,
                    device,
                    sources_per_context=int(args.sources_per_context),
                    reference_positions=reference_positions,
                )
                eval_rows.append(
                    {
                        "arm": arm,
                        "epoch": int(spec.epoch) + 1,
                        "update": int(spec.update),
                        **eval_metrics,
                    }
                )
                last_eval_update = int(spec.update)
        if last_eval_update != len(train_schedule):
            eval_metrics, _ = _evaluate_schedule(
                model,
                eval_schedule,
                device,
                sources_per_context=int(args.sources_per_context),
                reference_positions=reference_positions,
            )
            eval_rows.append(
                {
                    "arm": arm,
                    "epoch": int(args.epochs),
                    "update": len(train_schedule),
                    **eval_metrics,
                }
            )
        if args.save_arm_checkpoints:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_args": model_args,
                    "audit_semantics": AUDIT_SEMANTICS,
                    "arm": arm,
                    "tau_f": tau_f,
                    "data_split": "train_development_only",
                },
                output_dir / f"{arm}_final.pth",
            )

    update_rows.sort(key=lambda row: (str(row["arm"]), int(row["update"])))
    eval_rows.sort(key=lambda row: (str(row["arm"]), int(row["update"])))
    _record_csv(output_dir / "updates.csv", update_rows)
    _record_csv(output_dir / "evaluation.csv", eval_rows)
    update_summary = {
        arm: _summarize_updates([row for row in update_rows if row["arm"] == arm])
        for arm in ARM_NAMES
    }
    summary = {
        "audit_semantics": AUDIT_SEMANTICS,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "data_split": "train_development_only",
        "validation_opened": False,
        "external_30_opened": False,
        "final_test_opened": False,
        "config": {
            **vars(args),
            "executed_updates": len(train_schedule),
            "tau_f": tau_f,
            "tau_source": tau_source,
        },
        "calibration": calibration_metrics,
        "initial_fixed_evaluation": initial_eval_metrics,
        "update_summary": update_summary,
        "final_fixed_evaluation": {
            arm: max(
                [row for row in eval_rows if row["arm"] == arm],
                key=lambda row: int(row["update"]),
            )
            for arm in ARM_NAMES
        },
        "initial_state_sha256": hashlib.sha256(
            b"".join(
                value.detach().cpu().contiguous().numpy().tobytes()
                for _, value in sorted(initial_state.items())
            )
        ).hexdigest(),
    }
    _write_json(output_dir / "summary.json", summary)
    figures = _plot_results(output_dir, eval_rows, update_rows, tau_f)
    _write_report(
        output_dir / "REPORT.md",
        config=summary["config"],
        calibration=calibration_metrics,
        eval_rows=eval_rows,
        update_summary=update_summary,
        figures=figures,
    )
    print(f"Saved distribution-constrained audit to {output_dir}")
    print(json.dumps(update_summary, indent=2, ensure_ascii=False))
    return output_dir


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_audit(args)


if __name__ == "__main__":
    main()
