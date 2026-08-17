#!/usr/bin/env python3
"""Development-only MGDA geometry audit for the direct-cost Stage 2 model.

This module is deliberately separate from ``train_flow.py``.  It performs no
optimizer step and never opens validation, external, or final-test data.  For
each fixed train batch it evaluates the current deployment-point forward pass,
gets one full-parameter gradient for each raw privileged cost, solves the
three-objective MGDA problem by enumerating the simplex active sets, and then
optionally verifies the direction with temporary parameter perturbations.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from grad_optimizer import PRIVILEGED_COST_SEMANTICS
from map_config import MAP_CONFIG, SAFETY_COST_CONFIG
from posterior_pipeline import (
    DIRECT_COST_CONTEXT_SEMANTICS,
    DIRECT_COST_STAGE2_SEMANTICS,
    _direct_cost_forward,
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
PAIR_NAMES = (("F", "S"), ("F", "K"), ("S", "K"))
PAIR_COLUMNS = {
    ("F", "S"): "FS",
    ("F", "K"): "FK",
    ("S", "K"): "SK",
}
STEP_SCALES = (
    ("eta", 1.0),
    ("eta_half", 0.5),
    ("eta_quarter", 0.25),
    ("eta_eighth", 0.125),
)

DEFAULT_CHECKPOINT = Path("data/pmf_s2_stability_only/stage2_last.pth")
DEFAULT_OUTPUT = Path("diagnostics/direct_cost_mgda_geometry_20260807")


def _jsonable(value: Any):
    """Convert torch/numpy values and non-finite floats to JSON values."""
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


class FixedMaskVariantDataset(Dataset):
    """Expose one deterministic Stage-2 mask variant through DataLoader."""

    def __init__(self, dataset, mask_variant: int):
        self.dataset = dataset
        self.mask_variant = int(mask_variant)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.get_item(
            int(index),
            mask_variant=self.mask_variant,
        )


def _as_square_symmetric_gram(gram: torch.Tensor) -> torch.Tensor:
    gram = torch.as_tensor(gram, dtype=torch.float64, device="cpu")
    if gram.shape != (3, 3):
        raise ValueError(f"MGDA Gram matrix must have shape (3,3), got {gram.shape}")
    if not bool(torch.isfinite(gram).all()):
        raise FloatingPointError("MGDA Gram matrix contains NaN or Inf")
    return (gram + gram.T) * 0.5


def _candidate_from_alpha(
    gram: torch.Tensor,
    alpha: torch.Tensor,
    *,
    label: str,
    support: tuple[int, ...],
    feasibility_tol: float,
):
    alpha = torch.as_tensor(alpha, dtype=torch.float64, device="cpu").reshape(3)
    if not bool(torch.isfinite(alpha).all()):
        return None
    if abs(float(alpha.sum()) - 1.0) > feasibility_tol:
        return None
    if float(alpha.min()) < -feasibility_tol:
        return None
    # Remove only numerical negative zero and normalize the simplex sum.
    alpha = alpha.clamp_min(0.0)
    total = float(alpha.sum())
    if total <= 0.0:
        return None
    alpha = alpha / total
    objective = torch.dot(alpha, gram @ alpha)
    return {
        "alpha": alpha,
        "objective": float(objective),
        "label": label,
        "support": support,
    }


def solve_mgda_active_set(
    gram: torch.Tensor,
    *,
    feasibility_tol: float = 1e-8,
    degeneracy_tol: float = 1e-14,
) -> dict[str, Any]:
    """Solve the three-objective MGDA simplex problem exactly by active sets.

    The candidates are the equality-constrained interior solution, all three
    two-objective edges using their closed-form minimum-norm solution, and the
    three vertices.  An unconstrained edge solution outside ``[0, 1]`` is not
    admitted; its endpoint is covered explicitly by the vertex candidates.
    The tiny KKT system is solved directly, with least-squares fallback only
    for a singular Gram matrix.  No general QP solver is used.
    """
    gram = _as_square_symmetric_gram(gram)
    if feasibility_tol <= 0.0 or degeneracy_tol <= 0.0:
        raise ValueError("MGDA tolerances must be positive")

    candidates = []
    ones = torch.ones(3, dtype=torch.float64)
    # Equality-constrained interior candidate: H alpha + lambda 1 = 0,
    # 1^T alpha = 1.  Solving the KKT system also handles non-diagonal H.
    kkt = torch.zeros((4, 4), dtype=torch.float64)
    kkt[:3, :3] = gram
    kkt[:3, 3] = ones
    kkt[3, :3] = ones
    rhs = torch.zeros(4, dtype=torch.float64)
    rhs[3] = 1.0
    try:
        kkt_solution = torch.linalg.solve(kkt, rhs)
    except RuntimeError:
        kkt_solution = torch.linalg.lstsq(kkt, rhs).solution
    residual = torch.linalg.vector_norm(kkt @ kkt_solution - rhs)
    if float(residual) <= feasibility_tol * max(1.0, float(rhs.norm())):
        candidate = _candidate_from_alpha(
            gram,
            kkt_solution[:3],
            label="interior",
            support=(0, 1, 2),
            feasibility_tol=feasibility_tol,
        )
        if candidate is not None:
            candidates.append(candidate)

    # On an edge with alpha_i=t and alpha_j=1-t, the closed-form solution is
    # t=(H_jj-H_ij)/(H_ii-2H_ij+H_jj).
    for i, j in ((0, 1), (0, 2), (1, 2)):
        denominator = float(gram[i, i] - 2.0 * gram[i, j] + gram[j, j])
        scale = max(
            1.0,
            abs(float(gram[i, i])),
            abs(float(gram[i, j])),
            abs(float(gram[j, j])),
        )
        if denominator <= degeneracy_tol * scale:
            coefficient_i = 0.5
        else:
            coefficient_i = float(gram[j, j] - gram[i, j]) / denominator
        coefficient_j = 1.0 - coefficient_i
        alpha = torch.zeros(3, dtype=torch.float64)
        alpha[i] = coefficient_i
        alpha[j] = coefficient_j
        candidate = _candidate_from_alpha(
            gram,
            alpha,
            label=f"{OBJECTIVE_NAMES[i]}-{OBJECTIVE_NAMES[j]}",
            support=(i, j),
            feasibility_tol=feasibility_tol,
        )
        if candidate is not None:
            candidates.append(candidate)

    # Vertices are necessary when a closed-form edge minimizer falls outside
    # its segment, and are also the exact solution for a single active cost.
    for index, name in enumerate(OBJECTIVE_NAMES):
        alpha = torch.zeros(3, dtype=torch.float64)
        alpha[index] = 1.0
        candidate = _candidate_from_alpha(
            gram,
            alpha,
            label=name,
            support=(index,),
            feasibility_tol=feasibility_tol,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        raise FloatingPointError("MGDA active-set enumeration produced no feasible candidate")
    selected = min(candidates, key=lambda item: item["objective"])
    alpha = selected["alpha"]
    if selected["label"] == "interior" and float(alpha.min()) <= feasibility_tol:
        active = [
            OBJECTIVE_NAMES[index]
            for index, coefficient in enumerate(alpha.tolist())
            if coefficient > feasibility_tol
        ]
        selected_label = "-".join(active) if active else selected["label"]
    else:
        selected_label = selected["label"]
    return {
        "alpha": alpha,
        "objective": float(selected["objective"]),
        "label": selected_label,
        "support": selected["support"],
        "candidate_count": len(candidates),
        "candidates": [
            {
                "label": item["label"],
                "objective": item["objective"],
                "alpha": item["alpha"],
            }
            for item in candidates
        ],
    }


def _flatten_gradients_to_device(parameters, gradients, device) -> torch.Tensor:
    """Flatten a parameter-gradient tuple on the source device."""
    chunks = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            chunks.append(
                torch.zeros(
                    parameter.numel(),
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            chunks.append(gradient.detach().to(dtype=torch.float32).reshape(-1))
    vector = torch.cat(chunks)
    if not bool(torch.isfinite(vector).all()):
        raise FloatingPointError("Parameter gradient contains NaN or Inf")
    return vector


def _gram_from_device_vectors(vectors: Mapping[str, torch.Tensor], device):
    """Compute the tiny Gram matrix on GPU before copying vectors to CPU."""
    gram = torch.zeros((3, 3), dtype=torch.float64, device=device)
    for i, name_i in enumerate(OBJECTIVE_NAMES):
        for j in range(i, len(OBJECTIVE_NAMES)):
            name_j = OBJECTIVE_NAMES[j]
            # Accumulate in float32 on the GPU and retain float64 scalars for
            # the 3x3 active-set solve.  This avoids a 50M-dimensional CPU dot
            # for every pair and every batch.
            value = torch.dot(vectors[name_i], vectors[name_j]).double()
            gram[i, j] = value
            gram[j, i] = value
    return _as_square_symmetric_gram(gram.detach().cpu())


def _component_gradients(model, output, parameters, device):
    """Get raw F/S/K gradients from one retained forward graph."""
    losses = {
        name: output["components"][key].mean()
        for name, key in COMPONENT_KEYS.items()
    }
    vectors = {}
    device_vectors = {}
    parameter_count = sum(parameter.numel() for parameter in parameters)
    names = list(OBJECTIVE_NAMES)
    for index, name in enumerate(names):
        loss = losses[name]
        if not loss.requires_grad:
            device_vector = torch.zeros(
                parameter_count,
                dtype=torch.float32,
                device=device,
            )
        else:
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=index < len(names) - 1,
                allow_unused=True,
            )
            device_vector = _flatten_gradients_to_device(
                parameters,
                gradients,
                device,
            )
            del gradients
        device_vectors[name] = device_vector
        vectors[name] = device_vector.detach().cpu()
    gram = _gram_from_device_vectors(device_vectors, device)
    del device_vectors
    values = {name: float(losses[name].detach().cpu()) for name in OBJECTIVE_NAMES}
    return values, vectors, gram


def _combine_vectors(
    vectors: Mapping[str, torch.Tensor], coefficients: Sequence[float]
) -> torch.Tensor:
    result = torch.zeros_like(vectors[OBJECTIVE_NAMES[0]])
    for name, coefficient in zip(OBJECTIVE_NAMES, coefficients):
        result.add_(vectors[name], alpha=float(coefficient))
    return result


def _parameter_snapshot(parameters):
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _restore_parameters(parameters, snapshot) -> float:
    maximum_error = 0.0
    with torch.no_grad():
        for parameter, original in zip(parameters, snapshot):
            parameter.copy_(original.to(device=parameter.device, dtype=parameter.dtype))
            maximum_error = max(
                maximum_error,
                float(
                    (parameter.detach() - original.to(device=parameter.device)).abs().max()
                ),
            )
    return maximum_error


def _apply_parameter_vector(parameters, vector: torch.Tensor, eta: float) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            update = vector[offset : offset + count].reshape_as(parameter)
            parameter.add_(update.to(device=parameter.device, dtype=parameter.dtype), alpha=-float(eta))
            offset += count
    if offset != vector.numel():
        raise ValueError("Gradient vector length does not match model parameters")


def _source_digest(source: torch.Tensor) -> str:
    return hashlib.sha256(
        source.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _batch_identifiers(batch, environments: Sequence[str]) -> dict[str, Any]:
    dataset_indices = [int(value) for value in batch["dataset_index"].tolist()]
    env_indices = [int(value) for value in batch["env_index"].tolist()]
    return {
        "dataset_indices": ",".join(str(value) for value in dataset_indices),
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


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def _stats(values: Sequence[float]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str, predicate=lambda value: bool(value)) -> float | None:
    if not rows:
        return None
    return float(sum(predicate(row.get(key)) for row in rows) / len(rows))


def _direction_summary(direction_rows: Sequence[Mapping[str, Any]]):
    grouped = defaultdict(list)
    for row in direction_rows:
        grouped[(str(row["method"]), str(row["step_label"]))].append(row)
    result = {}
    for (method, step_label), values in sorted(grouped.items()):
        result.setdefault(method, {})[step_label] = {
            "count": len(values),
            "all_three_decreased_rate": float(
                np.mean([bool(value["all_three_decreased"]) for value in values])
            ),
            "first_order_all_decreased_rate": float(
                np.mean([bool(value["first_order_all_decreased"]) for value in values])
            ),
            "delta_F": _stats([float(value["delta_F"]) for value in values]),
            "delta_S": _stats([float(value["delta_S"]) for value in values]),
            "delta_K": _stats([float(value["delta_K"]) for value in values]),
            "first_order_delta_F": _stats(
                [float(value["first_order_delta_F"]) for value in values]
            ),
            "first_order_delta_S": _stats(
                [float(value["first_order_delta_S"]) for value in values]
            ),
            "first_order_delta_K": _stats(
                [float(value["first_order_delta_K"]) for value in values]
            ),
        }
    return result


def _summarize(rows, direction_rows, *, fixed_weights, config, checkpoint_metadata):
    pair_summary = {}
    for pair in PAIR_NAMES:
        suffix = PAIR_COLUMNS[pair]
        key = f"cos_{suffix}"
        values = _finite_values(rows, key)
        pair_summary[suffix] = {
            "objectives": list(pair),
            "cosine": _stats(values),
            "conflict_rate_cosine_lt_0": float(
                sum(value < 0.0 for value in values) / len(values)
            )
            if values
            else None,
        }

    alpha_summary = {
        name: _stats(_finite_values(rows, f"alpha_{name}"))
        for name in OBJECTIVE_NAMES
    }
    alpha_entropy = _finite_values(rows, "alpha_entropy_normalized")
    active_counts = Counter(str(row["mgda_active_set"]) for row in rows)
    mgda_ratio_values = _finite_values(rows, "mgda_ratio")
    mgda_norm_values = _finite_values(rows, "mgda_norm")
    summary = {
        "audit": {
            "development_only": True,
            "network_training_performed": False,
            "optimizer_created": False,
            "data_split": "train",
            "validation_read": False,
            "external_30_read": False,
            "final_test_read": False,
            "batch_count": len(rows),
            "context_count": int(sum(int(row["contexts"]) for row in rows)),
            "trajectory_count": int(sum(int(row["trajectories"]) for row in rows)),
            "direction_validation_batch_count": len(
                {int(row["batch_index"]) for row in direction_rows}
            ),
        },
        "checkpoint": checkpoint_metadata,
        "fixed_weights": fixed_weights,
        "config": config,
        "costs": {
            name: _stats(_finite_values(rows, f"J_{name}"))
            for name in OBJECTIVE_NAMES
        },
        "gradient_norms": {
            name: _stats(_finite_values(rows, f"norm_g_{name}"))
            for name in OBJECTIVE_NAMES
        },
        "gram": {
            key: _stats(_finite_values(rows, f"H_{key}"))
            for key in ("FF", "FS", "FK", "SS", "SK", "KK")
        },
        "pairwise_cosines": pair_summary,
        "mgda": {
            "alpha": alpha_summary,
            "alpha_entropy_normalized": _stats(alpha_entropy),
            "active_set_counts": dict(sorted(active_counts.items())),
            "dominant_max_alpha_rate": _rate(
                rows,
                "max_alpha",
                lambda value: value is not None and float(value) >= 0.9,
            ),
            "dominant_F_rate": _rate(
                rows, "alpha_F", lambda value: value is not None and float(value) >= 0.9
            ),
            "dominant_S_rate": _rate(
                rows, "alpha_S", lambda value: value is not None and float(value) >= 0.9
            ),
            "dominant_K_rate": _rate(
                rows, "alpha_K", lambda value: value is not None and float(value) >= 0.9
            ),
            "norm": _stats(mgda_norm_values),
            "ratio": _stats(mgda_ratio_values),
            "common_descent_rate": _rate(rows, "common_descent"),
            "approximate_pareto_stationary_rate": _rate(
                rows, "approximate_pareto_stationary"
            ),
            "dot_gF_gMGDA": _stats(_finite_values(rows, "dot_F_MGDA")),
            "dot_gS_gMGDA": _stats(_finite_values(rows, "dot_S_MGDA")),
            "dot_gK_gMGDA": _stats(_finite_values(rows, "dot_K_MGDA")),
        },
        "fixed_weight_comparison": {
            "norm": _stats(_finite_values(rows, "fixed_norm")),
            "common_descent_rate": _rate(rows, "fixed_common_descent"),
            "worsens_any_cost_first_order_rate": _rate(
                rows, "fixed_worsens_any_first_order"
            ),
            "worsens_F_rate": _rate(rows, "fixed_worsens_F_first_order"),
            "worsens_S_rate": _rate(rows, "fixed_worsens_S_first_order"),
            "worsens_K_rate": _rate(rows, "fixed_worsens_K_first_order"),
            "dot_F_fixed": _stats(_finite_values(rows, "dot_F_fixed")),
            "dot_S_fixed": _stats(_finite_values(rows, "dot_S_fixed")),
            "dot_K_fixed": _stats(_finite_values(rows, "dot_K_fixed")),
        },
        "direction_validation": _direction_summary(direction_rows),
    }
    return summary


def _format_percent(value):
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _format_number(value, digits=4):
    return "n/a" if value is None else f"{float(value):.{digits}g}"


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    pair_items = summary["pairwise_cosines"]
    most_conflicting = min(
        pair_items.items(),
        key=lambda item: float(item[1]["cosine"]["mean"])
        if item[1]["cosine"]["mean"] is not None
        else float("inf"),
    )
    mgda = summary["mgda"]
    fixed = summary["fixed_weight_comparison"]
    direction = summary["direction_validation"]
    n_batches = summary["audit"]["batch_count"]
    n_direction = summary["audit"]["direction_validation_batch_count"]
    checkpoint_metadata = summary["checkpoint"]
    environment_source = checkpoint_metadata.get(
        "audit_environment_source_checkpoint",
        checkpoint_metadata.get("path"),
    )
    model_checkpoint_p_mask = checkpoint_metadata.get("model_checkpoint_p_mask")
    audit_p_mask = summary["config"].get("p_mask")
    alpha_means = ", ".join(
        f"{name}={_format_number(mgda['alpha'][name]['mean'])}"
        for name in OBJECTIVE_NAMES
    )

    lines = [
        "# Direct-Cost MGDA Gradient Geometry Audit",
        "",
        "## Scope",
        "",
        "This is a development-only, no-training audit of the current direct-cost "
        "Stage 2 forward/cost geometry.",
        "",
        f"- Model checkpoint: `{checkpoint_metadata['path']}` ({checkpoint_metadata.get('stage')}).",
        f"- Train environment source: `{environment_source}`.",
        "- Data: `data/dataset1/train` only, using the recorded Stage 2 train environments.",
        f"- Audit mask probability: `p_mask={audit_p_mask}`."
        + (
            f" The Stage 1 checkpoint metadata records its original training `p_mask={model_checkpoint_p_mask}`;"
            " the audit intentionally follows the current Stage 2 mask protocol."
            if model_checkpoint_p_mask is not None
            and audit_p_mask is not None
            and not np.isclose(float(model_checkpoint_p_mask), float(audit_p_mask))
            else ""
        ),
        "- Validation, external 30, and final-test data were not read.",
        f"- Batches: `{n_batches}`; direction-validation batches: `{n_direction}`.",
        "- The generator was kept in evaluation mode. No Adam/optimizer update was performed.",
        "- Raw gradients are gradients of the batch means `J_F=components['forbidden_region']`, `J_S=components['stability']`, and `J_K=components['curvature']` (analytic curvature); fixed weights are used only after those gradients are available for the comparator.",
        "",
        "## Main Findings",
        "",
        "1. **共同下降方向**：MGDA 的三项一阶内积均为正的 batch 比例为 "
        f"`{_format_percent(mgda['common_descent_rate'])}`。"
        + (
            "因此在当前 train-only 审计中多数 batch 存在共同下降方向。"
            if mgda["common_descent_rate"] is not None
            and mgda["common_descent_rate"] > 0.5
            else "因此当前 train-only 审计不能称为多数 batch 都有共同下降方向。"
        ),
        "2. **MGDA 权重退化**：平均权重为 "
        f"`{alpha_means}`；最大权重不小于 0.9 的比例为 "
        f"`{_format_percent(mgda['dominant_max_alpha_rate'])}`。"
        + (
            "这表示权重长期明显退化到单一 cost。"
            if mgda["dominant_max_alpha_rate"] is not None
            and mgda["dominant_max_alpha_rate"] > 0.5
            else "没有观察到多数 batch 的单一 cost 退化。"
        ),
        "3. **最严重冲突**：按平均 pairwise cosine，最严重的是 "
        f"`{most_conflicting[0]}`，均值为 "
        f"`{_format_number(most_conflicting[1]['cosine']['mean'])}`，"
        f"cosine<0 的比例为 `{_format_percent(most_conflicting[1]['conflict_rate_cosine_lt_0'])}`。",
        "4. **Pareto stationary**：`mgda_ratio` 的中位数为 "
        f"`{_format_number(mgda['ratio']['median'])}`，"
        "按相对阈值 `mgda_ratio <= 1e-3` 或绝对阈值 `1e-12` 标记的近零比例为 "
        f"`{_format_percent(mgda['approximate_pareto_stationary_rate'])}`。"
        + (
            "若该比例较高，说明许多 batch 更接近真实的 Pareto tradeoff。"
            if mgda["approximate_pareto_stationary_rate"] is not None
            and mgda["approximate_pareto_stationary_rate"] > 0.1
            else "当前没有看到大量接近零的 MGDA 方向。"
        ),
        "5. **真实 forward 验证**：见下表；它在同一 condition/mask/source 上重新 forward，"
        "不依赖一阶近似。",
        "6. **固定加权对照**：固定方向使至少一项 cost 一阶上变坏的比例为 "
        f"`{_format_percent(fixed['worsens_any_cost_first_order_rate'])}`，"
        f"而 MGDA 的 common-descent 比例为 `{_format_percent(mgda['common_descent_rate'])}`。"
        + (
            "这支持 MGDA 在一阶梯度 tradeoff 上确实解决了固定加权方向的部分冲突。"
            if fixed["worsens_any_cost_first_order_rate"] is not None
            and mgda["common_descent_rate"] is not None
            and fixed["worsens_any_cost_first_order_rate"] > 0.0
            and mgda["common_descent_rate"] > fixed["common_descent_rate"]
            else "该 train-only 结果不足以宣称 MGDA 普遍优于固定加权方向。"
        ),
        "",
        "## Metrics",
        "",
        "### Pairwise Gradient Cosines",
        "",
        "| pair | mean | median | cosine < 0 |",
        "|---|---:|---:|---:|",
    ]
    for suffix in ("FS", "FK", "SK"):
        item = pair_items[suffix]
        lines.append(
            f"| {suffix} | {_format_number(item['cosine']['mean'])} | "
            f"{_format_number(item['cosine']['median'])} | "
            f"{_format_percent(item['conflict_rate_cosine_lt_0'])} |"
        )
    lines.extend(
        [
            "",
            "### MGDA and Fixed-Weight Summary",
            "",
            f"- Fixed weights `(F,S,K)`: `{summary['fixed_weights']}`.",
            f"- MGDA active sets: `{mgda['active_set_counts']}`.",
            f"- Normalized alpha entropy mean: `{_format_number(mgda['alpha_entropy_normalized']['mean'])}` (1 is uniform).",
            f"- MGDA norm median: `{_format_number(mgda['norm']['median'])}`; ratio median: `{_format_number(mgda['ratio']['median'])}`.",
            f"- Fixed direction common-descent rate: `{_format_percent(fixed['common_descent_rate'])}`.",
            f"- Fixed direction first-order worsening rates: F `{_format_percent(fixed['worsens_F_rate'])}`, S `{_format_percent(fixed['worsens_S_rate'])}`, K `{_format_percent(fixed['worsens_K_rate'])}`.",
            "",
            "### Same-Batch Forward Direction Check",
            "",
            "`all_three_decreased_rate` means all three raw costs decreased strictly after the temporary parameter update.",
            "",
            "| method | step | all three decreased | mean delta F | mean delta S | mean delta K |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(direction):
        for step_label in (label for label, _ in STEP_SCALES):
            if step_label not in direction[method]:
                continue
            item = direction[method][step_label]
            lines.append(
                f"| {method} | {step_label} | {_format_percent(item['all_three_decreased_rate'])} | "
                f"{_format_number(item['delta_F']['mean'])} | "
                f"{_format_number(item['delta_S']['mean'])} | "
                f"{_format_number(item['delta_K']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- These are batch-level local gradient geometry results on the development train split, not validation/OOD/final-test evidence.",
            "- A positive MGDA inner product is a first-order statement. The direction table checks the corresponding finite perturbation by re-running the same forward pass.",
            "- `approximate_pareto_stationary` is a numerical diagnostic based on the reported relative/absolute thresholds; it is not a proof of global Pareto stationarity.",
            "- Parameter restoration was checked after each direction-validation batch; the audit does not produce a trained checkpoint.",
            "",
            "## Artifacts",
            "",
            "- `summary.json` contains the frozen protocol, aggregate statistics, and direction-validation summaries.",
            "- `per_batch.csv` contains raw per-batch cost, Gram, cosine, norm, alpha, dot-product, and fixed-weight comparison fields.",
            "- `direction_validation.csv` contains the long-form same-batch finite-forward checks.",
            "- `gradient_cosines.png`, `mgda_alphas.png`, `gradient_norms.png`, and `direction_validation.png` visualize the audit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_plots(output_dir: Path, rows, direction_rows) -> list[str]:
    if not rows:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - environment-specific fallback
        (output_dir / "plot_error.txt").write_text(str(error) + "\n", encoding="utf-8")
        return []

    x = np.arange(len(rows))
    created = []

    figure, axis = plt.subplots(figsize=(10, 4.5))
    for suffix, color in (("FS", "tab:blue"), ("FK", "tab:orange"), ("SK", "tab:green")):
        axis.plot(
            x,
            [float(row.get(f"cos_{suffix}", float("nan"))) for row in rows],
            label=suffix,
            linewidth=1.0,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("batch")
    axis.set_ylabel("gradient cosine")
    axis.set_title("Raw privileged-cost gradient cosines")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_dir / "gradient_cosines.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    created.append(path.name)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(rows))
    for name, color in (("F", "tab:red"), ("S", "tab:blue"), ("K", "tab:green")):
        values = np.asarray([float(row[f"alpha_{name}"]) for row in rows])
        axis.fill_between(x, bottom, bottom + values, step="mid", alpha=0.75, label=f"alpha_{name}", color=color)
        bottom += values
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("batch")
    axis.set_ylabel("MGDA alpha")
    axis.set_title("Exact active-set MGDA weights")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_dir / "mgda_alphas.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    created.append(path.name)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    for key, label, color in (
        ("norm_g_F", "||g_F||", "tab:red"),
        ("norm_g_S", "||g_S||", "tab:blue"),
        ("norm_g_K", "||g_K||", "tab:green"),
        ("mgda_norm", "||g_MGDA||", "black"),
    ):
        axis.semilogy(
            x,
            np.maximum(np.asarray([float(row[key]) for row in rows]), 1e-30),
            label=label,
            linewidth=1.0,
            color=color,
        )
    axis.set_xlabel("batch")
    axis.set_ylabel("gradient norm")
    axis.set_title("Raw and MGDA gradient norms")
    axis.legend()
    axis.grid(alpha=0.25, which="both")
    figure.tight_layout()
    path = output_dir / "gradient_norms.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    created.append(path.name)

    if direction_rows:
        grouped = defaultdict(list)
        for row in direction_rows:
            grouped[(row["method"], row["step_label"])].append(row)
        labels = []
        heights = []
        colors = []
        for method in ("mgda", "fixed"):
            for step_label, _ in STEP_SCALES:
                values = grouped.get((method, step_label), [])
                if not values:
                    continue
                labels.append(f"{method}\n{step_label}")
                heights.append(np.mean([bool(value["all_three_decreased"]) for value in values]))
                colors.append("tab:purple" if method == "mgda" else "tab:gray")
        figure, axis = plt.subplots(figsize=(11, 4.5))
        axis.bar(np.arange(len(labels)), heights, color=colors)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("all three decreased rate")
        axis.set_title("Same-batch finite-forward direction validation")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / "direction_validation.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path.name)
    return created


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--environment-checkpoint",
        type=Path,
        default=None,
        help=(
            "For a Stage-1 audit, current direct-cost Stage-2 checkpoint "
            "providing the identical development train environment split."
        ),
    )
    parser.add_argument("--data-folder", type=Path, default=Path("data/dataset1"))
    # Keeping this as a single choice is an intentional guard against opening
    # validation or sealed data from this development-only audit.
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sources-per-context", type=int, default=4)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--mask-variant", type=int, default=1)
    parser.add_argument("--p-mask", type=float, default=1.0)
    parser.add_argument("--vehicle-radius-meters", type=float, default=SAFETY_COST_CONFIG.vehicle_radius_meters)
    parser.add_argument("--source-seed", type=int, default=20260807)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-batches", type=int, default=None, help="Smoke only: stop after this many train batches.")
    parser.add_argument("--environment-limit", type=int, default=None, help="Smoke only: use the first N selected train environments.")
    parser.add_argument("--all-train-environments", action="store_true", help="Use every environment in data/dataset1/train instead of the checkpoint's Stage-2 train list.")
    parser.add_argument("--direction-batches", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1e-5)
    parser.add_argument("--stationary-ratio-threshold", type=float, default=1e-3)
    parser.add_argument("--stationary-absolute-threshold", type=float, default=1e-12)
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but CUDA is unavailable")
    return device


def _checkpoint_metadata(checkpoint_path: Path, checkpoint: Mapping[str, Any], model_args):
    environment_split = checkpoint.get("stage2_environment_split") or {}
    return {
        "path": str(checkpoint_path),
        "stage": checkpoint.get("stage"),
        "stage2_epoch": checkpoint.get("stage2_epoch"),
        "stage2_global_update": checkpoint.get("stage2_global_update"),
        "stage2_objective_semantics": checkpoint.get("stage2_objective_semantics"),
        "stage2_context_semantics": checkpoint.get("stage2_context_semantics"),
        "stage2_training_protocol_semantics": checkpoint.get("stage2_training_protocol_semantics"),
        "privileged_cost_semantics": checkpoint.get("privileged_cost_semantics"),
        "representation_semantic_version": checkpoint.get("representation_semantic_version"),
        "input_mask_semantics": checkpoint.get("input_mask_semantics"),
        "demo_target_semantics": checkpoint.get("demo_target_semantics"),
        "stage2_train_environments": list(environment_split.get("train", [])),
        "model_args": dict(model_args),
    }


def _select_train_environments(checkpoint, args):
    if args.all_train_environments:
        from map_config import discover_environments

        environments = discover_environments(
            args.data_folder / "train",
            expected_count=MAP_CONFIG.expected_environments,
        )
    else:
        split = checkpoint.get("stage2_environment_split") or {}
        environments = sorted(str(value) for value in split.get("train", []))
        if not environments:
            raise ValueError(
                "Checkpoint has no Stage-2 train environment list; use a current direct-cost checkpoint"
            )
    if args.environment_limit is not None:
        if args.environment_limit <= 0:
            raise ValueError("--environment-limit must be positive")
        environments = environments[: int(args.environment_limit)]
    if not environments:
        raise ValueError("No development train environments selected")
    if any("val" in str(value).lower() or "test" in str(value).lower() for value in environments):
        raise ValueError("Selected environment names violate the train-only audit guard")
    return environments


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.split != "train":
        raise ValueError("This development-only audit accepts only --split train")
    if args.batch_size <= 0 or args.sources_per_context <= 0:
        raise ValueError("batch size and sources per context must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    if args.direction_batches < 0:
        raise ValueError("--direction-batches must be non-negative")
    if args.eta <= 0.0:
        raise ValueError("--eta must be positive")
    if not 0.0 < args.p_mask <= 1.0:
        raise ValueError("--p-mask must be in (0,1]")
    if args.stationary_ratio_threshold <= 0.0 or args.stationary_absolute_threshold <= 0.0:
        raise ValueError("stationary thresholds must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _resolve_device(args.device)

    # Load the large checkpoint on CPU first so its optimizer/state archive is
    # never mapped onto the 6 GB audit GPU.
    model, model_args, checkpoint = load_model(args.checkpoint, torch.device("cpu"))
    _require_current_mask_semantics(checkpoint)
    _require_demo_target_semantics(checkpoint)
    _require_main_method_model(model)
    checkpoint_stage = checkpoint.get("stage")
    if checkpoint_stage not in {"stage1", "stage2_direct_privileged_cost"}:
        raise ValueError(
            "The audit checkpoint must be the current Stage-1 checkpoint or "
            "current direct-cost Stage-2 checkpoint"
        )

    # A Stage-1 checkpoint has no Stage-2 environment split.  In that case the
    # split is read explicitly from the matching direct-cost Stage-2 checkpoint
    # so the Stage-1 and Stage-2 audits consume identical train batches.
    environment_checkpoint = checkpoint
    environment_checkpoint_path = args.checkpoint
    if checkpoint_stage == "stage1":
        if args.environment_checkpoint is None:
            raise ValueError(
                "A Stage-1 audit requires --environment-checkpoint pointing to "
                "the matching current direct-cost Stage-2 checkpoint"
            )
        environment_checkpoint_path = Path(args.environment_checkpoint)
        if environment_checkpoint_path.resolve() == Path(args.checkpoint).resolve():
            raise ValueError(
                "--environment-checkpoint must be a Stage-2 checkpoint when "
                "--checkpoint is Stage-1"
            )
        environment_checkpoint = torch.load(
            environment_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    _require_current_mask_semantics(environment_checkpoint)
    _require_demo_target_semantics(environment_checkpoint)
    if environment_checkpoint.get("stage") != "stage2_direct_privileged_cost":
        raise ValueError(
            "The environment checkpoint must come from current direct-cost Stage 2"
        )
    if environment_checkpoint.get("stage2_objective_semantics") != DIRECT_COST_STAGE2_SEMANTICS:
        raise ValueError(
            "Environment checkpoint direct-cost objective semantics do not match "
            "current implementation: "
            f"{environment_checkpoint.get('stage2_objective_semantics')!r}"
        )
    if environment_checkpoint.get("stage2_context_semantics") != DIRECT_COST_CONTEXT_SEMANTICS:
        raise ValueError(
            "Environment checkpoint Stage-2 context semantics do not match "
            "current implementation"
        )
    if environment_checkpoint.get("privileged_cost_semantics") != PRIVILEGED_COST_SEMANTICS:
        raise ValueError(
            "Environment checkpoint privileged-cost semantics do not match "
            "current implementation: "
            f"{environment_checkpoint.get('privileged_cost_semantics')!r}"
        )
    if checkpoint.get("mask_seed") is not None and int(checkpoint["mask_seed"]) != int(args.mask_seed):
        raise ValueError("--mask-seed differs from the checkpoint's direct-cost mask seed")
    if checkpoint_stage == "stage2_direct_privileged_cost" and checkpoint.get("p_mask") is not None and not np.isclose(float(checkpoint["p_mask"]), args.p_mask):
        raise ValueError("--p-mask differs from the checkpoint's direct-cost mask probability")
    if checkpoint.get("vehicle_radius_meters") is not None and not np.isclose(float(checkpoint["vehicle_radius_meters"]), args.vehicle_radius_meters):
        raise ValueError("--vehicle-radius-meters differs from the checkpoint configuration")
    if environment_checkpoint.get("mask_seed") is not None and int(environment_checkpoint["mask_seed"]) != int(args.mask_seed):
        raise ValueError("--mask-seed differs from the environment checkpoint's mask seed")
    if environment_checkpoint.get("p_mask") is not None and not np.isclose(float(environment_checkpoint["p_mask"]), args.p_mask):
        raise ValueError("--p-mask differs from the environment checkpoint's direct-cost mask probability")
    if environment_checkpoint.get("vehicle_radius_meters") is not None and not np.isclose(float(environment_checkpoint["vehicle_radius_meters"]), args.vehicle_radius_meters):
        raise ValueError("--vehicle-radius-meters differs from the environment checkpoint configuration")

    metadata = _checkpoint_metadata(args.checkpoint, checkpoint, model_args)
    environments = _select_train_environments(environment_checkpoint, args)
    metadata["audit_environment_source_checkpoint"] = str(environment_checkpoint_path)
    metadata["audit_environment_source_stage"] = environment_checkpoint.get("stage")
    metadata["audit_train_environments"] = list(environments)
    metadata["audit_p_mask"] = float(args.p_mask)
    metadata["model_checkpoint_p_mask"] = checkpoint.get("p_mask")
    if environment_checkpoint is not checkpoint:
        del environment_checkpoint
    del checkpoint
    gc.collect()
    model = model.to(device)
    original_training = model.training
    original_checkpoint_mode = getattr(model, "use_gradient_checkpoint", None)
    model.eval()
    if hasattr(model, "use_gradient_checkpoint"):
        model.use_gradient_checkpoint = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters for shared-generator gradient audit")

    base_dataset, discovered_environments = make_partial_dataset(
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
    dataset = FixedMaskVariantDataset(base_dataset, args.mask_variant)
    if args.max_batches is not None:
        max_items = min(len(dataset), int(args.max_batches) * int(args.batch_size))
        dataset = Subset(dataset, range(max_items))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    source_generator = torch.Generator(device=device).manual_seed(int(args.source_seed))
    fixed_weights = {
        "F": float(SAFETY_COST_CONFIG.forbidden_weight),
        "S": float(SAFETY_COST_CONFIG.obstacle_weight),
        "K": float(SAFETY_COST_CONFIG.curvature_weight),
    }
    rows = []
    direction_rows = []
    restore_errors = []
    stationary_ratio_threshold = float(args.stationary_ratio_threshold)
    stationary_absolute_threshold = float(args.stationary_absolute_threshold)

    try:
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= int(args.max_batches):
                break
            count = int(batch["map"].shape[0]) * int(args.sources_per_context)
            source = torch.randn(
                count,
                model.num_edges,
                2,
                dtype=torch.float32,
                device=device,
                generator=source_generator,
            )
            output = _direct_cost_forward(
                model,
                batch,
                device,
                sources_per_context=args.sources_per_context,
                source=source,
            )
            values, raw_gradients, gram = _component_gradients(
                model,
                output,
                parameters,
                device,
            )
            solution = solve_mgda_active_set(gram)
            alpha = solution["alpha"]
            alpha_float = alpha.tolist()
            fixed_weight_tensor = torch.tensor(
                [fixed_weights[name] for name in OBJECTIVE_NAMES],
                dtype=torch.float64,
            )
            raw_norms = {
                name: math.sqrt(max(float(gram[index, index]), 0.0))
                for index, name in enumerate(OBJECTIVE_NAMES)
            }
            mgda_dot_tensor = gram @ alpha
            fixed_dot_tensor = gram @ fixed_weight_tensor
            mgda_objective = max(float(solution["objective"]), 0.0)
            fixed_objective = max(
                float(fixed_weight_tensor @ gram @ fixed_weight_tensor),
                0.0,
            )
            mgda_norm = math.sqrt(mgda_objective)
            fixed_norm = math.sqrt(fixed_objective)
            denominator = float(np.mean([raw_norms[name] for name in OBJECTIVE_NAMES]))
            mgda_ratio = mgda_norm / denominator if denominator > 0.0 else float("nan")
            cosines = {}
            for pair in PAIR_NAMES:
                suffix = PAIR_COLUMNS[pair]
                first_index = OBJECTIVE_NAMES.index(pair[0])
                second_index = OBJECTIVE_NAMES.index(pair[1])
                denominator_pair = raw_norms[pair[0]] * raw_norms[pair[1]]
                cosines[suffix] = (
                    float(gram[first_index, second_index]) / denominator_pair
                    if denominator_pair > 0.0
                    else float("nan")
                )
            mgda_dots = {
                name: float(mgda_dot_tensor[index])
                for index, name in enumerate(OBJECTIVE_NAMES)
            }
            fixed_dots = {
                name: float(fixed_dot_tensor[index])
                for index, name in enumerate(OBJECTIVE_NAMES)
            }
            common_descent = all(mgda_dots[name] > 0.0 for name in OBJECTIVE_NAMES)
            fixed_common_descent = all(fixed_dots[name] > 0.0 for name in OBJECTIVE_NAMES)
            fixed_worsens = {
                name: fixed_dots[name] < 0.0 for name in OBJECTIVE_NAMES
            }
            entropy = -sum(
                float(coefficient) * math.log(max(float(coefficient), 1e-30))
                for coefficient in alpha.tolist()
                if float(coefficient) > 0.0
            ) / math.log(3.0)
            approximate_stationary = (
                mgda_norm <= stationary_absolute_threshold
                or (
                    math.isfinite(mgda_ratio)
                    and mgda_ratio <= stationary_ratio_threshold
                )
            )
            row = {
                "batch_index": batch_index,
                "contexts": int(batch["map"].shape[0]),
                "trajectories": count,
                **_batch_identifiers(batch, discovered_environments),
                "source_sha256": _source_digest(source),
                "J_F": values["F"],
                "J_S": values["S"],
                "J_K": values["K"],
                "norm_g_F": raw_norms["F"],
                "norm_g_S": raw_norms["S"],
                "norm_g_K": raw_norms["K"],
                "H_FF": float(gram[0, 0]),
                "H_FS": float(gram[0, 1]),
                "H_FK": float(gram[0, 2]),
                "H_SS": float(gram[1, 1]),
                "H_SK": float(gram[1, 2]),
                "H_KK": float(gram[2, 2]),
                "cos_FS": cosines["FS"],
                "cos_FK": cosines["FK"],
                "cos_SK": cosines["SK"],
                "alpha_F": float(alpha[0]),
                "alpha_S": float(alpha[1]),
                "alpha_K": float(alpha[2]),
                "alpha_entropy_normalized": entropy,
                "max_alpha": float(alpha.max()),
                "mgda_active_set": solution["label"],
                "mgda_candidate_count": int(solution["candidate_count"]),
                "mgda_objective": float(solution["objective"]),
                "mgda_norm": mgda_norm,
                "mgda_ratio": mgda_ratio,
                "dot_F_MGDA": mgda_dots["F"],
                "dot_S_MGDA": mgda_dots["S"],
                "dot_K_MGDA": mgda_dots["K"],
                "common_descent": common_descent,
                "approximate_pareto_stationary": approximate_stationary,
                "fixed_norm": fixed_norm,
                "dot_F_fixed": fixed_dots["F"],
                "dot_S_fixed": fixed_dots["S"],
                "dot_K_fixed": fixed_dots["K"],
                "fixed_common_descent": fixed_common_descent,
                "fixed_worsens_F_first_order": fixed_worsens["F"],
                "fixed_worsens_S_first_order": fixed_worsens["S"],
                "fixed_worsens_K_first_order": fixed_worsens["K"],
                "fixed_worsens_any_first_order": any(fixed_worsens.values()),
            }

            mgda_gradient = None
            fixed_gradient = None
            if batch_index < int(args.direction_batches):
                mgda_gradient = _combine_vectors(raw_gradients, alpha_float)
                fixed_gradient = _combine_vectors(
                    raw_gradients,
                    [fixed_weights[name] for name in OBJECTIVE_NAMES],
                )
                snapshot = _parameter_snapshot(parameters)
                initial_values = dict(values)
                try:
                    for method, direction in (
                        ("mgda", mgda_gradient),
                        ("fixed", fixed_gradient),
                    ):
                        for step_label, scale in STEP_SCALES:
                            _restore_parameters(parameters, snapshot)
                            step_eta = float(args.eta) * float(scale)
                            _apply_parameter_vector(parameters, direction, step_eta)
                            with torch.no_grad():
                                perturbed = _direct_cost_forward(
                                    model,
                                    batch,
                                    device,
                                    sources_per_context=args.sources_per_context,
                                    source=source,
                                )
                                after_values = {
                                    name: float(
                                        perturbed["components"][COMPONENT_KEYS[name]]
                                        .mean()
                                        .cpu()
                                    )
                                    for name in OBJECTIVE_NAMES
                                }
                            del perturbed
                            deltas = {
                                name: after_values[name] - initial_values[name]
                                for name in OBJECTIVE_NAMES
                            }
                            first_order_deltas = {
                                name: -step_eta * (
                                    mgda_dots[name]
                                    if method == "mgda"
                                    else fixed_dots[name]
                                )
                                for name in OBJECTIVE_NAMES
                            }
                            validation_row = {
                                "batch_index": batch_index,
                                "method": method,
                                "step_label": step_label,
                                "eta": step_eta,
                                "J_F_before": initial_values["F"],
                                "J_S_before": initial_values["S"],
                                "J_K_before": initial_values["K"],
                                "J_F_after": after_values["F"],
                                "J_S_after": after_values["S"],
                                "J_K_after": after_values["K"],
                                "delta_F": deltas["F"],
                                "delta_S": deltas["S"],
                                "delta_K": deltas["K"],
                                "first_order_delta_F": first_order_deltas["F"],
                                "first_order_delta_S": first_order_deltas["S"],
                                "first_order_delta_K": first_order_deltas["K"],
                                "all_three_decreased": all(
                                    deltas[name] < 0.0 for name in OBJECTIVE_NAMES
                                ),
                                "first_order_all_decreased": all(
                                    first_order_deltas[name] < 0.0
                                    for name in OBJECTIVE_NAMES
                                ),
                            }
                            direction_rows.append(validation_row)
                            prefix = f"{method}_{step_label}"
                            row[f"{prefix}_all_three_decreased"] = validation_row[
                                "all_three_decreased"
                            ]
                            for name in OBJECTIVE_NAMES:
                                row[f"{prefix}_delta_{name}"] = deltas[name]
                            row[f"{prefix}_restore_error"] = _restore_parameters(
                                parameters, snapshot
                            )
                    restore_errors.append(
                        max(_restore_parameters(parameters, snapshot), 0.0)
                    )
                finally:
                    restore_errors.append(
                        max(_restore_parameters(parameters, snapshot), 0.0)
                    )
                    del snapshot
            rows.append(row)
            del output, raw_gradients, gram, solution, mgda_gradient, fixed_gradient
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if hasattr(model, "use_gradient_checkpoint") and original_checkpoint_mode is not None:
            model.use_gradient_checkpoint = original_checkpoint_mode
        model.train(original_training)
        if any(parameter.grad is not None for parameter in parameters):
            raise AssertionError("Audit unexpectedly left parameter .grad values populated")

    config = {
        "checkpoint": str(args.checkpoint),
        "data_folder": str(args.data_folder),
        "split": "train",
        "selected_train_environments": environments,
        "batch_size": int(args.batch_size),
        "sources_per_context": int(args.sources_per_context),
        "mask_seed": int(args.mask_seed),
        "mask_variant": int(args.mask_variant),
        "p_mask": float(args.p_mask),
        "vehicle_radius_meters": float(args.vehicle_radius_meters),
        "source_seed": int(args.source_seed),
        "eta": float(args.eta),
        "direction_batches": int(args.direction_batches),
        "stationary_ratio_threshold": stationary_ratio_threshold,
        "stationary_absolute_threshold": stationary_absolute_threshold,
        "fixed_weights": fixed_weights,
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_tensor_count": len(parameters),
        "parameter_restore_max_abs_error": max(restore_errors) if restore_errors else 0.0,
    }
    summary = _summarize(
        rows,
        direction_rows,
        fixed_weights=fixed_weights,
        config=config,
        checkpoint_metadata=metadata,
    )
    summary["artifacts"] = {
        "report": str(output_dir / "REPORT.md"),
        "summary": str(output_dir / "summary.json"),
        "per_batch": str(output_dir / "per_batch.csv"),
        "direction_validation": str(output_dir / "direction_validation.csv"),
    }
    _write_csv(output_dir / "per_batch.csv", rows)
    _write_csv(output_dir / "direction_validation.csv", direction_rows)
    _write_json(output_dir / "summary.json", summary)
    plots = _make_plots(output_dir, rows, direction_rows)
    summary["artifacts"]["plots"] = [str(output_dir / name) for name in plots]
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "REPORT.md", summary)
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_audit(args)
    print(json.dumps(_jsonable(summary["audit"]), indent=2, ensure_ascii=False))
    print(f"REPORT.md: {Path(args.output_dir) / 'REPORT.md'}")


if __name__ == "__main__":
    main()
