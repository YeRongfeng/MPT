"""Pure inference audit for the direct spatial-map Stage-1 checkpoint.

The audit intentionally does not import a training loop or mutate model
parameters.  It replays the deployed t=1,r=0 forward pass and uses temporary
forward code for CA interventions so the checkpoint and production model code
remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dit.Models import SpatialMapPathMeanFlowTransformer
from grad_optimizer import build_signed_mask_distance_map, trajectory_validity_metrics
from map_config import MAP_CONFIG
from posterior_pipeline import (
    make_partial_dataset,
    normalize_poses,
    path_coordinates_from_batch,
)
from train_compact_stage1 import build_model


DEFAULT_RUN_DIR = Path("diagnostics/spatial_map_resolution/single_12")
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "causal_audit"
LAYERS = 6
NUM_MAP_TOKENS = 144
NUM_EDGES = 22


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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


def _write_json(path: Path, value: Any):
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def _distribution(values):
    values = torch.as_tensor(values, dtype=torch.float32).flatten().cpu()
    if values.numel() == 0:
        return {"count": 0}
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


def _repeat_context_tensor(value, repeats):
    return value.repeat_interleave(int(repeats), dim=0)


def _validity_for_states(model, states, batch, device, repeats=1):
    """Run the repository hard-validity operator on decoded states."""
    states = model.project_zero_sum(states.float().to(device))
    physical_start = batch["start_pose"].float().to(device)
    physical_goal = batch["goal_pose"].float().to(device)
    cost_map = batch["cost_map"].float().to(device)
    mask = batch["mask"].float().to(device)
    start, goal = normalize_poses(
        physical_start,
        physical_goal,
        model.coordinate_scale,
        device,
    )
    if int(repeats) > 1:
        start = _repeat_context_tensor(start, repeats)
        goal = _repeat_context_tensor(goal, repeats)
        physical_start = _repeat_context_tensor(physical_start, repeats)
        physical_goal = _repeat_context_tensor(physical_goal, repeats)
        cost_map = _repeat_context_tensor(cost_map, repeats)
        mask = _repeat_context_tensor(mask, repeats)
    geometry = model.evaluate_trajectory_state(states, start, goal)
    curvature_audit = model.audit_trajectory_state_curvature(states, start, goal)
    signed_mask = build_signed_mask_distance_map(
        mask,
        MAP_CONFIG.cost_map_info(),
        device=device,
    )
    condition_ids = torch.arange(
        int(batch["map"].shape[0]), device=device, dtype=torch.long
    ).repeat_interleave(int(repeats))
    validity = trajectory_validity_metrics(
        geometry["position"],
        cost_map,
        MAP_CONFIG.cost_map_info(),
        analytic_yaw=geometry["yaw"],
        analytic_curvature=geometry["curvature"],
        analytic_curvature_audit=curvature_audit,
        mask=mask,
        signed_mask_distance_map=signed_mask,
        start_pose=physical_start,
        goal_pose=physical_goal,
        condition_ids=condition_ids,
    )
    return geometry, validity


def _prepare_state_inputs(model, map_input, noisy_path, timestep, timestep_r,
                          start_pose, goal_pose, token_permutation=None,
                          map_content_permutation=None):
    if map_content_permutation is None:
        map_tokens = model._build_map_memory(map_input)
    else:
        # Reorder CNN spatial content before adding the fixed 2-D PE.  This
        # tests spatial alignment; reordering already position-encoded tokens
        # would be a permutation-invariant K/V operation.
        levels = model._encode_map_levels(map_input)
        memories = []
        for _, features in levels.items():
            tokens = model.reorder_dims(features)
            tokens = tokens[:, map_content_permutation, :]
            tokens = model.map_position_enc(tokens, conv_shape=features.shape[-2:])
            memories.append(tokens)
        map_tokens = torch.cat(memories, dim=1)
    if token_permutation is not None:
        map_tokens = map_tokens[:, token_permutation, :]
    path_tokens = model.path_patchify(noisy_path)
    x = model.layer_norm(path_tokens + model.path_pos_embed)
    x = model.dropout(x)
    t_emb = model.time_embedder(timestep)
    h_emb = model.time_embedder(torch.clamp(timestep - timestep_r, 0.0, 1.0))
    s_emb = model.pose_embedder(start_pose)
    g_emb = model.pose_embedder(goal_pose)
    condition = model.cond_mlp(torch.cat([t_emb, h_emb, s_emb, g_emb], dim=-1))
    return x, map_tokens, condition


def _attention_forward(attention, query, key, value, token_mask=None):
    batch, query_length, width = query.shape

    def split_heads(tensor):
        return tensor.view(batch, -1, attention.n_heads, attention.head_dim).transpose(1, 2)

    q = split_heads(attention.q_proj(query))
    k = split_heads(attention.k_proj(key))
    v = split_heads(attention.v_proj(value))
    logits = torch.matmul(q, k.transpose(-2, -1)) * attention.scale
    if token_mask is not None:
        logits = logits.masked_fill(token_mask[:, None, None, :], float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    output = torch.matmul(weights, v).transpose(1, 2).contiguous()
    output = attention.out_proj(output.view(batch, query_length, width))
    return output, weights


def audit_forward(
    model,
    map_input,
    noisy_path,
    timestep,
    timestep_r,
    start_pose,
    goal_pose,
    *,
    disable_ca_all=False,
    disable_ca_layer=None,
    token_permutation=None,
    map_content_permutation=None,
    token_mask=None,
    collect_attention=False,
):
    """Exact eval-mode block replay with temporary CA interventions."""
    x, map_tokens, condition = _prepare_state_inputs(
        model,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        token_permutation=token_permutation,
        map_content_permutation=map_content_permutation,
    )
    layer_records = []
    for layer_index, block in enumerate(model.dit_blocks):
        (
            shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_ffn, scale_ffn, gate_ffn,
        ) = block.adaLN_modulation(condition).chunk(9, dim=-1)
        normalized = block.norm1(x)
        normalized = normalized * (1.0 + scale_sa[:, None]) + shift_sa[:, None]
        x = x + gate_sa[:, None] * block.self_attn(normalized, normalized, normalized)

        normalized = block.norm2(x)
        normalized = normalized * (1.0 + scale_ca[:, None]) + shift_ca[:, None]
        if disable_ca_all or disable_ca_layer == layer_index:
            ca_output = torch.zeros_like(normalized)
            weights = None
        else:
            ca_output, weights = _attention_forward(
                block.map_cross_attn,
                normalized,
                map_tokens,
                map_tokens,
                token_mask=token_mask,
            )
        ca_delta = gate_ca[:, None] * ca_output
        # Keep both denominators explicit: the requested ratio is relative to
        # the main path state immediately before CA, while the AdaLN-input
        # ratio is retained as a secondary diagnostic for comparability.
        ca_state_norm = torch.linalg.vector_norm(x, dim=-1)
        ca_input_norm = torch.linalg.vector_norm(normalized, dim=-1)
        ca_delta_norm = torch.linalg.vector_norm(ca_delta, dim=-1)
        state_ratio = ca_delta_norm.sum(dim=1) / ca_state_norm.sum(dim=1).clamp_min(1e-12)
        input_ratio = ca_delta_norm.sum(dim=1) / ca_input_norm.sum(dim=1).clamp_min(1e-12)
        x = x + ca_delta

        normalized = block.norm3(x)
        normalized = normalized * (1.0 + scale_ffn[:, None]) + shift_ffn[:, None]
        x = x + gate_ffn[:, None] * block.ffn(normalized)
        record = {
            "ca_ratio_state": state_ratio.detach(),
            "ca_ratio_input": input_ratio.detach(),
        }
        if collect_attention and weights is not None:
            record["attention"] = weights.detach()
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1)
            record["attention_entropy"] = entropy.detach()
        layer_records.append(record)
    x = model.layer_norm(x)
    state = model.project_zero_sum(model.main_pred(x))
    return state, layer_records


def _metrics_summary(validity, repeats, contexts):
    result = {}
    for key in ("strict_valid", "forbidden_region_ok", "stability_ok", "curvature_ok"):
        result[key] = float(validity[key].float().mean())
    strict = validity["strict_valid"].reshape(contexts, repeats)
    result["safe_at_1"] = float(strict[:, 0].float().mean())
    result["safe_at_k"] = float(strict.any(dim=1).float().mean())
    return result


def _metric_delta_summary(model, states, batch, baseline_states, device, repeats):
    geometry, validity = _validity_for_states(model, states, batch, device, repeats)
    base_geometry, _ = _validity_for_states(model, baseline_states, batch, device, repeats)
    position_delta = geometry["position"] - base_geometry["position"]
    rms = torch.sqrt(position_delta.square().mean(dim=(1, 2)))
    mean_disp = torch.linalg.vector_norm(position_delta, dim=-1).mean(dim=1)
    return {
        **_metrics_summary(validity, repeats, int(batch["map"].shape[0])),
        "state_rms": float(torch.sqrt((states - baseline_states).square().mean())),
        "trajectory_rms_m": float(rms.mean()),
        "trajectory_mean_displacement_m": float(mean_disp.mean()),
        "trajectory_rms_m_p90": float(torch.quantile(rms, 0.9)),
        "trajectory_mean_displacement_m_p90": float(torch.quantile(mean_disp, 0.9)),
    }


def _slice_batch_rows(batch, row_mask):
    """Slice only batch-leading tensors for a high/low context subgroup."""
    batch_size = int(batch["map"].shape[0])
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            result[key] = value[row_mask]
        else:
            result[key] = value
    return result


def _failure_error_summary(validity, rms, mean_disp):
    result = {}
    categories = {
        "safe": validity["strict_valid"],
        "curvature_fail": ~validity["curvature_ok"],
        "stability_fail": ~validity["stability_ok"],
        "forbidden_fail": ~validity["forbidden_region_ok"],
    }
    for name, selector in categories.items():
        selector = selector.detach().bool()
        result[name] = {
            "count": int(selector.sum()),
            "trajectory_rms_m": float(rms[selector].mean()) if selector.any() else None,
            "trajectory_mean_displacement_m": float(mean_disp[selector].mean()) if selector.any() else None,
            "trajectory_rms_m_p90": float(torch.quantile(rms[selector], 0.9)) if selector.any() else None,
        }
    return result


def _load_validation(args, summary):
    split = summary["environment_split"]
    dataset, _ = make_partial_dataset(
        args.data_folder,
        "val",
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=0.5,
        mask_mode="stage1_demo_valid",
        environment_names=split["validation"],
        dynamic_mask_noise=False,
    )
    if len(dataset) < args.contexts:
        raise ValueError(f"validation dataset has {len(dataset)} contexts, requested {args.contexts}")
    indices = np.linspace(0, len(dataset) - 1, args.contexts, dtype=int).tolist()
    return dataset, indices


def _safe_basin_audit(model, dataset, indices, args, device):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    source_generator = torch.Generator(device=device).manual_seed(args.source_seed)
    source_bank = torch.empty(
        (len(indices), args.sources, model.num_edges, 2), dtype=torch.float32
    )
    validity_arrays = {
        key: torch.zeros((len(indices), args.sources), dtype=torch.bool)
        for key in ("strict_valid", "forbidden_region_ok", "stability_ok", "curvature_ok")
    }
    error_arrays = {
        "trajectory_rms_m": torch.zeros((len(indices), args.sources)),
        "trajectory_mean_displacement_m": torch.zeros((len(indices), args.sources)),
    }
    expert_rows = []
    cursor = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            contexts = int(batch["map"].shape[0])
            maps = batch["map"].float().to(device)
            physical_start = batch["start_pose"].float().to(device)
            physical_goal = batch["goal_pose"].float().to(device)
            start, goal = normalize_poses(
                physical_start, physical_goal, model.coordinate_scale, device
            )
            target = path_coordinates_from_batch(batch, device)
            target_geometry, expert_validity = _validity_for_states(
                model, target, batch, device, repeats=1
            )
            for row in range(contexts):
                expert_rows.append({
                    "dataset_index": int(batch["dataset_index"][row]),
                    "env_index": int(batch["env_index"][row]),
                    "path_index": int(batch["path_index"][row]),
                    "forbidden_pass": bool(expert_validity["forbidden_region_ok"][row]),
                    "stability_pass": bool(expert_validity["stability_ok"][row]),
                    "curvature_pass": bool(expert_validity["curvature_ok"][row]),
                    "safe_pass": bool(expert_validity["strict_valid"][row]),
                    "fit_curvature_feasible": bool(batch["trajectory_state_fit_curvature_feasible"][row]),
                    "fit_max_curvature": float(batch["trajectory_state_fit_max_curvature"][row]),
                    "fit_rmse_m": float(batch["trajectory_state_fit_rmse_m"][row]),
                })
            sources = torch.randn(
                contexts,
                args.sources,
                model.num_edges,
                2,
                dtype=maps.dtype,
                device=device,
                generator=source_generator,
            )
            source_bank[cursor : cursor + contexts] = sources.cpu()
            expert_positions = target_geometry["position"]
            for offset in range(0, args.sources, args.source_chunk):
                repeats = min(args.source_chunk, args.sources - offset)
                source = sources[:, offset : offset + repeats].reshape(-1, model.num_edges, 2)
                map_k = maps.repeat_interleave(repeats, dim=0)
                start_k = start.repeat_interleave(repeats, dim=0)
                goal_k = goal.repeat_interleave(repeats, dim=0)
                one = torch.ones(source.shape[0], device=device)
                zero = torch.zeros_like(one)
                states = model(map_k, source, one, zero, start_k, goal_k)
                geometry, validity = _validity_for_states(model, states, batch, device, repeats)
                for key in validity_arrays:
                    validity_arrays[key][cursor : cursor + contexts, offset : offset + repeats] = (
                        validity[key].reshape(contexts, repeats).cpu()
                    )
                expert_k = expert_positions.repeat_interleave(repeats, dim=0)
                delta = geometry["position"] - expert_k
                error_arrays["trajectory_rms_m"][cursor : cursor + contexts, offset : offset + repeats] = (
                    torch.sqrt(delta.square().mean(dim=(1, 2))).reshape(contexts, repeats).cpu()
                )
                error_arrays["trajectory_mean_displacement_m"][cursor : cursor + contexts, offset : offset + repeats] = (
                    torch.linalg.vector_norm(delta, dim=-1).mean(dim=1).reshape(contexts, repeats).cpu()
                )
            cursor += contexts
            if batch_index == 1 or batch_index % 20 == 0 or cursor == len(indices):
                print(
                    f"safe_basin contexts={cursor}/{len(indices)} sources={args.sources}",
                    flush=True,
                )
    p_safe = validity_arrays["strict_valid"].float().mean(dim=1)
    bins = {
        "p_safe_eq_0": int((p_safe == 0).sum()),
        "0_lt_p_safe_le_0.05": int(((p_safe > 0) & (p_safe <= 0.05)).sum()),
        "0.05_lt_p_safe_le_0.25": int(((p_safe > 0.05) & (p_safe <= 0.25)).sum()),
        "0.25_lt_p_safe_le_0.50": int(((p_safe > 0.25) & (p_safe <= 0.50)).sum()),
        "p_safe_gt_0.50": int((p_safe > 0.50).sum()),
    }
    safe_at_k = {
        str(k): float(validity_arrays["strict_valid"][:, :k].any(dim=1).float().mean())
        for k in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if k <= args.sources
    }
    flattened_validity = {
        key: value.reshape(-1) for key, value in validity_arrays.items()
    }
    failure_errors = _failure_error_summary(
        flattened_validity,
        error_arrays["trajectory_rms_m"].reshape(-1),
        error_arrays["trajectory_mean_displacement_m"].reshape(-1),
    )
    context_rows = []
    for i, row in enumerate(expert_rows):
        context_rows.append({
            **row,
            "p_safe": float(p_safe[i]),
            "p_forbidden_pass": float(validity_arrays["forbidden_region_ok"][i].float().mean()),
            "p_stability_pass": float(validity_arrays["stability_ok"][i].float().mean()),
            "p_curvature_pass": float(validity_arrays["curvature_ok"][i].float().mean()),
        })
    expert_summary = {
        "contexts": len(expert_rows),
        "forbidden_pass_rate": float(np.mean([row["forbidden_pass"] for row in expert_rows])),
        "stability_pass_rate": float(np.mean([row["stability_pass"] for row in expert_rows])),
        "curvature_pass_rate": float(np.mean([row["curvature_pass"] for row in expert_rows])),
        "safe_pass_rate": float(np.mean([row["safe_pass"] for row in expert_rows])),
        "fit_curvature_feasible_rate": float(np.mean([row["fit_curvature_feasible"] for row in expert_rows])),
    }
    summary = {
        "contexts": len(indices),
        "sources_per_context": args.sources,
        "source_seed": args.source_seed,
        "p_safe_distribution": _distribution(p_safe),
        "p_forbidden_pass_distribution": _distribution(validity_arrays["forbidden_region_ok"].float().mean(dim=1)),
        "p_stability_pass_distribution": _distribution(validity_arrays["stability_ok"].float().mean(dim=1)),
        "p_curvature_pass_distribution": _distribution(validity_arrays["curvature_ok"].float().mean(dim=1)),
        "p_safe_bins": bins,
        "safe_at_k": safe_at_k,
        "expert_reference": expert_summary,
        "sample_error_by_outcome": failure_errors,
    }
    arrays = {
        "source_bank": source_bank.numpy(),
        "strict_valid": validity_arrays["strict_valid"].numpy(),
        "forbidden_region_ok": validity_arrays["forbidden_region_ok"].numpy(),
        "stability_ok": validity_arrays["stability_ok"].numpy(),
        "curvature_ok": validity_arrays["curvature_ok"].numpy(),
        "trajectory_rms_m": error_arrays["trajectory_rms_m"].numpy(),
        "trajectory_mean_displacement_m": error_arrays["trajectory_mean_displacement_m"].numpy(),
        "p_safe": p_safe.numpy(),
    }
    return summary, context_rows, expert_rows, arrays


def _build_selected_batches(dataset, selected_indices, batch_size):
    return DataLoader(
        Subset(dataset, selected_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _causal_audit(
    model,
    dataset,
    selected_indices,
    validation_indices,
    p_safe,
    source_bank,
    args,
    device,
):
    position_by_index = {
        int(index): position for position, index in enumerate(validation_indices)
    }
    selected_position_by_index = {
        int(index): position for position, index in enumerate(selected_indices)
    }
    rng = np.random.default_rng(args.map_seed)
    donor_order = rng.permutation(selected_indices)
    if len(selected_indices) > 1:
        donor_order = np.roll(donor_order, 1)
    donor_maps = {
        int(source_index): dataset[int(donor_index)]["map"]
        for source_index, donor_index in zip(selected_indices, donor_order)
    }
    token_permutation = torch.as_tensor(
        rng.permutation(NUM_MAP_TOKENS), dtype=torch.long, device=device
    )
    variant_rows = []
    layer_rows = []
    heads = model.dit_blocks[0].map_cross_attn.n_heads
    attention_weight_samples = torch.zeros(
        (args.attention_samples, LAYERS, heads, NUM_EDGES, NUM_MAP_TOKENS)
    )
    half_heatmaps = max(1, args.attention_samples // 2)
    heatmap_positions = list(range(half_heatmaps)) + list(
        range(len(selected_indices) - half_heatmaps, len(selected_indices))
    )
    attention_slot_by_position = {
        position: slot for slot, position in enumerate(heatmap_positions)
    }
    attention_entropy_sum = torch.zeros((LAYERS, heads))
    attention_entropy_count = torch.zeros((LAYERS, heads))
    ca_ratio_state_values = [[] for _ in range(LAYERS)]
    ca_ratio_input_values = [[] for _ in range(LAYERS)]
    baseline_states_all = {}
    selected_cursor = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(
            _build_selected_batches(dataset, selected_indices, args.causal_batch_size),
            start=1,
        ):
            contexts = int(batch["map"].shape[0])
            batch_indices = [int(v) for v in batch["dataset_index"]]
            maps = batch["map"].float().to(device)
            swapped_maps = torch.stack([donor_maps[index] for index in batch_indices]).float().to(device)
            physical_start = batch["start_pose"].float().to(device)
            physical_goal = batch["goal_pose"].float().to(device)
            start, goal = normalize_poses(physical_start, physical_goal, model.coordinate_scale, device)
            sources = torch.as_tensor(
                source_bank[
                    [position_by_index[index] for index in batch_indices],
                    :args.causal_sources,
                ],
                dtype=torch.float32,
                device=device,
            )
            repeats = int(args.causal_sources)
            source = sources.reshape(-1, model.num_edges, 2)
            map_k = maps.repeat_interleave(repeats, dim=0)
            swap_k = swapped_maps.repeat_interleave(repeats, dim=0)
            start_k = start.repeat_interleave(repeats, dim=0)
            goal_k = goal.repeat_interleave(repeats, dim=0)
            one = torch.ones(source.shape[0], device=device)
            zero = torch.zeros_like(one)
            baseline, records = audit_forward(model, map_k, source, one, zero, start_k, goal_k, collect_attention=True)
            reference = model(map_k, source, one, zero, start_k, goal_k)
            if not torch.allclose(baseline, reference, atol=3e-5, rtol=3e-5):
                raise AssertionError(f"audit forward mismatch: {float((baseline-reference).abs().max())}")
            baseline_states_all.update({index: baseline[i * repeats : (i + 1) * repeats].detach().cpu() for i, index in enumerate(batch_indices)})
            for layer_index, record in enumerate(records):
                ca_ratio_state_values[layer_index].append(record["ca_ratio_state"].cpu())
                ca_ratio_input_values[layer_index].append(record["ca_ratio_input"].cpu())
                weights = record["attention"]
                for local_context in range(contexts):
                    slot = attention_slot_by_position.get(
                        selected_cursor + local_context
                    )
                    if slot is not None:
                        attention_weight_samples[slot, layer_index] = (
                            weights[local_context * repeats].cpu()
                        )
                entropy = record["attention_entropy"].mean(dim=2).cpu()
                attention_entropy_sum[layer_index, : entropy.shape[1]] += entropy.sum(dim=0)
                attention_entropy_count[layer_index, : entropy.shape[1]] += entropy.shape[0]

            variants = {
                "baseline": baseline,
                "all_map_ca_disabled": audit_forward(
                    model, map_k, source, one, zero, start_k, goal_k, disable_ca_all=True
                )[0],
                "map_swapped": audit_forward(
                    model, swap_k, source, one, zero, start_k, goal_k
                )[0],
                "map_tokens_shuffled": audit_forward(
                    model, map_k, source, one, zero, start_k, goal_k,
                    map_content_permutation=token_permutation,
                )[0],
            }
            for name, states in variants.items():
                if name == "baseline":
                    base = states
                summary = _metric_delta_summary(model, states, batch, base, device, repeats)
                summary.update({
                    "variant": name,
                    "contexts": contexts,
                    "sources": contexts * repeats,
                    "group": "selected_high_low",
                })
                variant_rows.append(summary)
                state_view = states.reshape(contexts, repeats, *states.shape[1:])
                base_view = base.reshape(contexts, repeats, *base.shape[1:])
                high_mask = torch.as_tensor(
                    [
                        selected_position_by_index[index]
                        < len(selected_indices) // 2
                        for index in batch_indices
                    ],
                    dtype=torch.bool,
                )
                for group_name, group_mask in (
                    ("high", high_mask),
                    ("low", ~high_mask),
                ):
                    if not bool(group_mask.any()):
                        continue
                    group_batch = _slice_batch_rows(batch, group_mask)
                    group_states = state_view[group_mask].reshape(-1, *states.shape[1:])
                    group_base = base_view[group_mask].reshape(-1, *base.shape[1:])
                    group_summary = _metric_delta_summary(
                        model,
                        group_states,
                        group_batch,
                        group_base,
                        device,
                        repeats,
                    )
                    group_summary.update({
                        "variant": name,
                        "contexts": int(group_mask.sum()),
                        "sources": int(group_mask.sum()) * repeats,
                        "group": group_name,
                    })
                    variant_rows.append(group_summary)
            for layer_index in range(LAYERS):
                states = audit_forward(
                    model, map_k, source, one, zero, start_k, goal_k,
                    disable_ca_layer=layer_index,
                )[0]
                summary = _metric_delta_summary(model, states, batch, base, device, repeats)
                summary.update({
                    "layer": layer_index,
                    "contexts": contexts,
                    "sources": contexts * repeats,
                })
                layer_rows.append(summary)
            selected_cursor += contexts
            print(
                f"causal contexts={selected_cursor}/{len(selected_indices)}",
                flush=True,
            )

    ca_state_ratios = {
        str(layer): _distribution(torch.cat(ca_ratio_state_values[layer]))
        for layer in range(LAYERS)
    }
    ca_input_ratios = {
        str(layer): _distribution(torch.cat(ca_ratio_input_values[layer]))
        for layer in range(LAYERS)
    }
    entropy = attention_entropy_sum / attention_entropy_count.clamp_min(1.0)
    # Token-level ablation on one source for eight representative contexts.
    heatmap_indices = list(selected_indices[:half_heatmaps]) + list(
        selected_indices[-half_heatmaps:]
    )
    heatmap_states = []
    heatmap_labels = []
    heatmap_safe = []
    heatmap_causal = []
    position_errors = []
    with torch.inference_mode():
        for local, index in enumerate(heatmap_indices):
            item = dataset[int(index)]
            batch = {key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim > 0 else value for key, value in item.items()}
            map_input = batch["map"].float().to(device)
            physical_start = batch["start_pose"].float().to(device)
            physical_goal = batch["goal_pose"].float().to(device)
            start, goal = normalize_poses(physical_start, physical_goal, model.coordinate_scale, device)
            source = torch.as_tensor(
                source_bank[position_by_index[index], 0],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            one = torch.ones(1, device=device)
            zero = torch.zeros(1, device=device)
            base_state, _ = audit_forward(model, map_input, source, one, zero, start, goal, collect_attention=False)
            base_geometry = model.evaluate_trajectory_state(base_state, start, goal)
            importance = torch.zeros(NUM_MAP_TOKENS, device=device)
            for offset in range(0, NUM_MAP_TOKENS, args.token_chunk):
                count = min(args.token_chunk, NUM_MAP_TOKENS - offset)
                maps = map_input.repeat(count, 1, 1, 1)
                src = source.repeat(count, 1, 1)
                st = start.repeat(count, 1)
                gl = goal.repeat(count, 1)
                mask = torch.zeros((count, NUM_MAP_TOKENS), dtype=torch.bool, device=device)
                mask[torch.arange(count), torch.arange(offset, offset + count)] = True
                ablated, _ = audit_forward(model, maps, src, torch.ones(count, device=device), torch.zeros(count, device=device), st, gl, token_mask=mask)
                geometry = model.evaluate_trajectory_state(ablated, st, gl)
                delta = geometry["position"] - base_geometry["position"]
                importance[offset : offset + count] = torch.sqrt(delta.square().mean(dim=(1, 2)))
            heatmap_causal.append(importance.cpu().reshape(12, 12))
            heatmap_labels.append("high" if local < len(heatmap_indices) // 2 else "low")
            heatmap_safe.append(float(p_safe[position_by_index[index]]))
            heatmap_states.append(base_state.cpu())
            position_errors.append(float(importance.mean()))
    attention_heatmaps = attention_weight_samples.mean(dim=(2, 3)).reshape(
        args.attention_samples, LAYERS, 12, 12
    )
    causal_arrays = {
        "attention_weights": attention_weight_samples.numpy(),
        "attention_heatmaps": attention_heatmaps.numpy(),
        "causal_importance_heatmaps": torch.stack(heatmap_causal).numpy(),
        "heatmap_context_indices": np.asarray(heatmap_indices),
        "heatmap_p_safe": np.asarray(heatmap_safe),
    }
    return {
        "selected_contexts": selected_indices,
        "high_contexts": selected_indices[: len(selected_indices) // 2],
        "low_contexts": selected_indices[len(selected_indices) // 2 :],
        "map_swap_donor_indices": [int(donor_order[list(selected_indices).index(index)]) for index in selected_indices],
        "token_permutation": token_permutation.cpu().tolist(),
        "variant_rows": variant_rows,
        "layer_rows": layer_rows,
        "ca_ratio_by_layer": ca_state_ratios,
        "ca_input_ratio_by_layer": ca_input_ratios,
        "attention_entropy_by_layer_head": entropy.tolist(),
        "token_heatmap_mean_importance": float(np.mean(position_errors)),
        "heatmap_labels": heatmap_labels,
        "heatmap_context_indices": heatmap_indices,
        "heatmap_p_safe": heatmap_safe,
    }, causal_arrays


def _save_plots(output_dir, causal_arrays):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"plots": "matplotlib_unavailable"}
    paths = {}
    causal = causal_arrays["causal_importance_heatmaps"]
    attention = causal_arrays["attention_heatmaps"]
    for name, array, cmap in (
        ("causal_importance_heatmaps", causal, "magma"),
        ("attention_heatmaps", attention, "viridis"),
    ):
        if array.ndim == 3:
            array = array[:, None]
        samples, layers, _, _ = array.shape
        fig, axes = plt.subplots(samples, layers, figsize=(2.0 * layers, 2.0 * samples), squeeze=False)
        for sample in range(samples):
            for layer in range(layers):
                axes[sample, layer].imshow(array[sample, layer], cmap=cmap)
                axes[sample, layer].set_xticks([])
                axes[sample, layer].set_yticks([])
                if sample == 0:
                    axes[sample, layer].set_title(f"L{layer}")
        fig.tight_layout()
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[name] = str(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-folder", default="data/dataset1")
    parser.add_argument("--contexts", type=int, default=400)
    parser.add_argument("--sources", type=int, default=512)
    parser.add_argument("--source-chunk", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--causal-batch-size", type=int, default=4)
    parser.add_argument("--causal-sources", type=int, default=8)
    parser.add_argument("--attention-samples", type=int, default=8)
    parser.add_argument("--token-chunk", type=int, default=16)
    parser.add_argument("--source-seed", type=int, default=20260813)
    parser.add_argument("--map-seed", type=int, default=20260816)
    parser.add_argument("--mask-seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.sources < 256:
        raise ValueError("Safe Basin audit requires at least 256 sources per context")
    if args.sources < 256 or any(k > args.sources for k in (1, 2, 4, 8, 16, 32, 64, 128, 256)):
        raise ValueError("sources must be at least 256")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic audit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_path = args.run_dir / "stage1_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("map_layout", "single_12") != "single_12":
        raise ValueError("audit is restricted to single_12")
    if checkpoint.get("output_head") != "tokenwise_22x2":
        raise ValueError("unexpected output head")
    model, _ = build_model("spatial_map", device, map_layout="single_12")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.use_gradient_checkpoint = False
    dataset, indices = _load_validation(args, summary)
    print(f"audit checkpoint={checkpoint_path} device={device} contexts={len(indices)} sources={args.sources}", flush=True)
    safe_summary, context_rows, expert_rows, safe_arrays = _safe_basin_audit(
        model, dataset, indices, args, device
    )
    _write_json(args.output_dir / "safe_basin_summary.json", safe_summary)
    _write_csv(args.output_dir / "safe_basin_contexts.csv", context_rows)
    _write_csv(args.output_dir / "expert_reference_contexts.csv", expert_rows)
    np.savez_compressed(args.output_dir / "safe_basin_arrays.npz", **safe_arrays)
    p_safe = safe_arrays["p_safe"]
    order = np.argsort(p_safe)
    low = [int(indices[i]) for i in order[: min(32, len(order))]]
    high = [int(indices[i]) for i in order[-min(32, len(order)) :][::-1]]
    selected = high + low
    causal_summary, causal_arrays = _causal_audit(
        model,
        dataset,
        selected,
        indices,
        p_safe,
        safe_arrays["source_bank"],
        args,
        device,
    )
    causal_summary["high_p_safe_distribution"] = _distribution(p_safe[order[-min(32, len(order)) :]])
    causal_summary["low_p_safe_distribution"] = _distribution(p_safe[order[: min(32, len(order))]])
    causal_summary["plots"] = _save_plots(args.output_dir, causal_arrays)
    _write_json(args.output_dir / "map_causal_summary.json", causal_summary)
    _write_csv(args.output_dir / "map_causal_variants.csv", causal_summary["variant_rows"])
    _write_csv(args.output_dir / "map_causal_layerwise.csv", causal_summary["layer_rows"])
    np.savez_compressed(args.output_dir / "map_causal_heatmaps.npz", **causal_arrays)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "map_layout": checkpoint.get("map_layout"),
        "output_head": checkpoint.get("output_head"),
        "environment_split": summary["environment_split"],
        "validation_indices": indices,
        "contexts": args.contexts,
        "sources_per_context": args.sources,
        "source_seed": args.source_seed,
        "causal_sources_per_context": args.causal_sources,
        "map_seed": args.map_seed,
        "no_training": True,
    }
    _write_json(args.output_dir / "audit_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "safe_basin": {
                    "p_safe_bins": safe_summary["p_safe_bins"],
                    "safe_at_k": safe_summary["safe_at_k"],
                    "expert_reference": safe_summary["expert_reference"],
                },
                "causal": {
                    "selected_contexts": len(causal_summary["selected_contexts"]),
                    "token_heatmap_mean_importance": causal_summary[
                        "token_heatmap_mean_importance"
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
