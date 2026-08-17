"""Formal Privileged Source Transport training for Path MeanFlow Stage 2.

Unsafe Stage-1 sources are transported to capsizing-safe targets sampled from
the frozen privileged archive. Safe sources receive only a deployment-endpoint
function anchor, so ordinary self-PMF gradients cannot move them at the Stage-1
initialization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp

from dataLoader_dit import UnevenPathDataLoader
from grad_optimizer import trajectory_validity_metrics
from map_config import MAP_CONFIG, SAFETY_COST_CONFIG, discover_environments
from posterior_pipeline import (
    _direct_cost_forward,
    _require_current_mask_semantics,
    _require_demo_target_semantics,
    _require_main_method_model,
    load_model,
    make_partial_dataset,
    normalize_poses,
)


PROTOCOL_VERSION = "privileged_source_transport_v2_train_mode"
ENDPOINT_ANCHOR_WEIGHT = 0.25


def _stable_seed(*parts) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, dict):
            return {str(key): convert(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(entry) for entry in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (np.generic,)):
            return item.item()
        if isinstance(item, torch.Tensor):
            return convert(item.detach().cpu().tolist())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(convert(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stack_items(items):
    return {
        key: torch.stack([item[key].float() for item in items])
        for key in ("map", "start_pose", "goal_pose", "cost_map", "mask")
    }


def _validity(output):
    return trajectory_validity_metrics(
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


def _deterministic_endpoint(model, batch, source, device):
    was_training = model.training
    model.eval()
    try:
        start, goal = normalize_poses(
            batch["start_pose"].float(),
            batch["goal_pose"].float(),
            model.coordinate_scale,
            device,
        )
        one = torch.ones(len(source), device=device)
        zero = torch.zeros(len(source), device=device)
        return model.project_zero_sum(
            model(
                batch["map"].float().to(device),
                source.float().to(device),
                one,
                zero,
                start,
                goal,
            )
        )
    finally:
        model.train(was_training)


def _source_pool(model, entries, args, device):
    sources = []
    time_t = []
    time_r = []
    for entry in entries:
        source_generator = torch.Generator(device=device).manual_seed(
            int(entry["source_seed"])
        )
        sources.append(
            torch.randn(
                int(args.stage2_sources_per_context),
                model.num_edges,
                2,
                device=device,
                generator=source_generator,
            )
        )
        time_generator = torch.Generator(device=device).manual_seed(
            int(entry["time_seed"])
        )
        first = model.sample_timesteps(1, device, generator=time_generator)
        second = model.sample_timesteps(1, device, generator=time_generator)
        t = torch.maximum(first, second)
        r = torch.minimum(first, second)
        if bool(
            torch.rand(1, device=device, generator=time_generator).item()
            < float(args.stage2_endpoint_probability)
        ):
            t = torch.ones_like(t)
            r = torch.zeros_like(r)
        time_t.append(t)
        time_r.append(r)
    return torch.stack(sources), torch.cat(time_t), torch.cat(time_r)


@torch.no_grad()
def _reference_endpoints(reference, items, sources, device):
    contexts, pool_size = sources.shape[:2]
    stacked = _stack_items(items)
    output = _direct_cost_forward(
        reference,
        stacked,
        device,
        sources_per_context=pool_size,
        source=sources.reshape(contexts * pool_size, reference.num_edges, 2),
    )
    validity = _validity(output)
    anchor_batch = {
        key: stacked[key].float().to(device).repeat_interleave(pool_size, dim=0)
        for key in ("map", "start_pose", "goal_pose")
    }
    with torch.enable_grad():
        anchor_state = _deterministic_endpoint(
            reference,
            anchor_batch,
            sources.reshape(contexts * pool_size, reference.num_edges, 2),
            device,
        ).detach()
    start, goal = normalize_poses(
        anchor_batch["start_pose"],
        anchor_batch["goal_pose"],
        reference.coordinate_scale,
        device,
    )
    position = reference.evaluate_trajectory_state(
        anchor_state, start, goal
    )["position"]
    return {
        "state": anchor_state.reshape(contexts, pool_size, reference.num_edges, 2),
        "position": position.reshape(contexts, pool_size, position.shape[1], 2),
        "safe": (
            validity["forbidden_region_ok"].bool()
            & validity["stability_ok"].bool()
        ).reshape(contexts, pool_size),
    }


def _target_probabilities(group, temperature):
    indices = torch.nonzero(group["capsizing_safe"].bool()).flatten()
    curvature = group["max_curvature"][indices].double()
    excess = torch.relu(
        curvature / float(SAFETY_COST_CONFIG.curvature_limit) - 1.0
    )
    return indices, torch.softmax(-excess / float(temperature), dim=0)


@torch.no_grad()
def _target_positions(reference, states, group, device):
    start, goal = normalize_poses(
        group["start_pose"].float().to(device).repeat(len(states), 1),
        group["goal_pose"].float().to(device).repeat(len(states), 1),
        reference.coordinate_scale,
        device,
    )
    return reference.evaluate_trajectory_state(
        states.float().to(device), start, goal
    )["position"]


def _build_pairs(reference, groups, entries, sources, bundle, args, device):
    unsafe_source = []
    unsafe_target = []
    unsafe_context = []
    safe_source = []
    safe_target = []
    safe_context = []
    for local, entry in enumerate(entries):
        group = groups[int(entry["group_index"])]
        safe_indices = torch.nonzero(bundle["safe"][local]).flatten()
        unsafe_indices = torch.nonzero(~bundle["safe"][local]).flatten()
        for source_index in safe_indices.tolist():
            safe_source.append(sources[local, source_index])
            safe_target.append(bundle["state"][local, source_index])
            safe_context.append(local)
        if not len(unsafe_indices):
            continue
        eligible, probabilities = _target_probabilities(
            group, float(args.stage2_target_temperature)
        )
        generator = torch.Generator().manual_seed(int(entry["target_seed"]))
        sampled_local = torch.multinomial(
            probabilities,
            len(unsafe_indices),
            replacement=True,
            generator=generator,
        )
        sampled = eligible[sampled_local]
        target_state = group["state"][sampled].float().to(device)
        target_position = _target_positions(reference, target_state, group, device)
        distance = (
            (bundle["position"][local, unsafe_indices, None] - target_position[None])
            .square()
            .sum(dim=-1)
            .mean(dim=-1)
            .sqrt()
        )
        rows, columns = linear_sum_assignment(
            distance.detach().cpu().double().numpy()
        )
        if list(rows) != list(range(len(unsafe_indices))):
            raise AssertionError("Unexpected Hungarian row ordering")
        assigned = [int(sampled[int(column)]) for column in columns]
        if Counter(int(value) for value in sampled) != Counter(assigned):
            raise AssertionError("Hungarian changed the sampled target multiset")
        for row, source_index in enumerate(unsafe_indices.tolist()):
            column = int(columns[row])
            unsafe_source.append(sources[local, source_index])
            unsafe_target.append(target_state[column])
            unsafe_context.append(local)

    def stack(values):
        if values:
            return torch.stack(values)
        return torch.empty(0, reference.num_edges, 2, device=device)

    return {
        "unsafe_source": stack(unsafe_source),
        "unsafe_target": stack(unsafe_target),
        "unsafe_context": torch.as_tensor(unsafe_context, dtype=torch.long, device=device),
        "safe_source": stack(safe_source),
        "safe_target": stack(safe_target),
        "safe_context": torch.as_tensor(safe_context, dtype=torch.long, device=device),
    }


def _repeated_batch(items, indices, device):
    values = [int(value) for value in indices.detach().cpu().tolist()]
    return {
        key: torch.stack([items[index][key].float() for index in values]).to(device)
        for key in ("map", "start_pose", "goal_pose")
    }


def _fixed_time_pmf_loss(
    model, batch, source, target, time_t, time_r, args, device
):
    map_input = batch["map"].float().to(device)
    source = model.project_zero_sum(source.float().to(device))
    target = model.project_zero_sum(target.float().to(device))
    start, goal = normalize_poses(
        batch["start_pose"].float(),
        batch["goal_pose"].float(),
        model.coordinate_scale,
        device,
    )
    t = time_t.float().to(device).clone().requires_grad_(True)
    r = time_r.float().to(device)
    target_velocity = model.project_zero_sum(source - target)
    z_t = model.project_zero_sum(
        (1.0 - t[:, None, None]) * target + t[:, None, None] * source
    )
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def average_velocity(z_arg, t_arg, r_arg):
        output = functional_call(
            model,
            {**params, **buffers},
            (map_input, z_arg, t_arg, r_arg, start, goal),
        )
        output = model.project_zero_sum(output)
        return model.project_zero_sum(
            (z_arg - output) / (t_arg[:, None, None] + 1e-5)
        )

    was_training = model.training
    model.eval()
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            velocity, derivative = jvp(
                average_velocity,
                (z_t, t, r),
                (target_velocity, torch.ones_like(t), torch.zeros_like(r)),
            )
    finally:
        model.train(was_training)
    mean_velocity = model.project_zero_sum(
        velocity
        + (t - r)[:, None, None] * derivative.detach().clamp(-5.0, 5.0)
    )
    flow = (mean_velocity - target_velocity).square().mean(dim=(1, 2))
    one = torch.ones(len(source), device=device)
    zero = torch.zeros(len(source), device=device)
    endpoint_state = model(map_input, source, one, zero, start, goal)
    endpoint = (
        model.project_zero_sum(endpoint_state) - target
    ).square().mean(dim=(1, 2))
    audit_curvature = model.audit_trajectory_state_curvature(
        endpoint_state, start, goal
    )
    excess = torch.relu(
        audit_curvature / float(SAFETY_COST_CONFIG.curvature_limit) - 1.0
    )
    if args.stage2_curvature_penalty == "bounded":
        excess = excess / (1.0 + excess)
    elif args.stage2_curvature_penalty == "log1p":
        excess = torch.log1p(excess)
    else:
        raise ValueError(f"Unknown curvature penalty: {args.stage2_curvature_penalty}")
    tail_count = max(
        1,
        int(math.ceil(excess.shape[1] * float(args.stage2_curvature_tail_ratio))),
    )
    curvature = torch.topk(
        excess, k=tail_count, dim=1, largest=True, sorted=False
    ).values.mean(dim=1)
    return (
        flow
        + 0.25 * endpoint
        + float(args.stage2_curvature_weight) * curvature
    )


def _training_items(dataset, groups, entries):
    items = []
    for entry in entries:
        group = groups[int(entry["group_index"])]
        item = dataset.get_item(
            int(group["dataset_index"]), mask_variant=int(group["mask_variant"])
        )
        if _tensor_sha256(item["map"], item["mask"]) != str(
            group["observed_input_sha256"]
        ):
            raise RuntimeError(f"Observation lineage mismatch for group {group['group_id']}")
        if not torch.allclose(
            item["start_pose"].float(), group["start_pose"].float(), atol=1e-7, rtol=0
        ) or not torch.allclose(
            item["goal_pose"].float(), group["goal_pose"].float(), atol=1e-7, rtol=0
        ):
            raise RuntimeError(f"Pose lineage mismatch for group {group['group_id']}")
        items.append(item)
    return items


def _make_validation_dataset(root: Path, mask_seed: int, environment_count: int):
    available_environments = discover_environments(root / "val")
    if int(environment_count) > len(available_environments):
        raise ValueError(
            "Stage-2 validation requests "
            f"{environment_count}/{len(available_environments)} environments"
        )
    environments = sorted(available_environments)[: int(environment_count)]
    sealed_environments = sorted(set(available_environments) - set(environments))
    dataset = UnevenPathDataLoader(
        environments,
        str(root / "val"),
        compute_stability_map=True,
        use_precomputed_stability=True,
        compute_stability_if_missing=False,
        partial_observation=True,
        include_mask=True,
        mask_seed=int(mask_seed),
        p_mask=1.0,
        dynamic_mask_noise=False,
        mask_mode="stage2_independent",
        vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
        encode_path_coordinates=True,
    )
    return dataset, environments, sealed_environments


def _validation_indices(dataset, context_limit: int):
    by_environment = {}
    for dataset_index, (environment_index, _) in enumerate(dataset.indexDict):
        by_environment.setdefault(int(environment_index), []).append(dataset_index)
    ordered = []
    depth = 0
    while len(ordered) < min(int(context_limit), len(dataset)):
        added = False
        for environment_index in sorted(by_environment):
            values = by_environment[environment_index]
            if depth < len(values):
                ordered.append(values[depth])
                added = True
                if len(ordered) >= int(context_limit):
                    break
        if not added:
            break
        depth += 1
    return ordered


@torch.no_grad()
def _evaluate(model, dataset, indices, args, device):
    """Evaluate without leaking ``eval()`` into the following train update."""
    was_training = model.training
    try:
        model.eval()
        return _evaluate_in_eval_mode(model, dataset, indices, args, device)
    finally:
        model.train(was_training)


@torch.no_grad()
def _evaluate_in_eval_mode(model, dataset, indices, args, device):
    model.eval()
    k = int(args.stage2_validation_sources)
    rows = []
    for dataset_index in indices:
        item = dataset.get_item(
            int(dataset_index),
            mask_variant=(
                _stable_seed(args.mask_seed, "stage2-validation", dataset_index)
                % (2**31 - 1)
            ),
        )
        generator = torch.Generator(device=device).manual_seed(
            _stable_seed(args.stage2_eval_seed, dataset_index)
        )
        source = torch.randn(
            k, model.num_edges, 2, device=device, generator=generator
        )
        output = _direct_cost_forward(
            model,
            _stack_items([item]),
            device,
            sources_per_context=k,
            source=source,
        )
        validity = _validity(output)
        capsizing = (
            validity["forbidden_region_ok"].bool()
            & validity["stability_ok"].bool()
        )
        paths = output["geometry"]["position"]
        if k > 1:
            distances = torch.pdist(paths.flatten(1), p=2) / math.sqrt(paths.shape[1])
            diversity = float(distances.median())
        else:
            diversity = 0.0
        rows.append(
            {
                "capsizing_safe_count": int(capsizing.sum()),
                "safe_at_k": int(capsizing.any()),
                "forbidden_count": int(validity["forbidden_region_ok"].sum()),
                "stability_count": int(validity["stability_ok"].sum()),
                "curvature_count": int(validity["curvature_ok"].sum()),
                "diversity_m": diversity,
                "curvature": validity["max_curvature"].detach().cpu().tolist(),
            }
        )
    proposals = len(rows) * k
    curvature = np.asarray(
        [value for row in rows for value in row["curvature"]], dtype=np.float64
    )
    return {
        "contexts": len(rows),
        "proposals": proposals,
        "capsizing_safe_rate": sum(row["capsizing_safe_count"] for row in rows) / proposals,
        "safe_at_k": sum(row["safe_at_k"] for row in rows) / len(rows),
        "forbidden_pass_rate": sum(row["forbidden_count"] for row in rows) / proposals,
        "stability_pass_rate": sum(row["stability_count"] for row in rows) / proposals,
        "curvature_pass_rate": sum(row["curvature_count"] for row in rows) / proposals,
        "diversity_median": float(np.median([row["diversity_m"] for row in rows])),
        "curvature_q95": float(np.quantile(curvature, 0.95)),
        "curvature_q99": float(np.quantile(curvature, 0.99)),
        "curvature_max": float(curvature.max()),
    }


def _save_checkpoint(path, model, optimizer, model_args, metadata):
    payload = {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_args": model_args,
            "stage": "stage2_privileged_source_transport",
            "representation_semantic_version": metadata["representation_semantic_version"],
            "input_mask_semantics": metadata["input_mask_semantics"],
            "demo_target_semantics": metadata["demo_target_semantics"],
            **metadata,
        }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def train_stage2_source_transport(args, device):
    """Train the current Stage-2 generator and return its best checkpoint."""
    if not args.prior_checkpoint:
        raise ValueError("Stage 2 requires --prior_checkpoint")
    output = Path(args.fileDir)
    output.mkdir(parents=True, exist_ok=True)
    target_archive_path = Path(args.stage2_target_archive)
    archive = torch.load(target_archive_path, map_location="cpu", weights_only=False)
    checkpoint_hash = _sha256_file(args.prior_checkpoint)
    if archive["protocol"]["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("Target archive was generated by another Stage-1 checkpoint")
    groups = [
        group for group in archive["groups"] if bool(group["capsizing_safe"].any())
    ]
    if not groups:
        raise RuntimeError("Target archive contains no capsizing-safe groups")
    if args.max_contexts is not None:
        groups = groups[: int(args.max_contexts)]

    dataset, _ = make_partial_dataset(
        args.dataFolder,
        "train",
        compute_stability_map=True,
        mask_seed=int(args.mask_seed),
        p_mask=1.0,
        mask_mode="stage2_independent",
        dynamic_mask_noise=False,
    )
    (
        validation_dataset,
        validation_environments,
        sealed_validation_environments,
    ) = _make_validation_dataset(
        Path(args.stage2_validation_data),
        int(args.mask_seed),
        int(args.stage2_validation_environments),
    )
    validation_indices = _validation_indices(
        validation_dataset, int(args.stage2_validation_contexts)
    )

    reference, model_args, checkpoint = load_model(args.prior_checkpoint, device)
    model, _, _ = load_model(args.prior_checkpoint, device)
    for value in (reference, model):
        _require_main_method_model(value)
        value.use_gradient_checkpoint = False
    _require_current_mask_semantics(checkpoint)
    _require_demo_target_semantics(checkpoint)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.stage2_lr),
        weight_decay=float(args.stage2_weight_decay),
    )

    start_epoch = 1
    global_update = 0
    if args.stage2_resume:
        resumed = torch.load(args.stage2_resume, map_location=device, weights_only=False)
        if resumed.get("stage") != "stage2_privileged_source_transport":
            raise ValueError("--stage2_resume is not a source-transport checkpoint")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        start_epoch = int(resumed["epoch"]) + 1
        global_update = int(resumed["optimizer_update"])
    if start_epoch > int(args.stage2_epochs):
        raise ValueError(
            "--stage2_epochs must be greater than the epoch stored in "
            "--stage2_resume"
        )

    protocol = {
        "version": PROTOCOL_VERSION,
        "stage1_checkpoint": str(Path(args.prior_checkpoint).resolve()),
        "stage1_checkpoint_sha256": checkpoint_hash,
        "target_archive": str(target_archive_path.resolve()),
        "target_archive_sha256": _sha256_file(target_archive_path),
        "training_groups": len(groups),
        "validation_data": str(Path(args.stage2_validation_data).resolve()),
        "validation_environments": validation_environments,
        "sealed_validation_environments": sealed_validation_environments,
        "sealed_validation_environment_count": len(sealed_validation_environments),
        "validation_contexts": len(validation_indices),
        "sources_per_context": int(args.stage2_sources_per_context),
        "order_seed": int(args.stage2_order_seed),
        "source_seed": int(args.stage2_source_seed),
        "target_seed": int(args.stage2_target_seed),
        "all_parameters_trainable": True,
        "safe_source_objective": "deployment_endpoint_anchor_only",
        "unsafe_source_objective": "full_pmf_to_sampled_safe_target",
        "target_coupling": "sample_from_curvature_soft_prior_then_hungarian",
        "train_mode_restored_after_validation": True,
        "dropout_seed_scheme": "stable(stage2-order-seed,dropout,update,micro_offset)",
        "final_test_opened": False,
    }
    _write_json(output / "stage2_protocol.json", protocol)
    baseline = _evaluate(reference, validation_dataset, validation_indices, args, device)
    _write_json(output / "stage2_validation_baseline.json", baseline)
    best_score = (baseline["safe_at_k"], baseline["capsizing_safe_rate"])
    best_path = output / "stage2_best.pth"
    last_path = output / "stage2_last.pth"
    trace = []
    validation_trace = []
    stop = False

    for epoch in range(start_epoch, int(args.stage2_epochs) + 1):
        indices = list(range(len(groups)))
        random.Random(
            _stable_seed(args.stage2_order_seed, "t2-order", epoch)
        ).shuffle(indices)
        for offset in range(0, len(indices), int(args.stage2_batch_size)):
            selected = indices[offset : offset + int(args.stage2_batch_size)]
            if not selected:
                continue
            entries = []
            for group_index in selected:
                group_id = int(groups[group_index]["group_id"])
                entries.append(
                    {
                        "group_index": group_index,
                        "source_seed": _stable_seed(
                            args.stage2_source_seed,
                            "t2-source",
                            epoch,
                            group_id,
                        ),
                        "time_seed": _stable_seed(
                            args.stage2_source_seed,
                            "t2-time",
                            epoch,
                            group_id,
                        ),
                        "target_seed": _stable_seed(
                            args.stage2_target_seed,
                            "t2-target",
                            epoch,
                            group_id,
                        ),
                    }
                )
            items = _training_items(dataset, groups, entries)
            sources, time_t, time_r = _source_pool(reference, entries, args, device)
            bundle = _reference_endpoints(reference, items, sources, device)
            pairs = _build_pairs(
                reference, groups, entries, sources, bundle, args, device
            )
            slots = len(entries) * int(args.stage2_sources_per_context)
            optimizer.zero_grad(set_to_none=True)
            # Validation and deterministic endpoint helpers use eval mode.  The
            # train objective must explicitly restore dropout for every update;
            # otherwise the first validation silently changes all later updates.
            model.train()
            unsafe_sum = 0.0
            for micro_offset in range(
                0, len(pairs["unsafe_source"]), int(args.stage2_source_microbatch)
            ):
                selection = slice(
                    micro_offset, micro_offset + int(args.stage2_source_microbatch)
                )
                context = pairs["unsafe_context"][selection]
                batch = _repeated_batch(items, context, device)
                dropout_seed = _stable_seed(
                    args.stage2_order_seed,
                    "dropout",
                    global_update + 1,
                    micro_offset,
                )
                torch.manual_seed(dropout_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(dropout_seed)
                loss = _fixed_time_pmf_loss(
                    model,
                    batch,
                    pairs["unsafe_source"][selection],
                    pairs["unsafe_target"][selection],
                    time_t[context],
                    time_r[context],
                    args,
                    device,
                )
                (loss.sum() / float(slots)).backward()
                unsafe_sum += float(loss.detach().sum())
            anchor_sum = 0.0
            if len(pairs["safe_source"]):
                anchor_batch = _repeated_batch(items, pairs["safe_context"], device)
                prediction = _deterministic_endpoint(
                    model, anchor_batch, pairs["safe_source"], device
                )
                anchor = (
                    prediction - pairs["safe_target"].detach()
                ).square().mean(dim=(1, 2))
                (
                    ENDPOINT_ANCHOR_WEIGHT * anchor.sum() / float(slots)
                ).backward()
                anchor_sum = float(anchor.detach().sum())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.stage2_grad_clip_norm)
            )
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(
                    f"Non-finite Stage-2 gradient at update {global_update + 1}"
                )
            optimizer.step()
            global_update += 1
            trace.append(
                {
                    "epoch": epoch,
                    "update": global_update,
                    "contexts": len(entries),
                    "unsafe_sources": len(pairs["unsafe_source"]),
                    "safe_sources": len(pairs["safe_source"]),
                    "unsafe_pmf_loss_per_slot": unsafe_sum / slots,
                    "anchor_mse_per_slot": anchor_sum / slots,
                    "gradient_norm_pre_clip": float(gradient_norm),
                }
            )
            if global_update == 1 or global_update % 10 == 0:
                print(
                    f"stage2 update={global_update} epoch={epoch} "
                    f"unsafe={len(pairs['unsafe_source'])}/{slots} "
                    f"loss={unsafe_sum / slots:.6f} grad={float(gradient_norm):.3f}",
                    flush=True,
                )
            at_epoch_end = offset + int(args.stage2_batch_size) >= len(indices)
            should_evaluate = (
                global_update % int(args.stage2_eval_every_updates) == 0
                or at_epoch_end
            )
            if should_evaluate:
                metrics = _evaluate(
                    model, validation_dataset, validation_indices, args, device
                )
                score = (metrics["safe_at_k"], metrics["capsizing_safe_rate"])
                diversity_ok = (
                    metrics["diversity_median"]
                    >= 0.9 * baseline["diversity_median"]
                )
                forbidden_ok = (
                    metrics["forbidden_pass_rate"]
                    >= baseline["forbidden_pass_rate"] - 0.01
                )
                selected_best = score > best_score and diversity_ok and forbidden_ok
                row = {
                    "epoch": epoch,
                    "update": global_update,
                    **metrics,
                    "diversity_safeguard": int(diversity_ok),
                    "forbidden_safeguard": int(forbidden_ok),
                    "selected_best": int(selected_best),
                }
                validation_trace.append(row)
                print("stage2 validation " + json.dumps(row, sort_keys=True), flush=True)
                if selected_best:
                    best_score = score
                    _save_checkpoint(
                        best_path,
                        model,
                        None,
                        model_args,
                        {
                            "epoch": epoch,
                            "optimizer_update": global_update,
                            "protocol": protocol,
                            "validation_metrics": metrics,
                            "representation_semantic_version": checkpoint["representation_semantic_version"],
                            "input_mask_semantics": checkpoint["input_mask_semantics"],
                            "demo_target_semantics": checkpoint["demo_target_semantics"],
                        },
                    )
            if (
                args.stage2_max_updates is not None
                and global_update >= int(args.stage2_max_updates)
            ):
                stop = True
                break

        last_metrics = validation_trace[-1] if validation_trace else baseline
        _save_checkpoint(
            last_path,
            model,
            optimizer,
            model_args,
            {
                "epoch": epoch,
                "optimizer_update": global_update,
                "protocol": protocol,
                "validation_metrics": last_metrics,
                "representation_semantic_version": checkpoint["representation_semantic_version"],
                "input_mask_semantics": checkpoint["input_mask_semantics"],
                "demo_target_semantics": checkpoint["demo_target_semantics"],
            },
        )
        _write_csv(output / "stage2_training.csv", trace)
        _write_csv(output / "stage2_validation.csv", validation_trace)
        if stop:
            break

    summary = {
        "protocol": protocol,
        "baseline": baseline,
        "best_score": {"safe_at_k": best_score[0], "capsizing_safe_rate": best_score[1]},
        "best_checkpoint": str(best_path if best_path.exists() else last_path),
        "last_checkpoint": str(last_path),
        "updates": global_update,
        "completed_epochs": int(last_path.exists() and epoch or 0),
        "final_test_opened": False,
    }
    _write_json(output / "stage2_summary.json", summary)
    return best_path if best_path.exists() else last_path
