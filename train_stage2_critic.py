#!/usr/bin/env python3
"""Train the deployable Stage-2 stability critic as a two-part risk model.

The existing scaled critic only supervised ``safe_logit``.  This experiment
keeps that probability objective and additionally trains ``risk_head`` to
predict the conditional severity of a capsizing violation.  Privileged terrain
is used only to regenerate labels for the already frozen candidate sources.

The two outputs have fixed semantics:

* sigmoid(safe_logit) = P(stability-safe | deployment inputs)
* predicted_risk = E[log1p(V_stability / d_safe) | stability-unsafe, inputs]

Candidate paths, masks, and latent sources are regenerated deterministically
and checked against the frozen archives before any label is accepted.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import random
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataLoader_dit import UnevenPathDataLoader
from grad_optimizer import trajectory_validity_metrics
from map_config import MAP_CONFIG, SAFETY_COST_CONFIG
from posterior_pipeline import (
    _direct_cost_forward,
    _require_current_mask_semantics,
    _require_demo_target_semantics,
    _require_main_method_model,
    load_model,
    make_partial_dataset,
)
from stage2_critic import (
    PathAlignedCandidateCritic,
    sha256_file,
)


EXPERIMENT_VERSION = "stability_critic_dual_head_v1"
LABEL_VERSION = "stability_critic_conditional_severity_labels_v1"
DEPLOYABLE_FIELDS = (
    "observed_map",
    "start_condition",
    "goal_condition",
    "candidate_path",
    "candidate_curvature",
)
SEVERITY_FIELDS = (
    "stability_violation_max",
    "stability_violation_mean",
    "stability_violation_ratio",
    "min_stability_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scaled-archive-dir",
        type=Path,
        default=Path(
            "tests/audits/stage2_stability_critic_scaled_v1_20260803"
        ),
    )
    parser.add_argument(
        "--external-candidate-archive",
        type=Path,
        default=Path(
            "data/path_meanflow_stage2_reweighting/"
            "external_validation_candidates_v1.pt"
        ),
    )
    parser.add_argument(
        "--initial-critic-checkpoint",
        type=Path,
        default=Path(
            "tests/audits/stage2_stability_critic_scaled_v1_20260803/"
            "critic_scaled_6400.pth"
        ),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=Path("data/path_meanflow/stage1_best.pth"),
    )
    parser.add_argument("--data-folder", type=Path, default=Path("data/dataset1"))
    parser.add_argument(
        "--external-dataset-root", type=Path, default=Path("data/dataset1_val")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/path_meanflow_stage2/critic"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--collection-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--path-check-tolerance", type=float, default=2e-5)
    parser.add_argument(
        "--fit-environment-limit",
        type=int,
        default=None,
        help="Smoke only. Limit the number of fit environments.",
    )
    parser.add_argument(
        "--validation-environment-limit",
        type=int,
        default=None,
        help="Smoke only. Limit the number of validation environments.",
    )
    return parser.parse_args()


def chunks(values: Sequence[Any], size: int):
    for offset in range(0, len(values), int(size)):
        yield values[offset : offset + int(size)]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_identity(group: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(group["environment"]),
        int(group["path_num"]),
        int(group["mask_variant"]),
    )


def severity_target(violation_max: torch.Tensor) -> torch.Tensor:
    """Dimensionless conditional-severity regression target."""
    normalized = violation_max.clamp_min(0.0) / float(
        SAFETY_COST_CONFIG.d_safe_meters
    )
    return torch.log1p(normalized)


def conditional_severity_loss(
    predicted_risk: torch.Tensor,
    violation_max: torch.Tensor,
    constraint_ok: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if predicted_risk.shape != violation_max.shape:
        raise ValueError("risk prediction and continuous label shapes differ")
    if predicted_risk.shape != constraint_ok.shape[:2]:
        raise ValueError("risk prediction and constraint labels do not align")
    forbidden = constraint_ok[..., 0].bool()
    unsafe = ~constraint_ok[..., 1].bool()
    mask = forbidden & unsafe
    count = int(mask.sum())
    if count == 0:
        return predicted_risk.sum() * 0.0, 0
    target = severity_target(violation_max)
    return F.smooth_l1_loss(predicted_risk[mask], target[mask]), count


def curve_preference(curvature_max: torch.Tensor) -> torch.Tensor:
    limit = float(SAFETY_COST_CONFIG.curvature_limit)
    return torch.clamp(limit / curvature_max.clamp_min(limit), max=1.0)


def dual_head_score(
    predicted_risk: torch.Tensor,
    safe_logit: torch.Tensor,
    curvature_max: torch.Tensor,
) -> torch.Tensor:
    """Safety probability with conditional expected-risk and curve penalties."""
    safe_probability = torch.sigmoid(safe_logit)
    expected_severity = (1.0 - safe_probability) * predicted_risk
    return (
        safe_probability
        * curve_preference(curvature_max)
        * torch.exp(-expected_severity)
    )


def stable_int(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stack_items(items):
    return {
        key: torch.stack([item[key].float() for item in items])
        for key in ("map", "start_pose", "goal_pose", "cost_map", "mask")
    }


def batch_for_contexts(dataset, contexts, variant: int, *, metadata=False):
    items = [
        dataset.get_item(
            int(context["dataset_index"]),
            mask_variant=int(variant),
            return_mask_metadata=metadata,
        )
        for context in contexts
    ]
    return _stack_items(items), items


def fixed_sources(
    contexts,
    *,
    count: int,
    num_edges: int,
    seed: int,
    phase: str,
    occurrence: int,
):
    result = []
    for context in contexts:
        generator = torch.Generator().manual_seed(
            stable_int(
                seed,
                phase,
                occurrence,
                context["environment"],
                context["path_num"],
            )
            % (2**63 - 1)
        )
        result.append(torch.randn(int(count), int(num_edges), 2, generator=generator))
    return torch.cat(result, dim=0)


def validity_from_output(output):
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


def stability_probability_loss(safe_logit, constraint_ok):
    if safe_logit.shape != constraint_ok.shape[:2]:
        raise ValueError("safe logits and constraint labels do not align")
    forbidden = constraint_ok[..., 0].bool()
    stability = constraint_ok[..., 1].float()
    if not bool(forbidden.any()):
        raise ValueError("Training batch has no forbidden-valid candidate")
    return F.binary_cross_entropy_with_logits(
        safe_logit[forbidden], stability[forbidden]
    )


def stability_pairwise_ranking_loss(safe_logit, constraint_ok):
    if safe_logit.shape != constraint_ok.shape[:2]:
        raise ValueError("safe logits and constraint labels do not align")
    forbidden = constraint_ok[..., 0].bool()
    stability = constraint_ok[..., 1].float()
    target_delta = stability[:, :, None] - stability[:, None, :]
    logit_delta = safe_logit[:, :, None] - safe_logit[:, None, :]
    pair_mask = (
        torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
        & forbidden[:, :, None]
        & forbidden[:, None, :]
        & (target_delta.abs() > 0.5)
    )
    if not bool(pair_mask.any()):
        return safe_logit.sum() * 0.0
    return F.softplus(
        -target_delta.sign()[pair_mask] * logit_delta[pair_mask]
    ).mean()


def _binary_auc(probability, target):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    positive = probability[target]
    negative = probability[~target]
    if not len(positive) or not len(negative):
        return None
    comparison = positive[:, None] - negative[None, :]
    return float(
        (np.sum(comparison > 0.0) + 0.5 * np.sum(comparison == 0.0))
        / comparison.size
    )


def calibration_metrics(probability, target, bins=10):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not len(probability):
        raise ValueError("No calibration examples")
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    nll = -np.mean(
        target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)
    )
    brier = np.mean((probability - target) ** 2)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    calibration_bins = []
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        active = (probability >= lower) & (
            probability <= upper if index + 1 == bins else probability < upper
        )
        if not np.any(active):
            continue
        confidence = float(probability[active].mean())
        frequency = float(target[active].mean())
        ece += float(active.mean()) * abs(confidence - frequency)
        calibration_bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(active.sum()),
                "confidence": confidence,
                "frequency": frequency,
            }
        )
    return {
        "examples": int(len(probability)),
        "positive_rate": float(target.mean()),
        "nll": float(nll),
        "brier": float(brier),
        "ece_10": float(ece),
        "auc": _binary_auc(probability, target.astype(bool)),
        "bins": calibration_bins,
    }


def make_external_dataset(root: Path, environments, args):
    return UnevenPathDataLoader(
        list(environments),
        str(root / "val"),
        compute_stability_map=True,
        use_precomputed_stability=True,
        compute_stability_if_missing=False,
        partial_observation=True,
        include_mask=True,
        mask_seed=int(args.mask_seed),
        p_mask=1.0,
        dynamic_mask_noise=False,
        mask_mode="stage2_independent",
        vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
        encode_path_coordinates=True,
    )


@torch.no_grad()
def regenerated_records(
    model,
    dataset,
    contexts: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[int],
    candidates: int,
    source_seed: int,
    source_phase,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model.eval()
    for variant in variants:
        for context_batch in chunks(contexts, int(batch_size)):
            batch, _ = batch_for_contexts(
                dataset, context_batch, int(variant), metadata=True
            )
            phase = (
                source_phase(context_batch[0])
                if callable(source_phase)
                else str(source_phase)
            )
            source = fixed_sources(
                context_batch,
                count=int(candidates),
                num_edges=model.num_edges,
                seed=int(source_seed),
                phase=phase,
                occurrence=0,
            )
            output = _direct_cost_forward(
                model,
                batch,
                device,
                sources_per_context=int(candidates),
                source=source,
            )
            validity = validity_from_output(output)
            context_count = len(context_batch)
            paths = output["geometry"]["position"].reshape(
                context_count, int(candidates), -1, 2
            )
            constraint_ok = torch.stack(
                [
                    validity["forbidden_region_ok"],
                    validity["stability_ok"],
                    validity["curvature_ok"],
                ],
                dim=-1,
            ).reshape(context_count, int(candidates), 3)
            fields = {
                key: validity[key].reshape(context_count, int(candidates))
                for key in SEVERITY_FIELDS
            }
            for index, context in enumerate(context_batch):
                record = {
                    "environment": str(context["environment"]),
                    "path_num": int(context["path_num"]),
                    "mask_variant": int(variant),
                    "candidate_path": paths[index].cpu(),
                    "constraint_ok": constraint_ok[index].cpu(),
                }
                record.update({key: value[index].cpu() for key, value in fields.items()})
                records.append(record)
    return records


def attach_verified_labels(
    groups: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen = {group_identity(group): group for group in groups}
    generated = {group_identity(record): record for record in records}
    if set(frozen) != set(generated):
        raise ValueError("Regenerated label identities differ from frozen candidates")
    enriched: list[dict[str, Any]] = []
    maximum_path_error = 0.0
    constraint_mismatches = 0
    for group in groups:
        identity = group_identity(group)
        record = generated[identity]
        path_error = float(
            (group["candidate_path"].float() - record["candidate_path"].float())
            .abs()
            .max()
        )
        maximum_path_error = max(maximum_path_error, path_error)
        if path_error > float(tolerance):
            raise ValueError(
                f"Candidate regeneration changed {identity}: max error={path_error}"
            )
        mismatches = int(
            (group["constraint_ok"].bool() != record["constraint_ok"].bool())
            .any(dim=-1)
            .sum()
        )
        constraint_mismatches += mismatches
        if mismatches:
            raise ValueError(
                f"Privileged labels changed for {identity}: {mismatches} candidates"
            )
        item = dict(group)
        for key in SEVERITY_FIELDS:
            value = record[key].float()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Non-finite regenerated label: {key}")
            item[key] = value
        enriched.append(item)
    return enriched, {
        "groups": len(enriched),
        "maximum_candidate_path_error": maximum_path_error,
        "constraint_mismatches": constraint_mismatches,
    }


def load_fit_groups_and_labels(
    args: argparse.Namespace,
    dataset,
    model,
    device: torch.device,
    stage1_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_dir = args.scaled_archive_dir / "candidate_shards"
    shard_paths = sorted(source_dir.glob("env*.pt"))
    if args.fit_environment_limit is not None:
        shard_paths = shard_paths[: int(args.fit_environment_limit)]
    if not shard_paths:
        raise FileNotFoundError(f"No scaled candidate shards in {source_dir}")
    label_dir = args.output_dir / "fit_label_shards"
    label_dir.mkdir(parents=True, exist_ok=True)
    all_groups: list[dict[str, Any]] = []
    audits = []
    for index, source_path in enumerate(shard_paths, start=1):
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if source.get("stage1_checkpoint_sha256") != stage1_hash:
            raise ValueError(f"Stage-1 mismatch: {source_path}")
        groups = source["groups"]
        config = source["collection_config"]
        label_path = label_dir / source_path.name
        if label_path.exists():
            label_archive = torch.load(
                label_path, map_location="cpu", weights_only=False
            )
            if label_archive.get("label_version") != LABEL_VERSION:
                raise ValueError(f"Label version mismatch: {label_path}")
            records = label_archive["records"]
            action = "reused"
        else:
            records = regenerated_records(
                model,
                dataset,
                source["contexts"],
                variants=config["mask_variants"],
                candidates=int(config["candidates"]),
                source_seed=int(config["source_seed"]),
                source_phase=str(config["source_phase"]),
                batch_size=int(args.collection_batch_size),
                device=device,
            )
            label_archive = {
                "label_version": LABEL_VERSION,
                "stage1_checkpoint_sha256": stage1_hash,
                "source_candidate_shard": str(source_path.resolve()),
                "records": records,
            }
            temporary = label_path.with_suffix(".pt.tmp")
            torch.save(label_archive, temporary)
            temporary.replace(label_path)
            action = "collected"
        enriched, audit = attach_verified_labels(
            groups, records, tolerance=float(args.path_check_tolerance)
        )
        all_groups.extend(enriched)
        audits.append(audit)
        print(
            f"fit labels {action} {index:03d}/{len(shard_paths):03d} "
            f"{source_path.stem}: groups={len(groups)} "
            f"path_error={audit['maximum_candidate_path_error']:.3g}",
            flush=True,
        )
    return all_groups, {
        "environment_shards": len(shard_paths),
        "groups": len(all_groups),
        "maximum_candidate_path_error": max(
            value["maximum_candidate_path_error"] for value in audits
        ),
        "constraint_mismatches": sum(
            value["constraint_mismatches"] for value in audits
        ),
    }


def load_external_groups_and_labels(
    args: argparse.Namespace,
    model,
    device: torch.device,
    stage1_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive = torch.load(
        args.external_candidate_archive, map_location="cpu", weights_only=False
    )
    if archive.get("source_checkpoint_sha256") != stage1_hash:
        raise ValueError("External candidates use another Stage-1 checkpoint")
    groups = archive["groups"]
    config = archive["collection_config"]
    split = archive["environment_split"]
    allowed_environments = {
        value for values in split.values() for value in values
    }
    if args.validation_environment_limit is not None:
        limited = set(sorted(allowed_environments)[: int(args.validation_environment_limit)])
        allowed_environments = limited
        groups = [
            group
            for group in groups
            if str(group["environment"]).split("::")[-1] in limited
        ]
    dataset = make_external_dataset(
        args.external_dataset_root,
        sorted(allowed_environments),
        args,
    )
    cache_path = args.output_dir / "external_severity_labels.pt"
    if cache_path.exists():
        label_archive = torch.load(cache_path, map_location="cpu", weights_only=False)
        if label_archive.get("label_version") != LABEL_VERSION:
            raise ValueError("External label cache version mismatch")
        records = label_archive["records"]
        action = "reused"
    else:
        records = []
        contexts_by_partition = archive["contexts"]
        for partition, contexts in contexts_by_partition.items():
            selected_contexts = [
                context
                for context in contexts
                if str(context["raw_environment"]) in allowed_environments
            ]
            records.extend(
                regenerated_records(
                    model,
                    dataset,
                    selected_contexts,
                    variants=config["mask_variants"][partition],
                    candidates=int(config["candidates"]),
                    source_seed=int(config["source_seed"]),
                    source_phase=f"stability-critic-v3-{partition}",
                    batch_size=int(args.collection_batch_size),
                    device=device,
                )
            )
        label_archive = {
            "label_version": LABEL_VERSION,
            "stage1_checkpoint_sha256": stage1_hash,
            "source_candidate_archive": str(args.external_candidate_archive.resolve()),
            "records": records,
        }
        temporary = cache_path.with_suffix(".pt.tmp")
        torch.save(label_archive, temporary)
        temporary.replace(cache_path)
        action = "collected"
    enriched, audit = attach_verified_labels(
        groups, records, tolerance=float(args.path_check_tolerance)
    )
    audit["environments"] = len(allowed_environments)
    print(
        f"external labels {action}: groups={len(enriched)} "
        f"path_error={audit['maximum_candidate_path_error']:.3g}",
        flush=True,
    )
    return enriched, audit


class CandidateGroups(Dataset):
    def __init__(self, groups: Sequence[Mapping[str, Any]]):
        self.groups = list(groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.groups[index]


def collate_groups(groups: Sequence[Mapping[str, Any]]):
    keys = (*DEPLOYABLE_FIELDS, "constraint_ok", *SEVERITY_FIELDS)
    return {
        **{key: torch.stack([group[key] for group in groups]) for key in keys},
        "environment": [str(group["environment"]) for group in groups],
        "partition": [str(group.get("partition", "fit")) for group in groups],
    }


def move_batch(batch, device: torch.device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def critic_objective(critic, batch, args):
    predicted_risk, safe_logit = critic(
        *(batch[key] for key in DEPLOYABLE_FIELDS)
    )
    classification = stability_probability_loss(
        safe_logit, batch["constraint_ok"]
    )
    ranking = stability_pairwise_ranking_loss(
        safe_logit, batch["constraint_ok"]
    )
    risk, risk_count = conditional_severity_loss(
        predicted_risk,
        batch["stability_violation_max"],
        batch["constraint_ok"],
    )
    loss = (
        classification
        + float(args.ranking_weight) * ranking
        + float(args.risk_weight) * risk
    )
    return loss, classification, ranking, risk, risk_count


@torch.no_grad()
def validation_objective(critic, groups, args, device):
    loader = DataLoader(
        CandidateGroups(groups),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_groups,
        num_workers=0,
    )
    critic.eval()
    sums = np.zeros(4, dtype=np.float64)
    batches = 0
    unsafe_candidates = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        loss, classification, ranking, risk, risk_count = critic_objective(
            critic, batch, args
        )
        sums += np.asarray(
            [float(loss), float(classification), float(ranking), float(risk)]
        )
        batches += 1
        unsafe_candidates += int(risk_count)
    return {
        "loss": float(sums[0] / max(batches, 1)),
        "classification": float(sums[1] / max(batches, 1)),
        "ranking": float(sums[2] / max(batches, 1)),
        "risk": float(sums[3] / max(batches, 1)),
        "unsafe_candidates": unsafe_candidates,
    }


def train_critic(fit_groups, validation_groups, args, device):
    initial = torch.load(
        args.initial_critic_checkpoint, map_location="cpu", weights_only=False
    )
    config = initial.get("critic_config", {})
    critic = PathAlignedCandidateCritic(
        map_channels=int(config.get("map_channels", 4)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    ).to(device)
    critic.load_state_dict(initial["critic_state_dict"], strict=True)
    initial_state = {
        key: value.detach().cpu().clone() for key, value in critic.state_dict().items()
    }
    loader = DataLoader(
        CandidateGroups(fit_groups),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(int(args.seed) + 1),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    trace = []
    best_state = None
    best_epoch = None
    best_loss = math.inf
    stale = 0
    for epoch in range(int(args.epochs)):
        critic.train()
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        gradient_sum = 0.0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, classification, ranking, risk, _ = critic_objective(
                critic, batch, args
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite dual-head critic loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), float(args.grad_clip_norm)
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("Non-finite dual-head critic gradient")
            optimizer.step()
            sums += np.asarray(
                [float(loss), float(classification), float(ranking), float(risk)]
            )
            gradient_sum += float(gradient_norm)
            batches += 1
        validation = validation_objective(critic, validation_groups, args, device)
        row = {
            "epoch": epoch + 1,
            "loss": float(sums[0] / max(batches, 1)),
            "classification": float(sums[1] / max(batches, 1)),
            "ranking": float(sums[2] / max(batches, 1)),
            "risk": float(sums[3] / max(batches, 1)),
            "gradient_norm_pre_clip": gradient_sum / max(batches, 1),
            "validation_loss": validation["loss"],
            "validation_classification": validation["classification"],
            "validation_ranking": validation["ranking"],
            "validation_risk": validation["risk"],
        }
        trace.append(row)
        if validation["loss"] < best_loss - 1e-6:
            best_loss = validation["loss"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in critic.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch + 1:03d} loss={row['loss']:.6f} "
            f"cls={row['classification']:.6f} risk={row['risk']:.6f} "
            f"val={row['validation_loss']:.6f} "
            f"val_risk={row['validation_risk']:.6f} "
            f"grad={row['gradient_norm_pre_clip']:.4f}",
            flush=True,
        )
        if stale >= int(args.early_stopping_patience):
            print(
                f"early_stop={epoch + 1:03d} best_epoch={best_epoch:03d} "
                f"best_val={best_loss:.6f}",
                flush=True,
            )
            break
    if best_state is None or best_epoch is None:
        raise RuntimeError("No dual-head critic checkpoint was selected")
    critic.load_state_dict(best_state, strict=True)
    return critic, initial_state, trace, best_epoch, best_loss, config


@torch.no_grad()
def predict_groups(critic, groups, args, device):
    loader = DataLoader(
        CandidateGroups(groups),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_groups,
        num_workers=0,
    )
    critic.eval()
    result = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        risk, logit = critic(*(batch[key] for key in DEPLOYABLE_FIELDS))
        result.extend(zip(risk.cpu(), logit.cpu()))
    return result


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def risk_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if len(targets) == 0:
        return {"count": 0, "mae": None, "rmse": None, "spearman": None}
    error = predictions - targets
    if len(targets) < 2 or np.std(predictions) == 0.0 or np.std(targets) == 0.0:
        spearman = None
    else:
        spearman = float(
            np.corrcoef(rankdata(predictions), rankdata(targets))[0, 1]
        )
    return {
        "count": len(targets),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "spearman": spearman,
        "prediction_mean": float(np.mean(predictions)),
        "target_mean": float(np.mean(targets)),
    }


def evaluate_critic(critic, groups, args, device, name: str):
    predictions = predict_groups(critic, groups, args, device)
    probabilities = []
    labels = []
    risk_predictions = []
    risk_targets = []
    rows = []
    for group, (risk, logit) in zip(groups, predictions):
        constraint = group["constraint_ok"].bool()
        forbidden = constraint[:, 0]
        stability = constraint[:, 1]
        curvature_max = group["candidate_curvature"].amax(dim=1)
        probability = torch.sigmoid(logit)
        current_score = probability * curve_preference(curvature_max)
        combined_score = dual_head_score(risk, logit, curvature_max)
        allowed = torch.where(forbidden)[0]
        if len(allowed):
            current_index = int(allowed[current_score[allowed].argmax()])
            combined_index = int(allowed[combined_score[allowed].argmax()])
            current_safe = bool(stability[current_index])
            combined_safe = bool(stability[combined_index])
        else:
            current_index = -1
            combined_index = -1
            current_safe = False
            combined_safe = False
        support = bool((forbidden & stability).any())
        rows.append(
            {
                "critic": name,
                "partition": str(group.get("partition", "validation")),
                "environment": str(group["environment"]),
                "path_num": int(group["path_num"]),
                "mask_variant": int(group["mask_variant"]),
                "forbidden_output": bool(forbidden.any()),
                "capsizing_support": support,
                "current_product_safe": current_safe,
                "dual_head_safe": combined_safe,
                "current_index": current_index,
                "dual_head_index": combined_index,
            }
        )
        probabilities.extend(probability[forbidden].tolist())
        labels.extend(stability[forbidden].tolist())
        unsafe = forbidden & (~stability)
        risk_predictions.extend(risk[unsafe].tolist())
        risk_targets.extend(
            severity_target(group["stability_violation_max"])[unsafe].tolist()
        )
    calibration = calibration_metrics(probabilities, labels)
    metrics = {
        "groups": len(groups),
        "environments": len({str(group["environment"]) for group in groups}),
        "calibration": calibration,
        "conditional_severity": risk_metrics(
            np.asarray(risk_predictions), np.asarray(risk_targets)
        ),
        "forbidden_output_rate": float(
            np.mean([row["forbidden_output"] for row in rows])
        ),
        "capsizing_support_rate": float(
            np.mean([row["capsizing_support"] for row in rows])
        ),
        "current_product_safe_rate": float(
            np.mean([row["current_product_safe"] for row in rows])
        ),
        "dual_head_safe_rate": float(
            np.mean([row["dual_head_safe"] for row in rows])
        ),
        "current_to_dual_rescue": sum(
            (not row["current_product_safe"]) and row["dual_head_safe"]
            for row in rows
        ),
        "current_to_dual_regression": sum(
            row["current_product_safe"] and (not row["dual_head_safe"])
            for row in rows
        ),
    }
    return metrics, rows


def label_statistics(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = []
    target = []
    for group in groups:
        constraint = group["constraint_ok"].bool()
        mask = constraint[:, 0] & (~constraint[:, 1])
        raw.extend(group["stability_violation_max"][mask].tolist())
        target.extend(severity_target(group["stability_violation_max"])[mask].tolist())
    raw_array = np.asarray(raw)
    target_array = np.asarray(target)
    return {
        "unsafe_forbidden_valid_candidates": len(raw),
        "violation_max_m": {
            "mean": float(raw_array.mean()),
            "q50": float(np.quantile(raw_array, 0.50)),
            "q90": float(np.quantile(raw_array, 0.90)),
            "q99": float(np.quantile(raw_array, 0.99)),
            "max": float(raw_array.max()),
        },
        "log1p_normalized_target": {
            "mean": float(target_array.mean()),
            "q50": float(np.quantile(target_array, 0.50)),
            "q90": float(np.quantile(target_array, 0.90)),
            "q99": float(np.quantile(target_array, 0.99)),
            "max": float(target_array.max()),
        },
    }


def render_report(result: Mapping[str, Any]) -> str:
    before = result["validation"]["initial_scaled_critic"]
    after = result["validation"]["dual_head_critic"]
    before_risk = before["conditional_severity"]
    after_risk = after["conditional_severity"]
    return "\n".join(
        [
            "# Dual-Head Stability Critic Retraining",
            "",
            "The safe head retains BCE plus within-pool ranking supervision. The risk head predicts conditional log-normalized maximum stability violation on forbidden-valid, stability-unsafe candidates only.",
            "",
            "| Metric | Initial scaled critic | Dual-head critic |",
            "|---|---:|---:|",
            f"| Validation AUC | {before['calibration']['auc']:.3f} | {after['calibration']['auc']:.3f} |",
            f"| Validation ECE | {before['calibration']['ece_10']:.3f} | {after['calibration']['ece_10']:.3f} |",
            f"| Severity MAE | {before_risk['mae']:.4f} | {after_risk['mae']:.4f} |",
            f"| Severity RMSE | {before_risk['rmse']:.4f} | {after_risk['rmse']:.4f} |",
            f"| Severity Spearman | {before_risk['spearman']:.3f} | {after_risk['spearman']:.3f} |",
            f"| Current probability x curvature selected-safe | {before['current_product_safe_rate']:.2%} | {after['current_product_safe_rate']:.2%} |",
            f"| Dual-head score selected-safe | {before['dual_head_safe_rate']:.2%} | {after['dual_head_safe_rate']:.2%} |",
            "",
            f"Dual-head rescues/regressions relative to its own probability x curvature selector: {after['current_to_dual_rescue']}/{after['current_to_dual_regression']}.",
            f"Best epoch: {result['training']['best_epoch']}; validation objective: {result['training']['best_validation_loss']:.6f}.",
            "",
            f"All {result['validation_label_audit']['environments']} selected dataset1_val terrains form one validation set. Their historical calibration/admission labels are retained only as row metadata and do not create separate claims.",
            "Final test remains unopened.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    for name in (
        "collection_batch_size",
        "batch_size",
        "epochs",
        "early_stopping_patience",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(args.learning_rate) <= 0.0 or float(args.risk_weight) <= 0.0:
        raise ValueError("learning rate and risk weight must be positive")
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Completed result already exists: {result_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smoke = (
        args.fit_environment_limit is not None
        or args.validation_environment_limit is not None
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    stage1_hash = sha256_file(args.stage1_checkpoint)
    protocol = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "frozen_before_label_regeneration_and_training",
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "stage1_checkpoint_sha256": stage1_hash,
        "initial_critic_checkpoint": str(args.initial_critic_checkpoint.resolve()),
        "initial_critic_checkpoint_sha256": sha256_file(
            args.initial_critic_checkpoint
        ),
        "scaled_archive_dir": str(args.scaled_archive_dir.resolve()),
        "external_candidate_archive": str(
            args.external_candidate_archive.resolve()
        ),
        "external_candidate_archive_sha256": sha256_file(
            args.external_candidate_archive
        ),
        "fit_dataset": str(args.data_folder.resolve()),
        "validation_dataset": str(args.external_dataset_root.resolve()),
        "training": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "ranking_weight": float(args.ranking_weight),
            "risk_weight": float(args.risk_weight),
            "early_stopping_patience": int(args.early_stopping_patience),
            "initialization": "scaled_6400_checkpoint",
            "optimized_parameters": "all_critic_parameters",
        },
        "mask_seed": int(args.mask_seed),
        "safe_head_target": "stability_ok_conditioned_on_forbidden_valid",
        "risk_head_target": (
            "log1p(stability_violation_max/d_safe)_conditioned_on_unsafe"
        ),
        "dual_head_score": (
            "p_safe * p_curve * exp(-(1-p_safe)*predicted_conditional_severity)"
        ),
        "validation_uses_all_external_archive_environments": (
            args.validation_environment_limit is None
        ),
        "final_test_opened": False,
        "smoke": smoke,
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != jsonable(protocol):
            raise ValueError("Existing resumable run uses a different protocol")
    else:
        write_json(protocol_path, protocol)

    seed_everything(args.seed)
    stage1, model_args, stage1_checkpoint = load_model(
        args.stage1_checkpoint, device
    )
    _require_main_method_model(stage1)
    _require_current_mask_semantics(stage1_checkpoint)
    _require_demo_target_semantics(stage1_checkpoint)
    stage1.eval()
    stage1.use_gradient_checkpoint = False
    fit_dataset, _ = make_partial_dataset(
        args.data_folder,
        "train",
        compute_stability_map=True,
        mask_seed=int(args.mask_seed),
        p_mask=1.0,
        mask_mode="stage2_independent",
        dynamic_mask_noise=False,
    )
    fit_groups, fit_label_audit = load_fit_groups_and_labels(
        args, fit_dataset, stage1, device, stage1_hash
    )
    validation_groups, validation_label_audit = load_external_groups_and_labels(
        args, stage1, device, stage1_hash
    )
    del fit_dataset, stage1, stage1_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_everything(args.seed)
    critic, initial_state, trace, best_epoch, best_loss, critic_config = train_critic(
        fit_groups, validation_groups, args, device
    )
    initial_critic = PathAlignedCandidateCritic(
        map_channels=int(critic_config.get("map_channels", 4)),
        hidden_dim=int(critic_config.get("hidden_dim", 128)),
    ).to(device)
    initial_critic.load_state_dict(initial_state, strict=True)
    initial_metrics, initial_rows = evaluate_critic(
        initial_critic, validation_groups, args, device, "initial_scaled_critic"
    )
    final_metrics, final_rows = evaluate_critic(
        critic, validation_groups, args, device, "dual_head_critic"
    )

    checkpoint_path = args.output_dir / "critic_dual_head_6400.pth"
    checkpoint = {
        "checkpoint_version": 1,
        "stability_critic_semantics": (
            "forbidden_conditioned_stability_hurdle_critic_dual_head_v1"
        ),
        "critic_state_dict": critic.state_dict(),
        "critic_config": {
            "map_channels": int(critic_config.get("map_channels", 4)),
            "hidden_dim": int(critic_config.get("hidden_dim", 128)),
        },
        "source_checkpoint": str(args.stage1_checkpoint.resolve()),
        "source_checkpoint_sha256": stage1_hash,
        "initial_critic_checkpoint": str(args.initial_critic_checkpoint.resolve()),
        "training_trace": trace,
        "epoch": int(best_epoch) - 1,
        "best_validation_loss": float(best_loss),
        "safe_head_semantics": "P(stability_ok | forbidden_ok, deployment_inputs)",
        "risk_head_semantics": (
            "E[log1p(stability_violation_max/d_safe) | stability_unsafe, "
            "forbidden_ok, deployment_inputs]"
        ),
        "d_safe_meters": float(SAFETY_COST_CONFIG.d_safe_meters),
        "training_config": protocol["training"],
        "fit_label_statistics": label_statistics(fit_groups),
        "validation_metrics": final_metrics,
        "privileged_fields_are_labels_only": True,
        "final_test_opened": False,
    }
    torch.save(checkpoint, checkpoint_path)
    write_json(args.output_dir / "training_trace.json", trace)
    write_csv(args.output_dir / "validation_rows.csv", initial_rows + final_rows)
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "completed_smoke" if smoke else "completed_retraining",
        "model_args": model_args,
        "fit_label_audit": fit_label_audit,
        "validation_label_audit": validation_label_audit,
        "fit_label_statistics": label_statistics(fit_groups),
        "validation_label_statistics": label_statistics(validation_groups),
        "training": {
            "best_epoch": int(best_epoch),
            "stop_epoch": len(trace),
            "best_validation_loss": float(best_loss),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "validation": {
            "initial_scaled_critic": initial_metrics,
            "dual_head_critic": final_metrics,
        },
        "validation_terrain_semantics": (
            "all_dataset1_val_terrains_used_as_one_development_validation_set"
        ),
        "final_test_opened": False,
        "smoke": smoke,
    }
    write_json(result_path, result)
    (args.output_dir / "REPORT.md").write_text(
        render_report(result), encoding="utf-8"
    )
    print(render_report(result), flush=True)


if __name__ == "__main__":
    main()
